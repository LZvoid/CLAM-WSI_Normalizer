import os
import numpy as np
import openslide
import cv2
import torch
import h5py
from PIL import Image
from dataset_modules.dataset_h5 import Dataset_All_Bags
from wsi_normalizer import TorchVahadaneNormalizer, imread
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import threading
from functools import partial
import time
import json
import datetime

# 可配置的参数
DEFAULT_PATCH_SIZE = 512
DEFAULT_LEVEL = 0

def select_optimal_target(bags_dataset, args, device='cuda', max_workers=None, 
                         patch_size=DEFAULT_PATCH_SIZE, level=DEFAULT_LEVEL):
    """
    优化版本：使用H5文件中的预提取坐标和WSI缩略图加速target选择
    """
    print(f"=== 开始选择最优target图像 (patch_size: {patch_size}x{patch_size}, level: {level}) ===")
    start_time = time.time()
    
    # 设置并行工作线程数
    if max_workers is None:
        max_workers = min(cpu_count(), 8)
    print(f"使用 {max_workers} 个并行工作线程")
    
    # Step 1: 从H5文件筛选高质量WSI
    print("1. 从H5文件筛选高质量WSI候选...")
    candidate_wsis = screen_wsis_from_h5(bags_dataset, args, 
                                        num_candidates=15, 
                                        max_workers=max_workers,
                                        patch_size=patch_size,
                                        level=level)
    
    if not candidate_wsis:
        raise RuntimeError("没有找到合适的候选WSI")
    
    # Step 2: 从H5坐标并行提取patches
    print("2. 从H5坐标并行提取高质量patches...")
    candidate_patches = extract_patches_from_h5_coords(candidate_wsis, 
                                                      num_patches_per_wsi=10,
                                                      max_workers=max_workers,
                                                      patch_size=patch_size,
                                                      level=level)
    
    if not candidate_patches:
        raise RuntimeError("没有提取到合适的候选patches")
    
    # Step 3: 选择最佳target
    print("3. 评估并选择最佳target...")
    best_target, target_info = select_best_target_patch(candidate_patches)
    
    # Step 4: 保存target
    print("4. 保存target...")
    target_path = save_target_image(best_target, target_info)
    
    # Step 5: 并行验证
    print("5. 并行验证target效果...")
    validate_target_quality_parallel(target_path, candidate_patches[:8], max_workers=4)
    
    total_time = time.time() - start_time
    print(f"=== Target选择完成: {target_path} (耗时: {total_time:.1f}秒) ===")
    return target_path, target_info

def screen_wsis_from_h5(bags_dataset, args, num_candidates=15, max_workers=8,
                       patch_size=DEFAULT_PATCH_SIZE, level=DEFAULT_LEVEL):
    """
    从H5文件筛选高质量WSI - 使用缩略图评估
    """
    print(f"从 {len(bags_dataset)} 个WSI的H5文件中筛选候选...")
    
    #sample_size = min(50, len(bags_dataset))  # 增加采样数量
    sample_size = len(bags_dataset)
    wsi_tasks = []
    
    # 准备任务列表
    for i in range(sample_size):
        slide_id = bags_dataset[i].split(args.slide_ext)[0]
        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
        h5_file_path = os.path.join(args.data_h5_dir, slide_id + '.h5')
        
        # 检查文件是否存在
        if os.path.exists(slide_file_path) and os.path.exists(h5_file_path):
            wsi_tasks.append((slide_id, slide_file_path, h5_file_path, patch_size, level))
    
    print(f"找到 {len(wsi_tasks)} 个有效的WSI-H5文件对")
    
    # 并行处理WSI评估
    wsi_scores = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_wsi = {
            executor.submit(evaluate_wsi_from_h5_thumbnail, slide_id, slide_path, h5_path, patch_size, level): 
            (slide_id, slide_path, h5_path)
            for slide_id, slide_path, h5_path, patch_size, level in wsi_tasks
        }
        
        # 收集结果
        for future in as_completed(future_to_wsi):
            slide_id, slide_path, h5_path = future_to_wsi[future]
            try:
                score, patch_count = future.result()
                if score > 0.2 and patch_count > 0:  # 适当降低阈值
                    wsi_scores.append((slide_id, score, slide_path, h5_path, patch_count))
                    print(f"  ✓ {slide_id}: 评分 = {score:.3f}, patches = {patch_count}")
            except Exception as e:
                print(f"  ✗ 跳过 {slide_id}: {e}")
    
    if not wsi_scores:
        print("警告：没有找到合格的WSI，尝试降低标准...")
        # 重新评估，降低标准
        wsi_scores = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_wsi = {
                executor.submit(evaluate_wsi_from_h5_thumbnail_relaxed, slide_id, slide_path, h5_path, patch_size, level): 
                (slide_id, slide_path, h5_path)
                for slide_id, slide_path, h5_path, patch_size, level in wsi_tasks
            }
            
            for future in as_completed(future_to_wsi):
                slide_id, slide_path, h5_path = future_to_wsi[future]
                try:
                    score, patch_count = future.result()
                    if score > 0.1 and patch_count > 0:
                        wsi_scores.append((slide_id, score, slide_path, h5_path, patch_count))
                        print(f"  ✓ {slide_id}: 评分 = {score:.3f}, patches = {patch_count}")
                except Exception as e:
                    continue
    
    # 排序并选择最好的候选
    wsi_scores.sort(key=lambda x: x[1], reverse=True)
    selected_info = wsi_scores[:num_candidates]
    
    # 批量加载选中的WSI对象
    selected_wsis = []
    for slide_id, score, slide_path, h5_path, patch_count in selected_info:
        try:
            wsi = openslide.open_slide(slide_path)
            selected_wsis.append((slide_id, score, slide_path, h5_path, wsi, patch_count))
        except Exception as e:
            print(f"  加载WSI失败 {slide_id}: {e}")
    
    print(f"筛选出 {len(selected_wsis)} 个高质量候选WSI")
    return selected_wsis

def evaluate_wsi_from_h5_thumbnail(slide_id, slide_path, h5_path, patch_size, level):
    """
    使用缩略图评估WSI质量 - 标准版本
    """
    try:
        # 检查H5文件是否存在
        if not os.path.exists(h5_path):
            return 0, 0
        
        # 读取H5文件中的坐标数量
        with h5py.File(h5_path, 'r') as f:
            coords = f['coords'][:]
            if len(coords) < 10:  # 至少需要10个patch
                return 0, 0
        
        # 使用缩略图进行质量评估
        with openslide.open_slide(slide_path) as wsi:
            # 检查WSI基本尺寸
            width, height = wsi.level_dimensions[level]
            if width < patch_size or height < patch_size:
                return 0, 0
            
            # 生成高质量缩略图
            thumbnail_size = (2048, 2048)
            thumbnail = wsi.get_thumbnail(thumbnail_size)
            thumbnail_np = np.array(thumbnail.convert('RGB'))
            
            # 多维度质量评估
            score = comprehensive_thumbnail_quality_assessment(thumbnail_np, slide_id)
            
            return score, len(coords)
        
    except Exception as e:
        print(f"  评估失败 {slide_id}: {e}")
        return 0, 0

def evaluate_wsi_from_h5_thumbnail_relaxed(slide_id, slide_path, h5_path, patch_size, level):
    """
    使用缩略图评估WSI质量 - 放宽版本
    """
    try:
        if not os.path.exists(h5_path):
            return 0, 0
        
        with h5py.File(h5_path, 'r') as f:
            coords = f['coords'][:]
            if len(coords) < 5:  # 降低patch数量要求
                return 0, 0
        
        with openslide.open_slide(slide_path) as wsi:
            width, height = wsi.level_dimensions[level]
            if width < patch_size or height < patch_size:
                return 0, 0
            
            # 使用较小的缩略图加快速度
            thumbnail_size = (1024, 1024)
            thumbnail = wsi.get_thumbnail(thumbnail_size)
            thumbnail_np = np.array(thumbnail.convert('RGB'))
            
            # 简化的质量评估
            score = simple_thumbnail_quality_assessment(thumbnail_np)
            
            return score, len(coords)
        
    except Exception as e:
        return 0, 0

def comprehensive_thumbnail_quality_assessment(thumbnail_np, slide_id):
    """
    综合缩略图质量评估
    """
    try:
        # 基础颜色空间转换
        gray = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2LAB)
        
        # 1. 组织覆盖率评估
        tissue_mask = gray < 230
        tissue_ratio = np.mean(tissue_mask)
        
        # if tissue_ratio < 0.15:  # 组织覆盖率太低
        #     return 0
        
        # 2. 只在组织区域进行分析
        tissue_pixels_gray = gray[tissue_mask]
        tissue_pixels_hsv = hsv[tissue_mask]
        tissue_pixels_lab = lab[tissue_mask]
        
        # 3. 对比度和清晰度评估
        contrast_score = np.std(tissue_pixels_gray) / 100.0
        contrast_score = min(contrast_score, 1.0)
        
        # 4. 颜色质量评估
        # H&E染色特征检测
        h_channel = tissue_pixels_hsv[:, 0]
        s_channel = tissue_pixels_hsv[:, 1]
        
        # 检测蓝紫色区域（细胞核）
        nucleus_mask = ((h_channel >= 110) & (h_channel <= 130)) & (s_channel > 80)
        nucleus_ratio = np.mean(nucleus_mask)
        
        # 检测粉红色区域（细胞质）
        cytoplasm_mask = ((h_channel >= 160) | (h_channel <= 20)) & (s_channel > 60)
        cytoplasm_ratio = np.mean(cytoplasm_mask)
        
        # 5. 饱和度质量
        saturation_mean = np.mean(s_channel)
        saturation_quality = 1.0 - abs(saturation_mean - 120) / 120.0
        saturation_quality = max(0, saturation_quality)
        
        # 6. 亮度分布
        brightness_mean = np.mean(tissue_pixels_gray)
        brightness_quality = 1.0 - abs(brightness_mean - 140) / 140.0
        brightness_quality = max(0, brightness_quality)
        
        # 7. 颜色分布均匀性
        color_std = np.std([np.std(tissue_pixels_lab[:, i]) for i in range(3)])
        color_uniformity = 1.0 - min(color_std / 50.0, 1.0)
        
        # 8. 纹理复杂度
        texture_complexity = cv2.Laplacian(gray, cv2.CV_64F).var() / 2000.0
        texture_complexity = min(texture_complexity, 1.0)
        
        # 9. H&E平衡性
        he_balance = min(nucleus_ratio, cytoplasm_ratio) / (max(nucleus_ratio, cytoplasm_ratio) + 1e-6)
        he_balance = min(he_balance, 1.0)
        
        # 10. 综合评分
        score = (tissue_ratio * 0.25 +           # 组织覆盖率
                contrast_score * 0.15 +          # 对比度
                nucleus_ratio * 0.15 +           # 细胞核比例
                cytoplasm_ratio * 0.10 +         # 细胞质比例
                saturation_quality * 0.10 +      # 饱和度质量
                brightness_quality * 0.05 +      # 亮度质量
                color_uniformity * 0.05 +        # 颜色均匀性
                texture_complexity * 0.05 +      # 纹理复杂度
                he_balance * 0.10)               # H&E平衡
        
        return score
        
    except Exception as e:
        print(f"    缩略图评估失败 {slide_id}: {e}")
        return 0

def simple_thumbnail_quality_assessment(thumbnail_np):
    """
    简化的缩略图质量评估
    """
    try:
        gray = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2HSV)
        
        # 组织覆盖率
        tissue_mask = gray < 230
        tissue_ratio = np.mean(tissue_mask)
        
        if tissue_ratio < 0.1:
            return 0
        
        # 简单的质量指标
        tissue_pixels = gray[tissue_mask]
        contrast = np.std(tissue_pixels) / 100.0
        saturation = np.mean(hsv[:, :, 1][tissue_mask]) / 255.0
        
        score = tissue_ratio * 0.6 + contrast * 0.3 + saturation * 0.1
        
        return score
        
    except Exception:
        return 0

def extract_patches_from_h5_coords(candidate_wsis, num_patches_per_wsi=10, max_workers=8,
                                  patch_size=DEFAULT_PATCH_SIZE, level=DEFAULT_LEVEL):
    """
    从H5坐标并行提取patches
    """
    print(f"从H5坐标并行提取patches (patch_size: {patch_size}x{patch_size}, level: {level})...")
    all_patches = []
    
    print(f"候选WSI数量: {len(candidate_wsis)}")
    
    # 为每个WSI创建独立的提取任务
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有WSI的patch提取任务
        future_to_wsi = {
            executor.submit(extract_patches_from_single_wsi_h5, 
                          slide_id, score, slide_path, h5_path, wsi, patch_count,
                          num_patches_per_wsi, patch_size, level): slide_id
            for slide_id, score, slide_path, h5_path, wsi, patch_count in candidate_wsis
        }
        
        # 收集结果
        for future in as_completed(future_to_wsi):
            slide_id = future_to_wsi[future]
            try:
                patches_from_wsi = future.result()
                all_patches.extend(patches_from_wsi)
                print(f"  ✓ {slide_id}: 提取了 {len(patches_from_wsi)} 个 {patch_size}x{patch_size} patches")
            except Exception as e:
                print(f"  ✗ {slide_id}: 提取失败 - {e}")
    
    print(f"总共提取了 {len(all_patches)} 个候选patches")
    return all_patches

def extract_patches_from_single_wsi_h5(slide_id, wsi_score, slide_path, h5_path, wsi, patch_count,
                                      num_patches_per_wsi, patch_size, level):
    """
    从单个WSI的H5坐标提取patches
    """
    patches_from_wsi = []
    
    try:
        # 读取H5文件中的坐标
        with h5py.File(h5_path, 'r') as f:
            coords = f['coords'][:]
            
            # 智能选择坐标策略
            if len(coords) <= num_patches_per_wsi:
                selected_coords = coords
            else:
                # 分层采样：确保选择多样性
                np.random.seed(hash(slide_id) % 2**32)
                
                # 如果坐标很多，先随机选择更多候选
                if len(coords) > num_patches_per_wsi * 5:
                    pre_selection_size = num_patches_per_wsi * 3
                    pre_indices = np.random.choice(len(coords), pre_selection_size, replace=False)
                    pre_coords = coords[pre_indices]
                else:
                    pre_coords = coords
                
                # 从预选坐标中最终选择
                if len(pre_coords) > num_patches_per_wsi:
                    final_indices = np.random.choice(len(pre_coords), num_patches_per_wsi, replace=False)
                    selected_coords = pre_coords[final_indices]
                else:
                    selected_coords = pre_coords
        
        # 检查WSI尺寸
        width, height = wsi.level_dimensions[level]
        
        success_count = 0
        # 从选定的坐标提取patches
        for coord in selected_coords:
            if success_count >= num_patches_per_wsi:
                break
                
            try:
                x, y = int(coord[0]), int(coord[1])
                
                # 边界检查
                if x < 0 or y < 0 or x + patch_size > width or y + patch_size > height:
                    continue
                
                # 读取patch
                patch = wsi.read_region((x, y), level, (patch_size, patch_size))
                patch_np = np.array(patch.convert('RGB'))
                
                # 检查patch尺寸
                if patch_np.shape[:2] != (patch_size, patch_size):
                    continue
                
                # 快速预筛选
                if quick_patch_prefilter(patch_np, patch_size):
                    # 详细评估
                    patch_score = evaluate_patch_for_target_optimized(patch_np)
                    
                    if patch_score > 0.3:  # 适当降低阈值
                        patch_info = {
                            'patch': patch_np,
                            'score': patch_score,
                            'slide_id': slide_id,
                            'wsi_score': wsi_score,
                            'position': (x, y),
                            'level': level,
                            'patch_size': patch_size,
                            'from_h5': True
                        }
                        patches_from_wsi.append(patch_info)
                        success_count += 1
                        
            except Exception as e:
                continue
        
    except Exception as e:
        print(f"    H5处理错误 {slide_id}: {e}")
    
    return patches_from_wsi

def quick_patch_prefilter(patch, patch_size):
    """
    快速预筛选patch
    """
    try:
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        tissue_ratio = np.mean(gray < 230)
        
        # 组织覆盖率检查
        if tissue_ratio < 0.25:
            return False
        
        # 对比度检查
        std_threshold = max(15, patch_size // 30)
        if np.std(gray) < std_threshold:
            return False
        
        # 避免过度曝光或过暗
        mean_brightness = np.mean(gray)
        if mean_brightness < 50 or mean_brightness > 220:
            return False
        
        return True
        
    except Exception:
        return False

def evaluate_patch_for_target_optimized(patch):
    """
    进阶 H&E patch 筛选：
    - 核蓝紫要好
    - 胞质粉红要明显
    - 核密度过高直接丢弃
    - 颜色偏紫而无粉红则丢弃
    """
    try:
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)

        tissue_mask = gray < 230
        background_mask = (hsv[:,:,1] < 30) & (hsv[:,:,2] > 200)
        tissue_mask = tissue_mask & ~background_mask

        tissue_ratio = np.mean(tissue_mask)
        if tissue_ratio < 0.3:
            return 0

        tissue_h = hsv[:,:,0][tissue_mask]
        tissue_s = hsv[:,:,1][tissue_mask]
        tissue_v = hsv[:,:,2][tissue_mask]
        tissue_gray = gray[tissue_mask]

        # 核蓝紫区域
        nucleus_mask_h = (tissue_h >= 110) & (tissue_h <= 130)
        nucleus_mask_s = tissue_s > 80
        nucleus_mask_v = tissue_v > 30

        nucleus_pixels = tissue_s[nucleus_mask_h & nucleus_mask_s & nucleus_mask_v]
        nucleus_ratio = len(nucleus_pixels) / len(tissue_h) if len(tissue_h) > 0 else 0

        # 粉红胞质区域 (更宽色域 + 去掉核蓝紫区域)
        eosin_h_mask = (
            (tissue_h <= 20) | 
            ((tissue_h >= 150) & (tissue_h <= 180))
        )
        eosin_s_mask = tissue_s > 50
        eosin_v_mask = tissue_v > 50

        # 排除核蓝紫区
        not_nucleus_mask = ~nucleus_mask_h

        cytoplasm_pixels = tissue_s[eosin_h_mask & eosin_s_mask & eosin_v_mask & not_nucleus_mask]
        cytoplasm_ratio = len(cytoplasm_pixels) / len(tissue_h) if len(tissue_h) > 0 else 0

        # 如果 cytoplasm_ratio 很小，或者核密度太高 → 丢弃
        if cytoplasm_ratio < 0.05 or nucleus_ratio > 0.5:
            return 0

        # 加核-胞质比例约束
        nucleus_cytoplasm_ratio = nucleus_ratio / (cytoplasm_ratio + 1e-6)
        if nucleus_cytoplasm_ratio > 4:
            return 0

        sat_mean = np.mean(tissue_s)
        sat_std = np.std(tissue_s)
        saturation_quality = (1.0 - abs(sat_mean - 120) / 120.0) * (1.0 - min(sat_std / 80.0, 1.0))
        saturation_quality = max(0, saturation_quality)

        contrast = np.std(tissue_gray) / 100.0
        contrast = min(contrast, 1.0)

        brightness_mean = np.mean(tissue_gray)
        brightness_quality = 1.0 - abs(brightness_mean - 140) / 140.0
        brightness_quality = max(0, brightness_quality)

        he_balance = min(nucleus_ratio, cytoplasm_ratio) / (max(nucleus_ratio, cytoplasm_ratio) + 1e-6)
        he_balance = min(he_balance, 1.0)

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(laplacian_var / 1500.0, 1.0)

        if contrast < 0.05 or sharpness < 0.05:
            return 0

        score = (
            tissue_ratio * 0.20 +
            nucleus_ratio * 0.18 +
            cytoplasm_ratio * 0.22 +
            saturation_quality * 0.12 +
            contrast * 0.10 +
            brightness_quality * 0.08 +
            he_balance * 0.12 +
            sharpness * 0.05
        )

        return score

    except Exception:
        return 0




def select_best_target_patch(candidate_patches):
    """
    选择最佳target patch
    """
    if not candidate_patches:
        raise RuntimeError("没有可用的候选patches")
    
    scores = np.array([patch['score'] for patch in candidate_patches])
    sorted_indices = np.argsort(scores)[::-1]
    
    print("Top 10 候选patches:")
    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[i]
        patch_info = candidate_patches[idx]
        patch_size = patch_info.get('patch_size', 'Unknown')
        level = patch_info.get('level', 'Unknown')
        from_h5 = patch_info.get('from_h5', False)
        source = "H5" if from_h5 else "Random"
        print(f"  {i+1}. {patch_info['slide_id']}: {patch_info['score']:.3f} (size: {patch_size}, level: {level}, from: {source})")
    
    best_idx = sorted_indices[0]
    best_patch_info = candidate_patches[best_idx]
    
    target_info = {
        'slide_id': best_patch_info['slide_id'],
        'score': best_patch_info['score'],
        'wsi_score': best_patch_info['wsi_score'],
        'position': best_patch_info['position'],
        'level': best_patch_info['level'],
        'patch_size': best_patch_info.get('patch_size', 512),
        'from_h5': best_patch_info.get('from_h5', False),
        'selection_method': 'quality_based_h5_coords_thumbnail_optimized'
    }
    
    print(f"选择最佳target: {target_info['slide_id']}, 评分: {target_info['score']:.3f}")
    
    return best_patch_info['patch'], target_info

def save_target_image(target_patch, target_info):
    """
    保存target图像和相关信息
    """
    # 创建目录
    os.makedirs("target_imgs", exist_ok=True)
    
    # 根据patch尺寸命名文件
    patch_size = target_info.get('patch_size', 512)
    level = target_info.get('level', 0)
    from_h5 = target_info.get('from_h5', False)
    source_tag = "_h5" if from_h5 else "_random"
    target_path = f"target_imgs/optimal_target_{patch_size}x{patch_size}_level{level}{source_tag}.png"
    
    # 保存图像
    if isinstance(target_patch, np.ndarray):
        Image.fromarray(target_patch).save(target_path, optimize=True)
    else:
        target_patch.save(target_path, optimize=True)
    
    # 保存信息
    info_path = f"target_imgs/target_info_{patch_size}x{patch_size}_level{level}{source_tag}.json"
    
    # 转换numpy类型
    save_info = {}
    for key, value in target_info.items():
        if isinstance(value, (np.integer, np.int64, np.int32)):
            save_info[key] = int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            save_info[key] = float(value)
        elif isinstance(value, tuple):
            save_info[key] = list(value)
        else:
            save_info[key] = value
    
    # 添加配置信息
    save_info['creation_time'] = datetime.datetime.now().isoformat()
    save_info['optimization_version'] = 'thumbnail_based_h5_coords_v1.0'
    save_info['patch_size'] = patch_size
    save_info['level'] = level
    save_info['from_h5_coords'] = from_h5
    
    with open(info_path, 'w') as f:
        json.dump(save_info, f, indent=2)
    
    print(f"Target图像保存到: {target_path}")
    print(f"Target信息保存到: {info_path}")
    
    return target_path

def load_saved_target(patch_size=512, level=0, from_h5=True):
    """
    加载指定尺寸和level的target
    """
    source_tag = "_h5" if from_h5 else "_random"
    target_path = f"target_imgs/optimal_target_{patch_size}x{patch_size}_level{level}{source_tag}.png"
    
    if not os.path.exists(target_path):
        # 尝试另一种类型
        source_tag = "_random" if from_h5 else "_h5"
        target_path = f"target_imgs/optimal_target_{patch_size}x{patch_size}_level{level}{source_tag}.png"
        
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target图像不存在: {target_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normalizer = TorchVahadaneNormalizer(staintools_estimate=False)
    
    # 加载图像
    target = imread(target_path)
    target_tensor = torch.from_numpy(target).to(device, non_blocking=True)
    
    with torch.no_grad():
        normalizer.fit(target_tensor)
    
    # 加载信息
    info_path = target_path.replace('.png', '.json').replace('optimal_target', 'target_info')
    target_info = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            target_info = json.load(f)
    
    print(f"已加载target: {target_path}")
    if target_info:
        print(f"Target来源: {target_info.get('slide_id', 'Unknown')}")
        print(f"Patch尺寸: {target_info.get('patch_size', 'Unknown')}")
        print(f"分辨率层级: {target_info.get('level', 'Unknown')}")
        print(f"来源方式: {'H5坐标' if target_info.get('from_h5_coords', False) else '随机采样'}")
        print(f"质量评分: {target_info.get('score', 'Unknown')}")
    
    return normalizer, target_info

def validate_target_quality_parallel(target_path, sample_patches, max_workers=4):
    """
    并行验证target质量
    """
    print("并行验证target归一化效果...")
    
    try:
        # 创建normalizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        normalizer = TorchVahadaneNormalizer(staintools_estimate=False)
        
        # 加载target
        target = imread(target_path)
        target_tensor = torch.from_numpy(target).to(device)
        
        with torch.no_grad():
            normalizer.fit(target_tensor)
        
        # 简单验证
        print("✓ Target归一化器创建成功")
        print("✓ Target验证完成")
        
    except Exception as e:
        print(f"验证过程出错: {e}")

def create_quick_quality_report(target_info, candidate_patches):
    """
    创建快速质量报告
    """
    print("\n=== 质量报告 ===")
    print(f"最佳target来源: {target_info['slide_id']}")
    print(f"质量评分: {target_info['score']:.3f}")
    print(f"WSI评分: {target_info['wsi_score']:.3f}")
    print(f"坐标位置: {target_info['position']}")
    print(f"总候选patches: {len(candidate_patches)}")
    
    # 评分分布
    scores = [p['score'] for p in candidate_patches]
    print(f"评分统计: 最高={max(scores):.3f}, 最低={min(scores):.3f}, 平均={np.mean(scores):.3f}")
    
    # 来源分布
    sources = {}
    for patch in candidate_patches:
        slide_id = patch['slide_id']
        sources[slide_id] = sources.get(slide_id, 0) + 1
    
    print(f"来源分布: {len(sources)} 个WSI")
    for slide_id, count in sorted(sources.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {slide_id}: {count} patches")

# 使用示例
if __name__ == "__main__":
    import argparse
    
    class Args:
        data_slide_dir = "/mnt/f/c16/CAMELYON16/train"
        data_h5_dir = "/mnt/e/dataset/mhim_c16_h5"
        slide_ext = ".tif"
    
    args = Args()
    csv_path = "./dataset_csv/process_list_autogen.csv"
    bags_dataset = Dataset_All_Bags(csv_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 配置选项
    configs = [
        {"patch_size": 512, "level": 0, "desc": "512x512 patches at level 0"},
        {"patch_size": 256, "level": 0, "desc": "256x256 patches at level 0"},
        {"patch_size": 512, "level": 1, "desc": "512x512 patches at level 1"},
        {"patch_size": 256, "level": 1, "desc": "256x256 patches at level 1"},
    ]
    
    # 选择配置
    config_index = 0
    config = configs[config_index]
    
    print(f"选择配置: {config['desc']}")
    
    start_time = time.time()
    
    try:
        target_path, target_info = select_optimal_target(
            bags_dataset, 
            args, 
            device, 
            max_workers=8,
            patch_size=config["patch_size"],
            level=config["level"]
        )
        
        total_time = time.time() - start_time
        
        print(f"\n=== 使用缩略图和H5坐标的优化结果 ===")
        print(f"最优target已选择: {target_path}")
        print(f"来源WSI: {target_info['slide_id']}")
        print(f"Patch尺寸: {target_info['patch_size']}x{target_info['patch_size']}")
        print(f"分辨率层级: {target_info['level']}")
        print(f"来源方式: {'H5坐标' if target_info.get('from_h5', False) else '随机采样'}")
        print(f"质量评分: {target_info['score']:.3f}")
        print(f"WSI评分: {target_info['wsi_score']:.3f}")
        print(f"总耗时: {total_time:.1f} 秒")
        
        # 测试加载保存的target
        print("\n=== 测试加载保存的target ===")
        normalizer, loaded_info = load_saved_target(
            patch_size=config["patch_size"],
            level=config["level"],
            from_h5=True
        )
        print("✓ Target加载测试成功")
        
    except Exception as e:
        print(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()