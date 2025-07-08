import time
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import pdb
from functools import partial

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
from tqdm import tqdm
import numpy as np
from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder
import cv2
from wsi_normalizer import imread, TorchVahadaneNormalizer, MacenkoNormalizer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 全局初始化normalizer
normalizer = TorchVahadaneNormalizer(staintools_estimate=False)
target = imread("target_imgs/optimal_target_512x512_level0_h5.png")
target = torch.from_numpy(target).to(device)
normalizer.fit(target)

def batch_normalize_efficient(batch, normalizer):
    """
    高效的批量归一化处理
    """
    batch_size, c, h, w = batch.shape
    
    # 一次性转换整个batch到HWC格式
    batch_hwc = batch.permute(0, 2, 3, 1)  # (B,C,H,W) -> (B,H,W,C)
    
    # 一次性转换到uint8
    batch_uint8 = (batch_hwc * 255).clamp(0, 255).to(torch.uint8)
    
    # 批量归一化处理
    normalized_batch = []
    
    # 如果normalizer支持批量处理，使用批量模式
    if hasattr(normalizer, 'transform_batch') and callable(getattr(normalizer, 'transform_batch')):
        # 使用批量归一化（如果支持）
        try:
            norm_batch = normalizer.transform_batch(batch_uint8)
            norm_batch = norm_batch.permute(0, 3, 1, 2).float() / 255.0
            return norm_batch
        except:
            pass
    
    # 优化的逐个处理（减少数据传输开销）
    for i in range(batch_size):
        img_tensor = batch_uint8[i]  # 已经在GPU上
        norm_img = normalizer.transform(img_tensor)
        normalized_batch.append(norm_img)
    
    # 一次性堆叠和转换
    batch_norm = torch.stack(normalized_batch, dim=0)  # (B,H,W,C)
    batch_norm = batch_norm.permute(0, 3, 1, 2).float() / 255.0  # (B,C,H,W)
    
    return batch_norm

def compute_w_loader_optimized(output_path, loader, model, normalizer, verbose=0):
    """
    优化的特征提取函数
    """
    if verbose > 0:
        print(f'processing a total of {len(loader)} batches')

    mode = 'w'
    
    # 预分配内存用于缓存
    features_cache = []
    coords_cache = []
    cache_size = 0
    max_cache_size = 1000  # 每1000个样本写入一次

    for count, data in enumerate(tqdm(loader)):
        with torch.inference_mode():
            batch = data['img']
            coords = data['coord'].numpy().astype(np.int32)
            batch = batch.to(device, non_blocking=True).float()
            
            # 高效批量归一化
            batch_normalized = batch_normalize_efficient(batch, normalizer)
            
            # 特征提取
            features = model(batch_normalized)
            features = features.cpu().numpy().astype(np.float32)
            
            # 缓存结果
            features_cache.append(features)
            coords_cache.append(coords)
            cache_size += len(features)
            
            # 批量写入，减少I/O次数
            if cache_size >= max_cache_size or count == len(loader) - 1:
                # 合并缓存的数据
                combined_features = np.vstack(features_cache)
                combined_coords = np.vstack(coords_cache)
                
                asset_dict = {'features': combined_features, 'coords': combined_coords}
                save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
                mode = 'a'
                
                # 清空缓存
                features_cache = []
                coords_cache = []
                cache_size = 0
    
    return output_path

def preload_normalizer():
    """
    预加载和预热normalizer
    """
    # 创建一个小的测试图像来预热GPU
    test_img = torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8, device=device)
    
    # 预热normalizer
    try:
        _ = normalizer.transform(test_img)
        print("Normalizer预热完成")
    except Exception as e:
        print(f"Normalizer预热失败: {e}")

class OptimizedDataLoader:
    """
    优化的数据加载器，使用预取和并行加载
    """
    def __init__(self, dataset, batch_size, num_workers=8, prefetch_factor=2):
        self.loader = DataLoader(
            dataset=dataset, 
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            persistent_workers=True if num_workers > 0 else False,
            drop_last=False
        )
    
    def __iter__(self):
        return iter(self.loader)
    
    def __len__(self):
        return len(self.loader)

def main():
    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('--data_h5_dir', type=str, default=None)
    parser.add_argument('--data_slide_dir', type=str, default=None)
    parser.add_argument('--slide_ext', type=str, default='.svs')
    parser.add_argument('--csv_path', type=str, default=None)
    parser.add_argument('--feat_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='resnet50_trunc', 
                       choices=['resnet50_trunc', 'uni_v1', 'conch_v1', 'resnet_18',
                               'resnet50.a2_in1k','resnet50.b1k_in1k','resnet50.tv2_in1k'])
    parser.add_argument('--batch_size', type=int, default=32)  # 增加batch size
    parser.add_argument('--no_auto_skip', default=False, action='store_true')
    parser.add_argument('--target_patch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--prefetch_factor', type=int, default=4)
    args = parser.parse_args()

    print('初始化dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    bags_dataset = Dataset_All_Bags(csv_path)
    
    # 创建输出目录
    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

    # 预热normalizer
    preload_normalizer()
    
    # 加载模型
    model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
    model = model.eval().to(device)
    
    # 编译模型以加速（PyTorch 2.0+）
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='max-autotune')
            print("模型编译成功，性能将得到优化")
        except:
            print("模型编译失败，使用常规模式")
    
    total = len(bags_dataset)

    # 主处理循环
    for bag_candidate_idx in tqdm(range(total), desc="处理WSI"):
        slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, bag_name)
        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
        
        print(f'\n进度: {bag_candidate_idx+1}/{total}')
        print(f'处理: {slide_id}')

        if not args.no_auto_skip and slide_id + '.pt' in dest_files:
            print(f'跳过已处理的文件: {slide_id}')
            continue 

        output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
        time_start = time.time()
        
        try:
            wsi = openslide.open_slide(slide_file_path)
            dataset = Whole_Slide_Bag_FP(
                file_path=h5_file_path, 
                wsi=wsi, 
                img_transforms=img_transforms
            )

            # 使用优化的数据加载器
            optimized_loader = OptimizedDataLoader(
                dataset, 
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor
            )
            
            # 使用优化的计算函数
            output_file_path = compute_w_loader_optimized(
                output_path, 
                loader=optimized_loader, 
                model=model, 
                normalizer=normalizer,
                verbose=1
            )

            time_elapsed = time.time() - time_start
            print(f'\n{slide_id} 特征提取完成，耗时: {time_elapsed:.1f}s')

            # 验证输出并保存pt文件
            with h5py.File(output_file_path, "r") as file:
                features = file['features'][:]
                coords = file['coords'][:]
                print(f'特征维度: {features.shape}')
                print(f'坐标维度: {coords.shape}')

            features = torch.from_numpy(features)
            bag_base, _ = os.path.splitext(bag_name)
            torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base + '.pt'))
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f'处理 {slide_id} 时出错: {e}')
            continue
        finally:
            if 'wsi' in locals():
                wsi.close()

    print("所有文件处理完成！")

if __name__ == '__main__':
    main()