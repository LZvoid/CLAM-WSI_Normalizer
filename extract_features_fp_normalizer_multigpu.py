import time
import os
# Set a default value for HF_ENDPOINT if it is not already defined
os.environ['HF_ENDPOINT'] = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')

# 必须在导入torch之前设置多进程启动方法
import multiprocessing as mp
try:
	mp.set_start_method('spawn', force=True)
	print("多进程启动方法设置为 'spawn'")
except RuntimeError as e:
	print(f"多进程启动方法设置失败或已设置: {e}")

import argparse
from functools import partial
from multiprocessing import Process, Queue, Manager
import threading

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
from wsi_normalizer import imread, TorchVahadaneNormalizer ,MacenkoNormalizer  # or ReinhardNormalizer, VahadaneNormalizer, TorchVahadaneNormalizer

# 检查可用的GPU数量
if torch.cuda.is_available():
	device = torch.device('cuda')
	num_gpus = torch.cuda.device_count()
	print(f"Available GPUs: {num_gpus}")
	for i in range(num_gpus):
		print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
	device = torch.device('cpu')
	num_gpus = 0
	print("CUDA not available, using CPU")

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
		try:
			norm_img = normalizer.transform(img_tensor)
			normalized_batch.append(norm_img)
		except Exception as e:
			# 如果归一化失败，使用原始图像
			norm_img = img_tensor.float() / 255.0
			norm_img = norm_img.permute(2, 0, 1)  # HWC -> CHW
			normalized_batch.append(norm_img.permute(1, 2, 0))  # 保持HWC格式用于stack
	
	# 一次性堆叠和转换
	batch_norm = torch.stack(normalized_batch, dim=0)  # (B,H,W,C)
	batch_norm = batch_norm.permute(0, 3, 1, 2).float() / 255.0  # (B,C,H,W)
	
	return batch_norm

def preload_normalizer(device, target_path="target_imgs/optimal_target_512x512_level0_h5.png"):
	"""
	预加载和预热normalizer
	"""
	normalizer = TorchVahadaneNormalizer(staintools_estimate=False)
	target = imread(target_path)
	target = torch.from_numpy(target).to(device)
	normalizer.fit(target)
	
	# 创建一个小的测试图像来预热GPU
	test_img = torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8, device=device)
	
	# 预热normalizer
	try:
		_ = normalizer.transform(test_img)
		print(f"GPU {device}: Normalizer预热完成")
	except Exception as e:
		print(f"GPU {device}: Normalizer预热失败: {e}")
	
	return normalizer

class OptimizedDataLoader:
	"""
	优化的数据加载器，使用预取和并行加载
	在多进程环境中使用时会自动调整参数避免pickle错误
	"""
	def __init__(self, dataset, batch_size, num_workers=8, prefetch_factor=2):
		# 在已经处于多进程环境中时，避免嵌套多进程
		if num_workers > 0:
			try:
				# 检查是否在子进程中
				if mp.current_process().name != 'MainProcess':
					num_workers = 0  # 在子进程中强制使用单进程
					print(f"检测到子进程环境，设置num_workers=0避免pickle错误")
			except:
				pass
		
		# 构建DataLoader参数
		loader_kwargs = {
			'dataset': dataset,
			'batch_size': batch_size,
			'num_workers': num_workers,
			'pin_memory': True if num_workers > 0 else False,
			'persistent_workers': True if num_workers > 0 else False,
			'drop_last': False
		}
		
		# 只有当num_workers > 0时才设置prefetch_factor
		if num_workers > 0:
			loader_kwargs['prefetch_factor'] = prefetch_factor
		
		self.loader = DataLoader(**loader_kwargs)
		
		print(f"DataLoader配置: batch_size={batch_size}, num_workers={num_workers}")
	
	def __iter__(self):
		return iter(self.loader)
	
	def __len__(self):
		return len(self.loader)
	"""
	args:
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		verbose: level of feedback
		gpu_id: specific GPU ID for this process
	"""
def compute_w_loader_optimized(output_path, loader, model, normalizer, verbose=0, gpu_id=0):
	"""
	优化的特征提取函数，使用高效批量归一化
	args:
		output_path: directory to save computed features (.h5 file)
		loader: pytorch data loader
		model: pytorch model
		normalizer: 归一化器
		verbose: level of feedback
		gpu_id: specific GPU ID for this process
	"""
	device = torch.device(f'cuda:{gpu_id}')
	model = model.to(device)
	
	if verbose > 0:
		print(f'GPU {gpu_id}: processing a total of {len(loader)} batches')

	mode = 'w'
	
	# 预分配内存用于缓存
	features_cache = []
	coords_cache = []
	cache_size = 0
	max_cache_size = 1000  # 每1000个样本写入一次
	
	total_patches = 0
	failed_patches = 0

	for count, data in enumerate(tqdm(loader, desc=f'GPU {gpu_id}')):
		with torch.inference_mode():
			batch = data['img']
			coords = data['coord'].numpy().astype(np.int32)
			batch = batch.to(device, non_blocking=True).float()
			
			# 显示当前batch在哪个GPU上处理
			if verbose > 1 and count % 10 == 0:
				print(f"GPU {gpu_id} - Batch {count}: data shape {batch.shape}")
			
			total_patches += len(batch)
			
			try:
				# 高效批量归一化
				batch_normalized = batch_normalize_efficient(batch, normalizer)
				
				# 特征提取
				features = model(batch_normalized)
				features = features.cpu().numpy().astype(np.float32)
				
			except Exception as e:
				# 如果批量归一化失败，使用原始图像
				failed_patches += len(batch)
				if verbose > 0:
					print(f"GPU {gpu_id}: Batch normalization failed for batch {count}: {e}")
				
				features = model(batch)
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
	
	# 计算失败百分比
	failure_percentage = (failed_patches / total_patches * 100) if total_patches > 0 else 0
	print(f"\nGPU {gpu_id} Statistics:")
	print(f"Total patches processed: {total_patches}")
	print(f"Failed normalizations: {failed_patches}")
	print(f"Failure percentage: {failure_percentage:.2f}%")
	
	return output_path, failure_percentage


def process_single_wsi(wsi_info, args, gpu_id, results_queue, progress_queue):
	"""
	处理单个WSI的函数，在指定GPU上运行
	"""
	wsi = None  # 初始化WSI对象
	try:
		# 在子进程中重新设置CUDA设备和环境
		torch.cuda.set_device(gpu_id)
		device = torch.device(f'cuda:{gpu_id}')
		
		# 清理CUDA缓存
		torch.cuda.empty_cache()
		
		slide_id, bag_candidate_idx, total = wsi_info
		bag_name = slide_id + '.h5'
		h5_file_path = os.path.join(args.data_h5_dir, bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
		
		print(f'GPU {gpu_id}: Processing {slide_id} ({bag_candidate_idx+1}/{total})')
		
		# 检查文件是否存在
		if not os.path.exists(h5_file_path):
			print(f'GPU {gpu_id}: H5 file not found: {h5_file_path}')
			progress_queue.put(1)
			return
		
		if not os.path.exists(slide_file_path):
			print(f'GPU {gpu_id}: Slide file not found: {slide_file_path}')
			progress_queue.put(1)
			return
		
		# 检查是否跳过
		dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))
		if not args.no_auto_skip and slide_id + '.pt' in dest_files:
			print(f'GPU {gpu_id}: Skipped {slide_id}')
			progress_queue.put(1)
			return
		
		# 在子进程中重新初始化模型和归一化器
		model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
		model = model.eval().to(device)
		
		# 为当前GPU创建和预热normalizer
		normalizer = preload_normalizer(device)
		
		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()
		
		wsi = openslide.open_slide(slide_file_path)
		dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
									wsi=wsi, 
									img_transforms=img_transforms)

		# 使用单进程数据加载器避免pickle问题
		optimized_loader = OptimizedDataLoader(
			dataset, 
			batch_size=args.batch_size,
			num_workers=0,  # 重要：设置为0避免嵌套多进程导致pickle错误
			prefetch_factor=2  # 当num_workers=0时会被自动忽略
		)
		
		output_file_path, failure_percentage = compute_w_loader_optimized(output_path, 
																		  loader=optimized_loader, 
																		  model=model,
																		  normalizer=normalizer,
																		  verbose=1, 
																		  gpu_id=gpu_id)
		
		time_elapsed = time.time() - time_start
		print(f'GPU {gpu_id}: Computing features for {slide_id} took {time_elapsed:.2f}s')
		print(f'GPU {gpu_id}: Normalization failure rate for {slide_id}: {failure_percentage:.2f}%')
		
		# 保存.pt文件
		with h5py.File(output_file_path, "r") as file:
			features = file['features'][:]
			print(f'GPU {gpu_id}: features size for {slide_id}: {features.shape}')
			print(f'GPU {gpu_id}: coordinates size for {slide_id}: {file["coords"].shape}')

		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base + '.pt'))
		
		# 清理GPU缓存
		torch.cuda.empty_cache()
		
		# 将结果放入队列
		result = {
			'slide_id': slide_id,
			'failure_percentage': failure_percentage,
			'processing_time': time_elapsed,
			'gpu_id': gpu_id
		}
		results_queue.put(result)
		progress_queue.put(1)
		
	except Exception as e:
		print(f'GPU {gpu_id}: Error processing {slide_id}: {str(e)}')
		import traceback
		traceback.print_exc()
		progress_queue.put(1)
	finally:
		if wsi is not None:
			try:
				wsi.close()
			except:
				pass


def distribute_wsis_to_gpus(wsi_list, num_gpus):
	"""
	将WSI列表分配给不同的GPU
	"""
	gpu_assignments = [[] for _ in range(num_gpus)]
	
	for i, wsi in enumerate(wsi_list):
		gpu_id = i % num_gpus
		gpu_assignments[gpu_id].append(wsi)
	
	return gpu_assignments


def process_gpu_batch(wsi_batch, args, gpu_id, results_queue, progress_queue):
	"""
	在指定GPU上处理一批WSI
	"""
	try:
		# 在子进程中重新设置CUDA设备
		torch.cuda.set_device(gpu_id)
		print(f"GPU {gpu_id}: Processing {len(wsi_batch)} WSIs")
		
		# 在子进程中重新初始化CUDA上下文
		device = torch.device(f'cuda:{gpu_id}')
		torch.cuda.empty_cache()
		
		for wsi_info in wsi_batch:
			process_single_wsi(wsi_info, args, gpu_id, results_queue, progress_queue)
			
	except Exception as e:
		print(f"GPU {gpu_id}: Error in batch processing: {str(e)}")
		import traceback
		traceback.print_exc()


def compute_w_loader_batch_parallel(output_path, loader, model, verbose=0, num_gpus=1, normalizer=None):
	"""
	批次级并行的优化计算函数
	"""
	device = next(model.parameters()).device
	
	if verbose > 0:
		print(f'processing a total of {len(loader)} batches')
		print(f'Using {num_gpus} GPU(s) for processing')

	mode = 'w'
	
	# 预分配内存用于缓存
	features_cache = []
	coords_cache = []
	cache_size = 0
	max_cache_size = 1000  # 每1000个样本写入一次
	
	total_patches = 0
	failed_patches = 0
	
	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():
			batch = data['img']
			coords = data['coord'].numpy().astype(np.int32)
			batch = batch.to(device, non_blocking=True).float()
			
			# 显示当前batch在哪些GPU上处理
			if verbose > 1 and count % 10 == 0:
				print(f"Batch {count}: data shape {batch.shape}, device: {batch.device}")
			
			total_patches += len(batch)
			
			try:
				# 高效批量归一化
				batch_normalized = batch_normalize_efficient(batch, normalizer)
				
				# DataParallel会自动将batch分配到多个GPU
				features = model(batch_normalized)
				features = features.cpu().numpy().astype(np.float32)
				
			except Exception as e:
				# 如果批量归一化失败，使用原始图像
				failed_patches += len(batch)
				if verbose > 0:
					print(f"Batch normalization failed for batch {count}: {e}")
				
				features = model(batch)
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
	
	# 计算失败百分比
	failure_percentage = (failed_patches / total_patches * 100) if total_patches > 0 else 0
	print(f"\nNormalization Statistics:")
	print(f"Total patches processed: {total_patches}")
	print(f"Failed normalizations: {failed_patches}")
	print(f"Failure percentage: {failure_percentage:.2f}%")
	print(f"GPU utilization: {num_gpus} GPU(s)")
	
	return output_path, failure_percentage


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default='.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--model_name', type=str, default='resnet50_trunc', choices=['resnet50_trunc', 'uni_v1', 'conch_v1', 'resnet_18','resnet50.a2_in1k','resnet50.b1k_in1k','resnet50.tv2_in1k'])
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
parser.add_argument('--gpu_ids', type=str, default=None, help='Comma-separated list of GPU IDs to use (e.g., "0,1,2")')
parser.add_argument('--monitor_gpu', default=False, action='store_true', help='Monitor GPU memory usage')
parser.add_argument('--num_processes', type=int, default=None, help='Number of parallel processes (defaults to number of GPUs)')
parser.add_argument('--data_parallel_mode', type=str, default='wsi', choices=['wsi', 'batch'], 
                   help='Data parallelism mode: "wsi" for WSI-level parallelism, "batch" for batch-level parallelism')
parser.add_argument('--prefetch_factor', type=int, default=4, help='Prefetch factor for data loading optimization')
parser.add_argument('--cache_size', type=int, default=1000, help='Cache size for batch writing optimization')
args = parser.parse_args()

def print_gpu_memory():
	"""打印GPU内存使用情况"""
	if torch.cuda.is_available():
		for i in range(torch.cuda.device_count()):
			allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
			cached = torch.cuda.memory_reserved(i) / (1024**3)  # GB
			print(f"GPU {i}: Allocated {allocated:.2f}GB, Cached {cached:.2f}GB")


if __name__ == '__main__':
	# 确保多进程启动方法设置正确
	try:
		current_method = mp.get_start_method()
		if current_method != 'spawn':
			mp.set_start_method('spawn', force=True)
			print(f"多进程启动方法从 '{current_method}' 改为 'spawn'")
		else:
			print("多进程启动方法已正确设置为 'spawn'")
	except Exception as e:
		print(f"设置多进程启动方法时出错: {e}")
	
	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	# 设置使用的GPU
	if args.gpu_ids is not None:
		gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
		print(f"Using GPUs: {gpu_ids}")
		os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
		# 重新检查设备
		if torch.cuda.is_available():
			num_gpus = len(gpu_ids)
		else:
			num_gpus = 0
	else:
		# 使用所有可用GPU
		gpu_ids = list(range(num_gpus)) if num_gpus > 0 else []
		print(f"Using all available GPUs: {gpu_ids}")

	# 设置进程数量
	if args.num_processes is None:
		args.num_processes = num_gpus if num_gpus > 0 else 1
	
	print(f"Data parallel mode: {args.data_parallel_mode}")
	print(f"Number of processes: {args.num_processes}")

	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)

	total = len(bags_dataset)
	
	# 根据并行模式选择处理方式
	if args.data_parallel_mode == 'wsi' and num_gpus > 1:
		print("Using WSI-level data parallelism")
		
		# 准备WSI列表
		wsi_list = []
		for bag_candidate_idx in range(total):
			slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
			wsi_list.append((slide_id, bag_candidate_idx, total))
		
		# 分配WSI到不同GPU
		gpu_assignments = distribute_wsis_to_gpus(wsi_list, num_gpus)
		
		# 使用spawn上下文创建多进程
		ctx = mp.get_context('spawn')
		manager = ctx.Manager()
		results_queue = manager.Queue()
		progress_queue = manager.Queue()
		
		# 启动进程
		processes = []
		for gpu_id in range(num_gpus):
			if gpu_assignments[gpu_id]:  # 只为有分配任务的GPU启动进程
				p = ctx.Process(target=process_gpu_batch, 
						   args=(gpu_assignments[gpu_id], args, gpu_id, results_queue, progress_queue))
				p.start()
				processes.append(p)
				print(f"Started process for GPU {gpu_id} with {len(gpu_assignments[gpu_id])} WSIs")
		
		# 监控进度
		completed = 0
		pbar = tqdm(total=total, desc="Overall progress")
		
		wsi_statistics = {}
		while completed < total:
			try:
				progress_queue.get(timeout=1)
				completed += 1
				pbar.update(1)
			except:
				continue
		
		pbar.close()
		
		# 等待所有进程完成
		for p in processes:
			p.join()
		
		# 收集结果
		while not results_queue.empty():
			result = results_queue.get()
			wsi_statistics[result['slide_id']] = {
				'failure_percentage': result['failure_percentage'],
				'processing_time': result['processing_time'],
				'gpu_id': result['gpu_id']
			}
		
	else:
		print("Using batch-level data parallelism (original mode)")
		
		# 使用原始的batch-level并行方式
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		
		# 创建全局normalizer并预热
		normalizer = preload_normalizer(device)
		
		dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))
		model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
		model = model.eval().to(device)
		
		# 如果有多个GPU，使用DataParallel
		if num_gpus > 1:
			print(f"Using DataParallel with {num_gpus} GPUs")
			model = nn.DataParallel(model, device_ids=gpu_ids)
			effective_batch_size = args.batch_size * num_gpus
			print(f"Effective batch size: {effective_batch_size}")
		else:
			effective_batch_size = args.batch_size
		
		# 编译模型以加速（PyTorch 2.0+）
		if hasattr(torch, 'compile'):
			try:
				model = torch.compile(model, mode='max-autotune')
				print("模型编译成功，性能将得到优化")
			except:
				print("模型编译失败，使用常规模式")
		
		loader_kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if device.type == "cuda" else {'num_workers': args.num_workers}
		wsi_statistics = {}
		
		for bag_candidate_idx in tqdm(range(total)):
			slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
			bag_name = slide_id + '.h5'
			h5_file_path = os.path.join(args.data_h5_dir, bag_name)
			slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
			
			print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
			print(slide_id)

			if not args.no_auto_skip and slide_id + '.pt' in dest_files:
				print('skipped {}'.format(slide_id))
				continue 

			output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
			time_start = time.time()
			
			try:
				wsi = openslide.open_slide(slide_file_path)
				dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
											wsi=wsi, 
											img_transforms=img_transforms)

				# 使用优化的数据加载器
				optimized_loader = OptimizedDataLoader(
					dataset, 
					batch_size=effective_batch_size,
					num_workers=args.num_workers,
					prefetch_factor=args.prefetch_factor
				)
				
				if args.monitor_gpu and torch.cuda.is_available():
					print("GPU memory before processing:")
					print_gpu_memory()
				
				# 使用优化的compute_w_loader函数
				output_file_path, failure_percentage = compute_w_loader_batch_parallel(output_path, 
																					   loader=optimized_loader, 
																					   model=model, 
																					   verbose=1, 
																					   num_gpus=num_gpus,
																					   normalizer=normalizer)
				
				if args.monitor_gpu and torch.cuda.is_available():
					print("GPU memory after processing:")
					print_gpu_memory()

				time_elapsed = time.time() - time_start
				print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
				print(f'Normalization failure rate for {slide_id}: {failure_percentage:.2f}%')
				
				wsi_statistics[slide_id] = {
					'failure_percentage': failure_percentage,
					'processing_time': time_elapsed
				}

				with h5py.File(output_file_path, "r") as file:
					features = file['features'][:]
					print('features size: ', features.shape)
					print('coordinates size: ', file['coords'].shape)

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
	
	# 保存统计信息到文件
	statistics_file = os.path.join(args.feat_dir, 'normalization_statistics.txt')
	with open(statistics_file, 'w') as f:
		f.write("WSI Normalization Statistics\n")
		f.write("=" * 70 + "\n")
		f.write(f"Parallel Mode: {args.data_parallel_mode}\n")
		f.write(f"GPU Configuration: {num_gpus} GPU(s) - {gpu_ids}\n")
		f.write(f"Number of Processes: {args.num_processes}\n")
		if args.data_parallel_mode == 'batch':
			f.write(f"Effective Batch Size: {effective_batch_size}\n")
		f.write(f"Number of Workers: {args.num_workers}\n")
		f.write("-" * 70 + "\n")
		f.write(f"{'WSI ID':<30} {'Failure %':<12} {'Time (s)':<10} {'GPU ID':<8}\n")
		f.write("-" * 70 + "\n")
		
		total_failure_rate = 0
		processed_count = 0
		total_processing_time = 0
		
		for wsi_id, stats in wsi_statistics.items():
			gpu_info = f"GPU {stats.get('gpu_id', 'N/A')}" if 'gpu_id' in stats else "Multi-GPU"
			f.write(f"{wsi_id:<30} {stats['failure_percentage']:<12.2f} {stats['processing_time']:<10.2f} {gpu_info:<8}\n")
			total_failure_rate += stats['failure_percentage']
			total_processing_time += stats['processing_time']
			processed_count += 1
		
		if processed_count > 0:
			avg_failure_rate = total_failure_rate / processed_count
			avg_processing_time = total_processing_time / processed_count
			f.write("-" * 70 + "\n")
			f.write(f"Average failure rate: {avg_failure_rate:.2f}%\n")
			f.write(f"Average processing time: {avg_processing_time:.2f}s\n")
			f.write(f"Total processing time: {total_processing_time:.2f}s\n")
			f.write(f"Total WSIs processed: {processed_count}\n")
	
	print(f"\nStatistics saved to: {statistics_file}")
	print(f"Total processing completed using {num_gpus} GPU(s) in {args.data_parallel_mode} mode")



