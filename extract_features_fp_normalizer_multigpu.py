import time
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import pdb
from functools import partial
import multiprocessing as mp
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

def compute_w_loader(output_path, loader, model, verbose = 0, gpu_id = 0):
	"""
	args:
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		verbose: level of feedback
		gpu_id: specific GPU ID for this process
	"""
	device = torch.device(f'cuda:{gpu_id}')
	model = model.to(device)
	
	# 为当前GPU创建normalizer
	normalizer = TorchVahadaneNormalizer(staintools_estimate=False)
	target = imread("target_imgs/c16_1.png")
	target = torch.from_numpy(target).to(device)
	normalizer.fit(target)
	
	if verbose > 0:
		print(f'GPU {gpu_id}: processing a total of {len(loader)} batches')

	mode = 'w'
	total_patches = 0
	failed_patches = 0
	
	for count, data in enumerate(tqdm(loader, desc=f'GPU {gpu_id}')):
		with torch.inference_mode():
			batch = data['img']
			coords = data['coord'].numpy().astype(np.int32)
			batch = batch.to(device, non_blocking=True).float()
			batch_norm = []
			
			# 显示当前batch在哪个GPU上处理
			if verbose > 1 and count % 10 == 0:
				print(f"GPU {gpu_id} - Batch {count}: data shape {batch.shape}")
			
			for img_tensor in batch:
				total_patches += 1
				# 转 HWC
				img_tensor = img_tensor.permute(1, 2, 0)  # (C,H,W) -> (H,W,C)

				# 转 uint8
				img_tensor = (img_tensor * 255).clamp(0, 255).to(torch.uint8)

				try:
					# ⚠️ 如果 normalizer 用 StainExtractorGPU，需要 staintools_estimate=False
					#注意，使用 TorchVahadaneNormalizer 时，需要修改其transform方法的返回值
					#将out = out.detach().cpu().numpy()这一行注释掉
					norm_img = normalizer.transform(img_tensor)  # 返回 torch.Tensor (H,W,C) on CUDA
					
					# 转回 CHW + float
					norm_img = norm_img.permute(2, 0, 1).float() / 255.0
				except Exception as e:
					# 如果归一化失败，使用原始图像
					failed_patches += 1
					if verbose > 0:
						print(f"GPU {gpu_id}: Normalization failed for patch {total_patches}: {e}")
					# 确保 img_tensor 是 torch tensor
					if isinstance(img_tensor, np.ndarray):
						img_tensor = torch.from_numpy(img_tensor).to(device)
					norm_img = img_tensor.permute(2, 0, 1).float() / 255.0

				batch_norm.append(norm_img)

			# 拼 batch
			batch = torch.stack(batch_norm, dim=0)  # (B,C,H,W)，已在 CUDA
			
			# 单GPU处理
			features = model(batch)
			features = features.cpu().numpy().astype(np.float32)

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
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
	try:
		torch.cuda.set_device(gpu_id)
		device = torch.device(f'cuda:{gpu_id}')
		
		slide_id, bag_candidate_idx, total = wsi_info
		bag_name = slide_id + '.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
		
		print(f'GPU {gpu_id}: Processing {slide_id} ({bag_candidate_idx+1}/{total})')
		
		# 检查是否跳过
		dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))
		if not args.no_auto_skip and slide_id + '.pt' in dest_files:
			print(f'GPU {gpu_id}: Skipped {slide_id}')
			progress_queue.put(1)
			return
		
		# 初始化模型
		model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
		model = model.eval().to(device)
		
		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()
		
		wsi = openslide.open_slide(slide_file_path)
		dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
									wsi=wsi, 
									img_transforms=img_transforms)

		loader_kwargs = {'num_workers': max(1, args.num_workers // args.num_processes), 'pin_memory': True}
		loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
		
		output_file_path, failure_percentage = compute_w_loader(output_path, 
															   loader=loader, 
															   model=model, 
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
		# 设置CUDA设备
		torch.cuda.set_device(gpu_id)
		print(f"GPU {gpu_id}: Processing {len(wsi_batch)} WSIs")
		
		for wsi_info in wsi_batch:
			process_single_wsi(wsi_info, args, gpu_id, results_queue, progress_queue)
			
	except Exception as e:
		print(f"GPU {gpu_id}: Error in batch processing: {str(e)}")
		import traceback
		traceback.print_exc()


def compute_w_loader_batch_parallel(output_path, loader, model, verbose=0, num_gpus=1, normalizer=None):
	"""
	批次级并行的计算函数（保持与原始版本兼容）
	"""
	device = next(model.parameters()).device
	
	if verbose > 0:
		print(f'processing a total of {len(loader)} batches')
		print(f'Using {num_gpus} GPU(s) for processing')

	mode = 'w'
	total_patches = 0
	failed_patches = 0
	
	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():
			batch = data['img']
			coords = data['coord'].numpy().astype(np.int32)
			batch = batch.to(device, non_blocking=True).float()
			batch_norm = []
			
			# 显示当前batch在哪些GPU上处理
			if verbose > 1 and count % 10 == 0:
				print(f"Batch {count}: data shape {batch.shape}, device: {batch.device}")
			
			for img_tensor in batch:
				total_patches += 1
				# 转 HWC
				img_tensor = img_tensor.permute(1, 2, 0)  # (C,H,W) -> (H,W,C)

				# 转 uint8
				img_tensor = (img_tensor * 255).clamp(0, 255).to(torch.uint8)

				try:
					# ⚠️ 如果 normalizer 用 StainExtractorGPU，需要 staintools_estimate=False
					#注意，使用 TorchVahadaneNormalizer 时，需要修改其transform方法的返回值
					#将out = out.detach().cpu().numpy()这一行注释掉
					norm_img = normalizer.transform(img_tensor)  # 返回 torch.Tensor (H,W,C) on CUDA
					
					# 转回 CHW + float
					norm_img = norm_img.permute(2, 0, 1).float() / 255.0
				except Exception as e:
					# 如果归一化失败，使用原始图像
					failed_patches += 1
					if verbose > 0:
						print(f"Normalization failed for patch {total_patches}: {e}")
					# 确保 img_tensor 是 torch tensor
					if isinstance(img_tensor, np.ndarray):
						img_tensor = torch.from_numpy(img_tensor).to(device)
					norm_img = img_tensor.permute(2, 0, 1).float() / 255.0

				batch_norm.append(norm_img)

			# 拼 batch
			batch = torch.stack(batch_norm, dim=0)  # (B,C,H,W)，已在 CUDA
			
			# DataParallel会自动将batch分配到多个GPU
			features = model(batch)
			features = features.cpu().numpy().astype(np.float32)

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
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
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--model_name', type=str, default='resnet50_trunc', choices=['resnet50_trunc', 'uni_v1', 'conch_v1', 'resnet_18','resnet50.a2_in1k','resnet50.b1k_in1k','resnet50.tv2_in1k'])
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
parser.add_argument('--gpu_ids', type=str, default=None, help='Comma-separated list of GPU IDs to use (e.g., "0,1,2")')
parser.add_argument('--monitor_gpu', default=False, action='store_true', help='Monitor GPU memory usage')
parser.add_argument('--num_processes', type=int, default=None, help='Number of parallel processes (defaults to number of GPUs)')
parser.add_argument('--data_parallel_mode', type=str, default='wsi', choices=['wsi', 'batch'], 
                   help='Data parallelism mode: "wsi" for WSI-level parallelism, "batch" for batch-level parallelism')
args = parser.parse_args()

def print_gpu_memory():
	"""打印GPU内存使用情况"""
	if torch.cuda.is_available():
		for i in range(torch.cuda.device_count()):
			allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
			cached = torch.cuda.memory_reserved(i) / (1024**3)  # GB
			print(f"GPU {i}: Allocated {allocated:.2f}GB, Cached {cached:.2f}GB")


if __name__ == '__main__':
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
		
		# 创建多进程
		processes = []
		manager = Manager()
		results_queue = manager.Queue()
		progress_queue = manager.Queue()
		
		# 启动进程
		for gpu_id in range(num_gpus):
			if gpu_assignments[gpu_id]:  # 只为有分配任务的GPU启动进程
				p = Process(target=process_gpu_batch, 
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
		
		# 创建全局normalizer
		normalizer = TorchVahadaneNormalizer(staintools_estimate=False)
		target = imread("target_imgs/c16_1.png")
		target = torch.from_numpy(target).to(device)
		normalizer.fit(target)
		
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
		
		loader_kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if device.type == "cuda" else {'num_workers': args.num_workers}
		wsi_statistics = {}
		
		for bag_candidate_idx in tqdm(range(total)):
			slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
			bag_name = slide_id + '.h5'
			h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
			slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
			
			print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
			print(slide_id)

			if not args.no_auto_skip and slide_id + '.pt' in dest_files:
				print('skipped {}'.format(slide_id))
				continue 

			output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
			time_start = time.time()
			wsi = openslide.open_slide(slide_file_path)
			dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
										wsi=wsi, 
										img_transforms=img_transforms)

			loader = DataLoader(dataset=dataset, batch_size=effective_batch_size, **loader_kwargs)
			
			if args.monitor_gpu and torch.cuda.is_available():
				print("GPU memory before processing:")
				print_gpu_memory()
			
			# 使用原始的compute_w_loader函数（需要适配）
			output_file_path, failure_percentage = compute_w_loader_batch_parallel(output_path, 
																				   loader=loader, 
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
	
	# 保存统计信息到文件
	statistics_file = os.path.join(args.feat_dir, 'normalization_statistics.txt')
	with open(statistics_file, 'w') as f:
		f.write("WSI Normalization Statistics\n")
		f.write("=" * 70 + "\n")
		f.write(f"Parallel Mode: {args.data_parallel_mode}\n")
		f.write(f"GPU Configuration: {num_gpus} GPU(s) - {gpu_ids}\n")
		f.write(f"Number of Processes: {args.num_processes}\n")
		if args.data_parallel_mode == 'batch':
			f.write(f"Effective Batch Size: {effective_batch_size if 'effective_batch_size' in locals() else args.batch_size}\n")
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



