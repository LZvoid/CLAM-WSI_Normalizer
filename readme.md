# CLAM 扩展版

本项目基于 CLAM（Clustering-constrained Attention Multiple Instance Learning）项目，**新增了提取染色归一化后 patch 特征的功能**，适用于病理切片的高通量特征分析和多中心数据一致性处理。

## 主要功能

- 支持在特征提取前对每个 patch 进行染色归一化（支持 Macenko、Vahadane、TorchVahadane 等方法）
- 可自定义归一化目标图像，提升多批次、多中心数据的色彩一致性
- 兼容原有 CLAM 的 patch 切割、特征提取、下游 MIL 训练流程

## 主要脚本

- `create_patches_fp.py`  
  切割 WSI 并保存 patch 信息到 h5 文件
- `extract_features_fp.py`  
  提取 patch 特征（无归一化）
- `extract_features_fp_normalizer.py`  
  **提取染色归一化后的 patch 特征**，需指定目标归一化图像

## 使用方法

### 1. 切割 patch

```bash
python create_patches_fp.py --source /path/to/WSI --save_dir /path/to/output --patch_size 512 --patch --seg --stitch
```

### 2. 提取归一化特征

```bash
python extract_features_fp_normalizer.py \
  --data_h5_dir /path/to/output \
  --data_slide_dir /path/to/WSI \
  --csv_path /path/to/output/process_list_autogen.csv \
  --feat_dir /path/to/features \
  --batch_size 16 \
  --slide_ext .svs \
  --target_patch_size 512
```

> **注意：**  
> - 需在 `extract_features_fp_normalizer.py` 中设置好归一化目标图像路径（如 `target_imgs/c16_1.png`）。
> - 染色归一化方法可选 Macenko、Vahadane、TorchVahadane 等，按需切换。

## 环境依赖

- Python 3.8+
- torch, torchvision
- timm
- h5py
- openslide-python
- opencv-python
- wsi_normalizer
- tqdm
- 其他依赖详见 `requirements.txt`

## 致谢

- [CLAM 原始项目](https://github.com/mahmoodlab/CLAM)
- [wsi_normalizer](https://github.com/EIDOSLAB/wsi-normalizer)

---