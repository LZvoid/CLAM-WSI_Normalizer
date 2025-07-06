# CLAM-WSI Normalizer

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

基于 [CLAM](https://github.com/mahmoodlab/CLAM)（Clustering-constrained Attention Multiple Instance Learning）的扩展版本，**专为病理切片染色归一化和多中心数据处理而设计**。本项目新增了染色归一化功能，显著提升了跨中心、跨批次病理数据的一致性和模型泛化能力。

## ✨ 主要特性

### 🎨 染色归一化支持
- **多种归一化算法**：支持 Macenko、Vahadane、TorchVahadane 等主流归一化方法
- **自定义目标图像**：可指定标准化参考图像，确保色彩一致性
- **GPU 加速**：利用 TorchVahadane 实现高效 GPU 加速归一化

### 🔬 完整的 WSI 处理流程
- **智能分割**：自动组织区域分割和背景移除
- **灵活切块**：支持多尺度 patch 提取（256px, 512px 等）
- **特征提取**：基于预训练模型（ResNet50、UNI、CONCH 等）
- **MIL 学习**：支持 CLAM-SB/MB 等多实例学习模型

### 📊 可视化与分析
- **热力图生成**：可视化模型关注区域
- **批量处理**：支持大规模数据集自动化处理
- **实验管理**：完整的实验配置和结果追踪

## 🚀 快速开始

### 环境安装

```bash
# 创建 conda 环境
conda env create -f env.yml
conda activate clam_latest

# 或使用 pip 安装
pip install torch torchvision timm h5py pandas opencv-python scikit-learn matplotlib tqdm openslide-python
pip install git+https://github.com/EIDOSLAB/wsi-normalizer.git
```

### 基本使用流程

#### 1️⃣ WSI 切块处理

```bash
# 标准切块（无归一化）
python create_patches_fp.py \
  --source /path/to/WSI \
  --save_dir /path/to/output \
  --patch_size 512 \
  --patch --seg --stitch

# 归一化切块处理
python create_patches_fp_normalizer.py \
  --source /path/to/WSI \
  --save_dir /path/to/output \
  --patch_size 512 \
  --patch --seg --stitch
```

#### 2️⃣ 特征提取

```bash
# 标准特征提取
python extract_features_fp.py \
  --data_h5_dir /path/to/patches \
  --data_slide_dir /path/to/WSI \
  --csv_path /path/to/process_list.csv \
  --feat_dir /path/to/features \
  --batch_size 512 \
  --slide_ext .svs

# 染色归一化特征提取
python extract_features_fp_normalizer.py \
  --data_h5_dir /path/to/patches \
  --data_slide_dir /path/to/WSI \
  --csv_path /path/to/process_list.csv \
  --feat_dir /path/to/features \
  --batch_size 16 \
  --slide_ext .svs \
  --target_patch_size 512
```

#### 3️⃣ 数据集分割

```bash
python create_splits_seq.py \
  --label_frac 0.75 \
  --seed 1 \
  --k 10 \
  --task task_1_tumor_vs_normal \
  --val_frac 0.1 \
  --test_frac 0.1
```

#### 4️⃣ 模型训练

```bash
python main.py \
  --drop_out 0.25 \
  --early_stopping \
  --lr 2e-4 \
  --k 10 \
  --label_frac 1 \
  --exp_code tumor_vs_normal \
  --weighted_sample \
  --bag_loss ce \
  --inst_loss svm \
  --task task_1_tumor_vs_normal \
  --model_type clam_sb \
  --log_data \
  --data_root_dir /path/to/features
```

#### 5️⃣ 热力图生成

```bash
python create_heatmaps.py --config heatmaps/configs/config_template.yaml
```

## 📁 项目结构

```
CLAM-WSI_Normalizer/
├── 📄 主要脚本
│   ├── create_patches_fp.py              # WSI 切块处理
│   ├── create_patches_fp_normalizer.py   # 归一化切块处理
│   ├── extract_features_fp.py            # 标准特征提取
│   ├── extract_features_fp_normalizer.py # 归一化特征提取
│   ├── main.py                           # 模型训练主脚本
│   ├── eval.py                           # 模型评估
│   └── create_heatmaps.py                # 热力图生成
├── 📂 核心模块
│   ├── models/                           # 模型定义（CLAM、MIL 等）
│   ├── dataset_modules/                  # 数据加载器
│   ├── utils/                           # 工具函数
│   ├── wsi_core/                        # WSI 处理核心
│   └── vis_utils/                       # 可视化工具
├── 📂 配置与数据
│   ├── presets/                         # 预设配置文件
│   ├── heatmaps/configs/                # 热力图配置
│   ├── target_imgs/                     # 归一化目标图像
│   └── splits/                          # 数据集分割结果
└── 📄 环境配置
    ├── env.yml                          # Conda 环境文件
    └── LICENSE.md                       # GPL v3 许可证
```

## ⚙️ 核心参数说明

### 切块参数
- `--patch_size`：patch 尺寸（256, 512 等）
- `--patch_level`：提取层级（0 为最高分辨率）
- `--overlap`：patch 重叠率
- `--seg_level`：分割处理层级

### 归一化参数
- `--normalizer`：归一化方法（Macenko, Vahadane, TorchVahadane）
- `--target_img`：目标参考图像路径
- `--target_patch_size`：归一化后 patch 尺寸

### 模型参数
- `--model_type`：模型类型（clam_sb, clam_mb, mil）
- `--bag_loss`：bag-level 损失函数（ce, nll）
- `--inst_loss`：instance-level 损失函数（svm, ce）
- `--embed_dim`：特征维度（1024, 2048）

## 🔧 高级配置

### 1. 自定义归一化目标

```python
# 在 extract_features_fp_normalizer.py 中修改
target = imread("target_imgs/your_target.png")
normalizer = TorchVahadaneNormalizer(staintools_estimate=False)
normalizer.fit(target)
```

### 2. 多 GPU 训练

```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py [其他参数]
```

### 3. 自定义预训练模型

```python
# 在 models/builder.py 中添加新模型
def get_encoder(model_name, target_img_size=224):
    if model_name == 'your_model':
        model = YourModel(pretrained=True)
        # ... 配置代码
```

## 📈 性能优化建议

1. **批次大小调优**：归一化特征提取建议 batch_size=16-32，标准特征提取可用 512+
2. **内存管理**：处理大型 WSI 时建议设置 `--no_auto_skip` 避免内存溢出
3. **GPU 利用**：使用 TorchVahadane 可显著加速归一化过程
4. **并行处理**：支持多进程 WSI 处理，提升整体效率

## 🐛 常见问题

<details>
<summary><strong>Q: OpenSlide 安装失败怎么办？</strong></summary>

**A:** 在 Windows 上：
```bash
conda install -c conda-forge openslide
```
在 Linux 上：
```bash
sudo apt-get install openslide-tools
pip install openslide-python
```
</details>

<details>
<summary><strong>Q: 归一化后颜色异常？</strong></summary>

**A:** 检查目标图像质量，确保：
- 目标图像为高质量 H&E 染色切片
- 图像格式为 RGB（非 RGBA）
- 尝试不同的归一化算法
</details>

<details>
<summary><strong>Q: 内存不足错误？</strong></summary>

**A:** 降低批次大小，或使用：
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
```
</details>

## 📚 相关资源

- **论文**: [Data-efficient and weakly supervised computational pathology on whole-slide images](https://www.nature.com/articles/s41551-021-00732-z)
- **原始 CLAM**: [mahmoodlab/CLAM](https://github.com/mahmoodlab/CLAM)
- **WSI Normalizer**: [EIDOSLAB/wsi-normalizer](https://github.com/EIDOSLAB/wsi-normalizer)
- **CLAM 教程**: [官方文档](https://github.com/mahmoodlab/CLAM/blob/master/docs/TUTORIAL.md)

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！在贡献前请：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 [GPL v3](LICENSE.md) 许可证 - 详见 LICENSE.md 文件。

## 🙏 致谢

- [CLAM 原始项目](https://github.com/mahmoodlab/CLAM) - Mahmood Lab, Harvard Medical School
- [WSI Normalizer](https://github.com/EIDOSLAB/wsi-normalizer) - EIDOSLAB
- [Staintools](https://github.com/Peter554/StainTools) - 染色归一化算法实现

---

<p align="center">
  <sub>⭐ 如果这个项目对您有帮助，请给个 Star！</sub>
</p>