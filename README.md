# Wave-Mamba for Flare Removal

基于 Wave-Mamba (ACMMM2024) 改造的炫光去除模型

## 核心改造

1. **输出机制转换**：从加性残差改为减性残差（`clean = input - flare_pred`）
2. **损失函数**：光晕预测损失 + 感知损失 + FFT 频域损失
3. **轻量化设计**：wf=16, n_l_blocks=[1,1,1], n_h_blocks=[1,1,1]

## 文件结构

```
flare-removal/
├── Wave-Mamba/
│   ├── basicsr/
│   │   ├── archs/
│   │   │   ├── wavemamba_flare_arch.py   # 新架构（核心改造）
│   │   │   └── wavemamba_arch.py          # 原版架构
│   │   ├── models/
│   │   │   ├── wavemamba_flare_model.py   # 训练模型类
│   │   │   ├── base_model.py
│   │   │   └── cal_ssim.py
│   │   ├── losses/
│   │   │   └── losses.py                  # 新增 L_Abs_pure, L_percepture
│   │   ├── data/
│   │   │   ├── flare7kpp_dataset.py       # Flare7K++ 数据集加载
│   │   │   └── flare7k_dataset.py
│   │   ├── utils/
│   │   ├── train.py
│   │   └── __init__.py
│   └── options/
│       └── train_wavemamba_flare.yml       # 训练配置
└── datasets/                               # 需要软链接或下载
```

## 环境安装

```bash
conda create -n flare_removal python=3.10
conda activate flare_removal

# PyTorch
pip install torch torchvision

# Mamba 依赖
pip install causal_conv1d
pip install mamba-ssm

# BasicSR 框架
pip install basicsr edict pyiqa
pip install 'numpy<2'

# 可选：OpenCV 等
pip install opencv-python
```

## 数据集准备

```bash
# 软链接 Flare7K++ 数据集
ln -s /path/to/Flare7K++ flare-removal/datasets/Flare7K++
ln -s /path/to/Flickr24K flare-removal/datasets/Flickr24K
```

## 训练

```bash
cd flare-removal/Wave-Mamba
conda activate flare_removal

python -m basicsr.train -opt options/train_wavemamba_flare.yml
```

## 训练结果

| 指标 | 值 |
|------|------|
| 最佳 PSNR | 27.80 (@590k iter) |
| 最佳 SSIM | 0.949 (@550k iter) |
| LPIPS | 0.053 |
| 参数量 | 0.29M (wf=16) |

## 参考

- 原始 Wave-Mamba: https://github.com/AlexZou14/Wave-Mamba
- Flare7K++: https://github.com/yby1305693/Flare7K
