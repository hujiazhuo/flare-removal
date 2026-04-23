# Wave-Mamba for Flare Removal

基于 Wave-Mamba (ACMMM2024) 改造的炫光去除模型

## 核心改造

1. **输出机制转换**：从加性残差改为减性残差（`clean = input - flare_pred`）
2. **损失函数**：光晕预测损失 + 感知损失 + FFT 频域损失
3. **原始模型规模**：wf=32, n_l_blocks=[1,2,4], n_h_blocks=[1,1,2] (1.51M 参数)

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

下载 Flare7K++ 数据集（https://github.com/yby1305693/Flare7K），创建软链接：

```bash
# 假设数据集放在 /data/flare 目录
mkdir -p datasets
ln -s /data/flare/Flare7K++ datasets/Flare7Kpp
ln -s /data/flare/Flickr24K datasets/Flickr24K
```

然后修改 `options/train_wavemamba_flare.yml` 中的路径。

## 训练

```bash
cd Wave-Mamba
conda activate flare_removal

python -m basicsr.train -opt options/train_wavemamba_flare.yml
```

## 模型配置

| 参数 | 值 | 说明 |
|------|------|------|
| wf | 32 | 特征通道宽度（原版） |
| n_l_blocks | [1, 2, 4] | 低频块数量（原版） |
| n_h_blocks | [1, 1, 2] | 高频块数量（原版） |
| 参数量 | 1.51M | - |
| batch_size | 2 | 可根据显存调整 |

## 训练结果 (轻量化版本 wf=16)

| 指标 | 值 |
|------|------|
| 最佳 PSNR | 27.80 (@590k iter) |
| 最佳 SSIM | 0.949 (@550k iter) |
| LPIPS | 0.053 |
| 参数量 | 0.29M (wf=16) |

## 参考

- 原始 Wave-Mamba: https://github.com/AlexZou14/Wave-Mamba
- Flare7K++: https://github.com/yby1305693/Flare7K
