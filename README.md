# PointNeXt ModelNet40 Demo

这是一个 ModelNet40 点云分类项目。当前主方案是：

```text
PointNeXt-S/C64 + x,y,z,nx,ny,nz + 1024 points + AdamW + cosine scheduler + test-time voting
```

代码采用“可复用 Python 模块 + 命令行脚本 + Jupyter Notebook”的结构。训练入口会自动使用 CUDA GPU；正式训练命令建议加 `--require-cuda`，这样没有 GPU 时会直接停止，避免误用 CPU 跑长任务。

## 1. 环境准备

进入项目目录：

```powershell
cd D:\pointnext_modelnet40_demo
```

建议创建虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

先确认机器有 NVIDIA GPU 和驱动：

```powershell
nvidia-smi
```

安装 CUDA 版 PyTorch。请优先到 PyTorch 官方安装页选择你的系统、Python、CUDA 版本并复制命令：

```text
https://pytorch.org/get-started/locally/
```

示例，CUDA 12.1 环境可用：

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

然后安装项目依赖：

```powershell
python -m pip install -r requirements.txt
```

验证 PyTorch 能看到 GPU：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

必须看到：

```text
True
你的 NVIDIA GPU 名称
```

## 2. 数据目录

当前默认训练数据路径是：

```text
modelnet40_train_data/modelnet40_normal_resampled/
  airplane/
    airplane_0001.txt
  bathtub/
  ...
  xbox/
```

每个点云文件每行 6 个数：

```text
x,y,z,nx,ny,nz
```

代码也支持下面这种标准 split 结构：

```text
data/
  train/
    airplane/
      airplane_0001.txt
  test/
    sample_0001.txt
```

## 3. 代码检查

```powershell
python -m compileall src
```

训练时建议始终保留 `--require-cuda`。如果没有 GPU，命令会报错停止。

## 4. 正式 GPU 训练

推荐先训练 PointNeXt-S/C64：

```powershell
python -m src.pointnext_demo.train `
  --data-root modelnet40_train_data/modelnet40_normal_resampled `
  --variant s `
  --width 64 `
  --nsample 32 `
  --num-points 1024 `
  --use-normals `
  --epochs 600 `
  --batch-size 32 `
  --lr 0.001 `
  --weight-decay 0.05 `
  --label-smoothing 0.2 `
  --out-dir runs/pointnext_s_c64_normals `
  --require-cuda
```

训练开始时应看到类似输出：

```text
dataset=9843 train=8367 val=1476 channels=6 classes=40 device=cuda
gpu=NVIDIA ...
```

训练结束后输出：

```text
runs/pointnext_s_c64_normals/
  best.pt
  history.json
```

`best.pt` 是验证集准确率最高的模型参数，后续预测就加载这个文件。`history.json` 保存每轮 `train_loss`、`train_acc`、`val_loss`、`val_acc` 和学习率。

如果显存充足，可尝试更强的 PointNeXt-B/C64：

```powershell
python -m src.pointnext_demo.train `
  --data-root modelnet40_train_data/modelnet40_normal_resampled `
  --variant b `
  --width 64 `
  --nsample 32 `
  --num-points 1024 `
  --use-normals `
  --epochs 600 `
  --batch-size 16 `
  --out-dir runs/pointnext_b_c64_normals `
  --require-cuda
```

## 5. 使用训练好的模型预测

如果测试集在默认 `data_root/test` 下：

```powershell
python -m src.pointnext_demo.predict `
  --checkpoint runs/pointnext_s_c64_normals/best.pt `
  --out-csv submit.csv `
  --votes 10
```

如果测试集在其他目录：

```powershell
python -m src.pointnext_demo.predict `
  --data-root your_test_data_root `
  --split test `
  --checkpoint runs/pointnext_s_c64_normals/best.pt `
  --out-csv submit.csv `
  --votes 10
```

输出格式：

```text
sample_0001,chair
sample_0002,airplane
```

`--votes 10` 表示测试时对同一个样本做 10 次预测并平均类别概率，通常比单次预测更稳定，但预测耗时约为 `--votes 1` 的 10 倍。

## 6. Jupyter Notebook 训练

也可以使用 Notebook：

```powershell
jupyter notebook notebooks/modelnet40_pointnext_training.ipynb
```

Notebook 默认 `require_cuda=True`，没有 CUDA GPU 时会停止。它复用同一套 `src/pointnext_demo` 模块，不是另一套独立算法。

## 7. 当前模型架构

当前选择的是自包含 PointNeXt 风格分类网络，不依赖完整 OpenPoints 工程：

- 输入：默认 `x,y,z,nx,ny,nz`，即 6 通道；可用 `--no-normals` 改为仅 `x,y,z`。
- 点数：默认每个样本采样 `1024` 个点。
- 预处理：中心化、单位球归一化、固定点数采样。
- 数据增强：随机缩放、平移、jitter、point dropout；随机 Y 轴旋转默认关闭，可用 `--random-rotate` 开启。
- 局部特征：FPS 采样 + kNN 分组，默认 `nsample=32`。
- 主干：PointNeXt/PointNet++ 风格 Set Abstraction + residual MLP。
- 默认宽度：`width=64`，即 C64。
- 分类头：全局 max pooling + avg pooling 后接 `Linear 512 -> 256 -> 40`。
- 损失：`CrossEntropyLoss(label_smoothing=0.2)`。
- 优化器：`AdamW(lr=0.001, weight_decay=0.05)`。
- 调度器：`CosineAnnealingLR`。

训练中真正被学习和保存的模型权重包括：

- stem 的 `Conv1d` 和 `BatchNorm1d` 参数；
- 每个 Set Abstraction 层中的局部 `Conv2d`、`BatchNorm2d` 参数；
- residual MLP 中的 `Conv1d`、`BatchNorm1d` 参数；
- 分类头 `Linear`、`BatchNorm1d` 参数。

这些权重保存在 `best.pt` 的 `model` 字段中。

## 8. 精度目标与模型选择

你的目标是：

```text
Test Instance Accuracy >= 92%
Class Accuracy >= 90%
```

当前 PointNeXt-S/C64 方案是合适的。官方 OpenPoints / PointNeXt ModelNet40 结果中，PointNeXt-S(C=64) 报告约 `94.0` overall accuracy 和 `91.1` mean class accuracy，理论上高于你的目标线。

选择当前模型设计的原因：

- PointNet 结构简单、训练快，但主要依赖全局特征，对局部几何结构建模较弱，冲击 `Class Accuracy >= 90%` 的风险更高。
- PointNet++ 能通过层次化采样和局部分组学习局部几何，ModelNet40 上通常能达到较强基线；PointNeXt 正是在 PointNet++ 思路上通过更现代的训练策略、残差 MLP、优化器和增强设置进一步提升效果。
- 当前数据天然包含 `x,y,z,nx,ny,nz`，PointNeXt-S/C64 可以同时利用坐标和法向量信息，比仅用坐标的轻量模型更适合当前任务。
- PointNeXt-S/C64 在精度、显存和训练时间之间比较均衡；相比更大的 PointNeXt-B，它更容易在普通单卡 GPU 上完整训练 600 epoch。
- 项目没有直接引入完整 OpenPoints，是为了降低 Windows 环境下安装 CUDA 扩展和复杂依赖的风险；当前实现保留 PointNeXt 的关键思想，同时保持代码可读、可改、可在本目录直接运行。

但需要注意：本项目是轻量自包含实现，不是完整复刻官方 OpenPoints。它更适合课程项目、可读性和本地直接运行；如果你必须最大化最终测试成绩，优先级建议如下：

1. 首选：当前 `PointNeXt-S/C64 + normals + votes=10`，先完整训练 600 epoch。
2. 如果验证集 `val_acc` 长期低于 92%，尝试 `PointNeXt-B/C64`、更长训练、调整 batch size。
3. 如果最终成绩必须尽可能接近官方最佳结果，使用官方 OpenPoints/PointNeXt 代码和官方配置训练或加载官方预训练模型会更稳。

参考资料：

- PyTorch 官方安装页：https://pytorch.org/get-started/locally/
- OpenPoints ModelNet40 示例：https://guochengqian.github.io/PointNeXt/examples/modelnet/
- OpenPoints / PointNeXt Model Zoo：https://guochengqian.github.io/PointNeXt/modelzoo/
- PointNeXt 论文：https://arxiv.org/abs/2206.04670
