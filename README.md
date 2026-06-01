# PointNeXt ModelNet40 Demo

这是一个 ModelNet40 点云分类项目。当前主方案是：

```text
PointNeXt-S/B + x,y,z,nx,ny,nz + 1024-point training + 2048-point inference + AdamW + cosine scheduler
```

代码采用“可复用 Python 模块 + 命令行脚本 + Jupyter Notebook”的结构。训练和预测入口通过配置项 `use_gpu` 控制是否启用 GPU 加速：设置为 `true` 时会检测 CUDA，机器有可用 GPU 就使用 GPU，否则自动回落到 CPU；设置为 `false` 时始终使用 CPU。

## 项目结构

```text
pointnext_modelnet40_demo/
  .gitignore
  README.md
  requirements.txt
  configs/
    pointnext_s_c64.yaml          # 训练/预测默认配置，可手动调整超参数
    pointnext_b_c64_no_rotate/    # B/C64 无旋转两阶段配置
      stage1.yaml
      stage2.yaml
    pointnext_b_c64_rotate/       # B/C64 随机 Y 轴旋转两阶段配置
      stage1.yaml
      stage2.yaml
    pointnext_b_c96_no_rotate/    # B/C96 高容量无旋转两阶段配置
      stage1.yaml
      stage2.yaml
    predict_selected_model/
      predict.yaml                # 统一预测入口，通过改配置切换 checkpoint
  labels/
    modelnet40.txt                # ModelNet40 类别名称
  notebooks/
    modelnet40_pointnext_training.ipynb
  src/pointnext_demo/
    __init__.py
    data.py                       # 点云读取、采样、归一化、增强、Dataset
    model.py                      # PointNeXt 风格分类模型
    train.py                      # 训练入口，读取配置并保存 best.pt/history.json
    predict.py                    # 预测入口，加载 best.pt 输出 CSV
    utils.py                      # 随机种子、配置、标签、保存工具
```

训练数据目录 `modelnet40_train_data/` 被 `.gitignore` 排除，不会提交到 GitHub。

## 1. 环境准备

进入项目目录：

```powershell
cd pointnext_modelnet40_demo
```

建议创建虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1 # Windows 环境
# source .venv/bin/activate # Linux/Mac 环境
python -m pip install --upgrade pip
```

如果希望启用 GPU 加速，先确认机器有 NVIDIA GPU 和驱动：

```powershell
nvidia-smi
```

如果要使用 GPU，安装 CUDA 版 PyTorch。请优先到 PyTorch 官方安装页选择你的系统、Python、CUDA 版本并复制命令：

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

可选：验证 PyTorch 能看到 GPU：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

如果准备使用 GPU，应看到：

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

训练前可通过配置文件的 `use_gpu` 控制设备选择。`use_gpu: true` 会优先使用 CUDA，没有可用 GPU 时自动使用 CPU。

## 4. 配置文件

基础模型训练参数集中到：

```text
configs/pointnext_s_c64.yaml
```

多模型实验配置放在 `configs/<model_name>/` 子目录中，避免不同实验互相覆盖。配置文件中每个参数都有注释说明，包括：

- 数据路径：`data_root`、`labels`、`out_dir`
- 模型结构：`variant`、`width`、`nsample`
- 输入与增强：`num_points`、`use_normals`、`random_rotate`
- 训练设置：`epochs`、`batch_size`、`lr`、`weight_decay`、`label_smoothing`
- 验证与运行：`val_ratio`、`num_workers`、`seed`、`use_gpu`
- 重训增强：`use_class_weights`、`class_weight_power`、`augment_strength`、`warmup_epochs`、`early_stop_patience`、`early_stop_metric`、`resume_checkpoint`
- 预测与提交：`test_data_root`、`checkpoint`、`out_csv`、`votes`、`predict_num_points`、`predict_batch_size`、`eval_on_test`

当前配置已按 RTX 4090 24GB 显存设置：

```yaml
variant: s
width: 64
num_points: 1024
batch_size: 128
use_gpu: true
```

命令行参数仍可临时覆盖配置，例如：

```powershell
python -m src.pointnext_demo.train --config configs/pointnext_s_c64.yaml --batch-size 8
```

当前保留的训练配置：

| 模型名 | 配置 | 说明 |
| --- | --- | --- |
| `pointnext_s_c64_base_v2` | `configs/pointnext_s_c64.yaml` | 优化后的 S/C64 基础模型，无旋转、法向量、轻增强、类别权重、2048 点推理 |
| `pointnext_b_c64_no_rotate_stage1/2` | `configs/pointnext_b_c64_no_rotate/stage1.yaml`、`stage2.yaml` | 主力 B/C64 两阶段方案，无随机旋转，默认 `votes: 1` |
| `pointnext_b_c64_rotate_stage1/2` | `configs/pointnext_b_c64_rotate/stage1.yaml`、`stage2.yaml` | 旋转增强对照实验，默认 `votes: 3` |
| `pointnext_b_c96_no_rotate_stage1/2` | `configs/pointnext_b_c96_no_rotate/stage1.yaml`、`stage2.yaml` | 更大容量备选方案，显存充足时再训练 |
| `predict_selected_model` | `configs/predict_selected_model/predict.yaml` | 统一预测配置，修改 YAML 即可切换模型 |

## 5. 正式训练

推荐先训练优化后的 PointNeXt-S/C64 基础模型：

```powershell
python -m src.pointnext_demo.train --config configs/pointnext_s_c64.yaml
```

如果机器有可用 CUDA GPU 且 `use_gpu: true`，训练开始时应看到类似输出：

```text
dataset=9843 train=8367 val=1476 channels=6 classes=40 device=cuda
gpu=NVIDIA ...
```

如果没有可用 CUDA GPU，或配置为 `use_gpu: false`，输出中的设备会是：

```text
device=cpu
```

训练结束后输出：

```text
runs/pointnext_s_c64_base_v2/
  best.pt
  history.json
```

`best.pt` 是验证集准确率最高的模型参数，后续预测就加载这个文件。`history.json` 保存每轮 `train_loss`、`train_acc`、`val_loss`、`val_acc` 和学习率。

训练日志会显示：

- `train_loss`：训练集平均损失；
- `train_instance_acc`：训练集整体样本准确率；
- `train_class_acc`：训练集按类别平均准确率；
- `val_loss`：验证集平均损失；
- `val_instance_acc`：验证集整体样本准确率，可作为没有独立测试集时的 Test Instance Accuracy 近似参考；
- `val_class_acc`：验证集按类别平均准确率，可作为没有独立测试集时的 Class Accuracy 近似参考。

严格来说，最终 `Test Instance Accuracy` 和 `Class Accuracy` 应由独立测试集计算；当前项目只有训练集，因此训练中报告的是留出验证集指标。

### 5.1 多模型训练顺序

第一轮实验表明：未经过旋转增强训练的模型，使用随机旋转投票会明显降低测试精度；但 2048 点推理有提升。因此当前主线优先训练无旋转模型，旋转增强作为对照实验保留。

建议训练顺序：

1. `pointnext_s_c64_base_v2`：确认优化后的基础模型是否超过第一轮。
2. `pointnext_b_c64_no_rotate_stage1/2`：主力提分模型，重点冲 `Class Accuracy >= 90%`。
3. `pointnext_b_c64_rotate_stage1/2`：旋转增强对照，只有实测超过无旋转模型才作为最终模型。
4. `pointnext_b_c96_no_rotate_stage1/2`：更大容量备选，B/C64 仍不足时再训练。

训练基础 S/C64：

```powershell
python -m src.pointnext_demo.train --config configs/pointnext_s_c64.yaml
```

训练 B/C64 无旋转两阶段：

```bash
source .venv/bin/activate
cd ~/workspace/pointnext_modelnet40_demo
python -m src.pointnext_demo.train_two_stage --stage1-config configs/pointnext_b_c64_no_rotate/stage1.yaml --stage2-config configs/pointnext_b_c64_no_rotate/stage2.yaml
```

训练 B/C64 旋转增强两阶段：

```bash
python -m src.pointnext_demo.train_two_stage --stage1-config configs/pointnext_b_c64_rotate/stage1.yaml --stage2-config configs/pointnext_b_c64_rotate/stage2.yaml
```

训练 B/C96 无旋转两阶段：

```bash
python -m src.pointnext_demo.train_two_stage --stage1-config configs/pointnext_b_c96_no_rotate/stage1.yaml --stage2-config configs/pointnext_b_c96_no_rotate/stage2.yaml
```

也可以分别训练某一阶段，例如：

```powershell
python -m src.pointnext_demo.train --config configs/pointnext_b_c64_no_rotate/stage1.yaml
python -m src.pointnext_demo.train --config configs/pointnext_b_c64_no_rotate/stage2.yaml
```

每个模型会写入独立输出目录，例如：

```text
runs/pointnext_s_c64_base_v2/
runs/pointnext_b_c64_no_rotate_stage1/
runs/pointnext_b_c64_no_rotate_stage2/
runs/pointnext_b_c64_rotate_stage1/
runs/pointnext_b_c64_rotate_stage2/
runs/pointnext_b_c96_no_rotate_stage1/
runs/pointnext_b_c96_no_rotate_stage2/
```

### 5.2 单阶段 / 旧版 S 模型

旧的第一轮输出保留在 `runs/pointnext_s_c64_normals/`。当前 `configs/pointnext_s_c64.yaml` 已改为新的 `pointnext_s_c64_base_v2` 输出目录，不会覆盖第一轮结果。显存不足时将 `batch_size` 调小即可。

## 6. 使用训练好的模型预测

预测参数与训练一样写在 YAML 中。推荐固定使用统一预测配置：

```powershell
python -m src.pointnext_demo.predict --config configs/predict_selected_model/predict.yaml
```

如果在 Windows 里使用本项目虚拟环境，先进入虚拟环境，再运行同一条预测命令：

```bat
venv\Scripts\activate.bat
python -m src.pointnext_demo.predict --config configs/predict_selected_model/predict.yaml
```

需要切换模型时，只修改 `configs/predict_selected_model/predict.yaml`，不需要改执行命令。必须同步检查并修改这些字段：

- `selected_model`：当前选择的模型名称，用于人工记录和结果追踪；脚本不会根据它自动填充其他字段。
- `checkpoint`：要加载的权重文件，通常是对应目录下的 `best.pt`。
- `out_csv`：预测结果输出路径。建议每个模型使用独立文件名，避免覆盖之前的结果。
- `variant`、`width`、`nsample`、`use_normals`：模型结构参数，必须和训练该 `best.pt` 时一致，否则可能无法加载权重或得到错误结果。
- `predict_num_points`、`votes`、`predict_batch_size`：推理参数。切换到更大的模型或更多点数时，要同步调小 batch；无旋转模型通常保持 `votes: 1`。

例如切换到 `pointnext_b_c64_no_rotate_stage2` 时，`predict.yaml` 中至少应同步改成：

```yaml
selected_model: pointnext_b_c64_no_rotate_stage2
checkpoint: runs/pointnext_b_c64_no_rotate_stage2/best.pt
out_csv: runs/pointnext_b_c64_no_rotate_stage2/test_predictions_stage2.csv
variant: b
width: 64
nsample: 32
use_normals: true
predict_num_points: 2048
votes: 1
predict_batch_size: 32
```

常用模型的推理参数可参考：

| 模型 | `variant` | `width` | `nsample` | `predict_num_points` | `votes` | `predict_batch_size` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `pointnext_s_c64_base_v2` | `s` | 64 | 32 | 2048 | 1 | 64 |
| `pointnext_b_c64_no_rotate_stage2` | `b` | 64 | 32 | 2048 | 1 | 32 |
| `pointnext_b_c64_rotate_stage2` | `b` | 64 | 32 | 2048 | 1 或 3 | 32 |
| `pointnext_b_c96_no_rotate_stage2` | `b` | 96 | 32 | 2048 | 1 | 24 |

无旋转模型建议 `votes: 1`；旋转增强模型可以比较 `votes: 1` 和 `votes: 3`，只有实测提升时再保留更高投票数。`out_csv` 建议写成模型专属文件名，例如 `test_predictions_base_v2.csv`、`test_predictions_stage2.csv`，便于后续对比分析。

各模型训练配置末尾也包含预测字段，可以直接用对应配置预测。例如：

```powershell
python -m src.pointnext_demo.predict --config configs/pointnext_b_c64_no_rotate/stage2.yaml
```

常用预测字段包括：

- `test_data_root`：测试集根目录（其下应有 `test/<class>/*.txt`）
- `checkpoint`：权重路径，例如 `runs/pointnext_b_c64_no_rotate_stage2/best.pt`
- `out_csv`：提交文件名与路径
- `votes`：测试时投票次数（Y 轴旋转平均概率）
- `predict_num_points`：推理采样点数（可与训练 `num_points` 不同，无需重训即可试 2048）
- `predict_batch_size`：推理 batch，点数增多时适当减小
- `eval_on_test: true`：若测试目录带类别标签，预测结束后打印 Test Instance / Class Accuracy

需要临时实验时仍可用命令行覆盖 YAML，例如只改投票次数：

```powershell
python -m src.pointnext_demo.predict --config configs/predict_selected_model/predict.yaml --votes 3
```

正式保留结果时，优先把参数写回 `configs/predict_selected_model/predict.yaml`，再使用固定命令运行，避免忘记当时命令行覆盖了哪些字段。

输出格式（无表头，英文逗号分隔）：

```text
airplane_0627,airplane
airplane_0628,chair
```

`votes` 越大耗时近似线性增加。对当前无旋转训练模型，第一轮实测 `votes=10` 明显低于 `votes=1`，因此不要盲目增加投票次数。

## 7. Jupyter Notebook 训练

也可以使用 Notebook：

```powershell
jupyter notebook notebooks/modelnet40_pointnext_training.ipynb
```

Notebook 默认 `use_gpu=True`，有可用 CUDA GPU 时使用 GPU，否则自动回落到 CPU。它复用同一套 `src/pointnext_demo` 模块，不是另一套独立算法。

## 8. 当前模型架构

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

训练过程中自动调整的是上述神经网络权重和 BatchNorm 统计量。下面这些是训练前设定的超参数，训练过程不会自动改变，除非手动改命令行参数：

- `width`：模型通道宽度，默认 `64`；显存不足可降到 `32`，想提高容量可尝试 `96`。
- `nsample`：每个局部分组的邻居点数，默认 `32`；通常在 `16-48` 之间调。
- `num_points`：每个样本采样点数，默认 `1024`；常见选择是 `1024` 或 `2048`。
- `batch_size`：默认 `16`；由 GPU 显存决定，显存不足就降到 `8`，显存充足可尝试更大。
- `lr`：学习率，默认 `0.001`；AdamW 下通常在 `0.0005-0.002` 之间调。
- `weight_decay`：权重衰减，默认 `0.05`；参考 PointNeXt / OpenPoints 配置，通常在 `0.01-0.05`。
- `label_smoothing`：默认 `0.2`；常见范围 `0.1-0.2`，用于提升泛化。
- `epochs`：默认 `600`；如果验证集指标还在提升，可以继续增加。

这些范围来自 PointNeXt / OpenPoints 在 ModelNet40 上的常用配置，以及 GPU 显存、训练稳定性和验证集表现之间的折中。实际调参时以 `val_instance_acc` 和 `val_class_acc` 是否提升为准。

## 9. 精度目标与模型选择

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

1. 首选：`pointnext_b_c64_no_rotate_stage1/2 + normals + predict_num_points=2048 + votes=1`。
2. 对照：`pointnext_b_c64_rotate_stage1/2`，只在实测优于无旋转模型时使用。
3. 如果 Class Accuracy 仍低于 90%，尝试 `pointnext_b_c96_no_rotate_stage1/2` 或微调类别权重。
4. 如果最终成绩必须尽可能接近官方最佳结果，使用官方 OpenPoints/PointNeXt 代码和官方配置训练或加载官方预训练模型会更稳。

参考资料：

- PyTorch 官方安装页：https://pytorch.org/get-started/locally/
- OpenPoints ModelNet40 示例：https://guochengqian.github.io/PointNeXt/examples/modelnet/
- OpenPoints / PointNeXt Model Zoo：https://guochengqian.github.io/PointNeXt/modelzoo/
- PointNeXt 论文：https://arxiv.org/abs/2206.04670
