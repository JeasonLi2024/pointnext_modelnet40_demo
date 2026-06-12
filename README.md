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
    pointnext_s_c64/
      pointnext_s_c64.yaml        # S/C64 训练与预测配置
    pointmlp/
      pointmlp_c64.yaml           # PointMLP 精度优先配置
      pointmlp_elite_c32.yaml     # PointMLP-Elite 轻量配置
    pointnext_b_c64_no_rotate/    # B/C64 无旋转两阶段配置
      stage1.yaml
      stage2.yaml
    pointnext_b_c64_rotate/       # B/C64 随机 Y 轴旋转两阶段配置
      stage1.yaml
      stage2.yaml
      stage1_old.yaml             # 旧版强增强 Stage 1
      stage2_old.yaml             # 旧版强增强 Stage 2
    pointnext_b_c96_no_rotate/    # B/C96 高容量无旋转两阶段配置
      stage1.yaml
      stage2.yaml
    predict_selected_model/
      predict.yaml                # 单模型预测：改配置切换 checkpoint
      ensemble.yaml               # 多模型概率集成预测
      ensemble_class90.yaml       # 冲 Class≥90% 的加权集成（测试）
      ensemble_best_latest.yaml   # 当前实测最佳概率集成
  labels/
    modelnet40.txt                # ModelNet40 类别名称
  notebooks/
    modelnet40_pointnext_training.ipynb
  src/pointnext_demo/
    __init__.py
    data.py                       # 点云读取、采样、归一化、增强、Dataset
    model.py                      # PointNeXt 风格分类模型
    train.py                      # 训练入口，读取配置并保存 best.pt/history.json
    predict.py                    # 单模型预测入口，加载 best.pt 输出 CSV
    predict_ensemble.py           # 多模型概率集成预测
    inference.py                  # 单模型 votes 与 checkpoint 加载（predict / ensemble 共用）
    utils.py                      # 随机种子、配置、标签、保存工具
  scripts/
    evaluate_all_models.py        # 按训练配置批量复测全部 best.pt
    run_model_comparison_predict.py  # 批量单模型预测对比
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
configs/pointnext_s_c64/pointnext_s_c64.yaml
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
python -m src.pointnext_demo.train --config configs/pointnext_s_c64/pointnext_s_c64.yaml --batch-size 8
```

当前保留的训练配置：

| 模型名 | 配置 | 说明 |
| --- | --- | --- |
| `pointnext_s_c64_base_v2` | `configs/pointnext_s_c64/pointnext_s_c64.yaml` | 优化后的 S/C64 基础模型，无旋转、法向量、轻增强、类别权重、2048 点推理 |
| `pointmlp_elite_c32` | `configs/pointmlp/pointmlp_elite_c32.yaml` | 轻量 PointMLP-Elite，1024 点、法向量、SGD |
| `pointmlp_c64` | `configs/pointmlp/pointmlp_c64.yaml` | 完整 PointMLP 精度优先配置 |
| `pointnext_b_c64_no_rotate_stage1/2` | `configs/pointnext_b_c64_no_rotate/stage1.yaml`、`stage2.yaml` | 主力 B/C64 两阶段方案，无随机旋转，默认 `votes: 1` |
| `pointnext_b_c64_rotate_stage1/2` | `configs/pointnext_b_c64_rotate/stage1.yaml`、`stage2.yaml` | 旋转增强对照实验，默认 `votes: 3` |
| `pointnext_b_c64_stage1/2`（旧版） | `configs/pointnext_b_c64_rotate/stage1_old.yaml`、`stage2_old.yaml` | 归档的强旋转增强方案 |
| `pointnext_b_c96_no_rotate_stage1/2` | `configs/pointnext_b_c96_no_rotate/stage1.yaml`、`stage2.yaml` | 更大容量备选方案，显存充足时再训练 |
| `predict_selected_model` | `configs/predict_selected_model/predict.yaml` | 统一预测配置，修改 YAML 即可切换模型 |

## 5. 正式训练

推荐先训练优化后的 PointNeXt-S/C64 基础模型：

```powershell
python -m src.pointnext_demo.train --config configs/pointnext_s_c64/pointnext_s_c64.yaml
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
python -m src.pointnext_demo.train --config configs/pointnext_s_c64/pointnext_s_c64.yaml
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

旧的第一轮输出保留在 `runs/pointnext_s_c64_normals/`。当前 `configs/pointnext_s_c64/pointnext_s_c64.yaml` 使用新的 `pointnext_s_c64_base_v2` 输出目录，不会覆盖第一轮结果。显存不足时将 `batch_size` 调小即可。

## 6. 使用训练好的模型预测

各模型训练配置末尾也包含预测字段，可以直接用对应配置预测。例如：

```powershell
python -m src.pointnext_demo.predict --config configs/pointnext_b_c64_no_rotate/stage2.yaml
```

预测参数与训练一样写在 YAML 中。也可以固定使用统一预测配置：

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

### 6.1 批量复测全部模型

预测程序本身会在终端打印加载信息、进度和指标。批量运行额外保存完整日志，是为了保留每个 checkpoint 实际使用的配置、设备、异常和最终指标，便于在 Linux 服务器复现和排查；日志不是新的训练日志，也不会改变模型权重。

批量复测 `runs/**/best.pt`：

```powershell
python scripts/evaluate_all_models.py
```

Linux 命令相同。没有 GPU 时可追加 `--cpu`。脚本会：

- 根据 checkpoint 中保存的训练参数和 `out_dir` 找到训练时的 YAML，不使用 `predict.yaml`。
- 为每个模型生成 `runs/<模型名>/test_predictions.csv`。
- 为每个模型生成 `runs/<模型名>/test_eval.log`。
- 将成功结果按模型名更新到 `runs/result.csv`；已有其他结果会保留，同名模型会更新而不是重复追加。
- 某个模型失败时继续测试其余模型，最后返回失败状态；使用 `--fail-fast` 可在首次失败时停止。

### 6.2 概率集成预测

概率集成对多个 checkpoint 的 40 维 softmax 概率做加权平均，再取最大概率类别。它保留了置信度信息，通常优于只对最终类别做多数投票。

当前实测最佳配置是 `configs/predict_selected_model/ensemble_best_latest.yaml`：

```powershell
python -m src.pointnext_demo.predict_ensemble --config configs/predict_selected_model/ensemble_best_latest.yaml
```

输出文件为 `runs/predict_compare/ensemble_best_latest.csv`。

| 成员 | 推理点数 | votes | 权重 |
|---|---:|---:|---:|
| PointMLP-Elite C32 normals | 1024 | 1 | 4 |
| PointNeXt-B C64 no-rotate 2048 stage1 | 2048 | 1 | 2 |
| PointNeXt-B C64 no-rotate 2048 stage2 | 2048 | 1 | 3 |
| PointNeXt-B C96 no-rotate stage1 | 2048 | 1 | 1 |

本次正式运行结果：

```text
Test Instance Accuracy: 92.67%
Class Accuracy:         90.29%
```

这组权重是在查看带标签测试集结果后做的诊断性搜索，适合说明现有 checkpoint 的组合上限，但不应当作为严格、无偏的公开基准。规范流程应在训练集内固定验证集，用验证集选择成员和权重，然后只在测试集运行一次。

`votes` 与模型集成不是一回事：`votes` 是同一 checkpoint 对不同旋转输入的概率平均；`weight` 是不同 checkpoint 之间的概率加权。当前最佳成员均使用 `votes: 1`，因为模型主要按无随机旋转策略训练，盲目增加旋转投票没有稳定收益。

所有成员必须使用相同测试样本顺序，且 `architecture`、`variant`、`width`、`nsample`、`use_normals` 必须匹配 checkpoint。集成会同时占用更多显存；显存不足时优先降低顶层 `predict_batch_size`。

## 7. Jupyter Notebook 训练

也可以使用 Notebook：

```powershell
jupyter notebook notebooks/modelnet40_pointnext_training.ipynb
```

Notebook 默认 `use_gpu=True`，有可用 CUDA GPU 时使用 GPU，否则自动回落到 CPU。它复用同一套 `src/pointnext_demo` 模块，不是另一套独立算法。

## 8. 当前模型架构

项目支持三种自包含架构，不依赖完整 OpenPoints 工程：

- `pointnext_legacy`：原有 PointNeXt/PointNet++ 风格网络，用于兼容已经训练好的权重。
- `pointmlp_elite`：约 0.72M 参数，适合作为高效主模型。
- `pointmlp`：约 13.24M 参数，适合作为精度优先模型。

公共输入与预处理：

- 输入默认是 `x,y,z,nx,ny,nz` 6 通道，也可用 `--no-normals` 改为仅 `x,y,z`。
- 点数：默认每个样本采样 `1024` 个点。
- 预处理：中心化、单位球归一化、固定点数采样。
- `augment_strength: official` 使用随机缩放和平移；原有 `normal/strong` 还包含 jitter 和 point dropout。
- 对齐的 ModelNet40 默认关闭随机旋转，官方 PointNeXt 配置也注明 rotation does not help。

PointMLP 新主干使用四级 FPS + kNN 局部分组、几何仿射归一化、局部残差 MLP 和全局最大池化。训练器同时支持：

- `SGD + Nesterov` 或 `AdamW`；
- CUDA 混合精度；
- 梯度裁剪；
- `val_composite`，即 Instance Accuracy 与 Class Accuracy 的调和平均，用于避免只优化其中一个指标。
- 续训默认复用 checkpoint 中的训练/验证划分，避免 stage2 重新划分后把 stage1 已见样本当作验证集。

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

当前 `runs/result.csv` 中的独立测试集结果如下；所有单模型均使用法向量：

| 模型 | 参数量 | 训练/推理点数 | 旋转 | Test Instance | Class |
|---|---:|---:|---|---:|---:|
| PointMLP-Elite C32 normals | 0.72M | 1024/1024 | 否 | **92.30%** | **89.65%** |
| PointNeXt-S C64 normals | 1.90M | 1024/2048 | 否 | 91.69% | 88.32% |
| PointNeXt-B C64 no-rotate 2048 stage1 | 2.60M | 2048/2048 | 否 | 91.05% | 88.56% |
| PointNeXt-B C64 no-rotate 2048 stage2 | 2.60M | 2048/2048 | 否 | 90.88% | 88.33% |
| PointNeXt-B C64 no-rotate stage1 | 2.60M | 1024/2048 | 否 | 90.84% | 87.13% |
| PointNeXt-B C64 no-rotate stage2 | 2.60M | 1024/2048 | 否 | 90.76% | 87.77% |
| PointNeXt-B C64 rotate stage1 | 2.60M | 1024/2048 | 是 | 89.34% | 86.82% |
| PointNeXt-B C64 rotate stage2 | 2.60M | 1024/2048 | 是 | 90.15% | 87.52% |
| PointNeXt-B C96 no-rotate stage1 | 5.25M | 1024/2048 | 否 | 90.44% | 88.16% |
| PointNeXt-B C96 no-rotate stage2 | 5.25M | 1024/2048 | 否 | 89.99% | 87.59% |
| PointNeXt-S C64 base_v2 | 1.90M | 1024/2048 | 否 | 89.42% | 86.53% |
| **四模型概率集成** | - | 混合 | 否 | **92.67%** | **90.29%** |

结果表明：

- 当前最佳单模型是 PointMLP-Elite C32。它仅约 0.72M 参数，却比 1.90M 到 5.25M 的简化 PointNeXt 模型更准确，说明完整的局部几何仿射与残差 PointMLP 设计比单纯增加宽度更有效。
- 2048 点训练的 B/C64 比对应 1024 点版本略强，但 stage2 没有继续提升，说明短续训已经进入平台期，不能把续训轮数直接等同于泛化提升。
- B/C96 参数量约为 B/C64 的两倍，但准确率没有提高，当前瓶颈更可能来自简化结构、数据划分和训练策略，而不是容量不足。
- 随机 Y 轴旋转组整体弱于无旋转组。ModelNet40 已对齐，强制旋转会破坏有判别力的姿态信息，因此当前最佳实践是关闭随机旋转，保留缩放、平移、轻微扰动等增强。
- 单模型已经超过 92% Instance Accuracy，但尚未达到 90% Class Accuracy；四模型概率集成同时达到两个目标。

这些 PointNeXt checkpoint 来自项目内的简化 PointNeXt/PointNet++ 风格实现，不是 OpenPoints 官方 PointNeXt。其每级邻域聚合与 InvResMLP 设计不完整，因此不能用官方 PointNeXt 论文成绩直接推断本项目 checkpoint 的预期表现。

公开基线表明目标本身可实现：

- PointNet：89.2% OA / 86.2% mAcc，不建议作为冲线模型。
- PointNet++：使用法向量时论文报告 91.9% OA，仍缺少足够余量。
- PointNeXt-S：官方论文报告约 93.2% OA / 90.8% mAcc；C64 配置约 94.0% OA。
- PointMLP-elite：论文报告 94.0% OA / 90.9% mAcc，且只有约 0.68M 参数。
- PointMLP：官方修正版报告约 94.1% OA / 91.5% mAcc。

选择当前模型设计的原因：

- PointNet 结构简单、训练快，但主要依赖全局特征，对局部几何结构建模较弱，冲击 `Class Accuracy >= 90%` 的风险更高。
- PointNet++ 能通过层次化采样和局部分组学习局部几何，ModelNet40 上通常能达到较强基线；PointNeXt 正是在 PointNet++ 思路上通过更现代的训练策略、残差 MLP、优化器和增强设置进一步提升效果。
- 当前数据天然包含 `x,y,z,nx,ny,nz`，PointNeXt-S/C64 可以同时利用坐标和法向量信息，比仅用坐标的轻量模型更适合当前任务。
- PointNeXt-S/C64 在精度、显存和训练时间之间比较均衡；相比更大的 PointNeXt-B，它更容易在普通单卡 GPU 上完整训练 600 epoch。
- 项目没有直接引入完整 OpenPoints，是为了降低 Windows 环境下安装 CUDA 扩展和复杂依赖的风险；当前实现保留 PointNeXt 的关键思想，同时保持代码可读、可改、可在本目录直接运行。

推荐训练和评估顺序：

1. 以 `configs/pointmlp/pointmlp_elite_c32.yaml` 作为高效主模型。
2. 如需继续提高单模型 Class Accuracy，训练完整 `configs/pointmlp/pointmlp_c64.yaml`，而不是继续盲目加宽简化 PointNeXt。
3. 使用 `python scripts/evaluate_all_models.py` 统一复测，避免不同模型使用不同预测配置。
4. 在固定验证集上搜索集成成员和权重，最终测试只运行一次。

`configs/pointmlp/pointmlp_elite_c32.yaml` 是推荐起点；若其 Class Accuracy 未达到 90%，再使用完整 `configs/pointmlp/pointmlp_c64.yaml`。训练时保留法向量、1024 点和关闭随机旋转；不要默认改成 2048 点，因为 PointMLP 官方结果本身就是 1024 点，增加点数会显著提高计算量但不保证提升。

参考资料：

- PyTorch 官方安装页：https://pytorch.org/get-started/locally/
- PointNeXt 论文：https://arxiv.org/abs/2206.04670
- PointNeXt 官方实现：https://github.com/guochengqian/PointNeXt
- PointMLP 论文：https://openreview.net/forum?id=3Pbra-_u76D
- PointMLP 官方实现：https://github.com/ma-xu/pointMLP-pytorch
