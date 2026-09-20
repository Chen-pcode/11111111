# Kaggle 运行与实验指南

## 1. 更新代码与环境

Notebook Settings 中开启 Internet，选择 GPU。已有仓库时运行：

```python
%cd /kaggle/working/11111111
!git pull --ff-only
%pip install -r requirements.txt
!git rev-parse --short HEAD
!python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
!python train.py --help
```

首次运行时，先执行以下两行，再执行上面的安装和检查命令。`git clone` 后面是纯 URL，不包含 Markdown 链接括号。

```python
%cd /kaggle/working
!git clone https://github.com/Chen-pcode/11111111.git
```

`requirements.txt` 补充了 `pytorch-wavelets` 和 `PyWavelets`。先使用 Kaggle 现有的 torch/torchvision 配套版本，不主动升级它们；每次实验会在 `config.json` 记录实际依赖版本。原论文的旧环境版本与本项目的现代环境不是同一个复现条件。

## 2. 检查你的数据路径

当前提供的路径是 `/kaggle/input/datasets/zichengdoctor/isic2017`。该目录应直接包含 `train/images`、`train/masks`、`val/images`、`val/masks`。

```python
from pathlib import Path
from research_support import paired_files

DATA = Path('/kaggle/input/datasets/zichengdoctor/isic2017')
for split in ('train', 'val'):
    pairs = paired_files(DATA, split)
    print(split, len(pairs), pairs[0])
```

如果找不到目录，运行以下代码寻找实际挂载位置，并把后续 `--data` 替换成打印出来的目录：

```python
from pathlib import Path
for images in Path('/kaggle/input').rglob('train/images'):
    root = images.parent.parent
    if all((root / suffix).is_dir() for suffix in ('train/masks', 'val/images', 'val/masks')):
        print(root)
```

支持图像与掩码同 stem，或 `ISIC_001.jpg` 对应 `ISIC_001_segmentation.png`。缺少配对、重复 ID、空目录会报出具体问题。仓库不上传数据集，Kaggle 必须单独挂载数据。

## 3. 先分别运行一轮

```python
!python train.py --dataset isic17 --data /kaggle/input/datasets/zichengdoctor/isic2017 --experiment A0 --epochs 1 --device cuda --out /kaggle/working/check_A0
!python train.py --dataset isic17 --data /kaggle/input/datasets/zichengdoctor/isic2017 --experiment A6 --epochs 1 --device cuda --out /kaggle/working/check_A6
```

这里每轮会遍历整个训练集和验证集，仅用于检查运行流程。A0 应打印 `active_stages=[]`；A6 应打印 `frequency=both, gate=mask, active_stages=[1, 2, 3]`。每次重新检查使用新的 `--out` 路径，以免混用结果。

默认 `python train.py` 对应 A0，频率分支关闭。频率门控主实验需要明确使用 `--experiment A6`。

## 4. 正式训练和恢复

一轮检查成功后，使用新的输出目录正式训练，保留相同 seed、输入尺寸、batch size、优化器和数据划分：

```python
!python train.py --dataset isic17 --data /kaggle/input/datasets/zichengdoctor/isic2017 --experiment A0 --seed 42 --epochs 300 --out /kaggle/working/A0_s42
!python train.py --dataset isic17 --data /kaggle/input/datasets/zichengdoctor/isic2017 --experiment A6 --seed 42 --epochs 300 --out /kaggle/working/A6_s42
```

单次运行使用一张 GPU。可先分别完成 A0 和 A6，确认收益后再展开全套消融。每个实验保存 `config.json`、`manifest.json`、`log/train.info.log`、TensorBoard 日志、`checkpoints/latest.pth`、`checkpoints/best.pth` 和预测图。

```python
!python train.py --resume --out /kaggle/working/A6_s42
!python train.py --evaluate --out /kaggle/working/A6_s42
```

恢复和评估会读取原配置，并核对源码、数据文件哈希和依赖版本。`--resume` 恢复优化器、调度器和 epoch，但目前不保存中断时的 Python/NumPy/Torch 随机数状态，因此不能宣称与未中断训练逐步一致。不能把一轮检查目录直接延长成正式实验。

Kaggle 会话结束前下载或保存完整输出目录；新会话中的 `/kaggle/working` 不保证保留。跨会话恢复时，代码版本、依赖版本和挂载路径需要匹配。目录内容需放到可写位置。

## 5. 消融设计与论文边界

| 实验 | 频率模式 | 高频门控 | 用途 |
| --- | --- | --- | --- |
| A0 | off | 不启用 | 原始结构基线 |
| A1 | spatial | mask | 双空间分支对照，与 A6 学习参数量一致，但计算量不同 |
| A2 | low | none | 仅低频修正 |
| A3 | high | none | 仅高频修正 |
| A4 | both | none | 高低频组合，无门控 |
| A5 | both | feature | 由频带特征生成门控 |
| A6 | both | mask | 在频带特征基础上加入预测不确定性 |

A1–A6 默认插入 GAB1、GAB2、GAB3。模型先增强低层 skip 特征，再进入原 GAB，保留 GHPA、GAB 和深监督。门控输入包含频带绝对值均值和 `4*p*(1-p)` 不确定性，`p=sigmoid(detached mask logits)`；门控只调节高频修正，不是显式边界监督或概率校准。高频既可能包含边界，也可能包含毛发和噪声，效果需要实验证据。

先比较 A0/A6，再用 A4/A5/A6 分离频率与不确定性门控的贡献。阶段消融可用 `--mode both --gate mask --stages 1 2 3`，不能与 `--experiment` 同时使用。

原论文报告 ISIC2017/2018、7:3 划分、256x256、300 epochs、batch size 8，并重复 5 次报告均值和标准差。后续建议预先固定 5 个 seed，例如 42、43、44、45、46，基线和改进使用相同 seed 列表，并保存每次结果。单次提升不能证明创新有效。

当前实现按验证 loss 选 best，最终指标仍来自同一个 val，必须标明验证集结果。正式研究应预先确定独立测试集或嵌套评估方案，避免用报告集反复调参；与原论文数字比较时需要说明划分、选择规则、软件环境和统计口径的差异。当前日志的 `miou` 实际计算前景 IoU，不是前景/背景两类 IoU 的均值。

后续还需记录参数量、推理延迟、显存和运算量。THOP 不一定覆盖 DWT/IDWT，自定义算子缺失时不能把输出直接当作完整 FLOPs。是否足以形成论文贡献，还需要相关工作检索和实验证据。

## 6. 后续报错反馈

提供完整 traceback、执行命令、`git rev-parse --short HEAD` 输出，以及错误前的数据数量和 `Experiment=...` 行。本地修复按编号追加到 `MODIFICATION_LOG.md`。不要包含访问令牌或私钥。

频率图导出尚未接入训练入口，传入非空 `--visualize-ids` 会明确报错。
