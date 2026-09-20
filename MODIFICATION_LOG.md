# 修改与实验记录

本文件逐次记录用户报告的问题、原因、修改、验证结果和后续操作。未执行的实验不填写指标；代码修复通过不代表性能提升。

## FIX-001 | 2026-09-20 | Kaggle 入口缺失模块与频率配置未生效

### 报错和背景

- 起始代码：`728a53a`，仓库 `Chen-pcode/11111111`。
- Kaggle 执行 `python train.py`，报 `ModuleNotFoundError: No module named 'research_support'`。
- 用户数据路径：`/kaggle/input/datasets/zichengdoctor/isic2017`，尚待 Kaggle 验证实际目录结构。
- 阅读用户提供的 EGE-UNet 论文文本，作为实验设置参考。

### 根因

1. `author_protocol.py` 导入的 `research_support.py`、`run_frequency.py` 未包含在仓库中。
2. `models/egeunet.py` 调用频率分支的行有错误缩进，会在导入模型时失败。
3. `train.py` 未传入 `freq_mode`、`freq_gate`、`freq_stages`，即使选择 A6，模型仍使用默认 off。
4. 安装命令缺少 `pytorch-wavelets`/`PyWavelets`，启用小波模块时会缺依赖。

### 修改步骤

1. 新增 `research_support.py`，实现图像/掩码 ID 配对和依赖版本记录；将 `package_versions` 一并放在此处，移除对不存在的 `run_frequency.py` 的引用。
2. 修复频率调用缩进，将三项频率配置传入模型；打印实际启用阶段与参数量。默认 A0，A6 显式开启频率门控。
3. 新增 `requirements.txt`，包含小波和现有训练依赖。
4. 数据读取按 ID 配对，识别 `_segmentation` 后缀，对路径、空目录、重复和缺失配对给出明确错误；有效 ISIC 文件的配对不变。
5. 接通现有 `experiment_spec`，保存配置、依赖版本、源码哈希和数据文件清单；恢复和评估核对这些信息。
6. 禁止新训练自动复用非空输出目录，恢复须显式 `--resume`；接通 `--evaluate`，保留 `best.pth` 固定文件名。保存 Python float 类型的 loss，使新 checkpoint 可用 `weights_only=True` 加载。
7. 模型和 batch 遵循 `--device`，支持 CPU 流程测试；TensorBoard step 每 batch 加 1 并保存；混淆矩阵固定二分类标签以兼容单类样本。
8. 修正模块文档：频率模块增强原 GAB 输入，门控只作用于高频修正。将最终评估标为验证集评估。尚未实现的频率图导出明确报错。
9. 新增 `KAGGLE_GUIDE.md`，记录给定数据路径、更新/安装/检查/训练命令、消融表和论文评估注意事项。

### 验证

- 本机独立 `.venv`：Python 3.11.9、torch 2.14.0+cpu、torchvision 0.29.0+cpu、pytorch-wavelets 1.3.0。
- `python -m unittest discover -s tests -v`：4 项测试通过，耗时 55.363 秒。覆盖 A0–A6 的输出尺寸、有限 loss/梯度、频率参数参与反向传播、共同主干初始化一致，以及数据配对错误。
- A0/A6 各完成合成数据的一轮 CPU 训练（32x32，train/val 各 2 张），验证配置与清单落盘、A6 checkpoint 包含频率权重、best/latest 保存、评估加载、已完成实验恢复加载、非空目录防覆盖。
- `python train.py --help` 正常；`pip check` 无依赖冲突；`git diff --check` 无新增空白错误。
- 本地未完成真实数据集训练；上述合成数据仅验证流程，不用于论文精度结论。
- Kaggle GPU 和完整数据训练待用户运行后反馈。

### 后续实验记录

| 日期 | 代码提交 | 实验/seed | 数据划分 | 输出目录 | 状态 | Dice / 前景 IoU |
| --- | --- | --- | --- | --- | --- | --- |
| 待运行 | FIX-001 修复版本 | A0 / 42 | 用户 Kaggle train/val | `/kaggle/working/check_A0` | 一轮流程检查待执行 | 未测 |
| 待运行 | FIX-001 修复版本 | A6 / 42 | 同上 | `/kaggle/working/check_A6` | 一轮流程检查待执行 | 未测 |

### 已知边界

- 本轮不改变损失、数据增强、归一化、阈值、默认学习率和原模型主体。
- 原有随机旋转策略为初始化时采样角度、逐样本按概率应用；保留以减少对比条件变化。
- `--resume` 尚未恢复随机数生成器状态，不能承诺精确延续随机轨迹。
- val 同时用于选模与报告，不是独立测试集；正式实验协议仍需单独确定。
- 本轮不声称提升精度，也不证明方法新颖性。
