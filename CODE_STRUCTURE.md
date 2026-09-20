# EGE-UNet 代码框架说明

本文档按训练时的数据流梳理工程结构，方便定位后续修改点。

## 目录结构

```text
EGE-UNet/
├── train.py                    # 训练入口：创建配置、数据集、模型、损失、优化器并驱动训练
├── engine.py                   # 单轮训练、验证、测试逻辑
├── utils.py                    # 日志、随机种子、优化器/调度器、损失函数、数据增强、结果保存
├── configs/
│   └── config_setting.py       # 实验配置：数据路径、模型宽度、训练超参、增强策略
├── datasets/
│   └── dataset.py              # ISIC 数据读取，返回 image/mask 张量
├── models/
│   └── egeunet.py              # EGE-UNet 网络结构
├── data/
│   └── isic2017/               # 默认数据目录，内部应含 train/val 的 images 和 masks
└── results/                    # 训练输出：日志、权重、预测可视化
```

## 训练主流程

1. `train.py` 读取 `setting_config`。
2. 创建结果目录、日志、TensorBoard writer。
3. 用 `NPY_datasets` 加载训练集和验证集。
4. 根据 `config.network` 创建 `EGEUNet`。
5. 从 `utils.py` 获取损失函数、优化器和学习率调度器。
6. 循环调用 `engine.train_one_epoch()` 和 `engine.val_one_epoch()`。
7. 保存 `latest.pth`，并在验证 loss 最小时保存 `best.pth`。
8. 训练结束后加载最佳模型，调用 `engine.test_one_epoch()` 保存预测图。

## 模型主结构

`EGEUNet` 是一个轻量 U-Net 变体：

- Encoder：前 3 层使用普通卷积，后 3 层使用 `Grouped_multi_axis_Hadamard_Product_Attention`。
- Decoder：逐级上采样，并与 encoder 的跳跃特征相加。
- `group_aggregation_bridge`：用高层语义特征、低层细节特征和深监督 mask 引导跳跃连接融合。
- `gt_ds=True`：开启深监督，多个 decoder 层都会输出辅助预测，并参与损失计算。
- 最终输出：`out0` 为原图大小的二分类分割概率图。

## 常改位置

- 换数据集路径：改 `configs/config_setting.py` 中的 `datasets` 和 `data_path`。
- 改输入尺寸：改 `input_size_h`、`input_size_w`。
- 改模型宽度：改 `model_config['c_list']`，数值越大模型越宽、显存占用越高。
- 关闭桥接模块：改 `model_config['bridge'] = False`，但当前 `forward()` 默认仍调用 GAB，若要关闭需要同步改模型前向逻辑。
- 关闭深监督：改 `model_config['gt_ds'] = False`，同时损失函数也要换成 `BceDiceLoss`，否则 `GT_BceDiceLoss` 会期待辅助输出。
- 改优化器/学习率：改 `opt`、`lr`、`weight_decay`。
- 改评价阈值：改 `threshold`。

## 数据格式要求

默认读取：

```text
data/isic2017/
├── train/
│   ├── images/
│   └── masks/
└── val/
    ├── images/
    └── masks/
```

`images` 和 `masks` 文件名排序后一一配对。mask 会被转成灰度图，并除以 255 得到 0 到 1 的标签。
