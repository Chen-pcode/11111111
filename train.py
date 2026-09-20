from author_protocol import make_config
import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.egeunet import EGEUNet

from engine import *
import os
import sys

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")


def main(config):
    """
    训练主入口。

    主流程很标准：
    1. 创建日志和输出目录
    2. 初始化 GPU 和随机种子
    3. 构建训练/验证数据集
    4. 创建模型、损失函数、优化器、调度器
    5. 循环执行 train + val
    6. 保存最佳权重和最新断点
    7. 训练完成后再用最佳权重跑一次测试可视化
    """

    # 结果目录由配置文件中的 `work_dir` 决定，默认包含时间戳，
    # 这样每次训练都会写到一个新目录，避免覆盖旧实验。
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)

    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    # 当前代码是按 GPU 环境写的，默认走 CUDA。
    # 如果后续要在 CPU 上调试，需要把 `.cuda()` 相关逻辑一起改掉。
    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    # 数据集类名叫 NPY_datasets，但实际上读取的是图像文件，而不是 `.npy`。
    # train=True 读 `data_path/train/`，train=False 读 `data_path/val/`。
    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers
    )

    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=True
    )

    # 根据配置选择网络结构。当前仓库只实现了 egeunet 这一种。
    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'egeunet':
        model = EGEUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            c_list=model_cfg['c_list'],
            bridge=model_cfg['bridge'],
            gt_ds=model_cfg['gt_ds'],
        )
    else:
        raise Exception('network in not right!')
    model = model.cuda()

    # 损失函数、优化器和学习率调度器都在 config 里集中定义。
    # 当前默认是：
    # - criterion: GT_BceDiceLoss
    # - optimizer: AdamW
    # - scheduler: CosineAnnealingLR
    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    # 如果存在 latest.pth，说明之前训练中断过，或者用户希望继续训练。
    # 这里不仅恢复模型参数，也恢复优化器和调度器状态，保证学习率轨迹连续。
    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = (
            f'resuming model from {resume_model}. '
            f'resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, '
            f'min_epoch: {min_epoch}, loss: {loss:.4f}'
        )
        logger.info(log_info)

    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()

        # 训练一个 epoch，返回更新后的 TensorBoard 全局 step。
        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )

        # 在验证集上评估当前 epoch。
        loss = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config
        )

        # 当前项目用验证 loss 判断“最佳模型”。
        # 如果你后续更关心 Dice / IoU，也可以改成按指标保存。
        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        # `latest.pth` 保存完整训练现场，方便断点恢复。
        # 与 `best.pth` 不同，这里不仅保存模型参数，还保存优化器和调度器状态。
        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },
            os.path.join(checkpoint_dir, 'latest.pth')
        )

    # 训练结束后，重新加载最佳模型，在验证集上跑一遍测试逻辑，
    # 主要是为了输出最终指标和部分预测可视化图。
    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
            val_loader,
            model,
            criterion,
            logger,
            config,
        )

        # 训练结束后把 best.pth 重命名，直接把最佳 epoch 和 loss 写进文件名。
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )


if __name__ == '__main__':
    config = setting_config
    main(make_config())
