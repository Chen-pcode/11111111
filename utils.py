import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt


def set_seed(seed):
    """
    固定训练中常见的随机源，尽量保证实验可复现。

    这里同时固定了：
    1. Python 原生随机数
    2. NumPy 随机数
    3. PyTorch CPU/GPU 随机数
    4. cuDNN 的确定性行为

    代价是有些操作可能会比完全非确定性模式稍慢，但对论文复现更友好。
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    """
    创建日志记录器。

    参数：
    - name: logger 名称，当前训练入口里传入的是 `train`
    - log_dir: 日志目录，最终会生成类似 `train.info.log` 的日志文件

    这里使用 `TimedRotatingFileHandler`，按天滚动日志文件。对于当前项目来说，
    核心作用是把训练过程中打印的 loss、指标和配置保存下来，便于后续复现实验。
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(
        info_name,
        when='D',
        encoding='utf-8'
    )
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    info_handler.setFormatter(formatter)
    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    """
    把配置类中的公开属性写入日志。

    训练结束很久之后再回来看结果时，最常见的问题不是“代码跑没跑”，
    而是“不知道这次实验到底用了什么参数”。这个函数就是专门解决这个问题的。
    """
    config_dict = config.__dict__
    logger.info('#----------Config info----------#')
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        logger.info(f'{k}: {v},')


def get_optimizer(config, model):
    """
    根据配置创建优化器。

    当前仓库虽然列出了多种优化器分支，但一次训练只会命中 `config.opt`
    对应的那一支。默认配置使用 `AdamW`。
    """
    assert config.opt in [
        'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'
    ], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr=config.lr,
            rho=config.rho,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr=config.lr,
            lr_decay=config.lr_decay,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr=config.lr,
            lambd=config.lambd,
            alpha=config.alpha,
            t0=config.t0,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            alpha=config.alpha,
            eps=config.eps,
            centered=config.centered,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr=config.lr,
            etas=config.etas,
            step_sizes=config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            dampening=config.dampening,
            nesterov=config.nesterov
        )
    else:
        return torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.05,
        )


def get_scheduler(config, optimizer):
    """
    根据配置创建学习率调度器。

    默认配置使用 `CosineAnnealingLR`，即余弦退火。
    代码里也保留了 warmup + cosine / warmup + multistep 等分支，
    如果后面调参，可以直接在配置文件里切换。
    """
    assert config.sch in [
        'StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR',
        'ReduceLROnPlateau', 'CosineAnnealingWarmRestarts',
        'WP_MultiStepLR', 'WP_CosineLR'
    ], 'Unsupported scheduler!'

    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.milestones,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.T_max,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            patience=config.patience,
            threshold=config.threshold,
            threshold_mode=config.threshold_mode,
            cooldown=config.cooldown,
            min_lr=config.min_lr,
            eps=config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma ** len(
            [m for m in config.milestones if m <= epoch]
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
            math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler


def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    """
    保存预测可视化图。

    每次保存三张内容拼成的一张图：
    1. 原图
    2. 真实 mask
    3. 预测 mask

    作用很直接：训练指标告诉你“数值上好不好”，这张图告诉你“模型到底分到了哪”。
    """
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img

    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)

    plt.figure(figsize=(7, 15))

    plt.subplot(3, 1, 1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(3, 1, 2)
    plt.imshow(msk, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 1, 3)
    plt.imshow(msk_pred, cmap='gray')
    plt.axis('off')

    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    plt.savefig(save_path + str(i) + '.png')
    plt.close()


class BCELoss(nn.Module):
    """
    像素级 BCE 损失。

    输入和标签原本是 `[B, 1, H, W]`，这里会展平为 `[B, H*W]`。
    这样本质上就是把一张分割图看成很多个独立像素二分类问题。
    """

    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    """
    Dice 损失。

    Dice 更关注预测区域和真实区域的整体重叠程度，对医学图像分割很常见，
    因为病灶区域往往只占图像一小部分，单用 BCE 容易被大量背景像素主导。
    """

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size
        return dice_loss


class BceDiceLoss(nn.Module):
    """
    BCE + Dice 的组合损失。

    组合思路：
    - BCE 约束单个像素的分类正确性
    - Dice 强调整体目标区域的重叠质量

    医学分割里这类组合比单独用一种损失更常见，也更稳。
    """

    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)
        loss = self.wd * diceloss + self.wb * bceloss
        return loss


class GT_BceDiceLoss(nn.Module):
    """
    带深监督的组合损失。

    参数：
    - gt_pre: 来自多个 decoder 尺度的辅助预测
    - out: 最终输出
    - target: 原图尺度的真实 mask

    计算逻辑：
    1. 最终输出 `out` 计算一份主损失
    2. 5 个辅助输出也分别和原图标签计算损失
    3. 越浅层、越接近最终输出的辅助预测权重越大

    这样做的目的，是让中间层也尽早学到“哪里是病灶”，帮助梯度传播和收敛。
    """

    def __init__(self, wb=1, wd=1):
        super(GT_BceDiceLoss, self).__init__()
        self.bcedice = BceDiceLoss(wb, wd)

    def forward(self, gt_pre, out, target):
        bcediceloss = self.bcedice(out, target)
        gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre

        gt_loss = (
            self.bcedice(gt_pre5, target) * 0.1 +
            self.bcedice(gt_pre4, target) * 0.2 +
            self.bcedice(gt_pre3, target) * 0.3 +
            self.bcedice(gt_pre2, target) * 0.4 +
            self.bcedice(gt_pre1, target) * 0.5
        )
        return bcediceloss + gt_loss


class myToTensor:
    """
    把 NumPy 图像转成 PyTorch Tensor，并把通道维从 HWC 改成 CHW。

    模型卷积层要求输入是 `[C, H, W]`，而 PIL/NumPy 图像常见格式是 `[H, W, C]`，
    所以这里必须做一次维度调整。
    """

    def __init__(self):
        pass

    def __call__(self, data):
        image, mask = data
        return torch.tensor(image).permute(2, 0, 1), torch.tensor(mask).permute(2, 0, 1)


class myResize:
    """
    同时缩放图像和 mask。

    分割任务里图像和标签必须做完全一致的几何变换，否则监督关系会错位。
    """

    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w

    def __call__(self, data):
        image, mask = data
        return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w])


class myRandomHorizontalFlip:
    """
    随机水平翻转。

    注意这里不是只翻图像，mask 也必须同步翻转。
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.hflip(image), TF.hflip(mask)
        else:
            return image, mask


class myRandomVerticalFlip:
    """
    随机垂直翻转。
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.vflip(image), TF.vflip(mask)
        else:
            return image, mask


class myRandomRotation:
    """
    随机旋转增强。

    这里有个值得注意的实现细节：
    `self.angle` 是在对象初始化时采样一次的，而不是每次调用时重新采样。
    这意味着同一个 transform 实例在整个训练过程中会使用固定角度，只是按概率决定
    当前样本是否旋转。这个实现能跑，但严格来说不算“每张图随机角度旋转”。
    如果你后续想把增强改得更合理，这里是一个优先修改点。
    """

    def __init__(self, p=0.5, degree=[0, 360]):
        self.angle = random.uniform(degree[0], degree[1])
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.rotate(image, self.angle), TF.rotate(mask, self.angle)
        else:
            return image, mask


class myNormalize:
    """
    按数据集统计量做归一化。

    作者没有使用 ImageNet 均值/方差，而是使用了针对各数据集统计得到的数值。
    这在医学图像里很常见，因为图像分布通常与自然图像差异较大。
    """

    def __init__(self, data_name, train=True):
        if data_name == 'isic18':
            if train:
                self.mean = 157.561
                self.std = 26.706
            else:
                self.mean = 149.034
                self.std = 32.022
        elif data_name == 'isic17':
            if train:
                self.mean = 159.922
                self.std = 28.871
            else:
                self.mean = 148.429
                self.std = 25.748
        elif data_name == 'isic18_82':
            if train:
                self.mean = 156.2899
                self.std = 26.5457
            else:
                self.mean = 149.8485
                self.std = 35.3346

    def __call__(self, data):
        img, msk = data

        # 先减均值除标准差，再做 min-max 拉伸回 0~255。
        # 这不是最常见的标准化写法，但作者原始代码就是这样做的。
        img_normalized = (img - self.mean) / self.std
        img_normalized = (
            (img_normalized - np.min(img_normalized)) /
            (np.max(img_normalized) - np.min(img_normalized))
        ) * 255.
        return img_normalized, msk
