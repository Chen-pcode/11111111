import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs


def train_one_epoch(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    step,
                    logger,
                    config,
                    writer):
    """
    训练一个 epoch。

    数据流：
    1. 从 `train_loader` 取出一个 batch
    2. 前向计算得到辅助预测和最终预测
    3. 用损失函数计算总 loss
    4. 反向传播并更新参数
    5. 记录日志和 TensorBoard

    输入张量形状：
    - images:  [B, 3, H, W]
    - targets: [B, 1, H, W]
    """
    model.train()
    loss_list = []

    for iter, data in enumerate(train_loader):
        # `step` 用于 TensorBoard 横轴，这里作者按“累计迭代次数”来记。
        step += iter

        optimizer.zero_grad()
        images, targets = data
        images = images.cuda(non_blocking=True).float()
        targets = targets.cuda(non_blocking=True).float()

        # 当 gt_ds=True 时，模型返回：
        # 1. 5 个辅助预测组成的元组 gt_pre
        # 2. 最终输出 out
        gt_pre, out = model(images)
        loss = criterion(gt_pre, out, targets)

        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)

    # 当前默认调度器是按 epoch 更新一次学习率，而不是按 iteration 更新。
    scheduler.step()
    return step


def val_one_epoch(test_loader,
                  model,
                  criterion,
                  epoch,
                  logger,
                  config):
    """
    验证一个 epoch。

    验证阶段不更新参数，只做前向推理和指标统计。
    指标计算方式是：把整套验证集的预测和真值都展平为一维数组，再统一计算混淆矩阵。

    当前统计的指标包括：
    - loss
    - mIoU
    - F1 / Dice
    - accuracy
    - specificity
    - sensitivity
    """
    model.eval()
    preds = []
    gts = []
    loss_list = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img = img.cuda(non_blocking=True).float()
            msk = msk.cuda(non_blocking=True).float()

            gt_pre, out = model(img)
            loss = criterion(gt_pre, out, msk)

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())

            # 这里做了兼容写法：如果 out 是 tuple，就取第一个元素。
            # 但按当前 EGEUNet 实现，out 本身是单个 tensor。
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        # 概率图按阈值二值化，默认阈值是 0.5。
        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = (
            f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, '
            f'f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, '
            f'specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        )
        print(log_info)
        logger.info(log_info)
    else:
        # 非验证间隔 epoch 只记录 loss，不做完整指标统计。
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)


def test_one_epoch(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   test_data_name=None):
    """
    测试阶段逻辑。

    与验证阶段几乎一致，区别主要有两点：
    1. 会按 `save_interval` 周期性保存预测可视化图
    2. 日志文案写成 test，而不是 val
    """
    model.eval()
    preds = []
    gts = []
    loss_list = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img = img.cuda(non_blocking=True).float()
            msk = msk.cuda(non_blocking=True).float()

            gt_pre, out = model(img)
            loss = criterion(gt_pre, out, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)

            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

            if i % config.save_interval == 0:
                save_imgs(
                    img,
                    msk,
                    out,
                    i,
                    config.work_dir + 'outputs/',
                    config.datasets,
                    config.threshold,
                    test_data_name=test_data_name
                )

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)

        log_info = (
            f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, '
            f'f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, '
            f'specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        )
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)
