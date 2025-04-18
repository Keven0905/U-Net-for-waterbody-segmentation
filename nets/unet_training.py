import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def CE_Loss(inputs, target, cls_weights, num_classes=2):
    """Cross-Entropy Loss with spatial dimension alignment
    Args:
        inputs: Network output [N, C, H, W]
        target: Ground truth [N, H, W]
        cls_weights: Class weighting tensor
        num_classes: Number of semantic classes (ignore index)
    """
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()

    # Spatial resolution alignment
    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # Flatten tensors for loss calculation
    temp_inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, c)
    temp_target = target.view(-1)

    return nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)


def Focal_Loss(inputs, target, cls_weights, num_classes=2, alpha=0.5, gamma=2):
    """Focal Loss for class imbalance mitigation
    Args:
        alpha: Balancing parameter (0.5 for default class balance)
        gamma: Focusing parameter (>0 reduces well-classified example weight)
    """
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()

    # Spatial alignment
    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # Tensor reshaping
    temp_inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, c)
    temp_target = target.view(-1)

    # Focal loss computation
    logpt = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes,
                                 reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha  # Class balancing
    loss = -((1 - pt) ** gamma) * logpt  # Example focusing
    return loss.mean()


def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    """Dice Loss with F-beta score formulation
    Args:
        beta: Weighting between precision and recall (β>1 emphasizes recall)
        smooth: Numerical stability constant
    """
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()

    # Resolution matching
    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # Probability normalization
    temp_inputs = torch.softmax(inputs.permute(0, 2, 3, 1).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # Confusion matrix components
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, dim=[0, 1])  # True positives
    fp = torch.sum(temp_inputs, dim=[0, 1]) - tp  # False positives
    fn = torch.sum(temp_target[..., :-1], dim=[0, 1]) - tp  # False negatives

    # F-beta score calculation
    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    return 1 - torch.mean(score)  # Dice loss formulation


def weights_init(net, init_type='normal', init_gain=0.02):
    """Parameter initialization strategies
    Supports: Normal, Xavier, Kaiming, Orthogonal initialization
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'Unsupported initialization: {init_type}')
        elif 'BatchNorm2d' in classname:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print(f'Initializing network with {init_type} initialization')
    net.apply(init_func)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters,
                     warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    """Learning rate scheduling strategies
    Returns:
        Function that computes lr based on iteration count
    """

    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters,
                          warmup_lr_start, no_aug_iter, iters):
        """YOLOX-style warmup + cosine annealing scheduler
        1. Quadratic warmup phase
        2. Cosine annealing
        3. Final constant phase
        """
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * (iters / warmup_total_iters) ** 2 + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            ratio = (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter)
            lr = min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * ratio))
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        """Stepwise learning rate decay"""
        n = iters // step_size
        return lr * (decay_rate ** n)

    # Scheduler configuration
    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        return partial(yolox_warm_cos_lr, lr, min_lr, total_iters,
                       warmup_total_iters, warmup_lr_start, no_aug_iter)

    # Step decay configuration
    decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
    step_size = total_iters / step_num
    return partial(step_lr, lr, decay_rate, step_size)


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    """Updates optimizer's learning rate using scheduling function"""
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr