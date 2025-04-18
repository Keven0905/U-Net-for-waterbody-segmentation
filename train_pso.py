import random
import numpy as np
import torch
import os
from functools import partial
from torch.utils.data import DataLoader
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import seed_everything
from utils.utils_fit import fit_one_epoch
from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory
import datetime

# PSO参数配置
NUM_PARTICLES = 10  # 粒子数量
MAX_ITERATIONS = 5  # 最大迭代次数
W = 0.5  # 惯性权重
C1 = 1.5  # 个体学习因子
C2 = 2.0  # 群体学习因子

# 超参数搜索空间定义
PARAM_BOUNDS = {
    'Init_lr': (1e-5, 1e-2),  # 初始学习率范围
    'Min_lr': (1e-6, 1e-3),  # 最小学习率范围
    'Freeze_batch_size': [4, 8, 16],  # 冻结阶段批次大小选项
    'Unfreeze_batch_size': [4, 8, 16],  # 解冻阶段批次大小选项
    'optimizer_type': ['adam', 'sgd'],  # 优化器类型选项
    'momentum': (0.8, 0.99)  # 动量系数范围
}


class Particle:
    """粒子类"""

    def __init__(self):
        # 初始化位置（参数组合）
        self.position = {
            'Init_lr': 10 ** random.uniform(np.log10(PARAM_BOUNDS['Init_lr'][0]),
                                            np.log10(PARAM_BOUNDS['Init_lr'][1])),
            'Min_lr': 10 ** random.uniform(np.log10(PARAM_BOUNDS['Min_lr'][0]),
                                           np.log10(PARAM_BOUNDS['Min_lr'][1])),
            'Freeze_batch_size': random.choice(PARAM_BOUNDS['Freeze_batch_size']),
            'Unfreeze_batch_size': random.choice(PARAM_BOUNDS['Unfreeze_batch_size']),
            'optimizer_type': random.choice(PARAM_BOUNDS['optimizer_type']),
            'momentum': random.uniform(*PARAM_BOUNDS['momentum'])
        }

        # 初始化速度
        self.velocity = {
            'Init_lr': 0,
            'Min_lr': 0,
            'Freeze_batch_size': 0,
            'Unfreeze_batch_size': 0,
            'momentum': 0
        }

        # 个体最优记录
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')


def pso_optimization(model, train_lines, val_lines, input_shape, num_classes,
                     VOCdevkit_path, Cuda, device, distributed, rank,
                     ngpus_per_node, local_rank, epoch_steps):
    """PSO主优化函数"""
    # 初始化粒子群
    particles = [Particle() for _ in range(NUM_PARTICLES)]
    global_best_position = None
    global_best_fitness = float('inf')

    # 优化循环
    for iteration in range(MAX_ITERATIONS):
        print(f"\n=== Iteration {iteration + 1}/{MAX_ITERATIONS} ===")

        # 评估所有粒子
        for idx, particle in enumerate(particles):
            print(f"Evaluating particle {idx + 1}/{NUM_PARTICLES}")

            # 应用边界约束
            particle.position['Init_lr'] = np.clip(particle.position['Init_lr'],
                                                   *PARAM_BOUNDS['Init_lr'])
            particle.position['Min_lr'] = np.clip(particle.position['Min_lr'],
                                                  *PARAM_BOUNDS['Min_lr'])
            particle.position['momentum'] = np.clip(particle.position['momentum'],
                                                    *PARAM_BOUNDS['momentum'])

            # 处理离散参数
            particle.position['Freeze_batch_size'] = closest_value(
                particle.position['Freeze_batch_size'],
                PARAM_BOUNDS['Freeze_batch_size']
            )
            particle.position['Unfreeze_batch_size'] = closest_value(
                particle.position['Unfreeze_batch_size'],
                PARAM_BOUNDS['Unfreeze_batch_size']
            )

            # 评估适应度
            try:
                fitness = evaluate_model(
                    particle.position, model, train_lines, val_lines, input_shape, num_classes,
                    VOCdevkit_path, Cuda, device, distributed, rank, ngpus_per_node, local_rank,
                    epoch_steps
                )
            except Exception as e:
                print(f"Evaluation failed: {str(e)}")
                fitness = float('inf')

            # 更新个体最优
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()
                print(f"New personal best: {fitness:.4f}")

            # 更新全局最优
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position.copy()
                print(f"New global best: {fitness:.4f}")

        # 更新粒子速度和位置
        for particle in particles:
            # 连续参数更新
            for key in ['Init_lr', 'Min_lr', 'momentum']:
                r1 = random.random()
                r2 = random.random()
                # 速度更新公式
                particle.velocity[key] = W * particle.velocity[key] + \
                                         C1 * r1 * (particle.best_position[key] - particle.position[key]) + \
                                         C2 * r2 * (global_best_position[key] - particle.position[key])
                # 位置更新
                particle.position[key] += particle.velocity[key]

            # 离散参数更新
            for key in ['Freeze_batch_size', 'Unfreeze_batch_size']:
                # 离散参数速度更新
                r1 = random.random()
                r2 = random.random()
                particle.velocity[key] = W * particle.velocity[key] + \
                                         C1 * r1 * (particle.best_position[key] - particle.position[key]) + \
                                         C2 * r2 * (global_best_position[key] - particle.position[key])
                # 位置更新并映射到最近有效值
                new_value = particle.position[key] + particle.velocity[key]
                particle.position[key] = closest_value(new_value, PARAM_BOUNDS[key])

        # 打印当前最优
        print(f"\nCurrent Best Fitness: {global_best_fitness:.4f}")
        print("Best Parameters:")
        for k, v in global_best_position.items():
            print(f"  {k:>20}: {v}")

    return global_best_position


def closest_value(value, options):
    """将连续值映射到最近的离散选项"""
    return min(options, key=lambda x: abs(x - value))


def evaluate_model(params, model, train_lines, val_lines, input_shape, num_classes,
                   VOCdevkit_path, Cuda, device, distributed, rank, ngpus_per_node,
                   local_rank, epoch_steps):
    """模型评估函数"""
    # 确保模型在正确设备
    model.to(device)
    model.train()

    # 参数边界最终检查
    params['momentum'] = np.clip(params['momentum'], 0.8, 0.99)
    params['Init_lr'] = np.clip(params['Init_lr'], 1e-5, 1e-2)
    params['Min_lr'] = np.clip(params['Min_lr'], 1e-6, 1e-3)

    # 优化器初始化
    try:
        if params['optimizer_type'] == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=params['Init_lr'],
                betas=(params['momentum'], 0.999),
                weight_decay=0
            )
        else:
            optimizer = optim.SGD(
                model.parameters(),
                lr=params['Init_lr'],
                momentum=params['momentum'],
                nesterov=True,
                weight_decay=0
            )
    except Exception as e:
        print(f"Optimizer init failed: {str(e)}")
        return float('inf')

    # 数据加载
    train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
    val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

    gen = DataLoader(
        train_dataset,
        batch_size=int(params['Freeze_batch_size']),
        shuffle=True,
        num_workers=4,
        collate_fn=unet_dataset_collate,
        pin_memory=True
    )
    gen_val = DataLoader(
        val_dataset,
        batch_size=int(params['Unfreeze_batch_size']),
        shuffle=False,
        num_workers=4,
        collate_fn=unet_dataset_collate,
        pin_memory=True
    )

    # 训练验证流程
    loss_history = LossHistory('logs', model, input_shape=input_shape)
    try:
        fit_one_epoch(
            model, model, loss_history, None, optimizer, 0,
            epoch_steps, epoch_steps, gen, gen_val,
            100, Cuda, True, True, np.ones([num_classes], np.float32),
            num_classes, False, None, 5, 'logs', 0
        )
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return float('inf')

    return loss_history.best_loss


if __name__ == "__main__":
    # 硬件配置
    Cuda = True
    seed = 11
    distributed = False
    device = torch.device('cuda' if Cuda and torch.cuda.is_available() else 'cpu')
    seed_everything(seed)

    # 模型参数
    num_classes = 2
    backbone = "resnet50"
    pretrained = False
    model_path = "model_data/unet_resnet_voc.pth"
    input_shape = [512, 512]

    # 初始化模型
    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).to(device)
    if not pretrained:
        weights_init(model)

    # 加载数据
    VOCdevkit_path = 'VOCdevkit'
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()

    epoch_steps = len(train_lines) // 4  # 初始批次大小

    # 运行PSO优化
    print("\nStarting PSO Hyperparameter Optimization...")
    best_params = pso_optimization(
        model, train_lines, val_lines, input_shape, num_classes,
        VOCdevkit_path, Cuda, device, distributed, 0,
        torch.cuda.device_count(), 0, epoch_steps
    )

    # 输出最终结果
    print("\n=== Optimization Completed ===")
    print("Best Hyperparameters Found:")
    for k, v in best_params.items():
        print(f"{k:>25}: {v}")
    print(f"Best Validation Loss: {min_loss:.4f}")

    # 使用最优参数训练最终模型
    print("\nTraining Final Model with Best Parameters...")
    # ...（此处添加完整训练流程）...