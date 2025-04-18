import random
import numpy as np
import torch
import os
from functools import partial
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import seed_everything
from utils.utils_fit import fit_one_epoch
from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory
import datetime

# Ant Colony Optimization (ACO) hyperparameters
NUM_ANTS = 10  # Population size
ITERATIONS = 5  # Number of optimization iterations
EVAPORATION_RATE = 0.5  # Pheromone evaporation coefficient
ALPHA = 1  # Pheromone influence weight
BETA = 2  # Heuristic information weight


# Hyperparameter search space definition
def random_hyperparameters():
    """Generate random hyperparameters within predefined ranges"""
    Init_lr = 10 ** random.uniform(-5, -2)  # Initial learning rate (1e-5 to 1e-2)
    return {
        'Init_lr': max(Init_lr, 1e-5),  # Ensure minimum learning rate threshold
        'Min_lr': 10 ** random.uniform(-6, -3),  # Minimum learning rate (1e-6 to 1e-3)
        'Freeze_batch_size': random.choice([4, 8, 16]),  # Batch size during freeze phase
        'Unfreeze_batch_size': random.choice([4, 8, 16]),  # Batch size during unfreeze phase
        'optimizer_type': random.choice(['adam', 'sgd']),  # Optimization algorithm
        'momentum': random.uniform(0.8, 0.99),  # Momentum parameter range
    }

# Model evaluation function
def evaluate_model(params, model, train_lines, val_lines, input_shape, num_classes, VOCdevkit_path, Cuda, device,
                   distributed, rank, ngpus_per_node, local_rank, optimizer_type, epoch_steps, eval_flag=True,
                   eval_period=5):
    """Evaluate model performance with given hyperparameters using validation loss"""
    model.train()
    batch_size = params['Freeze_batch_size']

    def validate_lr(lr):
        """Ensure learning rate validity"""
        return max(lr, 1e-4) if lr <= 0 else lr

    params['Init_lr'] = validate_lr(params['Init_lr'])

    # Configure optimizer based on parameters
    optimizer = {
        'adam': optim.Adam(model.parameters(), params['Init_lr'], betas=(params['momentum'], 0.999), weight_decay=0),
        'sgd': optim.SGD(model.parameters(), params['Init_lr'], momentum=params['momentum'], nesterov=True,
                         weight_decay=0)
    }[params['optimizer_type']]

    # Prepare datasets and data loaders
    train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
    val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

    # Configure learning rate scheduler
    lr_scheduler_func = get_lr_scheduler('cos', params['Init_lr'], params['Min_lr'], 100)

    # Initialize data loaders
    gen = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
                     collate_fn=unet_dataset_collate)
    gen_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=unet_dataset_collate)

    # Initialize loss tracking
    loss_history = LossHistory('logs', model, input_shape=input_shape)

    # Execute training epoch for evaluation
    fit_one_epoch(model, model, loss_history, None, optimizer, 0, epoch_steps, epoch_steps, gen, gen_val, 100, Cuda,
                  True, True, np.ones([num_classes], np.float32), num_classes, False, None, 5, 'logs', 0)

    return loss_history.best_loss  # Return validation loss as fitness metric

# ACO main algorithm implementation
def ant_colony_optimization(model, train_lines, val_lines, input_shape, num_classes, VOCdevkit_path, Cuda, device,
                            distributed, rank, ngpus_per_node, local_rank, epoch_steps, eval_flag=True, eval_period=5):
    """Implement Ant Colony Optimization for hyperparameter tuning"""
    # Initialize pheromone matrix with random values
    pheromones = {
        'Init_lr': [random.uniform(-5, -2) for _ in range(NUM_ANTS)],
        'Min_lr': [random.uniform(-6, -3) for _ in range(NUM_ANTS)],
        'Freeze_batch_size': [random.choice([4, 6, 8]) for _ in range(NUM_ANTS)],
        'Unfreeze_batch_size': [random.choice([4, 6, 8]) for _ in range(NUM_ANTS)],
        'optimizer_type': [random.choice(['adam', 'sgd']) for _ in range(NUM_ANTS)],
        'momentum': [random.uniform(0.8, 0.99) for _ in range(NUM_ANTS)],
        'fitness': [float('inf') for _ in range(NUM_ANTS)]
    }

    # Optimization loop
    for iteration in range(ITERATIONS):
        print(f"Iteration {iteration + 1}/{ITERATIONS}")

        # Evaluate each ant's solution
        for i in range(NUM_ANTS):
            params = {k: pheromones[k][i] for k in pheromones if k != 'fitness'}
            fitness = evaluate_model(params, model, train_lines, val_lines, input_shape, num_classes, VOCdevkit_path,
                                     Cuda, device, distributed, rank, ngpus_per_node, local_rank, 'adam', epoch_steps,
                                     eval_flag, eval_period)
            pheromones['fitness'][i] = fitness

        # Update pheromone trails
        best_fitness_idx = np.argmin(pheromones['fitness'])
        best_fitness = pheromones['fitness'][best_fitness_idx]

        # Apply pheromone evaporation
        for param in pheromones:
            if param != 'fitness':
                pheromones[param] = [v * (1 - EVAPORATION_RATE) for v in pheromones[param]]

        # Deposit new pheromones based on best solution
        for param in pheromones:
            if param != 'fitness':
                best_value = pheromones[param][best_fitness_idx]
                pheromones[param] = [v + ALPHA * best_value for v in pheromones[param]]

    # Extract best solution
    best_ant_idx = np.argmin(pheromones['fitness'])
    best_params = {k: pheromones[k][best_ant_idx] for k in pheromones if k != 'fitness'}

    print(f"Optimal hyperparameters: {best_params}")
    return best_params

# Main training execution
if __name__ == "__main__":
    # Hardware configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Cuda = True
    seed = 11
    distributed = False
    sync_bn = False
    fp16 = True

    # Model architecture parameters
    num_classes = 2
    backbone = "resnet50"
    pretrained = False
    model_path = "model_data/unet_resnet_voc.pth"
    input_shape = [512, 512]

    # Training schedule parameters
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 4
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 4
    Freeze_Train = True

    # Optimization parameters
    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    lr_decay_type = 'cos'

    # Experiment management
    save_period = 5
    save_dir = 'logs'
    eval_flag = True
    eval_period = 5

    # Dataset configuration
    VOCdevkit_path = 'VOCdevkit'
    dice_loss = True
    focal_loss = True
    cls_weights = np.ones([num_classes], np.float32)
    num_workers = 4

    # Initialize environment
    seed_everything(seed)
    ngpus_per_node = torch.cuda.device_count()

    # Model initialization
    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train()
    if not pretrained:
        weights_init(model)

    # Load pretrained weights
    if model_path != '':
        model_dict = model.state_dict()
        model.to(device)
        pretrained_dict = torch.load(model_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # Load dataset metadata
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()

    epoch_steps = len(train_lines) // Freeze_batch_size

    # Execute hyperparameter optimization
    best_params = ant_colony_optimization(model, train_lines, val_lines, input_shape, num_classes, VOCdevkit_path, Cuda,
                                          device, distributed, 0, ngpus_per_node, 0, epoch_steps, eval_flag=eval_flag,
                                          eval_period=eval_period)

    print("Optimized hyperparameters: ", best_params)