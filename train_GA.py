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

# Genetic Algorithm Parameters
POPULATION_SIZE = 10  # Population size
GENERATION_COUNT = 5  # Number of evolutionary generations
MUTATION_RATE = 0.1  # Probability of mutation
CROSSOVER_RATE = 0.7  # Probability of crossover
TOURNAMENT_SIZE = 3  # Tournament selection pool size


# Hyperparameter search space definition
def random_hyperparameters():
    """Generate random hyperparameters within predefined ranges"""
    return {
        'Init_lr': 10 ** random.uniform(-5, -2),  # Initial learning rate (1e-5 to 1e-2)
        'Min_lr': 10 ** random.uniform(-6, -3),  # Minimum learning rate (1e-6 to 1e-3)
        'Freeze_batch_size': random.choice([4, 6, 8]),  # Batch size during freeze phase
        'Unfreeze_batch_size': random.choice([4, 6, 8]),  # Batch size during unfreeze phase
        'optimizer_type': random.choice(['adam', 'sgd']),  # Optimization algorithm selection
        'momentum': random.uniform(0.8, 0.99),  # Momentum parameter range
    }


# Fitness evaluation function
def evaluate_model(params, model, train_lines, val_lines, input_shape, num_classes, VOCdevkit_path, Cuda, device,
                   distributed, rank, ngpus_per_node, local_rank, optimizer_type, epoch_steps, eval_flag=True,
                   eval_period=5):
    """Evaluate model performance with given hyperparameters using validation loss"""
    model.train()
    batch_size = params['Freeze_batch_size']

    # Configure optimizer based on parameters
    optimizer = {
        'adam': optim.Adam(model.parameters(), params['Init_lr'], betas=(params['momentum'], 0.999), weight_decay=0),
        'sgd': optim.SGD(model.parameters(), params['Init_lr'], momentum=params['momentum'], nesterov=True,
                         weight_decay=0)
    }[params['optimizer_type']]

    # Initialize datasets and data loaders
    train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
    val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

    # Configure learning rate scheduler
    lr_scheduler_func = get_lr_scheduler('cos', params['Init_lr'], params['Min_lr'], 100)

    # Create data loaders with current batch size
    gen = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
                     collate_fn=unet_dataset_collate)
    gen_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=unet_dataset_collate)

    # Initialize loss tracking
    loss_history = LossHistory('logs', model, input_shape=input_shape)

    # Execute training epoch for fitness evaluation
    fit_one_epoch(model, model, loss_history, None, optimizer, 0, epoch_steps, epoch_steps, gen, gen_val, 100, Cuda,
                  True, True, np.ones([num_classes], np.float32), num_classes, False, None, 5, 'logs', 0)

    return loss_history.best_loss  # Return validation loss as fitness metric


# Tournament selection operator
def tournament_selection(population, tournament_size):
    """Select individuals through tournament competition"""
    tournament = random.sample(population, tournament_size)
    tournament.sort(key=lambda x: x['fitness'])
    return tournament[0]  # Return best individual in tournament


# Single-point crossover operator
def crossover(parent1, parent2):
    """Perform genetic crossover between two parents"""
    child = parent1.copy()
    if random.random() < CROSSOVER_RATE:
        crossover_point = random.choice(list(parent1.keys()))
        child[crossover_point] = parent2[crossover_point]
    return child


# Mutation operator
def mutate(individual):
    """Introduce random mutations in genetic material"""
    if random.random() < MUTATION_RATE:
        mutation_point = random.choice(list(individual.keys()))
        if mutation_point in ['Init_lr', 'Min_lr']:
            individual[mutation_point] = 10 ** random.uniform(-5, -2)
        elif mutation_point in ['Freeze_batch_size', 'Unfreeze_batch_size']:
            individual[mutation_point] = random.choice([4, 8, 16])
        elif mutation_point == 'momentum':
            individual[mutation_point] = random.uniform(0.8, 0.99)
        elif mutation_point == 'optimizer_type':
            individual[mutation_point] = random.choice(['adam', 'sgd'])
    return individual


# Genetic Algorithm main process
def genetic_algorithm(model, train_lines, val_lines, input_shape, num_classes, VOCdevkit_path, Cuda, device,
                      distributed, rank, ngpus_per_node, local_rank, epoch_steps, eval_flag=True, eval_period=5):
    """Implement genetic evolution for hyperparameter optimization"""
    # Initialize population
    population = [random_hyperparameters() for _ in range(POPULATION_SIZE)]

    # Evaluate initial population fitness
    for individual in population:
        individual['fitness'] = evaluate_model(individual, model, train_lines, val_lines, input_shape, num_classes,
                                               VOCdevkit_path, Cuda, device, distributed, rank, ngpus_per_node,
                                               local_rank, 'adam', epoch_steps, eval_flag, eval_period)

    # Evolutionary loop
    for generation in range(GENERATION_COUNT):
        print(f"Generation {generation + 1}/{GENERATION_COUNT}")

        # Parent selection phase
        selected_parents = [tournament_selection(population, TOURNAMENT_SIZE) for _ in range(POPULATION_SIZE // 2)]

        # Crossover phase
        children = []
        for i in range(0, len(selected_parents), 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1]
            children.extend([crossover(parent1, parent2), crossover(parent2, parent1)])

        # Mutation phase
        children = [mutate(child) for child in children]

        # Evaluate offspring fitness
        for child in children:
            child['fitness'] = evaluate_model(child, model, train_lines, val_lines, input_shape, num_classes,
                                              VOCdevkit_path, Cuda, device, distributed, rank, ngpus_per_node,
                                              local_rank, 'adam', epoch_steps, eval_flag, eval_period)

        # Survival selection
        population += children
        population.sort(key=lambda x: x['fitness'])
        population = population[:POPULATION_SIZE]  # Keep top performers

    # Return best performing individual
    best_individual = min(population, key=lambda x: x['fitness'])
    print(f"Optimized hyperparameters: {best_individual}")
    return best_individual


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

    # Load pretrained weights with compatibility check
    if model_path != '':
        model_dict = model.state_dict()
        model.to(device)
        pretrained_dict = torch.load(model_path)

        # Filter matching parameters
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and model_dict[k].shape == v.shape}

        # Update model state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # Load dataset metadata
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()

    epoch_steps = len(train_lines) // Freeze_batch_size

    # Execute genetic optimization
    best_params = genetic_algorithm(model, train_lines, val_lines, input_shape, num_classes, VOCdevkit_path, Cuda,
                                    device, distributed, 0, ngpus_per_node, 0, epoch_steps,
                                    eval_flag=eval_flag, eval_period=eval_period)

    print("Final optimized hyperparameters: ", best_params)