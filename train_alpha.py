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

# Alpha Evolution Algorithm Parameters
POPULATION_SIZE = 10  # Population size
MAX_GENERATIONS = 5  # Maximum evolutionary iterations
ALPHA_DECAY = 0.9  # Perturbation decay coefficient
EVAPORATION_RATE = 0.2  # Path evaporation rate

# Hyperparameter search space definition
def random_hyperparameters():
    """Generate random hyperparameters within predefined ranges"""
    return {
        'Init_lr': 10 ** random.uniform(-5, -2),  # Initial learning rate (1e-5 to 1e-2)
        'Min_lr': 10 ** random.uniform(-6, -3),  # Minimum learning rate (1e-6 to 1e-3)
        'Freeze_batch_size': random.choice([4, 8, 16]),  # Batch size during freeze phase
        'Unfreeze_batch_size': random.choice([4, 8, 16]),  # Batch size during unfreeze phase
        'optimizer_type': random.choice(['adam', 'sgd']),  # Optimization algorithm selection
        'momentum': random.uniform(0.8, 0.99),  # Momentum coefficient
    }

def evaluate_model(params, model, train_lines, val_lines, input_shape, num_classes,
                   VOCdevkit_path, Cuda, device, distributed, rank, ngpus_per_node,
                   local_rank, optimizer_type, epoch_steps, eval_flag=True, eval_period=5):
    """Evaluate model performance with given hyperparameters using validation loss"""
    # Ensure model is on correct device
    model.to(device)  # <--- Critical fix for device placement
    model.train()

    batch_size = params['Freeze_batch_size']

    # Initialize optimizer configuration
    optimizer = {
        'adam': optim.Adam(model.parameters(), params['Init_lr'], betas=(params['momentum'], 0.999)),
        'sgd': optim.SGD(model.parameters(), params['Init_lr'], momentum=params['momentum'])
    }[params['optimizer_type']]

    # Data loading configuration
    train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
    val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

    gen = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                     num_workers=4, collate_fn=unet_dataset_collate)
    gen_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=4, collate_fn=unet_dataset_collate)

    # Training and validation execution
    loss_history = LossHistory('logs', model, input_shape=input_shape)

    # Critical: Ensure data is on same device as model
    fit_one_epoch(
        model, model, loss_history, None, optimizer, 0, epoch_steps, epoch_steps,
        gen, gen_val, 100, Cuda, True, True, np.ones([num_classes], np.float32),
        num_classes, False, None, 5, 'logs', 0
    )

    return loss_history.best_loss

# Alpha Evolution Algorithm Core Implementation
class AlphaEvolution:
    def __init__(self, population_size):
        """Initialize evolutionary algorithm components"""
        self.population = [random_hyperparameters() for _ in range(population_size)]
        self.pa = random_hyperparameters()  # Path vector Pa (exploration path)
        self.pb = random_hyperparameters()  # Path vector Pb (exploitation path)
        self.best_fitness = float('inf')
        self.best_individual = None

    def adaptive_alpha(self, generation, max_generations):
        """Calculate adaptive perturbation coefficient"""
        t = generation / max_generations
        return np.exp(np.log(1 - t) - (4 * t) ** 2) * ALPHA_DECAY ** generation

    def update_paths(self, individual, alpha):
        """Update evolutionary path vectors"""
        # Update path Pa (random sampling based)
        for key in self.pa:
            if random.random() < 0.5:
                self.pa[key] = (1 - alpha) * self.pa[key] + alpha * individual[key]

        # Update path Pb (fitness weighted)
        for key in self.pb:
            self.pb[key] = (1 - alpha) * self.pb[key] + alpha * individual[key] * individual['fitness']

    def evolve(self, model, train_lines, val_lines, input_shape, num_classes,
               VOCdevkit_path, Cuda, device, distributed, rank, ngpus_per_node,
               local_rank, epoch_steps):
        """Execute evolutionary optimization process"""
        # Evaluate initial population fitness
        for ind in self.population:
            ind['fitness'] = evaluate_model(ind, model, train_lines, val_lines, input_shape,
                                            num_classes, VOCdevkit_path, Cuda, device,
                                            distributed, rank, ngpus_per_node, local_rank,
                                            'adam', epoch_steps)

        # Evolutionary iteration loop
        for generation in range(MAX_GENERATIONS):
            alpha = self.adaptive_alpha(generation, MAX_GENERATIONS)

            # Generate new population
            new_population = []
            for i in range(POPULATION_SIZE):
                # Select evolutionary path
                if random.random() < 0.5:
                    base_params = self.pa.copy()
                else:
                    base_params = self.pb.copy()

                # Generate perturbation delta
                delta = {k: alpha * (random.random() * 2 - 1) for k in base_params}

                # Create new individual
                child = {}
                for key in base_params:
                    # Numerical parameter handling
                    if key in ['Init_lr', 'Min_lr', 'momentum']:
                        child[key] = base_params[key] + delta[key]
                        child[key] = max(child[key], 1e-6)  # Prevent negative values
                    elif key in ['Freeze_batch_size', 'Unfreeze_batch_size']:
                        child[key] = int(base_params[key] + delta[key])
                        child[key] = max(child[key], 4)  # Minimum batch size
                    else:
                        child[key] = base_params[key] if random.random() > 0.2 else random_hyperparameters()[key]

                # Apply parameter constraints
                child['Init_lr'] = np.clip(child['Init_lr'], 1e-5, 1e-2)
                child['Min_lr'] = np.clip(child['Min_lr'], 1e-6, 1e-3)
                child['momentum'] = np.clip(child['momentum'], 0.8, 0.99)

                # Evaluate new individual
                child['fitness'] = evaluate_model(child, model, train_lines, val_lines, input_shape,
                                                  num_classes, VOCdevkit_path, Cuda, device,
                                                  distributed, rank, ngpus_per_node, local_rank,
                                                  'adam', epoch_steps)

                # Update best solution
                if child['fitness'] < self.best_fitness:
                    self.best_fitness = child['fitness']
                    self.best_individual = child.copy()

                new_population.append(child)

                # Apply path evaporation
                for key in self.pa:
                    self.pa[key] *= (1 - EVAPORATION_RATE)
                for key in self.pb:
                    self.pb[key] *= (1 - EVAPORATION_RATE)

            # Population selection and replacement
            combined = self.population + new_population
            combined.sort(key=lambda x: x['fitness'])
            self.population = combined[:POPULATION_SIZE]

            print(f"Generation {generation + 1}, Best Fitness: {self.best_fitness:.4f}")

        return self.best_individual

# Main Training Execution
if __name__ == "__main__":
    # Hardware Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Cuda = True
    seed = 11
    distributed = False
    sync_bn = False
    fp16 = True

    # Model Architecture Parameters
    num_classes = 2
    backbone = "resnet50"
    pretrained = False
    model_path = "model_data/unet_resnet_voc.pth"
    input_shape = [512, 512]

    # Training Schedule Parameters
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 4
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 4
    Freeze_Train = True

    # Optimization Parameters
    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    lr_decay_type = 'cos'

    # Experiment Management
    save_period = 5
    save_dir = 'logs'
    eval_flag = True
    eval_period = 5

    # Dataset Configuration
    VOCdevkit_path = 'VOCdevkit'
    dice_loss = True
    focal_loss = True
    cls_weights = np.ones([num_classes], np.float32)
    num_workers = 4

    # Environment Initialization
    seed_everything(seed)
    ngpus_per_node = torch.cuda.device_count()

    # Model Initialization
    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train()

    # Dataset Loading
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()

    epoch_steps = len(train_lines) // Freeze_batch_size

    # Execute Alpha Evolution Optimization
    print("\nInitializing Alpha Evolution Optimization...")
    ae_optimizer = AlphaEvolution(POPULATION_SIZE)
    best_params = ae_optimizer.evolve(
        model, train_lines, val_lines, input_shape, num_classes, VOCdevkit_path,
        Cuda, device, distributed, 0, ngpus_per_node, 0, epoch_steps
    )

    print("\nOptimization Process Completed!")
    print("Optimal Hyperparameter Configuration:")
    for k, v in best_params.items():
        print(f"{k:>20} : {v}")