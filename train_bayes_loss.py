import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna

from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit_bayes import fit_one_epoch


def train_and_validate(lambda_, beta, num_epochs, model, model_train, loss_history, eval_callback, optimizer,
                       epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights,
                       num_classes, fp16, scaler, save_period, save_dir, local_rank):
    best_val_loss = float('inf')
    best_f1_score = 0.0  # Track best F1-score
    best_miou = 0.0  # Track best mean Intersection-over-Union
    best_alpha = 0.0  # Track best composite metric (F1 & mIoU)

    for epoch in range(num_epochs):
        # Adjust learning rate according to schedule
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        # Execute one training epoch
        fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                      epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss,
                      cls_weights, num_classes, fp16, scaler, save_period, save_dir, lambda_, beta, local_rank)

        # Extract performance metrics
        current_val_loss = loss_history.val_loss[-1]
        current_f1_score = eval_callback.f1_scores[-1]
        current_miou = eval_callback.mious[-1]

        # Calculate composite metric (equal weights for F1 and mIoU)
        current_alpha = 0.5 * current_f1_score + 0.5 * current_miou

        # Update best performing metrics
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
        if current_f1_score > best_f1_score:
            best_f1_score = current_f1_score
        if current_miou > best_miou:
            best_miou = current_miou
        if current_alpha > best_alpha:
            best_alpha = current_alpha

    # Optimization target
    return best_alpha


def objective(trial, Epoch):
    """Objective function for Bayesian optimization with Optuna"""
    # Hyperparameter sampling
    lambda_ = trial.suggest_float('lambda', 0.0, 1.0)
    beta = 1.0 - lambda_    # Complementary parameter

    # Execute training with current hyperparameters
    best_alpha = train_and_validate(lambda_, beta, num_epochs=10, model=model, model_train=model_train,
                                    loss_history=loss_history,
                                    eval_callback=eval_callback, optimizer=optimizer, epoch_step=epoch_step,
                                    epoch_step_val=epoch_step_val,
                                    gen=gen, gen_val=gen_val, Epoch=Epoch, cuda=Cuda, dice_loss=dice_loss,
                                    focal_loss=focal_loss,
                                    cls_weights=cls_weights, num_classes=num_classes, fp16=fp16, scaler=scaler,
                                    save_period=save_period,
                                    save_dir=save_dir, local_rank=local_rank)

    # return alpha
    return best_alpha


if __name__ == "__main__":
    # Hardware Configuration
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
    UnFreeze_Epoch = 200
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

    # Distributed Training Setup
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)
            dist.barrier()
        else:
            download_weights(backbone)

    # Model Initialization
    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train()
    if not pretrained:
        weights_init(model)
    # Load Pretrained Weights with Compatibility Check
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        model_dict = model.state_dict()
        # Parameter filtering for model compatibility
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape,
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train,
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type,
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )

    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            model.freeze_backbone()

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs = 16
        lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                             weight_decay=weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small to continue training, please expand the dataset.")

        train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=unet_dataset_collate, sampler=train_sampler,
                         worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=unet_dataset_collate, sampler=val_sampler,
                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        if local_rank == 0:
            eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda,
                                         eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None

            # Bayesian optimization with Optuna to maximize alpha
        if local_rank == 0:
            study = optuna.create_study(direction='maximize')  # maximize alpha
            study.optimize(partial(objective, Epoch=UnFreeze_Epoch), n_trials=10)  # Run 10 trials

            # print the best lambda and beta
            best_trial = study.best_trial
            best_lambda = best_trial.params['lambda']
            best_beta = 1.0 - best_lambda
            print(f"Best trial - lambda: {best_lambda}, beta: {best_beta}")
            print(f"Best alpha: {best_trial.value}")  # print the best alpha value

            # Complete training with the best lambda and beta
        train_and_validate(best_lambda, best_beta, num_epochs=UnFreeze_Epoch, model=model, model_train=model_train,
                           loss_history=loss_history,
                           eval_callback=eval_callback, optimizer=optimizer, epoch_step=epoch_step,
                           epoch_step_val=epoch_step_val,
                           gen=gen, gen_val=gen_val, Epoch=UnFreeze_Epoch, cuda=Cuda, dice_loss=dice_loss,
                           focal_loss=focal_loss,
                           cls_weights=cls_weights, num_classes=num_classes, fp16=fp16, scaler=scaler,
                           save_period=save_period,
                           save_dir=save_dir, local_rank=local_rank)

        if local_rank == 0:
            loss_history.writer.close()