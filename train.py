import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    # ---------------------------------#
    #   Cuda    Whether to use CUDA
    #           Set to False if no GPU available
    # ---------------------------------#
    Cuda = True
    # ----------------------------------------------#
    #   Seed    Fixed random seed for reproducibility
    #           Ensures consistent results across runs
    # ----------------------------------------------#
    seed = 11
    # ---------------------------------------------------------------------#
    #   distributed     Whether to use distributed training (multi-GPU)
    #                   CUDA_VISIBLE_DEVICES specifies GPU indices on Ubuntu
    #                   Windows only supports DP mode (automatically uses all GPUs)
    #   DP Mode:
    #       Set distributed = False
    #       Run command: CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP Mode:
    #       Set distributed = True
    #       Run command: CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    # ---------------------------------------------------------------------#
    distributed = False
    # ---------------------------------------------------------------------#
    #   sync_bn     Whether to use synchronized batch normalization (DDP mode only)
    # ---------------------------------------------------------------------#
    sync_bn = False
    # ---------------------------------------------------------------------#
    #   fp16        Whether to use mixed precision training
    #               Reduces VRAM usage by ~50%, requires PyTorch 1.7.1+
    # ---------------------------------------------------------------------#
    fp16 = True
    # -----------------------------------------------------#
    #   num_classes     Must modify for custom datasets
    #                   Number of classes + 1 (e.g., 2+1)
    # -----------------------------------------------------#
    num_classes = 2
    # -----------------------------------------------------#
    #   Backbone network selection
    #   Options: vgg, resnet50, densenet, mobilenet
    # -----------------------------------------------------#
    backbone = "resnet50"
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      Whether to use backbone pretrained weights (loaded during model initialization)
    #                   If model_path is specified, backbone weights are not loaded (pretrained becomes irrelevant)
    #                   If model_path='' and pretrained=True: load backbone weights only
    #                   If model_path='' and pretrained=False: train from scratch (not recommended)
    # ----------------------------------------------------------------------------------------------------------------------------#
    pretrained = False
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   model_path      Pretrained weights path (download links in README)
    #                   Pretrained weights are general across datasets due to feature reuse
    #                   Important: Backbone weights are critical for feature extraction
    #                   Set to empty string '' to disable pretrained weights
    #                   To resume training: specify path to checkpoint in logs folder
    # ----------------------------------------------------------------------------------------------------------------------------#
    model_path = "model_data/unet_resnet_voc.pth"  # "model_data/unet_resnet_voc.pth"
    # -----------------------------------------------------#
    #   input_shape     Input image size (must be multiple of 32)
    # -----------------------------------------------------#
    input_shape = [512, 512]

    # ----------------------------------------------------------------------------------------------------------------------------#
    #   Training phases: Freeze (initial) and Unfreeze (main)
    #   Freeze phase reduces VRAM usage for limited hardware
    #   Set Freeze_Epoch=UnFreeze_Epoch to only use freeze training
    #
    #   Configuration suggestions:
    #   (A) Full model pretraining:
    #       Adam: Init_lr=1e-4, Freeze_Epoch=50, UnFreeze_Epoch=100
    #       SGD: Init_lr=1e-2, Freeze_Epoch=50, UnFreeze_Epoch=100
    #   (B) Backbone-only pretraining:
    #       Adam: Init_lr=1e-4, Freeze_Epoch=50, UnFreeze_Epoch=100-300
    #       SGD: Init_lr=1e-2, Freeze_Epoch=50, UnFreeze_Epoch=120-300
    #   (C) Batch size: Maximize within GPU memory limits
    #       Freeze_batch_size = 1-2x Unfreeze_batch_size
    #       resnet50 requires batch_size > 1 due to BatchNorm
    # ----------------------------------------------------------------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Freeze Phase Training Parameters
    #   Freezes backbone for feature extraction network
    #   Init_Epoch       Starting epoch (can be >Freeze_Epoch for resuming)
    #   Freeze_Epoch     Number of freeze training epochs
    #   Freeze_batch_size Batch size during freeze phase
    # ------------------------------------------------------------------#
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 4
    # ------------------------------------------------------------------#
    #   Unfreeze Phase Parameters
    #   Unfreeze_Epoch      Total training epochs
    #   Unfreeze_batch_size Batch size after unfreezing
    # ------------------------------------------------------------------#
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 4
    # ------------------------------------------------------------------#
    #   Freeze_Train    Whether to perform freeze training first
    # ------------------------------------------------------------------#
    Freeze_Train = True

    # ------------------------------------------------------------------#
    #   Optimizer Parameters: Learning rate, scheduler, etc.
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Init_lr         Maximum learning rate
    #                   Adam: 1e-4, SGD: 1e-2 recommended
    #   Min_lr          Minimum learning rate (default: 0.01*Init_lr)
    # ------------------------------------------------------------------#
    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01
    # ------------------------------------------------------------------#
    #   optimizer_type  Optimizer type: adam or sgd
    #   momentum        Optimizer momentum parameter
    #   weight_decay    L2 regularization (set to 0 for Adam)
    # ------------------------------------------------------------------#
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    # ------------------------------------------------------------------#
    #   lr_decay_type   Learning rate decay: 'step' or 'cos'
    # ------------------------------------------------------------------#
    lr_decay_type = 'cos'
    # ------------------------------------------------------------------#
    #   save_period     Save checkpoint every N epochs
    # ------------------------------------------------------------------#
    save_period = 5
    # ------------------------------------------------------------------#
    #   save_dir        Directory to save checkpoints and logs
    # ------------------------------------------------------------------#
    save_dir = 'logs'
    # ------------------------------------------------------------------#
    #   eval_flag       Enable evaluation during training
    #   eval_period     Evaluate every N epochs (slows training)
    # ------------------------------------------------------------------#
    eval_flag = True
    eval_period = 5

    # ------------------------------#
    #   Dataset Path
    # ------------------------------#
    VOCdevkit_path = 'VOCdevkit'
    # ------------------------------------------------------------------#
    #   dice_loss      Recommended for small datasets or large batches
    # ------------------------------------------------------------------#
    dice_loss = True
    # ------------------------------------------------------------------#
    #   focal_loss     Mitigate class imbalance (often used with dice)
    # ------------------------------------------------------------------#
    focal_loss = True
    # ------------------------------------------------------------------#
    #   cls_weights    Class weights for loss balancing (numpy array)
    # ------------------------------------------------------------------#
    cls_weights = np.ones([num_classes], np.float32)
    # ------------------------------------------------------------------#
    #   num_workers    DataLoader threads (0 disables multiprocessing)
    # ------------------------------------------------------------------#
    num_workers = 4

    seed_everything(seed)
    # ------------------------------------------------------#
    #   GPU Configuration
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
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

    # ----------------------------------------------------#
    #   Download Pretrained Weights
    # ----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)
            dist.barrier()
        else:
            download_weights(backbone)

    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train()
    if not pretrained:
        weights_init(model)
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        # ------------------------------------------------------#
        #   Load weights matching model architecture
        # ------------------------------------------------------#
        model_dict = model.state_dict()
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
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nLoaded Keys:", len(load_key))
            print("\nFailed Keys:", str(no_load_key)[:500], "……\nFailed Keys:", len(no_load_key))
            print("\n\033[1;33;44mNote: Head layer mismatches are normal, backbone mismatches indicate errors.\033[0m")

    # ----------------------#
    #   Loss Tracking
    # ----------------------#
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # ------------------------------------------------------------------#
    #   Mixed Precision Training
    # ------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    # ----------------------------#
    #   Sync BatchNorm
    # ----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not supported in single GPU or non-distributed mode.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # ---------------------------#
    #   Dataset Preparation
    # ---------------------------#
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape, \
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type, \
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )
    # ------------------------------------------------------#
    #   Freeze Backbone for Initial Training
    # ------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            model.freeze_backbone()

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        # -------------------------------------------------------------------#
        #   Auto-adjust learning rate based on batch size
        # -------------------------------------------------------------------#
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
            raise ValueError("Dataset too small for training, please expand dataset.")

        train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
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
            eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                         eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None

        # ---------------------------------------#
        #   Main Training Loop
        # ---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs = 16
                lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                model.unfreeze_backbone()

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("Dataset too small for training, please expand dataset.")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=unet_dataset_collate, sampler=train_sampler,
                                 worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=unet_dataset_collate, sampler=val_sampler,
                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss,
                          cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()