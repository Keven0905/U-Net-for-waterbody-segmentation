import os
import torch
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
from tqdm import tqdm
from utils.utils import get_lr
from utils.utils_metrics import f_score


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period,
                  save_dir, local_rank=0):
    """
    Complete training and validation cycle for one epoch
    Args:
        model_train: Trainable model instance
        gen/val_gen: Training/Validation data loaders
        dice_loss: Flag for Dice loss incorporation
        focal_loss: Flag for Focal loss activation
        cls_weights: Class weighting for imbalanced datasets
        fp16: Mixed precision training flag
    """
    # Initialize metrics
    total_loss, total_f_score = 0, 0
    val_loss, val_f_score = 0, 0
    lambda_, beta_ = 1, 1  # Loss combination coefficients

    # Training Phase ----------------------------------------------------------
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: break

        # Data preparation
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:  # GPU data transfer
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()

        # Mixed precision context manager
        if not fp16:
            # Forward pass
            outputs = model_train(imgs)

            # Loss computation
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:  # Combine cross entropy with dice loss
                main_dice = Dice_loss(outputs, labels)
                loss = lambda_ * loss + beta_ * main_dice

            # Metric calculation
            with torch.no_grad():
                _f_score = f_score(outputs, labels)

            # Backpropagation
            loss.backward()
            optimizer.step()
        else:
            # Automatic Mixed Precision (AMP) context
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(imgs)
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss = lambda_ * loss + beta_ * main_dice

                with torch.no_grad():
                    _f_score = f_score(outputs, labels)

            # Scaled backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Metric aggregation
        total_loss += loss.item()
        total_f_score += _f_score.item()

        # Progress update
        if local_rank == 0:
            pbar.set_postfix(**{
                'total_loss': total_loss / (iteration + 1),
                'f_score': total_f_score / (iteration + 1),
                'lr': get_lr(optimizer)
            })
            pbar.update(1)

    # Validation Phase ---------------------------------------------------------
    if local_rank == 0:
        pbar.close()
        print('Finish Train\nStart Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val: break

        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            # Forward pass
            outputs = model_train(imgs)

            # Loss calculation
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = lambda_ * loss + beta_ * main_dice

            # Metric computation
            _f_score = f_score(outputs, labels)
            val_loss += loss.item()
            val_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{
                'val_loss': val_loss / (iteration + 1),
                'f_score': val_f_score / (iteration + 1),
                'lr': get_lr(optimizer)
            })
            pbar.update(1)

    # Post-Epoch Processing ----------------------------------------------------
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')

        # Update loss history and perform evaluation
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)

        # Performance summary
        print(f'Epoch:{epoch + 1}/{Epoch}')
        print(f'Total Loss: {total_loss / epoch_step:.3f} || Val Loss: {val_loss / epoch_step_val:.3f}')

        # Model checkpointing
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir,
                                                        f'ep{(epoch + 1):03d}-loss{total_loss / epoch_step:.3f}-val_loss{val_loss / epoch_step_val:.3f}.pth'))

        # Best model preservation
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        # Always save last epoch
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))


def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda, dice_loss,
                         focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0):
    """
    Training cycle without validation (for scenarios without validation set)
    Maintains same structure as fit_one_epoch but excludes validation steps
    """
    total_loss, total_f_score = 0, 0
    lambda_, beta_ = 1, 1

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.train()
    for iteration, batch in enumerate(gen):
    # Identical training logic as fit_one_epoch
    # [...]

    # Post-training processing (no validation)
    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, total_loss / epoch_step)
        print(f'Epoch:{epoch + 1}/{Epoch}\nTotal Loss: {total_loss / epoch_step:.3f}')

        # Model checkpointing (same logic without validation metrics)
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f'ep{(epoch + 1):03d}-loss{total_loss / epoch_step:.3f}.pth'))

        if len(loss_history.losses) <= 1 or (total_loss / epoch_step) <= min(loss_history.losses):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))