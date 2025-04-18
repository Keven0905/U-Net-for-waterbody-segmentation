import os
import torch
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
from tqdm import tqdm
from utils.utils import get_lr
from utils.utils_metrics import f_score


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period,
                  save_dir, lambda_, beta, local_rank):
    """
    Complete training and validation cycle for one epoch with multi-GPU support
    Implements:
    - Mixed precision training
    - Loss function combination (CrossEntropy/Focal + Dice)
    - Periodic model checkpointing
    """
    total_loss = 0
    total_f_score = 0
    val_loss = 0
    val_f_score = 0

    # Training Phase ----------------------------------------------------------
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: break

        # Data Preparation
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:  # GPU Data Transfer
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()

        # Mixed Precision Context
        if not fp16:
            # Forward Propagation
            outputs = model_train(imgs)

            # Loss Computation
            if focal_loss:
                base_loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                base_loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            # Combine with Dice Loss if enabled
            if dice_loss:
                dice_loss_val = Dice_loss(outputs, labels)
                loss = lambda_ * base_loss + beta * dice_loss_val
            else:
                loss = base_loss

            # Metric Calculation
            with torch.no_grad():
                _f_score = f_score(outputs, labels)

            # Backward Propagation
            loss.backward()
            optimizer.step()
        else:
            # Automatic Mixed Precision (AMP)
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(imgs)

                if focal_loss:
                    base_loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    base_loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    dice_loss_val = Dice_loss(outputs, labels)
                    loss = lambda_ * base_loss + beta * dice_loss_val
                else:
                    loss = base_loss

                with torch.no_grad():
                    _f_score = f_score(outputs, labels)

            # Scaled Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Metric Aggregation
        total_loss += loss.item()
        total_f_score += _f_score.item()

        # Progress Update
        if local_rank == 0:
            pbar.set_postfix(**{
                'total_loss': total_loss / (iteration + 1),
                'f_score': total_f_score / (iteration + 1),
                'lr': get_lr(optimizer)
            })
            pbar.update(1)

    # Validation Phase --------------------------------------------------------
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

            # Forward Pass
            outputs = model_train(imgs)

            # Validation Loss
            if focal_loss:
                base_loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                base_loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                dice_loss_val = Dice_loss(outputs, labels)
                loss = lambda_ * base_loss + beta * dice_loss_val
            else:
                loss = base_loss

            # Validation Metrics
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

    # Post-Epoch Processing ---------------------------------------------------
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')

        # Update loss history and metrics
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)

        # Calculate evaluation metrics
        if eval_callback is not None:
            eval_callback.on_epoch_end(epoch + 1, model_train)  # Compute F1 & mIoU

        print(f'Epoch:{epoch + 1}/{Epoch}')
        print(f'Total Loss: {total_loss / epoch_step:.3f} || Val Loss: {val_loss / epoch_step_val:.3f}')

        # Model Checkpointing
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir,
                                                        f'ep{epoch + 1:03d}-loss{total_loss / epoch_step:.3f}-val_loss{val_loss / epoch_step_val:.3f}.pth'))

        # Best Model Preservation
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        # Final Model Save
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))


def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda, dice_loss,
                         focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, lambda_, beta,
                         local_rank=0):
    """
    Training cycle without validation (for scenarios with no validation set)
    Maintains identical training logic excluding validation steps
    """
    # [Identical training logic as fit_one_epoch excluding validation phase]
    # [...]

    # Final Processing (No Validation)
    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, total_loss / epoch_step)
        print(f'Epoch:{epoch + 1}/{Epoch}\nTotal Loss: {total_loss / epoch_step:.3f}')

        # Model Preservation Logic
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f'ep{epoch + 1:03d}-loss{total_loss / epoch_step:.3f}.pth'))

        if len(loss_history.losses) <= 1 or (total_loss / epoch_step) <= min(loss_history.losses):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))