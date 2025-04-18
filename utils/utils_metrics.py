import csv
import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def f_score(inputs, target, beta=1, smooth=1e-5, threhold=0.5):
    """
    Computes F-beta score for semantic segmentation evaluation
    Args:
        inputs: Model outputs (before softmax) with shape [N, C, H, W]
        target: Ground truth labels in one-hot format [N, H, W, C+1]
        beta: Weighting factor between precision and recall (β=1 → F1-score)
    """
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()

    # Handle resolution mismatch with bilinear interpolation
    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # Reshape for pixel-wise comparison
    temp_inputs = torch.softmax(inputs.permute(0, 2, 3, 1).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # Threshold predictions and calculate confusion matrix components
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, dim=[0, 1])  # True positives
    fp = torch.sum(temp_inputs, dim=[0, 1]) - tp  # False positives
    fn = torch.sum(temp_target[..., :-1], dim=[0, 1]) - tp  # False negatives

    # F-beta score calculation with smoothing factor
    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    return torch.mean(score)


def fast_hist(a, b, n):
    """
    Computes confusion matrix for two 1D label arrays
    Args:
        a: Flattened ground truth array [H*W]
        b: Flattened prediction array [H*W]
        n: Number of semantic classes
    Returns:
        confusion_matrix: [n, n] matrix where diagonal represents correct predictions
    """
    valid_mask = (a >= 0) & (a < n)
    return np.bincount(n * a[valid_mask].astype(int) + b[valid_mask],
                       minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    """Calculates Intersection over Union (IoU) for each class"""
    return np.diag(hist) / np.maximum(hist.sum(1) + hist.sum(0) - np.diag(hist), 1)


def per_class_PA_Recall(hist):
    """Computes Recall (Pixel Accuracy) per class"""
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def per_class_Precision(hist):
    """Calculates Precision per class"""
    return np.diag(hist) / np.maximum(hist.sum(0), 1)


def per_Accuracy(hist):
    """Overall pixel accuracy"""
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)


def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None):
    """
    Computes mean Intersection-over-Union (mIoU) metrics
    Args:
        gt_dir: Directory containing ground truth segmentation masks
        pred_dir: Directory containing model predictions
        png_name_list: List of sample identifiers without extensions
    Returns:
        hist: Full confusion matrix
        IoUs: Per-class IoU values
        PA_Recall: Per-class Recall
        Precision: Per-class Precision
    """
    print('Num classes', num_classes)
    hist = np.zeros((num_classes, num_classes))  # Initialize confusion matrix

    # Prepare file paths
    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]

    for ind in range(len(gt_imgs)):
        # Load prediction and ground truth
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))

        # Skip mismatched resolutions
        if label.size != pred.size:
            print(f'Skipping: len(gt) = {len(label.flatten())}, len(pred) = {len(pred.flatten())}, '
                  f'{gt_imgs[ind]}, {pred_imgs[ind]}')
            continue

        # Update confusion matrix
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

        # Periodic progress reporting
        if name_classes and ind > 0 and ind % 10 == 0:
            print(f'{ind}/{len(gt_imgs)}: mIoU-{100 * np.nanmean(per_class_iu(hist)):.2f}%; '
                  f'mPA-{100 * np.nanmean(per_class_PA_Recall(hist)):.2f}%; '
                  f'Accuracy-{100 * per_Accuracy(hist):.2f}%')

    # Calculate final metrics
    IoUs = per_class_iu(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)

    # Print per-class metrics
    if name_classes:
        for idx in range(num_classes):
            print(f'===>{name_classes[idx]}:\tIoU-{IoUs[idx] * 100:.2f}; '
                  f'Recall-{PA_Recall[idx] * 100:.2f}; Precision-{Precision[idx] * 100:.2f}')

    # Print summary statistics
    print(f'===> mIoU: {np.nanmean(IoUs) * 100:.2f}; mPA: {np.nanmean(PA_Recall) * 100:.2f}; '
          f'Accuracy: {per_Accuracy(hist) * 100:.2f}')
    return hist.astype(np.int64), IoUs, PA_Recall, Precision


def adjust_axes(r, t, fig, axes):
    """Automatically adjust plot axes to accommodate text labels"""
    text_width = t.get_window_extent(renderer=r).width / fig.dpi
    current_width = fig.get_figwidth()
    axes.set_xlim([axes.get_xlim()[0], axes.get_xlim()[1] * (current_width + text_width) / current_width])


def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size=12, plt_show=True):
    """Generates horizontal bar plots for metric visualization"""
    fig, axes = plt.subplots()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)

    # Add value labels
    for i, val in enumerate(values):
        label = f'{val:.2f}' if val < 1.0 else str(val)
        t = plt.text(val, i, f' {label}', color='royalblue', va='center', fontweight='bold')
        if i == len(values) - 1:
            adjust_axes(fig.canvas.get_renderer(), t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show: plt.show()
    plt.close()


def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size=12):
    """Visualizes and saves evaluation metrics"""
    # Generate metric plots
    metrics = [
        (IoUs, "mIoU", "Intersection over Union"),
        (PA_Recall, "mPA", "Pixel Accuracy"),
        (PA_Recall, "mRecall", "Recall"),
        (Precision, "mPrecision", "Precision")
    ]

    for data, metric, label in metrics:
        path = os.path.join(miou_out_path, f"{metric}.png")
        draw_plot_func(data, name_classes, f"{metric} = {np.nanmean(data) * 100:.2f}%",
                       label, path, tick_font_size)
        print(f"Save {metric} to {path}")

    # Export confusion matrix
    csv_path = os.path.join(miou_out_path, "confusion_matrix.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([' '] + name_classes)
        for i, row in enumerate(hist):
            writer.writerow([name_classes[i]] + list(row))
    print(f"Save confusion matrix to {csv_path}")