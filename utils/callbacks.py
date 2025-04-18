import os
import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal

import cv2
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import cvtColor, preprocess_input, resize_image
from .utils_metrics import compute_mIoU, per_Accuracy, f_score


class LossHistory():
    def __init__(self, log_dir, model, input_shape, val_loss_flag=True):
        self.log_dir = log_dir
        self.val_loss_flag = val_loss_flag

        self.losses = []  # Training loss storage
        if self.val_loss_flag:
            self.val_loss = []  # Validation loss storage

        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        try:
            # Model graph visualization with dummy input
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss=None):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)  # Record training loss
        if self.val_loss_flag:
            self.val_loss.append(val_loss)  # Record validation loss

        # Persist losses to text files
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss) + "\n")
        if self.val_loss_flag:
            with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
                f.write(str(val_loss) + "\n")

        # TensorBoard logging
        self.writer.add_scalar('loss', loss, epoch)
        if self.val_loss_flag:
            self.writer.add_scalar('val_loss', val_loss, epoch)

        # Visualization of loss curves
        self.plot_train_loss()  # Training loss plot
        if self.val_loss_flag:
            self.plot_val_loss()  # Validation loss plot

    def plot_train_loss(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')

        # Apply Savitzky-Golay smoothing filter
        try:
            window_size = 5 if len(self.losses) < 25 else 15
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, window_size, 3),
                     'green', linestyle='--', linewidth=2, label='smoothed train loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.title('Training Loss Dynamics')
        plt.savefig(os.path.join(self.log_dir, "train_loss.png"))
        plt.cla()
        plt.close("all")

    def plot_val_loss(self):
        if not self.val_loss_flag:
            return

        iters = range(len(self.val_loss))

        plt.figure()
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')

        # Apply Savitzky-Golay smoothing filter
        try:
            window_size = 5 if len(self.val_loss) < 25 else 15
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, window_size, 3),
                     '#8B4513', linestyle='--', linewidth=2, label='smoothed val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.title('Validation Loss Dynamics')
        plt.savefig(os.path.join(self.log_dir, "val_loss.png"))
        plt.cla()
        plt.close("all")


class EvalCallback():
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda,
                 miou_out_path=".temp_miou_out", eval_flag=True, period=1):
        super(EvalCallback, self).__init__()
        # Network configuration parameters
        self.net = net
        self.input_shape = input_shape  # Model input dimensions [H,W]
        self.num_classes = num_classes  # Number of semantic classes
        self.image_ids = image_ids  # Dataset sample identifiers
        self.dataset_path = dataset_path  # PASCAL VOC directory path
        self.log_dir = log_dir  # Metric output directory
        self.cuda = cuda  # GPU acceleration flag
        self.miou_out_path = miou_out_path  # Temporary prediction storage
        self.eval_flag = eval_flag  # Evaluation enable flag
        self.period = period  # Evaluation frequency (epochs)

        # Metric tracking containers
        self.image_ids = [image_id.split()[0] for image_id in image_ids]
        self.mious = [0]  # Mean Intersection-over-Union
        self.mPAs = [0]  # Mean Pixel Accuracy
        self.Accuracy = [0]  # Global Accuracy
        self.mPrecisions = [0]  # Mean Precision
        self.epoches = [0]  # Epoch indices
        self.f1_scores = [0]  # F1-Scores (harmonic mean)

        # Metric file initialization
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write("0\n")

    def get_miou_png(self, image):
        """
        Generate segmentation mask for mIoU calculation
        Args:
            image: PIL Image object
        Returns:
            Image: Segmentation mask with class indices
        """
        # Convert to RGB to prevent channel mismatch
        image = cvtColor(image)
        orig_h, orig_w = np.array(image).shape[:2]

        # Resize with aspect ratio preservation
        image_data, new_w, new_h = resize_image(image, (self.input_shape[1], self.input_shape[0]))

        # Add batch dimension and normalize
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # Network forward pass
            pr = self.net(images)[0]
            # Class probability distribution
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            # Remove padding artifacts
            pr = pr[int((self.input_shape[0] - new_h) // 2): int((self.input_shape[0] - new_h) // 2 + new_h),
                 int((self.input_shape[1] - new_w) // 2): int((self.input_shape[1] - new_w) // 2 + new_w)]
            # Resize to original dimensions
            pr = cv2.resize(pr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            # Argmax for class indices
            pr = pr.argmax(axis=-1)

        return Image.fromarray(np.uint8(pr))

    def plot_metrics(self, metric_values, metric_name, y_label, file_name):
        """
        Visualize metric progression
        Args:
            metric_values: List of metric values
            metric_name: Name for legend
            y_label: Y-axis label
            file_name: Output filename
        """
        plt.figure()
        plt.plot(self.epoches, metric_values, 'red', linewidth=2, label=f'{metric_name}')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel(y_label)
        plt.title(f'{metric_name} Progression')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, file_name))
        plt.cla()
        plt.close("all")

    def on_epoch_end(self, epoch, model_eval):
        """Periodic evaluation at epoch end"""
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            gt_dir = os.path.join(self.dataset_path, "VOC2007/SegmentationClass/")
            pred_dir = os.path.join(self.miou_out_path, 'detection-results')

            # Directory initialization
            os.makedirs(self.miou_out_path, exist_ok=True)
            os.makedirs(pred_dir, exist_ok=True)

            print("Generating segmentation predictions...")
            for image_id in tqdm(self.image_ids):
                # Load and process image
                image_path = os.path.join(self.dataset_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
                image = Image.open(image_path)
                # Generate and save prediction
                self.get_miou_png(image).save(os.path.join(pred_dir, image_id + ".png"))

            print("Computing segmentation metrics...")
            # Calculate evaluation metrics
            hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir,
                                                            self.image_ids, self.num_classes, None)

            # Metric calculations
            temp_miou = np.nanmean(IoUs) * 100
            temp_mPA = np.nanmean(PA_Recall) * 100
            temp_Accuracy = np.nanmean(per_Accuracy(hist)) * 100
            temp_mPrecision = np.nanmean(Precision) * 100

            # Class-wise F1 computation
            f1_scores = []
            for i in range(self.num_classes):
                prec = Precision[i]
                rec = PA_Recall[i]
                f1 = 2 * (prec * rec) / (prec + rec + 1e-7) if (prec + rec) > 0 else 0
                f1_scores.append(f1)
            temp_f1_score = np.nanmean(f1_scores)  # Macro-average F1

            # Update metric records
            self.mious.append(temp_miou)
            self.mPAs.append(temp_mPA)
            self.Accuracy.append(temp_Accuracy)
            self.mPrecisions.append(temp_mPrecision)
            self.f1_scores.append(temp_f1_score)
            self.epoches.append(epoch)

            # Persist metrics to files
            metric_files = {
                "epoch_miou.txt": temp_miou,
                "epoch_mPA.txt": temp_mPA,
                "epoch_Accuracy.txt": temp_Accuracy,
                "epoch_mPrecision.txt": temp_mPrecision,
                "epoch_f1score.txt": temp_f1_score
            }
            for fname, value in metric_files.items():
                with open(os.path.join(self.log_dir, fname), 'a') as f:
                    f.write(f"{value}\n")

            # Visualize metric trajectories
            self.plot_metrics(self.mious, "mIoU", "mIoU (%)", "epoch_miou.png")
            self.plot_metrics(self.mPAs, "mPA", "Mean PA (%)", "epoch_mPA.png")
            self.plot_metrics(self.Accuracy, "Accuracy", "Accuracy (%)", "epoch_Accuracy.png")
            self.plot_metrics(self.mPrecisions, "Precision", "Precision (%)", "epoch_mPrecision.png")
            self.plot_metrics(self.f1_scores, "F1-Score", "F1-Score (%)", "epoch_f1score.png")

            print("Evaluation completed.")
            shutil.rmtree(self.miou_out_path)  # Clean temporary files