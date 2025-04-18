import os
from PIL import Image
from tqdm import tqdm
from unet import Unet
from utils.utils_metrics import compute_mIoU, show_results

if __name__ == "__main__":
    # ---------------------------------------------------------------------------#
    #   miou_mode controls computation phases:
    #   0: Full pipeline (prediction + metric computation)
    #   1: Prediction generation only
    #   2: Metric computation from existing predictions
    # ---------------------------------------------------------------------------#
    miou_mode = 0
    # ------------------------------#
    #   Number of classes + background
    #   Format: (num_classes + 1)
    # ------------------------------#
    num_classes = 2
    # --------------------------------------------#
    #   Class labels must match dataset annotations
    #   Ordered as [background, class1, class2...]
    # --------------------------------------------#
    name_classes = ["_background_", "water"]
    # -------------------------------------------------------#
    #   Path to PASCAL VOC dataset directory
    #   Default assumes VOCdevkit in root directory
    # -------------------------------------------------------#
    VOCdevkit_path = 'VOCdevkit'

    # Load validation set indices
    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), 'r').read().splitlines()
    # Path configurations
    gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")  # Ground truth directory
    miou_out_path = "miou_out"  # Metric output directory
    pred_dir = os.path.join(miou_out_path, 'detection-results')  # Prediction storage

    # Phase 1: Prediction Generation
    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Initializing segmentation model...")
        unet = Unet()
        print("Model loaded successfully.")

        print("Generating predictions...")
        for image_id in tqdm(image_ids):
            # Load and process image
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
            image = Image.open(image_path)
            # Generate segmentation mask
            image = unet.get_miou_png(image)
            # Save prediction
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Prediction generation completed.")

    # Phase 2: Metric Computation
    if miou_mode == 0 or miou_mode == 2:
        print("Computing segmentation metrics...")
        # Returns: confusion matrix, IoUs, Pixel Accuracy Recall, Precision
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)
        print("Metric computation completed.")
        # Visualize and save results
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)