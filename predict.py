import time
import cv2
import numpy as np
from PIL import Image
from unet import Unet_ONNX, Unet

if __name__ == "__main__":
    # -------------------------------------------------------------------------#
    #   Modify self.colors in __init__ to change class color mappings
    # -------------------------------------------------------------------------#
    # ----------------------------------------------------------------------------------------------------------#
    #   Operation modes:
    #   'predict'         - Single image inference
    #   'video'           - Real-time video/webcam processing
    #   'fps'             - Computational performance benchmarking
    #   'dir_predict'     - Batch prediction on directory
    #   'export_onnx'     - Model conversion to ONNX format (requires PyTorch 1.7.1+)
    #   'predict_onnx'    - Inference using exported ONNX model (modify Unet_ONNX params)
    # ----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    # -------------------------------------------------------------------------#
    #   count            - Enable pixel-wise counting and ratio calculation
    #   name_classes     - Class labels matching dataset annotations
    #
    #   Only effective in 'predict' mode
    # -------------------------------------------------------------------------#
    count = False
    name_classes = ["background", "1", "2", "3", "4", "5", "6", "7"]
    # ----------------------------------------------------------------------------------------------------------#
    #   video_path       - Input video path (0 for webcam)
    #   video_save_path  - Output video path (empty string disables saving)
    #   video_fps        - Target frames per second for output
    #
    #   Video processing parameters (only for 'video' mode)
    #   Note: Full video save requires natural termination or completion
    # ----------------------------------------------------------------------------------------------------------#
    video_path = 0
    video_save_path = ""
    video_fps = 25.0
    # ----------------------------------------------------------------------------------------------------------#
    #   test_interval    - Number of iterations for FPS measurement
    #   fps_image_path   - Test image for benchmarking
    #
    #   FPS measurement parameters (only for 'fps' mode)
    # ----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path = "img/street.jpg"
    # -------------------------------------------------------------------------#
    #   dir_origin_path  - Input directory for batch processing
    #   dir_save_path    - Output directory for processed images
    #
    #   Directory processing parameters (only for 'dir_predict' mode)
    # -------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path = "F:/DeepLearning/模型预测图片/2025.1.15"
    # -------------------------------------------------------------------------#
    #   simplify         - Enable ONNX graph optimization
    #   onnx_save_path  - Output path for ONNX model
    # -------------------------------------------------------------------------#
    simplify = True
    onnx_save_path = "model_data/models.onnx"

    if mode != "predict_onnx":
        unet = Unet()
    else:
        yolo = Unet_ONNX()

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = unet.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Failed to initialize camera/video capture")

        fps = 0.0
        while (True):
            t1 = time.time()
            # Frame acquisition
            ref, frame = capture.read()
            if not ref:
                break
            # Color space conversion BGR→RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            frame = Image.fromarray(np.uint8(frame))
            # Perform segmentation
            frame = np.array(unet.detect_image(frame))
            # Convert back to BGR for display
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Calculate smoothed FPS
            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break
        print("Video processing completed")
        capture.release()
        if video_save_path != "":
            print("Output video saved to: " + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open('img/street.jpg')
        tact_time = unet.get_FPS(img, test_interval)
        print(f'Inference time: {tact_time:.4f}s, FPS: {1 / tact_time:.2f} (@batch_size=1)')

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = unet.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))

    elif mode == "export_onnx":
        unet.convert_to_onnx(simplify, onnx_save_path)

    elif mode == "predict_onnx":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()

    else:
        raise AssertionError("Invalid mode specified. Valid options: 'predict', 'video', 'fps', 'dir_predict'")