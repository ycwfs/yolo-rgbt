import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('/data1/wangqiurui/code/competition/tianchi/unseen/YOLOv11-RGBT/ultralytics/cfg/models/11-RGBT/yolo11s-RGBT-midfusion.yaml')
    model = YOLO('/data1/wangqiurui/code/competition/tianchi/unseen/YOLOv11-RGBT/LLVIP-yolo11s-RGBT-midfusion-e300-16-.pt')
    model.info(True,True,imgsz=[720,1280])
    # model.load('/data1/wangqiurui/code/competition/tianchi/unseen/YOLOv11-RGBT/LLVIP-yolo11s-RGBT-midfusion-e300-16-pretrained.pt') # loading pretrain weights
    model.train(data='/data1/wangqiurui/code/competition/tianchi/unseen/YOLOv11-RGBT/ultralytics/cfg/datasets/unseen.yaml',
                cache=False,
                imgsz=1280,
                epochs=300,
                batch=12,
                close_mosaic=10,
                degrees=0,
                hsv_h=0.015,
                hsv_s=0.5,
                hsv_v=0.3, # lightness
                translate=0.0, # 边缘，部分物体 Translates the image horizontally and vertically by a fraction of the image size, aiding in learning to detect partially visible objects.
                flipud=0, # 上下翻转在这个场景不太适合 Flips the image upside down with the specified probability, increasing the data variability without affecting the object's characteristics.
                fliplr=0.8,
                # shear=2, # 图像四个角的变形 Shears the image by a specified angle, distorting the image while preserving the object's structure, useful for learning object orientation.
                scale=0,  # 0.9 ~ 1.1 Scales the image by a gain factor, simulating objects at different distances from the camera.
                mosaic=0.6, #	目标很多，拼接出来训练不一定好 这种增强虽然好，但是由于对裁剪拼接的数据进行了训练。它会破坏检测的完整性。也就是说，如果你的检测画面中存在目标的一小部分，它也会检测出来。有时候，可能我们并不想这样。拿检测汽车来说，如果你希望只检测出完整的汽车，那么mosaic这个开关要关掉。Combines four training images into one, simulating different scene compositions and object interactions. Highly effective for complex scene understanding.
                mixup=0,  # 混合两个图片，标签 S:0.05; M:0.15; L:0.15; X:0.2 Blends two images and their labels, creating a composite image. Enhances the model's ability to generalize by introducing label noise and visual variability.
                copy_paste=0.0,  # S:0.15; M:0.4; L:0.5; X:0.6
                workers=6,
                device='0',
                optimizer='SGD',  # using SGD
                save_period=20,
                # lr0=0.002,
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGBT",
                channels=4,
                project='runs/rgb_ir',
                name='yolo11n-RGBT-midfusion-finetuneLLVIP',
                )