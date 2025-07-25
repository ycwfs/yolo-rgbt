import warnings
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("/data1/wangqiurui/code/competition/tianchi/unseen/YOLOv11-RGBT/runs/rgb_ir/yolo11s-RGBT-midlate-fusion-finetuneM3DF/weights/best.pt") # select your model.pt path
    model.predict(source="/data1/wangqiurui/code/ossutil-v1.7.19-linux-amd64/rgb_ir/test/visible/images",
                  save=True, imgsz=[736,1280], conf=0.1, device="cuda:0", line_width=1, 
                  save_txt=True, save_conf=True,
                  project='/data1/wangqiurui/code/competition/tianchi/unseen/YOLOv11-RGBT/runs/rgb_ir/yolo11s-RGBT-midlate-fusion-finetuneM3DF', 
                  name='res_923',
                  use_simotm="RGBT",
                  channels=4,
                  # visualize=True # visualize model features maps
                )