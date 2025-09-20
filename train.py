# https://docs.ultralytics.com/modes/train/#train-settings
from ultralytics import YOLO

def train():
    model = YOLO('yolo11n-pose.pt')  # load a pretrained model (recommended for training)
    results = model.train(data="swimXYZ.yaml", epochs=1, imgsz=640, batch=16)

    print(results)  # print results to console

train()