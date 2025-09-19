from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt')  # load a pretrained model (recommended for training)
results = model.val(data="coco8.yaml", verbose=True)

print(results)  # print results to console