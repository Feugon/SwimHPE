import sys
import cv2
from ultralytics import YOLO

image_path = sys.argv[1] if len(sys.argv) > 1 else "examples/ex.jpeg"
model = YOLO('models/yolo26m-pose.pt')

results = model.predict(image_path, verbose=False)
vis = results[0].plot()

cv2.imshow('Prediction', vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
