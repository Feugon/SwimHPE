import cv2, time
from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt')  # tiny & fast
cap = cv2.VideoCapture(0)
cap.set(3, 640); cap.set(4, 360); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

t0, n = time.time(), 0
while True:
    ok, frame = cap.read()
    if not ok: break
    r = model.predict(frame, device='cpu', imgsz=640, verbose=False)
    vis = r[0].plot()  # <-- draws keypoints & skeleton
    n += 1
    if n % 30 == 0:
        print(f'Avg FPS: {n/(time.time()-t0):.1f}')
    cv2.imshow('YOLO-Pose', vis)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()