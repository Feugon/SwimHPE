import cv2, time, collections
from ultralytics import YOLO

model = YOLO('models/yolo26n-pose.mlpackage')
cap = cv2.VideoCapture(0)
cap.set(3, 640); cap.set(4, 360); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame_times = collections.deque(maxlen=30)
prev_time = time.time()
while True:
    ok, frame = cap.read()
    if not ok: break
    r = model.predict(frame, imgsz=640, verbose=False)
    vis = r[0].plot()

    curr_time = time.time()
    frame_times.append(curr_time - prev_time)
    prev_time = curr_time
    avg_fps = len(frame_times) / sum(frame_times)
    cv2.putText(vis, f'FPS: {avg_fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('YOLO-Pose', vis)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # q or Esc
        break

cap.release()
cv2.destroyAllWindows()
