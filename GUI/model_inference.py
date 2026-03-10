import cv2

try:
    # Try importing YOLO from ultralytics (YOLOv8/YOLOv11)
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")

_COCO17_FLIP_PAIRS = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]

class ModelInference:
    def __init__(self, model_path=None):
        self.model = None
        self.model_loaded = False
        self.model_path = model_path or "models/yolo26n-pose.pt"  # Default to YOLO26 nano pose model
        self._mlpackage_diag_done = False

    def load_model(self, model_path=None):
        """Load the pose estimation model"""
        if model_path:
            self.model_path = model_path

        try:
            if not YOLO_AVAILABLE:
                print("YOLO not available. Please install ultralytics: pip install ultralytics")
                return False

            print(f"Loading model: {self.model_path}")
            if self.model_path.endswith('.mlpackage'):
                self.model = YOLO(self.model_path, task='pose')
            else:
                self.model = YOLO(self.model_path)
            self.model_loaded = True
            self._mlpackage_diag_done = False
            print("Model loaded successfully!")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
            return False

    def _infer(self, frame):
        """Raw inference — returns list of persons."""
        kwargs = dict(verbose=False, imgsz=640)
        if not self.model_path.endswith('.mlpackage'):
            kwargs['half'] = False
        results = self.model(frame, **kwargs)
        if self.model_path.endswith('.mlpackage') and not self._mlpackage_diag_done:
            self._mlpackage_diag_done = True
            r = results[0]
            n_boxes = len(r.boxes) if r.boxes is not None else 0
            kp_shape = tuple(r.keypoints.data.shape) if r.keypoints is not None else None
            print(f"[mlpackage diag] boxes={n_boxes}, keypoints.data={kp_shape}")
        if len(results) > 0 and results[0].keypoints is not None:
            all_persons = []
            for person_kps in results[0].keypoints.data:
                kps = person_kps.cpu().numpy()
                all_persons.append([[float(x), float(y), float(c)] for x, y, c in kps])
            return all_persons
        return []

    def _merge_tta(self, persons_orig, persons_flip):
        """Match persons by center proximity, average their keypoints."""
        if not persons_orig:
            return persons_flip
        if not persons_flip:
            return persons_orig

        def center(person):
            visible = [kp for kp in person if kp[2] > 0]
            if not visible:
                return (0.0, 0.0)
            return (sum(kp[0] for kp in visible) / len(visible),
                    sum(kp[1] for kp in visible) / len(visible))

        merged = []
        used = set()
        for p_orig in persons_orig:
            cx, cy = center(p_orig)
            best_idx, best_dist = -1, float('inf')
            for i, p_flip in enumerate(persons_flip):
                if i in used:
                    continue
                fx, fy = center(p_flip)
                d = ((cx - fx)**2 + (cy - fy)**2)**0.5
                if d < best_dist:
                    best_dist, best_idx = d, i
            if best_idx >= 0:
                used.add(best_idx)
                p_flip = persons_flip[best_idx]
                merged.append([
                    [(kp_o[0]+kp_f[0])/2, (kp_o[1]+kp_f[1])/2, (kp_o[2]+kp_f[2])/2]
                    for kp_o, kp_f in zip(p_orig, p_flip)
                ])
            else:
                merged.append(p_orig)
        return merged

    def predict(self, frame, use_tta=False):
        """Run pose estimation on a frame.

        Returns a list of persons, each a list of 17 keypoints [x, y, conf].
        Returns an empty list if no detections.
        """
        if not self.model_loaded or self.model is None:
            return []

        try:
            persons_orig = self._infer(frame)
            if not use_tta:
                return persons_orig

            h, w = frame.shape[:2]
            flipped = cv2.flip(frame, 1)
            persons_flip = self._infer(flipped)

            # Unflip x-coords and swap left<->right keypoint pairs
            for person in persons_flip:
                for kp in person:
                    kp[0] = w - kp[0]
                for a, b in _COCO17_FLIP_PAIRS:
                    person[a], person[b] = person[b], person[a]

            return self._merge_tta(persons_orig, persons_flip)

        except Exception as e:
            print(f"Error during prediction: {e}")
            return []

    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model_loaded and self.model is not None:
            return {
                "model_path": self.model_path,
                "model_type": "YOLO Pose Estimation",
                "loaded": True
            }
        else:
            return {
                "model_path": self.model_path,
                "model_type": "Not loaded",
                "loaded": False
            }

    def set_model_path(self, path):
        """Set a custom model path"""
        self.model_path = path
        # Reload model with new path
        if self.model_loaded:
            self.load_model()
