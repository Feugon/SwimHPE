import cv2
import numpy as np
try:
    # Try importing YOLO from ultralytics (YOLOv8/YOLOv11)
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")

class ModelInference:
    def __init__(self, model_path=None):
        self.model = None
        self.model_loaded = False
        self.model_path = model_path or "models/yolo26n-pose.pt"  # Default to YOLO26 nano pose model
        
    def load_model(self, model_path=None):
        """Load the pose estimation model"""
        if model_path:
            self.model_path = model_path
            
        try:
            if not YOLO_AVAILABLE:
                print("YOLO not available. Please install ultralytics: pip install ultralytics")
                return False
                
            print(f"Loading model: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.model_loaded = True
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
            return False
    
    def predict(self, frame):
        """Run pose estimation on a frame"""
        if not self.model_loaded or self.model is None:
            return None
            
        try:
            # Run inference, using half precision for speed
            results = self.model(frame, verbose=False, half=False)
            
            # Extract keypoints from the first detection (if any)
            if len(results) > 0 and results[0].keypoints is not None:
                keypoints_data = results[0].keypoints.data
                
                if len(keypoints_data) > 0:
                    # Get the first person's keypoints
                    keypoints = keypoints_data[0].cpu().numpy()  # Shape: (17, 3) for COCO format
                    
                    # Convert from normalized coordinates to pixel coordinates
                    height, width = frame.shape[:2]
                    pixel_keypoints = []
                    
                    for kp in keypoints:
                        x, y, conf = kp
                        # YOLO returns absolute pixel coordinates for keypoints
                        pixel_keypoints.append([x, y, conf])
                    
                    return pixel_keypoints
            
            return None
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None
    
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