import os
import torch

# Environment fixes
os.environ['KMP_DUPLICATE_LIB_OK']='True'
torch.backends.nnpack.enabled = False

from ultralytics import YOLO

def train_and_save(epochs=2, device="cpu"):
    """
    This function now only trains the model and saves the final weights.
    """
    model = YOLO('models/yolo26n-pose.pt')
    
    # Train the model
    model.train(data="coco8-pose.yaml",
                epochs=epochs,
                imgsz=640,
                device=device,
                workers=0,
                plots=False,
                save=False, # Important: keep this false
                name="debug")

    print("\n✅ Training complete.")

    # Save the final weights using the stable PyTorch method
    final_weights_path = 'models/yolo26n-pose-final.pt'
    torch.save(model.state_dict(), final_weights_path)
    print(f"✅ Model weights saved to {final_weights_path}")
    return final_weights_path

if __name__ == '__main__':
    # This script will now run to completion without freezing.
    train_and_save(epochs=2, device="cpu")