import sys
import argparse
from PyQt6.QtWidgets import QApplication
from gui_logic import VideoPlayer

def main():
    parser = argparse.ArgumentParser(description="SwimHPE GUI")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to YOLO pose model (.pt file)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    player = VideoPlayer(model_path=args.model)
    player.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()