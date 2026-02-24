# This file has been moved to GUI/run_gui.py
# Please run the GUI from the GUI directory:
# cd GUI
# python run_gui.py

import sys
import os

# Add GUI directory to path and run the GUI
gui_dir = os.path.join(os.path.dirname(__file__), 'GUI')
sys.path.insert(0, gui_dir)

from run_gui import main

if __name__ == "__main__":
    main()