import tkinter as tk
from GUI import ImageGUI
import logging
import argparse

if __name__ == "__main__":
    # Disable pyradiomics logging
    logger = logging.getLogger("radiomics.glcm")
    logger.setLevel(logging.ERROR)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dev', action="store_true", help="Enable development mode. Displays SVM and feature selection buttons.")
    args = parser.parse_args()
    dev_mode = args.dev

    # Run GUI
    root = tk.Tk()
    gui = ImageGUI(root, dev_mode)
    root.mainloop()