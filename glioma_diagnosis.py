import tkinter as tk
from GUI import ImageGUI
import logging

if __name__ == "__main__":
    # Disable pyradiomics logging
    logger = logging.getLogger("radiomics.glcm")
    logger.setLevel(logging.ERROR)

    root = tk.Tk()
    gui = ImageGUI(root)
    root.mainloop()