import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import os
import csv
import cv2
from sklearn.decomposition import PCA
from skimage import color
from skimage.filters import threshold_otsu
import imageio
import h5py
from radiomics import featureextractor
import SimpleITK as sitk
from GUI import ImageGUI
import logging

if __name__ == "__main__":
    # Disable pyradiomics logging
    logger = logging.getLogger("radiomics.glcm")
    logger.setLevel(logging.ERROR)

    root = tk.Tk()
    gui = ImageGUI(root)
    root.mainloop()