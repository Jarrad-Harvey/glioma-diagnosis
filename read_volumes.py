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
from settings import *

# Load a volume, either by user selection or a specified volume folder.
def load_volume(volume_folder=''):
    if not volume_folder:
        # Open a folder selection dialog box to choose a directory
        volume_folder = filedialog.askdirectory(title="Select Slice Directory", initialdir='.')

    # Get the slice filenames.
    slice_files = sorted(os.listdir(volume_folder))

    # Determine the volume name and label
    name = slice_files[0].split('_')[1]
    label = 0 # TODO: XXX

    # Search the directory and track the H5 files according to the filename conventions of the downloaded dataset.
    slices = np.empty(slices_per_volume, dtype=object)
    for slice_file in slice_files:
        # Read file
        file_path = os.path.join(volume_folder, slice_file)
        slice_id = int(slice_file.split('_')[3].split('.')[0])
        file = h5py.File(file_path, 'r')
        
        # Extract image
        images = file['image'][:] * 100
        channels = [Image.fromarray(images[:, :, i]) for i in range(len(channel_options))]
        
        # Extract mask
        masks = file['mask'][:]
        merged_mask = merge_mask(masks)

        # Append slice
        slices[slice_id] = {
            "image": channels,
            "mask": merged_mask,
        }

    volume = {
        "slices": slices,
        "name": name,
        "label": label,
    }
    return volume


def get_current_volume(channel_ID, volume):
    # Extract 3D image and mask for the specified channel
    image_3d = [slice["image"][channel_ID] for slice in volume["slices"]]
    mask_3d = [slice["mask"] for slice in volume["slices"]]
    
    # Convert lists to numpy arrays
    image_3d = np.array(image_3d)
    mask_3d = np.array(mask_3d)
    
    return image_3d, mask_3d

def merge_mask(masks):
    # Reshape masks array
    mask_array = [masks[:, :, i] for i in range(masks.shape[2])]

    # Merge non-overlapping masks by addition
    merged_Mask = mask_array[0] + mask_array[1] + mask_array[2]
    return merged_Mask