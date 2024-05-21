import os
import shutil
import re

# 
#       Place this file inside the data folder with the 369*155 .h5 slices and execute with `python sort_slices.py`.
#  

# Path to the directory containing the slice files
base_dir = './'

# Regex pattern to match the filenames
pattern = re.compile(r'volume_(\d+)_slice_(\d+)\.h5')

# Loop through all files in the base directory
for filename in os.listdir(base_dir):
    match = pattern.match(filename)
    if match:
        volume_number = match.group(1)
        # Create a subdirectory for the volume if it doesn't exist
        volume_dir = os.path.join(base_dir, f'volume_{volume_number}')
        if not os.path.exists(volume_dir):
            os.makedirs(volume_dir)
        # Move the file into the corresponding volume subdirectory
        shutil.move(os.path.join(base_dir, filename), os.path.join(volume_dir, filename))

print("Files sorted into subdirectories successfully.")
