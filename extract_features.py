from tkinter import filedialog
import numpy as np
import os
import csv
import cv2
from sklearn.decomposition import PCA
import h5py
from radiomics import featureextractor
import SimpleITK as sitk
from read_volumes import load_volume, merge_mask, get_current_volume
from calculate_best_features import calculate_repeatability, select_top_features, extract_result
from settings import * 
import pprint

def extract_radiomic_features(current_channel_ID=0):
        feature_lists = []
        volume_lists = []
        hadHeader = False
        folder_path = filedialog.askdirectory(title="Select Volume Set Directory", initialdir='.')
        # Ensure the volume folders are sorted
        volume_folders = sorted(os.listdir(folder_path))


        for volume_folder in volume_folders:
            # Load volume
            path = os.path.join(folder_path, volume_folder)
            volume = load_volume(path)
            volume_lists.append(volume["name"])

            results = []
            for channel_ID in range (0,4):
                image_3d, mask_3d = get_current_volume(channel_ID, volume)
                # Convert 3D numpy arrays to SimpleITK images
                sitk_volume = sitk.GetImageFromArray(image_3d)
                sitk_mask = sitk.GetImageFromArray(mask_3d)
                
                # Execute feature extraction on the volume and mask
                print("Extracting radiomic features for volume " + volume['name'] + " channel " + str(channel_ID) + "...")
                extractor = featureextractor.RadiomicsFeatureExtractor()
                result = extractor.execute(sitk_volume, sitk_mask)

                results.append(result)
                if (current_channel_ID == channel_ID):
                    feature_lists.append(result)

            calculate_repeatability(*results)

        # Determine the best features for high repeatability across all volume
        # top_features = select_top_features()
        top_features = hardcoded_top_features
        print("\nBest features are:")
        pprint.pp(top_features)

        with open(radiomic_output_file, 'w', newline='') as csvfile:
            i = 0
            for result_to_show in feature_lists:
                result_to_show = extract_result(result_to_show, top_features)
                if(hadHeader is not True):
                    fieldnames = ['Volume'] + list(result_to_show.keys())  # Add 'Value' as the first column
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    hadHeader = True
                volume_number = volume_lists[i]
                i+=1
                Volume_Number = 'Volume_' + volume_number
                writer.writerow({'Volume': Volume_Number, **result_to_show})  # Write the data row
        print("Features extracted and saved to:", radiomic_output_file)
        return

def extract_conventional_features():
    # Ask user for folder path
    folder_path = filedialog.askdirectory(title="Select Slice Directory", initialdir='.')

    # Create or overwrite the CSV file to store the results
    with open(conventional_output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Volume', 'Max Tumor Area', 'Max Tumor Diameter', 'Outer Layer Involvement'])

        # Ensure the volume folders are sorted
        volume_folders = sorted(os.listdir(folder_path))

        for volume_name in volume_folders:            
            volume_folder = os.path.join(folder_path, volume_name)

            max_tumor_area = 0
            max_tumor_diameter = 0
            total_voxels = 0
            total_cortex_voxels = 0

            # Get all slice files in the volume folder and sort them
            slice_files = sorted(os.listdir(volume_folder))

            name = slice_files[0].split('_')[1]
            print("Extracting conventional features for volume " + name + "...")

            for slice_file in slice_files:
                if slice_file.endswith('.h5'):
                    file_path = os.path.join(volume_folder, slice_file)
                    file = h5py.File(file_path, 'r')
                    datasetMask = file['mask'][:]
                    datasetImage = file['image'][:]
                    merged_mask = merge_mask(datasetMask)
                    tumor_area = np.sum(merged_mask)
                    max_tumor_area = max(max_tumor_area, tumor_area)

                    max_tumor_diameter_slice = calculate_max_tumor_diameter(merged_mask)
                    max_tumor_diameter = max(max_tumor_diameter, max_tumor_diameter_slice)

                    Outer_layer = outer_contours(datasetImage[:, :, 1])
                    glioma_overlap = cv2.bitwise_and(Outer_layer.astype(np.uint8), merged_mask)
                    voxel_count = np.count_nonzero(glioma_overlap == 1)
                    total_voxels += voxel_count

                    cortex_voxels = np.count_nonzero(Outer_layer)
                    total_cortex_voxels += cortex_voxels

            outer_layer_involvement_avg = (total_voxels / total_cortex_voxels) * 100

            # Extract volume number from the volume name (assuming volume name has the format 'volume_X')
            volume_number = slice_files[0].split('_')[1]
            Volume_Number = 'Volume_' + volume_number
            csv_writer.writerow([Volume_Number, max_tumor_area, max_tumor_diameter, outer_layer_involvement_avg])

    print("Conventional features extracted and saved to:", conventional_output_file)

def outer_contours(image):
    # Set variables for the width and height of the image, and the number of layers required for the combined contour
    image_w = image.shape[1]
    image_h = image.shape[0]
    outer_layers = 5
    i = 0

    # Convert the grayscale image of the brain into a binarized format
    _, image_th = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)

    # Convert the image to the correct datatype and create a blank image to store the created contours
    image_uint8 = image_th.astype(np.uint8)
    collected_contours = np.zeros((image_w, image_h))

    # Loop to calculate the contours of the outer contour and accumulate them
    while i < outer_layers:
        contours, _ = cv2.findContours(image_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            i+= 1
            continue

        # Call the generate_Outer_Layers function to return an image of the contour
        contour_image = generate_Outer_Layer(contours, image_w, image_h)

        # Add the image of the contour to the accumulated image
        collected_contours += contour_image
        # Shrink image by 1 pixel
        image_uint8 =  shrink_image(image_uint8) 
        i+= 1

    # Return the combined image of the contours
    return collected_contours

def calculate_max_tumor_diameter(merged_mask):
    max_tumor_diameter = 0
    # Find coordinates of tumor pixels
    tumor_pixels = np.argwhere(merged_mask > 0)
    if len(tumor_pixels) > 1:
        # Perform PCA to find the longest linear measurement
        pca = PCA(n_components=2)
        pca.fit(tumor_pixels)
        eigen_vectors = pca.components_
        # Project tumor pixels onto the first principal component
        projected_tumor_pixels = np.dot(tumor_pixels - pca.mean_, eigen_vectors[0])
        # Calculate the longest linear measurement along the principal component
        max_tumor_diameter = np.max(projected_tumor_pixels) - np.min(projected_tumor_pixels)
        #max_tumor_diameter = max(max_tumor_diameter, tumor_diameter)
    return max_tumor_diameter

def generate_Outer_Layer(contours, image_w, image_h):
    # Create a blank image to store the contour
    Outer_Layer = np.zeros((image_w, image_h), dtype=np.uint8)

    # Draw each contour onto the blank image
    cv2.drawContours(Outer_Layer, contours, -1, 255, thickness=1)

    # Return the image of the contour
    return Outer_Layer

def shrink_image(image):
    # Define a kernel for erosion
    kernel = np.ones((2,2), np.uint8)
    # Erode the image
    eroded_image = cv2.erode(image, kernel, iterations=1)
    # Convert the eroded image back to np.uint8
    eroded_image = eroded_image.astype(np.uint8)
    return eroded_image