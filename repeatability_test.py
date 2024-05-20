import os
import pprint
from tkinter import filedialog
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from read_volumes import get_current_volume, load_volume
from settings import * 

scores = {}

def calculate_repeatability(*results):
    # Filter features in all results
    filtered_results = []
    for result in results:
        filtered_result = filter_features(result)
        filtered_results.append(filtered_result)
    # collect all keys
    keys = filtered_results[0].keys()
    
    for key in keys:
        values = []
        for result in filtered_results:
            if key in result:
                values.append(result[key])
        
        if len(values) < len(results): 
            continue
        mean_val = np.mean(values)
        std_dev = np.std(values)
        cv = std_dev / mean_val if mean_val != 0 else np.inf
        if key not in scores:
            scores[key] = 0
        scores[key] += cv
    return


def filter_features(result):
    # Extract keys from the result dictionary
    keys = result.keys()
    
    # Filter keys that contain any of the specified substrings
    filtered_keys = []
    for key in keys:
        if any(substring in key for substring in substrings):
            filtered_keys.append(key)
    
    # Create a new dictionary containing only the filtered keys and their corresponding values
    filtered_result = {}
    for key in filtered_keys:
        filtered_result[key] = result[key]
    
    return filtered_result


def select_top_features():
    # Get top scoring features in each category
    top_intensity_features = select_top_features_in_category('firstorder', 10)
    top_shape_features = select_top_features_in_category('shape', 10)
    top_texture_features = select_top_features_in_category('glcm', 3) + \
                            select_top_features_in_category('gldm', 3) + \
                            select_top_features_in_category('glrlm', 2) + \
                            select_top_features_in_category('glszm', 2)
    
    return {
        'intensity_features': top_intensity_features,
        'shape_features': top_shape_features,
        'texture_features': top_texture_features
    }
        

def select_top_features_in_category(category, top_n):
    # Filter repeatability scores for the specified category
    filtered_scores = {}
    for key, val in scores.items():
        if category in key:
            filtered_scores[key] = val
    
    # Sort the filtered scores by their repeatability values
    sorted_features = sorted(filtered_scores.items(), key=lambda item: item[1])
    
    # Extract top N features from the sorted list
    top_features = [feature for feature, _ in sorted_features[:top_n]]
    
    return top_features


def extract_result(result, top_features):
    # Merge top_features values into a single list
    keys = [item for sublist in top_features.values() for item in sublist]
    
    # Filter result to only contain keys in keys
    top_feature = {key: result[key] for key in result.keys() if key in keys}
    
    return top_feature

def perform_repeatability_test():
    folder_path = filedialog.askdirectory(title="Select Volume Set Directory", initialdir='.')
    # Ensure the volume folders are sorted
    volume_folders = sorted(os.listdir(folder_path))

    for volume_folder in volume_folders:
        # Load volume
        path = os.path.join(folder_path, volume_folder)
        volume = load_volume(path)

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

        calculate_repeatability(*results)

    # Determine the best features for high repeatability across all volume
    top_features = select_top_features()
    
    print("\nBest features are:")
    pprint.pp(top_features)