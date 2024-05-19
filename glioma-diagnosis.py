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



class ImageGUI:

    # Change detection method
    def onChanges(self, value):
        self.update_image()
   
    def __init__(self, master):
        self.master = master
        self.repeatability_scores = {}
        self.master.title("Glioma Diagnosis")
        self.substrings = ['shape', 'firstorder', 'glcm', 'gldm', 'glrlm', 'glszm']
        ##### Frames
        # Create a frame for the GUI.
        self.frame = tk.Frame(self.master)
        self.frame.pack(expand=True, padx=10, pady=10)
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        # Create subframes. (Image & 2 rows of options.)
        self.imageGrid = tk.Frame(self.frame)
        self.imageGrid.grid(row=0, column=0)
        self.options_row_1 = tk.Frame(self.frame, relief="groove")
        self.options_row_1.grid(row=1, column=0, sticky="nsew")
        self.options_row_2 = tk.Frame(self.frame, relief="groove")
        self.options_row_2.grid(row=2, column=0, sticky="nsew")
        self.options_row_3 = tk.Frame(self.frame, relief="groove")
        self.options_row_3.grid(row=3, column=0, sticky="nsew")
        self.options_row_4 = tk.Frame(self.frame, relief="groove")
        self.options_row_4.grid(row=4, column=0, sticky="nsew")
        
        ##### Image grid
        # Create a "Load Image" button
        self.load_button = tk.Button(self.options_row_1, text="Load Slice Directory", command=self.load_slice_directory)
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)
        # Create a label to display the original image
        self.original_image_label = tk.Label(self.imageGrid)
        self.original_image_label.pack(side=tk.LEFT, padx=5, pady=5)
    
        ##### Function buttons
        # Create a "Detect Edges" button
        self.conv_features_button = tk.Button(self.options_row_1, text="Extract conventional features", command=self.extract_conventional_features)
        self.conv_features_button.pack(side=tk.RIGHT, padx=5, pady=5)
        self.radio_features_button = tk.Button(self.options_row_2, text="Extract radiomic features", command=self.extract_radiomic_features)
        self.radio_features_button.pack(side=tk.RIGHT, padx=5, pady=5)
    
        ##### Option widgets
        # Create a drop down menu for switching the channel
        self.channel_options = ('T1', 'T1Gd', 'T2', 'T2-FLAI')
        self.annotation_label = tk.Label(self.options_row_4, text='Channel:')
        self.annotation_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.channel_var = tk.StringVar()
        self.channel_var.set(self.channel_options[0])
        self.channel_menu = tk.OptionMenu(self.options_row_4, self.channel_var, *self.channel_options, command=self.onChanges)
        self.channel_menu.pack(side=tk.LEFT, padx=5, pady=5)
        # Create drop down menus for toggling annotation visibility
        annotation_options = ('On', 'Off')
        self.annotation_var = tk.StringVar()
        self.annotation_var.set(annotation_options[0])
        self.annotation_menu = tk.OptionMenu(self.options_row_4, self.annotation_var, *annotation_options, command=self.onChanges)
        self.annotation_menu.pack(side=tk.RIGHT, padx=5, pady=5)
        self.annotation_label = tk.Label(self.options_row_4, text='Annotation:')
        self.annotation_label.pack(side=tk.RIGHT, padx=5, pady=5)
        # Create a Scale widget for the slice ID
        self.slice_ID_var = tk.IntVar()
        self.slice_ID_var.set(60)
        self.slice_ID_slider = tk.Scale(self.options_row_3, from_=0, to=154, resolution=1, orient=tk.HORIZONTAL, label="Slice ID", variable=self.slice_ID_var, command=self.onChanges)
        self.slice_ID_slider.pack(side=tk.RIGHT, padx=5, pady=5, fill='x', expand='true')

        # Initialise image
        self.load_slice_directory('example volume')


    # Load an image and display it
    def load_slice_directory(self, folder_path=''):
        if not folder_path:
            # Open a folder selection dialog box to choose a directory
            folder_path = filedialog.askdirectory(title="Select Slice Directory", initialdir='.')
        self.folder_path = folder_path

        # Search the directory and track the H5 files according to the filename conventions of the downloaded dataset.
        volume = []
        for i in range(155):
            # Read file
            file_path = folder_path + '/volume_1_slice_%i.h5' % i
            file = h5py.File(file_path, 'r')
            
            # Extract image
            image = file['image'][:] * 100                      #XXX: How should we normalise the array?
            channels = [Image.fromarray(image[:, :, i]) for i in range(len(self.channel_options))]
            
            # Extract mask
            masks = file['mask'][:]
            merged_mask = self.merge_mask(masks)

            volume.append({
                "image": channels,
                "mask": merged_mask
            })

        self.volume = volume
        
        # Display image
        self.update_image()
    

    # Resize and convert the image so it can be displayed.
    def prepare_photo(self, image, convertToPIL=True):
        # Save the filtered image
        self.filtered_image = image

        # Convert the image back to PIL format
        pil_image = Image.fromarray(image) if convertToPIL else image
        
        # Resize the image to fit in the label
        width, height = pil_image.size
        max_size = 500
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_width = int(width * (max_size / height))
            new_height = max_size
        pil_image = pil_image.resize((new_width, new_height))
        
        # Convert the image to Tkinter format
        photo = ImageTk.PhotoImage(pil_image)
        return photo


    # Display a new photo in the GUI
    def update_image(self):
        image, _ = self.get_current_image()
        photo = self.prepare_photo(image, convertToPIL=False)

        self.original_image_label.configure(image=photo)
        self.original_image_label.image = photo


    def get_current_image(self):
        slice_ID = self.slice_ID_var.get()
        channel_ID = self.channel_options.index(self.channel_var.get())

        image = self.volume[slice_ID]["image"][channel_ID]
        mask = self.volume[slice_ID]["mask"]
        return image, mask
    

    def get_current_volume(self, channel_ID, volume):
        # Extract 3D image and mask for the specified channel
        image_3d = [slice["image"][channel_ID] for slice in volume]
        mask_3d = [slice["mask"] for slice in volume]
        
        # Convert lists to numpy arrays
        image_3d = np.array(image_3d)
        mask_3d = np.array(mask_3d)
        
        return image_3d, mask_3d

    def filter_features(self, result):
        # Extract keys from the result dictionary
        keys = result.keys()
        
        # Filter keys that contain any of the specified substrings
        filtered_keys = []
        for key in keys:
            if any(substring in key for substring in self.substrings):
                filtered_keys.append(key)
        
        # Create a new dictionary containing only the filtered keys and their corresponding values
        filtered_result = {}
        for key in filtered_keys:
            filtered_result[key] = result[key]
        
        return filtered_result
    
    def calculate_repeatability(self, *results):
        # Filter features in all results
        filtered_results = []
        for result in results:
            filtered_result = self.filter_features(result)
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
            if key not in self.repeatability_scores:
                self.repeatability_scores[key] = 0
            self.repeatability_scores [key] += cv
        return
    def select_top_features(self, repeatability_scores, category, top_n):
        # Filter repeatability scores for the specified category
        filtered_scores = {}
        for key, val in repeatability_scores.items():
            if category in key:
                filtered_scores[key] = val
        
        # Sort the filtered scores by their repeatability values
        sorted_features = sorted(filtered_scores.items(), key=lambda item: item[1])
        
        # Extract top N features from the sorted list
        top_features = [feature for feature, _ in sorted_features[:top_n]]
        
        return top_features
    
    def group_and_select_features(self):
        #repeatability_scores = self.calculate_repeatability(*results)
        #print(self.repeatability_scores )
        top_intensity_features = self.select_top_features(self.repeatability_scores , 'firstorder', 10)
        top_shape_features = self.select_top_features(self.repeatability_scores , 'shape', 10)
        top_texture_features = self.select_top_features(self.repeatability_scores , 'glcm', 3) + \
                               self.select_top_features(self.repeatability_scores , 'gldm', 3) + \
                               self.select_top_features(self.repeatability_scores , 'glrlm', 2) + \
                               self.select_top_features(self.repeatability_scores , 'glszm', 2)
        
        return {
            'intensity_features': top_intensity_features,
            'shape_features': top_shape_features,
            'texture_features': top_texture_features
        }
    
    def extract_result(self, result, top_features):
        # Merge top_features values into a single list
        keys = [item for sublist in top_features.values() for item in sublist]
        
        # Filter result to only contain keys in keys
        top_feature = {key: result[key] for key in result.keys() if key in keys}
        
        return top_feature
    def extract_radiomic_features(self):
        Current_channel_ID = self.channel_options.index(self.channel_var.get())
        Feature_lists = []
        volume_lists = []
        hadHeader = False
        folder_path = filedialog.askdirectory(title="Select Slice Directory", initialdir='.')
        self.folder_path = folder_path
        # Ensure the volume folders are sorted
        volume_folders = sorted(os.listdir(self.folder_path))
        # Output result_to_show to a CSV file
        output_file = "radiomic_features.csv"

        for volume_name in volume_folders:
            volume_folder = os.path.join(self.folder_path, volume_name)
            # Get all slice files in the volume folder and sort them
            slice_files = sorted(os.listdir(volume_folder))
            volume = []
            volume_lists.append(slice_files[0].split('_')[1])
            for slice_file in slice_files:
                file_path = os.path.join(volume_folder, slice_file)
                file = h5py.File(file_path, 'r')
                # Extract image
                image = file['image'][:]
                channels = [Image.fromarray(image[:, :, i]) for i in range(len(self.channel_options))]
                
                # Extract mask
                masks = file['mask'][:]
                merged_mask = self.merge_mask(masks)

                volume.append({
                    "image": channels,
                    "mask": merged_mask
                })  
            result_list = []
            for channel_ID in range (0,4):
                image_3d, mask_3d = self.get_current_volume(channel_ID, volume)
                # Convert 3D numpy arrays to SimpleITK images
                sitk_volume = sitk.GetImageFromArray(image_3d)
                sitk_mask = sitk.GetImageFromArray(mask_3d)
                
                # Execute feature extraction on the volume and mask
                extractor = featureextractor.RadiomicsFeatureExtractor()
                result = extractor.execute(sitk_volume, sitk_mask)
                #result = self.filter_result(result)
                result_list.append(result)
                if (Current_channel_ID == channel_ID):
                    Feature_lists.append(result)
                    #result_to_show = result

            result1 = result_list[0]
            result2 = result_list[1]
            result3 = result_list[2]
            result4 = result_list[3]
            self.calculate_repeatability(result1, result2, result3, result4)
        top_features = self.group_and_select_features()
        #print(top_features)
        with open(output_file, 'w', newline='') as csvfile:
            i = 0
            for result_to_show in Feature_lists:
                result_to_show = self.extract_result(result_to_show, top_features)
                if(hadHeader is not True):
                    fieldnames = ['Volume'] + list(result_to_show.keys())  # Add 'Value' as the first column
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    hadHeader = True
                volume_number = volume_lists[i]
                i+=1
                Volume_Number = 'Volume_' + volume_number
                writer.writerow({'Volume': Volume_Number, **result_to_show})  # Write the data row
        print("Features extracted and saved to:", output_file)
        return
    
    def calculate_max_tumor_diameter(self, merged_mask):
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
    
    def generate_Outer_Layer(self, contours, image_w, image_h):
        # Create a blank image to store the contour
        Outer_Layer = np.zeros((image_w, image_h), dtype=np.uint8)

        # Draw each contour onto the blank image
        cv2.drawContours(Outer_Layer, contours, -1, 255, thickness=1)

        # Return the image of the contour
        return Outer_Layer
    
    def shrink_image(self, image):
        # Define a kernel for erosion
        kernel = np.ones((2,2), np.uint8)
        # Erode the image
        eroded_image = cv2.erode(image, kernel, iterations=1)
        # Convert the eroded image back to np.uint8
        eroded_image = eroded_image.astype(np.uint8)
        return eroded_image

    def outer_contours(self, image):
        # Set variables for the width and height of the image, and the number of layers required for the combined contour
        image_w = image.shape[1]
        image_h = image.shape[0]
        outer_layers = 5
        i = 0

        # Convert the grayscale image of the brain into a binarized format
        ret, image_th = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)

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
            contour_image = self.generate_Outer_Layer(contours, image_w, image_h)

            # Add the image of the contour to the accumulated image
            collected_contours += contour_image
            # Shrink image by 1 pixel
            image_uint8 =  self.shrink_image(image_uint8) 
            i+= 1

        # Return the combined image of the contours
        return collected_contours

    def merge_mask(self, masks):
        # Reshape masks array
        mask_array = [masks[:, :, i] for i in range(masks.shape[2])]

        # Merge non-overlapping masks by addition
        merged_Mask = mask_array[0] + mask_array[1] + mask_array[2]
        return merged_Mask

    def extract_conventional_features(self):
        folder_path = filedialog.askdirectory(title="Select Slice Directory", initialdir='.')
        self.folder_path = folder_path
        # Create or overwrite the CSV file to store the results
        csv_file_path = 'conventional_features.csv'
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Volume', 'Max Tumor Area', 'Max Tumor Diameter', 'Outer Layer Involvement'])

            # Ensure the volume folders are sorted
            volume_folders = sorted(os.listdir(self.folder_path))

            for volume_name in volume_folders:
                volume_folder = os.path.join(self.folder_path, volume_name)

                max_tumor_area = 0
                max_tumor_diameter = 0
                total_voxels = 0
                total_cortex_voxels = 0

                # Get all slice files in the volume folder and sort them
                slice_files = sorted(os.listdir(volume_folder))

                for slice_file in slice_files:
                    if slice_file.endswith('.h5'):
                        file_path = os.path.join(volume_folder, slice_file)
                        file = h5py.File(file_path, 'r')
                        datasetMask = file['mask'][:]
                        datasetImage = file['image'][:]
                        merged_mask = self.merge_mask(datasetMask)
                        tumor_area = np.sum(merged_mask)
                        max_tumor_area = max(max_tumor_area, tumor_area)

                        max_tumor_diameter_slice = self.calculate_max_tumor_diameter(merged_mask)
                        max_tumor_diameter = max(max_tumor_diameter, max_tumor_diameter_slice)

                        Outer_layer = self.outer_contours(datasetImage[:, :, 1])
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

        print("Conventional features extracted and saved to:", csv_file_path)
if __name__ == "__main__":
    root = tk.Tk()
    gui = ImageGUI(root)
    root.mainloop()