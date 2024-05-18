import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import os
import csv
import cv2
from sklearn.decomposition import PCA
import h5py
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances


class ImageGUI:

    # Change detection method
    def onChanges(self, value):
        self.update_image()
   
    def __init__(self, master):
        self.master = master
        self.master.title("Glioma Diagnosis")
    
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
        channel_options = ('T1', 'T1Gd', 'T2', 'T2-FLAI')
        self.annotation_label = tk.Label(self.options_row_4, text='Channel:')
        self.annotation_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.channel_var = tk.StringVar()
        self.channel_var.set(channel_options[0])
        self.channel_menu = tk.OptionMenu(self.options_row_4, self.channel_var, *channel_options, command=self.onChanges)
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
            file_path = folder_path + '/volume_1_slice_%i.h5' % i
            file = h5py.File(file_path, 'r')
            dataset = file['image'][:] * 100
            #print(dataset)                    #XXX: How should we normalise the array?
            image = Image.fromarray(dataset.astype("uint8"))
            volume.append(image)
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
        slice_ID = self.slice_ID_var.get()
        image = self.volume[slice_ID]
        photo = self.prepare_photo(image, convertToPIL=False)

        self.original_image_label.configure(image=photo)
        self.original_image_label.image = photo


    def select_top_features_by_group(self, features_sets_grouped):
        top_features = {}
        
        for group_name, features_sets in features_sets_grouped.items():
            n_features = features_sets[0].shape[0]
            repeatability_scores = np.zeros(n_features)
            
            # Calculate repeatability scores for each feature across all pairs of feature sets
            n_sets = len(features_sets)
            for i in range(n_sets):
                for j in range(i + 1, n_sets):
                    for feature_idx in range(n_features):
                        feature_set_1 = features_sets[i][feature_idx, :]
                        feature_set_2 = features_sets[j][feature_idx, :]
                        pearson_corr, cos_similarity, euclidean_dist = self.calculate_repeatability(feature_set_1, feature_set_2)
                        
                        # Combine the scores into a single repeatability score for this pair
                        repeatability_score = (pearson_corr + cos_similarity - euclidean_dist) / 3
                        repeatability_scores[feature_idx] += repeatability_score
            
            # Normalize by the number of comparisons
            n_comparisons = n_sets * (n_sets - 1) / 2
            repeatability_scores /= n_comparisons
            
            # Sort features based on repeatability scores
            sorted_features = sorted(enumerate(repeatability_scores), key=lambda x: x[1], reverse=True)
            
            # Select top 10 features
            top_10_features_indices = [idx for idx, _ in sorted_features[:10]]
            
            top_features[group_name] = top_10_features_indices
        
        return top_features
    
    # XXX
    def extract_radiomic_features(self):
        features_day1_intensity = np.random.rand(20, 10)
        features_day1_shape = np.random.rand(20, 10)
        features_day1_texture = np.random.rand(20, 10)

        features_day2_intensity = np.random.rand(20, 10)
        features_day2_shape = np.random.rand(20, 10)
        features_day2_texture = np.random.rand(20, 10)

        features_day3_intensity = np.random.rand(20, 10)
        features_day3_shape = np.random.rand(20, 10)
        features_day3_texture = np.random.rand(20, 10)

        features_day4_intensity = np.random.rand(20, 10)
        features_day4_shape = np.random.rand(20, 10)
        features_day4_texture = np.random.rand(20, 10)

        # Put all feature sets into a dictionary grouped by feature type
        features_sets_grouped = {
            'intensity': [
                features_day1_intensity, features_day2_intensity, features_day3_intensity, features_day4_intensity
            ],
            'shape': [
                features_day1_shape, features_day2_shape, features_day3_shape, features_day4_shape
            ],
            'texture': [
                features_day1_texture, features_day2_texture, features_day3_texture, features_day4_texture
            ]
        }

        # Select top 10 features for each group
        top_features = self.select_top_features_by_group(features_sets_grouped)

        print("Top 10 intensity features:", top_features['intensity'])
        print("Top 10 shape features:", top_features['shape'])
        print("Top 10 texture features:", top_features['texture'])
    
    def calculate_repeatability(self, feature_set_1, feature_set_2):
        # Pearson correlation coefficient
        pearson_corr, _ = pearsonr(feature_set_1, feature_set_2)
        
        # Cosine similarity
        cos_similarity = cosine_similarity([feature_set_1], [feature_set_2])[0][0]
        
        # Euclidean distance (we use the negative distance to be consistent with correlation and similarity measures)
        euclidean_dist = -euclidean_distances([feature_set_1], [feature_set_2])[0][0]
        
        return pearson_corr, cos_similarity, euclidean_dist

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

    def merge_mask(self, mask_array):
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

            for volume_idx, volume_path in enumerate(os.listdir(self.folder_path)):
                # Process each volume
                volume_folder = os.path.join(self.folder_path, volume_path)

                max_tumor_area = 0
                max_tumor_diameter = 0
                total_voxels = 0
                total_cortex_voxels = 0

                for slice_idx in range(155):
                    # Process each slice in the volume
                    file_path = os.path.join(volume_folder, 'volume_%d_slice_%d.h5' % (volume_idx + 1, slice_idx))
                    file = h5py.File(file_path, 'r')
                    datasetMask = file['mask'][:]
                    datasetImage = file['image'][:]
                    masks_list = [datasetMask[:, :, i] for i in range(datasetMask.shape[2])]
                    merged_mask = self.merge_mask(masks_list)
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
                Voulume_Number = 'Volume_' + str(volume_idx + 1)
                csv_writer.writerow([Voulume_Number, max_tumor_area, max_tumor_diameter, outer_layer_involvement_avg])

        print("Conventional features extracted and saved to:", csv_file_path)
if __name__ == "__main__":
    root = tk.Tk()
    gui = ImageGUI(root)
    root.mainloop()