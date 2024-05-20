import csv
import os
import re
import tkinter as tk
from tkinter import filedialog

from imblearn.over_sampling import SMOTE
import SimpleITK as sitk
import cv2
import h5py
import numpy as np
import pandas as pd
from PIL import ImageTk, Image
from radiomics import featureextractor
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold, cross_val_predict, \
    GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


def loadData():
    labels = ['HGG', 'LGG']
    df = pd.read_csv('name_mapping.csv')

    # Step 2: Extract Grade and BRATS 2020 Subject ID
    df_subset = df[['Grade', 'BraTS_2020_subject_ID']]

    # Step 3: Extract Volume Number
    def extract_volume_number(subject_id):
        match = re.search(r'\d+$', subject_id)
        if match:
            return f'Volume_{int(match.group())}'
        else:
            return None

    df_subset = df_subset.copy()
    df_subset['Volume'] = df_subset['BraTS_2020_subject_ID'].apply(extract_volume_number)

    df_subset['Volume_Number'] = df_subset['Volume'].apply(lambda x: int(x.split('_')[1]))
    df_sorted = df_subset.sort_values(by='Volume_Number')
    df_final = df_sorted[['Volume', 'Grade']]

    # Step 4: Combine Data
    # df_final = df_subset[['Volume', 'Grade']].iloc[:118]

    df_radiomic = pd.read_csv('radiomic_features.csv')
    df_radiomic = df_radiomic.copy()

    df_radiomic['Volume_Number'] = df_radiomic['Volume'].apply(lambda x: int(x.split('_')[1]))
    df_radiomic_sorted = df_radiomic.sort_values(by='Volume_Number')
    df_radiomic_final = df_radiomic_sorted.drop(columns=['Volume_Number'])

    # df_conventional = pd.read_csv('conventional_features.csv')
    # combined_features_df = pd.concat([df_radiomic, df_conventional], axis=1)
    # print(combined_features_df)
    # combined_features_df.to_csv('combined.csv')

    print('here after combined')
    df_merged = pd.merge(df_final, df_radiomic_final, on='Volume', suffixes=('', '_radiomic'))

    # df_final_features = pd.concat([df_radiomic_final, df_final], axis=1)
    df_merged.to_csv('volume_feature.csv', index=False)

    df_final_csv = df_merged.drop('Volume', axis=1)
    print('here after final')

    df_final_csv.to_csv('final_feature.csv', index=False)
    print('here after final')

    # Step 5: Save to CSV
    df_final.to_csv('output.csv', index=False)


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
        self.conv_features_button = tk.Button(self.options_row_1, text="Extract conventional features",
                                              command=self.extract_conventional_features)
        self.conv_features_button.pack(side=tk.RIGHT, padx=5, pady=5)
        self.radio_features_button = tk.Button(self.options_row_2, text="Extract radiomic features",
                                               command=self.extract_radiomic_features)
        self.radio_features_button.pack(side=tk.RIGHT, padx=5, pady=5)

        ##### Option widgets
        # Create a drop down menu for switching the channel
        # channel_options = ('T1', 'T1Gd', 'T2', 'T2-FLAI')
        self.channel_options = {
            'T2 Fluid Attenuated Inversion Recovery (T2-FLAIR)': 0,
            'native (T1)': 1,
            'post-contrast T1-weighted (T1Gd)': 2,
            'T2-weighted (T2)': 3

        }
        self.annotation_label = tk.Label(self.options_row_4, text='Channel:')
        self.annotation_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.channel_var = tk.StringVar()
        self.channel_var.set('T2 Fluid Attenuated Inversion Recovery (T2-FLAIR)')
        self.channel_menu = tk.OptionMenu(self.options_row_4, self.channel_var, *self.channel_options,
                                          command=self.onChanges)
        self.channel_menu.pack(side=tk.LEFT, padx=5, pady=5)
        # Create drop down menus for toggling annotation visibility
        annotation_options = ('On', 'Off')
        self.annotation_var = tk.StringVar()
        self.annotation_var.set(annotation_options[0])
        self.annotation_menu = tk.OptionMenu(self.options_row_4, self.annotation_var, *annotation_options,
                                             command=self.onChanges)
        self.annotation_menu.pack(side=tk.RIGHT, padx=5, pady=5)
        self.annotation_label = tk.Label(self.options_row_4, text='Annotation:')
        self.annotation_label.pack(side=tk.RIGHT, padx=5, pady=5)
        # Create a Scale widget for the slice ID
        self.slice_ID_var = tk.IntVar()
        self.slice_ID_var.set(60)
        self.slice_ID_slider = tk.Scale(self.options_row_3, from_=0, to=154, resolution=1, orient=tk.HORIZONTAL,
                                        label="Slice ID", variable=self.slice_ID_var, command=self.onChanges)
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
            image = file['image'][:] * 100  # XXX: How should we normalise the array?
            channels = [Image.fromarray(image[:, :, i]) for i in range(len(self.channel_options))]

            # Extract mask
            masks_sep = file['mask'][:] * 100
            masks = file['mask'][:]

            merged_mask = self.merge_mask(masks)

            # slice_dict={
            #     "image": channels,
            #     "mask": merged_mask,
            #     "mask_sep":masks_sep
            # }
            # print(slice_dict)
            volume.append({
                "image": channels,
                "mask": merged_mask,
                "mask_sep": masks_sep
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
        pil_image.convert('RGB')

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
        mask_flag = self.annotation_var.get()

        # slice_ID = self.slice_ID_var.get()
        # channel_ID = self.channel_options[self.channel_var.get()]
        # images = self.volume[slice_ID]['image'][channel_ID]
        image, mask, mask_sep = self.get_current_image()
        image = np.array(image)
        print("hey there its me")
        # print(self.volume[0]['image'][channel_ID])

        print("obi wan")

        print(type(image))

        # print(images)
        image_normalized = ((image - image.min()) / (image.max() - image.min())) * 255
        image_normalized = np.round(image_normalized).astype(np.uint8)
        print("this is mask flag")
        print(mask_flag)

        if mask_flag == "On":

            mask_image = mask_sep

            rgb_image = np.stack([image_normalized] * 3, axis=-1)
            color_mask = np.stack([mask_image[:, :, 0], mask_image[:, :, 1], mask_image[:, :, 2]], axis=-1)

            # mask = cv2.resize(mask_image, (image.shape[1], image.shape[0]))

            # blended = cv2.addWeighted(rgb_image, 1.0, color_mask, 0.5, 0)
            blended = np.clip(rgb_image * 1.0 + color_mask * 0.5, 0, 255).astype(np.uint8)

            # blended_image_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

            photo = self.prepare_photo(blended, convertToPIL=True)
        else:

            photo = self.prepare_photo(image_normalized, convertToPIL=True)

        self.original_image_label.configure(image=photo)
        self.original_image_label.image = photo

    def get_current_image(self):
        slice_ID = self.slice_ID_var.get()
        channel_ID = self.channel_options[self.channel_var.get()]

        image = self.volume[slice_ID]["image"][channel_ID]
        mask = self.volume[slice_ID]["mask"]
        mask_sep = self.volume[slice_ID]["mask_sep"]
        # print(image)
        return image, mask, mask_sep

    def get_current_volume(self, channel_ID, volume):
        # channel_ID = self.channel_options[self.channel_var.get()]

        # Extract 3D image and mask for the specified channel
        image_3d = [slice["image"][channel_ID] for slice in self.volume]
        mask_3d = [slice["mask"] for slice in self.volume]

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
            self.repeatability_scores[key] += cv
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
        # repeatability_scores = self.calculate_repeatability(*results)
        # print(self.repeatability_scores )
        top_intensity_features = self.select_top_features(self.repeatability_scores, 'firstorder', 10)
        top_shape_features = self.select_top_features(self.repeatability_scores, 'shape', 10)
        top_texture_features = self.select_top_features(self.repeatability_scores, 'glcm', 3) + \
                               self.select_top_features(self.repeatability_scores, 'gldm', 3) + \
                               self.select_top_features(self.repeatability_scores, 'glrlm', 2) + \
                               self.select_top_features(self.repeatability_scores, 'glszm', 2)

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
        Current_channel_ID = self.channel_options[self.channel_var.get()]
        Feature_lists = []
        volume_lists = []
        hadHeader = False
        folder_path = filedialog.askdirectory(title="Select Slice Directory", initialdir='.')
        self.folder_path = folder_path
        print(self.folder_path)
        # Ensure the volume folders are sorted

        # volume_folders = sorted(os.listdir(self.folder_path))
        volume_folders = sorted(
            [item.name for item in os.scandir(self.folder_path) if item.is_dir() and not item.name.startswith('.')])

        # volume_folders = sorted([item for item in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, item))])
        print(len(volume_folders))
        print(volume_folders)
        # Output result_to_show to a CSV file
        output_file = "radiomic_features.csv"

        for volume_name in volume_folders:
            volume_folder = os.path.join(self.folder_path, volume_name)
            # Get all slice files in the volume folder and sort them

            # slice_files = sorted([file for file in os.listdir(volume_folder) if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.h5'))])

            slice_files = sorted(os.listdir(volume_folder))
            volume = []
            print(slice_files)
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
            for channel_ID in range(0, 4):
                image_3d, mask_3d = self.get_current_volume(channel_ID, volume)
                # Convert 3D numpy arrays to SimpleITK images
                sitk_volume = sitk.GetImageFromArray(image_3d)
                sitk_mask = sitk.GetImageFromArray(mask_3d)

                # Execute feature extraction on the volume and mask
                extractor = featureextractor.RadiomicsFeatureExtractor()
                result = extractor.execute(sitk_volume, sitk_mask)
                # result = self.filter_result(result)
                result_list.append(result)
                if (Current_channel_ID == channel_ID):
                    Feature_lists.append(result)
                    # result_to_show = result

            result1 = result_list[0]
            result2 = result_list[1]
            result3 = result_list[2]
            result4 = result_list[3]
            self.calculate_repeatability(result1, result2, result3, result4)
        top_features = self.group_and_select_features()
        # print(top_features)
        with open(output_file, 'w', newline='') as csvfile:
            i = 0
            for result_to_show in Feature_lists:
                result_to_show = self.extract_result(result_to_show, top_features)
                if (hadHeader is not True):
                    fieldnames = ['Volume'] + list(result_to_show.keys())  # Add 'Value' as the first column
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    hadHeader = True
                volume_number = volume_lists[i]
                i += 1
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
            # max_tumor_diameter = max(max_tumor_diameter, tumor_diameter)
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
        kernel = np.ones((2, 2), np.uint8)
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
                i += 1
                continue

            # Call the generate_Outer_Layers function to return an image of the contour
            contour_image = self.generate_Outer_Layer(contours, image_w, image_h)

            # Add the image of the contour to the accumulated image
            collected_contours += contour_image
            # Shrink image by 1 pixel
            image_uint8 = self.shrink_image(image_uint8)
            i += 1

        # Return the combined image of the contours
        return collected_contours

    def merge_mask(self, masks):
        # Reshape masks array
        mask_array = [masks[:, :, i] for i in range(masks.shape[2])]

        # Merge non-overlapping masks by addition
        merged_Mask = mask_array[0] + mask_array[1] + mask_array[2]
        return merged_Mask

    def filter_samples(self, df, label_column, class_names, n_hidden_test=10, random_state=42):

        label_encoder = LabelEncoder()
        df_encoded = df.copy()
        df_encoded[label_column] = label_encoder.fit_transform(df_encoded[label_column])

        # Determine the mapping between original class names and encoded labels
        class_mapping = {label: name for label, name in enumerate(label_encoder.classes_)}

        # Filter samples based on the encoded labels corresponding to each class name
        filtered_samples = []
        hidden_test_samples = []
        for class_name in class_names:
            encoded_label = label_encoder.transform([class_name])[0]
            filtered_samples_class = df_encoded[df_encoded[label_column] == encoded_label]
            hidden_test_class = filtered_samples_class.sample(n=n_hidden_test, random_state=random_state)
            hidden_test_samples.append(hidden_test_class)
            filtered_samples.append(filtered_samples_class.drop(hidden_test_class.index))
            print(class_name)
            print(class_mapping)
            print(filtered_samples_class,flush=True)
        filtered_samples_combined = pd.concat(filtered_samples, axis=0)
        hidden_test_samples_combined = pd.concat(hidden_test_samples, axis=0)
        return filtered_samples_combined, hidden_test_samples_combined, class_mapping

    def train_SVM(self):
        features_df = pd.read_csv('final_feature.csv')
        class_names = ['HGG', 'LGG']
        filtered_samples, hidden_test_samples, class_mapping = self.filter_samples(features_df, 'Grade', class_names)
        print(filtered_samples.value_counts())
        print(filtered_samples,flush=True)

        # label_encoder = LabelEncoder()
        # features_df['Grade'] = label_encoder.fit_transform(features_df['Grade'])
        # class_mapping = {label: class_name for label, class_name in enumerate(label_encoder.classes_)}
        # print("Class Mapping:", class_mapping)
        # hgg_samples = features_df[features_df['Grade'] == 'HGG']
        # lgg_samples = features_df[features_df['Grade'] == 'LGG']
        #
        # hidden_test_hgg = hgg_samples.sample(n=10, random_state=42)
        # hidden_test_lgg = lgg_samples.sample(n=10, random_state=42)
        # hidden_test_set = pd.concat([hidden_test_hgg, hidden_test_lgg])
        # features_df = features_df.drop(hidden_test_set.index)
        # original_labels = features_df['Grade'].unique()
        #
        # # unique_encoded_labels = np.unique(features_df)
        # print("Unique encoded labels:", original_labels)
        # numerical_labels = label_encoder.classes_
        #
        # # Create a mapping dictionary
        # label_mapping = dict(zip(numerical_labels, original_labels))
        # print("numerical encoded labels:", original_labels)

        X = filtered_samples.drop('Grade', axis=1)  # Features
        print(X,flush=True)

        y = filtered_samples['Grade']
        # smote = SMOTE(random_state=42)
        # X_resampled, y_resampled = smote.fit_resample(X, y)
        # y_resampled_series = pd.Series(y_resampled)
        # class_counts_resampled = y_resampled_series.value_counts()
        #
        # # Print the value counts for each class
        # print("Class Counts in Resampled Data:")
        # print(class_counts_resampled)
        # print('lebgth')
        # print(len(X_resampled))
        # print(len(y_resampled))
        # print(X_resampled.shape)
        # print(y_resampled.shape)

        # hgg_samples = features_df[features_df['Grade'] == 'HGG']
        # lgg_samples = features_df[features_df['Grade'] == 'LGG']
        # print(hgg_samples.size)
        # print(lgg_samples.size)

        # Split training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X,  # Features
            y,  # Labels
            test_size=0.2,  # Size of hidden test set
            stratify=y,  # Stratify by grade to ensure proportional split
            random_state=42
        )
        #
        # # Check the sizes of each set
        # print("Hidden Test Set:", hidden_test_samples.shape)
        # print("Training Set:", X_train.shape)
        # print("Validation Set:", X_val.shape)
        #
        # svm_model = SVC(kernel='rbf', C=0.1, gamma='auto', random_state=42)
        #
        # cv_strategy = StratifiedKFold(n_splits=50, shuffle=True, random_state=42)
        # cv_strategy_hidden = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # cv_scores = cross_val_score(svm_model, X_resampled, y_resampled, cv=cv_strategy, scoring='accuracy')

        clf = GridSearchCV(SVC(gamma='auto'),{
            'C':[1,10,20],
            'kernel':['rbf','linear']
        },cv = 5,return_train_score = False)

        print(y_train.value_counts())
        clf.fit(X_train,y_train)
        print(clf.cv_results_)
        print(clf.best_score_)
        print(clf.best_params_)
        best_model = clf.best_estimator_
        best_model.fit(X_train,y_train)

        hidden_test_X = hidden_test_samples.drop('Grade', axis=1)
        hidden_test_y = hidden_test_samples['Grade']
        test_accuracy = best_model.score(hidden_test_X, hidden_test_y)
        print("Test Accuracy (hidden test set):", test_accuracy)



    # cv_strategies = [
    #     KFold(n_splits=5, shuffle=True, random_state=42),
    #     StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    #     # Add more cross-validation strategies as needed
    # ]
    # for cv_strategy in cv_strategies:
    #     scores = cross_val_score(svm_model, X_resampled, y_resampled, cv=cv_strategy, scoring='accuracy')
    #     print(f"Mean accuracy ({cv_strategy.__class__.__name__}): {scores.mean():.4f} (+/- {scores.std():.4f})")

    # print("Cross-validation scores:", cv_scores)
    #
    # # Step 5: Print mean and standard deviation of cross-validation scores
    # print("Mean CV accuracy:", np.mean(cv_scores))
    # print("Standard deviation of CV accuracy:", np.std(cv_scores))

    # Fit the SVM model to the training data
    # svm_model.fit(X_train, y_train)
    #
    # # Make predictions on the validation set
    # hidden_test_X = hidden_test_samples.drop('Grade', axis=1)
    # hidden_test_y = hidden_test_samples['Grade']
    # y_pred = cross_val_predict(svm_model, hidden_test_X, hidden_test_y, cv=cv_strategy_hidden)
    #
    # #
    # # # Evaluate the model
    # accuracy = accuracy_score(hidden_test_y, y_pred)
    # print("Hidden test Accuracy:", accuracy)

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
        loadData()
        self.train_SVM()


if __name__ == "__main__":
    root = tk.Tk()
    gui = ImageGUI(root)
    root.mainloop()
