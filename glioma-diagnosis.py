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


    # XXX
    def extract_radiomic_features(self):
        # TODO
        return
    
    def calculate_max_tumor_diameter(self, segmentation_masks):
        max_tumor_diameter = 0
        for mask in segmentation_masks:
            # Find coordinates of tumor pixels
            tumor_pixels = np.argwhere(mask > 0)
            if len(tumor_pixels) > 1:
                # Perform PCA to find the longest linear measurement
                pca = PCA(n_components=2)
                pca.fit(tumor_pixels)
                eigen_vectors = pca.components_
                # Project tumor pixels onto the first principal component
                projected_tumor_pixels = np.dot(tumor_pixels - pca.mean_, eigen_vectors[0])
                # Calculate the longest linear measurement along the principal component
                tumor_diameter = np.max(projected_tumor_pixels) - np.min(projected_tumor_pixels)
                max_tumor_diameter = max(max_tumor_diameter, tumor_diameter)
        return max_tumor_diameter
    
    # XXX
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
                    outer_layer_involvement_sum = 0

                    for slice_idx in range(155):
                        # Process each slice in the volume
                        file_path = os.path.join(volume_folder, 'volume_%d_slice_%d.h5' % (volume_idx + 1, slice_idx))
                        file = h5py.File(file_path, 'r')
                        datasetMask = file['mask'][:]
                        datasetImage = file['image'][:]
                        masks_list = [datasetMask[:, :, i] for i in range(datasetMask.shape[2])]
                        #grayscale_dataset = [color.rgb2gray(image) for image in dataset]
                        #image = Image.fromarray(dataset.astype("uint8"))
                        # Calculate mean and standard deviation
                        #mean_intensity = np.mean(dataset)
                        #std_intensity = np.std(dataset)
                        #threshold = mean_intensity - 2 * std_intensity
                        # Calculate maximum tumor area
                        tumor_area = np.sum(datasetMask)
                        max_tumor_area = max(max_tumor_area, tumor_area)
                        # Calculating maximum tumor diameter
                        max_tumor_diameter_slice = self.calculate_max_tumor_diameter(masks_list)
                        max_tumor_diameter = max(max_tumor_diameter, max_tumor_diameter_slice)

                        # Calculate outer layer involvement
                        threshold = np.percentile(datasetImage, 94.3) 
                        #threshold = threshold_otsu(datasetImage) 
                        outer_layer_pixels = datasetImage[:, :, 0:5]  # Assuming outer layer thickness is 5 pixels
                        outer_layer_involvement = np.mean(outer_layer_pixels > threshold)
                        outer_layer_involvement_sum += outer_layer_involvement

                    # Average outer layer involvement across all slices
                    outer_layer_involvement_avg = outer_layer_involvement_sum / (155 * len(os.listdir(self.folder_path))) *100
                    Voulume_Number = 'Volume_' + str(volume_idx + 1)
                    # Write results to CSV
                    csv_writer.writerow([Voulume_Number, max_tumor_area, max_tumor_diameter, outer_layer_involvement_avg])

            print("Conventional features extracted and saved to:", csv_file_path)
            
if __name__ == "__main__":
    root = tk.Tk()
    gui = ImageGUI(root)
    root.mainloop()