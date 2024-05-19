import tkinter as tk
from tkinter import filedialog

import cv2
import h5py
import numpy as np
from PIL import ImageTk, Image


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
    def load_slice_directory(self, folder_path):
        # if not folder_path:
        # Open a folder selection dialog box to choose a directory
        folder_path = filedialog.askdirectory(title="Select Slice Directory", initialdir='.')
        self.folder_path = folder_path

        # Search the directory and track the H5 files according to the filename conventions of the downloaded dataset.
        volume = []
        for i in range(155):
            file_path = folder_path + '/volume_1_slice_%i.h5' % i
            file = h5py.File(file_path, 'r')
            dataset = file['image'][:] * 100  # XXX: How should we normalise the array?


            mask = file['mask'][:] * 100
            slice_dict = {'image': dataset, 'mask': mask}
            # image = Image.fromarray(dataset.astype("uint8"))

            volume.append(slice_dict)
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
        slice_ID = self.slice_ID_var.get()
        mask_flag = self.annotation_var.get()
        channel_ID = self.channel_options[self.channel_var.get()]
        image = self.volume[slice_ID]['image']
        image = image[:, :, channel_ID]
        image_normalized = ((image - image.min()) / (image.max() - image.min())) * 255
        image_normalized = np.round(image_normalized).astype(np.uint8)


        if mask_flag == "On":

            mask_image = self.volume[slice_ID]['mask']



            rgb_image = np.stack([image_normalized] * 3, axis=-1)
            color_mask = np.stack([mask_image[:, :,0], mask_image[:, :,1], mask_image[:, :,2]], axis=-1)




            # mask = cv2.resize(mask_image, (image.shape[1], image.shape[0]))




            # blended = cv2.addWeighted(rgb_image, 1.0, color_mask, 0.5, 0)
            blended = np.clip(rgb_image * 1.0 + color_mask * 0.5, 0, 255).astype(np.uint8)



            # blended_image_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

            photo = self.prepare_photo(blended, convertToPIL=True)
        else:

            photo = self.prepare_photo(image_normalized, convertToPIL=True)

        self.original_image_label.configure(image=photo)
        self.original_image_label.image = photo

    # XXX
    def extract_radiomic_features(self):
        # TODO
        return

    # XXX
    def extract_conventional_features(self):
        # TODO
        return


if __name__ == "__main__":
    root = tk.Tk()
    gui = ImageGUI(root)
    root.mainloop()
