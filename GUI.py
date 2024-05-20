import tkinter as tk
from PIL import ImageTk, Image
from read_volumes import load_volume
from extract_features import extract_radiomic_features, extract_conventional_features
from calculate_best_features import perform_repeatability_test
from settings import * 

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
        # Create a label to display the original image
        self.image = tk.Label(self.imageGrid)
        self.image.pack(side=tk.LEFT, padx=5, pady=5)
    
        ##### Function buttons
        # Create a "Load Slice Directory" button
        self.load_button = tk.Button(self.options_row_1, text="Load Slice Directory", command=self.load_slice_directory)
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)
        # Create a "Perform repeatability test" button
        self.load_button = tk.Button(self.options_row_2, text="Perform repeatability test", command=perform_repeatability_test)
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)
        # Create an "Extract conventional features" button
        self.conv_features_button = tk.Button(self.options_row_1, text="Extract conventional features", command=extract_conventional_features)
        self.conv_features_button.pack(side=tk.RIGHT, padx=5, pady=5)
        # Create an "Extract radiomic features" button
        self.radio_features_button = tk.Button(self.options_row_2, text="Extract radiomic features", command=lambda: extract_radiomic_features(channel_options.index(self.channel_var.get())))
        self.radio_features_button.pack(side=tk.RIGHT, padx=5, pady=5)
    
        ##### Option widgets
        # Create a drop down menu for switching the channel
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
    

    def load_slice_directory(self, folderPath=''):
        self.volume = load_volume(folderPath)
        self.update_image()


    # Resize and convert the image so it can be displayed.
    def prepare_photo(self, image, convertToPIL=True):

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
        channel_ID = channel_options.index(self.channel_var.get())

        image = self.volume["slices"][slice_ID]["image"][channel_ID]

        photo = self.prepare_photo(image, convertToPIL=False)

        self.image.configure(image=photo)
        self.image.image = photo