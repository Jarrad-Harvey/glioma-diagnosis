import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import cv2

class ImageGUI:

    # Change detection method
    def onChanges(self, value):
        # Update
        return
   
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
        self.options_top = tk.Frame(self.frame, relief="groove")
        self.options_top.grid(row=1, column=0, sticky="nsew")
        self.options_bottom = tk.Frame(self.frame, relief="groove")
        self.options_bottom.grid(row=2, column=0, sticky="nsew")
        
        ##### Image grid
        # Create a "Load Image" button
        self.load_button = tk.Button(self.options_top, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)
        # Create a label to display the original image
        self.original_image_label = tk.Label(self.imageGrid)
        self.original_image_label.pack(side=tk.LEFT, padx=5, pady=5)
        # Create a label to display the filtered image
        self.filtered_image_label = tk.Label(self.imageGrid)
        self.filtered_image_label.pack(side=tk.LEFT, padx=5, pady=5)
    
        ##### Function buttons
        # Create a "Detect Edges" button
        self.detect_edges_button = tk.Button(self.options_top, text="Apply optimal thresholds", command=self.apply_optimal_threshold)
        self.detect_edges_button.pack(side=tk.RIGHT, padx=5, pady=5)
    
        ##### Option widgets
        # Create a Scale widget for the upper threshold
        self.upper_thres_var = tk.IntVar()
        self.upper_thres_var.set(60)
        self.upper_thres_scale = tk.Scale(self.options_bottom, from_=0, to=180, resolution=1, orient=tk.HORIZONTAL, label="Upper threshold", variable=self.upper_thres_var, command=self.onChanges)
        self.upper_thres_scale.pack(side=tk.RIGHT, padx=5, pady=5, fill='x', expand='true')
        # Create a Scale widget for the lower threshold
        self.lower_thres_var = tk.IntVar()
        self.lower_thres_var.set(33)
        self.lower_thres_scale = tk.Scale(self.options_bottom, from_=0, to=180, resolution=1, orient=tk.HORIZONTAL, label="Lower threshold", variable=self.lower_thres_var, command=self.onChanges)
        self.lower_thres_scale.pack(side=tk.RIGHT, padx=5, pady=5, fill='x', expand='true')

        # Initialise image
        self.load_image('peppers.png')


    # Load an image and display it
    def load_image(self, file_path=''):
        if not file_path:
            # Open a file selection dialog box to choose an image file
            file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif")])
        self.image_path = file_path
        
        # Load the chosen image using PIL
        image = Image.open(file_path)
        
        # Display image
        self.original_image = image
        photo = self.prepare_photo(self.original_image, convertToPIL=False)
        self.update_image(photo, which='both')

        self.apply_optimal_threshold()
    

    # Resize and convert the image so it can be displayed.
    def prepare_photo(self, image, convertToPIL=True):
        # Save the filtered image
        self.filtered_image = image

        # Convert the image back to PIL format
        pil_image = Image.fromarray(image) if convertToPIL else image
        
        # Resize the image to fit in the label
        width, height = pil_image.size
        max_size = 300
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
    def update_image(self, photo, which='right'):
        if which in ['left', 'both']:
            self.original_image_label.configure(image=photo)
            self.original_image_label.image = photo
        if which in ['right', 'both']:
            self.filtered_image_label.configure(image=photo)
            self.filtered_image_label.image = photo


    def apply_optimal_threshold(self):
        image_name = self.image_path

        # If the name is recognised, apply preset threshold values.
        if 'iris.jpg' in image_name:
            self.lower_thres_var.set(0)
            self.upper_thres_var.set(30)
        elif 'peppers.png' in image_name:
            self.lower_thres_var.set(30)
            self.upper_thres_var.set(94)
        
        self.hue_threshold()


    # Apply a mask to the image which filters by a hue threshold
    def hue_threshold(self):
        image = np.array(self.original_image)

        # Convert to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower = self.lower_thres_var.get()
        upper = self.upper_thres_var.get()

        # Generate the HSV mask
        lower_hue = np.array([lower if lower < upper else 0,0,0], np.uint8)
        upper_hue = np.array([upper,255,255], np.uint8)
        mask = cv2.inRange(hsv_image, lower_hue, upper_hue)

        # Generate the wrap-around HSV mask
        # Hue is a cyclical value so lower > upper is a valid threshold. 
        if lower > upper:
            lower_hue = np.array([lower,0,0], np.uint8)
            upper_hue = np.array([upper if lower < upper else 255,255,255], np.uint8)
            wrapped_mask = cv2.inRange(hsv_image, lower_hue, upper_hue)
            mask = cv2.bitwise_or(mask, wrapped_mask)

        # Apply the mask
        filtered_image = cv2.bitwise_and(image,image,mask = mask)

        # Display filtered image
        photo = self.prepare_photo(filtered_image)
        self.update_image(photo, which='right')

            
if __name__ == "__main__":
    root = tk.Tk()
    gui = ImageGUI(root)
    root.mainloop()
