import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
from tkinter import simpledialog
import webbrowser
import glob
import ultralytics
# from IPython.display import Image, display
HOME = os.getcwd()
print(HOME)
from roboflow import Roboflow
rf = Roboflow(api_key="QUWMCW70c1dPy9pc03sQ")
project = rf.workspace("hiteshram").project("object-detection-bounding-box-ftfs5")
dataset = project.version(1).download("yolov5")

class PotholeDetectorApp:
    def __init__(self, master):
        self.master = master
        master.title("Pothole Detection System")
        master.geometry("1200x700")
        master.configure(bg="#1e1e2d")
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.camera = None
        self.is_camera_running = False

        self.frame = ttk.Frame(master, padding=10)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.title_label = ttk.Label(self.frame, text="Pothole Detection System", font=("Arial", 24, "bold"))
        self.title_label.pack(pady=10)

        button_frame = ttk.Frame(self.frame)
        button_frame.pack(pady=20)

        self.upload_button = ttk.Button(button_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.grid(row=0, column=0, padx=10)

        self.detect_button = ttk.Button(button_frame, text="Detect Potholes", command=self.detect_potholes)
        self.detect_button.grid(row=0, column=1, padx=10)

        self.clear_button = ttk.Button(button_frame, text="Clear Image", command=self.clear_image)
        self.clear_button.grid(row=0, column=2, padx=10)

        self.crop_button = ttk.Button(button_frame, text="Crop Image", command=self.crop_image)
        self.crop_button.grid(row=0, column=3, padx=10)

        self.rotate_button = ttk.Button(button_frame, text="Rotate Image", command=self.rotate_image)
        self.rotate_button.grid(row=0, column=4, padx=10)

        self.save_button = ttk.Button(button_frame, text="Save Image", command=self.save_image)
        self.save_button.grid(row=0, column=5, padx=10)

        # self.camera_button = ttk.Button(button_frame, text="Open Camera", command=self.toggle_camera)
        # self.camera_button.grid(row=0, column=6, padx=10)

        self.info_button = ttk.Button(button_frame, text="More Info", command=self.open_info)
        self.info_button.grid(row=0, column=7, padx=10)

        self.image_label = ttk.Label(self.frame)
        self.image_label.pack(pady=20)

        self.status_label = ttk.Label(self.frame, text="Upload an image to detect potholes", font=("Arial", 12))
        self.status_label.pack(pady=10)

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if self.image_path:
            self.status_label.config(text=f"Selected Image: {os.path.basename(self.image_path)}")
            self.load_image(self.image_path)

    def load_image(self, image_path):
        self.original_image = cv2.imread(image_path)
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.display_image(self.original_image)

    def display_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((500, 400))
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo

    def detect_potholes(self):
        if self.processed_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pothole_count = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                pothole_count += 1
                cv2.drawContours(self.processed_image, [contour], -1, (0, 255, 0), 2)

        self.display_image(self.processed_image)
        self.status_label.config(text=f"Potholes Detected: {pothole_count}")

    def crop_image(self):
        if self.processed_image is not None:
            x = simpledialog.askinteger("Crop", "Enter X coordinate:")
            y = simpledialog.askinteger("Crop", "Enter Y coordinate:")
            w = simpledialog.askinteger("Crop", "Enter Width:")
            h = simpledialog.askinteger("Crop", "Enter Height:")
            if x is not None and y is not None and w is not None and h is not None:
                self.processed_image = self.processed_image[y:y + h, x:x + w]
                self.display_image(self.processed_image)
        else:
            messagebox.showerror("Error", "No image to crop.")

    def rotate_image(self):
        if self.processed_image is not None:
            self.processed_image = cv2.rotate(self.processed_image, cv2.ROTATE_90_CLOCKWISE)
            self.display_image(self.processed_image)
        else:
            messagebox.showerror("Error", "No image to rotate.")

    # def toggle_camera(self):
    #     if not self.is_camera_running:
    #         self.is_camera_running = True
    #         self.camera_button.config(text="Close Camera")
    #         self.camera = cv2.VideoCapture(0)
    #         self.update_camera()
    #     else:
    #         self.is_camera_running = False
    #         self.camera_button.config(text="Open Camera")
    #         self.camera.release()
    #         self.image_label.config(image='')

    # def update_camera(self):
    #     if self.is_camera_running and self.camera.isOpened():
    #         ret, frame = self.camera.read()
    #         if ret:
    #             self.processed_image = frame
    #             self.display_image(frame)
    #             self.master.after(10, self.update_camera)
    #         else:
    #             messagebox.showerror("Error", "Failed to capture camera frame")
    #             self.toggle_camera()

    def save_image(self):
        if self.processed_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
            if file_path:
                cv2.imwrite(file_path, self.processed_image)
                messagebox.showinfo("Success", f"Image saved at {file_path}")
        else:
            messagebox.showerror("Error", "No image to save.")

    def clear_image(self):
        self.image_label.config(image='')
        self.status_label.config(text="Upload an image to detect potholes")

    def open_info(self):
        webbrowser.open("https://www.example.com")

if __name__ == "__main__":
    root = tk.Tk()
    app = PotholeDetectorApp(root)
    root.mainloop()
