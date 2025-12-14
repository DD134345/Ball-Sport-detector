import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

# Ball sport classes (adjust these based on your training data)
BALL_CLASSES = [
    'Basketball',
    'Football',
    'Soccer Ball',
    'Tennis Ball',
    'Baseball',
    'Volleyball',
    'Golf Ball',
    'Bowling Ball',
    'Badminton Shuttlecock'
]

IMAGE_SIZE = (224, 224)
MODEL_PATH = 'best_ball_classifier.h5'


class BallDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ball Sport Detector")
        self.root.geometry("700x800")
        self.root.resizable(True, True)
        
        # Load model
        self.model = None
        self.load_model()
        
        # Create UI
        self.create_widgets()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(MODEL_PATH):
                self.model = load_model(MODEL_PATH)
                print(f"Model loaded successfully from {MODEL_PATH}")
            else:
                messagebox.showerror("Error", f"Model file not found: {MODEL_PATH}\nPlease ensure 'best_ball_classifier.h5' exists in the project directory.")
                self.model = None
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.model = None
    
    def create_widgets(self):
        """Create UI elements"""
        # Title
        title_label = tk.Label(self.root, text="🏀 Ball Sport Detector 🏀", font=("Arial", 24, "bold"))
        title_label.pack(pady=20)
        
        # Upload button
        upload_btn = tk.Button(self.root, text="📁 Upload Ball Image", command=self.upload_image, 
                              font=("Arial", 12), bg="#4CAF50", fg="white", padx=20, pady=10)
        upload_btn.pack(pady=10)
        
        # Image display
        self.image_label = tk.Label(self.root, bg="gray", width=60, height=20)
        self.image_label.pack(pady=20, padx=20)
        
        # Results frame
        results_frame = tk.Frame(self.root, bg="#f0f0f0", padx=20, pady=20)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        results_label = tk.Label(results_frame, text="Detection Results:", font=("Arial", 14, "bold"), bg="#f0f0f0")
        results_label.pack(anchor="w")
        
        # Results display
        self.results_text = tk.Text(results_frame, height=10, width=70, font=("Arial", 10), state=tk.DISABLED)
        self.results_text.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Clear button
        clear_btn = tk.Button(self.root, text="🔄 Clear", command=self.clear_results, 
                             font=("Arial", 11), bg="#2196F3", fg="white", padx=20, pady=8)
        clear_btn.pack(pady=10)
    
    def upload_image(self):
        """Handle image upload"""
        if self.model is None:
            messagebox.showerror("Error", "Model is not loaded. Please check the model file.")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select a ball image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            self.display_image(file_path)
            self.predict_ball(file_path)
    
    def display_image(self, file_path):
        """Display the selected image"""
        try:
            img = Image.open(file_path)
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            self.image_label.config(image=photo)
            self.image_label.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def predict_ball(self, file_path):
        """Predict the ball type"""
        try:
            # Load and preprocess image
            img = image.load_img(file_path, target_size=IMAGE_SIZE)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # Display results
            self.display_results(predictions[0], predicted_class, confidence)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to predict: {str(e)}")
    
    def display_results(self, predictions, predicted_class, confidence):
        """Display prediction results"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        # Top prediction
        result_text = f"🎯 TOP PREDICTION:\n"
        result_text += f"Ball Type: {BALL_CLASSES[predicted_class]}\n"
        result_text += f"Confidence: {confidence*100:.2f}%\n"
        result_text += f"\n{'='*50}\n"
        result_text += f"ALL PREDICTIONS (sorted by confidence):\n"
        result_text += f"{'='*50}\n\n"
        
        # Sort all predictions
        sorted_indices = np.argsort(predictions)[::-1]
        
        for rank, idx in enumerate(sorted_indices, 1):
            ball_type = BALL_CLASSES[idx]
            confidence_pct = predictions[idx] * 100
            bar_length = int(confidence_pct / 5)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            result_text += f"{rank}. {ball_type:<20} {confidence_pct:6.2f}% [{bar}]\n"
        
        self.results_text.insert(1.0, result_text)
        self.results_text.config(state=tk.DISABLED)
    
    def clear_results(self):
        """Clear all results"""
        self.image_label.config(image='')
        self.image_label.image = None
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = BallDetectorApp(root)
    root.mainloop()
