import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import threading
from pathlib import Path

# Ball sport classes (must match training_model.py output layer: units=6)
BALL_CLASSES = [
    'basketball',
    'billiard_ball',
    'bowling_ball',
    'football',
    'tennis_ball',
    'volleyball'
]

# Model and preprocessing settings (must match training_model.py)
IMAGE_SIZE = (192, 192)  # Must match training size
MODEL_PATH = 'Ball_sport_classifier.h5'
CONFIDENCE_THRESHOLD = 0.1  # Minimum confidence to display prediction


class BallDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ball Sport Detector")
        self.root.geometry("800x900")
        self.root.resizable(True, True)
        
        # Load model
        self.model = None
        self.is_loading = False
        self.load_model()
        
        # Create UI
        self.create_widgets()
    
    def load_model(self):
        """Load the trained model with error handling"""
        try:
            if not os.path.exists(MODEL_PATH):
                error_msg = (
                    f"Model file not found: {MODEL_PATH}\n\n"
                    f"Please ensure 'best_ball_classifier.h5' exists in: {os.path.abspath(MODEL_PATH)}\n\n"
                    f"Current directory: {os.getcwd()}"
                )
                messagebox.showerror("Model Loading Error", error_msg)
                self.model = None
                return
            
            self.model = load_model(MODEL_PATH)
            
            # Validate model output shape matches BALL_CLASSES
            try:
                output_shape = self.model.output_shape
                num_classes = output_shape[-1]
                
                if num_classes != len(BALL_CLASSES):
                    error_msg = (
                        f"Model output shape mismatch!\n"
                        f"Model outputs {num_classes} classes, "
                        f"but BALL_CLASSES has {len(BALL_CLASSES)} entries.\n\n"
                        f"Please ensure BALL_CLASSES matches your training data."
                    )
                    messagebox.showerror("Model Configuration Error", error_msg)
                    self.model = None
                    return
            except Exception as e:
                messagebox.showwarning("Model Validation", f"Could not fully validate model: {str(e)}")
            
            print(f"✓ Model loaded successfully from {MODEL_PATH}")
            if self.model is not None:
                print(f"✓ Model expects input shape: {self.model.input_shape}")
            print(f"✓ Model outputs {num_classes} classes")
            
        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Failed to load model:\n{str(e)}\n\nPlease check the model file and try again.")
            self.model = None
    
    def create_widgets(self):
        """Create UI elements"""
        # Title
        title_label = tk.Label(self.root, text="🏀 Ball Sport Detector 🏀", font=("Arial", 24, "bold"))
        title_label.pack(pady=20)
        
        # Model status
        self.status_label = tk.Label(
            self.root, 
            text="✓ Model Ready" if self.model else "✗ Model Not Loaded",
            font=("Arial", 11),
            fg="green" if self.model else "red"
        )
        self.status_label.pack(pady=5)
        
        # Upload button
        upload_btn = tk.Button(
            self.root, 
            text="📁 Upload Ball Image", 
            command=self.upload_image, 
            font=("Arial", 12), 
            bg="#4CAF50", 
            fg="white", 
            padx=20, 
            pady=10
        )
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
        self.results_text = tk.Text(results_frame, height=12, width=70, font=("Courier", 10), state=tk.DISABLED)
        self.results_text.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Progress label
        self.progress_label = tk.Label(self.root, text="", font=("Arial", 10), fg="blue")
        self.progress_label.pack(pady=5)
        
        # Clear button
        clear_btn = tk.Button(
            self.root, 
            text="🔄 Clear", 
            command=self.clear_results, 
            font=("Arial", 11), 
            bg="#2196F3", 
            fg="white", 
            padx=20, 
            pady=8
        )
        clear_btn.pack(pady=10)
    
    def upload_image(self):
        """Handle image upload"""
        if self.model is None:
            messagebox.showerror("Error", "Model is not loaded. Please restart the application and ensure the model file exists.")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select a ball image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*")]
        )
        
        if file_path:
            # Validate file exists and is readable
            try:
                if not os.path.isfile(file_path):
                    messagebox.showerror("Error", f"File not found: {file_path}")
                    return
                
                # Start prediction in background thread to prevent UI freezing
                thread = threading.Thread(target=self._process_image, args=(file_path,))
                thread.daemon = True
                thread.start()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process file: {str(e)}")
    
    def _process_image(self, file_path):
        """Process image in background thread"""
        try:
            self.progress_label.config(text="Processing image...", fg="blue")
            self.root.update()
            
            self.display_image(file_path)
            self.predict_ball(file_path)
            
            self.progress_label.config(text="")
            
        except Exception as e:
            self.progress_label.config(text="")
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
    
    def display_image(self, file_path):
        """Display the selected image with validation"""
        try:
            # Validate and load image
            img = Image.open(file_path)
            
            # Convert RGBA or grayscale to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Thumbnail for display
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            self.image_label.config(image=photo)
            self.image_label.config(image=photo)
            self.photo_reference = photo  # Store a reference to prevent garbage collection
            
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed to load image: {str(e)}")
    
    def predict_ball(self, file_path):
        """Predict the ball type with proper preprocessing"""
        try:
            # Load image with target size matching training
            img = image.load_img(file_path, target_size=IMAGE_SIZE)
            img_array = image.img_to_array(img)
            
            # Preprocessing: match training preprocessing (rescale=1./255)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize to [0, 1] like training
            
            # Validate preprocessing
            if np.isnan(img_array).any():
                raise ValueError("Image preprocessing resulted in NaN values")
            
            # Make prediction with low verbosity
            if self.model is None:
                raise ValueError("Model is not loaded. Please ensure the model file exists and is loaded correctly.")
            predictions = self.model.predict(img_array, verbose=0)
            
            # Validate predictions
            if predictions is None or len(predictions) == 0:
                raise ValueError("Model returned no predictions")
            
            prediction_scores = predictions[0]
            predicted_class = np.argmax(prediction_scores)
            confidence = prediction_scores[predicted_class]
            
            # Display results in main thread
            self.root.after(0, self.display_results, prediction_scores, predicted_class, confidence)
            
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Prediction Error", f"Failed to predict: {str(e)}")
    
    def display_results(self, predictions, predicted_class, confidence):
        """Display prediction results with confidence thresholds"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        # Validate predicted class
        if predicted_class >= len(BALL_CLASSES):
            self.results_text.insert(1.0, "ERROR: Invalid class index from model")
            self.results_text.config(state=tk.DISABLED)
            return
        
        # Top prediction
        result_text = f"🎯 TOP PREDICTION:\n"
        result_text += f"Ball Type: {BALL_CLASSES[predicted_class]}\n"
        result_text += f"Confidence: {confidence*100:.2f}%\n"
        result_text += f"\n{'='*60}\n"
        result_text += f"ALL PREDICTIONS (sorted by confidence):\n"
        result_text += f"{'='*60}\n\n"
        
        # Sort all predictions by confidence (descending)
        sorted_indices = np.argsort(predictions)[::-1]
        
        for rank, idx in enumerate(sorted_indices, 1):
            if idx >= len(BALL_CLASSES):
                continue
                
            ball_type = BALL_CLASSES[idx]
            confidence_pct = predictions[idx] * 100
            bar_length = int(confidence_pct / 5)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            
            # Highlight top prediction
            if rank == 1:
                result_text += f"→ {rank}. {ball_type:<20} {confidence_pct:6.2f}% [{bar}]\n"
            else:
                result_text += f"  {rank}. {ball_type:<20} {confidence_pct:6.2f}% [{bar}]\n"
        
        result_text += f"\n{'='*60}\n"
        result_text += f"Note: Model confidence varies by training data quality"
        
        self.results_text.insert(1.0, result_text)
        self.results_text.config(state=tk.DISABLED)
    
    def clear_results(self):
        """Clear all results and image"""
        self.image_label.config(image='')
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
        self.progress_label.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    app = BallDetectorApp(root)
    root.mainloop()
