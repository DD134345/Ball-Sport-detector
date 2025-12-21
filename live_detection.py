import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os
import threading
from pathlib import Path

# ============ CONFIGURATION ============
BALL_CLASSES = [
    'basketball',
    'billiard_ball',
    'bowling_ball',
    'football',
    'tennis_ball',
    'volleyball'
]

IMAGE_SIZE = (224, 224)  # Must match training size
MODEL_PATH = 'Ball_sport_classifier.h5'
CONFIDENCE_THRESHOLD = 0.2  # Minimum confidence to display prediction


class BallDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🏀 Ball Sport Detector")
        self.root.geometry("900x1000")
        self.root.resizable(True, True)
        self.root.config(bg='#f0f0f0')
        
        # Load model
        self.model = None
        self.is_loading = False
        self.current_image_path = None
        self.current_predictions = None
        self.load_model()
        
        # Create UI
        self.create_widgets()
    
    def load_model(self):
        """Load the trained model with error handling"""
        try:
            if not os.path.exists(MODEL_PATH):
                error_msg = (
                    f"Model file not found: {MODEL_PATH}\n\n"
                    f"Please ensure 'Ball_sport_classifier.h5' exists in: {os.path.abspath(MODEL_PATH)}\n\n"
                    f"Current directory: {os.getcwd()}\n\n"
                    f"Make sure you have trained the model using training_model.py"
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
        title_label = tk.Label(
            self.root, 
            text="🏀 Ball Sport Detector 🏀", 
            font=("Arial", 26, "bold"),
            bg='#f0f0f0',
            fg='#2196F3'
        )
        title_label.pack(pady=20)
        
        # Model status
        self.status_label = tk.Label(
            self.root, 
            text="✓ Model Ready" if self.model else "✗ Model Not Loaded",
            font=("Arial", 12, "bold"),
            fg="green" if self.model else "red",
            bg='#f0f0f0'
        )
        self.status_label.pack(pady=5)
        
        # Upload button
        upload_btn = tk.Button(
            self.root, 
            text="📁 Upload Ball Image", 
            command=self.upload_image, 
            font=("Arial", 13, "bold"), 
            bg="#4CAF50", 
            fg="white", 
            padx=25, 
            pady=12,
            cursor="hand2",
            relief=tk.RAISED,
            bd=2
        )
        upload_btn.pack(pady=15)
        
        # Image display with frame
        image_frame = tk.Frame(self.root, bg="white", relief=tk.SUNKEN, bd=2)
        image_frame.pack(pady=15, padx=20, fill=tk.BOTH, expand=True)
        
        self.image_label = tk.Label(image_frame, bg="lightgray", width=60, height=18)
        self.image_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        # Results frame
        results_frame = tk.LabelFrame(
            self.root, 
            text="Detection Results", 
            font=("Arial", 12, "bold"),
            bg="#f0f0f0",
            padx=15, 
            pady=15,
            fg='#2196F3'
        )
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Results display with scrollbar
        scrollbar = tk.Scrollbar(results_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_text = tk.Text(
            results_frame, 
            height=12, 
            width=75, 
            font=("Courier", 10),
            state=tk.DISABLED,
            yscrollcommand=scrollbar.set
        )
        self.results_text.pack(pady=5, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.results_text.yview)
        
        # Progress label
        self.progress_label = tk.Label(
            self.root, 
            text="", 
            font=("Arial", 10),
            fg="blue",
            bg='#f0f0f0'
        )
        self.progress_label.pack(pady=5)
        
        # Button frame
        button_frame = tk.Frame(self.root, bg='#f0f0f0')
        button_frame.pack(pady=15)
        
        # Clear button
        clear_btn = tk.Button(
            button_frame, 
            text="🔄 Clear", 
            command=self.clear_results, 
            font=("Arial", 11, "bold"), 
            bg="#2196F3", 
            fg="white", 
            padx=20, 
            pady=8,
            cursor="hand2",
            relief=tk.RAISED,
            bd=2
        )
        clear_btn.pack(side=tk.LEFT, padx=10)
        
        # Exit button
        exit_btn = tk.Button(
            button_frame,
            text="❌ Exit",
            command=self.root.quit,
            font=("Arial", 11, "bold"),
            bg="#f44336",
            fg="white",
            padx=20,
            pady=8,
            cursor="hand2",
            relief=tk.RAISED,
            bd=2
        )
        exit_btn.pack(side=tk.LEFT, padx=10)
    
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
            self.progress_label.config(text="⏳ Processing image...", fg="blue")
            self.root.update()
            
            # First display image without annotations
            self.display_image(file_path)
            
            # Then predict and update with annotations
            self.predict_ball(file_path)
            
            self.progress_label.config(text="✓ Detection Complete!", fg="green")
            
        except Exception as e:
            self.progress_label.config(text="")
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
    
    def display_image(self, file_path, predictions=None, predicted_class=None, confidence=None):
        """Display the selected image with detection annotations"""
        try:
            # Store current image path
            self.current_image_path = file_path
            
            # Validate and load image
            img = Image.open(file_path)
            
            # Convert RGBA or grayscale to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Create a copy for annotation
            img_copy = img.copy()
            
            # Add detection annotations if available
            if predictions is not None and predicted_class is not None:
                img_copy = self.annotate_image(img_copy, predicted_class, confidence)
            
            # Thumbnail for display (keep aspect ratio)
            display_size = (600, 600)
            img_copy.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img_copy)
            
            self.image_label.config(image=photo)
            self.image_reference = photo  # Store reference to prevent garbage collection
            
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed to load image: {str(e)}")
    
    def annotate_image(self, img, predicted_class, confidence):
        """Add detection annotations to the image"""
        try:
            # Create a drawing context
            draw = ImageDraw.Draw(img)
            
            # Get image dimensions
            width, height = img.size
            
            # Try to load a font, fallback to default if not available
            try:
                # Try to use a larger font
                font_large = ImageFont.truetype("arial.ttf", size=int(height * 0.08))
                font_medium = ImageFont.truetype("arial.ttf", size=int(height * 0.05))
            except:
                try:
                    font_large = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size=int(height * 0.08))
                    font_medium = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size=int(height * 0.05))
                except:
                    font_large = ImageFont.load_default()
                    font_medium = ImageFont.load_default()
            
            # Get ball name and color
            ball_name = BALL_CLASSES[predicted_class].upper()
            confidence_pct = confidence * 100
            
            # Define colors (RGB format for PIL)
            colors = {
                'basketball': (255, 140, 0),      # Orange
                'billiard_ball': (255, 255, 255), # White
                'bowling_ball': (0, 0, 0),        # Black
                'football': (139, 69, 19),        # Brown
                'tennis_ball': (0, 255, 0),       # Green
                'volleyball': (255, 255, 0)       # Yellow
            }
            
            ball_color = colors.get(BALL_CLASSES[predicted_class], (0, 255, 0))
            
            # Draw semi-transparent background rectangle for text
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            # Top banner
            banner_height = int(height * 0.15)
            overlay_draw.rectangle([(0, 0), (width, banner_height)], fill=(0, 0, 0, 180))
            
            # Bottom info box
            info_height = int(height * 0.12)
            overlay_draw.rectangle([(0, height - info_height), (width, height)], fill=(0, 0, 0, 180))
            
            # Blend overlay with original image
            img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(img)
            
            # Draw main detection text at top
            text_main = f"BALL DETECTED: {ball_name}"
            text_confidence = f"Confidence: {confidence_pct:.1f}%"
            
            # Get text bounding boxes for centering
            bbox_main = draw.textbbox((0, 0), text_main, font=font_large)
            text_width_main = bbox_main[2] - bbox_main[0]
            
            bbox_conf = draw.textbbox((0, 0), text_confidence, font=font_medium)
            text_width_conf = bbox_conf[2] - bbox_conf[0]
            
            # Draw text with outline for visibility
            x_main = (width - text_width_main) // 2
            y_main = int(banner_height * 0.3)
            
            # Draw outline (shadow effect)
            for adj in range(-2, 3):
                for adj2 in range(-2, 3):
                    draw.text((x_main + adj, y_main + adj2), text_main, 
                             font=font_large, fill=(0, 0, 0))
            draw.text((x_main, y_main), text_main, font=font_large, fill=ball_color)
            
            # Draw confidence
            x_conf = (width - text_width_conf) // 2
            y_conf = int(banner_height * 0.7)
            
            for adj in range(-2, 3):
                for adj2 in range(-2, 3):
                    draw.text((x_conf + adj, y_conf + adj2), text_confidence, 
                             font=font_medium, fill=(0, 0, 0))
            draw.text((x_conf, y_conf), text_confidence, font=font_medium, fill=(255, 255, 255))
            
            # Draw bottom info
            info_text = f"Detected Ball Type: {ball_name}"
            bbox_info = draw.textbbox((0, 0), info_text, font=font_medium)
            text_width_info = bbox_info[2] - bbox_info[0]
            x_info = (width - text_width_info) // 2
            y_info = height - int(info_height * 0.6)
            
            for adj in range(-1, 2):
                for adj2 in range(-1, 2):
                    draw.text((x_info + adj, y_info + adj2), info_text, 
                             font=font_medium, fill=(0, 0, 0))
            draw.text((x_info, y_info), info_text, font=font_medium, fill=(255, 255, 255))
            
            return img
            
        except Exception as e:
            print(f"Annotation error: {str(e)}")
            return img
    
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
            
            # Store predictions
            self.current_predictions = prediction_scores
            
            # Update image with annotations in main thread
            self.root.after(0, self.update_image_with_detection, file_path, prediction_scores, predicted_class, confidence)
            
            # Display results in main thread
            self.root.after(0, self.display_results, prediction_scores, predicted_class, confidence)
            
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Prediction Error", f"Failed to predict: {str(e)}")
    
    def update_image_with_detection(self, file_path, predictions, predicted_class, confidence):
        """Update the displayed image with detection annotations"""
        try:
            self.display_image(file_path, predictions, predicted_class, confidence)
        except Exception as e:
            print(f"Error updating image: {str(e)}")
    
    def display_results(self, predictions, predicted_class, confidence):
        """Display prediction results with confidence thresholds"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        # Validate predicted class
        if predicted_class >= len(BALL_CLASSES):
            self.results_text.insert(1.0, "ERROR: Invalid class index from model")
            self.results_text.config(state=tk.DISABLED)
            return
        
        # Top prediction with enhanced formatting
        result_text = f"🎯 TOP PREDICTION:\n"
        result_text += f"{'━'*60}\n"
        result_text += f"Ball Type: {BALL_CLASSES[predicted_class].upper()}\n"
        result_text += f"Confidence: {confidence*100:.2f}%\n"
        result_text += f"\n{'═'*60}\n"
        result_text += f"ALL PREDICTIONS (sorted by confidence):\n"
        result_text += f"{'═'*60}\n\n"
        
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
        
        result_text += f"\n{'═'*60}\n"
        result_text += f"Note: Higher confidence indicates more reliable detection"
        
        self.results_text.insert(1.0, result_text)
        self.results_text.config(state=tk.DISABLED)
    
    def clear_results(self):
        """Clear all results and image"""
        self.image_label.config(image='')
        self.current_image_path = None
        self.current_predictions = None
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
        self.progress_label.config(text="")


def main():
    """Main entry point"""
    try:
        root = tk.Tk()
        app = BallDetectorApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Error: {str(e)}")
        messagebox.showerror("Error", f"Application error: {str(e)}")


if __name__ == "__main__":
    main()
