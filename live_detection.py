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

IMAGE_SIZE = (192, 192)  # Must match training size
MODEL_PATH = 'Ball_sport_classifier.h5'
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to display prediction
DETECTION_WINDOW_SIZE = 192  # Size of sliding window
DETECTION_STEP_SIZE = 96  # Step size for sliding window (50% overlap)
MIN_DETECTION_CONFIDENCE = 0.4  # Minimum confidence for a detection to be considered


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
        self.detected_balls = []  # List of detected balls with bounding boxes
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
            self.progress_label.config(text="⏳ Loading image...", fg="blue")
            self.root.update()
            
            # First display image without annotations
            self.display_image(file_path)
            
            self.progress_label.config(text="⏳ Scanning for balls (this may take a moment)...", fg="blue")
            self.root.update()
            
            # Then detect multiple balls and update with annotations
            self.detect_multiple_balls(file_path)
            
            self.progress_label.config(text=f"✓ Detection Complete! Found {len(self.detected_balls)} ball(s)", fg="green")
            
        except Exception as e:
            self.progress_label.config(text="")
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
    
    def display_image(self, file_path, predictions=None, predicted_class=None, confidence=None, detected_balls=None):
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
            if detected_balls is not None and len(detected_balls) > 0:
                img_copy = self.annotate_multiple_balls(img_copy, detected_balls)
            elif predictions is not None and predicted_class is not None:
                img_copy = self.annotate_image(img_copy, predicted_class, confidence)
            
            # Thumbnail for display (keep aspect ratio) - larger size for better visibility
            display_size = (800, 800)
            img_copy.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img_copy)
            
            self.image_label.config(image=photo)
            self.image_reference = photo  # Store reference to prevent garbage collection
            
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed to load image: {str(e)}")
    
    def annotate_multiple_balls(self, img, detected_balls):
        """Add bounding boxes and labels for multiple detected balls"""
        try:
            draw = ImageDraw.Draw(img)
            width, height = img.size
            
            # Try to load fonts
            try:
                font_large = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size=max(20, int(height * 0.03)))
                font_medium = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size=max(16, int(height * 0.025)))
            except:
                font_large = ImageFont.load_default()
                font_medium = ImageFont.load_default()
            
            # Define colors (RGB format for PIL)
            colors = {
                'basketball': (255, 140, 0),      # Orange
                'billiard_ball': (255, 255, 255), # White
                'bowling_ball': (0, 0, 0),        # Black
                'football': (139, 69, 19),        # Brown
                'tennis_ball': (0, 255, 0),       # Green
                'volleyball': (255, 255, 0)       # Yellow
            }
            
            # Draw each detection
            for idx, detection in enumerate(detected_balls):
                x_min, y_min, x_max, y_max = detection['bbox']
                ball_class = detection['class']
                confidence = detection['confidence']
                ball_name = BALL_CLASSES[ball_class]
                ball_color = colors.get(ball_name, (255, 0, 0))
                
                # Draw bounding box
                box_width = 3
                draw.rectangle([x_min, y_min, x_max, y_max], outline=ball_color, width=box_width)
                
                # Draw label background
                label_text = f"{ball_name.upper()}: {confidence*100:.1f}%"
                bbox = draw.textbbox((0, 0), label_text, font=font_medium)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                label_y = max(0, y_min - text_height - 5)
                label_x = x_min
                
                # Draw semi-transparent background for label
                overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle([label_x, label_y, label_x + text_width + 10, label_y + text_height + 5], 
                                      fill=(0, 0, 0, 200))
                img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(img)
                
                # Draw label text
                draw.text((label_x + 5, label_y + 2), label_text, font=font_medium, fill=ball_color)
            
            # Draw summary at top
            summary_text = f"Detected {len(detected_balls)} ball(s)"
            bbox = draw.textbbox((0, 0), summary_text, font=font_large)
            summary_width = bbox[2] - bbox[0]
            summary_x = (width - summary_width) // 2
            summary_y = 10
            
            # Draw summary background
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle([summary_x - 10, summary_y - 5, summary_x + summary_width + 10, summary_y + bbox[3] - bbox[1] + 5], 
                                  fill=(0, 0, 0, 220))
            img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(img)
            
            # Draw summary text
            for adj in range(-2, 3):
                for adj2 in range(-2, 3):
                    draw.text((summary_x + adj, summary_y + adj2), summary_text, 
                             font=font_large, fill=(0, 0, 0))
            draw.text((summary_x, summary_y), summary_text, font=font_large, fill=(255, 255, 255))
            
            return img
            
        except Exception as e:
            print(f"Annotation error: {str(e)}")
            return img
    
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
    
    def detect_multiple_balls(self, file_path):
        """Detect multiple balls in an image using sliding window approach"""
        try:
            if self.model is None:
                raise ValueError("Model is not loaded. Please ensure the model file exists and is loaded correctly.")
            
            # Load full resolution image
            original_img = Image.open(file_path)
            if original_img.mode != 'RGB':
                original_img = original_img.convert('RGB')
            
            img_array_full = np.array(original_img)
            height, width = img_array_full.shape[:2]
            
            # Scale factors for different detection scales
            scales = [1.0, 0.75, 0.5, 1.25]  # Multiple scales to detect different sized balls
            all_detections = []
            
            print(f"🔍 Scanning image for balls ({width}x{height})...")
            
            for scale in scales:
                scaled_width = int(width * scale)
                scaled_height = int(height * scale)
                scaled_img = original_img.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
                scaled_array = np.array(scaled_img)
                
                # Sliding window detection
                for y in range(0, scaled_height - DETECTION_WINDOW_SIZE + 1, DETECTION_STEP_SIZE):
                    for x in range(0, scaled_width - DETECTION_WINDOW_SIZE + 1, DETECTION_STEP_SIZE):
                        # Extract window
                        window = scaled_array[y:y+DETECTION_WINDOW_SIZE, x:x+DETECTION_WINDOW_SIZE]
                        
                        # Preprocess window
                        window_img = Image.fromarray(window)
                        window_array = image.img_to_array(window_img)
                        window_array = np.expand_dims(window_array, axis=0)
                        window_array = window_array / 255.0
                        
                        # Predict
                        predictions = self.model.predict(window_array, verbose=0)
                        prediction_scores = predictions[0]
                        predicted_class = np.argmax(prediction_scores)
                        confidence = prediction_scores[predicted_class]
                        
                        # If confidence is high enough, record detection
                        if confidence >= MIN_DETECTION_CONFIDENCE:
                            # Convert back to original image coordinates
                            orig_x = int(x / scale)
                            orig_y = int(y / scale)
                            orig_size = int(DETECTION_WINDOW_SIZE / scale)
                            
                            all_detections.append({
                                'class': predicted_class,
                                'confidence': confidence,
                                'bbox': (orig_x, orig_y, orig_x + orig_size, orig_y + orig_size),
                                'all_scores': prediction_scores
                            })
            
            # Non-maximum suppression to remove overlapping detections
            detected_balls = self.non_max_suppression(all_detections, overlap_threshold=0.3)
            
            # Store detections
            self.detected_balls = detected_balls
            self.current_image_path = file_path
            
            # Update display
            self.root.after(0, self.update_image_with_multiple_detections, file_path, detected_balls)
            self.root.after(0, self.display_multiple_results, detected_balls)
            
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Detection Error", f"Failed to detect balls: {str(e)}")
    
    def non_max_suppression(self, detections, overlap_threshold=0.3):
        """Remove overlapping detections, keeping the one with highest confidence"""
        if not detections:
            return []
        
        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        for det in detections:
            # Check overlap with existing detections
            overlap = False
            for existing in filtered:
                if self.calculate_iou(det['bbox'], existing['bbox']) > overlap_threshold:
                    overlap = True
                    break
            
            if not overlap:
                filtered.append(det)
        
        return filtered
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def predict_ball(self, file_path):
        """Predict the ball type with proper preprocessing (legacy single detection)"""
        # Use multi-ball detection instead
        self.detect_multiple_balls(file_path)
    
    def update_image_with_detection(self, file_path, predictions, predicted_class, confidence):
        """Update the displayed image with detection annotations"""
        try:
            self.display_image(file_path, predictions, predicted_class, confidence)
        except Exception as e:
            print(f"Error updating image: {str(e)}")
    
    def update_image_with_multiple_detections(self, file_path, detected_balls):
        """Update the displayed image with multiple detection annotations"""
        try:
            self.display_image(file_path, detected_balls=detected_balls)
        except Exception as e:
            print(f"Error updating image: {str(e)}")
    
    def display_multiple_results(self, detected_balls):
        """Display results for multiple detected balls"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        if not detected_balls:
            result_text = "⚠️  NO BALLS DETECTED\n"
            result_text += f"{'═'*60}\n"
            result_text += "No balls were detected in this image.\n"
            result_text += "Try:\n"
            result_text += "  - Using a clearer image\n"
            result_text += "  - Ensuring balls are visible\n"
            result_text += "  - Checking image quality"
            self.results_text.insert(1.0, result_text)
            self.results_text.config(state=tk.DISABLED)
            return
        
        # Group detections by ball type
        ball_counts = {}
        for detection in detected_balls:
            ball_name = BALL_CLASSES[detection['class']]
            if ball_name not in ball_counts:
                ball_counts[ball_name] = []
            ball_counts[ball_name].append(detection)
        
        # Build results text
        result_text = f"🎯 DETECTION RESULTS: {len(detected_balls)} Ball(s) Found\n"
        result_text += f"{'═'*60}\n\n"
        
        # Summary by ball type
        result_text += "📊 SUMMARY BY BALL TYPE:\n"
        result_text += f"{'━'*60}\n"
        for ball_name, detections in sorted(ball_counts.items()):
            avg_confidence = np.mean([d['confidence'] for d in detections]) * 100
            result_text += f"  • {ball_name.upper()}: {len(detections)} detected (avg: {avg_confidence:.1f}%)\n"
        
        result_text += f"\n{'═'*60}\n"
        result_text += "📍 DETAILED DETECTIONS:\n"
        result_text += f"{'═'*60}\n\n"
        
        # List all detections
        for idx, detection in enumerate(detected_balls, 1):
            ball_name = BALL_CLASSES[detection['class']]
            confidence = detection['confidence'] * 100
            x_min, y_min, x_max, y_max = detection['bbox']
            
            result_text += f"Detection #{idx}:\n"
            result_text += f"  Ball Type: {ball_name.upper()}\n"
            result_text += f"  Confidence: {confidence:.2f}%\n"
            result_text += f"  Location: ({x_min}, {y_min}) to ({x_max}, {y_max})\n"
            result_text += f"  Size: {x_max-x_min}x{y_max-y_min} pixels\n"
            result_text += f"{'─'*60}\n\n"
        
        result_text += f"{'═'*60}\n"
        result_text += "💡 Tip: Bounding boxes are drawn around each detected ball"
        
        self.results_text.insert(1.0, result_text)
        self.results_text.config(state=tk.DISABLED)
    
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
        self.detected_balls = []
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
