import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os

# ============ CONFIGURATION ============
BALL_CLASSES = [
    'basketball',
    'billiard_ball',
    'bowling_ball',
    'football',
    'tennis_ball',
    'volleyball'
]

IMAGE_SIZE = (192, 192)
MODEL_PATH = 'Ball_sport_classifier.h5'
CONFIDENCE_THRESHOLD = 0.5  # Increased to reduce false detections
WINDOW_SIZE = 192
STEP_SIZE = 128  # Increased step size for fewer overlapping windows

# Colors for bounding boxes (RGB)
COLORS = {
    'basketball': (255, 140, 0),
    'billiard_ball': (255, 255, 255),
    'bowling_ball': (50, 50, 50),
    'football': (139, 69, 19),
    'tennis_ball': (0, 255, 0),
    'volleyball': (255, 255, 0)
}


class SimpleBallDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Ball Sport Detector")
        self.root.geometry("1200x900")
        self.root.config(bg='#2c3e50')
        
        self.model = None
        self.current_image = None
        self.detections = []
        
        self.load_model()
        self.create_ui()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if not os.path.exists(MODEL_PATH):
                error_msg = f"Model file not found: {MODEL_PATH}\nCurrent directory: {os.getcwd()}"
                print(error_msg)
                messagebox.showerror("Error", error_msg)
                self.model = None
                return
            
            print(f"Loading model from: {os.path.abspath(MODEL_PATH)}")
            self.model = load_model(MODEL_PATH, compile=False)
            
            # Recompile the model
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print(f"✓ Model loaded successfully")
            print(f"  Input shape: {self.model.input_shape}")
            print(f"  Output shape: {self.model.output_shape}")
            
        except Exception as e:
            error_msg = f"Failed to load model:\n{str(e)}\n\nFull path: {os.path.abspath(MODEL_PATH)}"
            print(error_msg)
            messagebox.showerror("Error", error_msg)
            self.model = None
    
    def create_ui(self):
        """Create the user interface"""
        # Title
        title = tk.Label(
            self.root,
            text="🏀 Ball Sport Detector",
            font=("Arial", 28, "bold"),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        title.pack(pady=20)
        
        # Upload button
        upload_btn = tk.Button(
            self.root,
            text="📁 Upload Image",
            command=self.upload_image,
            font=("Arial", 14, "bold"),
            bg="#27ae60",
            fg="white",
            padx=30,
            pady=15,
            cursor="hand2"
        )
        upload_btn.pack(pady=10)
        
        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Upload an image to detect balls",
            font=("Arial", 12),
            bg='#2c3e50',
            fg='#95a5a6'
        )
        self.status_label.pack(pady=5)
        
        # Image canvas with scrollbar
        canvas_frame = tk.Frame(self.root, bg='#34495e')
        canvas_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        # Scrollbars
        v_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        h_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        
        self.canvas = tk.Canvas(
            canvas_frame,
            bg='#34495e',
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set
        )
        
        v_scrollbar.config(command=self.canvas.yview)
        h_scrollbar.config(command=self.canvas.xview)
        
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Results text
        results_frame = tk.Frame(self.root, bg='#2c3e50')
        results_frame.pack(pady=10, padx=20, fill=tk.X)
        
        self.results_label = tk.Label(
            results_frame,
            text="",
            font=("Arial", 14, "bold"),
            bg='#2c3e50',
            fg='#ecf0f1',
            justify=tk.LEFT
        )
        self.results_label.pack()
    
    def upload_image(self):
        """Handle image upload and detection"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select Ball Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self.status_label.config(text="🔍 Detecting balls...", fg='#3498db')
            self.root.update()
            
            # Load and detect
            self.current_image = Image.open(file_path).convert('RGB')
            self.detections = self.detect_balls(self.current_image)
            
            # Draw detections
            annotated_image = self.draw_detections(self.current_image.copy(), self.detections)
            
            # Display image at full size
            self.display_image(annotated_image)
            
            # Show results
            self.show_results()
            
            self.status_label.config(
                text=f"✓ Found {len(self.detections)} ball(s)",
                fg='#27ae60'
            )
        
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed:\n{str(e)}")
            self.status_label.config(text="❌ Detection failed", fg='#e74c3c')
    
    def detect_balls(self, image):
        """Detect balls using sliding window with improved NMS"""
        if self.model is None:
            raise ValueError("Model is not loaded. Cannot perform detection.")
        
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        detections = []
        
        # Only use single scale for cleaner detection
        scale = 1.0
        scaled_array = img_array
        
        # Slide window with larger steps to reduce overlaps
        for y in range(0, height - WINDOW_SIZE + 1, STEP_SIZE):
            for x in range(0, width - WINDOW_SIZE + 1, STEP_SIZE):
                window = scaled_array[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
                
                # Predict
                window_normalized = window / 255.0
                window_batch = np.expand_dims(window_normalized, axis=0)
                predictions = self.model.predict(window_batch, verbose=0)[0]
                
                predicted_class = np.argmax(predictions)
                confidence = predictions[predicted_class]
                
                if confidence >= CONFIDENCE_THRESHOLD:
                    detections.append({
                        'class': predicted_class,
                        'confidence': confidence,
                        'bbox': (x, y, x + WINDOW_SIZE, y + WINDOW_SIZE)
                    })
        
        # Apply stricter NMS to remove overlaps
        return self.remove_overlaps(detections, iou_threshold=0.5)
    
    def remove_overlaps(self, detections, iou_threshold=0.5):
        """Remove overlapping detections using NMS with stricter threshold"""
        if not detections:
            return []
        
        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        
        for det in detections:
            # Check if this detection overlaps with any already accepted detection
            overlap = False
            for existing in filtered:
                iou = self.calculate_iou(det['bbox'], existing['bbox'])
                
                # If significant overlap, only keep if substantially higher confidence
                if iou > iou_threshold:
                    overlap = True
                    break
                # Also remove if boxes are very close (even with low IoU)
                elif iou > 0.2 and det['class'] == existing['class']:
                    if det['confidence'] < existing['confidence'] * 1.1:
                        overlap = True
                        break
            
            if not overlap:
                filtered.append(det)
        
        return filtered
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes on image"""
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        # Load font
        try:
            font_size = max(24, int(height * 0.025))
            font = ImageFont.truetype("arial.ttf", size=font_size)
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size=font_size)
            except:
                font = ImageFont.load_default()
        
        # Draw each detection
        for idx, det in enumerate(detections):
            x_min, y_min, x_max, y_max = det['bbox']
            ball_class = det['class']
            confidence = det['confidence']
            ball_name = BALL_CLASSES[ball_class]
            color = COLORS.get(ball_name, (255, 0, 0))
            
            # Draw box with thicker lines
            for offset in range(5):
                draw.rectangle(
                    [x_min-offset, y_min-offset, x_max+offset, y_max+offset], 
                    outline=color, 
                    width=1
                )
            
            # Draw label
            label = f"{ball_name}: {confidence*100:.0f}%"
            
            # Get text size
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Position label above box
            label_y = max(0, y_min - text_height - 15)
            label_x = x_min
            
            # Draw black background for text
            padding = 8
            draw.rectangle(
                [label_x - padding, label_y - padding, 
                 label_x + text_width + padding, label_y + text_height + padding],
                fill=(0, 0, 0)
            )
            
            # Draw text
            draw.text((label_x, label_y), label, font=font, fill=color)
        
        return image
    
    def display_image(self, image):
        """Display image at full size on canvas"""
        # Create PhotoImage
        photo = ImageTk.PhotoImage(image)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo  # Keep reference
        
        # Update scroll region
        self.canvas.config(scrollregion=(0, 0, image.width, image.height))
    
    def show_results(self):
        """Display detection results"""
        if not self.detections:
            self.results_label.config(text="No balls detected")
            return
        
        # Count by type
        counts = {}
        for det in self.detections:
            ball_name = BALL_CLASSES[det['class']]
            counts[ball_name] = counts.get(ball_name, 0) + 1
        
        # Format results
        result_text = f"Detected {len(self.detections)} ball(s): "
        result_text += ", ".join([f"{count}x {ball}" for ball, count in sorted(counts.items())])
        
        self.results_label.config(text=result_text)


def main():
    root = tk.Tk()
    app = SimpleBallDetector(root)
    root.mainloop()


if __name__ == "__main__":
    main()