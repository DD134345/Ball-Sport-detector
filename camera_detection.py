import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os
from datetime import datetime
from collections import deque
import threading
import time

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
CONFIDENCE_THRESHOLD = 0.35  # Minimum confidence to display prediction (lowered for better detection)

# Colors for display (BGR format for OpenCV)
CLASS_COLORS = {
    'basketball': (0, 165, 255),      # Orange
    'billiard_ball': (255, 255, 255), # White
    'bowling_ball': (0, 0, 0),        # Black
    'football': (0, 165, 255),        # Orange
    'tennis_ball': (0, 255, 0),       # Green
    'volleyball': (255, 255, 0)       # Cyan
}

class CameraBallDetector:
    def __init__(self, camera_id=0):
        """Initialize camera detector with model"""
        self.camera_id = camera_id
        self.model = None
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.predictions_history = deque(maxlen=15)  # Keep last 15 predictions for better smoothing
        self.frame_skip = 2  # Process every 2nd frame for faster performance
        self.frame_counter = 0
        self.is_running = True
        self.enhance_image = True  # Enable image enhancement for better detection
        self.detection_count = 0  # Track number of successful detections
        
        print("🔄 Initializing Ball Detector...")
        self.load_model()
        self.initialize_camera()
    
    def load_model(self):
        """Load the trained model with error handling"""
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(
                    f"Model file not found: {MODEL_PATH}\n"
                    f"Please ensure '{MODEL_PATH}' exists in: {os.path.abspath(MODEL_PATH)}"
                )
            
            self.model = load_model(MODEL_PATH)
            
            # Validate model output shape matches BALL_CLASSES
            output_shape = self.model.output_shape
            num_classes = output_shape[-1]
            
            if num_classes != len(BALL_CLASSES):
                raise ValueError(
                    f"Model output shape mismatch!\n"
                    f"Model outputs {num_classes} classes, "
                    f"but BALL_CLASSES has {len(BALL_CLASSES)} entries."
                )
            
            print(f"✓ Model loaded successfully from {MODEL_PATH}")
            print(f"✓ Model expects input shape: {self.model.input_shape}")
            print(f"✓ Model outputs {num_classes} classes")
            
        except Exception as e:
            print(f"✗ Failed to load model: {str(e)}")
            raise
    
    def initialize_camera(self):
        """Initialize webcam connection"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_id}")
            
            # Set camera properties for optimal performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
            
            print(f"✓ Camera {self.camera_id} initialized successfully")
            print(f"✓ Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            print(f"✓ FPS: {self.cap.get(cv2.CAP_PROP_FPS)}\n")
            
        except Exception as e:
            print(f"✗ Failed to initialize camera: {str(e)}")
            raise
    
    def enhance_frame(self, frame):
        """Enhance frame for better ball detection"""
        if not self.enhance_image:
            return frame
        
        try:
            # Convert to LAB color space for better enhancement
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Slight sharpening for better edge detection
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel * 0.1)
            
            # Blend original and sharpened
            enhanced = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
            
            return enhanced
        except Exception as e:
            # If enhancement fails, return original frame
            return frame
    
    def predict_frame(self, frame):
        """Predict ball type from a frame with optimized preprocessing and enhancement"""
        try:
            # Enhance frame for better detection
            enhanced_frame = self.enhance_frame(frame)
            
            # Preprocess frame for model
            # Resize to match training size
            resized_frame = cv2.resize(enhanced_frame, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
            
            # Convert BGR to RGB (model expects RGB)
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1] range (matching training preprocessing)
            img_array = np.array(rgb_frame, dtype=np.float32) / 255.0
            
            # Validate normalization
            if np.isnan(img_array).any() or np.isinf(img_array).any():
                raise ValueError("Invalid pixel values after normalization")
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction with lower verbosity
            if self.model is None:
                raise RuntimeError("Model is not loaded.")
            
            predictions = self.model.predict(img_array, verbose=0)
            prediction_scores = predictions[0]
            
            # Validate predictions
            if np.isnan(prediction_scores).any() or np.isinf(prediction_scores).any():
                raise ValueError("Model returned invalid predictions")
            
            # Get top prediction
            predicted_class = np.argmax(prediction_scores)
            confidence = prediction_scores[predicted_class]
            
            # Store in history for smoothing
            self.predictions_history.append({
                'class': predicted_class,
                'confidence': confidence,
                'all_scores': prediction_scores.copy()
            })
            
            # Increment detection count if confidence is above threshold
            if confidence >= CONFIDENCE_THRESHOLD:
                self.detection_count += 1
            
            return predicted_class, confidence, prediction_scores
            
        except Exception as e:
            print(f"✗ Prediction error: {str(e)}")
            return None, 0, None
    
    def get_smoothed_prediction(self):
        """Get smoothed prediction from history (reduces flickering) with improved algorithm"""
        if not self.predictions_history:
            return None, 0, None
        
        # Use weighted average based on recency (more recent = higher weight)
        classes = [p['class'] for p in self.predictions_history]
        confidences = [p['confidence'] for p in self.predictions_history]
        
        # Weight recent predictions more heavily
        weights = np.linspace(0.5, 1.0, len(self.predictions_history))
        
        # Get most common class from recent predictions (with weights)
        class_votes = {}
        for i, (cls, conf, weight) in enumerate(zip(classes, confidences, weights)):
            if cls not in class_votes:
                class_votes[cls] = 0
            # Vote strength = confidence * weight
            class_votes[cls] += conf * weight
        
        # Get class with highest weighted votes
        most_common_class = max(class_votes, key=lambda x: class_votes[x])
        
        # Weighted average confidence for that class
        weighted_confidences = [c * w for cls, c, w in zip(classes, confidences, weights) 
                               if cls == most_common_class]
        weights_for_class = [w for cls, w in zip(classes, weights) 
                           if cls == most_common_class]
        
        if weighted_confidences:
            avg_confidence = np.sum(weighted_confidences) / np.sum(weights_for_class)
        else:
            avg_confidence = confidences[-1]
        
        # Get latest scores
        latest_scores = self.predictions_history[-1]['all_scores']
        
        return most_common_class, avg_confidence, latest_scores
    
    def draw_predictions(self, frame, predicted_class, confidence, all_scores):
        """Draw predictions on frame with optimized rendering"""
        height, width = frame.shape[:2]
        
        # Semi-transparent background for text regions
        overlay = frame.copy()
        
        # Display title with background
        cv2.rectangle(overlay, (5, 5), (width - 5, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        cv2.putText(frame, "Ball Sport Detector - Real-time Detection", 
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", 
                    (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if predicted_class is not None and confidence >= CONFIDENCE_THRESHOLD:
            ball_name = BALL_CLASSES[predicted_class]
            color = CLASS_COLORS.get(ball_name, (0, 255, 0))
            
            # Draw main prediction box with semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 70), (width - 10, 200), color, -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            
            # Draw border with thicker line for visibility
            cv2.rectangle(frame, (10, 70), (width - 10, 200), color, 3)
            
            # Draw detection indicator
            cv2.circle(frame, (30, 90), 8, color, -1)
            cv2.putText(frame, "BALL DETECTED!", 
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw ball type with larger, bolder text
            cv2.putText(frame, f"Type: {ball_name.upper()}", 
                        (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(frame, f"Type: {ball_name.upper()}", 
                        (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            
            # Draw detection count
            cv2.putText(frame, f"Detections: {self.detection_count}", 
                        (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw confidence score with bar
            confidence_pct = confidence * 100
            bar_width = int((confidence_pct / 100) * 350)
            cv2.rectangle(frame, (10, 210), (360, 240), (200, 200, 200), 2)
            cv2.rectangle(frame, (10, 210), (10 + bar_width, 240), color, -1)
            cv2.putText(frame, f"Confidence: {confidence_pct:.1f}%", 
                        (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw top 3 predictions
            y_offset = 260
            cv2.putText(frame, "Top Predictions:", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            sorted_indices = np.argsort(all_scores)[::-1]
            
            for rank, idx in enumerate(sorted_indices[:3], 1):  # Show top 3
                ball_type = BALL_CLASSES[idx]
                score = all_scores[idx] * 100
                
                y_offset += 35
                prefix = "→ " if idx == predicted_class else "  "
                text = f"{prefix}{rank}. {ball_type}: {score:.1f}%"
                
                text_color = color if idx == predicted_class else (200, 200, 200)
                cv2.putText(frame, text, 
                            (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        else:
            # No confident prediction - show helpful message
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 70), (width - 10, 180), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.rectangle(frame, (10, 70), (width - 10, 180), (0, 0, 255), 3)
            
            cv2.putText(frame, "SEARCHING FOR BALL...", 
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, "Point camera at a ball", 
                        (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            # Show top prediction even if below threshold
            if predicted_class is not None and all_scores is not None:
                top_class_idx = np.argmax(all_scores)
                top_confidence = all_scores[top_class_idx] * 100
                top_ball_name = BALL_CLASSES[top_class_idx]
                cv2.putText(frame, f"Best guess: {top_ball_name} ({top_confidence:.1f}%)", 
                            (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 255), 1)
        
        # Draw instructions at bottom
        instruction_text = "Q:Quit | S:Save | R:Reset | SPACE:Pause | E:Enhance | C:Stats"
        cv2.putText(frame, instruction_text, 
                    (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Run real-time detection loop with optimizations"""
        print("\n" + "="*70)
        print("🎥 REAL-TIME BALL DETECTION STARTED")
        print("="*70)
        print("\nControls:")
        print("  'Q' - Quit application")
        print("  'S' - Save screenshot")
        print("  'R' - Reset prediction history")
        print("  'SPACE' - Pause/Resume detection")
        print("  'E' - Toggle image enhancement")
        print("  'C' - Show detection statistics")
        print("\n" + "="*70 + "\n")
        print("💡 TIP: Point the camera at a ball (basketball, football, tennis ball, etc.)")
        print("   The system will detect and identify the ball type in real-time!\n")
        
        prev_time = time.time()
        frame_count = 0
        is_paused = False
        
        while self.is_running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    print("✗ Camera is not initialized or failed to open")
                    break
                
                ret, frame = self.cap.read()
                
                if not ret:
                    print("✗ Failed to read frame from camera")
                    break
                
                # Calculate FPS
                current_time = time.time()
                frame_count += 1
                if current_time - prev_time >= 1.0:
                    self.fps = frame_count / (current_time - prev_time)
                    frame_count = 0
                    prev_time = current_time
                
                # Process every Nth frame
                self.frame_counter += 1
                if self.frame_counter % self.frame_skip == 0:
                    predicted_class, confidence, all_scores = self.predict_frame(frame)
                    
                    if not is_paused:
                        # Get smoothed prediction
                        smoothed_class, smoothed_conf, smoothed_scores = self.get_smoothed_prediction()
                        if smoothed_class is not None:
                            predicted_class, confidence, all_scores = smoothed_class, smoothed_conf, smoothed_scores
                else:
                    predicted_class, confidence, all_scores = None, 0, None
                
                # Draw results
                if predicted_class is not None:
                    frame = self.draw_predictions(frame, predicted_class, confidence, all_scores)
                else:
                    frame = self.draw_predictions(frame, None, 0, None)
                
                # Display frame
                cv2.imshow('Ball Sport Detector', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\n✓ Closing application...")
                    break
                elif key == ord('s') or key == ord('S'):
                    # Save screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"ball_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"✓ Screenshot saved: {filename}")
                elif key == ord('r') or key == ord('R'):
                    # Reset prediction history and detection count
                    self.predictions_history.clear()
                    self.detection_count = 0
                    print("✓ Prediction history and detection count reset")
                elif key == ord(' '):
                    # Pause/Resume
                    is_paused = not is_paused
                    status = "PAUSED" if is_paused else "RUNNING"
                    print(f"✓ Detection {status}")
                elif key == ord('e') or key == ord('E'):
                    # Toggle image enhancement
                    self.enhance_image = not self.enhance_image
                    status = "ENABLED" if self.enhance_image else "DISABLED"
                    print(f"✓ Image enhancement {status}")
                elif key == ord('c') or key == ord('C'):
                    # Print statistics
                    print(f"\n📊 Detection Statistics:")
                    print(f"  Total Detections: {self.detection_count}")
                    print(f"  FPS: {self.fps:.1f}")
                    print(f"  Image Enhancement: {'ON' if self.enhance_image else 'OFF'}")
                    if predicted_class is not None:
                        print(f"  Current Ball: {BALL_CLASSES[predicted_class]}")
                        print(f"  Confidence: {confidence*100:.2f}%")
                        if all_scores is not None:
                            print(f"  Top 3 Predictions:")
                            sorted_indices = np.argsort(all_scores)[::-1]
                            for rank, idx in enumerate(sorted_indices[:3], 1):
                                print(f"    {rank}. {BALL_CLASSES[idx]}: {all_scores[idx]*100:.2f}%")
                    print()
            
            except KeyboardInterrupt:
                print("\n\n✓ Detection interrupted by user")
                break
            except Exception as e:
                print(f"✗ Error during detection: {str(e)}")
                break
        
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.is_running = False
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("✓ Resources cleaned up successfully")
        except Exception as e:
            print(f"✗ Error during cleanup: {str(e)}")


def main():
    """Main entry point"""
    try:
        print("\n" + "="*70)
        print("🏀 BALL SPORT DETECTOR - LIVE CAMERA DETECTION 🏀")
        print("="*70 + "\n")
        
        # Try to use camera 0 (default webcam)
        detector = CameraBallDetector(camera_id=0)
        detector.run()
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {str(e)}")
        print(f"\nPlease make sure:")
        print(f"  1. The model file '{MODEL_PATH}' is in the same directory as this script")
        print(f"  2. You have trained the model using training_model.py")
    except RuntimeError as e:
        print(f"\n✗ Camera Error: {str(e)}")
        print(f"\nPlease make sure:")
        print(f"  1. Your webcam is connected and working")
        print(f"  2. No other application is using the camera")
        print(f"  3. Camera drivers are properly installed")
    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
