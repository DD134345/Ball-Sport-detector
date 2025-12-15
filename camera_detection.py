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
CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence to display prediction

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
        self.predictions_history = deque(maxlen=10)  # Keep last 10 predictions for smoothing
        self.frame_skip = 2  # Process every 2nd frame for faster performance
        self.frame_counter = 0
        self.is_running = True
        
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
    
    def predict_frame(self, frame):
        """Predict ball type from a frame with optimized preprocessing"""
        try:
            # Preprocess frame for model
            # Resize to match training size
            resized_frame = cv2.resize(frame, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            # Normalize
            img_array = np.array(rgb_frame, dtype=np.float32) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction with lower verbosity
            if self.model is None:
                raise RuntimeError("Model is not loaded.")
            
            predictions = self.model.predict(img_array, verbose=0)
            prediction_scores = predictions[0]
            
            # Get top prediction
            predicted_class = np.argmax(prediction_scores)
            confidence = prediction_scores[predicted_class]
            
            # Store in history for smoothing
            self.predictions_history.append({
                'class': predicted_class,
                'confidence': confidence,
                'all_scores': prediction_scores
            })
            
            return predicted_class, confidence, prediction_scores
            
        except Exception as e:
            print(f"✗ Prediction error: {str(e)}")
            return None, 0, None
    
    def get_smoothed_prediction(self):
        """Get smoothed prediction from history (reduces flickering)"""
        if not self.predictions_history:
            return None, 0, None
        
        # Use weighted average based on recency
        classes = [p['class'] for p in self.predictions_history]
        confidences = [p['confidence'] for p in self.predictions_history]
        
        # Get most common class from recent predictions
        most_common_class = max(set(classes), key=classes.count)
        
        # Average confidence for that class
        avg_confidence = np.mean([c for cls, c in zip(classes, confidences) 
                                  if cls == most_common_class])
        
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
            cv2.rectangle(overlay, (10, 70), (width - 10, 180), color, -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            
            # Draw border
            cv2.rectangle(frame, (10, 70), (width - 10, 180), color, 3)
            
            # Draw text
            cv2.putText(frame, "DETECTED BALL:", 
                        (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"{ball_name.upper()}", 
                        (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            # Draw confidence score with bar
            confidence_pct = confidence * 100
            bar_width = int((confidence_pct / 100) * 350)
            cv2.rectangle(frame, (10, 190), (360, 220), (200, 200, 200), 2)
            cv2.rectangle(frame, (10, 190), (10 + bar_width, 220), color, -1)
            cv2.putText(frame, f"Confidence: {confidence_pct:.1f}%", 
                        (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw top 3 predictions
            y_offset = 240
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
            # No confident prediction
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 70), (width - 10, 180), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.rectangle(frame, (10, 70), (width - 10, 180), (0, 0, 255), 3)
            
            cv2.putText(frame, "No ball detected or confidence too low", 
                        (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Draw instructions at bottom
        instruction_text = "Press 'Q' to quit | 'S' to save | 'R' to reset | 'SPACE' to pause"
        cv2.putText(frame, instruction_text, 
                    (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
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
        print("\n" + "="*70 + "\n")
        
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
                    # Reset prediction history
                    self.predictions_history.clear()
                    print("✓ Prediction history reset")
                elif key == ord(' '):
                    # Pause/Resume
                    is_paused = not is_paused
                    status = "PAUSED" if is_paused else "RUNNING"
                    print(f"✓ Detection {status}")
                elif key == ord('c') or key == ord('C'):
                    # Print statistics
                    if predicted_class is not None:
                        print(f"\n📊 Current Detection Statistics:")
                        print(f"  Detected Ball: {BALL_CLASSES[predicted_class]}")
                        print(f"  Confidence: {confidence*100:.2f}%")
                        print(f"  FPS: {self.fps:.1f}")
            
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
