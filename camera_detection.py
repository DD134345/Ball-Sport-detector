import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os
from datetime import datetime
from collections import deque

# Ball sport classes (must match training_model.py)
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
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to display prediction

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
        self.predictions_history = deque(maxlen=5)  # Keep last 5 predictions for smoothing
        
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
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"✓ Camera {self.camera_id} initialized successfully")
            print(f"✓ Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            
        except Exception as e:
            print(f"✗ Failed to initialize camera: {str(e)}")
            raise
    
    def predict_frame(self, frame):
        """Predict ball type from a frame"""
        try:
            # Preprocess frame for model
            # Resize to match training size
            resized_frame = cv2.resize(frame, IMAGE_SIZE)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            # Normalize
            img_array = np.array(rgb_frame, dtype=np.float32) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            if self.model is None:
                raise RuntimeError("Model is not loaded. Ensure 'load_model' is called successfully.")
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
            return None, 0
        
        # Use most common prediction from recent frames
        classes = [p['class'] for p in self.predictions_history]
        confidences = [p['confidence'] for p in self.predictions_history]
        
        # Get mode (most common class)
        most_common_class = max(set(classes), key=classes.count)
        avg_confidence = np.mean([c for cls, c in zip(classes, confidences) if cls == most_common_class])
        
        return most_common_class, avg_confidence
    
    def draw_predictions(self, frame, predicted_class, confidence, all_scores):
        """Draw predictions on frame"""
        height, width = frame.shape[:2]
        
        # Display title
        cv2.putText(frame, "Ball Sport Detector - Real-time Detection", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", 
                    (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if predicted_class is not None and confidence >= CONFIDENCE_THRESHOLD:
            ball_name = BALL_CLASSES[predicted_class]
            color = CLASS_COLORS.get(ball_name, (0, 255, 0))
            
            # Draw main prediction box
            cv2.rectangle(frame, (10, 60), (width - 10, 150), color, 2)
            cv2.putText(frame, "DETECTED BALL:", 
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"{ball_name.upper()}", 
                        (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            # Draw confidence score with bar
            confidence_pct = confidence * 100
            bar_width = int((confidence_pct / 100) * 300)
            cv2.rectangle(frame, (10, 160), (310, 185), (200, 200, 200), 1)
            cv2.rectangle(frame, (10, 160), (10 + bar_width, 185), color, -1)
            cv2.putText(frame, f"Confidence: {confidence_pct:.1f}%", 
                        (20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Draw all predictions
            y_offset = 240
            cv2.putText(frame, "All Predictions:", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            sorted_indices = np.argsort(all_scores)[::-1]
            
            for rank, idx in enumerate(sorted_indices[:3], 1):  # Show top 3
                ball_type = BALL_CLASSES[idx]
                score = all_scores[idx] * 100
                
                y_offset += 25
                prefix = "→ " if idx == predicted_class else "  "
                cv2.putText(frame, f"{prefix}{rank}. {ball_type}: {score:.1f}%", 
                            (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # No confident prediction
            cv2.putText(frame, "No ball detected or confidence too low", 
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw instructions
        cv2.putText(frame, "Press 'Q' to quit | 'S' to save screenshot | 'C' to capture stats", 
                    (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Run real-time detection loop"""
        print("\n" + "="*60)
        print("🎥 REAL-TIME BALL DETECTION STARTED")
        print("="*60)
        print("\nControls:")
        print("  'Q' - Quit application")
        print("  'S' - Save screenshot")
        print("  'C' - Capture prediction statistics")
        print("\nPress any key to continue...")
        print("="*60 + "\n")
        
        import time
        prev_time = time.time()
        frame_count = 0
        
        while True:
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
                
                # Make prediction
                predicted_class, confidence, all_scores = self.predict_frame(frame)
                
                # Draw results
                frame = self.draw_predictions(frame, predicted_class, confidence, all_scores)
                
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
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("✓ Resources cleaned up successfully")
        except Exception as e:
            print(f"✗ Error during cleanup: {str(e)}")


def main():
    """Main entry point"""
    try:
        print("\n🏀 Ball Sport Detector - Real-time Camera Feed 🏀\n")
        
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
