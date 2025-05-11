import argparse
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time

class FingerCounter:
    def __init__(self, model_path=None):
        """
        Initialize the finger counter with YOLOv8 model
        
        Args:
            model_path: Path to custom YOLOv8 model (if None, will use YOLOv8n)
        """
        print("Initializing FingerCounter...")
        
        # Load YOLOv8 model - we'll use the pretrained model for hand detection
        # Later we can fine-tune it specifically for our use case
        if model_path:
            self.model = YOLO(model_path)
        else:
            # Use YOLOv8n by default - we'll use it to detect hands
            self.model = YOLO("yolov8n.pt")
        
        print("Model loaded successfully!")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
    def process_image(self, image_path):
        """Process an image file to count fingers"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            return
        
        result = self.count_fingers(image)
        
        # Display results
        cv2.imshow("Finger Counter", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def process_webcam(self):
        """Process webcam feed to count fingers in real-time"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
            
        print("Webcam opened successfully. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break
                
            result = self.count_fingers(frame)
            
            # Display FPS
            cv2.imshow("Finger Counter (Press 'q' to quit)", result)
            
            # Exit if 'q' pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    def count_fingers(self, image):
        """
        Count fingers in an image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Processed image with hand detection and finger count
        """
        # Make a copy to draw on
        result_img = image.copy()
        h, w, _ = image.shape
        
        # Detect objects in the image using YOLOv8
        results = self.model(image, verbose=False)[0]
        
        # Track detected hands
        detected_hands = []
        
        # Process detections
        for detection in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = detection
            
            # If detection is a person (class 0) or we detect something with high confidence
            # Note: using a more general approach as we haven't trained for hands specifically yet
            if conf > 0.4:
                # Extract the hand region
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                hand_roi = image[y1:y2, x1:x2]
                
                # Skip if ROI is empty
                if hand_roi.size == 0:
                    continue
                
                # Simple placeholder for finger counting
                # This is where we'll implement the actual finger counting algorithm
                fingers = self.count_fingers_in_roi(hand_roi)
                
                # Add to detected hands
                detected_hands.append({
                    'box': (x1, y1, x2, y2),
                    'fingers': fingers
                })
                
                # Draw bounding box
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Display finger count
                text = f"Fingers: {fingers}"
                cv2.putText(result_img, text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display total hand and finger count
        total_hands = len(detected_hands)
        total_fingers = sum(hand['fingers'] for hand in detected_hands)
        
        info_text = f"Hands: {total_hands}, Total Fingers: {total_fingers}"
        cv2.putText(result_img, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return result_img
    
    def count_fingers_in_roi(self, hand_roi):
        """
        Count the number of fingers in hand region of interest
        
        This is a placeholder method. In a real implementation, we would:
        1. Convert to grayscale
        2. Apply skin color segmentation or thresholding
        3. Find contours 
        4. Apply convex hull to find finger tips
        5. Count extended fingers based on angles or convexity defects
        
        Args:
            hand_roi: Hand region image
            
        Returns:
            Number of fingers detected (0-5)
        """
        # Placeholder implementation - returns random finger count for demonstration
        # We'll replace this with actual finger counting logic
        
        # Convert to grayscale
        gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply thresholding
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # If no contours found, return 0
        if not contours:
            return 0
            
        # Find the largest contour (assuming it's the hand)
        max_contour = max(contours, key=cv2.contourArea)
        
        # Find convex hull and convexity defects
        hull = cv2.convexHull(max_contour, returnPoints=False)
        
        # If hull is too small, return 0
        if len(hull) < 3:
            return 0
            
        try:
            defects = cv2.convexityDefects(max_contour, hull)
            
            # Count fingers based on convexity defects
            finger_count = 1  # Start with 1 for the thumb
            
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(max_contour[s][0])
                    end = tuple(max_contour[e][0])
                    far = tuple(max_contour[f][0])
                    
                    # Calculate distance between points
                    a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    
                    # Calculate angle
                    angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                    
                    # If angle is less than 90 degrees, count as finger
                    if angle <= np.pi / 2:
                        finger_count += 1
            
            # Ensure reasonable range (0-5)
            return min(max(finger_count, 0), 5)
            
        except:
            # If there's an error in calculation, return a default value
            return 0
        
def parse_args():
    parser = argparse.ArgumentParser(description='Finger counter using YOLOv8')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='Path to input image')
    group.add_argument('--webcam', action='store_true', help='Use webcam input')
    parser.add_argument('--model', type=str, help='Path to custom YOLO model')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create finger counter instance
    counter = FingerCounter(model_path=args.model)
    
    # Process input based on arguments
    if args.image:
        print(f"Processing image: {args.image}")
        counter.process_image(args.image)
    elif args.webcam:
        print("Starting webcam mode...")
        counter.process_webcam()

if __name__ == "__main__":
    main()
