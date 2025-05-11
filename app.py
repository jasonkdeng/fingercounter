import argparse
import cv2
import numpy as np
import time
from ultralytics import YOLO
from hand_detector import HandDetector

class FingerCounterApp:
    def __init__(self, use_yolo=True, yolo_model_path=None, use_mediapipe=True):
        """
        Initialize the Finger Counter application
        
        Args:
            use_yolo: Whether to use YOLO for initial hand detection
            yolo_model_path: Path to custom YOLO model (None for default)
            use_mediapipe: Whether to use MediaPipe for landmark detection
        """
        print("Initializing Finger Counter Application...")
        
        self.use_yolo = use_yolo
        self.use_mediapipe = use_mediapipe
        
        # Initialize frame count and FPS variables
        self.frame_count = 0
        self.fps = 0
        self.fps_start_time = 0
        
        # Load YOLO model if enabled
        if use_yolo:
            print("Loading YOLO model...")
            if yolo_model_path:
                self.yolo_model = YOLO(yolo_model_path)
            else:
                self.yolo_model = YOLO("yolov8n.pt")
            print("YOLO model loaded successfully!")
        
        # Initialize MediaPipe hand detector if enabled
        if use_mediapipe:
            print("Initializing MediaPipe Hand Detector...")
            self.hand_detector = HandDetector(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            print("MediaPipe Hand Detector initialized successfully!")
    
    def process_image(self, image_path):
        """Process a single image file"""
        print(f"Processing image: {image_path}")
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            return
        
        # Process the image
        result = self.process_frame(image)
        
        # Display result
        cv2.imshow("Finger Counter", result)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def process_webcam(self):
        """Process webcam input in real-time"""
        print("Starting webcam mode...")
        
        # Initialize webcam capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
            
        print("Webcam opened successfully. Press 'q' to quit.")
        
        # Initialize FPS calculation
        self.fps_start_time = time.time()
        self.frame_count = 0
        
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Process the frame
            result = self.process_frame(frame)
            
            # Calculate FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.fps_start_time
            if elapsed_time > 1:
                self.fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.fps_start_time = time.time()
            
            # Display FPS
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(result, fps_text, (result.shape[1] - 120, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the result
            cv2.imshow("Finger Counter (Press 'q' to quit)", result)
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
    def process_frame(self, frame):
        """
        Process a single frame to detect hands and count fingers
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            Processed frame with annotations
        """
        result_frame = frame.copy()
        hand_regions = []
        
        # Step 1: Use YOLO to detect hands (if enabled)
        if self.use_yolo:
            # Detect objects using YOLO
            yolo_results = self.yolo_model(frame, verbose=False)[0]
            
            # Filter for potential hand detections
            for detection in yolo_results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = detection
                
                # Class 0 is person in COCO dataset, but we're mainly looking for high-confidence detections
                # This approach can be enhanced by training YOLO specifically for hands
                if conf > 0.3:
                    # Convert to integers
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Extract the region
                    region = frame[y1:y2, x1:x2].copy()
                    
                    # Skip empty regions
                    if region.size == 0:
                        continue
                    
                    # Add the region to our list with its coordinates
                    hand_regions.append({
                        'region': region,
                        'box': (x1, y1, x2, y2)
                    })
        
        # Step 2: If no hand regions detected by YOLO or YOLO is disabled, use the entire frame
        if not hand_regions or not self.use_yolo:
            hand_regions = [{
                'region': frame,
                'box': (0, 0, frame.shape[1], frame.shape[0])
            }]
        
        # Step 3: Use MediaPipe for detailed hand landmark detection (if enabled)
        if self.use_mediapipe:
            # Process the entire frame with MediaPipe
            _, results = self.hand_detector.detect_hands(frame)
            
            # Count fingers based on landmarks
            hands_data = self.hand_detector.count_fingers(results, frame.shape)
            
            # Draw finger count and hand information
            result_frame = self.hand_detector.draw_finger_count(frame, hands_data)
            
            # Draw detection boxes from YOLO if we used it
            if self.use_yolo and hand_regions[0]['box'] != (0, 0, frame.shape[1], frame.shape[0]):
                for hand_region in hand_regions:
                    x1, y1, x2, y2 = hand_region['box']
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
            return result_frame
        
        # Step 4: If MediaPipe is disabled, use our backup method on detected regions
        else:
            for hand_region in hand_regions:
                region = hand_region['region']
                x1, y1, x2, y2 = hand_region['box']
                
                # Use backup finger counting method
                finger_count = self._count_fingers_backup(region)
                
                # Draw bounding box
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Display finger count
                text = f"Fingers: {finger_count}"
                cv2.putText(result_frame, text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            return result_frame
    
    def _count_fingers_backup(self, hand_region):
        """
        Backup method to count fingers using simple image processing
        
        This is a simplified method that may not be as accurate as MediaPipe
        
        Args:
            hand_region: Region of interest containing the hand
            
        Returns:
            Estimated finger count (0-5)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
        
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
            # We need to be more conservative about counting the thumb in the backup method
            # Start with 0 and we'll decide about the thumb based on the overall shape
            finger_count = 0
            
            # Store all defect angles
            all_angles = []
            
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
                    all_angles.append(angle)
                    
                    # If angle is less than 90 degrees, count as finger
                    if angle <= np.pi / 2:
                        finger_count += 1
                
                # Determine thumb: 
                # In a typical hand, thumb creates distinctive patterns in the convex hull
                # If we have detected 1 or more fingers, and there are multiple defects
                # with small angles, we likely have a thumb
                if finger_count >= 1 and len(all_angles) >= 3:
                    # Sort angles and check if we have clusters of small angles
                    sorted_angles = sorted(all_angles)
                    if len(sorted_angles) >= 3 and sorted_angles[0] < np.pi/6:
                        # The thumb is likely already counted in finger_count
                        pass
                    else:
                        # No clear thumb pattern detected, ensure we don't overcount
                        finger_count = min(finger_count, 4)
            
            # Ensure reasonable range (0-5)
            return min(max(finger_count, 0), 5)
            
        except:
            # If there's an error in calculation, return a default value
            return 0

def parse_args():
    parser = argparse.ArgumentParser(description='Finger Counter Application')
    
    # Input source arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='Path to input image')
    group.add_argument('--webcam', action='store_true', help='Use webcam input')
    
    # Detection method arguments
    parser.add_argument('--no-yolo', action='store_true', help='Disable YOLO detection')
    parser.add_argument('--no-mediapipe', action='store_true', help='Disable MediaPipe detection')
    parser.add_argument('--yolo-model', type=str, help='Path to custom YOLO model')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create application instance
    app = FingerCounterApp(
        use_yolo=not args.no_yolo,
        yolo_model_path=args.yolo_model,
        use_mediapipe=not args.no_mediapipe
    )
    
    # Process input based on arguments
    if args.image:
        app.process_image(args.image)
    elif args.webcam:
        app.process_webcam()

if __name__ == "__main__":
    main()
