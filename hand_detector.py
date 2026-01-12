import cv2
import numpy as np
import mediapipe as mp
from enum import IntEnum

class HandLandmark(IntEnum):
    """Hand landmark positions based on MediaPipe's hand landmark model"""
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize HandDetector with MediaPipe Hands
        
        Args:
            static_image_mode: Whether to treat input as static images (True) or video stream (False)
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def detect_hands(self, image):
        """
        Detect hands in an image
        
        Args:
            image: BGR image (numpy array)
            
        Returns:
            processed_image: Image with hand landmarks drawn
            results: MediaPipe hand detection results
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and find hands
        results = self.hands.process(image_rgb)
        
        # Make a copy for drawing
        processed_image = image.copy()
        
        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    processed_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return processed_image, results
    
    def count_fingers(self, results, image_shape):
        """
        Count extended fingers based on hand landmarks
        
        Args:
            results: MediaPipe hand detection results
            image_shape: Shape of the input image (height, width)
            
        Returns:
            hands_data: List of dictionaries containing hand information (fingers, landmarks, etc.)
        """
        hands_data = []
        
        # If no hands detected, return empty list
        if not results.multi_hand_landmarks:
            return hands_data
        
        height, width, _ = image_shape
        
        # Process each detected hand
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get hand classification (Left/Right)
            handedness = results.multi_handedness[idx].classification[0].label if results.multi_handedness else "Unknown"
            # Swap Left/Right due to mirrored camera
            if handedness == "Left":
                handedness = "Right"
            elif handedness == "Right":
                handedness = "Left"
            
            # Extract landmark coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                # Convert normalized coordinates to pixel coordinates
                x, y = int(landmark.x * width), int(landmark.y * height)
                landmarks.append((x, y))
            
            # Count extended fingers
            extended_fingers = []
            finger_count = 0
            
            # Check thumb - simplified distance-based approach
            thumb_tip = landmarks[HandLandmark.THUMB_TIP]
            thumb_ip = landmarks[HandLandmark.THUMB_IP]
            thumb_mcp = landmarks[HandLandmark.THUMB_MCP]
            wrist = landmarks[HandLandmark.WRIST]
            
            # Calculate distance from thumb tip to wrist (extended thumb)
            thumb_tip_wrist_dist = np.sqrt((thumb_tip[0] - wrist[0])**2 + 
                                           (thumb_tip[1] - wrist[1])**2)
            
            # Calculate distance from thumb IP to wrist (bent thumb reference)
            thumb_ip_wrist_dist = np.sqrt((thumb_ip[0] - wrist[0])**2 + 
                                          (thumb_ip[1] - wrist[1])**2)
            
            # Thumb is extended if tip is farther from wrist than IP joint
            # This works regardless of hand orientation
            thumb_extended = thumb_tip_wrist_dist > thumb_ip_wrist_dist * 1.1
            
            if thumb_extended:
                extended_fingers.append("Thumb")
                finger_count += 1
            
            # Check fingers (index, middle, ring, pinky) using sequential joint distances
            # Each finger has 4 landmarks: MCP (base) -> PIP -> DIP -> TIP
            finger_data = [
                (HandLandmark.INDEX_FINGER_MCP, HandLandmark.INDEX_FINGER_PIP, 
                 HandLandmark.INDEX_FINGER_DIP, HandLandmark.INDEX_FINGER_TIP, "Index"),
                (HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.MIDDLE_FINGER_PIP, 
                 HandLandmark.MIDDLE_FINGER_DIP, HandLandmark.MIDDLE_FINGER_TIP, "Middle"),
                (HandLandmark.RING_FINGER_MCP, HandLandmark.RING_FINGER_PIP, 
                 HandLandmark.RING_FINGER_DIP, HandLandmark.RING_FINGER_TIP, "Ring"),
                (HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP, 
                 HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP, "Pinky")
            ]
            
            # For each finger, check if joints are progressively farther from wrist (extended)
            # This prevents false positives when fist is balled up
            for mcp_idx, pip_idx, dip_idx, tip_idx, name in finger_data:
                # Calculate distances from wrist to each joint
                mcp = landmarks[mcp_idx]
                pip = landmarks[pip_idx]
                dip = landmarks[dip_idx]
                tip = landmarks[tip_idx]
                
                pip_dist = np.sqrt((pip[0] - wrist[0])**2 + (pip[1] - wrist[1])**2)
                dip_dist = np.sqrt((dip[0] - wrist[0])**2 + (dip[1] - wrist[1])**2)
                tip_dist = np.sqrt((tip[0] - wrist[0])**2 + (tip[1] - wrist[1])**2)
                
                # Finger is extended if each joint is progressively farther from wrist
                # Allow small tolerance for measurement noise
                if (dip_dist > pip_dist * 0.95) and (tip_dist > dip_dist * 0.95):
                    extended_fingers.append(name)
                    finger_count += 1
            
            # Add hand data to result
            hand_info = {
                'handedness': handedness,
                'landmarks': landmarks,
                'finger_count': finger_count,
                'extended_fingers': extended_fingers
            }
            
            hands_data.append(hand_info)
        
        return hands_data
    
    def draw_finger_count(self, image, hands_data):
        """
        Draw finger count and hand information on image
        
        Args:
            image: Original image
            hands_data: List of dictionaries containing hand information
            
        Returns:
            Annotated image
        """
        result_image = image.copy()
        
        # If no hands detected, return original image
        if not hands_data:
            return result_image
        
        # Draw information for each hand
        for i, hand in enumerate(hands_data):
            # Get information
            handedness = hand['handedness']
            finger_count = hand['finger_count']
            extended_fingers = hand['extended_fingers']
            landmarks = hand['landmarks']
            
            # Draw a text box for finger count
            wrist = landmarks[HandLandmark.WRIST]
            text_pos = (wrist[0], wrist[1] - 20)
            
            # Create text
            text = f"{handedness} Hand: {finger_count} fingers"
            
            # Draw text background
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            text_w, text_h = text_size
            cv2.rectangle(result_image, 
                          (text_pos[0] - 5, text_pos[1] - text_h - 5),
                          (text_pos[0] + text_w + 5, text_pos[1] + 5),
                          (50, 50, 50), -1)
            
            # Draw text
            cv2.putText(result_image, text, text_pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw extended finger names
            finger_text = ", ".join(extended_fingers) if extended_fingers else "None"
            finger_text_pos = (wrist[0], wrist[1] - text_h - 25)
            
            # Draw text background
            finger_text_size, _ = cv2.getTextSize(finger_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            finger_text_w, finger_text_h = finger_text_size
            cv2.rectangle(result_image, 
                          (finger_text_pos[0] - 5, finger_text_pos[1] - finger_text_h - 5),
                          (finger_text_pos[0] + finger_text_w + 5, finger_text_pos[1] + 5),
                          (50, 50, 50), -1)
            
            # Draw text
            cv2.putText(result_image, finger_text, finger_text_pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw total count at the top of the image
        total_fingers = sum(hand['finger_count'] for hand in hands_data)
        total_hands = len(hands_data)
        
        total_text = f"Detected: {total_hands} hands, {total_fingers} fingers"
        cv2.putText(result_image, total_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return result_image
