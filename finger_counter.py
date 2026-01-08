import argparse
import cv2
import time
from hand_detector import HandDetector

class FingerCounter:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6):
        """
        Finger counter powered by MediaPipe Hand Landmarks.
        Draws a skeleton overlay and counts extended fingers per hand.
        """
        self.detector = HandDetector(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            return

        processed, results = self.detector.detect_hands(image)
        hands = self.detector.count_fingers(results, image.shape)
        annotated = self.detector.draw_finger_count(processed, hands)

        cv2.imshow("Finger Counter", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_webcam(self, cam_index=0):
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        print("Webcam opened. Press 'q' to quit.")

        prev = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: Failed to capture frame")
                break

            processed, results = self.detector.detect_hands(frame)
            hands = self.detector.count_fingers(results, frame.shape)
            annotated = self.detector.draw_finger_count(processed, hands)

            # FPS overlay
            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Finger Counter (Press 'q' to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="Finger counter with MediaPipe hand landmarks")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='Path to input image')
    group.add_argument('--webcam', action='store_true', help='Use webcam input')
    parser.add_argument('--cam-index', type=int, default=0, help='Webcam index (default: 0)')
    return parser.parse_args()


def main():
    args = parse_args()
    counter = FingerCounter()
    if args.image:
        print(f"Processing image: {args.image}")
        counter.process_image(args.image)
    else:
        print("Starting webcam mode...")
        counter.process_webcam(args.cam_index)


if __name__ == "__main__":
    main()
