# Finger Counter with Hand Landmarks

This project detects hands, overlays a 21‑point finger/hand skeleton, and counts how many fingers are extended in images or live webcam. It uses MediaPipe Hands for robust landmark detection and OpenCV for visualization.

## Features
- Real‑time hand landmark detection and skeleton overlay
- Per‑hand finger counting (thumb logic included)
- Left/Right hand labeling
- Image and webcam modes with FPS display

## Requirements
- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- (Optional) Ultralytics/YOLO if you later train a custom detector

See `requirements.txt` for exact versions.

## Setup
```bash
pip install -r requirements.txt
```

## Usage
- Image: 
```bash
python finger_counter.py --image path/to/image.jpg
```
- Webcam (default camera index 0):
```bash
python finger_counter.py --webcam
```
- Select a different camera:
```bash
python finger_counter.py --webcam --cam-index 1
```