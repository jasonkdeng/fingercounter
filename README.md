# Finger Counter

This project detects hands and counts fingers using a 21 point hand landmark overlay

## Features
- Real‑time hand landmark detection and skeleton overlay
- Left/Right hand labeling
- Image and webcam modes

## Requirements
- Python 3.8+
- OpenCV
- MediaPipe
- NumPy

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
