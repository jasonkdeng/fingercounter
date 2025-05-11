# Finger Counter with YOLO

This project uses YOLOv8 to detect hands and count how many fingers are being held up in images or video streams. It features robust finger detection that can accurately distinguish between extended fingers and those in a fist position.

## Features
- Hand detection using YOLOv8
- Advanced finger counting algorithm with accurate thumb detection
- Considers hand geometry and angles to detect extended fingers
- Support for both image files and webcam input

## Requirements
- Python 3.8+
- PyTorch
- Ultralytics (YOLOv8)
- OpenCV
- Other dependencies (see requirements.txt)

## Setup
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python finger_counter.py`

## Usage
- Image mode: `python finger_counter.py --image path/to/image.jpg`
- Webcam mode: `python finger_counter.py --webcam`
