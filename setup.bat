@echo off
echo Setting up Finger Counter project...

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed. Please install Python 3.8 or newer and try again.
    exit /b 1
)

REM Create and activate virtual environment
echo Setting up virtual environment...
if exist venv (
    echo Virtual environment already exists. Using existing environment.
) else (
    python -m venv venv
    echo Created new virtual environment.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing required packages...
pip install -r requirements.txt

REM Download sample images
echo Downloading sample images...
python download_samples.py

echo.
echo Setup completed successfully!
echo.
echo To run the application:
echo 1. Activate the virtual environment: venv\Scripts\activate.bat
echo 2. Run with webcam: python app.py --webcam
echo 3. Run with an image: python app.py --image sample_images/hand_five_fingers.jpg
echo.
echo Enjoy your finger counter application!

pause
