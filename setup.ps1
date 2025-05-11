# Setup script for the finger counter project
# Run this script to install dependencies and set up the project

# Check if Python and pip are installed
if (Get-Command python -ErrorAction SilentlyContinue) {
    Write-Host "Python is installed: " -NoNewline
    python --version
} else {
    Write-Host "Python is not installed. Please install Python 3.8 or newer and try again." -ForegroundColor Red
    exit 1
}

# Create and activate virtual environment
Write-Host "Setting up virtual environment..." -ForegroundColor Cyan

if (Test-Path "venv") {
    Write-Host "Virtual environment already exists. Using existing environment."
} else {
    python -m venv venv
    Write-Host "Created new virtual environment."
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "Installing required packages..." -ForegroundColor Cyan
pip install -r requirements.txt

# Download sample images
Write-Host "Downloading sample images..." -ForegroundColor Cyan
python download_samples.py

Write-Host "`nSetup completed successfully!" -ForegroundColor Green
Write-Host "`nTo run the application:" -ForegroundColor Yellow
Write-Host "1. Activate the virtual environment: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "2. Run with webcam: python app.py --webcam" -ForegroundColor Yellow
Write-Host "3. Run with an image: python app.py --image sample_images/hand_five_fingers.jpg" -ForegroundColor Yellow
Write-Host "`nEnjoy your finger counter application!" -ForegroundColor Green
