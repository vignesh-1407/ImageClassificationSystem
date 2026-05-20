@echo off
title AI Image Classification System - Setup
color 0A

echo.
echo  ============================================================
echo   Advanced AI-Based Image Classification System
echo   Setup Script
echo  ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo         Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

echo [OK] Python found.
python --version

:: Create virtual environment
echo.
echo [INFO] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

:: Activate venv
call venv\Scripts\activate.bat

:: Upgrade pip
echo.
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

:: Install dependencies
echo.
echo [INFO] Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [WARNING] Some packages may have failed. Trying individual installs...
    pip install tensorflow
    pip install opencv-python
    pip install numpy scikit-learn matplotlib seaborn Pillow
)

:: Create directories
echo.
echo [INFO] Creating project directories...
if not exist "data"          mkdir data
if not exist "models"        mkdir models
if not exist "outputs"       mkdir outputs
if not exist "sample_images" mkdir sample_images

echo.
echo  ============================================================
echo   Setup Complete!
echo  ============================================================
echo.
echo  Next steps:
echo    1. Activate venv:   venv\Scripts\activate
echo    2. Train model:     python main.py --mode train
echo    3. Evaluate:        python main.py --mode evaluate
echo    4. Predict image:   python main.py --mode predict --image sample_images\test.jpg
echo    5. Webcam:          python main.py --mode predict --webcam
echo.
pause
