@echo off
title AI Image Classification System
color 0B

echo.
echo  ============================================================
echo   Advanced AI-Based Image Classification System
echo  ============================================================
echo.
echo  Select a mode:
echo    [1] Train model
echo    [2] Evaluate model
echo    [3] Predict - single image
echo    [4] Predict - webcam (real-time)
echo    [5] Show model summary
echo    [6] Exit
echo.
set /p choice=" Enter your choice (1-6): "

:: Activate venv if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

if "%choice%"=="1" (
    echo.
    echo [INFO] Starting training on CIFAR-10...
    python main.py --mode train
)
if "%choice%"=="2" (
    echo.
    echo [INFO] Running evaluation...
    python main.py --mode evaluate
)
if "%choice%"=="3" (
    echo.
    set /p img_path=" Enter image path: "
    python main.py --mode predict --image "%img_path%"
)
if "%choice%"=="4" (
    echo.
    echo [INFO] Starting webcam prediction (press Q to quit)...
    python main.py --mode predict --webcam
)
if "%choice%"=="5" (
    echo.
    python main.py --mode summary
)
if "%choice%"=="6" (
    exit /b 0
)

echo.
pause
