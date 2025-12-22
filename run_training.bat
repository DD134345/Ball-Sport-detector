@echo off
REM Batch script to run training_model.py with proper path handling
REM This script handles paths with spaces correctly

REM Get the script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Check if virtual environment exists and activate it
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Using system Python.
)

REM Run the training script
echo Starting training...
python training_model.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo Press any key to exit...
    pause >nul
)

