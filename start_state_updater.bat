@echo off
echo Starting Enhanced Live Bot State Updater...
echo.

cd /d "c:\Users\ASUS\OneDrive\Documents\GitHub\my-trading-robot-1"

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo Running state updater...
echo Press Ctrl+C to stop
echo.

python live_trading\state_updater.py

echo.
echo State updater stopped.
pause
