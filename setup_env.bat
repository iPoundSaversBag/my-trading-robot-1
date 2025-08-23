@echo off
REM ============================================================================== 
REM
REM                      ENVIRONMENT SETUP SCRIPT (RESTORED)
REM
REM ============================================================================== 
REM Creates/activates venv and installs dependencies from requirements.txt
REM Adds optional upgrade and wheel build acceleration.
REM ============================================================================== 

SET VENV_DIR=venv

if not exist "%VENV_DIR%" (
    echo --- Creating Python Virtual Environment ---
    python -m venv "%VENV_DIR%"
) else (
    echo --- Python Virtual Environment already exists ---
)

echo.
echo --- Activating Virtual Environment ---
call "%VENV_DIR%\Scripts\activate.bat"

echo.
echo --- Upgrading pip/setuptools/wheel ---
python -m pip install --upgrade pip setuptools wheel

if exist requirements.txt (
    echo.
    echo --- Installing Dependencies from requirements.txt ---
    pip install -r requirements.txt
) else (
    echo requirements.txt not found. Skipping dependency install.
)

echo.
echo --- Environment Setup Complete ---
:END
