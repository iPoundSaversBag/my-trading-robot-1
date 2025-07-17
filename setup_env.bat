@echo off
REM ==============================================================================
REM
REM                      ENVIRONMENT SETUP SCRIPT
REM
REM ==============================================================================
REM
REM FILE: setup_env.bat
REM
REM PURPOSE:
REM   This batch script automates the entire process of setting up the necessary
REM   Python environment for the trading robot. It is designed to be the first
REM   script a new user runs to ensure that all dependencies are correctly
REM   installed and isolated from other Python projects.
REM
REM METHODOLOGY:
REM   The script performs the following steps in sequence:
REM   1.  Checks if a Python virtual environment directory (`venv`) already exists.
REM       If not, it creates one using the `python -m venv` command.
REM   2.  Activates the newly created or existing virtual environment.
REM   3.  Uses `pip` to install all the required Python libraries listed in the
REM       `requirements.txt` file.
REM
REM USAGE:
REM   Simply double-click this file or run it from the command line. It will
REM   create a self-contained environment and install all necessary packages,
REM   making the project ready to run.
REM
REM ==============================================================================

REM Check if the venv directory exists
if not exist "venv" (
    echo --- Creating Python Virtual Environment ---
    python -m venv venv
) else (
    echo --- Python Virtual Environment already exists ---
)

echo.
echo --- Activating Virtual Environment ---
call .\\venv\\Scripts\\activate.bat

echo.
echo --- Installing Dependencies ---
pip install -r requirements.txt

echo.
echo --- Environment Setup Complete ---
pause