@echo off
REM Trading Dashboard Server Launcher
REM This starts the HTTP server for the trading dashboard

echo.
echo ====================================
echo    TRADING DASHBOARD SERVER
echo ====================================
echo.
echo Starting HTTP server...
echo Dashboard will be available at: http://localhost:8080
echo.

cd /d "%~dp0.."
python scripts/dashboard_server.py

echo.
echo Server stopped.
pause
