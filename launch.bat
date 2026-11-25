@echo off
echo ============================================
echo  AI Traffic Violation Detection System
echo ============================================

cd /d "%~dp0"
set "TRAFFIC_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "TRAFFIC_PYTHON=.venv\Scripts\python.exe"

REM Check if Python is available
"%TRAFFIC_PYTHON%" --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

REM Check if the dashboard is installed
"%TRAFFIC_PYTHON%" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    "%TRAFFIC_PYTHON%" -m pip install -r requirements.txt
    if errorlevel 1 exit /b 1
)

echo.
echo Starting application...
echo Open your browser at: http://localhost:8501
echo.
"%TRAFFIC_PYTHON%" -m streamlit run app.py --server.maxUploadSize 500 --server.port 8501

pause
