@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating Python environment...
  py -3 -m venv .venv
  if errorlevel 1 (echo Python 3 was not found. Install Python and try again.& pause & exit /b 1)
)
echo Installing or checking dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (echo Dependency installation failed.& pause & exit /b 1)
echo Starting AI Vision Tracker...
start "" http://127.0.0.1:5000
".venv\Scripts\python.exe" app.py
pause
