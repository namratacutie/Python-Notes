@echo off
setlocal
cd /d "%~dp0"
echo --- Omniversal Tracking Studio Launcher ---
echo Searching for Python 3.11 Environment...

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment ".venv" not found.
    echo Please run: py -3.11 -m venv .venv
    pause
    exit /b
)

echo Environment found! Launching tracking...
".venv\Scripts\python.exe" touch_designer.py
if errorlevel 1 (
    echo.
    echo [ERROR] The program crashed. Check the messages above.
    pause
)
