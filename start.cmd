@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Please run install.cmd first.
    pause
    exit /b 1
)
start "Meeting Archive" ".venv\Scripts\pythonw.exe" "%~dp0run.py"
