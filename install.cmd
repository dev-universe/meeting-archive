@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
if errorlevel 1 (
    echo Installation failed. See the message above.
    pause
    exit /b 1
)
echo Installation complete. Double-click start.cmd to open the app.
pause
