@echo off
cd /d "%~dp0"

:: Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

:: Launch Flask app and open browser automatically
powershell -NoProfile -Command ^
  "Start-Process python -ArgumentList 'Site_Renderer_App\Site_Renderer_App.py' -WindowStyle Hidden; " ^
  "Start-Sleep 2; " ^
  "Start-Process 'http://localhost:5000'"
