@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=C:\project\FindInVideo\.venv\Scripts\python.exe"
cd /d "%SCRIPT_DIR%"
"%PYTHON_EXE%" "%SCRIPT_DIR%clip_video_for_upload_telegram.py"
pause
