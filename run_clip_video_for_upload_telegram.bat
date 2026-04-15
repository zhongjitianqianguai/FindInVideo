@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" "clip_video_for_upload_telegram.py"
pause
