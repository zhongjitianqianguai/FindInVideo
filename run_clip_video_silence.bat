@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" "clip_video_silence.py"
pause
