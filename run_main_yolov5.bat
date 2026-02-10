@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" "main_yolov5.py"
pause
