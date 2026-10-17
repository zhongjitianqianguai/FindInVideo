@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0logs" mkdir "%~dp0logs"
>>"%~dp0logs\launcher_events.log" echo [%date% %time%] event=launcher_started entrypoint=main_yolov5.py
".venv\Scripts\python.exe" "main_yolov5.py"
set "EXIT_CODE=%ERRORLEVEL%"
>>"%~dp0logs\launcher_events.log" echo [%date% %time%] event=launcher_exit entrypoint=main_yolov5.py exit_code=%EXIT_CODE%
pause
