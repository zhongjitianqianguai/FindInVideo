@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0logs" mkdir "%~dp0logs"
>>"%~dp0logs\launcher_events.log" echo [%date% %time%] event=launcher_started entrypoint=main_nipple.py
".venv\Scripts\python.exe" "main_nipple.py"
set "EXIT_CODE=%ERRORLEVEL%"
>>"%~dp0logs\launcher_events.log" echo [%date% %time%] event=launcher_exit entrypoint=main_nipple.py exit_code=%EXIT_CODE%
pause
