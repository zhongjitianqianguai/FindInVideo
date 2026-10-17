@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0logs" mkdir "%~dp0logs"
>>"%~dp0logs\launcher_events.log" echo [%date% %time%] event=launcher_started entrypoint=main.py
".venv\Scripts\python.exe" "main.py"
set "EXIT_CODE=%ERRORLEVEL%"
>>"%~dp0logs\launcher_events.log" echo [%date% %time%] event=launcher_exit entrypoint=main.py exit_code=%EXIT_CODE%
exit /b %EXIT_CODE%
