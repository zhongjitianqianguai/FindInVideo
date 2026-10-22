@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" "independent_launcher.py"
set "EXIT_CODE=%ERRORLEVEL%"
exit /b %EXIT_CODE%
