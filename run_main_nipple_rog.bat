@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo 错误：找不到虚拟环境 Python："%PYTHON%"
    pause
    exit /b 1
)

if not exist "%~dp0logs" mkdir "%~dp0logs"
>>"%~dp0logs\launcher_events.log" echo [%date% %time%] event=launcher_started entrypoint=main_nipple_rog.py
"%PYTHON%" "%~dp0main_nipple_rog.py"
set "EXIT_CODE=%ERRORLEVEL%"
>>"%~dp0logs\launcher_events.log" echo [%date% %time%] event=launcher_exit entrypoint=main_nipple_rog.py exit_code=%EXIT_CODE%
if not "%EXIT_CODE%"=="0" echo 错误：ROG 处理入口退出，代码 %EXIT_CODE%。
pause
exit /b %EXIT_CODE%
