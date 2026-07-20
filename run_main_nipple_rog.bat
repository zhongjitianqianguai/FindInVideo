@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo 错误：找不到虚拟环境 Python："%PYTHON%"
    pause
    exit /b 1
)

"%PYTHON%" "%~dp0main_nipple_rog.py"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" echo 错误：ROG 处理入口退出，代码 %EXIT_CODE%。
pause
exit /b %EXIT_CODE%
