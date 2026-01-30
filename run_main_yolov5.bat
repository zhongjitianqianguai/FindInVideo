@echo off
setlocal
cd /d %~dp0

if exist ".venv\Scripts\Activate.ps1" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%~dp0.venv\Scripts\Activate.ps1'"
)

python main_yolov5.py

if errorlevel 1 (
    echo.
    echo 运行失败，按任意键退出...
    pause >nul
)
endlocal
