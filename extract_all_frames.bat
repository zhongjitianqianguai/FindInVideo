@echo off
setlocal DisableDelayedExpansion

set "FFMPEG="
if defined FINDINVIDEO_FFMPEG_BIN (
    if exist "%FINDINVIDEO_FFMPEG_BIN%\ffmpeg.exe" set "FFMPEG=%FINDINVIDEO_FFMPEG_BIN%\ffmpeg.exe"
    if not defined FFMPEG if exist "%FINDINVIDEO_FFMPEG_BIN%" set "FFMPEG=%FINDINVIDEO_FFMPEG_BIN%"
)
if not defined FFMPEG (
    for /f "delims=" %%I in ('where ffmpeg 2^>nul') do if not defined FFMPEG set "FFMPEG=%%~fI"
)
if not defined FFMPEG if exist "C:\project\DouyinLiveRecorder\ffmpeg-7.0-essentials_build\bin\ffmpeg.exe" (
    set "FFMPEG=C:\project\DouyinLiveRecorder\ffmpeg-7.0-essentials_build\bin\ffmpeg.exe"
)

if not defined FFMPEG (
    echo 错误：找不到 ffmpeg，请配置 FINDINVIDEO_FFMPEG_BIN 或加入 PATH。
    pause
    exit /b 1
)

:check_arg
if "%~1"=="" (
    echo 用法：将视频文件拖到本脚本上。
    pause
    exit /b 1
)

set "VIDEO=%~f1"
if not exist "%VIDEO%" (
    echo 错误：输入文件不存在。
    shift
    goto check_arg
)

set "VIDEO_DIR=%~dp1"
set "VIDEO_NAME=%~nx1"
set "OUT_DIR=%VIDEO_DIR%%VIDEO_NAME%_frames"
set "RUN_DIR=%OUT_DIR%\run_%RANDOM%_%RANDOM%"

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
if errorlevel 1 (
    echo 错误：无法创建输出目录。
    pause
    exit /b 1
)
mkdir "%RUN_DIR%"
if errorlevel 1 (
    echo 错误：无法创建本轮输出目录。
    pause
    exit /b 1
)

echo 视频："%VIDEO%"
echo 本轮输出："%RUN_DIR%"
echo.

"%FFMPEG%" -hide_banner -i "%VIDEO%" -vsync 0 -qscale:v 2 "%RUN_DIR%\%%08d.jpg" -y
if errorlevel 1 (
    echo 错误：ffmpeg 提取失败。
    pause
    exit /b 1
)

echo.
set COUNT=0
for %%f in ("%RUN_DIR%\*.jpg") do if exist "%%~ff" set /a COUNT+=1
echo 完成：本轮共提取 %COUNT% 帧。

pause
exit /b 0
