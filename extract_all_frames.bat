@echo off
setlocal

set "FFMPEG=C:\project\DouyinLiveRecorder\ffmpeg-7.0-essentials_build\bin\ffmpeg.exe"

if not exist "%FFMPEG%" (
    echo ffmpeg not found: %FFMPEG%
    pause
    exit /b 1
)

:check_arg
if "%~1"=="" (
    echo Usage: drag a video file onto this script
    pause
    exit /b 1
)

set "VIDEO=%~1"

if not exist "%VIDEO%" (
    echo File not found: %VIDEO%
    shift
    goto check_arg
)

set "VIDEO_DIR=%~dp1"
set "VIDEO_NAME=%~n1"
set "OUT_DIR=%VIDEO_DIR%%VIDEO_NAME%"

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

echo Video: %VIDEO%
echo Output: %OUT_DIR%
echo.

"%FFMPEG%" -hide_banner -i "%VIDEO%" -vsync 0 -qscale:v 2 "%OUT_DIR%\%%08d.jpg" -y

echo.
set COUNT=0
for %%f in ("%OUT_DIR%\*.jpg") do set /a COUNT+=1
echo Done. %COUNT% frames extracted.

pause
