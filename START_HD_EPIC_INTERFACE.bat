@echo off
setlocal
cd /d "%~dp0"

set "PORT=8000"
set "TARGET=http://127.0.0.1:%PORT%/HD_EPIC_VQA_Interface.html"

echo Starting local server at %TARGET%
echo Keep this window open while using the interface.
echo.
start "" "%TARGET%"

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 -m http.server %PORT%
    goto :eof
)

where python >nul 2>nul
if %errorlevel%==0 (
    python -m http.server %PORT%
    goto :eof
)

where python3 >nul 2>nul
if %errorlevel%==0 (
    python3 -m http.server %PORT%
    goto :eof
)

echo Python was not found on PATH.
echo Install Python from https://www.python.org/downloads/ and try again.
pause
exit /b 1
