@echo off
title Plant Disease Detection - launcher
cd /d "%~dp0"

echo.
echo   ============================================================
echo    PLANT DISEASE DETECTION
echo   ============================================================
echo.
echo    This window will set everything up and then open the app.
echo    The first run takes 5 to 10 minutes (downloads about 700 MB).
echo    Do not close it. Later runs take about 15 seconds.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_and_run.ps1"
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" (
    echo   ------------------------------------------------------------
    echo    The launcher stopped with code %RC%.
    echo    Read the message above - it says exactly what to fix.
    echo   ------------------------------------------------------------
) else (
    echo   The app has closed. You can close this window.
)
echo.
pause
