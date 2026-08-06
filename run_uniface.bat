@echo off
chcp 65001 >nul
echo ===================================================
echo     ▶️ STARTING UNI-FACE
echo ===================================================
echo.
echo Launching Server...
echo.

call conda activate uniface
python api_server.py

pause
