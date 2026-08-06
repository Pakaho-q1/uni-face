@echo off
chcp 65001 >nul
echo ===================================================
echo     🚀 UNI-FACE Environment Setup Script
echo ===================================================
echo.

echo [1/3] Checking Conda Environment (facefusion)...
call conda activate facefusion
if errorlevel 1 (
    echo [ERROR] ไม่พบสภาพแวดล้อม "facefusion" 
    echo โปรดตรวจสอบให้แน่ใจว่าติดตั้ง FaceFusion เรียบร้อยแล้ว
    pause
    exit /b 1
)

echo.
echo [2/3] Installing Python Dependencies for Backend...
echo กำลังติดตั้งไลบรารีเสริม (FastAPI, Uvicorn, Websockets, Multipart)
pip install fastapi uvicorn python-multipart websockets

echo.
echo [3/3] Installing Node.js Dependencies for Frontend...
echo กำลังติดตั้งแพ็กเกจหน้าเว็บ (Web UI)
cd webui
call npm install
cd ..

echo.
echo ===================================================
echo  ✅ SETUP COMPLETE! ติดตั้งเสร็จสมบูรณ์
echo ===================================================
echo.
echo คุณสามารถรันระบบโดยการดับเบิ้ลคลิกไฟล์ run_uniface.bat ได้เลย
echo.
pause
