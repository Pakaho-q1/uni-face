@echo off
chcp 65001 >nul
echo ===================================================
echo     ▶️ STARTING UNI-FACE
echo ===================================================
echo.
echo Launching Backend (FastAPI) and Frontend (Vite)
echo Please wait a moment for the windows to open...

start "Uni-Face Backend" cmd /k "conda activate facefusion && python api_server.py"
start "Uni-Face Frontend" cmd /k "cd webui && npm run dev -- --host"

echo.
echo ระบบกำลังรันอยู่เบื้องหลัง! 
echo - หากต้องการใช้งานบนคอมพิวเตอร์เครื่องนี้ เข้าไปที่ http://localhost:5173
echo - หากต้องการใช้งานผ่านมือถือ ให้ดู IP Address ในหน้าต่าง Frontend
echo.
pause
