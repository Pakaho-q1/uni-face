# AGENT.md — uni-face (faceswap)

## Rules
- ห้ามเดา: ไม่ชัวร์ → ค้นเน็ตก่อน
- แก้ที่ root cause

## Env
- `conda activate facefusion``
- Models: `D:\WindowsMega\uni-face\models` (insightface_models/, parser/, restore/, swaper/, ffmpeg.exe)

## Architecture
- 1 module = 1 หน้าที่
- SSOT: `core/config.py` (paths/defaults)
- Central flow: pipeline ทั้งหมดผ่าน `core/dispatcher.py`
- **Backend/Frontend แยกกันเด็ดขาด**: `core/service.py` = business logic ชั้นเดียว (เรียก dispatcher ข้างใน) ทุก client (cli/api/gui) ต้องเรียกผ่าน service เท่านั้น ห้าม duplicate logic
- Pipeline (image=video, video=loop frame): detector → parser → swaper(config-driven backend) → restorer → compositor → output

## Structure
```
core/        config.py, state.py, dispatcher.py, service.py
modules/     detector.py, parser.py, restorer.py, compositor.py
modules/swaper/  base.py + inswapper.py, simswap.py, hyperswap.py (เพิ่ม backend ใหม่ = เพิ่มไฟล์+register เท่านั้น)
modules/io/  image_io.py, video_io.py
cli/         adapter: argv -> service (ไม่มี business logic)
api/         adapter: HTTP -> service (ไม่มี business logic)
tests/       test_<module>.py ต่อ module
configs/     yaml ต่อ backend
main.py      entrypoint เดียว, subcommand: cli | serve
```

## Entrypoint
- โหมด: `cli` (เรียก service ตรง in-process), `serve` (รัน API server ครอบ service เดียวกัน)
- GUI/client อื่นในอนาคต = เรียกผ่าน HTTP API เท่านั้น ไม่แตะ core โดยตรง
```
python main.py cli --source face.jpg --target vid.mp4 --output out.mp4 --backend hyperswap_1a
python main.py serve --port 8000
# client อื่น (gui/REST) เรียก POST http://localhost:8000/swap แทน
```

## Test
- unit test คู่ทุก module: happy/no-face/invalid/backend-switch
- mock onnxruntime, ห้ามโหลดโมเดลจริงใน test
- test service.py แยกจาก test cli/api adapter (adapter test = แค่ input parsing/response format)