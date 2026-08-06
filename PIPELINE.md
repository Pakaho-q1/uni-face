# PIPELINE.md — uni-face

รายงานกลาง (Central Report) ของ pipeline + สถานะโมดูล อ้างอิงโครงสร้างจาก facefusion (detection → landmark → recognition → manipulation → paste-back) แต่ **implement เอง ใช้ asset จริงที่มี** ห้าม copy logic จาก facefusion เข้ามาปนใน core ตรงๆ (license OpenRAIL-AS มีเงื่อนไข) — โค้ด facefusion ที่ดึงมาใช้ต้องอยู่หลัง adapter เท่านั้น (ดู AGENT.md > External/Vendor Integration)

## 1. System Pipeline Diagram

```mermaid
graph TD
    Client[Client / User]
    API[API Adapter - FastAPI]
    CLI[CLI Adapter - argparse]
    Client -- HTTP POST --> API
    Client -- Terminal --> CLI

    Service[core/service.py<br/>Business Logic - SSOT ของ 'ทำอะไร']
    API --> Service
    CLI --> Service

    Dispatcher[core/dispatcher.py<br/>Pipeline Controller]
    Config[core/config.py<br/>SSOT: model paths/defaults]
    Types[core/types.py<br/>Face / FrameResult dataclass]
    Service --> Dispatcher
    Config -.-> Dispatcher
    Types -.-> Dispatcher

    subgraph "Face Processing Pipeline (modules/)"
        Detector[1. detector.py<br/>yoloface + 2dfan4 + arcface_w600k_r50]
        Parser[2. parser.py<br/>xseg_1 occluder + bisenet_resnet_34 parser]
        Swaper[3. swaper/* backend<br/>config-driven: inswapper/simswap/hyperswap]
        Restorer[4. restorer.py<br/>gfpgan_1.4 / gpen_bfr]
        Compositor[5. compositor.py<br/>color match + blend]

        Detector --> Parser
        Parser --> Swaper
        Swaper -- "(frame, mask)" --> Restorer
        Restorer -- "(frame, mask)" --> Compositor
    end

    Dispatcher --> Detector
    Compositor --> Output[Result Image/Video]
```

**หมายเหตุสำคัญ:**
- Swaper/Restorer คืนค่ารูปแบบเดียวกันเสมอ `(frame, mask)` → เพิ่ม processor ใหม่เสียบเข้า pipeline ได้โดยไม่แก้ dispatcher
- `Face` dataclass กำหนดที่เดียวใน `core/types.py` ใช้ร่วมทุก stage ห้ามแต่ละ module สร้าง schema เอง

## 2. Data Contract ต่อ stage

| # | Stage | Module | Input | Output |
|---|-------|--------|-------|--------|
| 0 | Frame extraction | `modules/io/video_io.py` | video path | `List[frame:ndarray]` (image = 1 frame) |
| 1 | Detect+Landmark+Recognize | `modules/detector.py` | `frame` | `List[Face]` = `{bbox, landmark_5/106, embedding, gender_age}` |
| 2 | Face select | `core/service.py` | `List[Face]` + criteria | `Face` (target) |
| 3 | Masking | `modules/parser.py` | `frame` + `Face` | `mask:ndarray (H,W,1)` |
| 4 | Swap | `modules/swaper/<backend>.py` | `source_embedding` + `frame` + `Face` | `(frame, mask)` |
| 5 | Restore (optional) | `modules/restorer.py` | `(frame, mask)` | `(frame, mask)` |
| 6 | Composite | `modules/compositor.py` | `(frame, mask)` + original frame | `frame` (final) |
| 7 | Output creation | `modules/io/*.py` | `List[frame]` | image/video file |

## 3. Model Mapping (ใช้ default model set ของ facefusion — กำลังจะโหลดมาเพิ่ม)

| Stage | facefusion default model | Argument อ้างอิง | หมายเหตุ |
|-------|---------------------------|-------------------|---------|
| Face Detector | `yoloface` (yoloface_8n.onnx) | `--face-detector-model` (choices: yoloface, retinaface, scrfd, many) | ตัวหาบ bbox เบื้องต้น |
| Face Landmarker | `2dfan4` | `--face-landmarker-model` (choices: 2dfan4, peppa_wutz) | หา landmark ละเอียด (68 จุด) |
| Face Recognizer | `arcface_w600k_r50` | ผูกกับ swapper model | **ตรงกับที่มีอยู่แล้ว** `insightface_models/models/buffalo_l/w600k_r50.onnx` |
| Face Occluder (mask) | `xseg_1` | `--face-occluder-model` (choices: xseg_1/2/3, many) | มือ/ผม/สิ่งบังหน้า |
| Face Parser (mask) | `bisenet_resnet_34` | `--face-parser-model` (choices: bisenet_resnet_18/34) | region mask (ตา/ปาก/ฯลฯ) — แทน Segformer เดิม |
| Face Swapper | `inswapper_128` (default) | `--face-swapper-model` (choices: inswapper_128, inswapper_128_fp16, simswap_256, simswap_512_unofficial, blendswap_256, uniface_256, hyperswap_1a/1b/1c_256) | **config-driven** เลือกได้ตอนรัน ของเดิมที่มีอยู่แล้วใช้ต่อได้ (inswapper_128, simswap_256/512, hyperswap_1a/1b/1c) |
| Face Enhancer (restore) | `gfpgan_1.4` (default) | `--face-enhancer-model` (choices: codeformer, gfpgan_1.2/1.3/1.4, gpen_bfr_256/512/1024/2048, restoreformer_plus_plus) | ของเดิม GFPGANv1.4, GPEN-BFR-512/1024 ใช้ต่อได้ เพิ่ม codeformer ได้ถ้าโหลดมา |
| Video I/O | ffmpeg | — | มีอยู่แล้ว `models/ffmpeg.exe` |

**สิ่งที่ต้องไปโหลดเพิ่ม** (ยังไม่มีใน `models/` ปัจจุบัน): `yoloface_8n.onnx`, `2dfan4.onnx`, `xseg_1.onnx`, `bisenet_resnet_34.onnx` — โมเดล swapper/enhancer ที่มีอยู่แล้วใช้ต่อได้เลย ไม่ต้องโหลดซ้ำ

> เมื่อโหลดมาครบแล้ว ให้อัปเดต path จริงลงคอลัมน์นี้ และปรับ `core/config.py` ให้ชี้ตรง — ห้าม hardcode path ในโมดูล

## 4. สถานะโมดูล

| Module | Path | สถานะ | หมายเหตุ |
|--------|------|-------|---------|
| Config (SSOT paths) | `core/config.py` | 🟢 done | |
| Types (Face, FrameResult) | `core/types.py` | 🟢 done | |
| State | `core/state.py` | 🔴 not_started | |
| Service | `core/service.py` | 🔴 not_started | |
| Dispatcher | `core/dispatcher.py` | 🔴 not_started | |
| Detector | `modules/detector.py` | 🟢 done | yoloface+2dfan4+arcface (adapter, ไม่ copy logic facefusion) |
| Parser (mask) | `modules/parser.py` | 🟢 done | xseg_1 (occluder) + bisenet_resnet_34 (region) |
| Swaper base (interface) | `modules/swaper/base.py` | 🟢 done | abstract: load(), swap() |
| Swaper: inswapper | `modules/swaper/inswapper.py` | 🟢 done | affine transform + onnx session |
| Swaper: simswap | `modules/swaper/simswap.py` | 🔴 not_started | |
| Swaper: hyperswap | `modules/swaper/hyperswap.py` | 🔴 not_started | |
| Restorer | `modules/restorer.py` | 🟢 done | GFPGAN/GPEN, รับ-คืน (frame,mask) |
| Compositor | `modules/compositor.py` | 🔴 not_started | color match + seamless blend |
| Image IO | `modules/io/image_io.py` | 🔴 not_started | |
| Video IO (ffmpeg) | `modules/io/video_io.py` | 🔴 not_started | |
| CLI adapter | `cli/` | 🔴 not_started | argparse -> service เท่านั้น |
| API adapter | `api/` | 🔴 not_started | FastAPI -> service เท่านั้น |
| main.py | `main.py` | 🔴 not_started | subcommand: cli / serve |
| Tests | `tests/` | 🔴 not_started | mock onnxruntime, คู่ทุก module |

สถานะ: 🔴 not_started → 🟡 in_progress → 🟢 done → ✅ tested
**กติกา:** อัปเดตสถานะทันทีที่โมดูลเสร็จ+เทสผ่าน ห้ามข้ามไปโมดูลที่ dependency ยังไม่ 🟢/✅

## 5. Next Steps (MVP1: swap ภาพนิ่ง ไม่ปรับสี/ไม่ restore)
1. `core/types.py` + `core/config.py` — วาง data contract ให้นิ่งก่อน
2. `modules/detector.py` — insightface adapter
3. `modules/swaper/base.py` + `inswapper.py` — 1 backend ให้รันจบ pipeline ได้ก่อน
4. `core/dispatcher.py` — ต่อ detector → swaper → output แบบไม่มี mask/restore
5. Unit test คู่ทุกไฟล์ที่ทำ ก่อนไปโมดูลถัดไป
6. ค่อยเพิ่ม parser/restorer/compositor/backend อื่นๆ + CLI/API adapter
