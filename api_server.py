import os
import sys
import asyncio
import uuid
import shutil
import cv2
import base64
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import uvicorn
import threading
import numpy as np

# --- TRT Support: Auto-inject TensorRT libs into PATH ---
if sys.platform == 'win32':
    trt_path = os.path.join(sys.prefix, 'Lib', 'site-packages', 'tensorrt_libs')
    if os.path.exists(trt_path):
        os.environ['PATH'] = trt_path + os.pathsep + os.environ.get('PATH', '')

# Ensure we can import core modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.state import state
from core.video_service import process_video
from core.service import process_image

app = FastAPI(title="Uni-Face API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Replace with actual host/port if not localhost:8000
API_BASE_URL = "http://localhost:8000"



# --- WORKSPACE MANAGER ---
WORKSPACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")

def get_platform_dir(platform: str) -> str:
    if not platform: platform = "unknown"
    platform = platform.lower().replace(r'[^a-z0-9]', '_')
    return os.path.join(WORKSPACE_DIR, platform)

def ensure_workspace(platform: str):
    p_dir = get_platform_dir(platform)
    uploads_dir = os.path.join(p_dir, "uploads")
    outputs_dir = os.path.join(p_dir, "outputs")
    os.makedirs(uploads_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    return uploads_dir, outputs_dir

# --- JOB MANAGER ---
class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.cancel_events: Dict[str, threading.Event] = {}
        self.active_websockets: Dict[str, WebSocket] = {}

    def create_job(self, platform: str) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "id": job_id,
            "platform": platform,
            "status": "pending",
            "progress": 0.0,
            "frames_done": 0,
            "total_frames": 0,
            "preview_image": None,
            "output_path": None,
            "error": None
        }
        self.cancel_events[job_id] = threading.Event()
        return job_id

    def update_job(self, job_id: str, updates: Dict[str, Any]):
        if job_id in self.jobs:
            self.jobs[job_id].update(updates)
            
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)
        
    def cancel_job(self, job_id: str):
        if job_id in self.cancel_events:
            self.cancel_events[job_id].set()

job_manager = JobManager()

# --- API ENDPOINTS ---
@app.post("/api/v1/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    x_client_platform: str = Header("unknown")
):
    uploads_dir, _ = ensure_workspace(x_client_platform)
    results = []
    
    for file in files:
        file_id = f"{uuid.uuid4().hex[:8]}_{file.filename}"
        file_path = os.path.join(uploads_dir, file_id)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        results.append({"file_id": file_id, "filename": file.filename})
        
    return {"uploaded": results}


class JobStartRequest(BaseModel):
    source_file_id: str
    target_file_ids: list[str]
    preview_frequency: int = 15
    processors: list[str] = ["swap", "restore"]
    swap_model: str = "inswapper_128"
    swap_weight: float = 0.65
    swap_boost: int = 128
    restore_model: str = "gfpgan_1.4"
    restore_weight: float = 1.0
    restore_blend: int = 100
    mask_types: list[str] = ["box"]
    similarity: bool = False
    providers: list[str] = ["cpu"]

def run_job_background(job_id: str, req: JobStartRequest, x_client_platform: str):
    uploads_dir, outputs_dir = ensure_workspace(x_client_platform)
    source_path = os.path.join(uploads_dir, req.source_file_id)
    cancel_event = job_manager.cancel_events[job_id]
    
    state.init(parse_args=False)
    state.processors = req.processors
    state.swap_model = req.swap_model
    state.swap_weight = req.swap_weight
    state.swap_boost = req.swap_boost
    state.restore_model = req.restore_model
    state.restore_weight = req.restore_weight
    state.restore_blend = req.restore_blend
    state.mask_types = req.mask_types
    state.similarity = req.similarity
    if hasattr(state, "_parse_providers"):
        state._parse_providers(" ".join(req.providers))
    state.source_path = source_path
    
    try:
        from modules.detector import detect
        source_img = cv2.imread(source_path)
        if source_img is None:
            raise Exception("Could not read source image")
            
        source_faces = detect(source_img)
        if not source_faces:
            raise Exception("No face detected in source image")
            
        source_faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
        source_face = source_faces[0]
        
        total_targets = len(req.target_file_ids)
        last_output_path = None
        
        for idx, target_id in enumerate(req.target_file_ids):
            if cancel_event.is_set():
                break
                
            target_path = os.path.join(uploads_dir, target_id)
            import mimetypes
            mime_type, _ = mimetypes.guess_type(target_path)
            is_image = mime_type and mime_type.startswith('image')
            
            out_name = f"out_{target_id}"
            if is_image:
                ext = os.path.splitext(target_path)[1] or '.jpg'
                out_name += ext
            else:
                if not out_name.endswith('.mp4'):
                    out_name += ".mp4"
                    
            output_path = os.path.join(outputs_dir, out_name)
            last_output_path = output_path
            
            job_manager.update_job(job_id, {"status": "processing", "output_path": output_path})
            
            # Callback for progress and preview
            def progress_callback(current: int, total: int, frame: np.ndarray = None):
                file_pct = (current / total) * 100 if total > 0 else 0
                overall_pct = (idx / total_targets * 100) + (file_pct / total_targets)
                
                updates = {
                    "progress": round(overall_pct, 2),
                    "frames_done": current,
                    "total_frames": total
                }
                
                freq = max(1, req.preview_frequency)
                if frame is not None and (current == 1 or current % freq == 0 or current == total):
                    h, w = frame.shape[:2]
                    scale = 320 / max(h, w)
                    preview_frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
                    _, buffer = cv2.imencode('.jpg', preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    b64 = base64.b64encode(buffer).decode('utf-8')
                    updates["preview_image"] = f"data:image/jpeg;base64,{b64}"
                    
                job_manager.update_job(job_id, updates)
                
            if is_image:
                target_img = cv2.imread(target_path)
                if target_img is not None:
                    processed_img = process_image(source_face, target_img, verbose=False)
                    cv2.imwrite(output_path, processed_img)
                    progress_callback(1, 1, processed_img)
            else:
                process_video(source_face, target_path, output_path, progress_callback=progress_callback, cancel_event=cancel_event)
                
            # Cleanup target
            try:
                os.remove(target_path)
            except:
                pass
                
        if not cancel_event.is_set():
            job_manager.update_job(job_id, {"status": "completed", "progress": 100.0, "output_path": last_output_path})
        
        # Cleanup source
        try:
            os.remove(source_path)
        except:
            pass
            
    except Exception as e:
        job_manager.update_job(job_id, {"status": "failed", "error": str(e)})


@app.post("/api/v1/jobs")
async def create_job(
    req: JobStartRequest,
    x_client_platform: str = Header("unknown")
):
    job_id = job_manager.create_job(x_client_platform)
    
    t = threading.Thread(target=run_job_background, args=(job_id, req, x_client_platform))
    t.daemon = True
    t.start()
    
    return {"job_id": job_id, "status": "pending"}

@app.get("/api/v1/jobs/{job_id}")
async def get_job_status(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    resp = job.copy()
    resp.pop("preview_image", None)
    return resp

@app.post("/api/v1/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job_manager.cancel_job(job_id)
    return {"status": "cancelling"}

from fastapi.responses import FileResponse
@app.get("/api/v1/jobs/{job_id}/download")
async def download_job(job_id: str):
    job = job_manager.get_job(job_id)
    if not job or not job.get("output_path"):
        raise HTTPException(status_code=404, detail="Job output not found")
        
    output_path = job["output_path"]
    if not os.path.exists(output_path):
        # Maybe it was partially rendered
        base, ext = os.path.splitext(output_path)
        percent = int(job.get("progress", 0))
        partial_path = f"{base}_{percent}%{ext}"
        if os.path.exists(partial_path):
            output_path = partial_path
        else:
            raise HTTPException(status_code=404, detail="File not ready")
            
    import mimetypes
    media_type, _ = mimetypes.guess_type(output_path)
    if not media_type:
        media_type = "application/octet-stream"
        
    return FileResponse(output_path, media_type=media_type, filename=os.path.basename(output_path))

class DeleteHistoryRequest(BaseModel):
    filenames: list[str]

class DownloadHistoryRequest(BaseModel):
    filenames: list[str]

@app.get("/api/v1/history")
async def get_history(x_client_platform: str = Header("unknown"), skip: int = 0, limit: int = 50):
    _, outputs_dir = ensure_workspace(x_client_platform)
    files = []
    if os.path.exists(outputs_dir):
        for f in os.listdir(outputs_dir):
            path = os.path.join(outputs_dir, f)
            if os.path.isfile(path):
                import mimetypes
                mt, _ = mimetypes.guess_type(path)
                mtype = 'video' if mt and mt.startswith('video') else 'image'
                files.append({
                    "filename": f,
                    "url": f"/api/v1/history/{f}?platform={x_client_platform}",
                    "type": mtype,
                    "created_at": os.path.getmtime(path)
                })
        # Sort by newest first
        files.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {"history": files[skip:skip+limit], "total": len(files)}

@app.get("/api/v1/history/{filename}")
async def serve_history_file(filename: str, platform: str = "unknown"):
    _, outputs_dir = ensure_workspace(platform)
    file_path = os.path.join(outputs_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    import mimetypes
    media_type, _ = mimetypes.guess_type(file_path)
    return FileResponse(file_path, media_type=media_type or "application/octet-stream")

@app.delete("/api/v1/history")
async def delete_history(req: DeleteHistoryRequest, x_client_platform: str = Header("unknown")):
    _, outputs_dir = ensure_workspace(x_client_platform)
    deleted = 0
    for f in req.filenames:
        # Prevent directory traversal
        safe_f = os.path.basename(f)
        path = os.path.join(outputs_dir, safe_f)
        if os.path.exists(path):
            try:
                os.remove(path)
                deleted += 1
            except:
                pass
    return {"deleted": deleted}

import zipfile
@app.post("/api/v1/history/download")
async def download_history_bulk(req: DownloadHistoryRequest, x_client_platform: str = Header("unknown")):
    _, outputs_dir = ensure_workspace(x_client_platform)
    
    # If only 1 file, return it directly
    if len(req.filenames) == 1:
        safe_f = os.path.basename(req.filenames[0])
        path = os.path.join(outputs_dir, safe_f)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(path, filename=safe_f)
        
    # Multiple files -> zip
    downloads_dir = os.path.join(get_platform_dir(x_client_platform), "downloads")
    os.makedirs(downloads_dir, exist_ok=True)
    zip_filename = f"uni_face_export_{uuid.uuid4().hex[:8]}.zip"
    zip_path = os.path.join(downloads_dir, zip_filename)
    
    found = 0
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in req.filenames:
            safe_f = os.path.basename(f)
            path = os.path.join(outputs_dir, safe_f)
            if os.path.exists(path):
                zf.write(path, safe_f)
                found += 1
                
    if found == 0:
        raise HTTPException(status_code=404, detail="No files found")
        
    return FileResponse(zip_path, media_type="application/zip", filename="uni-face-export.zip")

# WebSocket for live progress and preview
@app.websocket("/api/v1/ws/jobs/{job_id}")
async def websocket_job_status(websocket: WebSocket, job_id: str):
    await websocket.accept()
    job_manager.active_websockets[job_id] = websocket
    
    try:
        while True:
            job = job_manager.get_job(job_id)
            if not job:
                await websocket.send_json({"error": "Job not found"})
                break
                
            await websocket.send_json(job)
            
            if job["status"] in ["completed", "failed"]:
                break
                
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    finally:
        if job_id in job_manager.active_websockets:
            del job_manager.active_websockets[job_id]

# Serve WebUI compiled dist
webui_dist = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui", "dist")
if os.path.exists(webui_dist):
    app.mount("/", StaticFiles(directory=webui_dist, html=True), name="webui")
else:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    @app.get("/")
    async def root():
        return RedirectResponse(url="/static/index.html")


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
