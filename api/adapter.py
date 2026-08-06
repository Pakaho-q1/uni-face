from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from core.service import SwapService

app = FastAPI(title="Uni-Face API", description="API adapter for face swapping service")

class SwapRequest(BaseModel):
    source_path: str
    target_path: str
    output_path: str
    backend: str = "inswapper"
    mode: str = "image"

@app.post("/swap")
def swap_faces(request: SwapRequest):
    try:
        result = SwapService.process_swap(
            source_path=request.source_path,
            target_path=request.target_path,
            output_path=request.output_path,
            backend=request.backend,
            mode=request.mode
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_server(port: int = 8000):
    uvicorn.run("api.adapter:app", host="0.0.0.0", port=port, reload=False)
