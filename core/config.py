import os
from pathlib import Path

# Base Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
INSIGHTFACE_MODELS_DIR = MODELS_DIR / "insightface_models" / "models" / "buffalo_l"

# Execution Providers (สามารถให้ user เปลี่ยนผ่าน CLI/API ได้ทีหลัง)
DEFAULT_EXECUTION_PROVIDERS = ['CPUExecutionProvider'] # เปลี่ยนเป็น ['CUDAExecutionProvider'] หากมี GPU

# Model Paths (SSOT) - ห้าม hardcode path ในโมดูลอื่น
MODEL_PATHS = {
    "yoloface": MODELS_DIR / "yoloface_8n.onnx",
    "2dfan4": MODELS_DIR / "2dfan4.onnx",
    "arcface": INSIGHTFACE_MODELS_DIR / "w600k_r50.onnx",
    "xseg_1": MODELS_DIR / "xseg_1.onnx",
    "bisenet_resnet_34": MODELS_DIR / "bisenet_resnet_34.onnx",
    "inswapper_128": MODELS_DIR / "inswapper_128.onnx",
    "inswapper_128_fp16": MODELS_DIR / "inswapper_128_fp16.onnx",
    "simswap_256": MODELS_DIR / "simswap_256.onnx",
    "hyperswap_1a_256": MODELS_DIR / "hyperswap_1a_256.onnx",
    "hyperswap_1b_256": MODELS_DIR / "hyperswap_1b_256.onnx",
    "hyperswap_1c_256": MODELS_DIR / "hyperswap_1c_256.onnx",
    "gfpgan_1.4": MODELS_DIR / "gfpgan_1.4.onnx",
    "codeformer": MODELS_DIR / "codeformer.onnx",
    "gpen_bfr_256": MODELS_DIR / "gpen_bfr_256.onnx",
    "gpen_bfr_512": MODELS_DIR / "gpen_bfr_512.onnx",
    "ffmpeg": MODELS_DIR / "ffmpeg.exe"
}

# Defaults
DEFAULT_BACKEND = "inswapper_128"
DEFAULT_MODE = "image"
