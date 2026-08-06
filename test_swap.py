import os
import cv2
import numpy as np
import onnxruntime
from insightface.app import FaceAnalysis
from insightface.utils import face_align  # ใช้สำหรับคำนวณ Matrix ตัดใบหน้า

# ==========================================
# 1. ตั้งค่า Path และโหลดโมเดล
# ==========================================
BASE_DIR = r"D:\WindowsMega\uni-face\models"
INSIGHTFACE_ROOT = os.path.join(BASE_DIR, "insightface_models")
SWAPPER_PATH = os.path.join(BASE_DIR, "swaper", "inswapper_128.onnx")
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# โหลด Detector
app = FaceAnalysis(name="buffalo_l", root=INSIGHTFACE_ROOT, providers=PROVIDERS)
app.prepare(ctx_id=0, det_size=(640, 640))

# โหลด HyperSwap Session โดยตรง (ไม่ผ่าน INSwapper)
session = onnxruntime.InferenceSession(SWAPPER_PATH, providers=PROVIDERS)

# อ่านชื่อ Input ของโมเดล (มักจะเป็น target image และ source embedding)
input_names = [inp.name for inp in session.get_inputs()]
output_names = [out.name for out in session.get_outputs()]

# ==========================================
# 2. ฟังก์ชันหลักสำหรับรัน HyperSwap โดยเฉพาะ
# ==========================================
def run_hyperswap(img_target, target_face, source_face):
    # --- ขั้นที่ 1: ตัดภาพใบหน้า (Crop & Align) เป็น 256x256 ---
    M = face_align.estimate_norm(target_face.kps, 256, mode='None')
    target_crop = cv2.warpAffine(img_target, M, (256, 256), borderValue=0.0)
    
    target_tensor = target_crop[:, :, ::-1] / 255.0 
    target_tensor = (target_tensor - 0.5) / 0.5
    target_tensor = np.transpose(target_tensor, (2, 0, 1))
    target_tensor = np.expand_dims(target_tensor, axis=0).astype(np.float32)

    source_embedding = source_face.embedding.reshape(1, -1)
    
    # --- ขั้นที่ 2: โยนเข้าโมเดล ONNX (แก้ไขแล้ว) ---
    inputs = {}
    for inp in session.get_inputs():
        if len(inp.shape) == 4:
            inputs[inp.name] = target_tensor
        elif len(inp.shape) == 2:
            inputs[inp.name] = source_embedding
            
    outputs = session.run(output_names, inputs)
    swapped_face = outputs[0][0] 
    
    # --- ขั้นที่ 3: แปะหน้ากลับ (Post-Process & Paste Back) ---
    swapped_face = np.transpose(swapped_face, (1, 2, 0))
    swapped_face = (swapped_face * 0.5 + 0.5) * 255.0
    swapped_face = swapped_face[:, :, ::-1].astype(np.uint8)
    
    IM = cv2.invertAffineTransform(M)
    mask = np.full((256, 256), 255, dtype=np.float32)
    
    img_white = np.full((img_target.shape[0], img_target.shape[1]), 255, dtype=np.float32)
    bgr_swapped = cv2.warpAffine(swapped_face, IM, (img_target.shape[1], img_target.shape[0]), borderValue=0.0)
    mask_swapped = cv2.warpAffine(mask, IM, (img_target.shape[1], img_target.shape[0]), borderValue=0.0)
    
    mask_alpha = mask_swapped / 255.0
    mask_alpha = np.expand_dims(mask_alpha, axis=2)
    
    res_img = (1.0 - mask_alpha) * img_target.astype(np.float32) + mask_alpha * bgr_swapped.astype(np.float32)
    return res_img.astype(np.uint8)

# ==========================================
# 3. ลองใช้งานฟังก์ชัน
# ==========================================
img_source = cv2.imread("source.jpg")
img_target = cv2.imread("target.jpg")

faces_source = app.get(img_source)
faces_target = app.get(img_target)

if faces_source and faces_target:
    print("กำลังสลับหน้าผ่านฟังก์ชัน Custom HyperSwap...")
    result = run_hyperswap(img_target, faces_target[0], faces_source[0])
    
    cv2.imwrite("result_hyperswap_custom.jpg", result)
    print("เสร็จสิ้น!")