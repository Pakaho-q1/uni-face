import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class Face:
    """
    Data Contract: แหล่งเก็บข้อมูลของ 1 ใบหน้าที่หาเจอในรูปภาพ
    ใช้แชร์ไปให้ทุกโมดูล (Parser, Swaper, Restorer) โดยไม่ต้องสร้าง Schema ใหม่
    """
    bbox: np.ndarray  # Bounding box [x1, y1, x2, y2]
    score: float = 0.0 # ความมั่นใจ (Confidence score)
    landmark_5: Optional[np.ndarray] = None  # จุด 5 จุดบนหน้า (ตา, จมูก, ปาก)
    landmark_106: Optional[np.ndarray] = None # จุด 106 จุดบนหน้า (โครงหน้าละเอียด)
    embedding: Optional[np.ndarray] = None  # Vector เอกลักษณ์ใบหน้า (สำหรับจำแนก หรือ swap)
    gender: Optional[int] = None  # 0 or 1
    age: Optional[int] = None
    
    # สำหรับเก็บค่าแปลกๆ เพิ่มเติมจากโมดูลอื่นๆ โดยไม่ต้องแก้ Schema
    attributes: Dict[str, Any] = field(default_factory=dict)

# ตัวแทน Output ของ Swaper/Restorer ที่ต้องมีรูปและ Mask
FrameResult = tuple[np.ndarray, np.ndarray] # (frame, mask)
