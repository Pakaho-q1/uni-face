import cv2
import numpy as np
import onnxruntime
from typing import Literal

from core.types import Face
from core.config import MODEL_PATHS
from core.state import state
from modules.utils import face_math
from modules.parser import get_combined_mask

class FaceRestorer:
    """
    Native implementation for Face Enhancement (Restoration).
    Supports GFPGAN and GPEN.
    """
    def __init__(self):
        self.providers = state.providers
        model_key = state.restore_model
        if model_key not in MODEL_PATHS:
            model_key = 'gfpgan_1.4'
            
        provider_names = [p if isinstance(p, str) else p[0] for p in self.providers]
        print(f"Loading Restorer Model: {model_key} {provider_names}")
        self.session = onnxruntime.InferenceSession(str(MODEL_PATHS[model_key]), providers=self.providers)
        
        # Set crop size based on model
        self.template = 'ffhq_512'
        if '256' in model_key:
            self.crop_size = (256, 256)
        else:
            self.crop_size = (512, 512)
        
        # Check if model has a 'weight' input (some enhancers support blending weight)
        self.has_weight = any(inp.name == 'weight' for inp in self.session.get_inputs())
        self.input_name = self.session.get_inputs()[0].name
        self.weight_name = 'weight' if self.has_weight else None

    def restore(self, target_face: Face, temp_vision_frame: np.ndarray, weight: float = 0.5, blend: float = 0.8) -> np.ndarray:
        """
        Enhance/restore a face in the full frame.
        
        Args:
            target_face: The Face object representing the face to restore.
            temp_vision_frame: The full frame image (BGR).
            weight: Model inference weight if supported.
            blend: Alpha blend factor for pasting back (1.0 = fully replace).
        """
        # 1. Warp face to 512x512 crop
        crop_vision_frame, affine_matrix = face_math.warp_face_by_face_landmark_5(
            temp_vision_frame, 
            target_face.landmark_5, 
            self.template, 
            self.crop_size
        )
        
        # 2. Prepare tensor (RGB, -1 to 1, NCHW)
        prepare_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
        prepare_vision_frame = (prepare_vision_frame - 0.5) / 0.5
        prepare_vision_frame = np.expand_dims(prepare_vision_frame.transpose(2, 0, 1), axis=0).astype(np.float32)
        
        # 3. Inference
        inputs = {self.input_name: prepare_vision_frame}
        if self.has_weight:
            inputs[self.weight_name] = np.array([weight], dtype=np.float64)
            
        enhanced_crop = self.session.run(None, inputs)[0][0]
        
        # 4. Denormalize tensor (NCHW to HWC, RGB to BGR, 0 to 255)
        enhanced_crop = np.clip(enhanced_crop, -1, 1)
        enhanced_crop = (enhanced_crop + 1) / 2
        enhanced_crop = enhanced_crop.transpose(1, 2, 0)
        enhanced_crop = (enhanced_crop * 255.0).round().astype(np.uint8)
        enhanced_crop = enhanced_crop[:, :, ::-1]
        
        # 5. Get Parsing Mask for seamless paste
        # This prevents restoring hair, hands, or background.
        crop_mask = get_combined_mask(enhanced_crop)
        
        # 6. Paste back
        paste_vision_frame = face_math.paste_back(temp_vision_frame, enhanced_crop, crop_mask, affine_matrix)
        
        # 7. Final Alpha Blend (if user doesn't want 100% sharp contrast)
        if blend < 1.0:
            temp_vision_frame = cv2.addWeighted(temp_vision_frame, 1 - blend, paste_vision_frame, blend, 0)
            return temp_vision_frame
        
        return paste_vision_frame

# Export a default instance
import threading
restorer_app = None
_lock = threading.Lock()

def restore(target_face: Face, frame: np.ndarray, weight: float = 0.5, blend: float = 1.0) -> np.ndarray:
    global restorer_app
    if restorer_app is None:
        with _lock:
            if restorer_app is None:
                restorer_app = FaceRestorer()
    return restorer_app.restore(target_face, frame, weight, blend)
