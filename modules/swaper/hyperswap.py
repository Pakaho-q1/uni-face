import numpy as np
import onnxruntime
from typing import Literal

from core.types import Face
from core.config import MODEL_PATHS
from core.state import state
from modules.utils import face_math
from modules.swaper.base import BaseSwaper

class Hyperswap(BaseSwaper):
    """
    Native implementation of Hyperswap 256.
    """

    def __init__(self):
        self.providers = state.providers
        model_key = state.swap_model
        if model_key not in MODEL_PATHS:
            model_key = "hyperswap_1a_256"
            
        model_path = str(MODEL_PATHS[model_key])
        provider_names = [p if isinstance(p, str) else p[0] for p in self.providers]
        print(f"Loading Swapper Model: {model_key} {provider_names}")
        
        # 1. Initialize ONNX Session
        self.session = onnxruntime.InferenceSession(model_path, providers=self.providers)
        
        self.template = 'arcface_128'
        self.crop_size = (256, 256)
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        
    def swap(self, source_face: Face, target_face: Face, temp_vision_frame: np.ndarray) -> np.ndarray:
        # 1. Warp target face
        crop_vision_frame, affine_matrix = face_math.warp_face_by_face_landmark_5(
            temp_vision_frame, 
            target_face.landmark_5, 
            self.template, 
            self.crop_size
        )
        
        # 2. Prepare target crop tensor
        crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0  # BGR to RGB, normalize 0-1
        crop_vision_frame = (crop_vision_frame - self.mean) / self.std
        crop_vision_frame = crop_vision_frame.transpose(2, 0, 1)    # HWC to CHW
        crop_vision_frame = np.expand_dims(crop_vision_frame, axis=0).astype(np.float32)
        
        # 3. Prepare source embedding
        weight = float(np.interp(state.swap_weight, [0, 1], [0.35, -0.35]))
        
        source_embedding = source_face.embedding.copy().reshape(1, -1)
        target_embedding = target_face.embedding.copy().reshape(1, -1)
        
        # Apply balance
        balanced_embedding = source_embedding * (1 - weight) + target_embedding * weight
        
        # Hyperswap takes the normalized embedding directly
        source_embedding_proj = balanced_embedding / np.linalg.norm(balanced_embedding)
            
        # 4. Run Inference
        inputs = {
            'source': source_embedding_proj,
            'target': crop_vision_frame
        }
        swapped_crop = self.session.run(None, inputs)[0][0]
        
        # 5. Denormalize swapped crop
        swapped_crop = swapped_crop.transpose(1, 2, 0) # CHW to HWC
        swapped_crop = swapped_crop * self.std + self.mean
        swapped_crop = swapped_crop.clip(0, 1)
        swapped_crop = swapped_crop[:, :, ::-1] * 255.0 # RGB to BGR
        
        # 6. Generate precise mask using Parser
        from modules.parser import get_combined_mask
        crop_mask = get_combined_mask(swapped_crop, state.mask_types, target_face, affine_matrix)
        
        # 7. Paste back
        paste_vision_frame = face_math.paste_back(temp_vision_frame, swapped_crop, crop_mask, affine_matrix)
        
        return paste_vision_frame
