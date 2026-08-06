import numpy as np
import cv2
from typing import Union
from core.types import Face

from core.state import state
from modules.detector import detect
from modules.swaper.swap import swap
from modules.restorer import restore
from modules.compositor import composite

class FaceService:
    def __init__(self):
        pass
        
    def process_image(self, source: Union[np.ndarray, Face], target_img: np.ndarray, verbose: bool = True) -> np.ndarray:
        # 1. Get/Detect Source Face
        if isinstance(source, np.ndarray):
            source_faces = detect(source)
            if not source_faces:
                if verbose:
                    print("No face detected in source image.")
                return target_img
            source_faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
            source_face = source_faces[0]
        else:
            source_face = source
        
        # 2. Detect Target Faces
        target_faces = detect(target_img)
        if not target_faces:
            if verbose:
                print("No face detected in target image.")
            return target_img
            
        target_faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
        target_face = target_faces[0]
        
        # 3. Dynamic Pipeline based on state.processors
        result_img = target_img.copy()
        
        if 'swap' in state.processors:
            result_img = swap(source_face, target_face, result_img)
            
        if 'restore' in state.processors:
            if verbose:
                print(f"Starting Restorer (model: {state.restore_model}, blend: {state.restore_blend})...")
            result_img = restore(target_face, result_img, blend=(state.restore_blend / 100.0))
            
        if 'color' in state.processors:
            if verbose:
                print("Starting Compositor (Color Match & Blend)...")
            result_img = composite(target_face, result_img, target_img)
            
        return result_img

service_app = FaceService()

def process_image(source: Union[np.ndarray, Face], target_img: np.ndarray, verbose: bool = True) -> np.ndarray:
    return service_app.process_image(source, target_img, verbose)
