import cv2
import numpy as np
import onnxruntime
from typing import List

from core.config import MODEL_PATHS, DEFAULT_EXECUTION_PROVIDERS

class MaskParser:
    """
    Native implementation of Face Masking (Occlusion, Region, Box).
    """
    def __init__(self):
        self.providers = DEFAULT_EXECUTION_PROVIDERS
        self.xseg_session = onnxruntime.InferenceSession(str(MODEL_PATHS["xseg_1"]), providers=self.providers)
        self.bisenet_session = onnxruntime.InferenceSession(str(MODEL_PATHS["bisenet_resnet_34"]), providers=self.providers)
        
        # Region mappings for BiseNet
        self.region_mapping = {
            'skin': 1, 'l_brow': 2, 'r_brow': 3, 'l_eye': 4, 'r_eye': 5,
            'eye_g': 6, 'l_ear': 7, 'r_ear': 8, 'ear_r': 9, 'nose': 10,
            'mouth': 11, 'u_lip': 12, 'l_lip': 13, 'neck': 14, 'neck_l': 15,
            'cloth': 16, 'hair': 17, 'hat': 18
        }
        
    def create_occlusion_mask(self, crop_vision_frame: np.ndarray) -> np.ndarray:
        """
        Create a mask of occlusions (e.g. hands, hair over face) using xseg_1.
        """
        model_size = (256, 256)
        
        # Prepare tensor for xseg (expects NHWC, BGR, 0-1)
        prepare_vision_frame = cv2.resize(crop_vision_frame, model_size)
        prepare_vision_frame = np.expand_dims(prepare_vision_frame, axis=0).astype(np.float32) / 255.0
        
        # Run Inference
        occlusion_mask = self.xseg_session.run(None, {self.xseg_session.get_inputs()[0].name: prepare_vision_frame})[0][0]
        
        # xseg_1 output might be (H, W, 1) or (1, H, W, 1). We took [0][0], so we likely have (H, W, 1) or similar.
        # Let's ensure it's (H, W)
        if occlusion_mask.ndim == 3 and occlusion_mask.shape[-1] == 1:
            occlusion_mask = np.squeeze(occlusion_mask, axis=-1)
            
        occlusion_mask = occlusion_mask.clip(0, 1).astype(np.float32)
        occlusion_mask = cv2.resize(occlusion_mask, crop_vision_frame.shape[:2][::-1])
        
        # Soften edges
        occlusion_mask = (cv2.GaussianBlur(occlusion_mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
        return occlusion_mask
        
    def create_region_mask(self, crop_vision_frame: np.ndarray, regions: List[str] = None) -> np.ndarray:
        """
        Create a mask for specific facial regions using bisenet_resnet_34.
        Default regions usually exclude hair and background to avoid pasting them.
        """
        if regions is None:
            # Default regions to keep (everything except hair, hat, cloth, background, etc)
            regions = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'nose', 'mouth', 'u_lip', 'l_lip']
            
        model_size = (512, 512)
        
        # Prepare tensor (expects NCHW, RGB, normalized with mean/std)
        prepare_vision_frame = cv2.resize(crop_vision_frame, model_size)
        prepare_vision_frame = prepare_vision_frame[:, :, ::-1].astype(np.float32) / 255.0
        prepare_vision_frame = np.subtract(prepare_vision_frame, np.array([0.485, 0.456, 0.406], dtype=np.float32))
        prepare_vision_frame = np.divide(prepare_vision_frame, np.array([0.229, 0.224, 0.225], dtype=np.float32))
        
        prepare_vision_frame = np.expand_dims(prepare_vision_frame, axis=0)
        prepare_vision_frame = prepare_vision_frame.transpose(0, 3, 1, 2)
        
        # Run Inference
        region_prediction = self.bisenet_session.run(None, {self.bisenet_session.get_inputs()[0].name: prepare_vision_frame})[0][0]
        
        # Output is (19, 512, 512) representing logits per class. argmax(0) gives (512, 512) with class indices.
        class_indices = region_prediction.argmax(axis=0)
        
        # Filter selected regions
        target_indices = [self.region_mapping[r] for r in regions if r in self.region_mapping]
        region_mask = np.isin(class_indices, target_indices).astype(np.float32)
        
        region_mask = cv2.resize(region_mask, crop_vision_frame.shape[:2][::-1])
        region_mask = (cv2.GaussianBlur(region_mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
        return region_mask
        
    def create_box_mask(self, crop_vision_frame: np.ndarray, padding: List[int] = [0, 0, 0, 0], blur: float = 0.3) -> np.ndarray:
        """
        Create a rectangular box mask with blurred edges to seamlessly blend the crop.
        padding: [top, right, bottom, left] in percentages.
        """
        crop_size = crop_vision_frame.shape[:2][::-1]
        blur_amount = int(crop_size[0] * 0.5 * blur)
        blur_area = max(blur_amount // 2, 1)
        
        box_mask = np.ones((crop_size[1], crop_size[0]), dtype=np.float32)
        
        # Apply padding (percentage based)
        box_mask[:max(blur_area, int(crop_size[1] * padding[0] / 100)), :] = 0
        box_mask[-max(blur_area, int(crop_size[1] * padding[2] / 100)):, :] = 0
        box_mask[:, :max(blur_area, int(crop_size[0] * padding[3] / 100))] = 0
        box_mask[:, -max(blur_area, int(crop_size[0] * padding[1] / 100)):] = 0
        
        if blur_amount > 0:
            box_mask = cv2.GaussianBlur(box_mask, (0, 0), blur_amount * 0.25)
            
        return box_mask

    def create_eyes_mask(self, crop_vision_frame: np.ndarray, target_face, affine_matrix) -> np.ndarray:
        """
        Create a priority mask for eyes to force them to be pasted, overriding subtractive masks.
        Transforms target landmarks into the crop space.
        """
        crop_size = crop_vision_frame.shape[:2][::-1]
        mask = np.zeros((crop_size[1], crop_size[0]), dtype=np.float32)
        
        if target_face is None or not hasattr(target_face, "landmark_5") or affine_matrix is None:
            return mask
            
        landmarks = target_face.landmark_5
        crop_landmarks = cv2.transform(np.array([landmarks]), affine_matrix)[0]
        
        left_eye = crop_landmarks[0]
        right_eye = crop_landmarks[1]
        
        eye_dist = float(np.linalg.norm(left_eye - right_eye))
        radius_x = max(10, int(eye_dist * 0.42))
        radius_y = max(8, int(radius_x * 0.65))
        
        for eye in (left_eye, right_eye):
            center = (int(eye[0]), int(eye[1]))
            cv2.ellipse(mask, center, (radius_x, radius_y), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
            
        # Adaptive blur based on crop size
        blur_amount = max(3, int(crop_size[0] * 0.05))
        if blur_amount % 2 == 0: blur_amount += 1
        
        mask = cv2.GaussianBlur(mask, (blur_amount, blur_amount), blur_amount * 0.2)
        mask = np.clip(mask, 0.0, 1.0)
        return mask

    def get_combined_mask(self, crop_vision_frame: np.ndarray, mask_types: List[str] = None, target_face = None, affine_matrix = None) -> np.ndarray:
        """
        Match FaceFusion exactly: dynamically reduce enabled mask types.
        FaceFusion default is ['box'].
        """
        if mask_types is None:
            mask_types = ['box']
            
        crop_masks = []
        
        if 'box' in mask_types:
            crop_masks.append(self.create_box_mask(crop_vision_frame))
            
        if 'occlusion' in mask_types:
            crop_masks.append(self.create_occlusion_mask(crop_vision_frame))
            
        if 'region' in mask_types:
            crop_masks.append(self.create_region_mask(crop_vision_frame))
            
        if not crop_masks:
            combined_mask = np.ones(crop_vision_frame.shape[:2], dtype=np.float32)
        else:
            combined_mask = np.minimum.reduce(crop_masks).clip(0, 1)
            
        # Priority additive masks (e.g., eyes)
        if 'eyes' in mask_types:
            eyes_mask = self.create_eyes_mask(crop_vision_frame, target_face, affine_matrix)
            combined_mask = np.maximum(combined_mask, eyes_mask)
            
        return combined_mask

# Export a default instance
parser_app = MaskParser()

def get_combined_mask(crop_vision_frame: np.ndarray, mask_types: List[str] = None, target_face = None, affine_matrix = None) -> np.ndarray:
    return parser_app.get_combined_mask(crop_vision_frame, mask_types, target_face, affine_matrix)
