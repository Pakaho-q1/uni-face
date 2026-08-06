import cv2
import numpy as np
import onnxruntime
from typing import List, Tuple

from core.types import Face
from core.config import MODEL_PATHS
from core.state import state
from modules.utils import face_math

class NativeDetector:
    """
    Native implementation of Face Detector, Landmarker, and Recognizer
    independent of any external facefusion dependency.
    """
    def __init__(self):
        # 1. Initialize ONNX Sessions directly from our paths
        self.providers = state.providers
        self.yoloface_session = onnxruntime.InferenceSession(str(MODEL_PATHS["yoloface"]), providers=self.providers)
        self.fan_session = onnxruntime.InferenceSession(str(MODEL_PATHS["2dfan4"]), providers=self.providers)
        self.arcface_session = onnxruntime.InferenceSession(str(MODEL_PATHS["arcface"]), providers=self.providers)
        
        # Configuration
        self.face_detector_size = (640, 640)
        self.face_detector_score = 0.5
        
        # arcface model template and size
        self.arcface_template = 'arcface_112_v2'
        self.arcface_size = (112, 112)
        
    def detect(self, frame: np.ndarray) -> List[Face]:
        """Detect faces, find landmarks, and extract embeddings."""
        faces = []
        
        # 1. Detect bounding boxes and 5-point landmarks
        bounding_boxes, face_scores, face_landmarks_5 = self._detect_yoloface(frame)
        
        for bbox, score, landmark_5 in zip(bounding_boxes, face_scores, face_landmarks_5):
            # 2. Refine with 2dfan4 to get 68-point landmarks
            landmark_68, _ = self._detect_2dfan4(frame, bbox)
            
            # If 2dfan4 fails or returns None, fallback to expanding the 5 points or ignore
            if landmark_68 is None:
                continue
                
            # Convert 68-point back to 5-point for arcface (more stable alignment)
            refined_landmark_5 = face_math.convert_to_face_landmark_5(landmark_68)
            
            # 3. Calculate embedding using ArcFace
            embedding = self._calculate_embedding(frame, refined_landmark_5)
            
            face_obj = Face(
                bbox=bbox,
                score=score,
                landmark_5=refined_landmark_5,
                landmark_106=landmark_68, # Store 68 points here for swapping
                embedding=embedding
            )
            faces.append(face_obj)
            
        return faces

    def _detect_yoloface(self, vision_frame: np.ndarray) -> Tuple[List[np.ndarray], List[float], List[np.ndarray]]:
        """Native implementation of yoloface inference."""
        face_detector_width, face_detector_height = self.face_detector_size
        
        # Restrict frame (scale down if larger than 640x640 while maintaining aspect ratio)
        temp_vision_frame = vision_frame.copy()
        height, width = temp_vision_frame.shape[:2]
        ratio_width = width / face_detector_width
        ratio_height = height / face_detector_height
        if ratio_width > 1 or ratio_height > 1:
            ratio = max(ratio_width, ratio_height)
            temp_vision_frame = cv2.resize(temp_vision_frame, (int(width / ratio), int(height / ratio)))
            
        ratio_height = vision_frame.shape[0] / temp_vision_frame.shape[0]
        ratio_width = vision_frame.shape[1] / temp_vision_frame.shape[1]
        
        # Pad to 640x640
        detect_vision_frame = np.zeros((face_detector_height, face_detector_width, 3), dtype=np.uint8)
        detect_vision_frame[:temp_vision_frame.shape[0], :temp_vision_frame.shape[1], :] = temp_vision_frame
        
        # Prepare tensor (CHW, float32, normalized 0-1)
        detect_vision_frame = np.expand_dims(detect_vision_frame.transpose(2, 0, 1), axis=0).astype(np.float32)
        detect_vision_frame = detect_vision_frame / 255.0
        
        # Run inference
        detection = self.yoloface_session.run(None, {self.yoloface_session.get_inputs()[0].name: detect_vision_frame})[0]
        detection = np.squeeze(detection).T
        
        bounding_boxes_raw, face_scores_raw, face_landmarks_5_raw = np.split(detection, [4, 5], axis=1)
        keep_indices = np.where(face_scores_raw > self.face_detector_score)[0]
        
        bounding_boxes = []
        face_scores = []
        face_landmarks_5 = []
        
        if keep_indices.size > 0:
            bounding_boxes_raw = bounding_boxes_raw[keep_indices]
            face_scores_raw = face_scores_raw[keep_indices]
            face_landmarks_5_raw = face_landmarks_5_raw[keep_indices]
            
            for bbox_raw in bounding_boxes_raw:
                bounding_boxes.append(np.array([
                    (bbox_raw[0] - bbox_raw[2] / 2) * ratio_width,
                    (bbox_raw[1] - bbox_raw[3] / 2) * ratio_height,
                    (bbox_raw[0] + bbox_raw[2] / 2) * ratio_width,
                    (bbox_raw[1] + bbox_raw[3] / 2) * ratio_height
                ]))
                
            face_scores = face_scores_raw.ravel().tolist()
            
            face_landmarks_5_raw[:, 0::3] = face_landmarks_5_raw[:, 0::3] * ratio_width
            face_landmarks_5_raw[:, 1::3] = face_landmarks_5_raw[:, 1::3] * ratio_height
            
            for landmark_raw_5 in face_landmarks_5_raw:
                face_landmarks_5.append(np.array(landmark_raw_5.reshape(-1, 3)[:, :2]))
                
            # Apply NMS
            nms_indices = face_math.apply_nms(bounding_boxes, face_scores, self.face_detector_score, 0.4)
            
            bounding_boxes = [bounding_boxes[i] for i in nms_indices]
            face_scores = [face_scores[i] for i in nms_indices]
            face_landmarks_5 = [face_landmarks_5[i] for i in nms_indices]
            
        return bounding_boxes, face_scores, face_landmarks_5

    def _detect_2dfan4(self, temp_vision_frame: np.ndarray, bounding_box: np.ndarray) -> Tuple[np.ndarray, float]:
        """Native implementation of 2dfan4 inference for 68 landmarks."""
        model_size = (256, 256)
        scale = 195 / np.subtract(bounding_box[2:], bounding_box[:2]).max().clip(1, None)
        translation = (model_size[0] - np.add(bounding_box[2:], bounding_box[:2]) * scale) * 0.5
        
        # Warp face by translation
        crop_vision_frame, affine_matrix = face_math.warp_face_by_translation(temp_vision_frame, translation, scale, model_size)
        
        # Optimize contrast (CLAHE on L channel)
        crop_vision_frame = cv2.cvtColor(crop_vision_frame, cv2.COLOR_RGB2Lab)
        if np.mean(crop_vision_frame[:, :, 0]) < 30:
            crop_vision_frame[:, :, 0] = cv2.createCLAHE(clipLimit=2).apply(crop_vision_frame[:, :, 0])
        crop_vision_frame = cv2.cvtColor(crop_vision_frame, cv2.COLOR_Lab2RGB)
        
        # Prepare tensor
        crop_vision_frame = crop_vision_frame.transpose(2, 0, 1).astype(np.float32) / 255.0
        crop_vision_frame = np.expand_dims(crop_vision_frame, axis=0)
        
        # Run inference
        prediction = self.fan_session.run(None, {self.fan_session.get_inputs()[0].name: crop_vision_frame})
        face_landmark_68, face_heatmap = prediction[0], prediction[1]
        
        face_landmark_68 = face_landmark_68[:, :, :2][0] / 64 * 256
        face_landmark_68 = face_math.transform_points(face_landmark_68, cv2.invertAffineTransform(affine_matrix))
        
        face_landmark_score_68 = np.amax(face_heatmap, axis=(2, 3))
        face_landmark_score_68 = np.mean(face_landmark_score_68)
        face_landmark_score_68 = np.interp(face_landmark_score_68, [0, 0.9], [0, 1])
        
        return face_landmark_68, face_landmark_score_68

    def _calculate_embedding(self, temp_vision_frame: np.ndarray, face_landmark_5: np.ndarray) -> np.ndarray:
        """Native implementation of arcface embedding extraction."""
        crop_vision_frame, matrix = face_math.warp_face_by_face_landmark_5(
            temp_vision_frame, face_landmark_5, self.arcface_template, self.arcface_size
        )
        
        # Normalize -1 to 1
        crop_vision_frame = crop_vision_frame / 127.5 - 1
        
        # Prepare tensor (BGR to RGB handled internally if needed, assuming input is BGR, insightface arcface expects BGR? 
        # Wait, Insightface arcface expects RGB usually, but facefusion does:
        # crop_vision_frame = crop_vision_frame[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32)
        crop_vision_frame = crop_vision_frame[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
        crop_vision_frame = np.expand_dims(crop_vision_frame, axis=0)
        
        # Run inference
        face_embedding = self.arcface_session.run(None, {self.arcface_session.get_inputs()[0].name: crop_vision_frame})[0]
        face_embedding = face_embedding.ravel()
        
        # Normalize vector
        face_embedding_norm = face_embedding / np.linalg.norm(face_embedding)
        return face_embedding_norm


# Export a default instance
import threading
detector_app = None
_lock = threading.Lock()

def detect(frame: np.ndarray) -> List[Face]:
    global detector_app
    if detector_app is None:
        with _lock:
            if detector_app is None:
                detector_app = NativeDetector()
    return detector_app.detect(frame)
