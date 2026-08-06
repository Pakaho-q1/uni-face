import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from modules.detector import NativeDetector, detect

class TestNativeDetector(unittest.TestCase):
    
    @patch('onnxruntime.InferenceSession')
    def test_detector_initialization(self, mock_ort_session):
        # Initializing the detector should create 3 ONNX sessions
        detector = NativeDetector()
        self.assertEqual(mock_ort_session.call_count, 3)
        
    @patch('onnxruntime.InferenceSession')
    @patch('modules.detector.face_math.apply_nms')
    def test_detect_returns_faces(self, mock_apply_nms, mock_ort_session):
        # Mock the ONNX session instances
        mock_yolo_session = MagicMock()
        mock_2dfan4_session = MagicMock()
        mock_arcface_session = MagicMock()
        
        # side_effect for the 3 creations in __init__
        mock_ort_session.side_effect = [mock_yolo_session, mock_2dfan4_session, mock_arcface_session]
        
        # 1. Mock Yolo output
        # Output should be (1, N, 15) wait, the detect code unpacks [0][0]. 
        # So we return a dummy tensor where some scores are > 0.5
        # The detect function extracts bounding boxes from the grid.
        # It's easier to mock `detect` method logic deeply or just test the tensor math.
        
        # Actually, let's test if the pipeline executes when we feed it a 640x640 dummy image.
        # The YOLO post processing might fail if the dummy tensor doesn't match expected shapes.
        # Yolo output shape: (1, 20, N) where N = 8400 usually.
        dummy_yolo_out = np.zeros((1, 20, 8400), dtype=np.float32)
        # Set one high confidence detection at index 0
        dummy_yolo_out[0, 4, 0] = 0.9 # Score
        # bbox: cx, cy, w, h
        dummy_yolo_out[0, 0, 0] = 50.0 # cx
        dummy_yolo_out[0, 1, 0] = 50.0 # cy
        dummy_yolo_out[0, 2, 0] = 20.0 # w
        dummy_yolo_out[0, 3, 0] = 20.0 # h
        # 5 landmarks (x,y,score) -> 15 values (indices 5 to 19)
        dummy_yolo_out[0, 5, 0] = 40.0 # L eye x
        dummy_yolo_out[0, 6, 0] = 40.0 # L eye y
        dummy_yolo_out[0, 7, 0] = 1.0  # score
        dummy_yolo_out[0, 8, 0] = 60.0 # R eye x
        dummy_yolo_out[0, 9, 0] = 40.0 # R eye y
        dummy_yolo_out[0, 10, 0] = 1.0
        dummy_yolo_out[0, 11, 0] = 50.0 # Nose x
        dummy_yolo_out[0, 12, 0] = 50.0 # Nose y
        dummy_yolo_out[0, 13, 0] = 1.0
        dummy_yolo_out[0, 14, 0] = 45.0 # L mouth x
        dummy_yolo_out[0, 15, 0] = 60.0 # L mouth y
        dummy_yolo_out[0, 16, 0] = 1.0
        dummy_yolo_out[0, 17, 0] = 55.0 # R mouth x
        dummy_yolo_out[0, 18, 0] = 60.0 # R mouth y
        dummy_yolo_out[0, 19, 0] = 1.0
        
        mock_yolo_session.run.return_value = [dummy_yolo_out]
        
        mock_apply_nms.return_value = [0] # Keep index 0
        
        # 2. Mock 2dfan4 output
        # Output is (1, 68, 3) for landmarks and (1, 68, 64, 64) for heatmaps
        dummy_heatmap = np.zeros((1, 68, 64, 64), dtype=np.float32)
        dummy_heatmap[0, :, 32, 32] = 1.0
        
        dummy_landmarks = np.zeros((1, 68, 3), dtype=np.float32)
        # Set some points to avoid colinearity (we only need the 68 points to not be all zeros)
        # The first 5 points are roughly used to convert back to 5 points
        for i in range(68):
            dummy_landmarks[0, i, 0] = float(i)
            dummy_landmarks[0, i, 1] = float(i % 10)
            
        mock_2dfan4_session.run.return_value = [dummy_landmarks, dummy_heatmap]
        
        # 3. Mock arcface output
        # Output is (1, 512)
        dummy_embedding = np.ones((1, 512), dtype=np.float32)
        mock_arcface_session.run.return_value = [dummy_embedding]
        
        detector = NativeDetector()
        
        # Create a dummy image
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Detect
        faces = detector.detect(frame)
        
        # Assertions
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0].embedding.shape, (512,))
        self.assertEqual(faces[0].landmark_106.shape, (68, 2))
        self.assertEqual(faces[0].landmark_5.shape, (5, 2))
        
        # Check if sessions were called
        mock_yolo_session.run.assert_called_once()
        mock_2dfan4_session.run.assert_called_once()
        mock_arcface_session.run.assert_called_once()

if __name__ == '__main__':
    unittest.main()
