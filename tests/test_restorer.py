import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from core.types import Face
from modules.restorer import FaceRestorer

class TestFaceRestorer(unittest.TestCase):

    @patch('onnxruntime.InferenceSession')
    def test_restorer_initialization(self, mock_ort_session):
        mock_session = MagicMock()
        # Mock inputs
        mock_input = MagicMock()
        mock_input.name = 'input'
        mock_session.get_inputs.return_value = [mock_input]
        
        mock_ort_session.return_value = mock_session
        
        restorer = FaceRestorer(model_name='gfpgan_1.4')
        
        mock_ort_session.assert_called_once()
        self.assertFalse(restorer.has_weight)

    @patch('modules.restorer.get_combined_mask')
    @patch('onnxruntime.InferenceSession')
    def test_restore_execution(self, mock_ort_session, mock_get_combined_mask):
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = 'input'
        mock_session.get_inputs.return_value = [mock_input]
        mock_ort_session.return_value = mock_session
        
        # GFPGAN/GPEN output shape (1, 3, 512, 512) for single batch
        dummy_enhanced_crop = np.zeros((3, 512, 512), dtype=np.float32)
        mock_session.run.return_value = [[dummy_enhanced_crop]]
        
        # Mask shape (512, 512)
        mock_get_combined_mask.return_value = np.ones((512, 512), dtype=np.float32)
        
        restorer = FaceRestorer(model_name='gfpgan_1.4')
        
        target_face = Face(
            bbox=np.array([50, 50, 150, 150]),
            landmark_5=np.array([[30,30], [70,30], [50,50], [40,70], [60,70]], dtype=np.float32),
            landmark_106=np.zeros((68, 2), dtype=np.float32),
            embedding=np.zeros((512,), dtype=np.float32)
        )
        
        temp_vision_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Test full replace
        result_frame = restorer.restore(target_face, temp_vision_frame, blend=1.0)
        
        self.assertEqual(result_frame.shape, (200, 200, 3))
        mock_session.run.assert_called_once()
        mock_get_combined_mask.assert_called_once()
        
        # Test blending
        result_frame_blended = restorer.restore(target_face, temp_vision_frame, blend=0.5)
        self.assertEqual(result_frame_blended.shape, (200, 200, 3))

if __name__ == '__main__':
    unittest.main()
