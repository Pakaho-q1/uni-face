import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from core.types import Face
from modules.swaper.inswapper import Inswapper

class TestInswapper(unittest.TestCase):

    @patch('modules.swaper.inswapper.onnx')
    @patch('onnxruntime.InferenceSession')
    def test_inswapper_initialization(self, mock_ort_session, mock_onnx):
        mock_model = MagicMock()
        mock_initializer = MagicMock()
        mock_model.graph.initializer = [mock_initializer]
        mock_onnx.load.return_value = mock_model
        
        swapper = Inswapper()
        
        mock_ort_session.assert_called_once()
        mock_onnx.load.assert_called_once()

    @patch('modules.parser.get_combined_mask')
    @patch('modules.swaper.inswapper.onnx')
    @patch('onnxruntime.InferenceSession')
    def test_swap_execution(self, mock_ort_session, mock_onnx, mock_get_combined_mask):
        mock_model = MagicMock()
        mock_onnx.load.return_value = mock_model
        
        # Mock ONNX helper for initializer to avoid errors
        mock_onnx.numpy_helper.to_array.return_value = np.ones((512, 512), dtype=np.float32)
        
        # Mock Session
        mock_session = MagicMock()
        mock_ort_session.return_value = mock_session
        
        # Mock returned crop from ONNX (CHW format)
        dummy_swapped_crop = np.zeros((3, 128, 128), dtype=np.float32)
        mock_session.run.return_value = [[dummy_swapped_crop]]
        
        # Mock mask from parser (HW format)
        mock_get_combined_mask.return_value = np.ones((128, 128), dtype=np.float32)
        
        swapper = Inswapper()
        
        source_face = Face(
            bbox=np.array([0, 0, 100, 100]),
            landmark_5=np.zeros((5, 2), dtype=np.float32),
            landmark_106=np.zeros((68, 2), dtype=np.float32),
            embedding=np.ones((512,), dtype=np.float32)
        )
        
        target_face = Face(
            bbox=np.array([50, 50, 150, 150]),
            landmark_5=np.array([[30,30], [70,30], [50,50], [40,70], [60,70]], dtype=np.float32), # Non-zero to avoid singular matrix
            landmark_106=np.zeros((68, 2), dtype=np.float32),
            embedding=np.zeros((512,), dtype=np.float32)
        )
        
        temp_vision_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        
        result_frame = swapper.swap(source_face, target_face, temp_vision_frame)
        
        self.assertEqual(result_frame.shape, (200, 200, 3))
        mock_session.run.assert_called_once()
        mock_get_combined_mask.assert_called_once()

if __name__ == '__main__':
    unittest.main()
