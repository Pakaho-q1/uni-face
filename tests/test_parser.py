import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from modules.parser import MaskParser, get_combined_mask

class TestMaskParser(unittest.TestCase):

    @patch('onnxruntime.InferenceSession')
    def test_create_occlusion_mask(self, mock_ort_session):
        mock_xseg_session = MagicMock()
        mock_bisenet_session = MagicMock()
        mock_ort_session.side_effect = [mock_xseg_session, mock_bisenet_session]
        
        # Mock xseg output (H, W, 1) or (1, H, W, 1) depending on model, parser takes [0][0]
        # Let's mock [0][0] as returning a 256x256x1 array
        dummy_xseg_out = np.ones((256, 256, 1), dtype=np.float32)
        
        mock_run_result = MagicMock()
        mock_run_result.__getitem__.return_value = MagicMock()
        mock_run_result.__getitem__().__getitem__.return_value = dummy_xseg_out
        # Or simpler:
        mock_xseg_session.run.return_value = [[dummy_xseg_out]]
        
        parser = MaskParser()
        crop_vision_frame = np.zeros((128, 128, 3), dtype=np.uint8)
        
        mask = parser.create_occlusion_mask(crop_vision_frame)
        
        self.assertEqual(mask.shape, (128, 128)) # Should be resized back to crop size
        mock_xseg_session.run.assert_called_once()
        
    @patch('onnxruntime.InferenceSession')
    def test_create_region_mask(self, mock_ort_session):
        mock_xseg_session = MagicMock()
        mock_bisenet_session = MagicMock()
        mock_ort_session.side_effect = [mock_xseg_session, mock_bisenet_session]
        
        # Mock bisenet output (19, 512, 512)
        dummy_bisenet_out = np.zeros((19, 512, 512), dtype=np.float32)
        # Class 1 is 'skin'. Let's set some pixels to class 1
        dummy_bisenet_out[1, 100:200, 100:200] = 10.0 # High logit for skin
        
        mock_bisenet_session.run.return_value = [[dummy_bisenet_out]]
        
        parser = MaskParser()
        crop_vision_frame = np.zeros((128, 128, 3), dtype=np.uint8)
        
        mask = parser.create_region_mask(crop_vision_frame, regions=['skin'])
        
        self.assertEqual(mask.shape, (128, 128))
        mock_bisenet_session.run.assert_called_once()

    def test_create_box_mask(self):
        # We don't need ONNX for box mask
        with patch('onnxruntime.InferenceSession'):
            parser = MaskParser()
            crop_vision_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            
            # padding: [top, right, bottom, left]
            mask = parser.create_box_mask(crop_vision_frame, padding=[10, 10, 10, 10], blur=0.0)
            
            self.assertEqual(mask.shape, (100, 100))
            self.assertEqual(mask[5, 50], 0.0)   # Top padding area
            self.assertEqual(mask[95, 50], 0.0)  # Bottom padding area
            self.assertEqual(mask[50, 50], 1.0)  # Center area

if __name__ == '__main__':
    unittest.main()
