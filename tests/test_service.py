import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from core.types import Face
from core.service import FaceService

class TestFaceService(unittest.TestCase):
    
    @patch('core.service.composite')
    @patch('core.service.restore')
    @patch('core.service.swap')
    @patch('core.service.detect')
    def test_process_image_success(self, mock_detect, mock_swap, mock_restore, mock_composite):
        # Setup mock faces
        mock_source_face = Face(
            bbox=np.array([0, 0, 100, 100]),
            landmark_5=np.zeros((5, 2)),
            landmark_106=np.zeros((68, 2)),
            embedding=np.zeros((512,))
        )
        
        mock_target_face = Face(
            bbox=np.array([50, 50, 150, 150]),
            landmark_5=np.zeros((5, 2)),
            landmark_106=np.zeros((68, 2)),
            embedding=np.zeros((512,))
        )
        
        # mock_detect will return a list with one face for source, and one for target
        # Since it's called twice, we use side_effect
        mock_detect.side_effect = [[mock_source_face], [mock_target_face]]
        
        # mock_swap and mock_restore return dummy images
        dummy_swapped = np.zeros((200, 200, 3), dtype=np.uint8)
        dummy_restored = np.ones((200, 200, 3), dtype=np.uint8) * 128
        dummy_composite = np.ones((200, 200, 3), dtype=np.uint8) * 255
        
        mock_swap.return_value = dummy_swapped
        mock_restore.return_value = dummy_restored
        mock_composite.return_value = dummy_composite
        
        service = FaceService()
        source_img = np.zeros((200, 200, 3), dtype=np.uint8)
        target_img = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Execute
        result_img = service.process_image(source_img, target_img)
        
        # Assertions
        self.assertEqual(mock_detect.call_count, 2)
        mock_swap.assert_called_once_with(mock_source_face, mock_target_face, target_img)
        mock_restore.assert_called_once_with(mock_target_face, dummy_swapped, blend=0.8)
        mock_composite.assert_called_once_with(mock_target_face, dummy_restored, target_img)
        
        # Result should be the composite image
        np.testing.assert_array_equal(result_img, dummy_composite)

    @patch('core.service.detect')
    def test_process_image_no_source_face(self, mock_detect):
        # If source has no face
        mock_detect.return_value = []
        
        service = FaceService()
        source_img = np.zeros((200, 200, 3), dtype=np.uint8)
        target_img = np.ones((200, 200, 3), dtype=np.uint8) * 128
        
        result_img = service.process_image(source_img, target_img)
        
        # Should return target img unchanged
        np.testing.assert_array_equal(result_img, target_img)
        self.assertEqual(mock_detect.call_count, 1)

if __name__ == '__main__':
    unittest.main()
