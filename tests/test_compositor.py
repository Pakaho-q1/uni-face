import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from core.types import Face
from modules.compositor import compositor_app, composite

class TestCompositor(unittest.TestCase):
    def setUp(self):
        # Create dummy crops (512x512)
        self.source_crop = np.full((512, 512, 3), 100, dtype=np.uint8) # Darker face
        self.target_crop = np.full((512, 512, 3), 200, dtype=np.uint8) # Lighter face
        
        # Create dummy target face
        self.target_face = Face(
            bbox=np.array([100, 100, 300, 300]),
            landmark_106=np.zeros((106, 2)),
            landmark_5=np.array([[120, 150], [280, 150], [200, 200], [150, 250], [250, 250]]),
            embedding=np.zeros((512,)),
            gender=1,
            age=30
        )
        
        # Create full frames (1024x1024)
        self.original_frame = np.full((1024, 1024, 3), 255, dtype=np.uint8)
        self.swapped_frame = np.full((1024, 1024, 3), 128, dtype=np.uint8)

    def test_equalize_color(self):
        # Should add color differences based on resized inputs
        result = compositor_app.equalize_color(self.source_crop, self.target_crop, (64, 64))
        self.assertEqual(result.shape, (512, 512, 3))
        # Since source is 100, target is 200: diff = -100
        # equalized = target(200) + diff(-100) = 100
        self.assertTrue(np.all(result == 100))

    def test_calculate_histogram_difference(self):
        # Same image should yield 1.0 correlation (hist_diff = 1.0 after interp)
        diff = compositor_app.calculate_histogram_difference(self.source_crop, self.source_crop)
        self.assertAlmostEqual(diff, 1.0, places=5)
        
    def test_match_color(self):
        matched = compositor_app.match_color(self.source_crop, self.target_crop)
        self.assertEqual(matched.shape, (512, 512, 3))

    @patch('modules.compositor.face_math')
    @patch('modules.compositor.get_combined_mask')
    def test_composite(self, mock_get_mask, mock_face_math):
        # Mock warp
        dummy_crop = np.zeros((512, 512, 3), dtype=np.uint8)
        dummy_matrix = np.eye(2, 3, dtype=np.float32)
        mock_face_math.warp_face_by_face_landmark_5.return_value = (dummy_crop, dummy_matrix)
        
        # Mock mask
        dummy_mask = np.ones((512, 512), dtype=np.float32)
        mock_get_mask.return_value = dummy_mask
        
        # Mock paste
        dummy_pasted = np.ones((1024, 1024, 3), dtype=np.uint8) * 50
        mock_face_math.paste_back.return_value = dummy_pasted
        
        result = composite(self.target_face, self.swapped_frame, self.original_frame)
        
        # Verify result
        self.assertEqual(result.shape, (1024, 1024, 3))
        
        # Verify warp was called twice (for swapped and original)
        self.assertEqual(mock_face_math.warp_face_by_face_landmark_5.call_count, 2)
        
        # Verify mask was retrieved
        mock_get_mask.assert_called_once()
        
        # Verify paste was called
        mock_face_math.paste_back.assert_called_once()

if __name__ == '__main__':
    unittest.main()
