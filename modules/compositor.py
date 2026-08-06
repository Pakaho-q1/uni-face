import cv2
import numpy as np

from core.types import Face
from modules.utils import face_math
from modules.parser import get_combined_mask

class FaceCompositor:
    """
    Native implementation for Color Matching and Seamless Compositing.
    Matches the skin tone of the swapped face to the original face, and applies feathered masking.
    """
    def __init__(self):
        # GFPGAN/GPEN uses ffhq_512, swapper usually uses arcface_128 or ffhq_512
        # We will work dynamically based on the input frame's crop size
        pass

    def equalize_color(self, source_crop: np.ndarray, target_crop: np.ndarray, size: tuple) -> np.ndarray:
        """
        Calculates the color difference at a specific low resolution (size) and adds it back to the target.
        """
        source_resized = cv2.resize(source_crop, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        target_resized = cv2.resize(target_crop, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        
        diff = np.subtract(source_resized, target_resized)
        diff_upscaled = cv2.resize(diff, target_crop.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
        
        equalized = np.add(target_crop, diff_upscaled).clip(0, 255).astype(np.uint8)
        return equalized

    def calculate_histogram_difference(self, source_crop: np.ndarray, target_crop: np.ndarray) -> float:
        """
        Compares the HSV histograms of source and target to determine how much blending is needed.
        """
        hist_source = cv2.calcHist([cv2.cvtColor(source_crop, cv2.COLOR_BGR2HSV)], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist_target = cv2.calcHist([cv2.cvtColor(target_crop, cv2.COLOR_BGR2HSV)], [0, 1], None, [50, 60], [0, 180, 0, 256])
        
        correlation = cv2.compareHist(hist_source, hist_target, cv2.HISTCMP_CORREL)
        # Interpolate correlation from [-1, 1] to [0, 1]
        hist_diff = float(np.interp(correlation, [-1, 1], [0, 1]))
        return hist_diff

    def match_color(self, source_crop: np.ndarray, target_crop: np.ndarray) -> np.ndarray:
        """
        Matches the color of target_crop to source_crop using multi-scale equalization.
        """
        h, w = target_crop.shape[:2]
        # Use multiple scales to match overall lighting and local color tone
        sizes = np.linspace(16, h, 3, endpoint=False).astype(int)
        
        matched_crop = target_crop.copy()
        for size in sizes:
            source_crop = self.equalize_color(source_crop, matched_crop, (size, size))
            
        matched_crop = self.equalize_color(source_crop, matched_crop, (w, h))
        return matched_crop

    def conditional_match_color(self, source_crop: np.ndarray, target_crop: np.ndarray) -> np.ndarray:
        """
        Conditionally matches color based on histogram differences to prevent over-correction.
        """
        hist_factor = self.calculate_histogram_difference(source_crop, target_crop)
        matched = self.match_color(source_crop, target_crop)
        
        # Blend the matched crop with the original target_crop based on the histogram factor
        final_matched = cv2.addWeighted(target_crop, 1 - hist_factor, matched, hist_factor, 0)
        return final_matched

    def composite(self, target_face: Face, swapped_frame: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        """
        Composites the swapped face seamlessly back into the original frame.
        
        Args:
            target_face: The original target Face object
            swapped_frame: The full frame containing the swapped (and optionally restored) face
            original_frame: The original untouched full frame
            
        Returns:
            The final composited frame.
        """
        # 1. Extract the swapped crop (which may have GFPGAN enhancement) and its affine matrix
        # Since GFPGAN/Restorer uses 512x512 ffhq_512, we will use that as standard for high-res composite
        template = 'ffhq_512'
        crop_size = (512, 512)
        
        swapped_crop, affine_matrix = face_math.warp_face_by_face_landmark_5(
            swapped_frame, 
            target_face.landmark_5, 
            template, 
            crop_size
        )
        
        # 2. Extract the original target crop (to serve as the color reference)
        original_crop, _ = face_math.warp_face_by_face_landmark_5(
            original_frame, 
            target_face.landmark_5, 
            template, 
            crop_size
        )
        
        # 3. Match color of swapped crop to original crop
        # Note: FaceFusion matches target to source. We want to match our swapped crop (which is our new 'source') 
        # to the original crop (which is our 'target' environment lighting)
        # So we want swapped_crop to look like original_crop.
        # Thus, conditional_match_color(original_crop, swapped_crop)
        matched_crop = self.conditional_match_color(original_crop, swapped_crop)
        
        # 4. Generate the mask for the matched crop
        # The mask determines which parts of the face we keep (excluding hair, background, occlusions)
        crop_mask = get_combined_mask(matched_crop)
        
        # Apply extra Gaussian blur to the mask for softer feathering at the edges
        blur_amount = int(crop_size[0] * 0.1) # 10% of crop size
        if blur_amount % 2 == 0:
            blur_amount += 1
        crop_mask = cv2.GaussianBlur(crop_mask, (blur_amount, blur_amount), 0)
        
        # 5. Paste the matched and feathered crop back onto the original frame!
        final_frame = face_math.paste_back(original_frame, matched_crop, crop_mask, affine_matrix)
        
        return final_frame

# Export default instance
compositor_app = FaceCompositor()

def composite(target_face: Face, swapped_frame: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
    return compositor_app.composite(target_face, swapped_frame, original_frame)
