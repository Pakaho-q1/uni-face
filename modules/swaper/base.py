from abc import ABC, abstractmethod
import numpy as np
from core.types import Face

class BaseSwaper(ABC):
    """
    Abstract interface for all face swapper backends.
    Any new backend (e.g. simswap, hyperswap) must implement this interface.
    """
    
    @abstractmethod
    def swap(self, source_face: Face, target_face: Face, temp_vision_frame: np.ndarray) -> np.ndarray:
        """
        Swap the target_face in temp_vision_frame with the source_face.
        
        Args:
            source_face: The Face object representing the identity to be swapped in.
            target_face: The Face object representing the destination face in the frame.
            temp_vision_frame: The full frame image (BGR) containing the target_face.
            
        Returns:
            np.ndarray: The modified full frame image with the swapped face.
        """
        pass
