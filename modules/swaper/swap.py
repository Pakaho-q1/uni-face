import numpy as np
from core.types import Face
from core.state import state

import threading

swapper_app = None
_lock = threading.Lock()

def get_swapper():
    global swapper_app
    if swapper_app is None:
        with _lock:
            if swapper_app is None:
                model_key = state.swap_model
                if 'hyperswap' in model_key:
                    from modules.swaper.hyperswap import Hyperswap
                    swapper_app = Hyperswap()
                else:
                    from modules.swaper.inswapper import Inswapper
                    swapper_app = Inswapper()
    return swapper_app

def swap(source_face: Face, target_face: Face, frame: np.ndarray) -> np.ndarray:
    return get_swapper().swap(source_face, target_face, frame)
