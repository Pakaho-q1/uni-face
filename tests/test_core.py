import unittest
from core.state import SwapState
from core.dispatcher import run_pipeline

class TestCore(unittest.TestCase):
    def test_state_creation(self):
        state = SwapState(source_path="src.jpg", target_path="tgt.jpg", output_path="out.jpg")
        self.assertEqual(state.source_path, "src.jpg")
        self.assertEqual(state.backend, "inswapper")

    def test_dispatcher(self):
        state = SwapState(source_path="src.jpg", target_path="tgt.jpg", output_path="out.jpg")
        # Should run without throwing errors
        run_pipeline(state)

if __name__ == '__main__':
    unittest.main()
