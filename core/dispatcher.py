from core.state import SwapState

def run_pipeline(state: SwapState):
    """
    Central flow for faceswap processing.
    Pipeline: detector -> parser -> swaper -> restorer -> compositor -> output
    """
    print(f"Starting pipeline in {state.mode} mode using backend {state.backend}")
    print(f"Source: {state.source_path}")
    print(f"Target: {state.target_path}")
    print(f"Output: {state.output_path}")
    
    # TODO: Initialize modules dynamically based on backend/mode
    # For now, just a placeholder.
    
    print("Pipeline execution completed.")
