import argparse
from core.service import SwapService
from core.config import DEFAULT_BACKEND, DEFAULT_MODE

def run_cli():
    parser = argparse.ArgumentParser(description="Uni-Face CLI Adapter")
    # Note: main.py already parsed the subcommand, so this parser handles the arguments for 'cli'
    parser.add_argument("--source", type=str, required=True, help="Source face image/video")
    parser.add_argument("--target", type=str, required=True, help="Target image/video")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    parser.add_argument("--backend", type=str, default=DEFAULT_BACKEND, help="Backend to use (e.g. inswapper, simswap)")
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE, choices=["image", "video"], help="Mode of operation")
    
    # We parse remaining args after the subcommand
    import sys
    # Subcommand 'cli' is at index 1
    args = parser.parse_args(sys.argv[2:])

    try:
        result = SwapService.process_swap(
            source_path=args.source,
            target_path=args.target,
            output_path=args.output,
            backend=args.backend,
            mode=args.mode
        )
        print("Success:", result)
    except Exception as e:
        print("Error processing swap:", str(e))
