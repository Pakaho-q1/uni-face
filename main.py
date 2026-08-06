import argparse
import sys
from cli.adapter import run_cli
from api.adapter import run_server

def main():
    parser = argparse.ArgumentParser(description="Uni-Face: Faceswap Project", add_help=False)
    # We only parse the first argument to determine the subcommand
    parser.add_argument("command", choices=["cli", "serve"], help="Command to run")
    
    # Parse known args so we don't error out on CLI flags meant for the adapter
    args, unknown = parser.parse_known_args(sys.argv[1:2])
    
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "cli":
        run_cli()
    elif args.command == "serve":
        # Additional parsing for serve command (e.g. --port)
        serve_parser = argparse.ArgumentParser(prog="main.py serve")
        serve_parser.add_argument("--port", type=int, default=8000, help="Port to run the API server on")
        serve_args = serve_parser.parse_args(sys.argv[2:])
        run_server(port=serve_args.port)

if __name__ == "__main__":
    main()
