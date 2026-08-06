import os
import sys
import subprocess

def run_command(command, cwd=None):
    print(f"> {' '.join(command)}")
    try:
        subprocess.check_call(command, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        sys.exit(1)

def install_python_deps():
    print("--- Installing Python Dependencies ---")
    
    # Base requirements
    reqs = ["-r", "requirements.txt"]
    
    # Basic install
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # If cuda or tensorrt is requested, we should make sure onnxruntime-gpu is installed
    # (By default requirements.txt just has onnxruntime, but pip will resolve it if we override)
    if "cuda" in " ".join(sys.argv).lower() or "tensorrt" in " ".join(sys.argv).lower():
        print("Hardware acceleration detected (CUDA/TensorRT). Ensuring GPU packages are installed...")
        run_command([sys.executable, "-m", "pip", "install", "onnxruntime-gpu"])
    
    run_command([sys.executable, "-m", "pip", "install"] + reqs)

def main():
    print("===========================================")
    print("      Uni-Face Native Installer")
    print("===========================================")
    install_python_deps()
    print("===========================================")
    print(" Installation Complete! 🎉")
    print(" You can now run the server with:")
    print(" python api_server.py")
    print("===========================================")

if __name__ == "__main__":
    main()
