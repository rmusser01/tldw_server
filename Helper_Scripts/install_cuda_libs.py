import os
import sys
import subprocess
import platform
import torch

def run_command(command, env_path):
    """Run a system command within the active Conda environment."""
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True, env=env_path
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e.stderr}")
        return None

def get_conda_env():
    """Retrieve the current active Conda environment path."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        print("Error: No active Conda environment detected! Activate a Conda environment and rerun the script.")
        sys.exit(1)
    return conda_prefix

def check_cuda():
    """Check if CUDA is available using PyTorch."""
    return torch.cuda.is_available()

def install_pytorch_cuda(conda_env_path):
    """Uninstalls CPU-only PyTorch and installs the GPU version in the active Conda environment."""
    print(f"CUDA detected! Upgrading PyTorch to the GPU version in the Conda environment: {conda_env_path}")

    # Set the environment path for the subprocess
    env_path = os.environ.copy()
    env_path["PATH"] = os.path.join(conda_env_path, "bin") + os.pathsep + env_path["PATH"]

    # Uninstall CPU-only PyTorch
    run_command("pip uninstall -y torch torchvision torchaudio", env_path)

    # Identify OS
    os_name = platform.system()

    # Install CUDA version
    if os_name == "Windows":
        run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121", env_path)
    elif os_name == "Linux":
        run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121", env_path)
    elif os_name == "Darwin":  # macOS
        print("CUDA is not supported on macOS. Keeping CPU version of PyTorch.")
        return
    else:
        print(f"Unsupported OS: {os_name}. Please install manually.")
        return

    print("✅ PyTorch has been upgraded to the CUDA version in the active Conda environment!")

def main():
    print("🔍 Checking for CUDA support...")
    conda_env_path = get_conda_env()

    # Warn if the user is in the base Conda environment
    if "base" in conda_env_path:
        print("⚠ Warning: You are in the base Conda environment! It's recommended to use a separate Conda environment for this setup.")
    
    if check_cuda():
        install_pytorch_cuda(conda_env_path)
    else:
        print("❌ No CUDA detected. Keeping the CPU version of PyTorch.")

if __name__ == "__main__":
    main()
