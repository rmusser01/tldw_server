# Local_LLM_Inference_Engine_Lib.py
#########################################
# Local LLM Inference Engine Library
# This library is used to handle downloading, configuring, and launching the Local LLM Inference Engine
#   via (llama.cpp via llamafile)
#
#
####
####################
# Function List
#
# 1.
#
####################
# Import necessary libraries
#import atexit
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
#
# Import 3rd-pary Libraries
import requests
#
# Import Local
from App_Function_Libraries.Utils.Utils import download_file, logging

#
#######################################################################################################################
# Function Definitions:


###############################################################
# LLM models information

llm_models = {
    "Mistral-7B-Instruct-v0.2-Q8.llamafile": {
        "name": "Mistral-7B-Instruct-v0.2-Q8.llamafile",
        "url": "https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.2-llamafile/resolve/main/mistral-7b-instruct-v0.2.Q8_0.llamafile?download=true",
        "filename": "mistral-7b-instruct-v0.2.Q8_0.llamafile",
        "hash": "1ee6114517d2f770425c880e5abc443da36b193c82abec8e2885dd7ce3b9bfa6"
    },
    "Samantha-Mistral-Instruct-7B-Bulleted-Notes-Q8.gguf": {
        "name": "Samantha-Mistral-Instruct-7B-Bulleted-Notes-Q8.gguf",
        "url": "https://huggingface.co/cognitivetech/samantha-mistral-instruct-7b-bulleted-notes-GGUF/resolve/main/samantha-mistral-instruct-7b-bulleted-notes.Q8_0.gguf?download=true",
        "filename": "samantha-mistral-instruct-7b-bulleted-notes.Q8_0.gguf",
        "hash": "6334c1ab56c565afd86535271fab52b03e67a5e31376946bce7bf5c144e847e4"
    },
    "Phi-3-mini-128k-instruct-Q8_0.gguf": {
        "name": "Phi-3-mini-128k-instruct-Q8_0.gguf",
        "url": "https://huggingface.co/gaianet/Phi-3-mini-128k-instruct-GGUF/resolve/main/Phi-3-mini-128k-instruct-Q8_0.gguf?download=true",
        "filename": "Phi-3-mini-128k-instruct-Q8_0.gguf",
        "hash": "6817b66d1c3c59ab06822e9732f0e594eea44e64cae2110906eac9d17f75d193"
    },
    "Meta-Llama-3-8B-Instruct.Q8_0.llamafile": {
        "name": "Meta-Llama-3-8B-Instruct.Q8_0.llamafile",
        "url": "https://huggingface.co/Mozilla/Meta-Llama-3-8B-Instruct-llamafile/resolve/main/Meta-Llama-3-8B-Instruct.Q8_0.llamafile?download=true",
        "filename": "Meta-Llama-3-8B-Instruct.Q8_0.llamafile",
        "hash": "406868a97f02f57183716c7e4441d427f223fdbc7fa42964ef10c4d60dd8ed37"
    }
}
#
###############################################################

# Function to download the latest llamafile from the Mozilla-Ocho/llamafile repo
def download_latest_llamafile(output_filename: str) -> str:
    """
    Downloads the latest llamafile binary from the Mozilla-Ocho/llamafile GitHub repository.
    """
    logging.info("Checking for and downloading Llamafile if it doesn't already exist...")
    if os.path.exists(output_filename):
        logging.debug(f"{output_filename} already exists. Skipping download.")
        return os.path.abspath(output_filename)

    repo = "Mozilla-Ocho/llamafile"
    asset_name_prefix = "llamafile-"
    latest_release_url = f"https://api.github.com/repos/{repo}/releases/latest"
    response = requests.get(latest_release_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch latest release info: {response.status_code}")

    latest_release_data = response.json()
    tag_name = latest_release_data['tag_name']

    release_details_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag_name}"
    response = requests.get(release_details_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch release details for tag {tag_name}: {response.status_code}")

    release_data = response.json()
    assets = release_data.get('assets', [])

    asset_url = None
    for asset in assets:
        if re.match(f"{asset_name_prefix}.*", asset['name']):
            asset_url = asset['browser_download_url']
            break

    if not asset_url:
        raise Exception(f"No asset found with prefix {asset_name_prefix}")

    logging.info("Downloading Llamafile...")
    download_file(asset_url, output_filename)

    logging.debug(f"Downloaded {output_filename} from {asset_url}")
    return os.path.abspath(output_filename)

def download_llm_model(model_name: str, model_url: str, model_filename: str, model_hash: str) -> str:
    """
    Downloads the specified LLM model if not already present.
    """
    logging.info(f"Checking availability of model: {model_name}")
    if os.path.exists(model_filename):
        logging.debug(f"Model '{model_name}' already exists. Skipping download.")
        return os.path.abspath(model_filename)

    logging.info(f"Downloading model: {model_name}")
    download_file(model_url, model_filename, expected_checksum=model_hash)
    logging.debug(f"Downloaded model '{model_name}' successfully.")
    return os.path.abspath(model_filename)

def launch_in_new_terminal(executable: str, args: List[str]) -> subprocess.Popen:
    """
    Launches the executable in a new terminal window based on the operating system.
    Returns the subprocess.Popen object.
    """
    useros = os.name
    if useros == "nt":
        # For Windows
        args_str = ' '.join(args)
        command = f'start cmd /k "{executable} {args_str}"'
    elif useros == "posix":
        # For Linux (assuming GNOME Terminal; adjust if necessary)
        args_str = ' '.join(args)
        command = f'gnome-terminal -- bash -c "{executable} {args_str}; exec bash"'
    else:
        # For macOS
        args_str = ' '.join(args)
        command = f'open -a Terminal.app "{executable}" --args {args_str}'

    try:
        process = subprocess.Popen(command, shell=True)
        logging.info(f"Launched {executable} with arguments: {args}")
        return process
    except Exception as e:
        logging.error(f"Failed to launch the process: {e}")
        raise

# Function to scan the directory for .gguf and .llamafile files
def get_gguf_llamafile_files(directory: str) -> List[str]:
    """
    Retrieves model files with extensions .gguf or .llamafile from the specified directory.
    """
    logging.debug(f"Scanning directory: {directory}")  # Debug print for directory

    try:
        dir_path = Path(directory)
        all_files = list(dir_path.iterdir())
        logging.debug(f"All files in directory: {[str(f) for f in all_files]}")
    except Exception as e:
        logging.error(f"Failed to list files in directory {directory}: {e}")
        return []

    try:
        gguf_files = list(dir_path.glob("*.gguf"))
        llamafile_files = list(dir_path.glob("*.llamafile"))

        logging.debug(f"Found .gguf files: {[str(f) for f in gguf_files]}")
        logging.debug(f"Found .llamafile files: {[str(f) for f in llamafile_files]}")

        return [f.name for f in gguf_files + llamafile_files]
    except Exception as e:
        logging.error(f"Error during glob operations in directory {directory}: {e}")
        return []


# Initialize process with type annotation
process: Optional[subprocess.Popen] = None
# Function to close out llamafile process on script exit.
def cleanup_process() -> None:
    """
    Terminates the external llamafile process if it is running.
    """
    global process
    if process is not None:
        process.kill()
        logging.debug("Terminated the external process")
        process = None  # Reset the process variable after killing

def signal_handler(sig, frame):
    """
    Handles termination signals to ensure the subprocess is cleaned up.
    """
    logging.info('Signal handler called with signal: %s', sig)
    cleanup_process()
    sys.exit(0)

# Register signal handlers
def setup_signal_handlers():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

setup_signal_handlers()

def start_llamafile(
    am_noob: bool,
    verbose_checked: bool,
    threads_checked: bool,
    threads_value: Optional[int],
    threads_batched_checked: bool,
    threads_batched_value: Optional[int],
    model_alias_checked: bool,
    model_alias_value: str,
    http_threads_checked: bool,
    http_threads_value: Optional[int],
    model_value: str,
    hf_repo_checked: bool,
    hf_repo_value: str,
    hf_file_checked: bool,
    hf_file_value: str,
    ctx_size_checked: bool,
    ctx_size_value: Optional[int],
    ngl_checked: bool,
    ngl_value: Optional[int],
    batch_size_checked: bool,
    batch_size_value: Optional[int],
    memory_f32_checked: bool,
    numa_checked: bool,
    server_timeout_value: Optional[int],
    host_checked: bool,
    host_value: str,
    port_checked: bool,
    port_value: Optional[int],
    api_key_checked: bool,
    api_key_value: Optional[str],
) -> str:
    """
    Starts the llamafile process based on provided configuration.
    """
    global process

    # Construct command based on checked values
    command = []
    if am_noob:
        # Define what 'am_noob' does, e.g., set default parameters
        command.append('--sane-defaults')  # Replace with actual flag if needed

    if verbose_checked:
        command.append('-v')

    if threads_checked and threads_value is not None:
        command.extend(['-t', str(threads_value)])

    if http_threads_checked and http_threads_value is not None:
        command.extend(['--threads', str(http_threads_value)])

    if threads_batched_checked and threads_batched_value is not None:
        command.extend(['-tb', str(threads_batched_value)])

    if model_alias_checked and model_alias_value:
        command.extend(['-a', model_alias_value])

    # Set model path
    model_path = os.path.abspath(model_value)
    command.extend(['-m', model_path])

    if hf_repo_checked and hf_repo_value:
        command.extend(['-hfr', hf_repo_value])

    if hf_file_checked and hf_file_value:
        command.extend(['-hff', hf_file_value])

    if ctx_size_checked and ctx_size_value is not None:
        command.extend(['-c', str(ctx_size_value)])

    if ngl_checked and ngl_value is not None:
        command.extend(['-ngl', str(ngl_value)])

    if batch_size_checked and batch_size_value is not None:
        command.extend(['-b', str(batch_size_value)])

    if memory_f32_checked:
        command.append('--memory-f32')

    if numa_checked:
        command.append('--numa')

    if host_checked and host_value:
        command.extend(['--host', host_value])

    if port_checked and port_value is not None:
        command.extend(['--port', str(port_value)])

    if api_key_checked and api_key_value:
        command.extend(['--api-key', api_key_value])

    try:
        useros = os.name
        output_filename = "llamafile.exe" if useros == "nt" else "llamafile"

        # Ensure llamafile is downloaded
        llamafile_path = download_latest_llamafile(output_filename)

        # Start llamafile process
        process = launch_in_new_terminal(llamafile_path, command)

        logging.info(f"Llamafile started with command: {' '.join(command)}")
        return f"Command built and ran: {' '.join(command)} \n\nLlamafile started successfully."

    except Exception as e:
        logging.error(f"Failed to start llamafile: {e}")
        return f"Failed to start llamafile: {e}"

#
# End of Local_LLM_Inference_Engine_Lib.py
#######################################################################################################################
