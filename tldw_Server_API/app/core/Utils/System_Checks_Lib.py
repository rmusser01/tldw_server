# System_Checks_Lib.py
#########################################
# System Checks Library
# This library is used to check the system for the necessary dependencies to run the script.
# It checks for the OS, the availability of the GPU, and the availability of the ffmpeg executable.
# If the GPU is available, it asks the user if they would like to use it for processing.
# If ffmpeg is not found, it asks the user if they would like to download it.
# The script will exit if the user chooses not to download ffmpeg.
####

####################
# Function List
#
# 1. platform_check()
# 2. cuda_check()
# 3. decide_cpugpu()
# 4. check_ffmpeg()
# 5. download_ffmpeg()
#
####################
# Import necessary libraries
import os
import platform
import shutil
# TASK-9941 Bandit B404 rationale: subprocess is limited to the local
# nvidia-smi availability probe; callers cannot supply commands and shell=True
# is not used.
import subprocess  # nosec B404

from tldw_Server_API.app.core.Utils.Utils import logging

userOS: str = "Unknown"
processing_choice: str = "cpu"
# Import Local Libraries
#from App_Function_Libraries import
#
#######################################################################################################################
# Function Definitions
#

def platform_check():
    """Detect the host operating system and record whether supported checks can run."""
    global userOS
    if platform.system() == "Linux":
        logging.info("Linux OS detected; running Linux-appropriate checks")
        userOS = "Linux"
    elif platform.system() == "Windows":
        logging.info("Windows OS detected; running Windows-appropriate checks")
        userOS = "Windows"
    else:
        logging.warning("Other/unknown OS detected; you may need to run steps manually")
        userOS = platform.system() or "Unknown"
        return False
    return True


# Check for NVIDIA GPU and CUDA availability
def cuda_check():
    """Probe for NVIDIA CUDA support using a resolved local nvidia-smi executable."""
    global processing_choice
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        logging.info(f"CUDA_VISIBLE_DEVICES is set: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        logging.info("CUDA_VISIBLE_DEVICES not set.")

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        logging.warning("CUDA is not installed or configured correctly.")
        processing_choice = "cpu"
        return False

    try:
        # Run nvidia-smi to capture its output
        # TASK-9941 Bandit B603 rationale: nvidia_smi comes from shutil.which("nvidia-smi"),
        # is executed as a fixed argv list, and shell=True is not used.
        nvidia_smi_output = subprocess.check_output([nvidia_smi], text=True)  # nosec B603

        # Look for CUDA version in the output
        if "CUDA Version" in nvidia_smi_output:
            cuda_version = next(
                (line.split(":")[-1].strip() for line in nvidia_smi_output.splitlines() if "CUDA Version" in line),
                "Not found")
            logging.info(f"NVIDIA GPU with CUDA Version {cuda_version} is available.")
            processing_choice = "cuda"
            return True #fix 'Asserion error: none is not true' in Tests\Summarization\test_summarize.py
        else:
            logging.warning("CUDA is not installed or configured correctly.")
            processing_choice = "cpu"
            return False

    except (subprocess.CalledProcessError, OSError) as e:
        logging.error(f"Failed to run 'nvidia-smi': {str(e)}")
        processing_choice = "cpu"
        return False
    except Exception as e:
        logging.error(f"An error occurred during CUDA detection: {str(e)}")
        processing_choice = "cpu"
        return False


# Ask user if they would like to use either their GPU or their CPU for transcription
def decide_cpugpu():
    """Prompt for CPU/GPU processing preference, defaulting to the current choice when non-interactive."""
    global processing_choice
    try:
        processing_input = input("Would you like to use your GPU or CPU for transcription? (1/cuda)GPU/(2/cpu)CPU): ")
    except EOFError:
        logging.debug("No interactive input available; defaulting to %s", processing_choice)
        processing_input = processing_choice

    if processing_input is None:
        return processing_choice

    if processing_choice == "cuda" and (processing_input.lower() == "cuda" or processing_input == "1"):
        logging.info("User selected GPU for processing.")
        logging.debug("GPU is being used for processing")
        processing_choice = "cuda"
    elif processing_input.lower() == "cpu" or processing_input == "2":
        logging.info("User selected CPU for processing.")
        logging.debug("CPU is being used for processing")
        processing_choice = "cpu"
    else:
        logging.warning("Invalid processing choice; please select GPU or CPU.")
    return processing_choice


# check for existence of ffmpeg
def check_ffmpeg():
    """Return whether ffmpeg is available from PATH or the local Bin fallback."""
    if shutil.which("ffmpeg"):
        logging.debug("ffmpeg found installed on the local system, in the local PATH, or in the './Bin' folder")
        return True #fix 'Asserion error: none is not true' in Tests\Summarization\test_summarize.py
    elif os.path.exists(os.path.join(".", "Bin", "ffmpeg.exe")): # Splitted for clearer loggic
        logging.debug("ffmpeg found in ./Bin directory.")
        return True
    else:
        logging.debug("ffmpeg not installed on the local system/in local PATH")
        logging.warning(
            "ffmpeg is not installed. You can install it manually or via your package manager. "
            "Windows builds: https://www.gyan.dev/ffmpeg/builds/"
        )
        userOS_guess = platform.system() if userOS == "Unknown" else userOS

        if userOS_guess == "Windows":
            return download_ffmpeg()

        elif userOS_guess == "Linux":
            logging.info(
                "Install ffmpeg using your platform's package manager (apt/dnf/pacman/etc.)."
            )
            return False
        else:
            logging.debug("running an unsupported OS")
            logging.warning("You're running an unsupported/un-tested OS")
            try:
                exit_script = input("Let's exit the script, unless you're feeling lucky? (y/n)")
            except EOFError:
                logging.debug("No interactive input available; defaulting to continue without ffmpeg auto-installation")
                return False
            if exit_script.lower() in ["y", "yes", "1"]:  # Handles 'Y' or 'y'
                return False
            return False


# Download ffmpeg
def download_ffmpeg():
    """Refuse automatic executable download and direct users to trusted installation channels."""
    logging.warning(
        "Automatic ffmpeg executable download is disabled. Install ffmpeg manually "
        "or through a trusted package manager."
    )
    return False
