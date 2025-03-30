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
import requests
import shutil
import subprocess
import zipfile

from PoC_Version.App_Function_Libraries.Utils.Utils import logging


# Import Local Libraries
#from App_Function_Libraries import
#
#######################################################################################################################
# Function Definitions
#

def platform_check():
    global userOS
    if platform.system() == "Linux":
        print("Linux OS detected \n Running Linux appropriate commands")
        userOS = "Linux"
    elif platform.system() == "Windows":
        print("Windows OS detected \n Running Windows appropriate commands")
        userOS = "Windows"
    else:
        print("Other OS detected \n Maybe try running things manually?")
        exit()


# Check for NVIDIA GPU and CUDA availability
def cuda_check():
    global processing_choice
    try:
        # Run nvidia-smi to capture its output
        nvidia_smi_output = subprocess.check_output("nvidia-smi", shell=True).decode()

        # Look for CUDA version in the output
        if "CUDA Version" in nvidia_smi_output:
            cuda_version = next(
                (line.split(":")[-1].strip() for line in nvidia_smi_output.splitlines() if "CUDA Version" in line),
                "Not found")
            print(f"NVIDIA GPU with CUDA Version {cuda_version} is available.")
            processing_choice = "cuda"
            return True #fix 'Asserion error: none is not true' in Tests\Summarization\test_summarize.py
        else:
            print("CUDA is not installed or configured correctly.")
            processing_choice = "cpu"
            return False

    except subprocess.CalledProcessError as e:
        print(f"Failed to run 'nvidia-smi': {str(e)}")
        processing_choice = "cpu"
        return False
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        processing_choice = "cpu"
        return False

    # Optionally, check for the CUDA_VISIBLE_DEVICES env variable as an additional check
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print("CUDA_VISIBLE_DEVICES is set:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("CUDA_VISIBLE_DEVICES not set.")


# Ask user if they would like to use either their GPU or their CPU for transcription
def decide_cpugpu():
    global processing_choice
    processing_input = input("Would you like to use your GPU or CPU for transcription? (1/cuda)GPU/(2/cpu)CPU): ")
    if processing_choice == "cuda" and (processing_input.lower() == "cuda" or processing_input == "1"):
        print("You've chosen to use the GPU.")
        logging.debug("GPU is being used for processing")
        processing_choice = "cuda"
    elif processing_input.lower() == "cpu" or processing_input == "2":
        print("You've chosen to use the CPU.")
        logging.debug("CPU is being used for processing")
        processing_choice = "cpu"
    else:
        print("Invalid choice. Please select either GPU or CPU.")


# check for existence of ffmpeg
def check_ffmpeg():
    if shutil.which("ffmpeg"):
        logging.debug("ffmpeg found installed on the local system, in the local PATH, or in the './Bin' folder")
        return True #fix 'Asserion error: none is not true' in Tests\Summarization\test_summarize.py
    elif os.path.exists(os.path.join("", "Bin", "ffmpeg.exe")): # Splitted for clearer loggic
        logging.debug("ffmpeg found in ./Bin directory.")
        return True
    else:
        logging.debug("ffmpeg not installed on the local system/in local PATH")
        print(
            "ffmpeg is not installed.\n\n You can either install it manually, or through your package manager of "
            "choice.\n Windows users, builds are here: https://www.gyan.dev/ffmpeg/builds/")
        if userOS == "Windows":
            if download_ffmpeg(): # call and check the return
                return True
            else:
                return False
            
        elif userOS == "Linux":
            print(
                "You should install ffmpeg using your platform's appropriate package manager, 'apt install ffmpeg',"
                "'dnf install ffmpeg' or 'pacman', etc."
                )
            return False
        else:
            logging.debug("running an unsupported OS")
            print("You're running an unsupported/Un-tested OS")
            exit_script = input("Let's exit the script, unless you're feeling lucky? (y/n)")
            if exit_script.lower() in ["y", "yes", "1"]:  # Handles 'Y' or 'y'
                return False
            return False 


# Download ffmpeg
def download_ffmpeg():
    FFMPEG_DOWNLOAD_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    user_choice = input("Do you want to download ffmpeg? (y/N): ")
    if user_choice.lower() not in ['y', 'yes', '1']:  # Simplified input check
        print("ffmpeg will not be downloaded.")
        return False

    print("Downloading ffmpeg...")
    try:
        response = requests.get(FFMPEG_DOWNLOAD_URL, stream=True)
        # Raise an exception for bad HTTP status codes (4xx or 5xx).
        response.raise_for_status()

        zip_path = "ffmpeg-release-essentials.zip"
        with open(zip_path, 'wb') as file:
            # Write the downloaded content in chunks to avoid memory issues.
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print("Extracting ffmpeg.exe...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            ffmpeg_path = None
            for file_info in zip_ref.infolist():
                if file_info.filename.endswith("ffmpeg.exe"):
                    ffmpeg_path = file_info.filename
                    break
            if ffmpeg_path is None:
                # Raise a FileNotFoundError if ffmpeg.exe is not found in the zip.
                raise FileNotFoundError("ffmpeg.exe not found in the zip file.")
            bin_folder = os.path.join("", "Bin")
            if not os.path.exists(bin_folder):
                os.makedirs(bin_folder)

            zip_ref.extract(ffmpeg_path, path=bin_folder)
            
            src_path = os.path.join(bin_folder, ffmpeg_path)
            dst_path = os.path.join(bin_folder, "ffmpeg.exe")
            shutil.move(src_path, dst_path)  # Move to the correct location (./Bin/ffmpeg.exe).

        os.remove(zip_path)  # Clean up: Delete the downloaded zip file.
        print("ffmpeg.exe has been successfully downloaded and extracted to the './Bin' folder.")
        return True # returns if the process was succesful

    # Handle potential errors during the download and extraction process.
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading ffmpeg: {e}")
        print(f"Error downloading ffmpeg: {e}")
        return False
    except (FileNotFoundError, zipfile.BadZipFile, OSError) as e:
        logging.error(f"Error extracting or moving ffmpeg: {e}")
        print(f"Error extracting or moving ffmpeg: {e}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}")
        return False

#
#
#######################################################################################################################
