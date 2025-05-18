# Local_File_Processing_Lib.py
#########################################
# Local File Processing and File Path Handling Library
# This library is used to handle processing local filepaths and URLs.
# It checks for the OS, the availability of the GPU, and the availability of the ffmpeg executable.
# If the GPU is available, it asks the user if they would like to use it for processing.
# If ffmpeg is not found, it asks the user if they would like to download it.
# The script will exit if the user chooses not to download ffmpeg.
####

####################
# Function List
#
# 1. read_paths_from_file(file_path)
# 2. process_path(path)
# 3. process_local_file(file_path)
# 4. read_paths_from_file(file_path: str) -> List[str]
#
####################

# Import necessary libraries
# Import Local
from App_Function_Libraries.Audio.Audio_Transcription_Lib import convert_to_wav
from App_Function_Libraries.Video_DL_Ingestion_Lib import *
from App_Function_Libraries.Video_DL_Ingestion_Lib import get_youtube
from App_Function_Libraries.Utils.Utils import normalize_title, create_download_directory

#######################################################################################################################
# Function Definitions
#

def read_paths_from_file(file_path):
    """ Reads a file containing URLs or local file paths and returns them as a list. """
    paths = []  # Initialize paths as an empty list
    with open(file_path, 'r') as file:
        paths = file.readlines()
    return [path.strip() for path in paths]


def process_path(path):
    """ Decides whether the path is a URL or a local file and processes accordingly. """
    if path.startswith('http'):
        logging.debug("file is a URL")
        # For YouTube URLs, modify to download and extract info
        return get_youtube(path)
    elif os.path.exists(path):
        logging.debug("File is a path")
        # For local files, define a function to handle them
        return process_local_file(path)
    else:
        logging.error(f"Path does not exist: {path}")
        return None


# FIXME - ingest_text is not used, need to confirm.
def process_local_file(file_path, ingest_text=False):
    logging.info(f"Processing local file: {file_path}")
    file_extension = os.path.splitext(file_path)[1].lower()

    if os.path.isfile(file_path):
        if file_path.lower().endswith('.txt'):
            if ingest_text:
                # Treat as content to be ingested
                return os.path.dirname(file_path), {'title': os.path.basename(file_path)}, file_path
            else:
                # Treat as potential list of URLs
                with open(file_path, 'r') as file:
                    urls = file.read().splitlines()
                return None, None, urls
        elif file_path.lower().endswith(('.mp4', '.avi', '.mov', '.wav', '.mp3', '.m4a')):
            # Handle video and audio files (existing code)
            title = normalize_title(os.path.splitext(os.path.basename(file_path))[0])
            info_dict = {'title': title}
            logging.debug(f"Creating {title} directory...")
            download_path = create_download_directory(title)
            logging.debug(f"Converting '{title}' to an audio file (wav).")
            audio_file = convert_to_wav(file_path)
            logging.debug(f"'{title}' successfully converted to an audio file (wav).")
            return download_path, info_dict, audio_file
    else:
        logging.error(f"File not found: {file_path}")
        return None, None, None





#
#
#######################################################################################################################