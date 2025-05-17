# Utils.py
#########################################
# General Utilities Library
# This library is used to hold random utilities used by various other libraries.
#
####
####################
# Function Categories
#
#     Config loading
#     Misc-Functions
#     File-saving Function Definitions
#     UUID-Functions
#     Sanitization/Verification Functions
#     DB Config Loading
#     File Handling Functions
#
####################
# Function List
#
# 1. extract_text_from_segments(segments: List[Dict]) -> str
# 2. download_file(url, dest_path, expected_checksum=None, max_retries=3, delay=5)
# 3. verify_checksum(file_path, expected_checksum)
# 4. create_download_directory(title)
# 5. sanitize_filename(filename)
# 6. normalize_title(title)
# 7.
#
####################
#
# Import necessary libraries
import mimetypes
import sys
import zipfile
from pathlib import Path

import chardet
import configparser
import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from datetime import timedelta, datetime
from typing import Union, AnyStr, Tuple, List, Optional, Protocol, cast
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
#
# 3rd-Party Imports
import requests
import unicodedata
from tqdm import tqdm
from loguru import logger
#
#######################################################################################################################
#
# Function Definitions

logging = logger

def extract_text_from_segments(segments, include_timestamps=True):
    logger.trace(f"Segments received: {segments}")
    logger.trace(f"Type of segments: {type(segments)}")

    def extract_text_recursive(data, include_timestamps):
        if isinstance(data, dict):
            text = data.get('Text', '')
            if include_timestamps and 'Time_Start' in data and 'Time_End' in data:
                return f"{data['Time_Start']}s - {data['Time_End']}s | {text}"
            for key, value in data.items():
                if key == 'Text':
                    return value
                elif isinstance(value, (dict, list)):
                    result = extract_text_recursive(value, include_timestamps)
                    if result:
                        return result
        elif isinstance(data, list):
            return '\n'.join(filter(None, [extract_text_recursive(item, include_timestamps) for item in data]))
        return None

    text = extract_text_recursive(segments, include_timestamps)

    if text:
        return text.strip()
    else:
        logging.error(f"Unable to extract text from segments: {segments}")
        return "Error: Unable to extract transcription"

#
#
#######################
# Temp file cleanup
#
# Global list to keep track of downloaded files
downloaded_files = []

def cleanup_downloads():
    """Function to clean up downloaded files when the server exits."""
    for file_path in downloaded_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Cleaned up file: {file_path}")
        except Exception as e:
            print(f"Error cleaning up file {file_path}: {e}")

#
#
#######################################################################################################################


#######################################################################################################################
# Config loading
#



def get_project_root():
    """Get the absolute path to the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    logging.trace(f"Project root: {project_root}")
    return project_root


def get_database_dir():
    """Get the absolute path to the database directory."""
    db_dir = os.path.join(get_project_root(), 'Databases')
    os.makedirs(db_dir, exist_ok=True)
    logging.trace(f"Database directory: {db_dir}")
    return db_dir


def get_database_path(db_name: str) -> str:
    """
    Get the full absolute path for a database file.
    Ensures the path is always within the Databases directory.
    """
    # Remove any directory traversal attempts
    safe_db_name = os.path.basename(db_name)
    path = os.path.join(get_database_dir(), safe_db_name)
    logging.trace(f"Database path for {safe_db_name}: {path}")
    return path


def get_project_relative_path(relative_path: Union[str, os.PathLike[AnyStr]]) -> str:
    """Convert a relative path to a path relative to the project root."""
    path = os.path.join(get_project_root(), str(relative_path))
    logging.trace(f"Project relative path for {relative_path}: {path}")
    return path

def get_chromadb_path():
    path = os.path.join(get_project_root(), 'Databases', 'chroma_db')
    logging.trace(f"ChromaDB path: {path}")
    return path

def ensure_directory_exists(path):
    """Ensure that a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)

# FIXME - update to include prompt path in return statement
# FIXME - remove api Key checks from config file and instead check .env file

global_api_endpoints = ["anthropic", "cohere", "google", "groq", "openai", "huggingface", "openrouter", "deepseek", "mistral", "custom_openai_api", "custom_openai_api_2", "llama", "ollama", "ooba", "kobold", "tabby", "vllm", "aphrodite"]

global_search_engines = ["baidu", "bing", "brave", "duckduckgo", "google", "kagi", "searx", "tavily", "yandex"]

openai_tts_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]





def format_api_name(api):
    name_mapping = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "cohere": "Cohere",
        "google": "Google",
        "groq": "Groq",
        "huggingface": "HuggingFace",
        "openrouter": "OpenRouter",
        "deepseek": "DeepSeek",
        "mistral": "Mistral",
        "custom_openai_api": "Custom-OpenAI-API",
        "custom_openai_api_2": "Custom-OpenAI-API-2",
        "llama": "Llama.cpp",
        "ooba": "Ooba",
        "kobold": "Kobold",
        "tabby": "Tabbyapi",
        "vllm": "VLLM",
        "ollama": "Ollama",
        "aphrodite": "Aphrodite"
    }
    return name_mapping.get(api, api.title())

#
# End of Config loading
#######################################################################################################################


#######################################################################################################################
#
# Misc-Functions

# Log file
# logging.basicConfig(filename='debug-runtime.log', encoding='utf-8', level=logging.DEBUG)

def format_metadata_as_text(metadata):
    if not metadata:
        return "No metadata available"

    formatted_text = "Video Metadata:\n"
    for key, value in metadata.items():
        if value is not None:
            if isinstance(value, list):
                # Join list items with commas
                formatted_value = ", ".join(str(item) for item in value)
            elif key == 'upload_date' and len(str(value)) == 8:
                # Format date as YYYY-MM-DD
                formatted_value = f"{value[:4]}-{value[4:6]}-{value[6:]}"
            elif key in ['view_count', 'like_count']:
                # Format large numbers with commas
                formatted_value = f"{value:,}"
            elif key == 'duration':
                # Convert seconds to HH:MM:SS format
                hours, remainder = divmod(value, 3600)
                minutes, seconds = divmod(remainder, 60)
                formatted_value = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                formatted_value = str(value)

            # Replace underscores with spaces in the key name
            formatted_key = key.replace('_', ' ').capitalize()
            formatted_text += f"{formatted_key}: {formatted_value}\n"
    return formatted_text.strip()

# # Example usage:
# example_metadata = {
#     'title': 'Sample Video Title',
#     'uploader': 'Channel Name',
#     'upload_date': '20230615',
#     'view_count': 1000000,
#     'like_count': 50000,
#     'duration': 3725,  # 1 hour, 2 minutes, 5 seconds
#     'tags': ['tag1', 'tag2', 'tag3'],
#     'description': 'This is a sample video description.'
# }
#
# print(format_metadata_as_text(example_metadata))


def convert_to_seconds(time_str):
    if not time_str:
        return 0

    # If it's already a number, assume it's in seconds
    if time_str.isdigit():
        return int(time_str)

    # Parse time string in format HH:MM:SS, MM:SS, or SS
    time_parts = time_str.split(':')
    if len(time_parts) == 3:
        return int(timedelta(hours=int(time_parts[0]),
                             minutes=int(time_parts[1]),
                             seconds=int(time_parts[2])).total_seconds())
    elif len(time_parts) == 2:
        return int(timedelta(minutes=int(time_parts[0]),
                             seconds=int(time_parts[1])).total_seconds())
    elif len(time_parts) == 1:
        return int(time_parts[0])
    else:
        raise ValueError(f"Invalid time format: {time_str}")


def truncate_content(content: Optional[str], max_length: int = 200) -> Optional[str]:
    """Truncate content to the specified maximum length with ellipsis."""
    if not content:
        return content

    if len(content) <= max_length:
        return content

    return content[:max_length - 3] + "..."

#
# End of Misc-Functions
#######################################################################################################################


#######################################################################################################################
#
# File-saving Function Definitions
def save_to_file(video_urls, filename):
    with open(filename, 'w') as file:
        file.write('\n'.join(video_urls))
    print(f"Video URLs saved to {filename}")


def save_segments_to_json(segments, file_name="transcription_segments.json"):
    """
    Save transcription segments to a JSON file.

    Parameters:
    segments (list): List of transcription segments
    file_name (str): Name of the JSON file to save (default: "transcription_segments.json")

    Returns:
    str: Path to the saved JSON file
    """
    # Ensure the Results directory exists
    os.makedirs("Results", exist_ok=True)

    # Full path for the JSON file
    json_file_path = os.path.join("Results", file_name)

    # Save segments to JSON file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(segments, json_file, ensure_ascii=False, indent=4)

    return json_file_path


def safe_download(url: str, tmp_dir: Path, ext: str) -> Path:
    """
    Wrapper around download_file() that:
      1) builds a random filename inside tmp_dir
      2) returns the Path on success
    """
    dst = tmp_dir / (f"{uuid.uuid4().hex}{ext}")
    # checksum=None, max_retries=3, delay=5 keep the defaults
    download_file(url, str(dst))          # raises on failure
    return dst

def smart_download(url: str, tmp_dir: Path) -> Path:
    """
    • Chooses a filename & extension automatically
    • Calls download_file(url, dest_path)
    • Returns Path to downloaded file

    Order of extension preference:
      1. The URL path (e.g. “.md”, “.rst”, “.txt” …)
      2. The HTTP Content‑Type header
      3. Fallback: “.bin”
    """
    # ---------- 1) try URL  -------------------------------------------------
    parsed = urlparse(url)
    guessed_ext = Path(parsed.path).suffix.lower()

    # ---------- 2) if no ext, probe HEAD  -----------------------------------
    if not guessed_ext:
        try:
            head = requests.head(url, allow_redirects=True, timeout=10)
            ctype = head.headers.get("content-type", "")
            guessed_ext = mimetypes.guess_extension(ctype.split(";")[0].strip()) or ""
        except Exception:
            guessed_ext = ""

    # ---------- 3) final fallback  ------------------------------------------
    if not guessed_ext:
        guessed_ext = ".bin"

    # ---------- 4) build dest path  -----------------------------------------
    dest = tmp_dir / f"{uuid.uuid4().hex}{guessed_ext}"

    # ---------- 5) download  ------------------------------------------------
    download_file(url, str(dest))          # inherits retries / resume
    return dest


def download_file(url, dest_path, expected_checksum=None, max_retries=3, delay=5):
    temp_path = dest_path + '.tmp'

    for attempt in range(max_retries):
        try:
            # Check if a partial download exists and get its size
            resume_header = {}
            if os.path.exists(temp_path):
                resume_header = {'Range': f'bytes={os.path.getsize(temp_path)}-'}

            response = requests.get(url, stream=True, headers=resume_header)
            response.raise_for_status()

            # Get the total file size from headers
            total_size = int(response.headers.get('content-length', 0))
            initial_pos = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0

            mode = 'ab' if 'Range' in response.headers else 'wb'
            with open(temp_path, mode) as temp_file, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=dest_path, initial=initial_pos, ascii=True
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive new chunks
                        temp_file.write(chunk)
                        pbar.update(len(chunk))

            # Verify the checksum if provided
            if expected_checksum:
                if not verify_checksum(temp_path, expected_checksum):
                    os.remove(temp_path)
                    raise ValueError("Downloaded file's checksum does not match the expected checksum")

            # Move the file to the final destination
            os.rename(temp_path, dest_path)
            print("Download complete and verified!")
            return dest_path

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Download failed.")
                raise

def download_file_if_missing(url: str, local_path: str) -> None:
    """
    Download a file from a URL if it does not exist locally.
    """
    if os.path.exists(local_path):
        logging.debug(f"File already exists locally: {local_path}")
        return
    logging.info(f"Downloading from {url} to {local_path}")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

def create_download_directory(title):
    base_dir = "Results"
    # Remove characters that are illegal in Windows filenames and normalize
    safe_title = normalize_title(title, preserve_spaces=False)
    logging.debug(f"{title} successfully normalized")
    session_path = os.path.join(base_dir, safe_title)
    if not os.path.exists(session_path):
        os.makedirs(session_path, exist_ok=True)
        logging.debug(f"Created directory for downloaded video: {session_path}")
    else:
        logging.debug(f"Directory already exists for downloaded video: {session_path}")
    return session_path


def safe_read_file(file_path):
    encodings = ['utf-8', 'utf-16', 'ascii', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-8-sig']

    logging.info(f"Attempting to read file: {file_path}")

    try:
        with open(file_path, 'rb') as file:
            logging.debug(f"Reading file in binary mode: {file_path}")
            raw_data = file.read()
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return f"File not found: {file_path}"
    except Exception as e:
        logging.error(f"An error occurred while reading the file: {e}")
        return f"An error occurred while reading the file: {e}"

    if not raw_data:
        logging.warning(f"File is empty: {file_path}")
        return ""

    # Use chardet to detect the encoding
    detected = chardet.detect(raw_data)
    if detected['encoding'] is not None:
        encodings.insert(0, detected['encoding'])
        logging.info(f"Detected encoding: {detected['encoding']}")

    for encoding in encodings:
        logging.info(f"Trying encoding: {encoding}")
        try:
            decoded_content = raw_data.decode(encoding)
            # Check if the content is mostly printable
            if sum(c.isprintable() for c in decoded_content) / len(decoded_content) > 0.90:
                logging.info(f"Successfully decoded file with encoding: {encoding}")
                return decoded_content
        except UnicodeDecodeError:
            logging.debug(f"Failed to decode with {encoding}")
            continue

    # If all decoding attempts fail, return the error message
    logging.error(f"Unable to decode the file {file_path}")
    return f"Unable to decode the file {file_path}"


#
# End of Files-saving Function Definitions
#######################################################################################################################


#######################################################################################################################
#
# UUID-Functions

def generate_unique_filename(base_path, base_filename):
    """Generate a unique filename by appending a counter if necessary."""
    filename = base_filename
    counter = 1
    while os.path.exists(os.path.join(base_path, filename)):
        name, ext = os.path.splitext(base_filename)
        filename = f"{name}_{counter}{ext}"
        counter += 1
    return filename


def generate_unique_identifier(file_path):
    filename = os.path.basename(file_path)
    timestamp = int(time.time())

    # Generate a hash of the file content
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    content_hash = hasher.hexdigest()[:8]  # Use first 8 characters of the hash

    return f"local:{timestamp}:{content_hash}:{filename}"

#
# End of UUID-Functions
#######################################################################################################################


#######################################################################################################################
#
# Sanitization/Verification Functions

# Helper function to validate URL format
def is_valid_url(url: str) -> bool:
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None


def verify_checksum(file_path, expected_checksum):
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b''):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == expected_checksum


def normalize_title(title, preserve_spaces=False):
    # Normalize the string to 'NFKD' form and encode to 'ascii' ignoring non-ascii characters
    title = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('ascii')

    if preserve_spaces:
        # Replace special characters with underscores, but keep spaces
        title = re.sub(r'[^\w\s\-.]', '_', title)
    else:
        # Replace special characters and spaces with underscores
        title = re.sub(r'[^\w\-.]', '_', title)

    # Replace multiple consecutive underscores with a single underscore
    title = re.sub(r'_+', '_', title)

    # Replace specific characters with underscores
    title = title.replace('/', '_').replace('\\', '_').replace(':', '_').replace('"', '_').replace('*', '_').replace(
        '?', '_').replace(
        '<', '_').replace('>', '_').replace('|', '_')

    return title.strip('_')


def clean_youtube_url(url):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    if 'list' in query_params:
        query_params.pop('list')
    cleaned_query = urlencode(query_params, doseq=True)
    cleaned_url = urlunparse(parsed_url._replace(query=cleaned_query))
    return cleaned_url

def sanitize_filename(filename):
    """
    Sanitizes the filename by:
      1) Removing forbidden characters entirely (rather than replacing them with '-')
      2) Collapsing consecutive whitespace into a single space
      3) Collapsing consecutive dashes into a single dash
    """
    # 1) Remove forbidden characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
    # 2) Replace runs of whitespace with a single space
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    # 3) Replace consecutive dashes with a single dash
    sanitized = re.sub(r'-{2,}', '-', sanitized)
    return sanitized


def format_transcription(content):
    # Replace '\n' with actual line breaks
    content = content.replace('\\n', '\n')
    # Split the content by newlines first
    lines = content.split('\n')
    formatted_lines = []
    for line in lines:
        # Add extra space after periods for better readability
        line = line.replace('.', '. ').replace('.  ', '. ')

        # Split into sentences using a more comprehensive regex
        sentences = re.split('(?<=[.!?]) +', line)

        # Trim whitespace from each sentence and add a line break
        formatted_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        # Join the formatted sentences
        formatted_lines.append(' '.join(formatted_sentences))

    # Join the lines with HTML line breaks
    formatted_content = '<br>'.join(formatted_lines)

    return formatted_content

def sanitize_user_input(message):
    """
    Removes or escapes '{{' and '}}' to prevent placeholder injection.

    Args:
        message (str): The user's message.

    Returns:
        str: Sanitized message.
    """
    # Replace '{{' and '}}' with their escaped versions
    message = re.sub(r'\{\{', '{ {', message)
    message = re.sub(r'\}\}', '} }', message)
    return message

def format_file_path(file_path, fallback_path=None):
    if file_path and os.path.exists(file_path):
        logging.debug(f"File exists: {file_path}")
        return file_path
    elif fallback_path and os.path.exists(fallback_path):
        logging.debug(f"File does not exist: {file_path}. Returning fallback path: {fallback_path}")
        return fallback_path
    else:
        logging.debug(f"File does not exist: {file_path}. No fallback path available.")
        return None

#
# End of Sanitization/Verification Functions
#######################################################################################################################


#######################################################################################################################
#
# DB Config Loading


def get_db_config():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to the project root directory (tldw)
    project_root = os.path.dirname(os.path.dirname(current_dir))
    # Construct the path to the config file
    config_path = os.path.join(project_root, 'Config_Files', 'config.txt')
    # Read the config file
    config = configparser.ConfigParser()
    config.read(config_path)
    # Return the database configuration
    return {
        'type': config['Database']['type'],
        'sqlite_path': config.get('Database', 'sqlite_path', fallback='./Databases/server_media_summary.db'),
        'elasticsearch_host': config.get('Database', 'elasticsearch_host', fallback='localhost'),
        'elasticsearch_port': config.getint('Database', 'elasticsearch_port', fallback=9200)
    }

#
# End of DB Config Loading
#######################################################################################################################


#######################################################################################################################
#
# File Handling Functions

# Track temp files for cleanup
temp_files = []

temp_file_paths = []

def save_temp_file(file):
    global temp_files
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, file.name)
    with open(temp_path, 'wb') as f:
        f.write(file.read())
    temp_files.append(temp_path)
    return temp_path

def cleanup_temp_files():
    global temp_files
    for file_path in temp_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"Removed temporary file: {file_path}")
            except Exception as e:
                logging.error(f"Failed to remove temporary file {file_path}: {e}")
    temp_files.clear()

def generate_unique_id():
    return f"uploaded_file_{uuid.uuid4()}"

class FileProcessor:
    """Handles file reading and name processing"""

    VALID_EXTENSIONS = {'.md', '.txt', '.zip'}
    ENCODINGS_TO_TRY = [
        'utf-8',
        'utf-16',
        'windows-1252',
        'iso-8859-1',
        'ascii'
    ]

    @staticmethod
    def detect_encoding(file_path: str) -> str:
        """Detect the file encoding using chardet"""
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8'

    @staticmethod
    def read_file_content(file_path: str) -> str:
        """Read file content with automatic encoding detection"""
        detected_encoding = FileProcessor.detect_encoding(file_path)

        # Try detected encoding first
        try:
            with open(file_path, 'r', encoding=detected_encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # If detected encoding fails, try others
            for encoding in FileProcessor.ENCODINGS_TO_TRY:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue

            # If all encodings fail, use utf-8 with error handling
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()

    @staticmethod
    def process_filename_to_title(filename: str) -> str:
        """Convert filename to a readable title"""
        # Remove extension
        name = os.path.splitext(filename)[0]

        # Look for date patterns
        date_pattern = r'(\d{4}[-_]?\d{2}[-_]?\d{2})'
        date_match = re.search(date_pattern, name)
        date_str = ""
        if date_match:
            try:
                date = datetime.strptime(date_match.group(1).replace('_', '-'), '%Y-%m-%d')
                date_str = date.strftime("%b %d, %Y")
                name = name.replace(date_match.group(1), '').strip('-_')
            except ValueError:
                pass

        # Replace separators with spaces
        name = re.sub(r'[-_]+', ' ', name)

        # Remove redundant spaces
        name = re.sub(r'\s+', ' ', name).strip()

        # Capitalize words, excluding certain words
        exclude_words = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        words = name.split()
        capitalized = []
        for i, word in enumerate(words):
            if i == 0 or word not in exclude_words:
                capitalized.append(word.capitalize())
            else:
                capitalized.append(word.lower())
        name = ' '.join(capitalized)

        # Add date if found
        if date_str:
            name = f"{name} - {date_str}"

        return name


class ZipValidator:
    """Validates zip file contents and structure"""

    MAX_ZIP_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_FILES = 100
    VALID_EXTENSIONS = {'.md', '.txt'}

    @staticmethod
    def validate_zip_file(zip_path: str) -> Tuple[bool, str, List[str]]:
        """
        Validate zip file and its contents
        Returns: (is_valid, error_message, valid_files)
        """
        try:
            # Check zip file size
            if os.path.getsize(zip_path) > ZipValidator.MAX_ZIP_SIZE:
                return False, "Zip file too large (max 100MB)", []

            valid_files = []
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Check number of files
                if len(zip_ref.filelist) > ZipValidator.MAX_FILES:
                    return False, f"Too many files in zip (max {ZipValidator.MAX_FILES})", []

                # Check for directory traversal attempts
                for file_info in zip_ref.filelist:
                    if '..' in file_info.filename or file_info.filename.startswith('/'):
                        return False, "Invalid file paths detected", []

                # Validate each file
                total_size = 0
                for file_info in zip_ref.filelist:
                    # Skip directories
                    if file_info.filename.endswith('/'):
                        continue

                    # Check file size
                    if file_info.file_size > ZipValidator.MAX_ZIP_SIZE:
                        return False, f"File {file_info.filename} too large", []

                    total_size += file_info.file_size
                    if total_size > ZipValidator.MAX_ZIP_SIZE:
                        return False, "Total uncompressed size too large", []

                    # Check file extension
                    ext = os.path.splitext(file_info.filename)[1].lower()
                    if ext in ZipValidator.VALID_EXTENSIONS:
                        valid_files.append(file_info.filename)

            if not valid_files:
                return False, "No valid markdown or text files found in zip", []

            return True, "", valid_files

        except zipfile.BadZipFile:
            return False, "Invalid or corrupted zip file", []
        except Exception as e:
            return False, f"Error processing zip file: {str(e)}", []

def format_text_with_line_breaks(text):
    # Split the text into sentences and add line breaks
    sentences = text.replace('. ', '.<br>').replace('? ', '?<br>').replace('! ', '!<br>')
    return sentences


def format_transcript(raw_text: str) -> str:
    """Convert timestamped transcript to readable format"""
    lines = []
    for line in raw_text.split('\n'):
        if '|' in line:
            timestamp, text = line.split('|', 1)
            lines.append(f"{text.strip()}")
        else:
            lines.append(line.strip())
    return '\n'.join(lines)

#
# End of File Handling Functions
#######################################################################################################################

def extract_media_id_from_result_string(result_msg: Optional[str]) -> Optional[str]:
    """
    Extracts the Media ID from a string expected to contain 'Media ID: <id>'.

    This function searches for the pattern "Media ID:" followed by optional
    whitespace and captures the subsequent sequence of non-whitespace characters
    as the ID.

    Args:
        result_msg: The input string potentially containing the Media ID message,
                    typically returned by processing functions like import_epub.

    Returns:
        The extracted Media ID as a string if the pattern is found.
        Returns None if the input string is None, empty, or the pattern
        "Media ID: <id>" is not found.

    Examples:
        >>> extract_media_id_from_result_string("Ebook imported successfully. Media ID: ebook_789")
        'ebook_789'
        >>> extract_media_id_from_result_string("Success. Media ID: db_mock_id")
        'db_mock_id'
        >>> extract_media_id_from_result_string("Error during processing.")
        None
        >>> extract_media_id_from_result_string(None)
        None
        >>> extract_media_id_from_result_string("Media ID: id-with-hyphens123") # Test hyphens/numbers
        'id-with-hyphens123'
        >>> extract_media_id_from_result_string("Media ID:id_no_space") # Test no space
        'id_no_space'
    """
    # Handle None or empty input string gracefully
    if not result_msg:
        return None

    # Regular expression pattern:
    # - Looks for the literal string "Media ID:" (case-sensitive).
    # - Allows for zero or more whitespace characters (\s*) after the colon.
    # - Captures (\(...\)) one or more non-whitespace characters (\S+).
    #   Using \S+ is generally safer than \w+ as IDs might contain hyphens or other symbols.
    #   If IDs are strictly alphanumeric + underscore, you could use (\w+) instead.
    # - We use re.search to find the pattern anywhere in the string.
    pattern = r"Media ID:\s*(\S+)"

    match = re.search(pattern, result_msg)

    # If a match is found, match.group(1) will contain the captured ID part
    if match:
        return match.group(1)
    else:
        # The pattern "Media ID: <id>" was not found in the string
        return None

def is_valid_date(date_string): # Placeholder
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def get_user_database_path():
    return None