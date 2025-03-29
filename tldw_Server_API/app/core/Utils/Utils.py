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
import sys
import zipfile

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
from typing import Union, AnyStr, Tuple, List, Protocol, cast
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
def load_comprehensive_config():
    # Get the directory of the current script (Utils.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logging.trace(f"Current directory: {current_dir}")

    # Go up two levels to the project root directory (tldw)
    project_root = os.path.dirname(os.path.dirname(current_dir))
    logging.trace(f"Project root directory: {project_root}")

    # Construct the path to the config file
    config_path = os.path.join(project_root, 'Config_Files', 'config.txt')
    logging.trace(f"Config file path: {config_path}")

    # Check if the config file exists
    if not os.path.exists(config_path):
        logging.error(f"Config file not found at {config_path}")
        raise FileNotFoundError(f"Config file not found at {config_path}")

    # Read the config file
    config = configparser.ConfigParser()
    config.read(config_path)

    # Log the sections found in the config file
    logging.trace(f"load_comprehensive_config(): Sections found in config: {config.sections()}")

    return config


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
def load_and_log_configs():
    logging.debug("load_and_log_configs(): Loading and logging configurations...")
    try:
        config = load_comprehensive_config()
        if config is None:
            logging.error("Config is None, cannot proceed")
            return None
        # API Keys
        anthropic_api_key = config.get('API', 'anthropic_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded Anthropic API Key: {anthropic_api_key[:5]}...{anthropic_api_key[-5:] if anthropic_api_key else None}")

        cohere_api_key = config.get('API', 'cohere_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded Cohere API Key: {cohere_api_key[:5]}...{cohere_api_key[-5:] if cohere_api_key else None}")

        groq_api_key = config.get('API', 'groq_api_key', fallback=None)
        # logging.debug(f"Loaded Groq API Key: {groq_api_key[:5]}...{groq_api_key[-5:] if groq_api_key else None}")

        openai_api_key = config.get('API', 'openai_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded OpenAI API Key: {openai_api_key[:5]}...{openai_api_key[-5:] if openai_api_key else None}")

        huggingface_api_key = config.get('API', 'huggingface_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded HuggingFace API Key: {huggingface_api_key[:5]}...{huggingface_api_key[-5:] if huggingface_api_key else None}")

        openrouter_api_key = config.get('API', 'openrouter_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded OpenRouter API Key: {openrouter_api_key[:5]}...{openrouter_api_key[-5:] if openrouter_api_key else None}")

        deepseek_api_key = config.get('API', 'deepseek_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded DeepSeek API Key: {deepseek_api_key[:5]}...{deepseek_api_key[-5:] if deepseek_api_key else None}")

        mistral_api_key = config.get('API', 'mistral_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded Mistral API Key: {mistral_api_key[:5]}...{mistral_api_key[-5:] if mistral_api_key else None}")

        google_api_key = config.get('API', 'google_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded Google API Key: {google_api_key[:5]}...{google_api_key[-5:] if google_api_key else None}")

        elevenlabs_api_key = config.get('API', 'elevenlabs_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded elevenlabs API Key: {elevenlabs_api_key[:5]}...{elevenlabs_api_key[-5:] if elevenlabs_api_key else None}")

        # LLM API Settings - streaming / temperature / top_p / min_p
        # Anthropic
        anthropic_api_key = config.get('API', 'anthropic_api_key', fallback=None)
        anthropic_model = config.get('API', 'anthropic_model', fallback='claude-3-5-sonnet-20240620')
        anthropic_streaming = config.get('API', 'anthropic_streaming', fallback='False')
        anthropic_temperature = config.get('API', 'anthropic_temperature', fallback='0.7')
        anthropic_top_p = config.get('API', 'anthropic_top_p', fallback='0.95')
        anthropic_top_k = config.get('API', 'anthropic_top_k', fallback='100')
        anthropic_max_tokens = config.get('API', 'anthropic_max_tokens', fallback='4096')
        anthropic_api_timeout = config.get('API', 'anthropic_api_timeout', fallback='90')
        anthropic_api_retries = config.get('API', 'anthropic_api_retry', fallback='3')
        anthropic_api_retry_delay = config.get('API', 'anthropic_api_retry_delay', fallback='5')

        # Cohere
        cohere_streaming = config.get('API', 'cohere_streaming', fallback='False')
        cohere_temperature = config.get('API', 'cohere_temperature', fallback='0.7')
        cohere_max_p = config.get('API', 'cohere_max_p', fallback='0.95')
        cohere_top_k = config.get('API', 'cohere_top_k', fallback='100')
        cohere_model = config.get('API', 'cohere_model', fallback='command-r-plus')
        cohere_max_tokens = config.get('API', 'cohere_max_tokens', fallback='4096')
        cohere_api_timeout = config.get('API', 'cohere_api_timeout', fallback='90')
        cohere_api_retries = config.get('API', 'cohere_api_retry', fallback='3')
        cohere_api_retry_delay = config.get('API', 'cohere_api_retry_delay', fallback='5')

        # Deepseek
        deepseek_streaming = config.get('API', 'deepseek_streaming', fallback='False')
        deepseek_temperature = config.get('API', 'deepseek_temperature', fallback='0.7')
        deepseek_top_p = config.get('API', 'deepseek_top_p', fallback='0.95')
        deepseek_min_p = config.get('API', 'deepseek_min_p', fallback='0.05')
        deepseek_model = config.get('API', 'deepseek_model', fallback='deepseek-chat')
        deepseek_max_tokens = config.get('API', 'deepseek_max_tokens', fallback='4096')
        deepseek_api_timeout = config.get('API', 'deepseek_api_timeout', fallback='90')
        deepseek_api_retries = config.get('API', 'deepseek_api_retry', fallback='3')
        deepseek_api_retry_delay = config.get('API', 'deepseek_api_retry_delay', fallback='5')

        # Groq
        groq_model = config.get('API', 'groq_model', fallback='llama3-70b-8192')
        groq_streaming = config.get('API', 'groq_streaming', fallback='False')
        groq_temperature = config.get('API', 'groq_temperature', fallback='0.7')
        groq_top_p = config.get('API', 'groq_top_p', fallback='0.95')
        groq_max_tokens = config.get('API', 'groq_max_tokens', fallback='4096')
        groq_api_timeout = config.get('API', 'groq_api_timeout', fallback='90')
        groq_api_retries = config.get('API', 'groq_api_retry', fallback='3')
        groq_api_retry_delay = config.get('API', 'groq_api_retry_delay', fallback='5')

        # Google
        google_model = config.get('API', 'google_model', fallback='gemini-1.5-pro')
        google_streaming = config.get('API', 'google_streaming', fallback='False')
        google_temperature = config.get('API', 'google_temperature', fallback='0.7')
        google_top_p = config.get('API', 'google_top_p', fallback='0.95')
        google_min_p = config.get('API', 'google_min_p', fallback='0.05')
        google_max_tokens = config.get('API', 'google_max_tokens', fallback='4096')
        google_api_timeout = config.get('API', 'google_api_timeout', fallback='90')
        google_api_retries = config.get('API', 'google_api_retry', fallback='3')
        google_api_retry_delay = config.get('API', 'google_api_retry_delay', fallback='5')

        # HuggingFace
        huggingface_model = config.get('API', 'huggingface_model', fallback='CohereForAI/c4ai-command-r-plus')
        huggingface_streaming = config.get('API', 'huggingface_streaming', fallback='False')
        huggingface_temperature = config.get('API', 'huggingface_temperature', fallback='0.7')
        huggingface_top_p = config.get('API', 'huggingface_top_p', fallback='0.95')
        huggingface_min_p = config.get('API', 'huggingface_min_p', fallback='0.05')
        huggingface_max_tokens = config.get('API', 'huggingface_max_tokens', fallback='4096')
        huggingface_api_timeout = config.get('API', 'huggingface_api_timeout', fallback='90')
        huggingface_api_retries = config.get('API', 'huggingface_api_retry', fallback='3')
        huggingface_api_retry_delay = config.get('API', 'huggingface_api_retry_delay', fallback='5')

        # Mistral
        mistral_model = config.get('API', 'mistral_model', fallback='mistral-large-latest')
        mistral_streaming = config.get('API', 'mistral_streaming', fallback='False')
        mistral_temperature = config.get('API', 'mistral_temperature', fallback='0.7')
        mistral_top_p = config.get('API', 'mistral_top_p', fallback='0.95')
        mistral_max_tokens = config.get('API', 'mistral_max_tokens', fallback='4096')
        mistral_api_timeout = config.get('API', 'mistral_api_timeout', fallback='90')
        mistral_api_retries = config.get('API', 'mistral_api_retry', fallback='3')
        mistral_api_retry_delay = config.get('API', 'mistral_api_retry_delay', fallback='5')

        # OpenAI
        openai_model = config.get('API', 'openai_model', fallback='gpt-4o')
        openai_streaming = config.get('API', 'openai_streaming', fallback='False')
        openai_temperature = config.get('API', 'openai_temperature', fallback='0.7')
        openai_top_p = config.get('API', 'openai_top_p', fallback='0.95')
        openai_max_tokens = config.get('API', 'openai_max_tokens', fallback='4096')
        openai_api_timeout = config.get('API', 'openai_api_timeout', fallback='90')
        openai_api_retries = config.get('API', 'openai_api_retry', fallback='3')
        openai_api_retry_delay = config.get('API', 'openai_api_retry_delay', fallback='5')

        # OpenRouter
        openrouter_model = config.get('API', 'openrouter_model', fallback='microsoft/wizardlm-2-8x22b')
        openrouter_streaming = config.get('API', 'openrouter_streaming', fallback='False')
        openrouter_temperature = config.get('API', 'openrouter_temperature', fallback='0.7')
        openrouter_top_p = config.get('API', 'openrouter_top_p', fallback='0.95')
        openrouter_min_p = config.get('API', 'openrouter_min_p', fallback='0.05')
        openrouter_top_k = config.get('API', 'openrouter_top_k', fallback='100')
        openrouter_max_tokens = config.get('API', 'openrouter_max_tokens', fallback='4096')
        openrouter_api_timeout = config.get('API', 'openrouter_api_timeout', fallback='90')
        openrouter_api_retries = config.get('API', 'openrouter_api_retry', fallback='3')
        openrouter_api_retry_delay = config.get('API', 'openrouter_api_retry_delay', fallback='5')

        # Logging Checks for model loads
        # logging.debug(f"Loaded Anthropic Model: {anthropic_model}")
        # logging.debug(f"Loaded Cohere Model: {cohere_model}")
        # logging.debug(f"Loaded Groq Model: {groq_model}")
        # logging.debug(f"Loaded OpenAI Model: {openai_model}")
        # logging.debug(f"Loaded HuggingFace Model: {huggingface_model}")
        # logging.debug(f"Loaded OpenRouter Model: {openrouter_model}")
        # logging.debug(f"Loaded Deepseek Model: {deepseek_model}")
        # logging.debug(f"Loaded Mistral Model: {mistral_model}")

        # Local-Models
        kobold_api_ip = config.get('Local-API', 'kobold_api_IP', fallback='http://127.0.0.1:5000/api/v1/generate')
        kobold_openai_api_IP = config.get('Local-API', 'kobold_openai_api_IP', fallback='http://127.0.0.1:5001/v1/chat/completions')
        kobold_api_key = config.get('Local-API', 'kobold_api_key', fallback='')
        kobold_streaming = config.get('Local-API', 'kobold_streaming', fallback='False')
        kobold_temperature = config.get('Local-API', 'kobold_temperature', fallback='0.7')
        kobold_top_p = config.get('Local-API', 'kobold_top_p', fallback='0.95')
        kobold_top_k = config.get('Local-API', 'kobold_top_k', fallback='100')
        kobold_max_tokens = config.get('Local-API', 'kobold_max_tokens', fallback='4096')
        kobold_api_timeout = config.get('Local-API', 'kobold_api_timeout', fallback='90')
        kobold_api_retries = config.get('Local-API', 'kobold_api_retry', fallback='3')
        kobold_api_retry_delay = config.get('Local-API', 'kobold_api_retry_delay', fallback='5')

        llama_api_IP = config.get('Local-API', 'llama_api_IP', fallback='http://127.0.0.1:8080/v1/chat/completions')
        llama_api_key = config.get('Local-API', 'llama_api_key', fallback='')
        llama_streaming = config.get('Local-API', 'llama_streaming', fallback='False')
        llama_temperature = config.get('Local-API', 'llama_temperature', fallback='0.7')
        llama_top_p = config.get('Local-API', 'llama_top_p', fallback='0.95')
        llama_min_p = config.get('Local-API', 'llama_min_p', fallback='0.05')
        llama_top_k = config.get('Local-API', 'llama_top_k', fallback='100')
        llama_max_tokens = config.get('Local-API', 'llama_max_tokens', fallback='4096')
        llama_api_timeout = config.get('Local-API', 'llama_api_timeout', fallback='90')
        llama_api_retries = config.get('Local-API', 'llama_api_retry', fallback='3')
        llama_api_retry_delay = config.get('Local-API', 'llama_api_retry_delay', fallback='5')

        ooba_api_IP = config.get('Local-API', 'ooba_api_IP', fallback='http://127.0.0.1:5000/v1/chat/completions')
        ooba_api_key = config.get('Local-API', 'ooba_api_key', fallback='')
        ooba_streaming = config.get('Local-API', 'ooba_streaming', fallback='False')
        ooba_temperature = config.get('Local-API', 'ooba_temperature', fallback='0.7')
        ooba_top_p = config.get('Local-API', 'ooba_top_p', fallback='0.95')
        ooba_min_p = config.get('Local-API', 'ooba_min_p', fallback='0.05')
        ooba_top_k = config.get('Local-API', 'ooba_top_k', fallback='100')
        ooba_max_tokens = config.get('Local-API', 'ooba_max_tokens', fallback='4096')
        ooba_api_timeout = config.get('Local-API', 'ooba_api_timeout', fallback='90')
        ooba_api_retries = config.get('Local-API', 'ooba_api_retry', fallback='3')
        ooba_api_retry_delay = config.get('Local-API', 'ooba_api_retry_delay', fallback='5')

        tabby_api_IP = config.get('Local-API', 'tabby_api_IP', fallback='http://127.0.0.1:5000/api/v1/generate')
        tabby_api_key = config.get('Local-API', 'tabby_api_key', fallback=None)
        tabby_model = config.get('models', 'tabby_model', fallback=None)
        tabby_streaming = config.get('Local-API', 'tabby_streaming', fallback='False')
        tabby_temperature = config.get('Local-API', 'tabby_temperature', fallback='0.7')
        tabby_top_p = config.get('Local-API', 'tabby_top_p', fallback='0.95')
        tabby_top_k = config.get('Local-API', 'tabby_top_k', fallback='100')
        tabby_min_p = config.get('Local-API', 'tabby_min_p', fallback='0.05')
        tabby_max_tokens = config.get('Local-API', 'tabby_max_tokens', fallback='4096')
        tabby_api_timeout = config.get('Local-API', 'tabby_api_timeout', fallback='90')
        tabby_api_retries = config.get('Local-API', 'tabby_api_retry', fallback='3')
        tabby_api_retry_delay = config.get('Local-API', 'tabby_api_retry_delay', fallback='5')

        vllm_api_url = config.get('Local-API', 'vllm_api_IP', fallback='http://127.0.0.1:500/api/v1/chat/completions')
        vllm_api_key = config.get('Local-API', 'vllm_api_key', fallback=None)
        vllm_model = config.get('Local-API', 'vllm_model', fallback=None)
        vllm_streaming = config.get('Local-API', 'vllm_streaming', fallback='False')
        vllm_temperature = config.get('Local-API', 'vllm_temperature', fallback='0.7')
        vllm_top_p = config.get('Local-API', 'vllm_top_p', fallback='0.95')
        vllm_top_k = config.get('Local-API', 'vllm_top_k', fallback='100')
        vllm_min_p = config.get('Local-API', 'vllm_min_p', fallback='0.05')
        vllm_max_tokens = config.get('Local-API', 'vllm_max_tokens', fallback='4096')
        vllm_api_timeout = config.get('Local-API', 'vllm_api_timeout', fallback='90')
        vllm_api_retries = config.get('Local-API', 'vllm_api_retry', fallback='3')
        vllm_api_retry_delay = config.get('Local-API', 'vllm_api_retry_delay', fallback='5')

        ollama_api_url = config.get('Local-API', 'ollama_api_IP', fallback='http://127.0.0.1:11434/api/generate')
        ollama_api_key = config.get('Local-API', 'ollama_api_key', fallback=None)
        ollama_model = config.get('Local-API', 'ollama_model', fallback=None)
        ollama_streaming = config.get('Local-API', 'ollama_streaming', fallback='False')
        ollama_temperature = config.get('Local-API', 'ollama_temperature', fallback='0.7')
        ollama_top_p = config.get('Local-API', 'ollama_top_p', fallback='0.95')
        ollama_max_tokens = config.get('Local-API', 'ollama_max_tokens', fallback='4096')
        ollama_api_timeout = config.get('Local-API', 'ollama_api_timeout', fallback='90')
        ollama_api_retries = config.get('Local-API', 'ollama_api_retry', fallback='3')
        ollama_api_retry_delay = config.get('Local-API', 'ollama_api_retry_delay', fallback='5')

        aphrodite_api_url = config.get('Local-API', 'aphrodite_api_IP', fallback='http://127.0.0.1:8080/v1/chat/completions')
        aphrodite_api_key = config.get('Local-API', 'aphrodite_api_key', fallback='')
        aphrodite_model = config.get('Local-API', 'aphrodite_model', fallback='')
        aphrodite_max_tokens = config.get('Local-API', 'aphrodite_max_tokens', fallback='4096')
        aphrodite_streaming = config.get('Local-API', 'aphrodite_streaming', fallback='False')
        aphrodite_api_timeout = config.get('Local-API', 'llama_api_timeout', fallback='90')
        aphrodite_api_retries = config.get('Local-API', 'aphrodite_api_retry', fallback='3')
        aphrodite_api_retry_delay = config.get('Local-API', 'aphrodite_api_retry_delay', fallback='5')

        custom_openai_api_key = config.get('API', 'custom_openai_api_key', fallback=None)
        custom_openai_api_ip = config.get('API', 'custom_openai_api_ip', fallback=None)
        custom_openai_api_model = config.get('API', 'custom_openai_api_model', fallback=None)
        custom_openai_api_streaming = config.get('API', 'custom_openai_api_streaming', fallback='False')
        custom_openai_api_temperature = config.get('API', 'custom_openai_api_temperature', fallback='0.7')
        custom_openai_api_top_p = config.get('API', 'custom_openai_api_top_p', fallback='0.95')
        custom_openai_api_min_p = config.get('API', 'custom_openai_api_top_k', fallback='100')
        custom_openai_api_max_tokens = config.get('API', 'custom_openai_api_max_tokens', fallback='4096')
        custom_openai_api_timeout = config.get('API', 'custom_openai_api_timeout', fallback='90')
        custom_openai_api_retries = config.get('API', 'custom_openai_api_retry', fallback='3')
        custom_openai_api_retry_delay = config.get('API', 'custom_openai_api_retry_delay', fallback='5')

        # 2nd Custom OpenAI API
        custom_openai2_api_key = config.get('API', 'custom_openai2_api_key', fallback=None)
        custom_openai2_api_ip = config.get('API', 'custom_openai2_api_ip', fallback=None)
        custom_openai2_api_model = config.get('API', 'custom_openai2_api_model', fallback=None)
        custom_openai2_api_streaming = config.get('API', 'custom_openai2_api_streaming', fallback='False')
        custom_openai2_api_temperature = config.get('API', 'custom_openai2_api_temperature', fallback='0.7')
        custom_openai2_api_top_p = config.get('API', 'custom_openai_api2_top_p', fallback='0.95')
        custom_openai2_api_min_p = config.get('API', 'custom_openai_api2_top_k', fallback='100')
        custom_openai2_api_max_tokens = config.get('API', 'custom_openai2_api_max_tokens', fallback='4096')
        custom_openai2_api_timeout = config.get('API', 'custom_openai2_api_timeout', fallback='90')
        custom_openai2_api_retries = config.get('API', 'custom_openai2_api_retry', fallback='3')
        custom_openai2_api_retry_delay = config.get('API', 'custom_openai2_api_retry_delay', fallback='5')

        # Logging Checks for Local API IP loads
        # logging.debug(f"Loaded Kobold API IP: {kobold_api_ip}")
        # logging.debug(f"Loaded Llama API IP: {llama_api_IP}")
        # logging.debug(f"Loaded Ooba API IP: {ooba_api_IP}")
        # logging.debug(f"Loaded Tabby API IP: {tabby_api_IP}")
        # logging.debug(f"Loaded VLLM API URL: {vllm_api_url}")

        # Retrieve default API choices from the configuration file
        default_api = config.get('API', 'default_api', fallback='openai')

        # Retrieve LLM API settings from the configuration file
        local_api_retries = config.get('Local-API', 'Settings', fallback='3')
        local_api_retry_delay = config.get('Local-API', 'local_api_retry_delay', fallback='5')

        # Retrieve output paths from the configuration file
        output_path = config.get('Paths', 'output_path', fallback='results')
        logging.trace(f"Output path set to: {output_path}")

        # Save video transcripts
        save_video_transcripts = config.get('Paths', 'save_video_transcripts', fallback='True')

        # Retrieve logging settings from the configuration file
        log_level = config.get('Logging', 'log_level', fallback='INFO')
        log_file = config.get('Logging', 'log_file', fallback='./Logs/tldw_logs.json')
        log_metrics_file = config.get('Logging', 'log_metrics_file', fallback='./Logs/tldw_metrics_logs.json')

        # Retrieve processing choice from the configuration file
        processing_choice = config.get('Processing', 'processing_choice', fallback='cpu')
        logging.trace(f"Processing choice set to: {processing_choice}")

        # [Chunking]
        # # Chunking Defaults
        # #
        # # Default Chunking Options for each media type
        chunking_method = config.get('Chunking', 'chunking_method', fallback='words')
        chunk_max_size = config.get('Chunking', 'chunk_max_size', fallback='400')
        chunk_overlap = config.get('Chunking', 'chunk_overlap', fallback='200')
        adaptive_chunking = config.get('Chunking', 'adaptive_chunking', fallback='False')
        chunking_multi_level = config.get('Chunking', 'chunking_multi_level', fallback='False')
        chunk_language = config.get('Chunking', 'chunk_language', fallback='en')
        #
        # Article Chunking
        article_chunking_method = config.get('Chunking', 'article_chunking_method', fallback='words')
        article_chunk_max_size = config.get('Chunking', 'article_chunk_max_size', fallback='400')
        article_chunk_overlap = config.get('Chunking', 'article_chunk_overlap', fallback='200')
        article_adaptive_chunking = config.get('Chunking', 'article_adaptive_chunking', fallback='False')
        article_chunking_multi_level = config.get('Chunking', 'article_chunking_multi_level', fallback='False')
        article_language = config.get('Chunking', 'article_language', fallback='english')
        #
        # Audio file Chunking
        audio_chunking_method = config.get('Chunking', 'audio_chunking_method', fallback='words')
        audio_chunk_max_size = config.get('Chunking', 'audio_chunk_max_size', fallback='400')
        audio_chunk_overlap = config.get('Chunking', 'audio_chunk_overlap', fallback='200')
        audio_adaptive_chunking = config.get('Chunking', 'audio_adaptive_chunking', fallback='False')
        audio_chunking_multi_level = config.get('Chunking', 'audio_chunking_multi_level', fallback='False')
        audio_language = config.get('Chunking', 'audio_language', fallback='english')
        #
        # Book Chunking
        book_chunking_method = config.get('Chunking', 'book_chunking_method', fallback='words')
        book_chunk_max_size = config.get('Chunking', 'book_chunk_max_size', fallback='400')
        book_chunk_overlap = config.get('Chunking', 'book_chunk_overlap', fallback='200')
        book_adaptive_chunking = config.get('Chunking', 'book_adaptive_chunking', fallback='False')
        book_chunking_multi_level = config.get('Chunking', 'book_chunking_multi_level', fallback='False')
        book_language = config.get('Chunking', 'book_language', fallback='english')
        #
        # Document Chunking
        document_chunking_method = config.get('Chunking', 'document_chunking_method', fallback='words')
        document_chunk_max_size = config.get('Chunking', 'document_chunk_max_size', fallback='400')
        document_chunk_overlap = config.get('Chunking', 'document_chunk_overlap', fallback='200')
        document_adaptive_chunking = config.get('Chunking', 'document_adaptive_chunking', fallback='False')
        document_chunking_multi_level = config.get('Chunking', 'document_chunking_multi_level', fallback='False')
        document_language = config.get('Chunking', 'document_language', fallback='english')
        #
        # Mediawiki Article Chunking
        mediawiki_article_chunking_method = config.get('Chunking', 'mediawiki_article_chunking_method', fallback='words')
        mediawiki_article_chunk_max_size = config.get('Chunking', 'mediawiki_article_chunk_max_size', fallback='400')
        mediawiki_article_chunk_overlap = config.get('Chunking', 'mediawiki_article_chunk_overlap', fallback='200')
        mediawiki_article_adaptive_chunking = config.get('Chunking', 'mediawiki_article_adaptive_chunking', fallback='False')
        mediawiki_article_chunking_multi_level = config.get('Chunking', 'mediawiki_article_chunking_multi_level', fallback='False')
        mediawiki_article_language = config.get('Chunking', 'mediawiki_article_language', fallback='english')
        #
        # Mediawiki Dump Chunking
        mediawiki_dump_chunking_method = config.get('Chunking', 'mediawiki_dump_chunking_method', fallback='words')
        mediawiki_dump_chunk_max_size = config.get('Chunking', 'mediawiki_dump_chunk_max_size', fallback='400')
        mediawiki_dump_chunk_overlap = config.get('Chunking', 'mediawiki_dump_chunk_overlap', fallback='200')
        mediawiki_dump_adaptive_chunking = config.get('Chunking', 'mediawiki_dump_adaptive_chunking', fallback='False')
        mediawiki_dump_chunking_multi_level = config.get('Chunking', 'mediawiki_dump_chunking_multi_level', fallback='False')
        mediawiki_dump_language = config.get('Chunking', 'mediawiki_dump_language', fallback='english')
        #
        # Obsidian Note Chunking
        obsidian_note_chunking_method = config.get('Chunking', 'obsidian_note_chunking_method', fallback='words')
        obsidian_note_chunk_max_size = config.get('Chunking', 'obsidian_note_chunk_max_size', fallback='400')
        obsidian_note_chunk_overlap = config.get('Chunking', 'obsidian_note_chunk_overlap', fallback='200')
        obsidian_note_adaptive_chunking = config.get('Chunking', 'obsidian_note_adaptive_chunking', fallback='False')
        obsidian_note_chunking_multi_level = config.get('Chunking', 'obsidian_note_chunking_multi_level', fallback='False')
        obsidian_note_language = config.get('Chunking', 'obsidian_note_language', fallback='english')
        #
        # Podcast Chunking
        podcast_chunking_method = config.get('Chunking', 'podcast_chunking_method', fallback='words')
        podcast_chunk_max_size = config.get('Chunking', 'podcast_chunk_max_size', fallback='400')
        podcast_chunk_overlap = config.get('Chunking', 'podcast_chunk_overlap', fallback='200')
        podcast_adaptive_chunking = config.get('Chunking', 'podcast_adaptive_chunking', fallback='False')
        podcast_chunking_multi_level = config.get('Chunking', 'podcast_chunking_multi_level', fallback='False')
        podcast_language = config.get('Chunking', 'podcast_language', fallback='english')
        #
        # Text Chunking
        text_chunking_method = config.get('Chunking', 'text_chunking_method', fallback='words')
        text_chunk_max_size = config.get('Chunking', 'text_chunk_max_size', fallback='400')
        text_chunk_overlap = config.get('Chunking', 'text_chunk_overlap', fallback='200')
        text_adaptive_chunking = config.get('Chunking', 'text_adaptive_chunking', fallback='False')
        text_chunking_multi_level = config.get('Chunking', 'text_chunking_multi_level', fallback='False')
        text_language = config.get('Chunking', 'text_language', fallback='english')
        #
        # Video Transcription Chunking
        video_chunking_method = config.get('Chunking', 'video_chunking_method', fallback='words')
        video_chunk_max_size = config.get('Chunking', 'video_chunk_max_size', fallback='400')
        video_chunk_overlap = config.get('Chunking', 'video_chunk_overlap', fallback='200')
        video_adaptive_chunking = config.get('Chunking', 'video_adaptive_chunking', fallback='False')
        video_chunking_multi_level = config.get('Chunking', 'video_chunking_multi_level', fallback='False')
        video_language = config.get('Chunking', 'video_language', fallback='english')
        #
        chunking_types = 'article', 'audio', 'book', 'document', 'mediawiki_article', 'mediawiki_dump', 'obsidian_note', 'podcast', 'text', 'video'

        # Retrieve Embedding model settings from the configuration file
        embedding_model = config.get('Embeddings', 'embedding_model', fallback='')
        logging.trace(f"Embedding model set to: {embedding_model}")
        embedding_provider = config.get('Embeddings', 'embedding_provider', fallback='')
        embedding_model = config.get('Embeddings', 'embedding_model', fallback='')
        onnx_model_path = config.get('Embeddings', 'onnx_model_path', fallback="./App_Function_Libraries/onnx_models/text-embedding-3-small.onnx")
        model_dir = config.get('Embeddings', 'model_dir', fallback="./App_Function_Libraries/onnx_models")
        embedding_api_url = config.get('Embeddings', 'embedding_api_url', fallback="http://localhost:8080/v1/embeddings")
        embedding_api_key = config.get('Embeddings', 'embedding_api_key', fallback='')
        chunk_size = config.get('Embeddings', 'chunk_size', fallback=400)
        overlap = config.get('Embeddings', 'overlap', fallback=200)

        # Prompts - FIXME
        prompt_path = config.get('Prompts', 'prompt_path', fallback='Databases/prompts.db')

        # Chat Dictionaries
        enable_chat_dictionaries = config.get('Chat-Dictionaries', 'enable_chat_dictionaries', fallback='False')
        post_gen_replacement = config.get('Chat-Dictionaries', 'post_gen_replacement', fallback='False')
        post_gen_replacement_dict = config.get('Chat-Dictionaries', 'post_gen_replacement_dict', fallback='')
        chat_dict_chat_prompts = config.get('Chat-Dictionaries', 'chat_dictionary_chat_prompts', fallback='')
        chat_dict_rag_prompts = config.get('Chat-Dictionaries', 'chat_dictionary_RAG_prompts', fallback='')
        chat_dict_replacement_strategy = config.get('Chat-Dictionaries', 'chat_dictionary_replacement_strategy', fallback='character_lore_first')
        chat_dict_max_tokens = config.get('Chat-Dictionaries', 'chat_dictionary_max_tokens', fallback='1000')
        default_rag_prompt = config.get('Chat-Dictionaries', 'default_rag_prompt', fallback='')

        # Auto-Save Values
        save_character_chats = config.get('Auto-Save', 'save_character_chats', fallback='False')
        save_rag_chats = config.get('Auto-Save', 'save_rag_chats', fallback='False')

        # Local API Timeout
        local_api_timeout = config.get('Local-API', 'local_api_timeout', fallback='90')

        # STT Settings
        default_stt_provider = config.get('STT-Settings', 'default_stt_provider', fallback='faster_whisper')

        # TTS Settings
        # FIXME
        local_tts_device = config.get('TTS-Settings', 'local_tts_device', fallback='cpu')
        default_tts_provider = config.get('TTS-Settings', 'default_tts_provider', fallback='openai')
        tts_voice = config.get('TTS-Settings', 'default_tts_voice', fallback='shimmer')
        # Open AI TTS
        default_openai_tts_model = config.get('TTS-Settings', 'default_openai_tts_model', fallback='tts-1-hd')
        default_openai_tts_voice = config.get('TTS-Settings', 'default_openai_tts_voice', fallback='shimmer')
        default_openai_tts_speed = config.get('TTS-Settings', 'default_openai_tts_speed', fallback='1')
        default_openai_tts_output_format = config.get('TTS-Settings', 'default_openai_tts_output_format', fallback='mp3')
        default_openai_tts_streaming = config.get('TTS-Settings', 'default_openai_tts_streaming', fallback='False')
        # Google TTS
        # FIXME - FIX THESE DEFAULTS
        default_google_tts_model = config.get('TTS-Settings', 'default_google_tts_model', fallback='en')
        default_google_tts_voice = config.get('TTS-Settings', 'default_google_tts_voice', fallback='en')
        default_google_tts_speed = config.get('TTS-Settings', 'default_google_tts_speed', fallback='1')
        # ElevenLabs TTS
        default_eleven_tts_model = config.get('TTS-Settings', 'default_eleven_tts_model', fallback='FIXME')
        default_eleven_tts_voice = config.get('TTS-Settings', 'default_eleven_tts_voice', fallback='FIXME')
        default_eleven_tts_language_code = config.get('TTS-Settings', 'default_eleven_tts_language_code', fallback='FIXME')
        default_eleven_tts_voice_stability = config.get('TTS-Settings', 'default_eleven_tts_voice_stability', fallback='FIXME')
        default_eleven_tts_voice_similiarity_boost = config.get('TTS-Settings', 'default_eleven_tts_voice_similiarity_boost', fallback='FIXME')
        default_eleven_tts_voice_style = config.get('TTS-Settings', 'default_eleven_tts_voice_style', fallback='FIXME')
        default_eleven_tts_voice_use_speaker_boost = config.get('TTS-Settings', 'default_eleven_tts_voice_use_speaker_boost', fallback='FIXME')
        default_eleven_tts_output_format = config.get('TTS-Settings', 'default_eleven_tts_output_format',
                                                      fallback='mp3_44100_192')
        # AllTalk TTS
        alltalk_api_ip = config.get('TTS-Settings', 'alltalk_api_ip', fallback='http://127.0.0.1:7851/v1/audio/speech')
        default_alltalk_tts_model = config.get('TTS-Settings', 'default_alltalk_tts_model', fallback='alltalk_model')
        default_alltalk_tts_voice = config.get('TTS-Settings', 'default_alltalk_tts_voice', fallback='alloy')
        default_alltalk_tts_speed = config.get('TTS-Settings', 'default_alltalk_tts_speed', fallback=1.0)
        default_alltalk_tts_output_format = config.get('TTS-Settings', 'default_alltalk_tts_output_format', fallback='mp3')

        # Kokoro TTS
        kokoro_model_path = config.get('TTS-Settings', 'kokoro_model_path', fallback='Databases/kokoro_models')
        default_kokoro_tts_model = config.get('TTS-Settings', 'default_kokoro_tts_model', fallback='pht')
        default_kokoro_tts_voice = config.get('TTS-Settings', 'default_kokoro_tts_voice', fallback='sky')
        default_kokoro_tts_speed = config.get('TTS-Settings', 'default_kokoro_tts_speed', fallback=1.0)
        default_kokoro_tts_output_format = config.get('TTS-Settings', 'default_kokoro_tts_output_format', fallback='wav')


        # Self-hosted OpenAI API TTS
        default_openai_api_tts_model = config.get('TTS-Settings', 'default_openai_api_tts_model', fallback='tts-1-hd')
        default_openai_api_tts_voice = config.get('TTS-Settings', 'default_openai_api_tts_voice', fallback='shimmer')
        default_openai_api_tts_speed = config.get('TTS-Settings', 'default_openai_api_tts_speed', fallback='1')
        default_openai_api_tts_output_format = config.get('TTS-Settings', 'default_openai_tts_api_output_format', fallback='mp3')
        default_openai_api_tts_streaming = config.get('TTS-Settings', 'default_openai_tts_streaming', fallback='False')


        # Search Engines
        search_provider_default = config.get('Search-Engines', 'search_provider_default', fallback='google')
        search_language_query = config.get('Search-Engines', 'search_language_query', fallback='en')
        search_language_results = config.get('Search-Engines', 'search_language_results', fallback='en')
        search_language_analysis = config.get('Search-Engines', 'search_language_analysis', fallback='en')
        search_default_max_queries = 10
        search_enable_subquery = config.get('Search-Engines', 'search_enable_subquery', fallback='True')
        search_enable_subquery_count_max = config.get('Search-Engines', 'search_enable_subquery_count_max', fallback=5)
        search_result_rerank = config.get('Search-Engines', 'search_result_rerank', fallback='True')
        search_result_max = config.get('Search-Engines', 'search_result_max', fallback=10)
        search_result_max_per_query = config.get('Search-Engines', 'search_result_max_per_query', fallback=10)
        search_result_blacklist = config.get('Search-Engines', 'search_result_blacklist', fallback='')
        search_result_display_type = config.get('Search-Engines', 'search_result_display_type', fallback='list')
        search_result_display_metadata = config.get('Search-Engines', 'search_result_display_metadata', fallback='False')
        search_result_save_to_db = config.get('Search-Engines', 'search_result_save_to_db', fallback='True')
        search_result_analysis_tone = config.get('Search-Engines', 'search_result_analysis_tone', fallback='')
        relevance_analysis_llm = config.get('Search-Engines', 'relevance_analysis_llm', fallback='False')
        final_answer_llm = config.get('Search-Engines', 'final_answer_llm', fallback='False')
        # Search Engine Specifics
        baidu_search_api_key = config.get('Search-Engines', 'search_engine_api_key_baidu', fallback='')
        # Bing Search Settings
        bing_search_api_key = config.get('Search-Engines', 'search_engine_api_key_bing', fallback='')
        bing_country_code = config.get('Search-Engines', 'search_engine_country_code_bing', fallback='us')
        bing_search_api_url = config.get('Search-Engines', 'search_engine_api_url_bing', fallback='')
        # Brave Search Settings
        brave_search_api_key = config.get('Search-Engines', 'search_engine_api_key_brave_regular', fallback='')
        brave_search_ai_api_key = config.get('Search-Engines', 'search_engine_api_key_brave_ai', fallback='')
        brave_country_code = config.get('Search-Engines', 'search_engine_country_code_brave', fallback='us')
        # DuckDuckGo Search Settings
        duckduckgo_search_api_key = config.get('Search-Engines', 'search_engine_api_key_duckduckgo', fallback='')
        # Google Search Settings
        google_search_api_url = config.get('Search-Engines', 'search_engine_api_url_google', fallback='')
        google_search_api_key = config.get('Search-Engines', 'search_engine_api_key_google', fallback='')
        google_search_engine_id = config.get('Search-Engines', 'search_engine_id_google', fallback='')
        google_simp_trad_chinese = config.get('Search-Engines', 'enable_traditional_chinese', fallback='0')
        limit_google_search_to_country = config.get('Search-Engines', 'limit_google_search_to_country', fallback='0')
        google_search_country = config.get('Search-Engines', 'google_search_country', fallback='us')
        google_search_country_code = config.get('Search-Engines', 'google_search_country_code', fallback='us')
        google_filter_setting = config.get('Search-Engines', 'google_filter_setting', fallback='1')
        google_user_geolocation = config.get('Search-Engines', 'google_user_geolocation', fallback='')
        google_ui_language = config.get('Search-Engines', 'google_ui_language', fallback='en')
        google_limit_search_results_to_language = config.get('Search-Engines', 'google_limit_search_results_to_language', fallback='')
        google_default_search_results = config.get('Search-Engines', 'google_default_search_results', fallback='10')
        google_safe_search = config.get('Search-Engines', 'google_safe_search', fallback='active')
        google_enable_site_search = config.get('Search-Engines', 'google_enable_site_search', fallback='0')
        google_site_search_include = config.get('Search-Engines', 'google_site_search_include', fallback='')
        google_site_search_exclude = config.get('Search-Engines', 'google_site_search_exclude', fallback='')
        google_sort_results_by = config.get('Search-Engines', 'google_sort_results_by', fallback='relevance')
        # Kagi Search Settings
        kagi_search_api_key = config.get('Search-Engines', 'search_engine_api_key_kagi', fallback='')
        # Searx Search Settings
        search_engine_searx_api = config.get('Search-Engines', 'search_engine_searx_api', fallback='')
        # Tavily Search Settings
        tavily_search_api_key = config.get('Search-Engines', 'search_engine_api_key_tavily', fallback='')
        # Yandex Search Settings
        yandex_search_api_key = config.get('Search-Engines', 'search_engine_api_key_yandex', fallback='')
        yandex_search_engine_id = config.get('Search-Engines', 'search_engine_id_yandex', fallback='')

        # Prompts
        sub_question_generation_prompt = config.get('Prompts', 'sub_question_generation_prompt', fallback='')
        search_result_relevance_eval_prompt = config.get('Prompts', 'search_result_relevance_eval_prompt', fallback='')
        analyze_search_results_prompt = config.get('Prompts', 'analyze_search_results_prompt', fallback='')

        # Web Scraper settings
        web_scraper_api_key = config.get('Web-Scraper', 'web_scraper_api_key', fallback='')
        web_scraper_api_url = config.get('Web-Scraper', 'web_scraper_api_url', fallback='')
        web_scraper_api_timeout = config.get('Web-Scraper', 'web_scraper_api_timeout', fallback='90')
        web_scraper_api_retries = config.get('Web-Scraper', 'web_scraper_api_retries', fallback='3')
        web_scraper_api_retry_delay = config.get('Web-Scraper', 'web_scraper_api_retry_delay', fallback='5')
        web_scraper_retry_count = config.get('Web-Scraper', 'web_scraper_retry_count', fallback='3')
        web_scraper_retry_timeout = config.get('Web-Scraper', 'web_scraper_retry_timeout', fallback='5')
        web_scraper_stealth_playwright = config.get('Web-Scraper', 'web_scraper_stealth_playwright', fallback='False')

        return {
            'anthropic_api': {
                'api_key': anthropic_api_key,
                'model': anthropic_model,
                'streaming': anthropic_streaming,
                'temperature': anthropic_temperature,
                'top_p': anthropic_top_p,
                'top_k': anthropic_top_k,
                'max_tokens': anthropic_max_tokens,
                'api_timeout': anthropic_api_timeout,
                'api_retries': anthropic_api_retries,
                'api_retry_delay': anthropic_api_retry_delay
            },
            'cohere_api': {
                'api_key': cohere_api_key,
                'model': cohere_model,
                'streaming': cohere_streaming,
                'temperature': cohere_temperature,
                'max_p': cohere_max_p,
                'top_k': cohere_top_k,
                'max_tokens': cohere_max_tokens,
                'api_timeout': cohere_api_timeout,
                'api_retries': cohere_api_retries,
                'api_retry_delay': cohere_api_retry_delay
            },
            'deepseek_api': {
                'api_key': deepseek_api_key,
                'model': deepseek_model,
                'streaming': deepseek_streaming,
                'temperature': deepseek_temperature,
                'top_p': deepseek_top_p,
                'min_p': deepseek_min_p,
                'max_tokens': deepseek_max_tokens,
                'api_timeout': deepseek_api_timeout,
                'api_retries': deepseek_api_retries,
                'api_retry_delay': deepseek_api_retry_delay
            },
            'google_api': {
                'api_key': google_api_key,
                'model': google_model,
                'streaming': google_streaming,
                'temperature': google_temperature,
                'top_p': google_top_p,
                'min_p': google_min_p,
                'max_tokens': google_max_tokens,
                'api_timeout': google_api_timeout,
                'api_retries': google_api_retries,
                'api_retry_delay': google_api_retry_delay
            },
            'groq_api': {
                'api_key': groq_api_key,
                'model': groq_model,
                'streaming': groq_streaming,
                'temperature': groq_temperature,
                'top_p': groq_top_p,
                'max_tokens': groq_max_tokens,
                'api_timeout': groq_api_timeout,
                'api_retries': groq_api_retries,
                'api_retry_delay': groq_api_retry_delay
            },
            'huggingface_api': {
                'api_key': huggingface_api_key,
                'model': huggingface_model,
                'streaming': huggingface_streaming,
                'temperature': huggingface_temperature,
                'top_p': huggingface_top_p,
                'min_p': huggingface_min_p,
                'max_tokens': huggingface_max_tokens,
                'api_timeout': huggingface_api_timeout,
                'api_retries': huggingface_api_retries,
                'api_retry_delay': huggingface_api_retry_delay
            },
            'mistral_api': {
                'api_key': mistral_api_key,
                'model': mistral_model,
                'streaming': mistral_streaming,
                'temperature': mistral_temperature,
                'top_p': mistral_top_p,
                'max_tokens': mistral_max_tokens,
                'api_timeout': mistral_api_timeout,
                'api_retries': mistral_api_retries,
                'api_retry_delay': mistral_api_retry_delay
            },
            'openrouter_api': {
                'api_key': openrouter_api_key,
                'model': openrouter_model,
                'streaming': openrouter_streaming,
                'temperature': openrouter_temperature,
                'top_p': openrouter_top_p,
                'min_p': openrouter_min_p,
                'top_k': openrouter_top_k,
                'max_tokens': openrouter_max_tokens,
                'api_timeout': openrouter_api_timeout,
                'api_retries': openrouter_api_retries,
                'api_retry_delay': openrouter_api_retry_delay
            },
            'openai_api': {
                'api_key': openai_api_key,
                'model': openai_model,
                'streaming': openai_streaming,
                'temperature': openai_temperature,
                'top_p': openai_top_p,
                'max_tokens': openai_max_tokens,
                'api_timeout': openai_api_timeout,
                'api_retries': openai_api_retries,
                'api_retry_delay': openai_api_retry_delay
            },
            'elevenlabs_api': {
                'api_key': elevenlabs_api_key,
            },
            'alltalk_api': {
                'api_ip': alltalk_api_ip,
                'default_alltalk_tts_model': default_alltalk_tts_model,
                'default_alltalk_tts_voice': default_alltalk_tts_voice,
                'default_alltalk_tts_speed': default_alltalk_tts_speed,
                'default_alltalk_tts_output_format': default_alltalk_tts_output_format,
            },
            'llama_api': {
                'api_ip': llama_api_IP,
                'api_key': llama_api_key,
                'streaming': llama_streaming,
                'temperature': llama_temperature,
                'top_p': llama_top_p,
                'min_p': llama_min_p,
                'top_k': llama_top_k,
                'max_tokens': llama_max_tokens,
                'api_timeout': llama_api_timeout,
                'api_retries': llama_api_retries,
                'api_retry_delay': llama_api_retry_delay
            },
            'ooba_api': {
                'api_ip': ooba_api_IP,
                'api_key': ooba_api_key,
                'streaming': ooba_streaming,
                'temperature': ooba_temperature,
                'top_p': ooba_top_p,
                'min_p': ooba_min_p,
                'top_k': ooba_top_k,
                'max_tokens': ooba_max_tokens,
                'api_timeout': ooba_api_timeout,
                'api_retries': ooba_api_retries,
                'api_retry_delay': ooba_api_retry_delay
            },
            'kobold_api': {
                'api_ip': kobold_api_ip,
                'api_streaming_ip': kobold_openai_api_IP,
                'api_key': kobold_api_key,
                'streaming': kobold_streaming,
                'temperature': kobold_temperature,
                'top_p': kobold_top_p,
                'top_k': kobold_top_k,
                'max_tokens': kobold_max_tokens,
                'api_timeout': kobold_api_timeout,
                'api_retries': kobold_api_retries,
                'api_retry_delay': kobold_api_retry_delay
            },
            'tabby_api': {
                'api_ip': tabby_api_IP,
                'api_key': tabby_api_key,
                'model': tabby_model,
                'streaming': tabby_streaming,
                'temperature': tabby_temperature,
                'top_p': tabby_top_p,
                'top_k': tabby_top_k,
                'min_p': tabby_min_p,
                'max_tokens': tabby_max_tokens,
                'api_timeout': tabby_api_timeout,
                'api_retries': tabby_api_retries,
                'api_retry_delay': tabby_api_retry_delay
            },
            'vllm_api': {
                'api_ip': vllm_api_url,
                'api_key': vllm_api_key,
                'model': vllm_model,
                'streaming': vllm_streaming,
                'temperature': vllm_temperature,
                'top_p': vllm_top_p,
                'top_k': vllm_top_k,
                'min_p': vllm_min_p,
                'max_tokens': vllm_max_tokens,
                'api_timeout': vllm_api_timeout,
                'api_retries': vllm_api_retries,
                'api_retry_delay': vllm_api_retry_delay
            },
            'ollama_api': {
                'api_url': ollama_api_url,
                'api_key': ollama_api_key,
                'model': ollama_model,
                'streaming': ollama_streaming,
                'temperature': ollama_temperature,
                'top_p': ollama_top_p,
                'max_tokens': ollama_max_tokens,
                'api_timeout': ollama_api_timeout,
                'api_retries': ollama_api_retries,
                'api_retry_delay': ollama_api_retry_delay
            },
            'aphrodite_api': {
                'api_ip': aphrodite_api_url,
                'api_key': aphrodite_api_key,
                'model': aphrodite_model,
                'max_tokens': aphrodite_max_tokens,
                'streaming': aphrodite_streaming,
                'api_timeout': aphrodite_api_timeout,
                'api_retries': aphrodite_api_retries,
                'api_retry_delay': aphrodite_api_retry_delay
            },
            'custom_openai_api': {
                'api_ip': custom_openai_api_ip,
                'api_key': custom_openai_api_key,
                'streaming': custom_openai_api_streaming,
                'model': custom_openai_api_model,
                'temperature': custom_openai_api_temperature,
                'max_tokens': custom_openai_api_max_tokens,
                'top_p': custom_openai_api_top_p,
                'min_p': custom_openai_api_min_p,
                'api_timeout': custom_openai_api_timeout,
                'api_retries': custom_openai_api_retries,
                'api_retry_delay': custom_openai_api_retry_delay
            },
            'custom_openai_api_2': {
                'api_ip': custom_openai2_api_ip,
                'api_key': custom_openai2_api_key,
                'streaming': custom_openai2_api_streaming,
                'model': custom_openai2_api_model,
                'temperature': custom_openai2_api_temperature,
                'max_tokens': custom_openai2_api_max_tokens,
                'top_p': custom_openai2_api_top_p,
                'min_p': custom_openai2_api_min_p,
                'api_timeout': custom_openai2_api_timeout,
                'api_retries': custom_openai2_api_retries,
                'api_retry_delay': custom_openai2_api_retry_delay
            },
            'llm_api_settings': {
                'default_api': default_api,
                'local_api_timeout': local_api_timeout,
                'local_api_retries': local_api_retries,
                'local_api_retry_delay': local_api_retry_delay,
            },
            'output_path': output_path,
            'system_preferences': {
                'save_video_transcripts': save_video_transcripts,
            },
            'processing_choice': processing_choice,
            'chat_dictionaries': {
                'enable_chat_dictionaries': enable_chat_dictionaries,
                'post_gen_replacement': post_gen_replacement,
                'post_gen_replacement_dict': post_gen_replacement_dict,
                'chat_dict_chat_prompts': chat_dict_chat_prompts,
                'chat_dict_RAG_prompts': chat_dict_rag_prompts,
                'chat_dict_replacement_strategy': chat_dict_replacement_strategy,
                'chat_dict_max_tokens': chat_dict_max_tokens,
                'default_rag_prompt': default_rag_prompt
            },
            'chunking_config': {
                'chunking_method': chunking_method,
                'chunk_max_size': chunk_max_size,
                'adaptive_chunking': adaptive_chunking,
                'multi_level': chunking_multi_level,
                'chunk_language': chunk_language,
                'chunk_overlap': chunk_overlap,
                'article_chunking_method': article_chunking_method,
                'article_chunk_max_size': article_chunk_max_size,
                'article_chunk_overlap': article_chunk_overlap,
                'article_adaptive_chunking': article_adaptive_chunking,
                'article_chunking_multi_level': article_chunking_multi_level,
                'article_language': article_language,
                'audio_chunking_method': audio_chunking_method,
                'audio_chunk_max_size': audio_chunk_max_size,
                'audio_chunk_overlap': audio_chunk_overlap,
                'audio_adaptive_chunking': audio_adaptive_chunking,
                'audio_chunking_multi_level': audio_chunking_multi_level,
                'audio_language': audio_language,
                'book_chunking_method': book_chunking_method,
                'book_chunk_max_size': book_chunk_max_size,
                'book_chunk_overlap': book_chunk_overlap,
                'book_adaptive_chunking': book_adaptive_chunking,
                'book_chunking_multi_level': book_chunking_multi_level,
                'book_language': book_language,
                'document_chunking_method': document_chunking_method,
                'document_chunk_max_size': document_chunk_max_size,
                'document_chunk_overlap': document_chunk_overlap,
                'document_adaptive_chunking': document_adaptive_chunking,
                'document_chunking_multi_level': document_chunking_multi_level,
                'document_language': document_language,
                'mediawiki_article_chunking_method': mediawiki_article_chunking_method,
                'mediawiki_article_chunk_max_size': mediawiki_article_chunk_max_size,
                'mediawiki_article_chunk_overlap': mediawiki_article_chunk_overlap,
                'mediawiki_article_adaptive_chunking': mediawiki_article_adaptive_chunking,
                'mediawiki_article_chunking_multi_level': mediawiki_article_chunking_multi_level,
                'mediawiki_article_language': mediawiki_article_language,
                'mediawiki_dump_chunking_method': mediawiki_dump_chunking_method,
                'mediawiki_dump_chunk_max_size': mediawiki_dump_chunk_max_size,
                'mediawiki_dump_chunk_overlap': mediawiki_dump_chunk_overlap,
                'mediawiki_dump_adaptive_chunking': mediawiki_dump_adaptive_chunking,
                'mediawiki_dump_chunking_multi_level': mediawiki_dump_chunking_multi_level,
                'mediawiki_dump_language': mediawiki_dump_language,
                'obsidian_note_chunking_method': obsidian_note_chunking_method,
                'obsidian_note_chunk_max_size': obsidian_note_chunk_max_size,
                'obsidian_note_chunk_overlap': obsidian_note_chunk_overlap,
                'obsidian_note_adaptive_chunking': obsidian_note_adaptive_chunking,
                'obsidian_note_chunking_multi_level': obsidian_note_chunking_multi_level,
                'obsidian_note_language': obsidian_note_language,
                'podcast_chunking_method': podcast_chunking_method,
                'podcast_chunk_max_size': podcast_chunk_max_size,
                'podcast_chunk_overlap': podcast_chunk_overlap,
                'podcast_adaptive_chunking': podcast_adaptive_chunking,
                'podcast_chunking_multi_level': podcast_chunking_multi_level,
                'podcast_language': podcast_language,
                'text_chunking_method': text_chunking_method,
                'text_chunk_max_size': text_chunk_max_size,
                'text_chunk_overlap': text_chunk_overlap,
                'text_adaptive_chunking': text_adaptive_chunking,
                'text_chunking_multi_level': text_chunking_multi_level,
                'text_language': text_language,
                'video_chunking_method': video_chunking_method,
                'video_chunk_max_size': video_chunk_max_size,
                'video_chunk_overlap': video_chunk_overlap,
                'video_adaptive_chunking': video_adaptive_chunking,
                'video_chunking_multi_level': video_chunking_multi_level,
                'video_language': video_language,
            },
            #chunking_types = 'article', 'audio', 'book', 'document', 'mediawiki_article', 'mediawiki_dump', 'obsidian_note', 'podcast', 'text', 'video'
            'db_config': {
                'prompt_path': get_project_relative_path(config.get('Prompts', 'prompt_path', fallback='Databases/prompts.db')),
                'db_type': config.get('Database', 'type', fallback='sqlite'),
                'sqlite_path': get_project_relative_path(config.get('Database', 'sqlite_path', fallback='Databases/media_summary.db')),
                'elasticsearch_host': config.get('Database', 'elasticsearch_host', fallback='localhost'),
                'elasticsearch_port': config.getint('Database', 'elasticsearch_port', fallback=9200),
                'chroma_db_path': get_project_relative_path(config.get('Database', 'chroma_db_path', fallback='Databases/chroma.db'))
            },
            'embedding_config': {
                'embedding_provider': embedding_provider,
                'embedding_model': embedding_model,
                'onnx_model_path': onnx_model_path,
                'model_dir': model_dir,
                'embedding_api_url': embedding_api_url,
                'embedding_api_key': embedding_api_key,
                'chunk_size': chunk_size,
                'chunk_overlap': overlap
            },
            'logging': {
                'log_level': log_level,
                'log_file': log_file,
                'log_metrics_file': log_metrics_file
            },
            'auto-save': {
                'save_character_chats': save_character_chats,
                'save_rag_chats': save_rag_chats,
            },
            'default_api': default_api,
            'local_api_timeout': local_api_timeout,
            'STT_Settings': {
                'default_stt_provider': default_stt_provider,
            },
            'tts_settings': {
                'default_tts_provider': default_tts_provider,
                'tts_voice': tts_voice,
                'local_tts_device': local_tts_device,
                # OpenAI
                'default_openai_tts_voice': default_openai_tts_voice,
                'default_openai_tts_speed': default_openai_tts_speed,
                'default_openai_tts_model': default_openai_tts_model,
                'default_openai_tts_output_format': default_openai_tts_output_format,
                # Google
                'default_google_tts_model': default_google_tts_model,
                'default_google_tts_voice': default_google_tts_voice,
                'default_google_tts_speed': default_google_tts_speed,
                # ElevenLabs
                'default_eleven_tts_model': default_eleven_tts_model,
                'default_eleven_tts_voice': default_eleven_tts_voice,
                'default_eleven_tts_language_code': default_eleven_tts_language_code,
                'default_eleven_tts_voice_stability': default_eleven_tts_voice_stability,
                'default_eleven_tts_voice_similiarity_boost': default_eleven_tts_voice_similiarity_boost,
                'default_eleven_tts_voice_style': default_eleven_tts_voice_style,
                'default_eleven_tts_voice_use_speaker_boost': default_eleven_tts_voice_use_speaker_boost,
                'default_eleven_tts_output_format': default_eleven_tts_output_format,
                # Open Source / Self-Hosted TTS
                # GPT SoVITS
                # 'default_gpt_tts_model': default_gpt_tts_model,
                # 'default_gpt_tts_voice': default_gpt_tts_voice,
                # 'default_gpt_tts_speed': default_gpt_tts_speed,
                # 'default_gpt_tts_output_format': default_gpt_tts_output_format
                # AllTalk
                'alltalk_api_ip': alltalk_api_ip,
                'default_alltalk_tts_model': default_alltalk_tts_model,
                'default_alltalk_tts_voice': default_alltalk_tts_voice,
                'default_alltalk_tts_speed': default_alltalk_tts_speed,
                'default_alltalk_tts_output_format': default_alltalk_tts_output_format,
                # Kokoro
                'default_kokoro_tts_model': default_kokoro_tts_model,
                'default_kokoro_tts_voice': default_kokoro_tts_voice,
                'default_kokoro_tts_speed': default_kokoro_tts_speed,
                'default_kokoro_tts_output_format': default_kokoro_tts_output_format,
                # Self-hosted OpenAI API
                'default_openai_api_tts_model': default_openai_api_tts_model,
                'default_openai_api_tts_voice': default_openai_api_tts_voice,
                'default_openai_api_tts_speed': default_openai_api_tts_speed,
                'default_openai_api_tts_output_format': default_openai_api_tts_output_format,
                'default_openai_api_tts_streaming': default_openai_api_tts_streaming,
            },
            'search_settings': {
                'default_search_provider': search_provider_default,
                'search_language_query': search_language_query,
                'search_language_results': search_language_results,
                'search_language_analysis': search_language_analysis,
                'search_default_max_queries': search_default_max_queries,
                'search_enable_subquery': search_enable_subquery,
                'search_enable_subquery_count_max': search_enable_subquery_count_max,
                'search_result_rerank': search_result_rerank,
                'search_result_max': search_result_max,
                'search_result_max_per_query': search_result_max_per_query,
                'search_result_blacklist': search_result_blacklist,
                'search_result_display_type': search_result_display_type,
                'search_result_display_metadata': search_result_display_metadata,
                'search_result_save_to_db': search_result_save_to_db,
                'search_result_analysis_tone': search_result_analysis_tone,
                'relevance_analysis_llm': relevance_analysis_llm,
                'final_answer_llm': final_answer_llm,
            },
            'search_engines': {
                'baidu_search_api_key': baidu_search_api_key,
                'bing_search_api_key': bing_search_api_key,
                'bing_country_code': bing_country_code,
                'bing_search_api_url': bing_search_api_url,
                'brave_search_api_key': brave_search_api_key,
                'brave_search_ai_api_key': brave_search_ai_api_key,
                'brave_country_code': brave_country_code,
                'duckduckgo_search_api_key': duckduckgo_search_api_key,
                'google_search_api_url': google_search_api_url,
                'google_search_api_key': google_search_api_key,
                'google_search_engine_id': google_search_engine_id,
                'google_simp_trad_chinese': google_simp_trad_chinese,
                'limit_google_search_to_country': limit_google_search_to_country,
                'google_search_country': google_search_country,
                'google_search_country_code': google_search_country_code,
                'google_search_filter_setting': google_filter_setting,
                'google_user_geolocation': google_user_geolocation,
                'google_ui_language': google_ui_language,
                'google_limit_search_results_to_language': google_limit_search_results_to_language,
                'google_site_search_include': google_site_search_include,
                'google_site_search_exclude': google_site_search_exclude,
                'google_sort_results_by': google_sort_results_by,
                'google_default_search_results': google_default_search_results,
                'google_safe_search': google_safe_search,
                'google_enable_site_search' : google_enable_site_search,
                'kagi_search_api_key': kagi_search_api_key,
                'searx_search_api_url': search_engine_searx_api,
                'tavily_search_api_key': tavily_search_api_key,
                'yandex_search_api_key': yandex_search_api_key,
                'yandex_search_engine_id': yandex_search_engine_id
            },
            'prompts': {
                'sub_question_generation_prompt': sub_question_generation_prompt,
                'search_result_relevance_eval_prompt': search_result_relevance_eval_prompt,
                'analyze_search_results_prompt': analyze_search_results_prompt,
            },
            'web_scraper':{
                'web_scraper_api_key': web_scraper_api_key,
                'web_scraper_api_url': web_scraper_api_url,
                'web_scraper_api_timeout': web_scraper_api_timeout,
                'web_scraper_api_retries': web_scraper_api_retries,
                'web_scraper_api_retry_delay': web_scraper_api_retry_delay,
                'web_scraper_retry_count': web_scraper_retry_count,
                'web_scraper_retry_timeout': web_scraper_retry_timeout,
                'web_scraper_stealth_playwright': web_scraper_stealth_playwright,
            }
        }
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        return None


global_api_endpoints = ["anthropic", "cohere", "google", "groq", "openai", "huggingface", "openrouter", "deepseek", "mistral", "custom_openai_api", "custom_openai_api_2", "llama", "ollama", "ooba", "kobold", "tabby", "vllm", "aphrodite"]

global_search_engines = ["baidu", "bing", "brave", "duckduckgo", "google", "kagi", "searx", "tavily", "yandex"]

openai_tts_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]


# Setup Default API Endpoint
try:
    loaded_config_data = load_and_log_configs()
    default_api_endpoint = loaded_config_data['default_api']
    print(f"Default API Endpoint: {default_api_endpoint}")
except Exception as e:
    logging.error(f"Error loading default API endpoint: {str(e)}")
    default_api_endpoint = "openai"


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
        'sqlite_path': config.get('Database', 'sqlite_path', fallback='./Databases/media_summary.db'),
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
