# config.py
# Description: Configuration settings for the tldw server application.
#
# Imports
import configparser
import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict


#
# 3rd-party Libraries
from loguru import logger#
# Local Imports
#
########################################################################################################################
#
# Functions:

# --- Constants ---
# Client ID used by the Server API itself when writing to sync logs
SERVER_CLIENT_ID = "SERVER_API_V1"

# --- File Validation/YARA Settings ---
YARA_RULES_PATH: Optional[str] = None # e.g., "/app/yara_rules/index.yar"
MAGIC_FILE_PATH: Optional[str] = os.getenv("MAGIC_FILE_PATH", None) # e.g., "/app/magic.mgc"

# --- Chunking Settings ---
global_default_chunk_language = "en"


# FIXME - TTS Config
APP_CONFIG = {
    "OPENAI_API_KEY": "sk-...",
    "KOKORO_ONNX_MODEL_PATH_DEFAULT": "path/to/your/downloaded/kokoro-v0_19.onnx",
    "KOKORO_ONNX_VOICES_JSON_DEFAULT": "path/to/your/downloaded/voices.json",
    "KOKORO_DEVICE_DEFAULT": "cpu", # or "cuda"
    "ELEVENLABS_API_KEY": "el-...",
    "local_kokoro_default_onnx": { # Specific overrides for this backend_id
        "KOKORO_DEVICE": "cuda:0"
    },
    "global_tts_settings": {
        # shared settings
    }
}

DATABASE_CONFIG = {
    }

RAG_SEARCH_CONFIG = {
    "fts_top_k": 10,
    "vector_top_k": 10,
    "web_vector_top_k": 10,
    "llm_context_document_limit": 10,
    "chat_context_limit": 10,
}

def load_openai_mappings() -> Dict:
    # Determine path relative to this file or use an absolute/configurable path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assume mappings.json is in a 'configs' folder at the project root
    # or adjust as needed. For example, if it's next to this router file:
    # mapping_path = os.path.join(current_dir, "openai_tts_mappings.json")

    # Example: if your project root is one level up from 'routers'
    # and configs is at root:
    project_root = os.path.dirname(os.path.dirname(current_dir))
    mapping_path = os.path.join(project_root, "Config_Files", "openai_tts_mappings.json")
    try:
        with open(mapping_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load OpenAI TTS mappings from {mapping_path}: {e}", exc_info=True)
        # Fallback to a default or raise an error
        return {
            "models": {"tts-1": "openai_official_tts-1"},
            "voices": {"alloy": "alloy"}
        }

_openai_mappings = load_openai_mappings()

openai_tts_mappings = {
    "models": {
        "tts-1": "openai_official_tts-1",
        "tts-1-hd": "openai_official_tts-1-hd",
        "eleven_monolingual_v1": "elevenlabs_english_v1",
        "kokoro": "local_kokoro_default_onnx"
    },
    "voices": {
        "alloy": "alloy", "echo": "echo", "fable": "fable",
        "onyx": "onyx", "nova": "nova", "shimmer": "shimmer",

        "RachelEL": "21m00Tcm4TlvDq8ikWAM",

        "k_bella": "af_bella",
        "k_adam" : "am_v0adam"
    }
}

# --- Helper Function (Optional but can keep dictionary creation clean) ---
def load_settings():
    """Loads all settings from environment variables or defaults into a dictionary."""

    # --- Application Mode ---
    single_user_mode_str = os.getenv("APP_MODE", "single").lower()
    single_user_mode = single_user_mode_str != "multi"

    # --- Single-User Settings ---
    # Use a fixed ID for the single user's database path and cache key
    single_user_fixed_id = int(os.getenv("SINGLE_USER_FIXED_ID", "0")) # Ensure it's an int
    # API Key for accessing the single-user instance
    single_user_api_key = os.getenv("API_KEY", "default-secret-key-for-single-user")

    # --- Multi-User Settings (JWT) ---
    jwt_secret_key = os.getenv("JWT_SECRET_KEY", "a_very_insecure_default_secret_key_for_dev_only")
    jwt_algorithm = "HS256"
    access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

    # --- Database Settings ---
    user_db_base_dir = Path(os.getenv("USER_DB_BASE_DIR", "./user_databases/"))
    # Flag to indicate if the central Users DB is configured
    users_db_configured = os.getenv("USERS_DB_ENABLED", "false").lower() == "true"
    database_url = os.getenv("DATABASE_URL", f"sqlite:///{Path('./tldw_data/databases/tldw.db').resolve()}") # Example path

    # --- Logging ---
    log_level = os.getenv("LOG_LEVEL", "I NFO").upper()

    # --- Build the Settings Dictionary ---
    config_dict = {
        # General App
        "APP_MODE_STR": single_user_mode_str, # Optional: Keep the raw string if needed elsewhere
        "SINGLE_USER_MODE": single_user_mode,
        "LOG_LEVEL": log_level,

        # Single User
        "SINGLE_USER_FIXED_ID": single_user_fixed_id,
        "SINGLE_USER_API_KEY": single_user_api_key,

        # Multi User / Auth
        "JWT_SECRET_KEY": jwt_secret_key,
        "JWT_ALGORITHM": jwt_algorithm,
        "ACCESS_TOKEN_EXPIRE_MINUTES": access_token_expire_minutes,

        # Database
        "DATABASE_URL": database_url, # Main DB URL if used
        "USER_DB_BASE_DIR": user_db_base_dir,
        "USERS_DB_CONFIGURED": users_db_configured,

        # Server Specific (Constants might also live here or stay module-level)
        "SERVER_CLIENT_ID": SERVER_CLIENT_ID,
    }

    # --- Warnings (can be placed after dictionary creation) ---
    if config_dict["SINGLE_USER_MODE"] and config_dict["SINGLE_USER_API_KEY"] == "default-secret-key-for-single-user":
        print("!!! WARNING: Using default API_KEY for single-user mode. Set the API_KEY environment variable for security. !!!")
        print("DEBUGPRING: Using default API_KEY for single-user mode: '{}'".format(config_dict["SINGLE_USER_API_KEY"]))
        #logging.debug("DEBUGPRING: Using default API_KEY for single-user mode: {}".format(config_dict["SINGLE_USER_API_KEY"]))
    if not config_dict["SINGLE_USER_MODE"] and config_dict["JWT_SECRET_KEY"] == "a_very_insecure_default_secret_key_for_dev_only":
        print("!!! SECURITY WARNING: Using default JWT_SECRET_KEY in multi-user mode. Set a strong JWT_SECRET_KEY environment variable! !!!")
    if not config_dict["SINGLE_USER_MODE"] and not config_dict["USERS_DB_CONFIGURED"]:
         print("!!! WARNING: Multi-user mode enabled (APP_MODE=multi), but USERS_DB_ENABLED is not 'true'. User authentication will likely fail. !!!")

    # Create necessary directories if they don't exist
    # Example: Ensure database directory exists
    db_path = Path(config_dict["DATABASE_URL"].replace("sqlite:///", ""))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    config_dict["USER_DB_BASE_DIR"].mkdir(parents=True, exist_ok=True)

    return config_dict


def load_comprehensive_config():
    current_file_path = Path(__file__).resolve()
    # Correct project_root calculation:
    # __file__ is .../tldw_Server_API/app/core/config.py
    # .parent -> .../app/core
    # .parent.parent -> .../app
    # .parent.parent.parent -> .../tldw_Server_API (This is the project root)
    project_root = current_file_path.parent.parent.parent

    config_path_obj = project_root / 'Config_Files' / 'config.txt'

    logger.info(f"Attempting to load comprehensive config from: {str(config_path_obj)}")

    if not config_path_obj.exists():
        logger.error(f"Config file not found at {str(config_path_obj)}")
        raise FileNotFoundError(f"Config file not found at {str(config_path_obj)}")

    config_parser = configparser.ConfigParser()
    try:
        config_parser.read(config_path_obj)  # configparser can read Path objects directly
    except configparser.Error as e:
        logger.error(f"Error parsing config file {str(config_path_obj)}: {e}", exc_info=True)
        raise  # Re-raise the parsing error to be caught by load_and_log_configs

    logger.info(f"load_comprehensive_config(): Sections found in config: {config_parser.sections()}")
    return config_parser

def load_and_log_configs():
    logger.debug("load_and_log_configs(): Loading and logging configurations...")
    try:
        # The 'config' variable below should be the result from load_comprehensive_config()
        config_parser_object = load_comprehensive_config()

        # This check might be redundant if load_comprehensive_config always raises on critical failure
        if config_parser_object is None:
            logger.error("Comprehensive config object is None, cannot proceed")  # Changed to logger
            return None
        # API Keys
        anthropic_api_key = config_parser_object.get('API', 'anthropic_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded Anthropic API Key: {anthropic_api_key[:5]}...{anthropic_api_key[-5:] if anthropic_api_key else None}")

        cohere_api_key = config_parser_object.get('API', 'cohere_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded Cohere API Key: {cohere_api_key[:5]}...{cohere_api_key[-5:] if cohere_api_key else None}")

        groq_api_key = config_parser_object.get('API', 'groq_api_key', fallback=None)
        # logging.debug(f"Loaded Groq API Key: {groq_api_key[:5]}...{groq_api_key[-5:] if groq_api_key else None}")

        openai_api_key = config_parser_object.get('API', 'openai_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded OpenAI API Key: {openai_api_key[:5]}...{openai_api_key[-5:] if openai_api_key else None}")

        huggingface_api_key = config_parser_object.get('API', 'huggingface_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded HuggingFace API Key: {huggingface_api_key[:5]}...{huggingface_api_key[-5:] if huggingface_api_key else None}")

        openrouter_api_key = config_parser_object.get('API', 'openrouter_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded OpenRouter API Key: {openrouter_api_key[:5]}...{openrouter_api_key[-5:] if openrouter_api_key else None}")

        deepseek_api_key = config_parser_object.get('API', 'deepseek_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded DeepSeek API Key: {deepseek_api_key[:5]}...{deepseek_api_key[-5:] if deepseek_api_key else None}")

        mistral_api_key = config_parser_object.get('API', 'mistral_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded Mistral API Key: {mistral_api_key[:5]}...{mistral_api_key[-5:] if mistral_api_key else None}")

        google_api_key = config_parser_object.get('API', 'google_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded Google API Key: {google_api_key[:5]}...{google_api_key[-5:] if google_api_key else None}")

        elevenlabs_api_key = config_parser_object.get('API', 'elevenlabs_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded elevenlabs API Key: {elevenlabs_api_key[:5]}...{elevenlabs_api_key[-5:] if elevenlabs_api_key else None}")

        # LLM API Settings - streaming / temperature / top_p / min_p
        # Anthropic
        anthropic_api_key = config_parser_object.get('API', 'anthropic_api_key', fallback=None)
        anthropic_model = config_parser_object.get('API', 'anthropic_model', fallback='claude-3-5-sonnet-20240620')
        anthropic_streaming = config_parser_object.get('API', 'anthropic_streaming', fallback='False')
        anthropic_temperature = config_parser_object.get('API', 'anthropic_temperature', fallback='0.7')
        anthropic_top_p = config_parser_object.get('API', 'anthropic_top_p', fallback='0.95')
        anthropic_top_k = config_parser_object.get('API', 'anthropic_top_k', fallback='100')
        anthropic_max_tokens = config_parser_object.get('API', 'anthropic_max_tokens', fallback='4096')
        anthropic_api_timeout = config_parser_object.get('API', 'anthropic_api_timeout', fallback='90')
        anthropic_api_retries = config_parser_object.get('API', 'anthropic_api_retry', fallback='3')
        anthropic_api_retry_delay = config_parser_object.get('API', 'anthropic_api_retry_delay', fallback='5')

        # Cohere
        cohere_streaming = config_parser_object.get('API', 'cohere_streaming', fallback='False')
        cohere_temperature = config_parser_object.get('API', 'cohere_temperature', fallback='0.7')
        cohere_max_p = config_parser_object.get('API', 'cohere_max_p', fallback='0.95')
        cohere_top_k = config_parser_object.get('API', 'cohere_top_k', fallback='100')
        cohere_model = config_parser_object.get('API', 'cohere_model', fallback='command-r-plus')
        cohere_max_tokens = config_parser_object.get('API', 'cohere_max_tokens', fallback='4096')
        cohere_api_timeout = config_parser_object.get('API', 'cohere_api_timeout', fallback='90')
        cohere_api_retries = config_parser_object.get('API', 'cohere_api_retry', fallback='3')
        cohere_api_retry_delay = config_parser_object.get('API', 'cohere_api_retry_delay', fallback='5')

        # Deepseek
        deepseek_streaming = config_parser_object.get('API', 'deepseek_streaming', fallback='False')
        deepseek_temperature = config_parser_object.get('API', 'deepseek_temperature', fallback='0.7')
        deepseek_top_p = config_parser_object.get('API', 'deepseek_top_p', fallback='0.95')
        deepseek_min_p = config_parser_object.get('API', 'deepseek_min_p', fallback='0.05')
        deepseek_model = config_parser_object.get('API', 'deepseek_model', fallback='deepseek-chat')
        deepseek_max_tokens = config_parser_object.get('API', 'deepseek_max_tokens', fallback='4096')
        deepseek_api_timeout = config_parser_object.get('API', 'deepseek_api_timeout', fallback='90')
        deepseek_api_retries = config_parser_object.get('API', 'deepseek_api_retry', fallback='3')
        deepseek_api_retry_delay = config_parser_object.get('API', 'deepseek_api_retry_delay', fallback='5')

        # Groq
        groq_model = config_parser_object.get('API', 'groq_model', fallback='llama3-70b-8192')
        groq_streaming = config_parser_object.get('API', 'groq_streaming', fallback='False')
        groq_temperature = config_parser_object.get('API', 'groq_temperature', fallback='0.7')
        groq_top_p = config_parser_object.get('API', 'groq_top_p', fallback='0.95')
        groq_max_tokens = config_parser_object.get('API', 'groq_max_tokens', fallback='4096')
        groq_api_timeout = config_parser_object.get('API', 'groq_api_timeout', fallback='90')
        groq_api_retries = config_parser_object.get('API', 'groq_api_retry', fallback='3')
        groq_api_retry_delay = config_parser_object.get('API', 'groq_api_retry_delay', fallback='5')

        # Google
        google_model = config_parser_object.get('API', 'google_model', fallback='gemini-1.5-pro')
        google_streaming = config_parser_object.get('API', 'google_streaming', fallback='False')
        google_temperature = config_parser_object.get('API', 'google_temperature', fallback='0.7')
        google_top_p = config_parser_object.get('API', 'google_top_p', fallback='0.95')
        google_min_p = config_parser_object.get('API', 'google_min_p', fallback='0.05')
        google_max_tokens = config_parser_object.get('API', 'google_max_tokens', fallback='4096')
        google_api_timeout = config_parser_object.get('API', 'google_api_timeout', fallback='90')
        google_api_retries = config_parser_object.get('API', 'google_api_retry', fallback='3')
        google_api_retry_delay = config_parser_object.get('API', 'google_api_retry_delay', fallback='5')

        # HuggingFace
        huggingface_use_router_url_format = config_parser_object.getboolean('API', 'huggingface_use_router_url_format', fallback=False)
        huggingface_router_base_url = config_parser_object.get('API', 'huggingface_router_base_url', fallback='https://router.huggingface.co/hf-inference')
        huggingface_api_base_url = config_parser_object.get('API', 'huggingface_api_base_url', fallback='https://router.huggingface.co/hf-inference/models')
        huggingface_model = config_parser_object.get('API', 'huggingface_model', fallback='/Qwen/Qwen3-235B-A22B')
        huggingface_streaming = config_parser_object.get('API', 'huggingface_streaming', fallback='False')
        huggingface_temperature = config_parser_object.get('API', 'huggingface_temperature', fallback='0.7')
        huggingface_top_p = config_parser_object.get('API', 'huggingface_top_p', fallback='0.95')
        huggingface_min_p = config_parser_object.get('API', 'huggingface_min_p', fallback='0.05')
        huggingface_max_tokens = config_parser_object.get('API', 'huggingface_max_tokens', fallback='4096')
        huggingface_api_timeout = config_parser_object.get('API', 'huggingface_api_timeout', fallback='90')
        huggingface_api_retries = config_parser_object.get('API', 'huggingface_api_retry', fallback='3')
        huggingface_api_retry_delay = config_parser_object.get('API', 'huggingface_api_retry_delay', fallback='5')

        # Mistral
        mistral_model = config_parser_object.get('API', 'mistral_model', fallback='mistral-large-latest')
        mistral_streaming = config_parser_object.get('API', 'mistral_streaming', fallback='False')
        mistral_temperature = config_parser_object.get('API', 'mistral_temperature', fallback='0.7')
        mistral_top_p = config_parser_object.get('API', 'mistral_top_p', fallback='0.95')
        mistral_max_tokens = config_parser_object.get('API', 'mistral_max_tokens', fallback='4096')
        mistral_api_timeout = config_parser_object.get('API', 'mistral_api_timeout', fallback='90')
        mistral_api_retries = config_parser_object.get('API', 'mistral_api_retry', fallback='3')
        mistral_api_retry_delay = config_parser_object.get('API', 'mistral_api_retry_delay', fallback='5')

        # OpenAI
        openai_model = config_parser_object.get('API', 'openai_model', fallback='gpt-4o')
        openai_streaming = config_parser_object.get('API', 'openai_streaming', fallback='False')
        openai_temperature = config_parser_object.get('API', 'openai_temperature', fallback='0.7')
        openai_top_p = config_parser_object.get('API', 'openai_top_p', fallback='0.95')
        openai_max_tokens = config_parser_object.get('API', 'openai_max_tokens', fallback='4096')
        openai_api_timeout = config_parser_object.get('API', 'openai_api_timeout', fallback='90')
        openai_api_retries = config_parser_object.get('API', 'openai_api_retry', fallback='3')
        openai_api_retry_delay = config_parser_object.get('API', 'openai_api_retry_delay', fallback='5')

        # OpenRouter
        openrouter_model = config_parser_object.get('API', 'openrouter_model', fallback='microsoft/wizardlm-2-8x22b')
        openrouter_streaming = config_parser_object.get('API', 'openrouter_streaming', fallback='False')
        openrouter_temperature = config_parser_object.get('API', 'openrouter_temperature', fallback='0.7')
        openrouter_top_p = config_parser_object.get('API', 'openrouter_top_p', fallback='0.95')
        openrouter_min_p = config_parser_object.get('API', 'openrouter_min_p', fallback='0.05')
        openrouter_top_k = config_parser_object.get('API', 'openrouter_top_k', fallback='100')
        openrouter_max_tokens = config_parser_object.get('API', 'openrouter_max_tokens', fallback='4096')
        openrouter_api_timeout = config_parser_object.get('API', 'openrouter_api_timeout', fallback='90')
        openrouter_api_retries = config_parser_object.get('API', 'openrouter_api_retry', fallback='3')
        openrouter_api_retry_delay = config_parser_object.get('API', 'openrouter_api_retry_delay', fallback='5')

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
        kobold_api_ip = config_parser_object.get('Local-API', 'kobold_api_IP', fallback='http://127.0.0.1:5000/api/v1/generate')
        kobold_openai_api_IP = config_parser_object.get('Local-API', 'kobold_openai_api_IP', fallback='http://127.0.0.1:5001/v1/chat/completions')
        kobold_api_key = config_parser_object.get('Local-API', 'kobold_api_key', fallback='')
        kobold_streaming = config_parser_object.get('Local-API', 'kobold_streaming', fallback='False')
        kobold_temperature = config_parser_object.get('Local-API', 'kobold_temperature', fallback='0.7')
        kobold_top_p = config_parser_object.get('Local-API', 'kobold_top_p', fallback='0.95')
        kobold_top_k = config_parser_object.get('Local-API', 'kobold_top_k', fallback='100')
        kobold_max_tokens = config_parser_object.get('Local-API', 'kobold_max_tokens', fallback='4096')
        kobold_api_timeout = config_parser_object.get('Local-API', 'kobold_api_timeout', fallback='90')
        kobold_api_retries = config_parser_object.get('Local-API', 'kobold_api_retry', fallback='3')
        kobold_api_retry_delay = config_parser_object.get('Local-API', 'kobold_api_retry_delay', fallback='5')

        llama_api_IP = config_parser_object.get('Local-API', 'llama_api_IP', fallback='http://127.0.0.1:8080/v1/chat/completions')
        llama_api_key = config_parser_object.get('Local-API', 'llama_api_key', fallback='')
        llama_streaming = config_parser_object.get('Local-API', 'llama_streaming', fallback='False')
        llama_temperature = config_parser_object.get('Local-API', 'llama_temperature', fallback='0.7')
        llama_top_p = config_parser_object.get('Local-API', 'llama_top_p', fallback='0.95')
        llama_min_p = config_parser_object.get('Local-API', 'llama_min_p', fallback='0.05')
        llama_top_k = config_parser_object.get('Local-API', 'llama_top_k', fallback='100')
        llama_max_tokens = config_parser_object.get('Local-API', 'llama_max_tokens', fallback='4096')
        llama_api_timeout = config_parser_object.get('Local-API', 'llama_api_timeout', fallback='90')
        llama_api_retries = config_parser_object.get('Local-API', 'llama_api_retry', fallback='3')
        llama_api_retry_delay = config_parser_object.get('Local-API', 'llama_api_retry_delay', fallback='5')

        ooba_api_IP = config_parser_object.get('Local-API', 'ooba_api_IP', fallback='http://127.0.0.1:5000/v1/chat/completions')
        ooba_api_key = config_parser_object.get('Local-API', 'ooba_api_key', fallback='')
        ooba_streaming = config_parser_object.get('Local-API', 'ooba_streaming', fallback='False')
        ooba_temperature = config_parser_object.get('Local-API', 'ooba_temperature', fallback='0.7')
        ooba_top_p = config_parser_object.get('Local-API', 'ooba_top_p', fallback='0.95')
        ooba_min_p = config_parser_object.get('Local-API', 'ooba_min_p', fallback='0.05')
        ooba_top_k = config_parser_object.get('Local-API', 'ooba_top_k', fallback='100')
        ooba_max_tokens = config_parser_object.get('Local-API', 'ooba_max_tokens', fallback='4096')
        ooba_api_timeout = config_parser_object.get('Local-API', 'ooba_api_timeout', fallback='90')
        ooba_api_retries = config_parser_object.get('Local-API', 'ooba_api_retry', fallback='3')
        ooba_api_retry_delay = config_parser_object.get('Local-API', 'ooba_api_retry_delay', fallback='5')

        tabby_api_IP = config_parser_object.get('Local-API', 'tabby_api_IP', fallback='http://127.0.0.1:5000/api/v1/generate')
        tabby_api_key = config_parser_object.get('Local-API', 'tabby_api_key', fallback=None)
        tabby_model = config_parser_object.get('models', 'tabby_model', fallback=None)
        tabby_streaming = config_parser_object.get('Local-API', 'tabby_streaming', fallback='False')
        tabby_temperature = config_parser_object.get('Local-API', 'tabby_temperature', fallback='0.7')
        tabby_top_p = config_parser_object.get('Local-API', 'tabby_top_p', fallback='0.95')
        tabby_top_k = config_parser_object.get('Local-API', 'tabby_top_k', fallback='100')
        tabby_min_p = config_parser_object.get('Local-API', 'tabby_min_p', fallback='0.05')
        tabby_max_tokens = config_parser_object.get('Local-API', 'tabby_max_tokens', fallback='4096')
        tabby_api_timeout = config_parser_object.get('Local-API', 'tabby_api_timeout', fallback='90')
        tabby_api_retries = config_parser_object.get('Local-API', 'tabby_api_retry', fallback='3')
        tabby_api_retry_delay = config_parser_object.get('Local-API', 'tabby_api_retry_delay', fallback='5')

        vllm_api_url = config_parser_object.get('Local-API', 'vllm_api_IP', fallback='http://127.0.0.1:500/api/v1/chat/completions')
        vllm_api_key = config_parser_object.get('Local-API', 'vllm_api_key', fallback=None)
        vllm_model = config_parser_object.get('Local-API', 'vllm_model', fallback=None)
        vllm_streaming = config_parser_object.get('Local-API', 'vllm_streaming', fallback='False')
        vllm_temperature = config_parser_object.get('Local-API', 'vllm_temperature', fallback='0.7')
        vllm_top_p = config_parser_object.get('Local-API', 'vllm_top_p', fallback='0.95')
        vllm_top_k = config_parser_object.get('Local-API', 'vllm_top_k', fallback='100')
        vllm_min_p = config_parser_object.get('Local-API', 'vllm_min_p', fallback='0.05')
        vllm_max_tokens = config_parser_object.get('Local-API', 'vllm_max_tokens', fallback='4096')
        vllm_api_timeout = config_parser_object.get('Local-API', 'vllm_api_timeout', fallback='90')
        vllm_api_retries = config_parser_object.get('Local-API', 'vllm_api_retry', fallback='3')
        vllm_api_retry_delay = config_parser_object.get('Local-API', 'vllm_api_retry_delay', fallback='5')

        ollama_api_url = config_parser_object.get('Local-API', 'ollama_api_IP', fallback='http://127.0.0.1:11434/api/generate')
        ollama_api_key = config_parser_object.get('Local-API', 'ollama_api_key', fallback=None)
        ollama_model = config_parser_object.get('Local-API', 'ollama_model', fallback=None)
        ollama_streaming = config_parser_object.get('Local-API', 'ollama_streaming', fallback='False')
        ollama_temperature = config_parser_object.get('Local-API', 'ollama_temperature', fallback='0.7')
        ollama_top_p = config_parser_object.get('Local-API', 'ollama_top_p', fallback='0.95')
        ollama_max_tokens = config_parser_object.get('Local-API', 'ollama_max_tokens', fallback='4096')
        ollama_api_timeout = config_parser_object.get('Local-API', 'ollama_api_timeout', fallback='90')
        ollama_api_retries = config_parser_object.get('Local-API', 'ollama_api_retry', fallback='3')
        ollama_api_retry_delay = config_parser_object.get('Local-API', 'ollama_api_retry_delay', fallback='5')

        aphrodite_api_url = config_parser_object.get('Local-API', 'aphrodite_api_IP', fallback='http://127.0.0.1:8080/v1/chat/completions')
        aphrodite_api_key = config_parser_object.get('Local-API', 'aphrodite_api_key', fallback='')
        aphrodite_model = config_parser_object.get('Local-API', 'aphrodite_model', fallback='')
        aphrodite_max_tokens = config_parser_object.get('Local-API', 'aphrodite_max_tokens', fallback='4096')
        aphrodite_streaming = config_parser_object.get('Local-API', 'aphrodite_streaming', fallback='False')
        aphrodite_api_timeout = config_parser_object.get('Local-API', 'llama_api_timeout', fallback='90')
        aphrodite_api_retries = config_parser_object.get('Local-API', 'aphrodite_api_retry', fallback='3')
        aphrodite_api_retry_delay = config_parser_object.get('Local-API', 'aphrodite_api_retry_delay', fallback='5')

        custom_openai_api_key = config_parser_object.get('API', 'custom_openai_api_key', fallback=None)
        custom_openai_api_ip = config_parser_object.get('API', 'custom_openai_api_ip', fallback=None)
        custom_openai_api_model = config_parser_object.get('API', 'custom_openai_api_model', fallback=None)
        custom_openai_api_streaming = config_parser_object.get('API', 'custom_openai_api_streaming', fallback='False')
        custom_openai_api_temperature = config_parser_object.get('API', 'custom_openai_api_temperature', fallback='0.7')
        custom_openai_api_top_p = config_parser_object.get('API', 'custom_openai_api_top_p', fallback='0.95')
        custom_openai_api_min_p = config_parser_object.get('API', 'custom_openai_api_top_k', fallback='100')
        custom_openai_api_max_tokens = config_parser_object.get('API', 'custom_openai_api_max_tokens', fallback='4096')
        custom_openai_api_timeout = config_parser_object.get('API', 'custom_openai_api_timeout', fallback='90')
        custom_openai_api_retries = config_parser_object.get('API', 'custom_openai_api_retry', fallback='3')
        custom_openai_api_retry_delay = config_parser_object.get('API', 'custom_openai_api_retry_delay', fallback='5')

        # 2nd Custom OpenAI API
        custom_openai2_api_key = config_parser_object.get('API', 'custom_openai2_api_key', fallback=None)
        custom_openai2_api_ip = config_parser_object.get('API', 'custom_openai2_api_ip', fallback=None)
        custom_openai2_api_model = config_parser_object.get('API', 'custom_openai2_api_model', fallback=None)
        custom_openai2_api_streaming = config_parser_object.get('API', 'custom_openai2_api_streaming', fallback='False')
        custom_openai2_api_temperature = config_parser_object.get('API', 'custom_openai2_api_temperature', fallback='0.7')
        custom_openai2_api_top_p = config_parser_object.get('API', 'custom_openai_api2_top_p', fallback='0.95')
        custom_openai2_api_min_p = config_parser_object.get('API', 'custom_openai_api2_top_k', fallback='100')
        custom_openai2_api_max_tokens = config_parser_object.get('API', 'custom_openai2_api_max_tokens', fallback='4096')
        custom_openai2_api_timeout = config_parser_object.get('API', 'custom_openai2_api_timeout', fallback='90')
        custom_openai2_api_retries = config_parser_object.get('API', 'custom_openai2_api_retry', fallback='3')
        custom_openai2_api_retry_delay = config_parser_object.get('API', 'custom_openai2_api_retry_delay', fallback='5')

        # Logging Checks for Local API IP loads
        # logging.debug(f"Loaded Kobold API IP: {kobold_api_ip}")
        # logging.debug(f"Loaded Llama API IP: {llama_api_IP}")
        # logging.debug(f"Loaded Ooba API IP: {ooba_api_IP}")
        # logging.debug(f"Loaded Tabby API IP: {tabby_api_IP}")
        # logging.debug(f"Loaded VLLM API URL: {vllm_api_url}")

        # Retrieve default API choices from the configuration file
        default_api = config_parser_object.get('API', 'default_api', fallback='openai')

        # Retrieve LLM API settings from the configuration file
        local_api_retries = config_parser_object.get('Local-API', 'Settings', fallback='3')
        local_api_retry_delay = config_parser_object.get('Local-API', 'local_api_retry_delay', fallback='5')

        # Retrieve output paths from the configuration file
        output_path = config_parser_object.get('Paths', 'output_path', fallback='results')
        logger.trace(f"Output path set to: {output_path}")

        # Save video transcripts
        save_video_transcripts = config_parser_object.get('Paths', 'save_video_transcripts', fallback='True')

        # Retrieve logging settings from the configuration file
        log_level = config_parser_object.get('Logging', 'log_level', fallback='INFO')
        log_file = config_parser_object.get('Logging', 'log_file', fallback='./Logs/tldw_logs.json')
        log_metrics_file = config_parser_object.get('Logging', 'log_metrics_file', fallback='./Logs/tldw_metrics_logs.json')

        # Retrieve processing choice from the configuration file
        processing_choice = config_parser_object.get('Processing', 'processing_choice', fallback='cpu')
        logger.trace(f"Processing choice set to: {processing_choice}")

        # [Chunking]
        # # Chunking Defaults
        # #
        # # Default Chunking Options for each media type
        chunking_method = config_parser_object.get('Chunking', 'chunking_method', fallback='words')
        chunk_max_size = config_parser_object.get('Chunking', 'chunk_max_size', fallback='400')
        chunk_overlap = config_parser_object.get('Chunking', 'chunk_overlap', fallback='200')
        adaptive_chunking = config_parser_object.get('Chunking', 'adaptive_chunking', fallback='False')
        chunking_multi_level = config_parser_object.get('Chunking', 'chunking_multi_level', fallback='False')
        chunk_language = config_parser_object.get('Chunking', 'chunk_language', fallback='en')
        #
        # Article Chunking
        article_chunking_method = config_parser_object.get('Chunking', 'article_chunking_method', fallback='words')
        article_chunk_max_size = config_parser_object.get('Chunking', 'article_chunk_max_size', fallback='400')
        article_chunk_overlap = config_parser_object.get('Chunking', 'article_chunk_overlap', fallback='200')
        article_adaptive_chunking = config_parser_object.get('Chunking', 'article_adaptive_chunking', fallback='False')
        article_chunking_multi_level = config_parser_object.get('Chunking', 'article_chunking_multi_level', fallback='False')
        article_language = config_parser_object.get('Chunking', 'article_language', fallback='english')
        #
        # Audio file Chunking
        audio_chunking_method = config_parser_object.get('Chunking', 'audio_chunking_method', fallback='words')
        audio_chunk_max_size = config_parser_object.get('Chunking', 'audio_chunk_max_size', fallback='400')
        audio_chunk_overlap = config_parser_object.get('Chunking', 'audio_chunk_overlap', fallback='200')
        audio_adaptive_chunking = config_parser_object.get('Chunking', 'audio_adaptive_chunking', fallback='False')
        audio_chunking_multi_level = config_parser_object.get('Chunking', 'audio_chunking_multi_level', fallback='False')
        audio_language = config_parser_object.get('Chunking', 'audio_language', fallback='english')
        #
        # Book Chunking
        book_chunking_method = config_parser_object.get('Chunking', 'book_chunking_method', fallback='words')
        book_chunk_max_size = config_parser_object.get('Chunking', 'book_chunk_max_size', fallback='400')
        book_chunk_overlap = config_parser_object.get('Chunking', 'book_chunk_overlap', fallback='200')
        book_adaptive_chunking = config_parser_object.get('Chunking', 'book_adaptive_chunking', fallback='False')
        book_chunking_multi_level = config_parser_object.get('Chunking', 'book_chunking_multi_level', fallback='False')
        book_language = config_parser_object.get('Chunking', 'book_language', fallback='english')
        #
        # Document Chunking
        document_chunking_method = config_parser_object.get('Chunking', 'document_chunking_method', fallback='words')
        document_chunk_max_size = config_parser_object.get('Chunking', 'document_chunk_max_size', fallback='400')
        document_chunk_overlap = config_parser_object.get('Chunking', 'document_chunk_overlap', fallback='200')
        document_adaptive_chunking = config_parser_object.get('Chunking', 'document_adaptive_chunking', fallback='False')
        document_chunking_multi_level = config_parser_object.get('Chunking', 'document_chunking_multi_level', fallback='False')
        document_language = config_parser_object.get('Chunking', 'document_language', fallback='english')
        #
        # Mediawiki Article Chunking
        mediawiki_article_chunking_method = config_parser_object.get('Chunking', 'mediawiki_article_chunking_method', fallback='words')
        mediawiki_article_chunk_max_size = config_parser_object.get('Chunking', 'mediawiki_article_chunk_max_size', fallback='400')
        mediawiki_article_chunk_overlap = config_parser_object.get('Chunking', 'mediawiki_article_chunk_overlap', fallback='200')
        mediawiki_article_adaptive_chunking = config_parser_object.get('Chunking', 'mediawiki_article_adaptive_chunking', fallback='False')
        mediawiki_article_chunking_multi_level = config_parser_object.get('Chunking', 'mediawiki_article_chunking_multi_level', fallback='False')
        mediawiki_article_language = config_parser_object.get('Chunking', 'mediawiki_article_language', fallback='english')
        #
        # Mediawiki Dump Chunking
        mediawiki_dump_chunking_method = config_parser_object.get('Chunking', 'mediawiki_dump_chunking_method', fallback='words')
        mediawiki_dump_chunk_max_size = config_parser_object.get('Chunking', 'mediawiki_dump_chunk_max_size', fallback='400')
        mediawiki_dump_chunk_overlap = config_parser_object.get('Chunking', 'mediawiki_dump_chunk_overlap', fallback='200')
        mediawiki_dump_adaptive_chunking = config_parser_object.get('Chunking', 'mediawiki_dump_adaptive_chunking', fallback='False')
        mediawiki_dump_chunking_multi_level = config_parser_object.get('Chunking', 'mediawiki_dump_chunking_multi_level', fallback='False')
        mediawiki_dump_language = config_parser_object.get('Chunking', 'mediawiki_dump_language', fallback='english')
        #
        # Obsidian Note Chunking
        obsidian_note_chunking_method = config_parser_object.get('Chunking', 'obsidian_note_chunking_method', fallback='words')
        obsidian_note_chunk_max_size = config_parser_object.get('Chunking', 'obsidian_note_chunk_max_size', fallback='400')
        obsidian_note_chunk_overlap = config_parser_object.get('Chunking', 'obsidian_note_chunk_overlap', fallback='200')
        obsidian_note_adaptive_chunking = config_parser_object.get('Chunking', 'obsidian_note_adaptive_chunking', fallback='False')
        obsidian_note_chunking_multi_level = config_parser_object.get('Chunking', 'obsidian_note_chunking_multi_level', fallback='False')
        obsidian_note_language = config_parser_object.get('Chunking', 'obsidian_note_language', fallback='english')
        #
        # Podcast Chunking
        podcast_chunking_method = config_parser_object.get('Chunking', 'podcast_chunking_method', fallback='words')
        podcast_chunk_max_size = config_parser_object.get('Chunking', 'podcast_chunk_max_size', fallback='400')
        podcast_chunk_overlap = config_parser_object.get('Chunking', 'podcast_chunk_overlap', fallback='200')
        podcast_adaptive_chunking = config_parser_object.get('Chunking', 'podcast_adaptive_chunking', fallback='False')
        podcast_chunking_multi_level = config_parser_object.get('Chunking', 'podcast_chunking_multi_level', fallback='False')
        podcast_language = config_parser_object.get('Chunking', 'podcast_language', fallback='english')
        #
        # Text Chunking
        text_chunking_method = config_parser_object.get('Chunking', 'text_chunking_method', fallback='words')
        text_chunk_max_size = config_parser_object.get('Chunking', 'text_chunk_max_size', fallback='400')
        text_chunk_overlap = config_parser_object.get('Chunking', 'text_chunk_overlap', fallback='200')
        text_adaptive_chunking = config_parser_object.get('Chunking', 'text_adaptive_chunking', fallback='False')
        text_chunking_multi_level = config_parser_object.get('Chunking', 'text_chunking_multi_level', fallback='False')
        text_language = config_parser_object.get('Chunking', 'text_language', fallback='english')
        #
        # Video Transcription Chunking
        video_chunking_method = config_parser_object.get('Chunking', 'video_chunking_method', fallback='words')
        video_chunk_max_size = config_parser_object.get('Chunking', 'video_chunk_max_size', fallback='400')
        video_chunk_overlap = config_parser_object.get('Chunking', 'video_chunk_overlap', fallback='200')
        video_adaptive_chunking = config_parser_object.get('Chunking', 'video_adaptive_chunking', fallback='False')
        video_chunking_multi_level = config_parser_object.get('Chunking', 'video_chunking_multi_level', fallback='False')
        video_language = config_parser_object.get('Chunking', 'video_language', fallback='english')
        #
        chunking_types = 'article', 'audio', 'book', 'document', 'mediawiki_article', 'mediawiki_dump', 'obsidian_note', 'podcast', 'text', 'video'

        # Retrieve Embedding model settings from the configuration file
        embedding_model = config_parser_object.get('Embeddings', 'embedding_model', fallback='')
        logger.trace(f"Embedding model set to: {embedding_model}")
        embedding_provider = config_parser_object.get('Embeddings', 'embedding_provider', fallback='')
        embedding_model = config_parser_object.get('Embeddings', 'embedding_model', fallback='')
        onnx_model_path = config_parser_object.get('Embeddings', 'onnx_model_path', fallback="./App_Function_Libraries/onnx_models/text-embedding-3-small.onnx")
        model_dir = config_parser_object.get('Embeddings', 'model_dir', fallback="./App_Function_Libraries/onnx_models")
        embedding_api_url = config_parser_object.get('Embeddings', 'embedding_api_url', fallback="http://localhost:8080/v1/embeddings")
        embedding_api_key = config_parser_object.get('Embeddings', 'embedding_api_key', fallback='')
        chunk_size = config_parser_object.get('Embeddings', 'chunk_size', fallback=400)
        overlap = config_parser_object.get('Embeddings', 'overlap', fallback=200)

        # Prompts - FIXME
        prompt_path = config_parser_object.get('Prompts', 'prompt_path', fallback='Databases/prompts.db')

        # Chat Dictionaries
        enable_chat_dictionaries = config_parser_object.get('Chat-Dictionaries', 'enable_chat_dictionaries', fallback='False')
        post_gen_replacement = config_parser_object.get('Chat-Dictionaries', 'post_gen_replacement', fallback='False')
        post_gen_replacement_dict = config_parser_object.get('Chat-Dictionaries', 'post_gen_replacement_dict', fallback='')
        chat_dict_chat_prompts = config_parser_object.get('Chat-Dictionaries', 'chat_dictionary_chat_prompts', fallback='')
        chat_dict_rag_prompts = config_parser_object.get('Chat-Dictionaries', 'chat_dictionary_RAG_prompts', fallback='')
        chat_dict_replacement_strategy = config_parser_object.get('Chat-Dictionaries', 'chat_dictionary_replacement_strategy', fallback='character_lore_first')
        chat_dict_max_tokens = config_parser_object.get('Chat-Dictionaries', 'chat_dictionary_max_tokens', fallback='1000')
        default_rag_prompt = config_parser_object.get('Chat-Dictionaries', 'default_rag_prompt', fallback='')

        # Auto-Save Values
        save_character_chats = config_parser_object.get('Auto-Save', 'save_character_chats', fallback='False')
        save_rag_chats = config_parser_object.get('Auto-Save', 'save_rag_chats', fallback='False')

        # Local API Timeout
        local_api_timeout = config_parser_object.get('Local-API', 'local_api_timeout', fallback='90')

        # STT Settings
        default_stt_provider = config_parser_object.get('STT-Settings', 'default_stt_provider', fallback='faster_whisper')

        # TTS Settings
        # FIXME
        local_tts_device = config_parser_object.get('TTS-Settings', 'local_tts_device', fallback='cpu')
        default_tts_provider = config_parser_object.get('TTS-Settings', 'default_tts_provider', fallback='openai')
        tts_voice = config_parser_object.get('TTS-Settings', 'default_tts_voice', fallback='shimmer')
        # Open AI TTS
        default_openai_tts_model = config_parser_object.get('TTS-Settings', 'default_openai_tts_model', fallback='tts-1-hd')
        default_openai_tts_voice = config_parser_object.get('TTS-Settings', 'default_openai_tts_voice', fallback='shimmer')
        default_openai_tts_speed = config_parser_object.get('TTS-Settings', 'default_openai_tts_speed', fallback='1')
        default_openai_tts_output_format = config_parser_object.get('TTS-Settings', 'default_openai_tts_output_format', fallback='mp3')
        default_openai_tts_streaming = config_parser_object.get('TTS-Settings', 'default_openai_tts_streaming', fallback='False')
        # Google TTS
        # FIXME - FIX THESE DEFAULTS
        default_google_tts_model = config_parser_object.get('TTS-Settings', 'default_google_tts_model', fallback='en')
        default_google_tts_voice = config_parser_object.get('TTS-Settings', 'default_google_tts_voice', fallback='en')
        default_google_tts_speed = config_parser_object.get('TTS-Settings', 'default_google_tts_speed', fallback='1')
        # ElevenLabs TTS
        default_eleven_tts_model = config_parser_object.get('TTS-Settings', 'default_eleven_tts_model', fallback='FIXME')
        default_eleven_tts_voice = config_parser_object.get('TTS-Settings', 'default_eleven_tts_voice', fallback='FIXME')
        default_eleven_tts_language_code = config_parser_object.get('TTS-Settings', 'default_eleven_tts_language_code', fallback='FIXME')
        default_eleven_tts_voice_stability = config_parser_object.get('TTS-Settings', 'default_eleven_tts_voice_stability', fallback='FIXME')
        default_eleven_tts_voice_similiarity_boost = config_parser_object.get('TTS-Settings', 'default_eleven_tts_voice_similiarity_boost', fallback='FIXME')
        default_eleven_tts_voice_style = config_parser_object.get('TTS-Settings', 'default_eleven_tts_voice_style', fallback='FIXME')
        default_eleven_tts_voice_use_speaker_boost = config_parser_object.get('TTS-Settings', 'default_eleven_tts_voice_use_speaker_boost', fallback='FIXME')
        default_eleven_tts_output_format = config_parser_object.get('TTS-Settings', 'default_eleven_tts_output_format',
                                                      fallback='mp3_44100_192')
        # AllTalk TTS
        alltalk_api_ip = config_parser_object.get('TTS-Settings', 'alltalk_api_ip', fallback='http://127.0.0.1:7851/v1/audio/speech')
        default_alltalk_tts_model = config_parser_object.get('TTS-Settings', 'default_alltalk_tts_model', fallback='alltalk_model')
        default_alltalk_tts_voice = config_parser_object.get('TTS-Settings', 'default_alltalk_tts_voice', fallback='alloy')
        default_alltalk_tts_speed = config_parser_object.get('TTS-Settings', 'default_alltalk_tts_speed', fallback=1.0)
        default_alltalk_tts_output_format = config_parser_object.get('TTS-Settings', 'default_alltalk_tts_output_format', fallback='mp3')

        # Kokoro TTS
        kokoro_model_path = config_parser_object.get('TTS-Settings', 'kokoro_model_path', fallback='Databases/kokoro_models')
        default_kokoro_tts_model = config_parser_object.get('TTS-Settings', 'default_kokoro_tts_model', fallback='pht')
        default_kokoro_tts_voice = config_parser_object.get('TTS-Settings', 'default_kokoro_tts_voice', fallback='sky')
        default_kokoro_tts_speed = config_parser_object.get('TTS-Settings', 'default_kokoro_tts_speed', fallback=1.0)
        default_kokoro_tts_output_format = config_parser_object.get('TTS-Settings', 'default_kokoro_tts_output_format', fallback='wav')


        # Self-hosted OpenAI API TTS
        default_openai_api_tts_model = config_parser_object.get('TTS-Settings', 'default_openai_api_tts_model', fallback='tts-1-hd')
        default_openai_api_tts_voice = config_parser_object.get('TTS-Settings', 'default_openai_api_tts_voice', fallback='shimmer')
        default_openai_api_tts_speed = config_parser_object.get('TTS-Settings', 'default_openai_api_tts_speed', fallback='1')
        default_openai_api_tts_output_format = config_parser_object.get('TTS-Settings', 'default_openai_tts_api_output_format', fallback='mp3')
        default_openai_api_tts_streaming = config_parser_object.get('TTS-Settings', 'default_openai_tts_streaming', fallback='False')


        # Search Engines
        search_provider_default = config_parser_object.get('Search-Engines', 'search_provider_default', fallback='google')
        search_language_query = config_parser_object.get('Search-Engines', 'search_language_query', fallback='en')
        search_language_results = config_parser_object.get('Search-Engines', 'search_language_results', fallback='en')
        search_language_analysis = config_parser_object.get('Search-Engines', 'search_language_analysis', fallback='en')
        search_default_max_queries = 10
        search_enable_subquery = config_parser_object.get('Search-Engines', 'search_enable_subquery', fallback='True')
        search_enable_subquery_count_max = config_parser_object.get('Search-Engines', 'search_enable_subquery_count_max', fallback=5)
        search_result_rerank = config_parser_object.get('Search-Engines', 'search_result_rerank', fallback='True')
        search_result_max = config_parser_object.get('Search-Engines', 'search_result_max', fallback=10)
        search_result_max_per_query = config_parser_object.get('Search-Engines', 'search_result_max_per_query', fallback=10)
        search_result_blacklist = config_parser_object.get('Search-Engines', 'search_result_blacklist', fallback='')
        search_result_display_type = config_parser_object.get('Search-Engines', 'search_result_display_type', fallback='list')
        search_result_display_metadata = config_parser_object.get('Search-Engines', 'search_result_display_metadata', fallback='False')
        search_result_save_to_db = config_parser_object.get('Search-Engines', 'search_result_save_to_db', fallback='True')
        search_result_analysis_tone = config_parser_object.get('Search-Engines', 'search_result_analysis_tone', fallback='')
        relevance_analysis_llm = config_parser_object.get('Search-Engines', 'relevance_analysis_llm', fallback='False')
        final_answer_llm = config_parser_object.get('Search-Engines', 'final_answer_llm', fallback='False')
        # Search Engine Specifics
        baidu_search_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_baidu', fallback='')
        # Bing Search Settings
        bing_search_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_bing', fallback='')
        bing_country_code = config_parser_object.get('Search-Engines', 'search_engine_country_code_bing', fallback='us')
        bing_search_api_url = config_parser_object.get('Search-Engines', 'search_engine_api_url_bing', fallback='')
        # Brave Search Settings
        brave_search_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_brave_regular', fallback='')
        brave_search_ai_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_brave_ai', fallback='')
        brave_country_code = config_parser_object.get('Search-Engines', 'search_engine_country_code_brave', fallback='us')
        # DuckDuckGo Search Settings
        duckduckgo_search_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_duckduckgo', fallback='')
        # Google Search Settings
        google_search_api_url = config_parser_object.get('Search-Engines', 'search_engine_api_url_google', fallback='')
        google_search_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_google', fallback='')
        google_search_engine_id = config_parser_object.get('Search-Engines', 'search_engine_id_google', fallback='')
        google_simp_trad_chinese = config_parser_object.get('Search-Engines', 'enable_traditional_chinese', fallback='0')
        limit_google_search_to_country = config_parser_object.get('Search-Engines', 'limit_google_search_to_country', fallback='0')
        google_search_country = config_parser_object.get('Search-Engines', 'google_search_country', fallback='us')
        google_search_country_code = config_parser_object.get('Search-Engines', 'google_search_country_code', fallback='us')
        google_filter_setting = config_parser_object.get('Search-Engines', 'google_filter_setting', fallback='1')
        google_user_geolocation = config_parser_object.get('Search-Engines', 'google_user_geolocation', fallback='')
        google_ui_language = config_parser_object.get('Search-Engines', 'google_ui_language', fallback='en')
        google_limit_search_results_to_language = config_parser_object.get('Search-Engines', 'google_limit_search_results_to_language', fallback='')
        google_default_search_results = config_parser_object.get('Search-Engines', 'google_default_search_results', fallback='10')
        google_safe_search = config_parser_object.get('Search-Engines', 'google_safe_search', fallback='active')
        google_enable_site_search = config_parser_object.get('Search-Engines', 'google_enable_site_search', fallback='0')
        google_site_search_include = config_parser_object.get('Search-Engines', 'google_site_search_include', fallback='')
        google_site_search_exclude = config_parser_object.get('Search-Engines', 'google_site_search_exclude', fallback='')
        google_sort_results_by = config_parser_object.get('Search-Engines', 'google_sort_results_by', fallback='relevance')
        # Kagi Search Settings
        kagi_search_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_kagi', fallback='')
        # Searx Search Settings
        search_engine_searx_api = config_parser_object.get('Search-Engines', 'search_engine_searx_api', fallback='')
        # Tavily Search Settings
        tavily_search_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_tavily', fallback='')
        # Yandex Search Settings
        yandex_search_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_yandex', fallback='')
        yandex_search_engine_id = config_parser_object.get('Search-Engines', 'search_engine_id_yandex', fallback='')

        # Prompts
        sub_question_generation_prompt = config_parser_object.get('Prompts', 'sub_question_generation_prompt', fallback='')
        search_result_relevance_eval_prompt = config_parser_object.get('Prompts', 'search_result_relevance_eval_prompt', fallback='')
        analyze_search_results_prompt = config_parser_object.get('Prompts', 'analyze_search_results_prompt', fallback='')

        # Web Scraper settings
        web_scraper_api_key = config_parser_object.get('Web-Scraper', 'web_scraper_api_key', fallback='')
        web_scraper_api_url = config_parser_object.get('Web-Scraper', 'web_scraper_api_url', fallback='')
        web_scraper_api_timeout = config_parser_object.get('Web-Scraper', 'web_scraper_api_timeout', fallback='90')
        web_scraper_api_retries = config_parser_object.get('Web-Scraper', 'web_scraper_api_retries', fallback='3')
        web_scraper_api_retry_delay = config_parser_object.get('Web-Scraper', 'web_scraper_api_retry_delay', fallback='5')
        web_scraper_retry_count = config_parser_object.get('Web-Scraper', 'web_scraper_retry_count', fallback='3')
        web_scraper_retry_timeout = config_parser_object.get('Web-Scraper', 'web_scraper_retry_timeout', fallback='5')
        web_scraper_stealth_playwright = config_parser_object.get('Web-Scraper', 'web_scraper_stealth_playwright', fallback='False')

        return_dict = {
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
                'huggingface_use_router_url_format': huggingface_use_router_url_format,
                'huggingface_router_base_url': huggingface_router_base_url,
                'api_base_url': huggingface_api_base_url,
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
        return return_dict
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        return None


# Global scope in config.py
try:
    loaded_config_data = load_and_log_configs()
    if loaded_config_data is None:  # Add a check here
        logger.critical("Failed to load configuration data at module import. `loaded_config_data` is None.")
        default_api_endpoint = "openai"  # Fallback
    else:
        default_api_endpoint = loaded_config_data.get('default_api', 'openai')  # Use .get() for safety
        logger.info(f"Default API Endpoint (from config.py global scope): {default_api_endpoint}")
except Exception as e:  # Should be less likely to hit this outer if inner one is robust
    logger.error(f"Critical error setting default_api_endpoint in config.py global scope: {str(e)}", exc_info=True)
    default_api_endpoint = "openai"  # Fallback


# --- Global Settings Object ---
# Load the settings when the module is imported
settings = load_settings()


# --- Optional: Export individual variables if needed for backward compatibility (less recommended) ---
# SINGLE_USER_MODE = settings["SINGLE_USER_MODE"]
# SINGLE_USER_FIXED_ID = settings["SINGLE_USER_FIXED_ID"]
# ... etc ...
#
# End of config.py
#######################################################################################################################