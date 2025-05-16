# config.py
# Description: Configuration settings for the tldw server application.
#
# Imports
import json
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
    mapping_path = os.path.join(project_root, "configs", "openai_tts_mappings.json")
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