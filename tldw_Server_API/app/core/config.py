# config.py
# Description: Configuration settings for the tldw server application.
#
# Imports
import os
from pathlib import Path


#
# 3rd-party Libraries
#
# Local Imports
#
########################################################################################################################
#
# Functions:

# config.py
# Description: Configuration settings for the tldw server application.
#
# Imports
import os
from pathlib import Path

# Imports from this project
# Note: It's generally better practice to keep config simple and avoid
# complex imports here if possible, but logging is often an exception.
# from tldw_Server_API.app.core.Utils.Utils import logging # Example if needed

########################################################################################################################

# --- Constants ---
# Client ID used by the Server API itself when writing to sync logs
SERVER_CLIENT_ID = "SERVER_API_V1"

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
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

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