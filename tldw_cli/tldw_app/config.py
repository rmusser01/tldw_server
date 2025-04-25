# tldw_cli/config.py
# Description: Configuration management for the tldw_cli application.
#
# Imports
import tomllib
import logging
import os
from pathlib import Path
import toml
from typing import Dict, Any, List, Optional
#
# Third-Party Imports
## No third-party imports in this file
# Local Imports
#
#######################################################################################################################
#
# Functions:

log = logging.getLogger(__name__)

# --- Default Configuration ---
# Define defaults here for resilience if config file is missing/corrupt
DEFAULT_CONFIG = {
    "general": {"default_tab": "chat", "log_level": "INFO"},
    "logging": {
        "log_filename": "tldw_cli_app.log",
        "file_log_level": "INFO",
        "log_max_bytes": 10 * 1024 * 1024, # 10 MB
        "log_backup_count": 5
    },
    "database": {
        "path": "~/.local/share/tldw_cli/tldw_cli_data.db"
    },
    "api_endpoints": {
         # Minimal defaults, user should override in config.toml
        "Ollama": "http://localhost:11434",
    },
    "providers": {
        # Minimal example providers/models if config file fails badly
        "Ollama": ["llama3:latest"],
        "Anthropic": ["claude-3-haiku-20240307"],
    },
    "chat_defaults": {
        "provider": "Ollama", "model": "llama3:latest",
        "system_prompt": "You are a helpful assistant.",
        "temperature": 0.7, "top_p": 1.0, "min_p": 0.0, "top_k": 0,
    },
    "character_defaults": {
        "provider": "Anthropic", "model": "claude-3-haiku-20240307",
        "system_prompt": "You are a helpful character.",
        "temperature": 0.8, "top_p": 1.0, "min_p": 0.0, "top_k": 0,
    }
    # Add other top-level sections with defaults if necessary
}

# --- Configuration Loading Function ---

def get_config_path() -> Path:
    """Determines the path to the configuration file."""
    # Priority: Environment variable > Default user location
    env_path = os.environ.get("TLDW_CLI_CONFIG_PATH")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        log.debug(f"Using config path from TLDW_CLI_CONFIG_PATH: {path}")
        return path

    # Default location (you could use XDG paths here too)
    config_dir = Path.home() / ".config" / "tldw_cli"
    default_path = config_dir / "config.toml"
    log.debug(f"Using default config path: {default_path}")
    return default_path


# Store loaded config globally within this module after first load
_APP_CONFIG: Optional[Dict[str, Any]] = None

def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Loads configuration from TOML file, merges with defaults.
    Ensures the config directory exists and creates a default config if needed.
    Caches the loaded configuration.
    """
    global _APP_CONFIG
    if _APP_CONFIG is not None:
        log.debug("Returning cached config.")
        return _APP_CONFIG

    if config_path is None:
        config_path = get_config_path()

    # Deep copy defaults to avoid modifying the original DEFAULT_CONFIG dict
    config = {k: v.copy() if isinstance(v, dict) else v for k, v in DEFAULT_CONFIG.items()}

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True) # Ensure parent dir exists
        log.info(f"Attempting to load configuration from: {config_path}")

        if config_path.exists():
            try:
                with open(config_path, "rb") as f:
                    user_config = tomllib.load(f)

                # Merge user config into defaults (simple one-level deep merge)
                # You could implement a recursive merge here if needed
                for section, section_config in user_config.items():
                    if section in config and isinstance(config[section], dict) and isinstance(section_config, dict):
                        config[section].update(section_config)
                    else:
                        config[section] = section_config # Overwrite or add new section
                log.info(f"Successfully loaded and merged config from {config_path}")

            except tomllib.TOMLDecodeError as e:
                log.error(f"Error decoding TOML file {config_path}: {e}", exc_info=True)
                log.warning("Using default configuration values due to TOML error.")
            except Exception as e:
                log.error(f"Unexpected error loading config file {config_path}: {e}", exc_info=True)
                log.warning("Using default configuration values due to loading error.")
        else:
            log.warning(f"Config file not found at {config_path}. Creating default config.")
            try:
                # Try to write the default config back for the user
                with open(config_path, "w", encoding="utf-8") as f:
                    toml.dump(DEFAULT_CONFIG, f)
                log.info(f"Created default configuration file at: {config_path}")
            except ImportError:
                 log.warning("`toml` library not found. Cannot write default config file.")
                 # Create a simple placeholder
                 with open(config_path, "w", encoding="utf-8") as f:
                     f.write("# Configuration file for tldw_cli\n")
                     f.write("# Please install 'toml' (`pip install toml`) and run again,\n")
                     f.write("# or manually create the config based on documentation.\n")
            except Exception as e:
                log.error(f"Failed to create default config file at {config_path}: {e}", exc_info=True)

    except OSError as e:
        log.error(f"OS error accessing config directory or file {config_path}: {e}", exc_info=True)
        log.warning("Using default configuration values due to OS error.")
    except Exception as e:
        log.error(f"General error during config loading for {config_path}: {e}", exc_info=True)
        log.warning("Using default configuration values due to unexpected error.")

    # Cache the result
    _APP_CONFIG = config
    log.debug(f"Configuration loaded: Sections={list(_APP_CONFIG.keys())}")
    return _APP_CONFIG

# --- Convenience Access Functions ---

def get_setting(section: str, key: str, default: Any = None) -> Any:
    """Gets a specific setting, returning a default if not found."""
    config = load_config() # Ensures config is loaded
    return config.get(section, {}).get(key, default)

def get_providers_and_models() -> Dict[str, List[str]]:
    """Returns the dictionary of providers and their models from config."""
    config = load_config()
    providers = config.get("providers", {})
    # Basic validation: ensure values are lists of strings
    valid_providers = {}
    for provider, models in providers.items():
        if isinstance(models, list) and all(isinstance(m, str) for m in models):
            valid_providers[provider] = models
        else:
            log.warning(f"Invalid model list for provider '{provider}' in config. Skipping.")
    return valid_providers

def get_database_path() -> Path:
    """Gets the resolved database path from config."""
    config = load_config()
    db_path_str = config.get("database", {}).get("path", DEFAULT_CONFIG["database"]["path"])
    db_path = Path(db_path_str).expanduser().resolve()
    # Ensure the parent directory exists *when accessed*
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.error(f"Could not create database directory {db_path.parent}: {e}", exc_info=True)
        # Depending on the use case, you might raise an error here
    return db_path

def get_log_file_path() -> Path:
    """Gets the full path for the log file, placing it relative to the DB."""
    db_dir = get_database_path().parent
    log_filename = get_setting("logging", "log_filename", DEFAULT_CONFIG["logging"]["log_filename"])
    return db_dir / log_filename


# Example usage (optional, for testing this module directly)
if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.DEBUG)
    print("--- Testing Config Loading ---")
    # Test with a temporary path
    temp_dir = Path("./temp_config_test")
    temp_dir.mkdir(exist_ok=True)
    temp_config_path = temp_dir / "test_config.toml"
    # Clean up previous test file if exists
    if temp_config_path.exists(): temp_config_path.unlink()

    # Test 1: File doesn't exist, create default
    print("\nTest 1: No config file")
    cfg1 = load_config(temp_config_path)
    print(f"Loaded config sections: {list(cfg1.keys())}")
    print(f"Default tab: {get_setting('general', 'default_tab')}")
    print(f"DB path: {get_database_path()}")
    assert temp_config_path.exists()

    # Test 2: File exists, load it
    print("\nTest 2: Existing config file")
    # Modify the created file
    with open(temp_config_path, "w") as f:
        f.write("[general]\n")
        f.write("default_tab = \"logs\"\n")
        f.write("[new_section]\n")
        f.write("my_value = 123\n")

    _APP_CONFIG = None # Clear cache
    cfg2 = load_config(temp_config_path)
    print(f"Loaded config sections: {list(cfg2.keys())}")
    print(f"Default tab (overridden): {get_setting('general', 'default_tab')}")
    print(f"New section value: {get_setting('new_section', 'my_value')}")
    assert get_setting('general', 'default_tab') == "logs"
    assert get_setting('new_section', 'my_value') == 123

    # Clean up test file and directory
    # temp_config_path.unlink()
    # temp_dir.rmdir()
    print("\n--- Config Loading Test Complete ---")

#
# End of tldw_cli/config.py
#######################################################################################################################
