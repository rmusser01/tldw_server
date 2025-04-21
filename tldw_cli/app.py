# tldw_cli - Textual CLI for LLMs
# Description: This file contains the main application logic for the tldw_cli, a Textual-based CLI for interacting with various LLM APIs.
#
# Imports
import asyncio
import logging  # Standard logging library
import logging.handlers  # For handlers
import tomllib # Use built-in tomllib for Python 3.11+
from pathlib import Path
import traceback
import os
from typing import Union, Generator, Any # For type hinting
#
# 3rd-Party Libraries
import chardet # For encoding detection
#import requests
#import tqdm
# --- Textual Imports ---
from textual.app import App, ComposeResult, ScreenStackError  # Removed RenderResult as it wasn't used
from textual.widgets import (
    Static, Button, Input, Header, Footer, RichLog, TextArea, Select
)
from textual.containers import Horizontal, Container, Vertical, VerticalScroll
from textual.reactive import reactive
# Removed Widget import as it wasn't used directly
# Removed Message import as it wasn't used directly
from textual.worker import Worker, WorkerState
from textual.binding import Binding
#
# --- Local API library Imports ---
# Adjust the path based on your project structure
try:
    # Assuming libs is a subdirectory
    from .Libs.LLM_API_Calls import (
        chat_with_openai, chat_with_anthropic, chat_with_cohere,
        chat_with_groq, chat_with_openrouter, chat_with_huggingface,
        chat_with_deepseek, chat_with_mistral, chat_with_google
    )
    from .Libs.LLM_API_Calls_Local import (
        # chat_with_local_llm, # Example if this exists
        chat_with_llama, chat_with_kobold,
        chat_with_oobabooga, chat_with_vllm, chat_with_tabbyapi,
        chat_with_aphrodite, chat_with_ollama,
        chat_with_custom_openai, chat_with_custom_openai_2 # Assuming these exist
    )
    API_IMPORTS_SUCCESSFUL = True
except ImportError as e:
    # Use logging right away if possible
    try:
        logging.error(f"Failed to import API libraries during initial load: {e}", exc_info=True)
    except Exception: # Fallback if logging itself fails early
        print(f"ERROR: Failed to import API libraries during initial load: {e}")

    # Set ALL potentially imported functions to None to prevent NameErrors later
    chat_with_openai = chat_with_anthropic = chat_with_cohere = chat_with_groq = \
    chat_with_openrouter = chat_with_huggingface = chat_with_deepseek = \
    chat_with_mistral = chat_with_google = \
    chat_with_llama = chat_with_kobold = chat_with_oobabooga = chat_with_vllm = \
    chat_with_tabbyapi = chat_with_aphrodite = chat_with_ollama = \
    chat_with_custom_openai = chat_with_custom_openai_2 = None # Add chat_with_local_llm if needed
    API_IMPORTS_SUCCESSFUL = False
    print("-" * 60)
    print("WARNING: Could not import one or more API library functions.")
    print("Check logs for details. Affected API functionality will be disabled.")
    print("-" * 60)
#
#######################################################################################################################
#
# Functions:

# --- Configuration Loading ---
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "tldw_cli" / "config.toml"
DEFAULT_CONFIG = {
    "general": {"default_tab": "chat", "log_level": "DEBUG"},
    "logging": {
        # "log_file": None, # Keep this if you want an *override* path separate from DB path logic
        "log_filename": "tldw_cli_app.log", # Default filename to use in DB dir
        "file_log_level": "INFO",
        "log_max_bytes": 10 * 1024 * 1024, # 10 MB
        "log_backup_count": 5
    },    "api_keys": { # Placeholders/Documentation
        "openai": "Set OPENAI_API_KEY env var", "anthropic": "Set ANTHROPIC_API_KEY env var",
        "google": "Set GOOGLE_API_KEY env var", "cohere": "Set COHERE_API_KEY env var",
        "groq": "Set GROQ_API_KEY env var", "mistral": "Set MISTRAL_API_KEY env var",
        "openrouter": "Set OPENROUTER_API_KEY env var", "deepseek": "Set DEEPSEEK_API_KEY env var",
    },
    "api_endpoints": { # Default URLs
        "ollama_url": "http://localhost:11434", "llama_cpp_url": "http://localhost:8080",
        "oobabooga_url": "http://localhost:5000/api", "kobold_url": "http://localhost:5001/api",
        "vllm_url": "http://localhost:8000", "custom_openai_url": "http://localhost:1234/v1",
        "custom_openai_2_url": "http://localhost:5678/v1",
    },
    "chat_defaults": { # Defaults for Chat Tab
        "provider": "Ollama", "model": "ollama/llama3:latest",
        "system_prompt": "You are a helpful assistant.", "temperature": 0.7,
        "top_p": 0.95, "min_p": 0.05, "top_k": 50,
    },
    "character_defaults": { # Defaults for Character Tab
        "provider": "Anthropic", "model": "claude-3-haiku-20240307",
        "system_prompt": "You are a character in a story.", "temperature": 0.8,
        "top_p": 0.9, "min_p": 0.0, "top_k": 100,
    },
    "database": {"path": str(Path.home() / ".local" / "share" / "tldw_cli" / "history.db")},
    "server": {"url": "http://localhost:8001/api", "token": None }
}

def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Loads configuration from TOML file, merging with defaults."""
    config = {k: v.copy() if isinstance(v, dict) else v for k, v in DEFAULT_CONFIG.items()} # Deep copy defaults

    config_path.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists

    if config_path.exists():
        try:
            with open(config_path, "rb") as f:
                user_config = tomllib.load(f)
            # Merge user config into defaults (simple one-level deep merge)
            for key, value in user_config.items():
                if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                    config[key].update(value)
                else:
                    config[key] = value
            logging.info(f"Loaded configuration from: {config_path}")
        except Exception as e:
            logging.error(f"Failed to load config from {config_path}: {e}", exc_info=True)
            logging.warning("Using default configuration values.")
    else:
        logging.warning(f"Config file not found at {config_path}. Using defaults and attempting to create.")
        try:
            with open(config_path, "w") as f:
                try:
                    # Use external toml library if available for writing
                    import toml
                    toml.dump(DEFAULT_CONFIG, f)
                    logging.info(f"Created default configuration file at: {config_path}")
                except ImportError:
                    f.write("# Default config file for tldw_cli\n")
                    f.write("# Install 'toml' library (`pip install toml`) to write defaults automatically.\n")
                    # Could write manually, but complex for nested dicts
                    logging.warning("`toml` library not installed. Created basic placeholder config file.")
        except Exception as e:
            logging.error(f"Failed to create default config file at {config_path}: {e}", exc_info=True)
    return config

# --- Constants ---
TAB_CHAT = "chat"; TAB_CHARACTER = "character"; TAB_MEDIA = "media"; TAB_SEARCH = "search"
TAB_INGEST = "ingest"; TAB_LOGS = "logs"; TAB_STATS = "stats"
ALL_TABS = [ TAB_CHAT, TAB_CHARACTER, TAB_MEDIA, TAB_SEARCH, TAB_INGEST, TAB_LOGS, TAB_STATS ]

# --- Define API Models (Combined Cloud & Local) ---
# (Keep your existing API_MODELS_BY_PROVIDER and LOCAL_PROVIDERS dictionaries)
API_MODELS_BY_PROVIDER = {
    "OpenAI": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    "Anthropic": ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
    "Google": ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest"],
    "MistralAI": ["mistral-large-latest", "mistral-small-latest", "open-mixtral-8x7b"],
    "Custom": ["custom-model-alpha", "custom-model-beta"]
}
LOCAL_PROVIDERS = {
    "Llama.cpp": ["llama-model-1"], "Oobabooga": ["ooba-model-a"], "KoboldCpp": ["kobold-model-x"],
    "Ollama": ["ollama/llama3:latest", "ollama/mistral:latest"], "vLLM": ["vllm-model-z"],
    "TabbyAPI": ["tabby-model"], "Aphrodite": ["aphrodite-engine"], "Custom-2": ["custom-model-gamma"],
    "Groq": ["llama3-70b-8192", "mixtral-8x7b-32768"], "Cohere": ["command-r-plus", "command-r"],
    "OpenRouter": ["meta-llama/llama-3-70b-instruct"], "HuggingFace": ["mistralai/Mixtral-8x7B-Instruct-v0.1"],
    "DeepSeek": ["deepseek-chat"],
}
ALL_API_MODELS = {**API_MODELS_BY_PROVIDER, **LOCAL_PROVIDERS}
AVAILABLE_PROVIDERS = list(ALL_API_MODELS.keys())

# --- API Function Map ---
# (Ensure this aligns with your actual imported functions or None values)
API_FUNCTION_MAP = {
    "OpenAI": chat_with_openai, "Anthropic": chat_with_anthropic, "Google": chat_with_google,
    "MistralAI": chat_with_mistral, "Custom": chat_with_custom_openai, "Llama.cpp": chat_with_llama,
    "Oobabooga": chat_with_oobabooga, "KoboldCpp": chat_with_kobold, "Ollama": chat_with_ollama,
    "vLLM": chat_with_vllm, "TabbyAPI": chat_with_tabbyapi, "Aphrodite": chat_with_aphrodite,
    "Custom-2": chat_with_custom_openai_2, "Groq": chat_with_groq, "Cohere": chat_with_cohere,
    "OpenRouter": chat_with_openrouter, "HuggingFace": chat_with_huggingface, "DeepSeek": chat_with_deepseek,
}

# --- ASCII Portrait ---
ASCII_PORTRAIT = r"""
  .--./)
 /.''.')
 | \ '/
 W `-'
 \\    '.
  '.    /
    `~~`
"""

# --- Custom Logging Handler ---
class RichLogHandler(logging.Handler):
    def __init__(self, rich_log_widget: RichLog):
        super().__init__()
        self.rich_log_widget = rich_log_widget
        self.log_queue = asyncio.Queue()
        self.formatter = logging.Formatter(
            "{asctime} [{levelname:<8}] {name}:{lineno:<4} : {message}",
            style="{", datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.setFormatter(self.formatter)
        self._queue_processor_task = None

    def start_processor(self, app: App):
        """Starts the log queue processing task."""
        if not self._queue_processor_task or self._queue_processor_task.done():
            self._queue_processor_task = app.create_task(self._process_log_queue(), name="RichLogProcessor")
            logging.debug("RichLog queue processor task started.")

    async def stop_processor(self):
        """Signals the queue processor to stop and waits for it."""
        if self._queue_processor_task and not self._queue_processor_task.done():
            logging.debug("Attempting to stop RichLog queue processor...")
            self._queue_processor_task.cancel()
            try:
                await self._queue_processor_task
            except asyncio.CancelledError:
                logging.debug("RichLog queue processor task cancelled successfully.")
            except Exception as e:
                logging.error(f"Error while waiting for RichLog processor cancellation: {e}", exc_info=True)
        self._queue_processor_task = None

    async def _process_log_queue(self):
        """Coroutine to process logs from the queue and write to the widget."""
        while True:
            try:
                message = await self.log_queue.get()
                if self.rich_log_widget.is_mounted and self.rich_log_widget.app:
                    self.rich_log_widget.write(message)
                self.log_queue.task_done()
            except asyncio.CancelledError:
                logging.debug("RichLog queue processor task received cancellation.")
                # Process any remaining items? Might be risky if app is shutting down.
                # while not self.log_queue.empty():
                #    try: message = self.log_queue.get_nowait(); # process...
                #    except asyncio.QueueEmpty: break
                break # Exit the loop on cancellation
            except Exception as e:
                print(f"!!! CRITICAL ERROR in RichLog processor: {e}") # Use print as fallback
                traceback.print_exc()
                # Avoid continuous loop on error, maybe sleep?
                await asyncio.sleep(1)

    def emit(self, record: logging.LogRecord):
        """Format the record and put it onto the async queue."""
        try:
            message = self.format(record)
            # Use call_soon_threadsafe if emit might be called from non-asyncio threads (workers)
            # For workers started with thread=True, this is necessary.
            if hasattr(self.rich_log_widget, 'app') and self.rich_log_widget.app:
                self.rich_log_widget.app._loop.call_soon_threadsafe(self.log_queue.put_nowait, message)
            else: # Fallback during startup/shutdown
                 if record.levelno >= logging.WARNING: print(f"LOG_FALLBACK: {message}")
        except Exception:
            print(f"!!!!!!!! ERROR within RichLogHandler.emit !!!!!!!!!!") # Use print as fallback
            traceback.print_exc()


# --- Global variable for config ---
APP_CONFIG = load_config()

# Configure root logger based on config BEFORE app starts fully
_initial_log_level_str = APP_CONFIG.get("general", {}).get("log_level", "INFO").upper()
_initial_log_level = getattr(logging, _initial_log_level_str, logging.INFO)
# Define a basic initial format
_initial_log_format = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
# Remove existing handlers before basicConfig to avoid duplicates if script is re-run
# logging.getLogger().handlers.clear() # Careful with this in complex setups
logging.basicConfig(level=_initial_log_level, format=_initial_log_format, force=True) # force=True might help override defaults
logging.info("Initial basic logging configured.")

# --- Helper Function for Sidebar ---
def create_settings_sidebar(id_prefix: str, config: dict) -> ComposeResult:
    """Yields the widgets for a settings sidebar, using config for defaults."""
    with VerticalScroll(id=f"{id_prefix}-sidebar", classes="sidebar"):
        defaults = config.get(f"{id_prefix}_defaults", config.get("chat_defaults", {}))
        default_provider = defaults.get("provider", AVAILABLE_PROVIDERS[0] if AVAILABLE_PROVIDERS else "")
        default_model = defaults.get("model", "")
        default_system_prompt = defaults.get("system_prompt", "")
        default_temp = str(defaults.get("temperature", 0.7))
        default_top_p = str(defaults.get("top_p", 0.95))
        default_min_p = str(defaults.get("min_p", 0.05))
        default_top_k = str(defaults.get("top_k", 50))

        yield Static("Settings", classes="sidebar-title")
        yield Static("API Provider", classes="sidebar-label")
        provider_options = [(provider, provider) for provider in AVAILABLE_PROVIDERS]
        yield Select(
            options=provider_options, prompt="Select Provider...", allow_blank=False,
            id=f"{id_prefix}-api-provider", value=default_provider
        )
        yield Static("Model", classes="sidebar-label")
        initial_models = ALL_API_MODELS.get(default_provider, [])
        model_options = [(model, model) for model in initial_models]
        current_model_value = default_model if default_model in initial_models else (initial_models[0] if initial_models else None)
        yield Select(
            options=model_options, prompt="Select Model...", allow_blank=True,
            id=f"{id_prefix}-api-model", value=current_model_value
        )
        yield Static("API Key (Set in config/env)", classes="sidebar-label", id=f"{id_prefix}-api-key-placeholder")
        yield Static("System prompt", classes="sidebar-label")
        yield TextArea(id=f"{id_prefix}-system-prompt", text=default_system_prompt, classes="sidebar-textarea")
        yield Static("Temperature", classes="sidebar-label")
        yield Input(placeholder="e.g., 0.7", id=f"{id_prefix}-temperature", value=default_temp, classes="sidebar-input")
        yield Static("Top-P", classes="sidebar-label")
        yield Input(placeholder="0.0 to 1.0", id=f"{id_prefix}-top-p", value=default_top_p, classes="sidebar-input")
        yield Static("Min-P", classes="sidebar-label")
        yield Input(placeholder="0.0 to 1.0", id=f"{id_prefix}-min-p", value=default_min_p, classes="sidebar-input")
        yield Static("Top-K", classes="sidebar-label")
        yield Input(placeholder="e.g., 50", id=f"{id_prefix}-top-k", value=default_top_k, classes="sidebar-input")

# --- Main App ---
class TldwCli(App):
    CSS_PATH = "tldw_cli.css"
    BINDINGS = [ Binding("ctrl+q", "quit", "Quit App", show=True) ]

    # Define reactive var at CLASS level
    current_tab = reactive(TAB_CHAT) # Default to CHAT, will be updated in __init__

    def __init__(self, config: dict):
        super().__init__()
        self.app_config = config
        # Determine initial tab from config and SET the reactive variable
        initial_tab_from_config = self.app_config.get("general", {}).get("default_tab", TAB_CHAT)
        if initial_tab_from_config not in ALL_TABS:
            logging.warning(f"Default tab '{initial_tab_from_config}' from config not valid. Falling back to '{TAB_CHAT}'.")
            self.initial_tab = TAB_CHAT
        else:
            self.initial_tab = initial_tab_from_config
        # Set the *initial value* of the reactive variable AFTER super().__init__()
        self.current_tab = self.initial_tab
        logging.info(f"App initialized. Initial tab set to: {self.current_tab}")

        self._rich_log_handler = None

    def on_mount(self) -> None:
        """Configure logging (RichLog and File) and start tasks."""
        logging.info("--- App Mounting ---")
        root_logger = logging.getLogger()  # Get root logger early

        # --- Setup RichLog Handler (TUI Display) ---
        try:
            log_display_widget = self.query_one("#app-log-display", RichLog)
            self._rich_log_handler = RichLogHandler(log_display_widget)

            # Clean up existing handlers if necessary (be cautious)
            for handler in root_logger.handlers[:]:
                 if isinstance(handler, logging.StreamHandler) and not isinstance(handler, (RichLogHandler, logging.FileHandler)):
                     logging.debug(f"Removing existing StreamHandler: {handler}")
                     root_logger.removeHandler(handler)
                 if isinstance(handler, RichLogHandler) and handler is not self._rich_log_handler:
                     logging.warning(f"Removing potentially stale RichLogHandler: {handler}")
                     root_logger.removeHandler(handler)

            # Configure and add the RichLog handler
            widget_handler_level_str = self.app_config.get("general", {}).get("log_level", "DEBUG").upper()
            widget_handler_level = getattr(logging, widget_handler_level_str, logging.DEBUG)
            self._rich_log_handler.setLevel(widget_handler_level)
            root_logger.addHandler(self._rich_log_handler)
            # Ensure root level is low enough for all handlers
            current_root_level = root_logger.getEffectiveLevel()
            lowest_level = min(current_root_level, widget_handler_level) # Consider file handler level too if added later
            root_logger.setLevel(logging.DEBUG if lowest_level <= logging.DEBUG else lowest_level)

            # Start the log queue processing task
            self._rich_log_handler.start_processor(self)
            logging.info(
                f"RichLog TUI logging configured. Widget Handler level: {logging.getLevelName(self._rich_log_handler.level)}")

        except Exception as e:
            # Use print as last resort if TUI logging setup fails
            print(f"!!!!!!!! FATAL ERROR setting up RichLog TUI Handler !!!!!!!!")
            traceback.print_exc()
            try:
                logging.exception("FATAL: Failed to configure RichLogHandler!")
            except Exception:
                pass  # Avoid error loops

        # --- Setup Rotating File Handler ---
        try:
            # 1. Get Database Path from Config
            db_path_str = self.app_config.get("database", {}).get("path")
            if not db_path_str:
                # Log this warning, but don't crash the app if DB path is missing
                logging.warning("database.path not found in config. Skipping file logging based on DB path.")
                raise ValueError("database.path not configured.")  # Raise to skip rest of try block

            db_path = Path(db_path_str).expanduser().resolve()
            log_dir = db_path.parent  # Directory containing the DB

            # 2. Get Log Filename from Config (or default)
            log_filename = self.app_config.get("logging", {}).get("log_filename", "tldw_cli_app.log")
            log_file_path = log_dir / log_filename

            # 3. Ensure Log Directory Exists
            log_dir.mkdir(parents=True, exist_ok=True)
            logging.debug(f"Ensured log directory exists: {log_dir}")

            # 4. Create RotatingFileHandler
            # Get rotation settings from config
            max_bytes = int(self.app_config.get("logging", {}).get("log_max_bytes", 10 * 1024 * 1024))
            backup_count = int(self.app_config.get("logging", {}).get("log_backup_count", 5))

            # Use UTF-8 encoding for the log file
            file_handler = logging.handlers.RotatingFileHandler(
                log_file_path, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
            )

            # 5. Set Log Level for File Handler
            file_log_level_str = self.app_config.get("logging", {}).get("file_log_level", "INFO").upper()
            file_log_level = getattr(logging, file_log_level_str, logging.INFO)
            file_handler.setLevel(file_log_level)

            # 6. Set Formatter (use same format as RichLog for consistency)
            file_formatter = logging.Formatter(
                "{asctime} [{levelname:<8}] {name}:{lineno:<4} : {message}",
                style="{", datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)

            # 7. Add Handler to Root Logger
            root_logger.addHandler(file_handler)

            # Adjust root logger level again if file handler needs lower level
            current_root_level = root_logger.getEffectiveLevel()
            lowest_level = min(current_root_level, file_log_level)
            root_logger.setLevel(logging.DEBUG if lowest_level <= logging.DEBUG else lowest_level)

            logging.info(
                f"File logging enabled: '{log_file_path}', Level: {logging.getLevelName(file_handler.level)}, Rotation: {max_bytes / (1024 * 1024):.1f}MB / {backup_count} backups")

        except Exception as e:
            # Log the error but allow the app to continue without file logging
            logging.error(f"Failed to configure file logging: {e}", exc_info=True)

        logging.info("App mount process completed.")  # Final message after all setup

    async def on_unmount(self) -> None:
        """Clean up logging resources on application exit."""
        logging.info("--- App Unmounting ---")
        if self._rich_log_handler:
            # Stop the processor task first
            await self._rich_log_handler.stop_processor()
            # Then remove the handler
            logging.getLogger().removeHandler(self._rich_log_handler)
            logging.info("RichLogHandler removed and processor stopped.")
        # File handlers usually closed by logging shutdown

    def compose(self) -> ComposeResult:
        yield Header()
        yield from self.compose_tabs()
        yield from self.compose_content_area()
        yield Footer()

    def compose_tabs(self) -> ComposeResult:
        with Horizontal(id="tabs"):
            for tab_id in ALL_TABS:
                yield Button(
                    tab_id.replace('_', ' ').capitalize(),
                    id=f"tab-{tab_id}",
                    classes="-active" if tab_id == self.current_tab else "" # Use current_tab after __init__
                )

    def compose_content_area(self) -> ComposeResult:
         # Pass the loaded config to the sidebar creator
        with Container(id="content"):
            # Determine visibility based on self.current_tab set in __init__
            # --- Chat Window ---
            with Container(id=f"{TAB_CHAT}-window", classes="window" if self.current_tab == TAB_CHAT else "window hidden"):
                yield from create_settings_sidebar(TAB_CHAT, self.app_config)
                with Container(id="chat-main-content"):
                    yield RichLog(id="chat-log", wrap=True, highlight=True, classes="chat-log")
                    with Horizontal(id="chat-input-area"):
                        yield TextArea(id="chat-input", classes="chat-input")
                        yield Button("Send", id="send-chat", classes="send-button")

            # --- Character Chat Window ---
            with Container(id=f"{TAB_CHARACTER}-window", classes="window" if self.current_tab == TAB_CHARACTER else "window hidden"):
                yield from create_settings_sidebar(TAB_CHARACTER, self.app_config)
                with Container(id="character-main-content"):
                    with Horizontal(id="character-top-area"):
                        yield RichLog(id="character-log", wrap=True, highlight=True, classes="chat-log")
                        yield Static(ASCII_PORTRAIT, id="character-portrait")
                    with Horizontal(id="character-input-area"):
                        yield TextArea(id="character-input", classes="chat-input")
                        yield Button("Send", id="send-character", classes="send-button")

            # --- Logs Window ---
            with Container(id=f"{TAB_LOGS}-window", classes="window" if self.current_tab == TAB_LOGS else "window hidden"):
                 yield RichLog(id="app-log-display", wrap=True, highlight=True, markup=True, auto_scroll=True)

            # --- Other Placeholder Windows ---
            for tab_id in ALL_TABS:
                if tab_id not in [TAB_CHAT, TAB_CHARACTER, TAB_LOGS]:
                    yield Static(
                        f"{tab_id.replace('_', ' ').capitalize()} Window Placeholder",
                        id=f"{tab_id}-window",
                        classes="window placeholder-window" + ("" if self.current_tab == tab_id else " hidden")
                    )

    # --- Reactive Watcher ---
    # This method name MUST match 'watch_' + attribute_name
    def watch_current_tab(self, old_tab: str, new_tab: str) -> None:
        """Called automatically when the 'current_tab' reactive variable changes."""
        # No need for manual calls if reactivity is working
        logging.critical(f"!!!! WATCHER watch_current_tab FIRED !!!! Switching from '{old_tab}' to '{new_tab}'")
        if not self.is_mounted: # Add check to prevent running before mount completes
            logging.warning("Watcher fired before app fully mounted. Skipping UI updates.")
            return
        try:
            # Deactivate old tab button and hide old window
            # Use query to safely handle potential missing elements (though they should exist)
            old_button_query = f"#tab-{old_tab}"
            old_window_query = f"#{old_tab}-window"

            old_buttons = self.query(old_button_query)
            if old_buttons:
                old_buttons.first().remove_class("-active")
                logging.debug(f"Watcher: Deactivated button {old_button_query}")
            else:
                logging.warning(f"Watcher: Could not find old button {old_button_query} to deactivate.")

            old_windows = self.query(old_window_query)
            if old_windows:
                 old_windows.first().display = False # More reliable than adding/removing classes sometimes
                 # old_windows.first().add_class("hidden") # Alternative using CSS class
                 logging.debug(f"Watcher: Hid window {old_window_query}")
            else:
                logging.warning(f"Watcher: Could not find old window {old_window_query} to hide.")

            # Activate new tab button and show new window
            new_button_query = f"#tab-{new_tab}"
            new_window_query = f"#{new_tab}-window"

            new_buttons = self.query(new_button_query)
            if new_buttons:
                new_buttons.first().add_class("-active")
                logging.debug(f"Watcher: Activated button {new_button_query}")
            else:
                 logging.error(f"Watcher: Could not find new button {new_button_query} to activate.")

            new_windows = self.query(new_window_query)
            if new_windows:
                 new_window = new_windows.first()
                 new_window.display = True # Make sure it's visible
                 # new_window.remove_class("hidden") # Alternative using CSS class
                 logging.debug(f"Watcher: Displayed window {new_window_query}")

                 # Try to focus input area in non-log tabs
                 if new_tab not in [TAB_LOGS]:
                     inputs = new_window.query("TextArea")
                     if inputs:
                         try:
                             # Need to defer focus slightly until after layout/display updates
                             self.set_timer(0.05, inputs.first().focus)
                             logging.debug(f"Watcher: Scheduled focus for TextArea in tab {new_tab}")
                         except Exception as focus_err:
                             logging.warning(f"Watcher: Could not schedule focus for TextArea in tab {new_tab}: {focus_err}")
                     else:
                         logging.debug(f"Watcher: No TextArea found to focus in tab {new_tab}")
            else:
                 logging.error(f"Watcher: Could not find new window {new_window_query} to display.")

        except Exception as e:
            # Catch specific ScreenStackError during init? Or just log generically.
            if isinstance(e, ScreenStackError) and not self.is_running:
                 logging.warning(f"Watcher: Ignored ScreenStackError during app init.")
            else:
                 logging.error(f"!!!! WATCHER ERROR !!!! Error during tab switch from {old_tab} to {new_tab}: {e}", exc_info=True)


    # --- Event Handlers ---
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle changes in Select widgets, specifically for API provider."""
        select_id = event.control.id
        if select_id and select_id.endswith("-api-provider"):
            id_prefix = select_id.removesuffix("-api-provider")
            new_provider = str(event.value) if event.value is not None else "" # Handle None case
            logging.info(f"Provider changed for '{id_prefix}': '{new_provider}'")

            model_select_id = f"#{id_prefix}-api-model"
            logging.debug(f"Attempting to find model select: {model_select_id}")
            try:
                model_select = self.query_one(model_select_id, Select)
            except Exception as e:
                logging.critical(f"CRITICAL: Could not find model select '{model_select_id}': {e}", exc_info=True)
                return

            models = ALL_API_MODELS.get(new_provider, []) if new_provider else []
            new_model_options = [(model, model) for model in models]

            logging.info(f"Updating models for '{id_prefix}': {models}")
            model_select.set_options(new_model_options)

            if models:
                config_defaults = self.app_config.get(f"{id_prefix}_defaults", {})
                config_default_model = config_defaults.get("model") if config_defaults.get("provider") == new_provider else None
                model_to_set = models[0] # Default to first if config doesn't match
                if config_default_model and config_default_model in models:
                    model_to_set = config_default_model
                    logging.debug(f"Setting model from config default: '{model_to_set}'")
                else:
                     logging.debug(f"Setting model to first available: '{model_to_set}'")

                model_select.value = model_to_set
                model_select.prompt = "Select Model..."
            else:
                model_select.value = None # Clear the value if no models
                model_select.prompt = "No models available" if new_provider else "Select Provider first"
                logging.info(f"No models available for '{id_prefix}'.")

            model_select.refresh()
            logging.debug(f"Refreshed model select widget: {model_select.id}")
        else:
             logging.debug(f"Ignoring Select.Changed event from {select_id or 'UNKNOWN'}")


    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses for tabs and sending messages."""
        button_id = event.button.id
        logging.debug(f"Button pressed: {button_id}")

        # --- Handle Tab Switching ---
        if button_id and button_id.startswith("tab-"):
            new_tab_id = button_id.replace("tab-", "")
            logging.info(f"Tab button {button_id} pressed. Requesting switch to '{new_tab_id}'")

            if new_tab_id == self.current_tab:
                logging.debug(f"Already on tab '{new_tab_id}'. Ignoring.")
                return

            # Assigning to the reactive variable SHOULD trigger the watcher
            self.current_tab = new_tab_id
            # No manual call needed if reactivity works correctly

        elif button_id and button_id.startswith("send-"):
            # --- Send message logic ---
            chat_id_part = button_id.replace("send-", "")
            prefix = chat_id_part
            logging.info(f"'Send' button pressed for '{chat_id_part}'")
            # --- Query Widgets ---
            try:
                text_area = self.query_one(f"#{prefix}-input", TextArea)
                chat_log_widget = self.query_one(f"#{prefix}-log", RichLog)
                provider_widget = self.query_one(f"#{prefix}-api-provider", Select)
                model_widget = self.query_one(f"#{prefix}-api-model", Select)
                system_prompt_widget = self.query_one(f"#{prefix}-system-prompt", TextArea)
                temp_widget = self.query_one(f"#{prefix}-temperature", Input)
                top_p_widget = self.query_one(f"#{prefix}-top-p", Input)
                min_p_widget = self.query_one(f"#{prefix}-min-p", Input)
                top_k_widget = self.query_one(f"#{prefix}-top-k", Input)
            except Exception as e:
                logging.error(f"Could not find required UI widgets for '{prefix}': {e}", exc_info=True)
                try: self.query_one(f"#{prefix}-log", RichLog).write("[bold red]Error: UI elements missing.[/]")
                except Exception: logging.error("Also failed to write UI error to chat log.")
                return

            # --- Get Values ---
            message = text_area.text.strip()
            selected_provider = str(provider_widget.value) if provider_widget.value else None
            selected_model = str(model_widget.value) if model_widget.value else None
            system_prompt = system_prompt_widget.text
            # Use helper for safe float/int conversion
            temperature = self._safe_float(temp_widget.value, 0.7, "temperature")
            top_p = self._safe_float(top_p_widget.value, 0.95, "top_p")
            min_p = self._safe_float(min_p_widget.value, 0.05, "min_p")
            top_k = self._safe_int(top_k_widget.value, 50, "top_k")

            # --- Basic Validation ---
            if not API_IMPORTS_SUCCESSFUL:
                chat_log_widget.write("[bold red]API libraries failed load. Cannot send.[/]")
                logging.error(f"Send attempt ('{prefix}') failed: API libraries not loaded.")
                return
            if not message:
                logging.debug(f"Empty message submitted in '{prefix}'.")
                text_area.clear(); text_area.focus()
                return
            if not selected_provider:
                chat_log_widget.write("[bold red]Select API Provider.[/]")
                logging.warning(f"Send attempt ('{prefix}') failed: Provider not selected.")
                return
            if not selected_model:
                chat_log_widget.write("[bold red]Select Model.[/]")
                logging.warning(f"Send attempt ('{prefix}') failed: Model not selected.")
                return

            # --- Log User Message & Clear Input ---
            chat_log_widget.write(f"You: {message}")
            text_area.clear()

            # --- Prepare and Dispatch API Call ---
            logging.info(f"API call: P='{selected_provider}', M='{selected_model}'")
            chat_log_widget.write(f"[dim]AI thinking ({selected_provider} / {selected_model})...[/]")

            api_function = API_FUNCTION_MAP.get(selected_provider)
            if api_function is None:
                error_msg = f"Error: API function for '{selected_provider}' not found/loaded."
                logging.error(error_msg)
                chat_log_widget.write(f"[bold red]{error_msg}[/]")
                return

            # --- Get API URL from Config ---
            api_endpoints = self.app_config.get("api_endpoints", {})
            api_url = self._get_api_name(selected_provider, api_endpoints)

            # --- Prepare API Arguments ---
            api_args = {
                "api_name": api_url, "api_key": None, "input_data": message, "model": selected_model,
                "system_message": system_prompt, "temp": temperature, "streaming": False,
                "topp": top_p, "top_p": top_p, "maxp": top_p, "topk": top_k, "minp": min_p,

            }

            # Use a distinct name like filtered_api_call_args
            filtered_api_call_args = {k: v for k, v in api_args.items() if v is not None or k in ["api_key", "system_message", "api_name"]}
            loggable_args = {k: v for k, v in filtered_api_call_args.items() if k != 'api_key'}
            func_name = getattr(api_function, '__name__', 'UNKNOWN_FUNCTION')
            logging.debug(f"Calling {func_name} with args: {loggable_args}")


        # --- Define the Worker Wrapper ---
        # This nested function will run in the worker thread.
        # It captures necessary variables from the surrounding scope.
        def api_call_wrapper():
            # Note: We are accessing 'self', 'api_function', 'filtered_api_call_args',
            # and 'chat_log_widget' from the enclosing 'on_button_pressed' scope.
            # Ensure 'chat_log_widget' is only used for logging *within* the worker
            # if absolutely necessary (better handled by on_worker_state_changed).
            # The main purpose is to pass args to _api_worker.
            return self._api_worker(api_function, filtered_api_call_args, chat_log_widget)

        # --- Run Worker ---
        # *** FIX: Pass the wrapper function as the worker, no other positional args ***
        self.run_worker(
            api_call_wrapper,         # The wrapper function is the worker now
            # NO positional arguments here for the worker itself
            # Keyword arguments FOR run_worker:
            name=f"API_Call_{prefix}",
            group="api_calls",
            exclusive=False,
            thread=True
        )

    # --- Helper methods for parsing inputs ---
    def _safe_float(self, value: str, default: float, name: str) -> float:
        if not value: return default
        try: return float(value)
        except ValueError:
            logging.warning(f"Invalid {name} value '{value}', using default {default}")
            return default

    def _safe_int(self, value: str, default: int, name: str) -> int:
        if not value: return default
        try: return int(value)
        except ValueError:
            logging.warning(f"Invalid {name} value '{value}', using default {default}")
            return default

    # --- Helper method for getting API URL ---
    def _get_api_name(self, provider: str, endpoints: dict) -> Union[str, None]:
        provider_key_map = {
            "Ollama": "ollama_url", "Llama.cpp": "llama_cpp_url", "Oobabooga": "oobabooga_url",
            "KoboldCpp": "kobold_url", "vLLM": "vllm_url", "Custom": "custom_openai_url",
            "Custom-2": "custom_openai_2_url", # Add other local providers mapped to URL keys
        }
        endpoint_key = provider_key_map.get(provider)
        if endpoint_key:
            url = endpoints.get(endpoint_key)
            if url:
                logging.debug(f"Using API endpoint '{url}' for provider '{provider}'")
                return url
            else:
                logging.warning(f"URL key '{endpoint_key}' for '{provider}' missing in config.")
                return None
        else:
             # Cloud providers or those not needing a URL explicitly set here
             logging.debug(f"No specific endpoint URL key configured for provider '{provider}'.")
             return None

    # --- Worker function ---
    def _api_worker(self, api_func: callable, api_args: dict, log_widget: RichLog) -> Union[str, Generator[Any, Any, None], None]:
        """Executes the synchronous API call in a thread."""
        # (Implementation remains the same as the last correct version)
        func_name = getattr(api_func, '__name__', 'UNKNOWN_FUNCTION')
        try:
            if api_func is None: raise ValueError("API function is None.")
            # Note: log_widget is passed here but might not be strictly needed inside
            # if all TUI updates happen in on_worker_state_changed.
            # Using it only for logging inside the worker is generally okay.
            logging.info(
                f"Worker executing: {func_name} with args: { {k: v for k, v in api_args.items() if k != 'api_key'} }")
            result = api_func(**api_args)  # Call the actual API function
            logging.info(f"Worker {func_name} finished successfully.")
            return result
        except Exception as e:
            logging.exception(f"Error during API call in worker ({func_name}): {e}")
            return f"[bold red]API Error ({func_name}):[/] {str(e)}"

    # --- Handle worker completion ---
    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Called when a worker changes state."""
        worker_name = event.worker.name or "Unknown Worker"
        logging.debug(f"Worker '{worker_name}' state changed to: {event.state}")

        if not worker_name.startswith("API_Call_"):
            logging.debug(f"Ignoring state change for non-API worker: {worker_name}")
            return

        prefix = worker_name.replace("API_Call_", "")
        try:
            chat_log_widget = self.query_one(f"#{prefix}-log", RichLog)

            if event.state == WorkerState.SUCCESS:
                result = event.worker.result
                if isinstance(result, str):
                    # Check if the result itself is an error message from the worker
                    if result.startswith("[bold red]API Error"):
                        logging.error(f"API call ({prefix}) worker returned an error message: {result}")
                        chat_log_widget.write(f"AI: {result}") # Display the error from worker
                    else:
                        logging.info(f"API call ({prefix}) successful. Result length: {len(result)}")
                        chat_log_widget.write(f"AI: {result}")
                elif isinstance(result, Generator):
                    # Basic generator handling (better implementation needed for true streaming)
                    logging.info(f"API call ({prefix}) successful (Generator started).")
                    chat_log_widget.write("AI: ") # Start the line
                    full_response = ""
                    try:
                        for chunk in result:
                            if isinstance(chunk, str):
                                chat_log_widget.write(chunk, shrink=False) # Append chunk without newline
                                full_response += chunk
                            else: # Handle potential non-string chunks if API yields them
                                logging.warning(f"Received non-string chunk from generator: {type(chunk)}")
                                chat_log_widget.write(str(chunk), shrink=False)
                                full_response += str(chunk)
                        logging.info(f"API call ({prefix}) generator finished. Total length: {len(full_response)}")
                    except Exception as gen_e:
                        logging.error(f"Error processing generator stream for '{prefix}': {gen_e}", exc_info=True)
                        chat_log_widget.write("[bold red] Error during streaming.[/]")
                elif result is None:
                     logging.error(f"API worker '{worker_name}' returned None.")
                     chat_log_widget.write("[bold red]AI: Error during processing (worker returned None).[/]")
                else:
                    logging.error(f"API worker '{worker_name}' returned unexpected type: {type(result)}")
                    chat_log_widget.write(f"[bold red]Error: Unexpected result type from API.[/]")

            elif event.state == WorkerState.ERROR:
                logging.error(f"Worker '{worker_name}' failed critically:", exc_info=event.worker.error)
                chat_log_widget.write(f"[bold red]AI Error: Processing failed critically. Check logs for details.[/]")

            # Try to focus the input area again after response/error
            try:
                text_area = self.query_one(f"#{prefix}-input", TextArea)
                self.set_timer(0.05, text_area.focus)
            except Exception:
                logging.debug(f"Could not refocus input for '{prefix}' after worker completion.")

        except Exception as e:
            logging.error(f"Error in on_worker_state_changed for worker '{worker_name}': {e}", exc_info=True)


# --- Configuration File Content (for reference or auto-creation) ---
CONFIG_TOML_CONTENT = """
# Configuration for tldw-cli TUI App
[general]
default_tab = "chat" # e.g., "chat", "character", "logs"
log_level = "DEBUG" # DEBUG, INFO, WARNING, ERROR, CRITICAL

[logging]
# Controls logging to the file located next to the database file.
log_filename = "tldw_cli_app.log" # The name of the log file.
file_log_level = "INFO"        # Level for the file log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
log_max_bytes = 10485760       # Max file size in bytes before rotation (e.g., 10MB = 10 * 1024 * 1024)
log_backup_count = 5           # Number of old log files to keep.

[api_keys]
# Use environment variables (e.g., OPENAI_API_KEY) primarily!
openai = "Set OPENAI_API_KEY environment variable"anthropic = "Set ANTHROPIC_API_KEY environment variable"
# ... other placeholders ...
[api_endpoints]
ollama_url = "http://localhost:11434"
llama_cpp_url = "http://localhost:8080"
oobabooga_url = "http://localhost:5000/api"
kobold_url = "http://localhost:5001/api"
vllm_url = "http://localhost:8000"
custom_openai_url = "http://localhost:1234/v1"
custom_openai_2_url = "http://localhost:5678/v1"

[chat_defaults]
provider = "Ollama"
model = "ollama/llama3:latest"
system_prompt = "You are a helpful AI assistant."
temperature = 0.7
top_p = 0.95
min_p = 0.05
top_k = 50

[character_defaults]
provider = "Anthropic"
model = "claude-3-haiku-20240307"
system_prompt = "You are roleplaying as a witty pirate captain."
temperature = 0.8
top_p = 0.9
min_p = 0.0
top_k = 100

[database]
path = "~/.local/share/tldw_cli/history.db"

[server]
url = "http://localhost:8001/api/v1"
token = null
"""

# --- Main execution block ---
if __name__ == "__main__":
    # Ensure config file exists (create default if missing)
    try:
        if not DEFAULT_CONFIG_PATH.exists():
            logging.info(f"Config file not found at {DEFAULT_CONFIG_PATH}, creating default.")
            DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(DEFAULT_CONFIG_PATH, "w") as f:
                f.write(CONFIG_TOML_CONTENT) # Write the example content
    except Exception as e:
        logging.error(f"Could not ensure creation of default config file: {e}", exc_info=True)

    # --- CSS definition ---
    # (Keep your CSS content here, make sure IDs match widgets)
    css_content = """
    Screen { layout: vertical; }
    Header { dock: top; height: 1; background: $accent-darken-1; }
    Footer { dock: bottom; height: 1; background: $accent-darken-1; }
    #tabs { dock: top; height: 3; background: $background; padding: 0 1; }
    #tabs Button { width: 1fr; height: 100%; border: none; background: $panel; color: $text-muted; }
    #tabs Button:hover { background: $panel-lighten-1; color: $text; }
    #tabs Button.-active { background: $accent; color: $text; text-style: bold; border: none; }
    #content { height: 1fr; width: 100%; }
    /* Use display: block/none for visibility */
    .window { height: 100%; width: 100%; layout: horizontal; overflow: hidden; display: block; }
    .window.hidden { display: none; } /* Keep hidden class definition if using classes */
    .placeholder-window { align: center middle; background: $panel; }
    /* Sidebar */
    .sidebar { width: 35; background: $boost; padding: 1 2; border-right: thick $background-darken-1; height: 100%; overflow-y: auto; overflow-x: hidden; }
    .sidebar-title { text-style: bold underline; margin-bottom: 1; width: 100%; text-align: center; }
    .sidebar-label { margin-top: 1; text-style: bold; }
    .sidebar-input { width: 100%; margin-bottom: 1; }
    .sidebar-textarea { width: 100%; height: 5; border: round $surface; margin-bottom: 1; }
    .sidebar Select { width: 100%; margin-bottom: 1; }
    #chat-api-key-placeholder, #character-api-key-placeholder { color: $text-muted; text-style: italic; margin-top: 1; }
    /* Chat Log */
    .chat-log { height: 1fr; width: 1fr; border: round $surface; padding: 0 1; margin-bottom: 1; }
    /* Chat Window Layout */
    #chat-main-content { layout: vertical; height: 100%; width: 1fr; }
    #chat-input-area, #character-input-area { height: auto; max-height: 12; width: 100%; align: left top; padding: 1 0 0 0; border-top: round $surface; }
    .chat-input { width: 1fr; height: auto; max-height: 10; border: round $surface; margin: 0 1 0 0; }
    .send-button { width: 10; height: 3; margin: 0; }
    /* Character Window Layout */
    #character-main-content { layout: vertical; height: 100%; width: 1fr; }
    #character-top-area { height: 1fr; width: 100%; layout: horizontal; margin-bottom: 1; }
    #character-top-area > .chat-log { margin: 0 1 0 0; height: 100%; margin-bottom: 0; }
    #character-portrait { width: 25; height: 100%; border: round $surface; padding: 1; margin: 0; overflow: hidden; align: center top; }
    /* Logs Tab */
    #logs-window { padding: 0; border: none; height: 100%; width: 100%; }
    #app-log-display { border: none; height: 1fr; width: 1fr; margin: 0; padding: 1; }
    """

    # --- CSS File Handling ---
    try:
        css_file = Path(TldwCli.CSS_PATH)
        if not css_file.is_file():
             css_file.parent.mkdir(parents=True, exist_ok=True)
             with open(css_file, "w") as f: f.write(css_content)
             logging.info(f"Created default CSS file: {css_file}")
    except Exception as e:
        logging.error(f"Error handling CSS file '{TldwCli.CSS_PATH}': {e}", exc_info=True)

    # --- Run the App ---
    logging.info("Starting Textual App...")
    # Pass the loaded config to the App instance
    app = TldwCli(config=APP_CONFIG)
    app.run()
    logging.info("Textual App finished.")

#
# End of app.py
#######################################################################################################################
