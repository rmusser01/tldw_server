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
from typing import Union, Generator, Any, Optional  # For type hinting

import toml
#
# 3rd-Party Libraries
# --- Textual Imports ---
from textual.app import App, ComposeResult
from textual.logging import TextualHandler
from textual.widgets import (
    Static, Button, Input, Header, Footer, RichLog, TextArea, Select
)
from textual.containers import Horizontal, Container, VerticalScroll
from textual.reactive import reactive
from textual.worker import Worker, WorkerState
from textual.binding import Binding
from textual.dom import DOMNode # For type hinting if needed
from textual.css.query import QueryError # For specific error handling
#
# --- Local API library Imports ---
from .config import get_setting, get_providers_and_models, log, get_log_file_path
from .Widgets.chat_message import ChatMessage
from .Widgets.settings_sidebar import create_settings_sidebar

# Adjust the path based on your project structure
try:
    # Import from the new 'api' directory
    from .api.LLM_API_Calls import (
        chat_with_openai, chat_with_anthropic, chat_with_cohere,
        chat_with_groq, chat_with_openrouter, chat_with_huggingface,
        chat_with_deepseek, chat_with_mistral, chat_with_google,
        )
    from .api.LLM_API_Calls_Local import (
        # Add local API functions if they are in the same file
        chat_with_llama, chat_with_kobold, chat_with_oobabooga,
        chat_with_vllm, chat_with_tabbyapi, chat_with_aphrodite,
        chat_with_ollama, chat_with_custom_openai, chat_with_custom_openai_2
    )
    # You'll need a map for these later, ensure names match
    API_FUNCTION_MAP = {
        "OpenAI": chat_with_openai, "Anthropic": chat_with_anthropic, # etc...
         # Make sure all providers from config have a mapping here or handle None
    }
    API_IMPORTS_SUCCESSFUL = True
    log.info("Successfully imported API functions from .api.llm_api")
except ImportError as e:
    log.error(f"Failed to import API libraries from .api.LLM_API_Calls / .api.LLM_API_Calls_Local: {e}", exc_info=True)
    # Set functions to None so the app doesn't crash later trying to use them
    chat_with_openai = chat_with_anthropic = chat_with_cohere = chat_with_groq = \
    chat_with_openrouter = chat_with_huggingface = chat_with_deepseek = \
    chat_with_mistral = chat_with_google = \
    chat_with_llama = chat_with_kobold = chat_with_oobabooga = chat_with_vllm = \
    chat_with_tabbyapi = chat_with_aphrodite = chat_with_ollama = \
    chat_with_custom_openai = chat_with_custom_openai_2 = None
    API_FUNCTION_MAP = {} # Clear the map on failure
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

# log = logging.getLogger(__name__)
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

    def start_processor(self, app: App): # Keep 'app' param for context if needed elsewhere, but don't use for run_task
        """Starts the log queue processing task using the widget's run_task."""
        if not self._queue_processor_task or self._queue_processor_task.done():
            try:
                # Get the currently running event loop
                loop = asyncio.get_running_loop()
                # Create the task using the standard asyncio function
                self._queue_processor_task = loop.create_task(
                    self._process_log_queue(),
                    name="RichLogProcessor"
                )
                logging.debug("RichLog queue processor task started via asyncio.create_task.")
            except RuntimeError as e:
                # Handle cases where the loop might not be running (shouldn't happen if called from on_mount)
                logging.error(f"Failed to get running loop to start log processor: {e}")
            except Exception as e:
                logging.error(f"Failed to start log processor task: {e}", exc_info=True)

    async def stop_processor(self):
        """Signals the queue processor task to stop and waits for it."""
        # This cancellation logic works for tasks created with asyncio.create_task
        if self._queue_processor_task and not self._queue_processor_task.done():
            logging.debug("Attempting to stop RichLog queue processor task...")
            self._queue_processor_task.cancel()
            try:
                # Wait for the task to acknowledge cancellation
                await self._queue_processor_task
            except asyncio.CancelledError:
                logging.debug("RichLog queue processor task cancelled successfully.")
            except Exception as e:
                # Log errors during cancellation itself
                logging.error(f"Error occurred while awaiting cancelled log processor task: {e}", exc_info=True)
            finally:
                 self._queue_processor_task = None # Ensure it's cleared

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
# --- Main App ---
class TldwCli(App[None]): # Specify return type for run() if needed, None is common
    """A Textual app for interacting with LLMs."""

    # Use forward slashes for paths, works cross-platform
    CSS_PATH = "css/tldw_cli.tcss"
    BINDINGS = [ Binding("ctrl+q", "quit", "Quit App", show=True) ]

    # Define reactive at class level with a placeholder default and type hint
    current_tab: reactive[str] = reactive("chat", layout=True)

    # Add state to hold the currently streaming AI message widget
    current_ai_message_widget: Optional[ChatMessage] = None

    def __init__(self):
        super().__init__()
        # Load config ONCE
        self.app_config = load_config() # Ensure this is called
        self.providers_models = get_providers_and_models() # Ensure this is called
        log.debug(f"__INIT__: Providers and Models loaded in __init__: {self.providers_models}")

        # Determine the *value* for the initial tab but don't set the reactive var yet
        initial_tab_from_config = get_setting("general", "default_tab", "chat")
        if initial_tab_from_config not in ALL_TABS:
            log.warning(f"Default tab '{initial_tab_from_config}' from config not valid. Falling back to 'chat'.")
            self._initial_tab_value = "chat"
        else:
            self._initial_tab_value = initial_tab_from_config

        log.info(f"App __init__: Determined initial tab value: {self._initial_tab_value}")
        self._rich_log_handler: Optional[RichLogHandler] = None # Initialize handler attribute

    def _setup_logging(self):
        """Sets up all logging handlers. Call from on_mount."""
        print("--- _setup_logging START ---")  # Use print for initial debug
        # Configure the root logger FIRST
        root_logger = logging.getLogger()
        initial_log_level_str = self.app_config.get("general", {}).get("log_level", "INFO").upper()
        initial_log_level = getattr(logging, initial_log_level_str, logging.INFO)
        # Set root level - handlers can have higher levels but not lower
        root_logger.setLevel(initial_log_level)
        print(f"Root logger level initially set to: {logging.getLevelName(root_logger.level)}")

        # Clear existing handlers added by basicConfig or previous runs (optional but safer)
        # for handler in root_logger.handlers[:]:
        #     root_logger.removeHandler(handler)
        # print("Cleared existing root logger handlers.")

        # Add TextualHandler for console (replaces basicConfig's default StreamHandler)
        # This integrates better with Textual's console capture.
        textual_console_handler = TextualHandler()
        textual_console_handler.setLevel(initial_log_level)  # Use general log level for console
        console_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        textual_console_handler.setFormatter(console_formatter)
        root_logger.addHandler(textual_console_handler)
        print(f"Added TextualHandler to root logger (Level: {logging.getLevelName(initial_log_level)}).")

        # --- Setup RichLog Handler ---
        try:
            log_display_widget = self.query_one("#app-log-display", RichLog)
            self._rich_log_handler = RichLogHandler(log_display_widget)
            # Set level for RichLog explicitly (e.g., DEBUG to see everything)
            self._rich_log_handler.setLevel(logging.DEBUG)  # Or read from config if needed
            # Formatter is set within RichLogHandler's __init__ now
            root_logger.addHandler(self._rich_log_handler)
            # Processor start moved to after mount completes
            print(f"Added RichLogHandler to root logger (Level: {logging.getLevelName(self._rich_log_handler.level)}).")

        except QueryError:
            print("!!! ERROR: Failed to find #app-log-display widget for RichLogHandler setup.")
            log.error("Failed to find #app-log-display widget for RichLogHandler setup.")
            self._rich_log_handler = None
        except Exception as e:
            print(f"!!! ERROR setting up RichLogHandler: {e}")
            log.exception("Error setting up RichLogHandler")
            self._rich_log_handler = None

        # --- Setup File Logging ---
        try:
            log_file_path = get_log_file_path()  # Get path from config module
            log_dir = log_file_path.parent
            log_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            print(f"Ensured log directory exists: {log_dir}")

            max_bytes = int(get_setting("logging", "log_max_bytes", DEFAULT_CONFIG["logging"]["log_max_bytes"]))
            backup_count = int(
                get_setting("logging", "log_backup_count", DEFAULT_CONFIG["logging"]["log_backup_count"]))
            file_log_level_str = get_setting("logging", "file_log_level", "INFO").upper()
            file_log_level = getattr(logging, file_log_level_str, logging.INFO)

            # Use standard RotatingFileHandler
            file_handler = logging.handlers.RotatingFileHandler(
                log_file_path, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
            )
            file_handler.setLevel(file_log_level)
            file_formatter = logging.Formatter(
                # Use standard %()s placeholders for standard handler
                "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            print(
                f"Added RotatingFileHandler to root logger (File: '{log_file_path}', Level: {logging.getLevelName(file_log_level)}).")

        except Exception as e:
            print(f"!!! ERROR setting up file logging: {e}")
            log.exception("Error setting up file logging")

        # Re-evaluate the lowest level needed for the root logger AFTER adding all handlers
        lowest_level = min(
            initial_log_level,  # Base level set initially
            self._rich_log_handler.level if self._rich_log_handler else logging.CRITICAL,
            file_handler.level if 'file_handler' in locals() else logging.CRITICAL
        )
        root_logger.setLevel(lowest_level)
        print(f"Final Root logger level set to: {logging.getLevelName(root_logger.level)}")
        log.info("Logging setup complete.")  # Now log using the configured system
        print("--- _setup_logging END ---")

    def compose(self) -> ComposeResult:
        log.debug("App composing UI...")
        yield Header()
        yield from self.compose_tabs()
        yield from self.compose_content_area()
        yield Footer()
        log.debug("App compose finished.")

    def compose_tabs(self) -> ComposeResult:
         with Horizontal(id="tabs"):
            for tab_id in ALL_TABS:
                yield Button(
                    tab_id.replace('_', ' ').capitalize(),
                    id=f"tab-{tab_id}",
                    # Initial active state based on the value determined in __init__
                    classes="-active" if tab_id == self._initial_tab_value else ""
                )

    def compose_content_area(self) -> ComposeResult:
        log.debug(f"Compose: Composing content area...")

        with Container(id="content"):
            # --- Chat Window ---
            with Container(id=f"{TAB_CHAT}-window", classes="window"):
                yield from create_settings_sidebar(TAB_CHAT, self.app_config)
                with Container(id="chat-main-content"):
                    # *** Use VerticalScroll for ChatMessages ***
                    yield VerticalScroll(id="chat-log")
                    with Horizontal(id="chat-input-area"):
                        yield TextArea(id="chat-input", classes="chat-input")
                        yield Button("Send", id="send-chat", classes="send-button")

            # --- Character Chat Window ---
            # NOTE: This still uses RichLog. Update if interactive messages needed.
            with Container(id=f"{TAB_CHARACTER}-window", classes="window"):
                 yield from create_settings_sidebar(TAB_CHARACTER, self.app_config)
                 with Container(id="character-main-content"):
                     with Horizontal(id="character-top-area"):
                         yield RichLog(id="character-log", wrap=True, highlight=True, classes="chat-log") # Still RichLog here
                         yield Static(ASCII_PORTRAIT, id="character-portrait") # Use ASCII art
                     with Horizontal(id="character-input-area"):
                         yield TextArea(id="character-input", classes="chat-input")
                         yield Button("Send", id="send-character", classes="send-button")

            # --- Logs Window ---
            with Container(id=f"{TAB_LOGS}-window", classes="window"):
                 yield RichLog(id="app-log-display", wrap=True, highlight=True, markup=True, auto_scroll=True)

            # --- Other Placeholder Windows ---
            for tab_id in ALL_TABS:
                if tab_id not in [TAB_CHAT, TAB_CHARACTER, TAB_LOGS]:
                    with Container(id=f"{tab_id}-window", classes="window placeholder-window"):
                         yield Static(f"{tab_id.replace('_', ' ').capitalize()} Window Placeholder")

    def on_mount(self) -> None:
        """Configure logging, set initial tab visibility, and start processors."""
        # Don't call super().on_mount() if not needed

        # Call the setup function
        self._setup_logging()

        # Start the RichLog processor AFTER mount is complete and event loop is running
        if self._rich_log_handler:
            log.debug("Starting RichLogHandler processor task...")
            self._rich_log_handler.start_processor(self)  # Pass the app instance

        # --- Set Initial Window Visibility ---
        log.debug(f"on_mount: Setting initial window visibility based on tab: {self._initial_tab_value}")
        for tab_id in ALL_TABS:
            try:
                window = self.query_one(f"#{tab_id}-window")
                is_visible = (tab_id == self._initial_tab_value)
                window.display = is_visible
                log.debug(f"  - Window #{tab_id}-window display set to {is_visible}")
            except QueryError:
                log.error(f"on_mount: Could not find window '#{tab_id}-window' to set initial display.")
            except Exception as e:
                log.error(f"on_mount: Error setting display for '#{tab_id}-window': {e}", exc_info=True)

        # *** Set the actual initial tab value AFTER UI is composed and mounted ***
        log.info(f"App on_mount: Setting current_tab reactive value to {self._initial_tab_value}")
        self.current_tab = self._initial_tab_value

        log.info("App mount process completed.")

        async def on_shutdown_request(self, event) -> None:
            log.info("--- App Shutdown Requested ---")
            if self._rich_log_handler:
                await self._rich_log_handler.stop_processor()
                log.info("RichLogHandler processor stopped.")

        # --- Set Initial Window Visibility ---
        log.debug(f"on_mount: Setting initial window visibility based on tab: {self._initial_tab_value}")
        for tab_id in ALL_TABS:
            try:
                window = self.query_one(f"#{tab_id}-window")
                is_visible = (tab_id == self._initial_tab_value)
                window.display = is_visible
                log.debug(f"  - Window #{tab_id}-window display set to {is_visible}")
            except QueryError:
                log.error(f"on_mount: Could not find window '#{tab_id}-window' to set initial display. Check IDs in compose_content_area.")
            except Exception as e:
                log.error(f"on_mount: Error setting display for '#{tab_id}-window': {e}", exc_info=True)

        # *** Set the actual initial tab value AFTER UI is composed and mounted ***
        log.info(f"App on_mount: Setting current_tab reactive value to {self._initial_tab_value}")
        self.current_tab = self._initial_tab_value

        log.info("App mount process completed.")

    async def on_unmount(self) -> None:
        """Clean up logging resources on application exit."""
        log.info("--- App Unmounting ---")
        # Processor should already be stopped by on_shutdown_request if graceful
        # Ensure handlers are removed here regardless
        if self._rich_log_handler:
            logging.getLogger().removeHandler(self._rich_log_handler)
            log.info("RichLogHandler removed.")
        # Find and remove file handler (more robustly)
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                try:
                    handler.close()  # Ensure file is closed
                    root_logger.removeHandler(handler)
                    log.info("File handler removed.")
                except Exception as e:
                    log.error(f"Error removing file handler: {e}")
        logging.shutdown()  # Ensure logs are flushed
        print("--- App Unmounted ---")  # Use print as logging might be shut down


    # WATCHER - Handles UI changes when current_tab's VALUE changes
    def watch_current_tab(self, old_tab: Optional[str], new_tab: str) -> None:
        """Shows/hides the relevant content window when the tab changes."""
        if not isinstance(new_tab, str) or not new_tab:
             log.error(f"Watcher received invalid new_tab value: {new_tab!r}. Aborting tab switch.")
             return
        if old_tab and not isinstance(old_tab, str):
             log.warning(f"Watcher received invalid old_tab value: {old_tab!r}.")
             old_tab = None

        log.debug(f"Watcher: Switching tab from '{old_tab}' to '{new_tab}'")

        if old_tab and old_tab != new_tab:
            try: self.query_one(f"#tab-{old_tab}").remove_class("-active")
            except QueryError: log.warning(f"Watcher: Could not find old button #tab-{old_tab}")
            except Exception as e: log.error(f"Watcher: Error deactivating old button: {e}")

            try: self.query_one(f"#{old_tab}-window").display = False
            except QueryError: log.warning(f"Watcher: Could not find old window #{old_tab}-window")
            except Exception as e: log.error(f"Watcher: Error hiding old window: {e}")

        try: self.query_one(f"#tab-{new_tab}").add_class("-active")
        except QueryError: log.error(f"Watcher: Could not find new button #tab-{new_tab}")
        except Exception as e: log.error(f"Watcher: Error activating new button: {e}")

        try:
            new_window = self.query_one(f"#{new_tab}-window")
            new_window.display = True

            # Focus input on relevant tabs
            if new_tab not in [TAB_LOGS]:
                input_widget: Optional[Union[TextArea, Input]] = None
                try: input_widget = new_window.query_one(TextArea)
                except QueryError:
                    try: input_widget = new_window.query_one(Input)
                    except QueryError: pass # No input found

                if input_widget:
                    def _focus_input():
                        try: input_widget.focus()
                        except Exception as focus_err: log.warning(f"Focus failed: {focus_err}")
                    self.set_timer(0.05, _focus_input)
                    log.debug(f"Watcher: Scheduled focus for input in '{new_tab}'")
                else:
                    log.debug(f"Watcher: No input found to focus in '{new_tab}'")

        except QueryError: log.error(f"Watcher: Could not find new window #{new_tab}-window")
        except Exception as e: log.error(f"Watcher: Error showing new window: {e}", exc_info=True)

        # --- Event Handlers ---
        def on_select_changed(self, event: Select.Changed) -> None:
            select_id = event.control.id
            new_value = str(event.value) if event.value is not None else ""
            log.debug(f"--- on_select_changed START --- ID='{select_id}', Value='{new_value}'")  # Use log now

            if select_id and select_id.endswith("-api-provider"):
                id_prefix = select_id.removesuffix("-api-provider")
                new_provider = new_value  # Already stringified
                log.info(f"Provider Select changed for '{id_prefix}'. New provider: '{new_provider}'")

                model_select_id = f"#{id_prefix}-api-model"
                log.debug(f"Attempting to query model select: {model_select_id}")

                try:
                    model_select = self.query_one(model_select_id, Select)
                    log.debug(f"Found model select widget: {model_select}")
                except QueryError as e:
                    log.error(f"QueryError finding model select '{model_select_id}': {e}", exc_info=True)
                    return
                except Exception as e:
                    log.error(f"Unexpected error querying model select '{model_select_id}': {e}", exc_info=True)
                    return

                # Log the source data
                log.debug(f"Using self.providers_models keys: {list(self.providers_models.keys())}")
                log.debug(f"Looking up models for provider key: '{new_provider}'")  # The exact key being used

                # Get models, ensuring case sensitivity is handled if necessary
                # The .get() handles missing keys gracefully.
                models = self.providers_models.get(new_provider, [])
                log.debug(f"Models retrieved: {models}")

                new_model_options = [(model, model) for model in models]
                log.debug(f"New model options generated: {new_model_options}")

                log.debug(f"Calling set_options on {model_select_id}...")
                try:
                    model_select.set_options(new_model_options)
                    log.debug(f"Finished set_options.")
                except Exception as e:
                    log.error(f"Error during set_options: {e}", exc_info=True)
                    # Optionally clear options or set a placeholder on error
                    model_select.set_options([])
                    model_select.prompt = "Error loading models"
                    return  # Stop further processing if options failed

                # Logic to set the value (seems okay, but log it)
                if models:
                    config_defaults = self.app_config.get(f"{id_prefix}_defaults", {})
                    config_default_model = config_defaults.get("model")
                    model_to_set = models[0]  # Default to first available
                    # Check if the default provider matches and the default model is valid for this provider
                    if config_defaults.get("provider") == new_provider and config_default_model in models:
                        model_to_set = config_default_model
                        log.debug(f"Using config default model: '{model_to_set}'")
                    else:
                        log.debug(f"Using first available model: '{model_to_set}'")

                    log.debug(f"Setting value of {model_select_id} to: '{model_to_set}'")
                    # --- This might be the crucial part ---
                    # Ensure the value being set is actually in the new options list
                    if model_to_set not in [opt[1] for opt in new_model_options]:
                        log.warning(
                            f"Model '{model_to_set}' not found in new options {new_model_options}. Falling back to first option or None.")
                        model_to_set = models[0] if models else None  # Fallback again

                    model_select.value = model_to_set  # Set the value AFTER set_options
                    model_select.prompt = "Select Model..."  # Reset prompt
                    log.debug(f"Model select value after setting: {model_select.value}")
                else:
                    log.debug(f"No models for '{new_provider}'. Clearing value.")
                    model_select.value = None  # Use None for empty value
                    model_select.prompt = "No models available" if new_provider else "Select Provider first"

                # Optional: force a refresh if updates seem inconsistent
                # model_select.refresh()
                # log.debug(f"Refreshed {model_select_id}")

            else:
                log.debug(f"Ignoring Select.Changed event from non-provider select: {select_id or 'UNKNOWN'}")
            log.debug(f"--- on_select_changed END --- ID='{select_id}'")

        async def on_button_pressed(self, event: Button.Pressed) -> None:
            """Handle button presses for tabs, sending messages, and message actions."""
            button = event.button
            button_id = button.id
            log.debug(f"Button pressed: {button_id}, Classes: {button.classes}")

            # --- Tab Switching ---
            if button_id and button_id.startswith("tab-"):
                new_tab_id = button_id.replace("tab-", "")
                log.info(f"Tab button {button_id} pressed. Requesting switch to '{new_tab_id}'")
                if new_tab_id != self.current_tab:
                    self.current_tab = new_tab_id
                else:
                    log.debug(f"Already on tab '{new_tab_id}'. Ignoring.")
                return  # Finished handling tab button

            # --- Send Message ---
            if button_id and button_id.startswith("send-"):
                chat_id_part = button_id.replace("send-", "")
                prefix = chat_id_part
                log.info(f"'Send' button pressed for '{chat_id_part}'")

                # --- Query Widgets ---
                try:
                    text_area = self.query_one(f"#{prefix}-input", TextArea)
                    chat_container = self.query_one(f"#{prefix}-log", VerticalScroll)  # Changed to VerticalScroll
                    provider_widget = self.query_one(f"#{prefix}-api-provider", Select)
                    model_widget = self.query_one(f"#{prefix}-api-model", Select)
                    system_prompt_widget = self.query_one(f"#{prefix}-system-prompt", TextArea)
                    temp_widget = self.query_one(f"#{prefix}-temperature", Input)
                    top_p_widget = self.query_one(f"#{prefix}-top-p", Input)
                    min_p_widget = self.query_one(f"#{prefix}-min-p", Input)
                    top_k_widget = self.query_one(f"#{prefix}-top-k", Input)
                except QueryError as e:
                    log.error(f"Send Button: Could not find UI widgets for '{prefix}': {e}")
                    return
                except Exception as e:
                    log.error(f"Send Button: Unexpected error querying widgets for '{prefix}': {e}")
                    return

                # --- Get Values ---
                message = text_area.text.strip()
                selected_provider = str(provider_widget.value) if provider_widget.value else None
                selected_model = str(model_widget.value) if model_widget.value else None
                system_prompt = system_prompt_widget.text
                temperature = self._safe_float(temp_widget.value, 0.7, "temperature")
                top_p = self._safe_float(top_p_widget.value, 0.95, "top_p")
                min_p = self._safe_float(min_p_widget.value, 0.05, "min_p")
                top_k = self._safe_int(top_k_widget.value, 50, "top_k")

                # --- Basic Validation ---
                if not API_IMPORTS_SUCCESSFUL:
                    # Maybe mount an error message?
                    await chat_container.mount(
                        ChatMessage("API libraries failed load. Cannot send.", role="AI", classes="-error"))
                    log.error(f"Send attempt ('{prefix}') failed: API libraries not loaded.")
                    return
                if not message: log.debug(
                    f"Empty message submitted in '{prefix}'."); text_area.clear(); text_area.focus(); return
                if not selected_provider: await chat_container.mount(
                    ChatMessage("Select API Provider.", role="AI", classes="-error")); return
                if not selected_model: await chat_container.mount(
                    ChatMessage("Select Model.", role="AI", classes="-error")); return

                # --- Mount User Message ---
                user_msg_widget = ChatMessage(message, role="User")
                await chat_container.mount(user_msg_widget)
                chat_container.scroll_end(animate=True)
                text_area.clear()
                text_area.focus()

                # --- Prepare and Dispatch API Call ---
                api_function = API_FUNCTION_MAP.get(selected_provider)
                if not api_function:
                    await chat_container.mount(
                        ChatMessage(f"Error: API backend for {selected_provider} not available.", role="AI",
                                    classes="-error"))
                    log.error(f"API function for provider {selected_provider} not found or loaded.")
                    return

                # Get API URL (if needed for local models)
                api_url = self._get_api_name(selected_provider, self.app_config.get("api_endpoints", {}))

                # TODO: Build chat history for context
                # For now, just send the current message
                input_for_api = message

                # TODO: Get API key securely (from env var or config, avoid passing directly if possible)
                # Placeholder: Assume key is retrieved somehow if needed by the function
                api_key_for_call = os.environ.get(f"{selected_provider.upper()}_API_KEY")  # Example

                # Determine streaming preference (example, adjust as needed)
                # should_stream = get_setting("providers", f"{selected_provider}_streaming", False)
                should_stream = False  # Default to non-streaming for simplicity initially

                # Prepare arguments dictionary based on what the specific API function expects
                # This needs careful mapping based on your api/llm_api.py functions
                api_args = {
                    "api_key": api_key_for_call,
                    "input_data": input_for_api,
                    "model": selected_model,
                    "custom_prompt_arg": "",  # Maybe combine prompt+input here? Adjust API funcs
                    "system_message": system_prompt,
                    "temp": temperature,
                    "streaming": should_stream,
                    "topp": top_p,  # Map UI value to expected param name
                    "top_k": top_k,
                    "minp": min_p,
                    # Add other relevant params like api_url if needed by local functions
                    "api_url": api_url if selected_provider in ["Ollama", "Llama.cpp", "Oobabooga", "KoboldCpp",
                                                                "vLLM"] else None,
                }

                # Remove None values unless specifically required by the API func
                filtered_api_call_args = {k: v for k, v in api_args.items() if v is not None}
                loggable_args = {k: v for k, v in filtered_api_call_args.items() if k != 'api_key'}
                func_name = getattr(api_function, '__name__', 'UNKNOWN_FUNCTION')
                log.debug(f"Calling {func_name} with filtered args: {loggable_args}")

                # --- Mount Placeholder AI Message ---
                ai_placeholder_widget = ChatMessage(message="AI thinking...", role="AI", generation_complete=False)
                await chat_container.mount(ai_placeholder_widget)
                chat_container.scroll_end(animate=False)
                self.current_ai_message_widget = ai_placeholder_widget

                # --- Define Worker Wrapper ---
                current_api_func = api_function
                current_api_args = filtered_api_call_args  # Use filtered args
                current_placeholder = ai_placeholder_widget

                def api_call_wrapper() -> Union[str, Generator[Any, Any, None], None]:
                    log.debug(f"Worker wrapper executing for {prefix}")
                    # Pass the filtered args to the worker
                    return self._api_worker(current_api_func, current_api_args, current_placeholder)

                # --- Run Worker ---
                log.debug(f"Running worker API_Call_{prefix}")
                self.run_worker(api_call_wrapper, name=f"API_Call_{prefix}", group="api_calls", thread=True)
                return  # Finished handling send button

            # --- Handle Action Buttons inside ChatMessage ---
            button_classes = button.classes
            action_widget: Optional[ChatMessage] = None
            node: Optional[DOMNode] = button
            while node is not None:
                if isinstance(node, ChatMessage): action_widget = node; break
                node = node.parent

            if action_widget:
                message_text = action_widget.message_text
                message_role = action_widget.role

                if "edit-button" in button_classes:
                    log.info(f"Action: Edit clicked for {message_role} message: '{message_text[:50]}...'")
                    try:
                        # Query for Static specifically
                        text_widget = action_widget.query_one(".message-text", Static)
                        text_widget.update(f"[EDITING...] {message_text}")
                    except QueryError:
                        log.error("Could not find .message-text Static widget for editing.")

                elif "copy-button" in button_classes:
                    log.info(f"Action: Copy clicked for {message_role} message: '{message_text[:50]}...'")
                    try:
                        self.app.set_clipboard(message_text)
                        log.info("Message copied to clipboard.")
                        button.label = "✅Copied"
                        self.set_timer(1.5, lambda: setattr(button, 'label', '📋'))
                    except Exception as e:
                        log.error(f"Clipboard action failed: {e}")


                elif "speak-button" in button_classes:
                    log.info(f"Action: Speak clicked for {message_role} message: '{message_text[:50]}...'")
                    try:
                        # Query for Static specifically
                        text_widget = action_widget.query_one(".message-text", Static)
                        text_widget.update(f"[SPEAKING...] {message_text}")
                    except QueryError:
                        log.error("Could not find .message-text Static widget for speaking placeholder.")

                elif "thumb-up-button" in button_classes:
                    log.info(f"Action: Thumb Up clicked for {message_role} message.")
                    button.label = "👍(OK)"  # Provide visual feedback

                elif "thumb-down-button" in button_classes:
                    log.info(f"Action: Thumb Down clicked for {message_role} message.")
                    button.label = "👎(OK)"  # Provide visual feedback


                elif "regenerate-button" in button_classes and message_role == "AI":
                    log.info(f"Action: Regenerate clicked for AI message.")
                    try:
                        text_widget = action_widget.query_one(".message-text", Static)
                        text_widget.update("[REGENERATING...]")
                    except QueryError:
                        log.error("Could not find .message-text Static widget for regenerating placeholder.")
                    except Exception as e:
                        log.error(f"Error updating regenerate placeholder: {e}")
                    # FIXME - Implement actual regeneration (find previous user message, call API worker again)
            else:
                # This handles buttons not inside a ChatMessage or unhandled IDs
                if not button_id.startswith("tab-"):  # Avoid logging tab clicks as warnings
                    log.warning(f"Button pressed with unhandled ID or context: {button_id}")

        # --- Worker function ---
        def _api_worker(self, api_func: callable, api_args: dict, placeholder_widget: ChatMessage) -> Union[
            str, Generator[Any, Any, None], None]:
            """Executes the API call in a thread."""
            func_name = getattr(api_func, '__name__', 'UNKNOWN_FUNCTION')
            try:
                log.info(
                    f"Worker {func_name} starting execution with args: {{k: v for k, v in api_args.items() if k != 'api_key'}}")
                # *** Call the actual API function with the filtered arguments ***
                result = api_func(**api_args)
                log.info(f"Worker {func_name} finished successfully.")
                return result
            except Exception as e:
                log.exception(f"Error during API call in worker ({func_name}): {e}")
                return f"[bold red]API Error ({func_name}):[/] {str(e)}"

        # --- Handle worker completion ---
        def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
            """Handles results from the API worker."""
            worker_name = event.worker.name or "Unknown Worker"
            log.debug(f"Worker '{worker_name}' state changed to: {event.state}")

            if not worker_name.startswith("API_Call_"): return

            prefix = worker_name.replace("API_Call_", "")
            ai_message_widget = self.current_ai_message_widget  # Use stored reference

            if not ai_message_widget:
                log.error(f"Worker finished for {prefix}, but no current_ai_message_widget found!")
                return

            try:
                chat_container = self.query_one(f"#{prefix}-log", VerticalScroll)  # Get container to scroll

                if event.state == WorkerState.SUCCESS:
                    result = event.worker.result
                    is_streaming = isinstance(result, Generator)

                    # Clear "AI thinking..." only if it hasn't been updated yet
                    if ai_message_widget.message_text == "AI thinking...":
                        ai_message_widget.message_text = ""
                        ai_message_widget.query_one(".message-text").update("")

                    if is_streaming:
                        log.info(f"API call ({prefix}) successful (Streaming started).")

                        async def process_stream():  # Use async task to process generator
                            full_response = ""
                            try:
                                async for chunk in result:  # Iterate async if API func returns async gen
                                    if isinstance(chunk, str):
                                        ai_message_widget.update_message_chunk(chunk)
                                        full_response += chunk
                                        chat_container.scroll_end(animate=False,
                                                                  duration=0.05)  # Scroll as text arrives
                                    else:
                                        log.warning(f"Received non-string chunk: {type(chunk)}")
                                        ai_message_widget.update_message_chunk(str(chunk))
                                        full_response += str(chunk)
                                # Streaming finished successfully
                                ai_message_widget.mark_generation_complete()
                                log.info(f"API call ({prefix}) streaming finished. Length: {len(full_response)}")
                            except Exception as gen_e:
                                log.error(f"Error processing stream for '{prefix}': {gen_e}", exc_info=True)
                                ai_message_widget.query_one(".message-text").update(
                                    ai_message_widget.message_text + "\n[bold red] Error during streaming.[/]")
                                ai_message_widget.mark_generation_complete()
                            finally:
                                self.current_ai_message_widget = None  # Clear reference once stream ends/errors
                                try:
                                    self.query_one(f"#{prefix}-input", TextArea).focus()
                                except Exception:
                                    pass

                        self.run_task(process_stream(), exclusive=True,
                                      group=f"stream_{prefix}")  # Run stream processing as Textual task

                    else:  # Non-streaming
                        if isinstance(result, str):
                            if result.startswith("[bold red]API Error"):
                                log.error(f"API call ({prefix}) worker returned error: {result}")
                                ai_message_widget.message_text = result
                                ai_message_widget.query_one(".message-text").update(result)
                            else:
                                log.info(f"API call ({prefix}) successful. Length: {len(result)}")
                                ai_message_widget.message_text = result
                                ai_message_widget.query_one(".message-text").update(result)
                        elif result is None:
                            log.error(f"API worker '{worker_name}' returned None.")
                            err_msg = "[bold red]AI: Error - No response received.[/]"
                            ai_message_widget.message_text = err_msg
                            ai_message_widget.query_one(".message-text").update(err_msg)
                        else:
                            log.error(f"Unexpected result type: {type(result)}")
                            err_msg = "[bold red]Error: Unexpected result type.[/]"
                            ai_message_widget.message_text = err_msg
                            ai_message_widget.query_one(".message-text").update(err_msg)

                        ai_message_widget.mark_generation_complete()  # Show buttons
                        self.current_ai_message_widget = None  # Clear reference
                        chat_container.scroll_end(animate=True)  # Scroll after non-streaming result
                        try:
                            self.query_one(f"#{prefix}-input", TextArea).focus()
                        except Exception:
                            pass

                elif event.state == WorkerState.ERROR:
                    log.error(f"Worker '{worker_name}' failed critically:", exc_info=event.worker.error)
                    err_msg = "[bold red]AI Error: Processing failed. Check logs.[/]"
                    ai_message_widget.message_text = err_msg
                    ai_message_widget.query_one(".message-text").update(err_msg)
                    ai_message_widget.mark_generation_complete()
                    self.current_ai_message_widget = None  # Clear reference
                    chat_container.scroll_end(animate=True)
                    try:
                        self.query_one(f"#{prefix}-input", TextArea).focus()
                    except Exception:
                        pass

            except QueryError:
                log.error(f"Failed to find log widget '#{prefix}-log' for worker '{worker_name}' completion.")
                self.current_ai_message_widget = None  # Clear reference even if UI fails
            except Exception as e:
                log.error(f"Error in on_worker_state_changed for worker '{worker_name}': {e}", exc_info=True)
                if ai_message_widget:
                    try:
                        ai_message_widget.query_one(".message-text").update(
                            "[bold red]Internal error handling response.[/]")
                        ai_message_widget.mark_generation_complete()
                    except Exception:
                        pass
                self.current_ai_message_widget = None

        # --- Helper methods ---
        def _safe_float(self, value: str, default: float, name: str) -> float:
            if not value: return default
            try:
                return float(value)
            except ValueError:
                log.warning(f"Invalid {name} '{value}', using {default}"); return default

        def _safe_int(self, value: str, default: int, name: str) -> int:
            if not value: return default
            try:
                return int(value)
            except ValueError:
                log.warning(f"Invalid {name} '{value}', using {default}"); return default

        def _get_api_name(self, provider: str, endpoints: dict) -> Optional[str]:
            # Map provider names (case-insensitive keys from config/UI) to endpoint keys in config.toml
            # Ensure these keys match your config.toml [api_endpoints] section
            provider_key_map = {
                "Ollama": "Ollama",  # Assuming key in config is "Ollama"
                "Llama.cpp": "Llama_cpp",
                "Oobabooga": "Oobabooga",
                "KoboldCpp": "KoboldCpp",
                "vLLM": "vLLM",
                "Custom": "Custom",
                "Custom-2": "Custom_2",
                # Add other mappings if needed (TabbyAPI, Aphrodite?)
            }
            endpoint_key = provider_key_map.get(provider)  # Case-sensitive lookup based on UI value
            if endpoint_key:
                url = endpoints.get(endpoint_key)  # Case-sensitive lookup in config dict
                if url:
                    log.debug(f"Using API endpoint '{url}' for provider '{provider}' (key: '{endpoint_key}')")
                    return url
                else:
                    log.warning(
                        f"URL key '{endpoint_key}' for provider '{provider}' missing in config [api_endpoints].")
                    return None
            else:
                # Cloud providers or those not needing a specific URL here
                log.debug(f"No specific endpoint URL key configured for provider '{provider}'.")
                return None


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
    app = TldwCli()
    app.run()
    logging.info("Textual App finished.")

#
# End of app.py
#######################################################################################################################
