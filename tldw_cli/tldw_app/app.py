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
from rich.text import Text
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
from .Chat.Chat_Functions import chat
from .config import get_setting, get_providers_and_models, get_log_file_path, DEFAULT_CONFIG, load_config, \
    DEFAULT_CONFIG_PATH, CONFIG_TOML_CONTENT
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
    logging.info("Successfully imported API functions from .api.llm_api")
except ImportError as e:
    logging.error(f"Failed to import API libraries from .api.LLM_API_Calls / .api.LLM_API_Calls_Local: {e}", exc_info=True)
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
# Statics


# Make sure these imports succeed first!
if API_IMPORTS_SUCCESSFUL:
    API_FUNCTION_MAP = {
        "OpenAI": chat_with_openai,
        "Anthropic": chat_with_anthropic,
        "Cohere": chat_with_cohere,
        "Groq": chat_with_groq,
        "OpenRouter": chat_with_openrouter,
        "HuggingFace": chat_with_huggingface,
        "DeepSeek": chat_with_deepseek,
        "MistralAI": chat_with_mistral, # Key from config
        "Google": chat_with_google,
        "Llama_cpp": chat_with_llama,    # Key from config
        "KoboldCpp": chat_with_kobold,   # Key from config
        "Oobabooga": chat_with_oobabooga,# Key from config
        "vLLM": chat_with_vllm,       # Key from config
        "TabbyAPI": chat_with_tabbyapi,  # Key from config
        "Aphrodite": chat_with_aphrodite,# Key from config
        "Ollama": chat_with_ollama,     # Key from config
        "Custom": chat_with_custom_openai, # Key from config
        "Custom_2": chat_with_custom_openai_2, # Key from config
        # "local-llm": chat_with_local_llm # Add if this is a distinct provider in config
    }
    logging.info(f"API_FUNCTION_MAP populated with {len(API_FUNCTION_MAP)} entries.")
else:
    API_FUNCTION_MAP = {}
    logging.error("API_FUNCTION_MAP is empty due to import failures.")

#
#
#####################################################################################################################
#
# Functions:

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

    # --- ADD REACTIVES FOR PROVIDER SELECTS ---
    # Initialize with a dummy value or fetch default from config here
    # Ensure the initial value matches what's set in compose/settings_sidebar
    # Fetching default provider from config:
    _default_chat_provider = APP_CONFIG.get("chat_defaults", {}).get("provider", "Ollama")
    _default_character_provider = APP_CONFIG.get("character_defaults", {}).get("provider", "Anthropic")

    chat_api_provider_value: reactive[Optional[str]] = reactive(_default_chat_provider)
    character_api_provider_value: reactive[Optional[str]] = reactive(_default_character_provider)
    # --- END ADD REACTIVES ---


    def __init__(self):
        super().__init__()
        # Load config ONCE
        self.app_config = load_config() # Ensure this is called
        logging.debug("__INIT__: Attempting to get providers and models...")
        try:
            # Call the function from the config module
            self.providers_models = get_providers_and_models()
            # *** ADD THIS LOGGING ***
            logging.info(
                f"__INIT__: Successfully retrieved providers_models. Count: {len(self.providers_models)}. Keys: {list(self.providers_models.keys())}")
        except Exception as e:
            logging.error(f"__INIT__: Failed to get providers and models: {e}", exc_info=True)
            self.providers_models = {}  # Set empty on error
        # Determine the *value* for the initial tab but don't set the reactive var yet
        initial_tab_from_config = get_setting("general", "default_tab", "chat")
        if initial_tab_from_config not in ALL_TABS:
            logging.warning(f"Default tab '{initial_tab_from_config}' from config not valid. Falling back to 'chat'.")
            self._initial_tab_value = "chat"
        else:
            self._initial_tab_value = initial_tab_from_config

        logging.info(f"App __init__: Determined initial tab value: {self._initial_tab_value}")
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

        # --- Add TextualHandler ---
        # Check if one already exists to prevent duplicates if setup runs multiple times
        has_textual_handler = any(isinstance(h, TextualHandler) for h in root_logger.handlers)
        if not has_textual_handler:
            textual_console_handler = TextualHandler()
            textual_console_handler.setLevel(initial_log_level)
            console_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            textual_console_handler.setFormatter(console_formatter)
            root_logger.addHandler(textual_console_handler)
            print(f"Added TextualHandler to root logger (Level: {logging.getLevelName(initial_log_level)}).")
        else:
            print("TextualHandler already exists on root logger.")

        # --- Setup RichLog Handler ---
        try:
            log_display_widget = self.query_one("#app-log-display", RichLog)
            # Prevent adding multiple RichLog Handlers
            if self._rich_log_handler and self._rich_log_handler in root_logger.handlers:
                 print("RichLogHandler already exists and is added.")
            elif not self._rich_log_handler:
                self._rich_log_handler = RichLogHandler(log_display_widget)
                self._rich_log_handler.setLevel(logging.DEBUG) # Set level explicitly
                root_logger.addHandler(self._rich_log_handler)
                print(f"Added RichLogHandler to root logger (Level: {logging.getLevelName(self._rich_log_handler.level)}).")
            else:
                 # Handler exists but wasn't added? Add it.
                 root_logger.addHandler(self._rich_log_handler)
                 print(f"Re-added existing RichLogHandler instance to root logger.")

        except QueryError:
            print("!!! ERROR: Failed to find #app-log-display widget for RichLogHandler setup.")
            logging.error("Failed to find #app-log-display widget for RichLogHandler setup.")
            self._rich_log_handler = None
        except Exception as e:
            print(f"!!! ERROR setting up RichLogHandler: {e}")
            logging.exception("Error setting up RichLogHandler")
            self._rich_log_handler = None

        # --- Setup File Logging ---
        try:
            log_file_path = get_log_file_path()  # Get path from config module
            log_dir = log_file_path.parent
            log_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            print(f"Ensured log directory exists: {log_dir}")

            # Prevent adding multiple File Handlers
            has_file_handler = any(isinstance(h, logging.handlers.RotatingFileHandler) and h.baseFilename == str(log_file_path) for h in root_logger.handlers)

            if not has_file_handler:
                max_bytes = int(get_setting("logging", "log_max_bytes", DEFAULT_CONFIG["logging"]["log_max_bytes"]))
                backup_count = int(get_setting("logging", "log_backup_count", DEFAULT_CONFIG["logging"]["log_backup_count"]))
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
            logging.exception("Error setting up file logging")

        # Re-evaluate lowest level
        all_handlers = root_logger.handlers
        if all_handlers:
             lowest_level = min(h.level for h in all_handlers if h.level > 0) # Ignore level 0 handlers
             root_logger.setLevel(lowest_level)
             print(f"Final Root logger level set to: {logging.getLevelName(root_logger.level)}")
        else:
             print(f"No handlers found on root logger after setup!")
        logging.info("Logging setup complete.")
        print("--- _setup_logging END ---")

    def compose(self) -> ComposeResult:
        logging.debug("App composing UI...")
        yield Header()
        yield from self.compose_tabs()
        yield from self.compose_content_area()
        yield Footer()
        logging.debug("App compose finished.")

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
        logging.debug(f"Compose: Composing content area...")

        with Container(id="content"):
            # --- Chat Window ---
            # Assign specific reactive variables to the Select widgets
            with Container(id=f"{TAB_CHAT}-window", classes="window"):
                yield from create_settings_sidebar(TAB_CHAT, self.app_config)

                with Container(id="chat-main-content"):
                    # *** Use VerticalScroll for ChatMessages ***
                    yield VerticalScroll(id="chat-log")
                    with Horizontal(id="chat-input-area"):
                        yield TextArea(id="chat-input", classes="chat-input")
                        yield Button("Send", id="send-chat", classes="send-button")

            # --- Character Chat Window ---
            # NOTE: This still uses Richlogging. Update if interactive messages needed.
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

    # --- Add explicit methods to update reactives from Select changes ---
    def update_chat_provider_reactive(self, new_value: Optional[str]) -> None:
        """Called when the chat provider Select value changes internally."""
        print(f">>> DEBUG: update_chat_provider_reactive called with: {new_value!r}")
        self.chat_api_provider_value = new_value

    def update_character_provider_reactive(self, new_value: Optional[str]) -> None:
        """Called when the character provider Select value changes internally."""
        print(f">>> DEBUG: update_character_provider_reactive called with: {new_value!r}")
        self.character_api_provider_value = new_value
    # --- END Add explicit methods ---



    def on_mount(self) -> None:
        """Configure logging, set initial tab, bind selects, and start processors."""
        self._setup_logging()

        if self._rich_log_handler:
            logging.debug("Starting RichLogHandler processor task...")
            self._rich_log_handler.start_processor(self)

        # --- REMOVE THIS BLOCK ---
        # print(f">>> DEBUG: on_mount: Setting initial window visibility for tab: {self._initial_tab_value}")
        # logging.debug(f"on_mount: Setting initial window visibility based on tab: {self._initial_tab_value}")
        # for tab_id in ALL_TABS:
        #     try:
        #         window = self.query_one(f"#{tab_id}-window")
        #         is_visible = (tab_id == self._initial_tab_value)
        #         print(f">>> DEBUG: Setting #{tab_id}-window display to {is_visible}")
        #         window.display = is_visible # <<<--- REMOVE THIS LOOP
        #         logging.debug(f"  - Window #{tab_id}-window display set to {is_visible}")
        #     except QueryError:
        #         print(f">>> DEBUG: ERROR - Could not find window #{tab_id}-window in on_mount")
        #         logging.error(f"on_mount: Could not find window '#{tab_id}-window' to set initial display.")
        #     except Exception as e:
        #         print(f">>> DEBUG: ERROR - Setting display for #{tab_id}-window: {e}")
        #         logging.error(f"on_mount: Error setting display for '#{tab_id}-window': {e}", exc_info=True)
        # --- END REMOVED BLOCK ---

        # --- Bind Select Widgets ---
        logging.info("App on_mount: Binding Select widgets to reactive updaters...")
        # (Keep your existing binding logic with self.watch here)
        try:
            chat_select = self.query_one(f"#{TAB_CHAT}-api-provider", Select)
            self.watch(chat_select, "value", self.update_chat_provider_reactive, init=False)
            logging.debug(f"Bound chat provider Select ({chat_select.id}) value to update_chat_provider_reactive")
            print(f">>> DEBUG: Bound chat provider Select to reactive update method.")
        except QueryError:
            logging.error(f"on_mount: Failed to find chat provider select: #{TAB_CHAT}-api-provider")
            print(f">>> DEBUG: ERROR - Failed to bind chat provider select.")
        except Exception as e:
            logging.error(f"on_mount: Error binding chat provider select: {e}", exc_info=True)
            print(f">>> DEBUG: ERROR - Exception during chat provider select binding: {e}")

        try:
            char_select = self.query_one(f"#{TAB_CHARACTER}-api-provider", Select)
            self.watch(char_select, "value", self.update_character_provider_reactive, init=False)
            logging.debug(f"Bound character provider Select ({char_select.id}) value to update_character_provider_reactive")
            print(f">>> DEBUG: Bound character provider Select to reactive update method.")
        except QueryError:
            logging.error(f"on_mount: Failed to find character provider select: #{TAB_CHARACTER}-api-provider")
            print(f">>> DEBUG: ERROR - Failed to bind character provider select.")
        except Exception as e:
            logging.error(f"on_mount: Error binding character provider select: {e}", exc_info=True)
            print(f">>> DEBUG: ERROR - Exception during character provider select binding: {e}")
        # --- END BINDING LOGIC ---

        # --- Set initial reactive tab value ---
        # This MUST be done AFTER the UI exists and AFTER bindings are set up (if they depend on it)
        # Crucially, this will trigger watch_current_tab to set the initial visibility.
        print(f">>> DEBUG: on_mount: Setting self.current_tab = {self._initial_tab_value} (will trigger watcher)")
        logging.info(f"App on_mount: Setting current_tab reactive value to {self._initial_tab_value} to trigger initial view.")
        self.current_tab = self._initial_tab_value

        logging.info("App mount process completed.")

    async def on_shutdown_request(self, event) -> None:
        """Stop background tasks before shutdown."""
        logging.info("--- App Shutdown Requested ---")
        if self._rich_log_handler:
            await self._rich_log_handler.stop_processor()
            logging.info("RichLogHandler processor stopped.")

    async def on_unmount(self) -> None:
        """Clean up logging resources on application exit."""
        logging.info("--- App Unmounting ---")
        # Processor should already be stopped by on_shutdown_request if graceful
        # Ensure handlers are removed here regardless
        if self._rich_log_handler:
            logging.getLogger().removeHandler(self._rich_log_handler)
            logging.info("RichLogHandler removed.")
        # Find and remove file handler (more robustly)
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                try:
                    handler.close()  # Ensure file is closed
                    root_logger.removeHandler(handler)
                    logging.info("File handler removed.")
                except Exception as e:
                    logging.error(f"Error removing file handler: {e}")
        logging.shutdown()  # Ensure logs are flushed
        print("--- App Unmounted ---")  # Use print as logging might be shut down


    # WATCHER - Handles UI changes when current_tab's VALUE changes
    def watch_current_tab(self, old_tab: Optional[str], new_tab: str) -> None:
        """Shows/hides the relevant content window when the tab changes."""
        # (Your existing watcher code is likely fine, just ensure the QueryErrors aren't hiding a problem)
        print(f"\n>>> DEBUG: watch_current_tab triggered! Old: '{old_tab}', New: '{new_tab}'")
        if not isinstance(new_tab, str) or not new_tab:
             print(f">>> DEBUG: watch_current_tab: Invalid new_tab '{new_tab!r}', aborting.")
             logging.error(f"Watcher received invalid new_tab value: {new_tab!r}. Aborting tab switch.")
             return
        if old_tab and not isinstance(old_tab, str):
             print(f">>> DEBUG: watch_current_tab: Invalid old_tab '{old_tab!r}', setting to None.")
             logging.warning(f"Watcher received invalid old_tab value: {old_tab!r}.")
             old_tab = None

        logging.debug(f"Watcher: Switching tab from '{old_tab}' to '{new_tab}'")

        # --- Hide Old Tab ---
        if old_tab and old_tab != new_tab:
            try:
                old_button = self.query_one(f"#tab-{old_tab}")
                old_button.remove_class("-active")
                print(f">>> DEBUG: Deactivated button #tab-{old_tab}")
            except QueryError:
                print(f">>> DEBUG: Could not find old button #tab-{old_tab}")
                logging.warning(f"Watcher: Could not find old button #tab-{old_tab}")
            except Exception as e:
                logging.error(f"Watcher: Error deactivating old button: {e}", exc_info=True)

            try:
                old_window = self.query_one(f"#{old_tab}-window")
                old_window.display = False # Set style directly
                print(f">>> DEBUG: Set display=False for window #{old_tab}-window")
            except QueryError:
                print(f">>> DEBUG: Could not find old window #{old_tab}-window")
                logging.warning(f"Watcher: Could not find old window #{old_tab}-window")
            except Exception as e:
                logging.error(f"Watcher: Error hiding old window: {e}", exc_info=True)

        # --- Show New Tab ---
        try:
            new_button = self.query_one(f"#tab-{new_tab}")
            new_button.add_class("-active")
            print(f">>> DEBUG: Activated button #tab-{new_tab}")
        except QueryError:
            print(f">>> DEBUG: Could not find new button #tab-{new_tab}")
            logging.error(f"Watcher: Could not find new button #tab-{new_tab}")
        except Exception as e:
            logging.error(f"Watcher: Error activating new button: {e}", exc_info=True)

        try:
            new_window = self.query_one(f"#{new_tab}-window")
            new_window.display = True # Set style directly
            print(f">>> DEBUG: Set display=True for window #{new_tab}-window")

            # Focus input (your existing logic here is fine)
            if new_tab not in [TAB_LOGS]:
                input_widget: Optional[Union[TextArea, Input]] = None
                try: input_widget = new_window.query_one(TextArea)
                except QueryError:
                    try: input_widget = new_window.query_one(Input)
                    except QueryError: pass

                if input_widget:
                    def _focus_input():
                        try: input_widget.focus()
                        except Exception as focus_err: logging.warning(f"Focus failed: {focus_err}")
                    # Slightly longer delay might sometimes help if layout is complex
                    self.set_timer(0.1, _focus_input)
                    logging.debug(f"Watcher: Scheduled focus for input in '{new_tab}'")
                else:
                    logging.debug(f"Watcher: No input found to focus in '{new_tab}'")

        except QueryError:
            print(f">>> DEBUG: Could not find new window #{new_tab}-window")
            logging.error(f"Watcher: Could not find new window #{new_tab}-window")
        except Exception as e:
            print(f">>> DEBUG: Error showing new window #{new_tab}-window: {e}")
            logging.error(f"Watcher: Error showing new window: {e}", exc_info=True)

        print(">>> DEBUG: watch_current_tab finished.")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses for tabs, sending messages, and message actions."""
        button = event.button
        button_id = button.id
        print(f"\n>>> DEBUG: on_button_pressed called! Button ID: {event.button.id}\n")
        logging.debug(f"Button pressed: {button_id}, Classes: {button.classes}")

        # --- Tab Switching --- (Keep this part as is)
        if button_id and button_id.startswith("tab-"):
            print(f">>> DEBUG: Tab button detected: {button_id}")
            new_tab_id = button_id.replace("tab-", "")
            logging.info(f"Tab button {button_id} pressed. Requesting switch to '{new_tab_id}'")
            if new_tab_id != self.current_tab:
                print(f">>> DEBUG: Changing current_tab from '{self.current_tab}' to '{new_tab_id}'")
                self.current_tab = new_tab_id
            else:
                print(f">>> DEBUG: Already on tab '{new_tab_id}'. Ignoring.")
                logging.debug(f"Already on tab '{new_tab_id}'. Ignoring.")
            return

        # --- Send Message ---
        if button_id and button_id.startswith("send-"):
            chat_id_part = button_id.replace("send-", "")
            prefix = chat_id_part
            logging.info(f"Send button pressed for '{chat_id_part}'")

            # --- Query Widgets ---
            try:
                text_area = self.query_one(f"#{prefix}-input", TextArea)
                chat_container = self.query_one(f"#{prefix}-log", VerticalScroll)
                provider_widget = self.query_one(f"#{prefix}-api-provider", Select)
                model_widget = self.query_one(f"#{prefix}-api-model", Select)
                system_prompt_widget = self.query_one(f"#{prefix}-system-prompt", TextArea)
                temp_widget = self.query_one(f"#{prefix}-temperature", Input)
                top_p_widget = self.query_one(f"#{prefix}-top-p", Input)
                min_p_widget = self.query_one(f"#{prefix}-min-p", Input)
                top_k_widget = self.query_one(f"#{prefix}-top-k", Input)
                # Add query for custom prompt if you have one
                # custom_prompt_widget = self.query_one(f"#{prefix}-custom-prompt", TextArea) # Example
            except QueryError as e:
                logging.error(f"Send Button: Could not find UI widgets for '{prefix}': {e}")
                # Optionally mount an error message in the chat_container
                await chat_container.mount(
                    ChatMessage(f"Internal Error: Missing UI elements for {prefix}.", role="AI", classes="-error"))
                return
            except Exception as e:
                logging.error(f"Send Button: Unexpected error querying widgets for '{prefix}': {e}")
                await chat_container.mount(
                    ChatMessage("Internal Error: Could not query UI elements.", role="AI", classes="-error"))
                return

            # --- Get Values ---
            message = text_area.text.strip()
            selected_provider = str(provider_widget.value) if provider_widget.value else None
            selected_model = str(model_widget.value) if model_widget.value else None
            system_prompt = system_prompt_widget.text
            temperature = self._safe_float(temp_widget.value, 0.7, "temperature")
            top_p = self._safe_float(top_p_widget.value, 0.95, "top_p")  # UI value for top_p
            min_p = self._safe_float(min_p_widget.value, 0.05, "min_p")  # UI value for min_p
            top_k = self._safe_int(top_k_widget.value, 50, "top_k")  # UI value for top_k
            custom_prompt = ""  # Fetch from UI if needed: custom_prompt_widget.text
            # Determine streaming based on config or UI element if you add one
            # Placeholder: Use False for now
            should_stream = False

            # --- Basic Validation ---
            if not message: logging.debug(
                f"Empty message submitted in '{prefix}'."); text_area.clear(); text_area.focus(); return
            if not selected_provider: await chat_container.mount(
                ChatMessage("Please select an API Provider.", role="AI", classes="-error")); return
            if not selected_model: await chat_container.mount(
                ChatMessage("Please select a Model.", role="AI", classes="-error")); return
            # Check if API functions loaded (check the flag)
            if not API_IMPORTS_SUCCESSFUL:
                await chat_container.mount(
                    ChatMessage("Error: Core API functions failed to load. Cannot send message.", role="AI",
                                classes="-error"))
                logging.error("Attempted to send message, but API imports failed.")
                return

            # --- Build History ---
            chat_history = []
            try:
                # Iterate through existing ChatMessage widgets in the container
                message_widgets = chat_container.query(ChatMessage)
                user_msg = None
                for msg_widget in message_widgets:
                    if msg_widget.role == "User":
                        user_msg = msg_widget.message_text  # Store user message
                    elif msg_widget.role == "AI" and user_msg is not None:
                        # Pair the last user message with this AI response
                        chat_history.append((user_msg, msg_widget.message_text))
                        user_msg = None  # Reset user message
                logging.debug(f"Built chat history with {len(chat_history)} turns.")
            except Exception as e:
                logging.error(f"Failed to build chat history: {e}", exc_info=True)
                # Decide whether to proceed without history or show an error
                await chat_container.mount(
                    ChatMessage("Internal Error: Could not retrieve chat history.", role="AI", classes="-error"))
                return

            # --- Mount User Message ---
            user_msg_widget = ChatMessage(message, role="User")
            await chat_container.mount(user_msg_widget)
            chat_container.scroll_end(animate=True)
            text_area.clear()
            text_area.focus()

            # --- Prepare and Dispatch API Call via Worker calling chat_wrapper ---
            # Fetch API key (adjust based on your actual key management)
            api_key_for_call = None
            config_key_found = False
            env_key_found = False

            # 1. Get provider-specific settings from the loaded config
            provider_settings_key = selected_provider.lower() # e.g., "openai", "anthropic"
            # Access the already loaded config (self.app_config or load_config() again if needed)
            # Assuming self.app_config holds the merged config dictionary
            provider_settings = self.app_config.get("api_settings", {}).get(provider_settings_key, {})

            if provider_settings:
                # 2. Check for hardcoded API key in config FIRST
                config_api_key = provider_settings.get("api_key", "").strip()
                if config_api_key:
                    api_key_for_call = config_api_key
                    config_key_found = True
                    logging.debug(f"Using API key for '{selected_provider}' from config file [api_settings.{provider_settings_key}].api_key.")
                else:
                    # 3. If no config key, check environment variable specified in config
                    env_var_name = provider_settings.get("api_key_env_var", "").strip()
                    if env_var_name:
                        env_api_key = os.environ.get(env_var_name, "").strip()
                        if env_api_key:
                            api_key_for_call = env_api_key
                            env_key_found = True
                            logging.debug(f"Using API key for '{selected_provider}' from environment variable '{env_var_name}'.")
                        else:
                            logging.debug(f"Environment variable '{env_var_name}' for '{selected_provider}' not found or empty.")
                    else:
                        logging.debug(f"No 'api_key_env_var' specified for '{selected_provider}' in config.")
            else:
                logging.warning(f"No [api_settings.{provider_settings_key}] section found in config for '{selected_provider}'. Cannot check for configured API key or ENV variable name.")


            # 4. Handle case where no key was found (neither config nor ENV)
            if not api_key_for_call:
                logging.warning(f"API Key for '{selected_provider}' not found in config file or specified environment variable.")
                # Define known cloud providers requiring keys
                providers_requiring_key = ["OpenAI", "Anthropic", "Google", "MistralAI", "Groq",
                                           "Cohere", "OpenRouter", "HuggingFace", "DeepSeek"] # Add any others
                # Check if the selected provider requires a key
                if selected_provider in providers_requiring_key:
                    logging.error(f"API call aborted: API Key for required provider '{selected_provider}' is missing.")
                    await chat_container.mount(
                        ChatMessage(f"API Key for {selected_provider} is missing.\n\n"
                                    f"Please add it to your config file under:\n"
                                    f"[api_settings.{provider_settings_key}]\n"
                                    f"api_key = \"YOUR_KEY\"\n\n"
                                    f"Or set the environment variable specified in 'api_key_env_var'.",
                                    role="AI", classes="-error"))
                    # Remove the placeholder "thinking" message if it exists
                    if self.current_ai_message_widget and self.current_ai_message_widget.is_mounted:
                         await self.current_ai_message_widget.remove()
                         self.current_ai_message_widget = None
                    return # Stop processing
                else:
                    # Assume it's a local model or one not needing a key
                    logging.info(f"Proceeding without API key for provider '{selected_provider}' (assumed local/no key required).")
                    api_key_for_call = None # Explicitly ensure it's None

            # --- Mount Placeholder AI Message ---
            ai_placeholder_widget = ChatMessage(message="AI thinking...", role="AI", generation_complete=False)
            await chat_container.mount(ai_placeholder_widget)
            chat_container.scroll_end(animate=False)
            self.current_ai_message_widget = ai_placeholder_widget  # Store reference

            # --- Define Worker Target using chat_wrapper ---
            # Use a lambda to capture current values and pass them to the wrapper
            # Note: We pass the UI parameter values directly. chat_wrapper calls chat(), which calls chat_api_call(),
            # and chat_api_call handles the mapping (e.g., top_p vs maxp vs topp).
            worker_target = lambda: self.chat_wrapper(
                message=message,
                history=chat_history,
                # --- Pass other necessary params ---
                api_endpoint=selected_provider,
                api_key=api_key_for_call,
                custom_prompt=custom_prompt,  # Pass the UI custom prompt
                temperature=temperature,
                system_message=system_prompt,
                streaming=should_stream,
                minp=min_p,  # Pass min_p from UI
                # maxp=top_p,    # Pass top_p from UI as maxp if chat_wrapper/chat expects it
                model=selected_model,
                topp=top_p,  # Pass top_p from UI as topp
                topk=top_k,  # Pass top_k from UI
                # Add placeholders for unused chat() params for now
                media_content={},
                selected_parts=[],
                chatdict_entries=None,
                max_tokens=500,  # Default or get from config/UI
                strategy="sorted_evenly"  # Default or get from config/UI
            )

            # --- Run Worker ---
            logging.debug(f"Running worker 'API_Call_{prefix}' to execute chat_wrapper")
            self.run_worker(
                worker_target,
                name=f"API_Call_{prefix}",
                group="api_calls",
                thread=True,  # Run in a separate thread
                description=f"Calling {selected_provider}"
            )
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
                logging.info(f"Action: Edit clicked for {message_role} message: '{message_text[:50]}...'")
                try:
                    # Query for Static specifically
                    text_widget = action_widget.query_one(".message-text", Static)
                    text_widget.update(Text(f"[EDITING...] {message_text}"))
                except QueryError:
                    logging.error("Could not find .message-text Static widget for editing.")

            elif "copy-button" in button_classes:
                logging.info(
                    "Action: Copy clicked for %s message: '%s…'",
                    message_role,
                    message_text[:50],
                )

                copied = False
                try:
                    # ── Textual ≥ 0.57 ────────────────────────────────────────────────
                    if hasattr(self, "copy_to_clipboard"):
                        self.copy_to_clipboard(message_text)
                        copied = True

                    # ── Early Textual builds (local clipboard only) ──────────────────
                    elif hasattr(self, "set_clipboard"):  # legacy helper
                        self.set_clipboard(message_text)
                        copied = True

                    # ── Fallback to pyperclip (optional dependency) ──────────────────
                    else:
                        import importlib.util
                        if importlib.util.find_spec("pyperclip"):
                            import pyperclip  # type: ignore
                            pyperclip.copy(message_text)
                            copied = True
                        else:
                            logging.warning("pyperclip not available; clipboard copy skipped.")

                except Exception as exc:  # noqa: BLE001
                    logging.error("Clipboard action failed: %s", exc, exc_info=True)

                # ── user feedback ─────────────────────────────────────────────────────
                if copied:
                    button.label = "✅Copied"
                    self.set_timer(1.5, lambda: setattr(button, "label", "📋"))

            elif "speak-button" in button_classes:
                logging.info(f"Action: Speak clicked for {message_role} message: '{message_text[:50]}...'")
                try:
                    # Query for Static specifically
                    text_widget = action_widget.query_one(".message-text", Static)
                    text_widget.update(Text(f"[SPEAKING...] {message_text}"))
                except QueryError:
                    logging.error("Could not find .message-text Static widget for speaking placeholder.")

            elif "thumb-up-button" in button_classes:
                logging.info(f"Action: Thumb Up clicked for {message_role} message.")
                button.label = "👍(OK)"  # Provide visual feedback

            elif "thumb-down-button" in button_classes:
                logging.info(f"Action: Thumb Down clicked for {message_role} message.")
                button.label = "👎(OK)"  # Provide visual feedback

            elif "delete-button" in button_classes:
                logging.info(
                    "Action: Delete clicked for %s message: '%s…'",
                    message_role,
                    message_text[:50],
                )
                try:
                    await action_widget.remove()  # removes the ChatMessage from the DOM
                    # If this was the AI placeholder being deleted, clear the reference
                    if action_widget is self.current_ai_message_widget:
                        self.current_ai_message_widget = None
                except Exception as exc:  # noqa: BLE001
                    logging.error("Failed to delete message: %s", exc, exc_info=True)

            elif "regenerate-button" in button_classes and message_role == "AI":
                logging.info(f"Action: Regenerate clicked for AI message.")
                try:
                    text_widget = action_widget.query_one(".message-text", Static)
                    text_widget.update(Text("[REGENERATING...]"))
                except QueryError:
                    logging.error("Could not find .message-text Static widget for regenerating placeholder.")
                except Exception as e:
                    logging.error(f"Error updating regenerate placeholder: {e}")
                # FIXME - Implement actual regeneration (find previous user message, call API worker again)
        else:
            # This handles buttons not inside a ChatMessage or unhandled IDs
            if not button_id.startswith("tab-"):  # Avoid logging tab clicks as warnings
                print(f">>> DEBUG: Button '{button_id}' didn't match tab or send prefixes.")
                logging.warning(f"Button pressed with unhandled ID or context: {button_id}")

    def chat_wrapper(self, message, history, api_endpoint, api_key, custom_prompt, temperature,
                     system_message, streaming, minp, model, topp, topk,
                     # Removed maxp if chat() doesn't use it directly
                     media_content, selected_parts, chatdict_entries, max_tokens, strategy):
        """
        This method runs in the worker thread and calls the main chat logic.
        It receives parameters captured by the lambda in on_button_pressed.
        """
        logging.debug(f"chat_wrapper executing for endpoint '{api_endpoint}'")
        try:
            # Call the imported chat function from Chat_Functions.py
            result = chat(
                message=message,
                history=history,
                media_content=media_content,
                selected_parts=selected_parts,
                api_endpoint=api_endpoint,
                api_key=api_key,
                custom_prompt=custom_prompt,  # Pass custom_prompt from UI
                temperature=temperature,
                system_message=system_message,
                streaming=streaming,
                minp=minp,
                # maxp=maxp, # Pass maxp if chat() needs it, otherwise remove
                model=model,
                topp=topp,  # Pass topp from UI
                topk=topk,  # Pass topk from UI
                chatdict_entries=chatdict_entries,
                max_tokens=max_tokens,
                strategy=strategy
            )
            logging.debug(f"chat_wrapper finished for '{api_endpoint}'. Result type: {type(result)}")
            return result
        except Exception as e:
            logging.exception(f"Error inside chat_wrapper for endpoint {api_endpoint}: {e}")
            # Return a formatted error string that on_worker_state_changed can display
            return f"[bold red]Error during chat processing:[/]\n{str(e)}"

    # --- Handle worker completion ---
    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle completion / failure of a background API worker."""
        worker_name = event.worker.name or "Unknown Worker"
        logging.debug("Worker '%s' state changed to %s", worker_name, event.state)

        if not worker_name.startswith("API_Call_"):
            return

        prefix = worker_name.replace("API_Call_", "")
        ai_message_widget = self.current_ai_message_widget  # previously stored

        if ai_message_widget is None:  # sanity-check
            logging.error("Worker finished for %s but AI placeholder is missing!", prefix)
            return

        try:
            chat_container: VerticalScroll = self.query_one(f"#{prefix}-log", VerticalScroll)

            # ────────────────────────────── SUCCESS ───────────────────────────────
            if event.state is WorkerState.SUCCESS:
                result = event.worker.result
                streaming = isinstance(result, Generator)

                # clear the “thinking…” placeholder once
                if ai_message_widget.message_text == "AI thinking...":
                    ai_message_widget.message_text = ""
                    ai_message_widget.query_one(".message-text", Static).update("")

                # ────────────── streaming response (async generator) ──────────────
                if streaming:
                    logging.info("API call (%s) returned a generator – streaming.", prefix)

                    async def process_stream() -> None:
                        full = ""
                        try:
                            async for chunk in result:  # type: ignore[misc]
                                text = str(chunk)
                                ai_message_widget.update_message_chunk(text)
                                full += text
                                chat_container.scroll_end(animate=False, duration=0.05)
                            ai_message_widget.mark_generation_complete()
                            logging.info("Stream finished (%d chars).", len(full))
                        except Exception as exc:  # noqa: BLE001
                            logging.exception("Stream failure (%s): %s", prefix, exc)
                            ai_message_widget.query_one(".message-text", Static).update(
                                f"{ai_message_widget.message_text}\n[bold red] Error during stream.[/]"
                            )
                            ai_message_widget.mark_generation_complete()
                        finally:
                            self.current_ai_message_widget = None
                            try:
                                self.query_one(f"#{prefix}-input", TextArea).focus()
                            except QueryError:
                                pass

                    # schedule the coroutine in Textual’s worker pool
                    self.run_worker(
                        process_stream,  # pass the coroutine *callable*
                        name=f"stream_{prefix}",
                        group="streams",
                        exclusive=True,
                    )

                # ──────────────── non-streaming (plain string / None) ─────────────
                else:
                    target = ai_message_widget.query_one(".message-text", Static)

                    if isinstance(result, str):
                        if result.startswith("[bold red]API Error"):
                            logging.error("API call (%s) returned an error.", prefix)
                        ai_message_widget.message_text = result
                        target.update(result)

                    elif result is None:
                        err = "[bold red]AI: Error – No response received.[/]"
                        ai_message_widget.message_text = err
                        target.update(err)

                    else:
                        err = "[bold red]Error: Unexpected result type.[/]"
                        logging.error("Unexpected result type: %r", type(result))
                        ai_message_widget.message_text = err
                        target.update(err)

                    ai_message_widget.mark_generation_complete()
                    self.current_ai_message_widget = None
                    chat_container.scroll_end(animate=True)
                    try:
                        self.query_one(f"#{prefix}-input", TextArea).focus()
                    except QueryError:
                        pass

            # ─────────────────────────────── ERROR ───────────────────────────────
            elif event.state is WorkerState.ERROR:
                err = "[bold red]AI Error: Processing failed. Check logs.[/]"
                logging.error("Worker '%s' failed.", worker_name, exc_info=event.worker.error)
                ai_message_widget.message_text = err
                ai_message_widget.query_one(".message-text", Static).update(err)
                ai_message_widget.mark_generation_complete()
                self.current_ai_message_widget = None
                chat_container.scroll_end(animate=True)
                try:
                    self.query_one(f"#{prefix}-input", TextArea).focus()
                except QueryError:
                    pass

        except QueryError:
            logging.error("Cannot find '#%s-log' for worker '%s'.", prefix, worker_name)
            self.current_ai_message_widget = None
        except Exception as exc:  # noqa: BLE001
            logging.exception("on_worker_state_changed – unexpected: %s", exc)
            ai_message_widget.query_one(".message-text", Static).update(
                "[bold red]Internal error handling response.[/]"
            )
            ai_message_widget.mark_generation_complete()
            self.current_ai_message_widget = None

    # --- Helper methods ---
    def _safe_float(self, value: str, default: float, name: str) -> float:
        if not value: return default
        try:
            return float(value)
        except ValueError:
            logging.warning(f"Invalid {name} '{value}', using {default}"); return default

    def _safe_int(self, value: str, default: int, name: str) -> int:
        if not value: return default
        try:
            return int(value)
        except ValueError:
            logging.warning(f"Invalid {name} '{value}', using {default}"); return default

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
                logging.debug(f"Using API endpoint '{url}' for provider '{provider}' (key: '{endpoint_key}')")
                return url
            else:
                logging.warning(
                    f"URL key '{endpoint_key}' for provider '{provider}' missing in config [api_endpoints].")
                return None
        else:
            # Cloud providers or those not needing a specific URL here
            logging.debug(f"No specific endpoint URL key configured for provider '{provider}'.")
            return None

    def watch_chat_api_provider_value(self, new_value: Optional[str]) -> None:
        """Watcher triggered when the value of the #chat-api-provider Select changes."""
        # Use print for immediate console feedback
        print(f"\n--- WATCHER TRIGGERED ---")
        print(f"Widget ID: chat-api-provider")
        print(f"New Value Observed: {new_value!r}")
        if new_value is None or new_value == Select.BLANK:
             print(f"Watcher: Value is blank or None. Clearing model options.")
             self._update_model_select("chat", []) # Pass empty list
             return

        # --- Re-introduce the model update logic here ---
        print(f"Watcher: Updating models for provider '{new_value}'")
        print(f"Watcher: Available Provider Keys: {list(self.providers_models.keys())}")
        models = self.providers_models.get(new_value, []) # Get models for the new provider key
        print(f"Watcher: Models retrieved: {models}")
        self._update_model_select("chat", models) # Call helper
        print(f"--- WATCHER END (chat-api-provider) ---\n")

    def watch_character_api_provider_value(self, new_value: Optional[str]) -> None:
        """Watcher triggered when the value of the #character-api-provider Select changes."""
        # Use print for immediate console feedback
        print(f"\n--- WATCHER TRIGGERED ---")
        print(f"Widget ID: character-api-provider")
        print(f"New Value Observed: {new_value!r}")
        if new_value is None or new_value == Select.BLANK:
            print(f"Watcher: Value is blank or None. Clearing model options.")
            self._update_model_select("character", [])
            return

        # --- Re-introduce the model update logic here ---
        print(f"Watcher: Updating models for provider '{new_value}'")
        print(f"Watcher: Available Provider Keys: {list(self.providers_models.keys())}")
        models = self.providers_models.get(new_value, [])
        print(f"Watcher: Models retrieved: {models}")
        self._update_model_select("character", models)
        print(f"--- WATCHER END (character-api-provider) ---\n")

    # --- ADD HELPER METHOD to update model select ---
    def _update_model_select(self, id_prefix: str, models: list[str]) -> None:
        """Helper function to update the model Select options and value."""
        model_select_id = f"#{id_prefix}-api-model"
        print(f"Helper _update_model_select: Targeting '{model_select_id}' with models: {models}")
        try:
            model_select = self.query_one(model_select_id, Select)
        except QueryError:
            print(f"Helper ERROR: Cannot find model select '{model_select_id}'")
            logging.error(f"Helper ERROR: Cannot find model select '{model_select_id}'")
            return

        new_model_options = [(model, model) for model in models]
        print(f"Helper: Setting model options: {new_model_options}")
        try:
            model_select.set_options(new_model_options)
        except Exception as e:
            print(f"Helper ERROR setting options: {e}")
            logging.error(f"Helper ERROR setting options for {model_select_id}: {e}")
            model_select.set_options([]) # Clear on error
            model_select.prompt = "Error"
            model_select.value = None
            return

        # Set value (e.g., to first model or a default)
        model_to_set = None
        if models:
            # You could try and get the default from config here if needed
            # default_model_for_provider = get_setting("api_settings", f"{new_value}.model") # Example
            model_to_set = models[0] # Simple default: first in list
            print(f"Helper: Setting value to first model: {model_to_set!r}")
        else:
            print(f"Helper: No models, clearing value.")

        try:
             # Explicitly set the value *after* setting options
             # This might be necessary if set_options clears the value
            model_select.value = model_to_set
            print(f">>> HELPER: Model select value AFTER explicit set: {model_select.value!r}")
        except Exception as e:
             print(f">>> HELPER ERROR setting value: {e}")
             logging.error(f"Helper ERROR setting value for {model_select_id}: {e}")

        model_select.prompt = "Select Model..." if models else "No models available"
        print(f"Helper: Model select value after update: {model_select.value!r}")


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

/* Base style for ALL windows. The watcher will set display: True/False */
.window {
    height: 100%;
    width: 100%;
    layout: horizontal; /* Or vertical if needed by default */
    overflow: hidden;
}

/* REMOVED .hidden class */

.placeholder-window { align: center middle; background: $panel; }

/* Sidebar Styling */
.sidebar { width: 35; background: $boost; padding: 1 2; border-right: thick $background-darken-1; height: 100%; overflow-y: auto; overflow-x: hidden; }
.sidebar-title { text-style: bold underline; margin-bottom: 1; width: 100%; text-align: center; }
.sidebar-label { margin-top: 1; text-style: bold; }
.sidebar-input { width: 100%; margin-bottom: 1; }
.sidebar-textarea { width: 100%; height: 5; border: round $surface; margin-bottom: 1; }
.sidebar Select { width: 100%; margin-bottom: 1; }

/* --- Chat Window specific layouts --- */
#chat-main-content {
    layout: vertical;
    height: 100%;
    width: 1fr;
}
/* VerticalScroll for chat messages */
#chat-log {
    height: 1fr; /* Takes remaining space */
    width: 100%;
    /* border: round $surface; Optional: Add border to scroll area */
    padding: 0 1; /* Padding around messages */
}

/* Input area styling (shared by chat and character) */
#chat-input-area, #character-input-area {
    height: auto;    /* Allow height to adjust */
    max-height: 12;  /* Limit growth */
    width: 100%;
    align: left top; /* Align children to top-left */
    padding: 1; /* Consistent padding */
    border-top: round $surface;
}
/* Input widget styling (shared) */
.chat-input { /* Targets TextArea */
    width: 1fr;
    height: auto;      /* Allow height to adjust */
    max-height: 100%;  /* Don't overflow parent */
    margin-right: 1; /* Space before button */
    border: round $surface;
}
/* Send button styling (shared) */
.send-button { /* Targets Button */
    width: 10;
    height: 3; /* Fixed height for consistency */
    /* align-self: stretch; REMOVED */
    margin-top: 0;
}

/* --- Character Chat Window specific layouts --- */
#character-main-content {
    layout: vertical;
    height: 100%;
    width: 1fr;
}
#character-top-area {
    height: 1fr; /* Top area takes remaining vertical space */
    width: 100%;
    layout: horizontal;
    margin-bottom: 1;
}
/* Log when next to portrait (Still RichLog here) */
#character-top-area > #character-log { /* Target by ID is safer */
    margin: 0 1 0 0;
    height: 100%;
    margin-bottom: 0; /* Override base margin */
    border: round $surface; /* Added border back for RichLog */
    padding: 0 1; /* Added padding back for RichLog */
    width: 1fr; /* Ensure it takes space */
}
/* Portrait styling */
#character-portrait {
    width: 25;
    height: 100%;
    border: round $surface;
    padding: 1;
    margin: 0;
    overflow: hidden;
    align: center top;
}

/* Logs Window */
#logs-window { padding: 0; border: none; height: 100%; width: 100%; }
#app-log-display { border: none; height: 1fr; width: 1fr; margin: 0; padding: 1; }

/* --- ChatMessage Styling --- */
ChatMessage {
    width: 100%;
    height: auto;
    margin-bottom: 1;
}
ChatMessage > Vertical {
    border: round $surface;
    background: $panel;
    padding: 0 1;
    width: 100%;
    height: auto;
}
ChatMessage.-user > Vertical {
    background: $boost; /* Different background for user */
    border: round $accent;
}
.message-header {
    width: 100%;
    padding: 0 1;
    background: $surface-darken-1;
    text-style: bold;
    height: 1; /* Ensure header is minimal height */
}
.message-text {
    padding: 1; /* Padding around the text itself */
    width: 100%;
    height: auto;
}
.message-actions {
    height: auto;
    width: 100%;
    padding: 1; /* Add padding around buttons */
    /* Use a VALID border type */
    border-top: solid $surface-lighten-1; /* CHANGED thin to solid */
    align: right middle; /* Align buttons to the right */
    display: block; /* Default display state */
}
.message-actions Button {
    min-width: 8;
    height: 1;
    margin: 0 0 0 1; /* Space between buttons */
    border: none;
    background: $surface-lighten-2;
    color: $text-muted;
}
.message-actions Button:hover {
    background: $surface;
    color: $text;
}
/* Initially hide AI actions until generation is complete */
ChatMessage.-ai .message-actions.-generating {
    display: none;
}
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
    print("--- INSTANTIATING TldwCli ---")
    logging.info("--- INSTANTIATING TldwCli ---")
    app = TldwCli()
    print("--- INSTANTIATED TldwCli ---")
    logging.info("--- INSTANTIATED TldwCli ---")
    print("--- CALLING app.run() ---")
    logging.info("--- CALLING app.run() ---")
    try:
        app.run()
    except Exception as e:
         print(f"--- CRITICAL ERROR DURING app.run() ---")
         logging.exception("--- CRITICAL ERROR DURING app.run() ---")
         traceback.print_exc() # Make sure traceback prints
    finally:
         # This might run even if app exits early internally in run()
         print("--- FINALLY block after app.run() ---")
         logging.info("--- FINALLY block after app.run() ---")

    print("--- AFTER app.run() call (if not crashed hard) ---")
    logging.info("--- AFTER app.run() call (if not crashed hard) ---")

#
# End of app.py
#######################################################################################################################
