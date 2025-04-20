import asyncio
import logging  # Standard logging library
import logging.handlers  # For handlers
from pathlib import Path
import traceback
import os # Needed if API keys rely on environment variables
from typing import Union, Generator, Any # For type hinting

# --- Textual Imports ---
from textual.app import App, ComposeResult, RenderResult
from textual.widgets import (
    Static, Button, Input, Header, Footer, RichLog, TextArea, Select
)
from textual.containers import Horizontal, Container, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.message import Message
from textual.worker import Worker, WorkerState

# --- Import your API libraries ---
# Adjust the path based on your project structure
try:
    # Assuming libs is a subdirectory
    from tldw_cli.Libs.LLM_API_Calls import (
        chat_with_openai, chat_with_anthropic, chat_with_cohere,
        chat_with_groq, chat_with_openrouter, chat_with_huggingface,
        chat_with_deepseek, chat_with_mistral, chat_with_google
    )
    from tldw_cli.Libs.LLM_API_Calls_Local import (
        chat_with_local_llm, chat_with_llama, chat_with_kobold,
        chat_with_oobabooga, chat_with_vllm, chat_with_tabbyapi,
        chat_with_aphrodite, chat_with_ollama,
        chat_with_custom_openai, chat_with_custom_openai_2 # Assuming these exist
    )
    # Ensure the config loader is available if needed directly
    # from tldw_Server_API.app.core.Utils.Utils import load_and_log_configs
except ImportError as e:
    logging.error(f"Failed to import API libraries: {e}")
    # Set ALL potentially imported functions to None to prevent NameErrors later
    chat_with_openai = None
    chat_with_anthropic = None
    chat_with_cohere = None
    chat_with_groq = None
    chat_with_openrouter = None
    chat_with_huggingface = None
    chat_with_deepseek = None
    chat_with_mistral = None
    chat_with_google = None
    chat_with_local_llm = None
    chat_with_llama = None
    chat_with_kobold = None
    chat_with_oobabooga = None
    chat_with_vllm = None
    chat_with_tabbyapi = None
    chat_with_aphrodite = None
    chat_with_ollama = None
    chat_with_custom_openai = None
    chat_with_custom_openai_2 = None
    print("-" * 60)
    print("ERROR: Could not import API library functions.")
    print("API calling functionality will be disabled.")
    print(f"Import Error Details: {e}")
    print("-" * 60)

# --- Constants ---
TAB_CHAT = "chat"
TAB_CHARACTER = "character"
TAB_MEDIA = "media"
TAB_SEARCH = "search"
TAB_INGEST = "ingest"
TAB_LOGS = "logs"
TAB_STATS = "stats"

ALL_TABS = [
    TAB_CHAT, TAB_CHARACTER, TAB_MEDIA, TAB_SEARCH,
    TAB_INGEST, TAB_LOGS, TAB_STATS
]

# --- Define the BASE Cloud Providers and their models ---
# This is the dictionary that was missing
API_MODELS_BY_PROVIDER = {
    "OpenAI": [
        "gpt-4o", "gpt-4-turbo", "gpt-4-1106-preview", "gpt-4-vision-preview",
        "gpt-4", "gpt-4-32k", "gpt-3.5-turbo-1106", "gpt-3.5-turbo", "gpt-3.5-turbo-16k",
    ],
    "Anthropic": [
        "claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307", "claude-2.1", "claude-2.0", "claude-instant-1.2",
    ],
    "Google": [
        "gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "gemini-pro", "gemini-pro-vision",
    ],
    "MistralAI": [
        "mistral-large-latest", "mistral-small-latest", "open-mixtral-8x7b", "open-mistral-7b",
    ],
     # Meta models usually run locally, maybe list under Ollama/Llama.cpp instead?
    # "Meta": ["llama3-70b-8192", "llama3-8b-8192"],
    "Custom": ["custom-model-alpha", "custom-model-beta", "legacy-model-v1"] # Keep if needed
}

# --- Define Local Providers and potentially others mapped to specific functions ---
LOCAL_PROVIDERS = {
    # Local Servers
    "Llama.cpp": ["llama-model-1", "llama-model-2"], # Add actual model names if known
    "Oobabooga": ["ooba-model-a", "ooba-model-b"],
    "KoboldCpp": ["kobold-model-x"],
    "Ollama": ["ollama/llama3:latest", "ollama/mistral:latest", "ollama/codellama:latest"], # Use names Ollama expects
    "vLLM": ["vllm-model-z"], # Add actual model names
    "TabbyAPI": ["tabby-model"],
    "Aphrodite": ["aphrodite-engine"],
    "Custom-2": ["custom-model-gamma"], # Example for a second custom entry

    # Cloud Services mapped to specific functions
    "Groq": ["llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
    "Cohere": ["command-r-plus", "command-r", "command", "command-light"],
    "OpenRouter": ["microsoft/wizardlm-2-8x22b", "google/gemini-flash-1.5", "meta-llama/llama-3-70b-instruct"],
    "HuggingFace": ["CohereForAI/c4ai-command-r-plus", "mistralai/Mixtral-8x7B-Instruct-v0.1"], # Model names HF uses
    "DeepSeek": ["deepseek-chat", "deepseek-coder"],
}

# --- Merge the dictionaries to create the comprehensive list for the UI ---
# This line now correctly uses the defined API_MODELS_BY_PROVIDER
ALL_API_MODELS = {**API_MODELS_BY_PROVIDER, **LOCAL_PROVIDERS}

# --- Create the list of provider names for the Select widget ---
AVAILABLE_PROVIDERS = list(ALL_API_MODELS.keys())

# --- API Function Map (defined after ALL_API_MODELS) ---
# This maps the *string* selected in the UI (a key from AVAILABLE_PROVIDERS)
# to the actual callable function from your libraries.
API_FUNCTION_MAP = {
    # Cloud Providers (ensure keys match API_MODELS_BY_PROVIDER)
    "OpenAI": chat_with_openai,
    "Anthropic": chat_with_anthropic,
    "Google": chat_with_google,
    "MistralAI": chat_with_mistral,
    # "Meta": chat_with_meta, # Add if you have a specific Meta function
    "Custom": chat_with_custom_openai,

    # Local/Other Providers (ensure keys match LOCAL_PROVIDERS)
    "Llama.cpp": chat_with_llama,
    "Oobabooga": chat_with_oobabooga,
    "KoboldCpp": chat_with_kobold,
    "Ollama": chat_with_ollama,
    "vLLM": chat_with_vllm,
    "TabbyAPI": chat_with_tabbyapi,
    "Aphrodite": chat_with_aphrodite,
    "Custom-2": chat_with_custom_openai_2,
    "Groq": chat_with_groq,
    "Cohere": chat_with_cohere,
    "OpenRouter": chat_with_openrouter,
    "HuggingFace": chat_with_huggingface,
    "DeepSeek": chat_with_deepseek,
}

ASCII_PORTRAIT = r"""
  .--./)
 /.''.')
 | \ '/
 W `-'
 \\    '.
  '.    /
    `~~`
"""

# Custom logging handler that writes to a RichLog widget
class RichLogHandler(logging.Handler):
    def __init__(self, rich_log_widget: RichLog):
        super().__init__()
        self.rich_log_widget = rich_log_widget

    def emit(self, record: logging.LogRecord):
        try:
            message = self.format(record)
            # print(f"    Formatted message: '{message[:100]}...'") # Print snippet
            # print(f"    Target widget: {self.rich_log_widget} (ID: {self.rich_log_widget.id})")
            # print(f"    App instance: {self.rich_log_widget.app}")

            # Check if the widget is still mounted (important check)
            if not self.rich_log_widget.is_mounted:
                # print("    ERROR in emit: RichLog widget is no longer mounted!")
                return

            # Check if app is available
            if not self.rich_log_widget.app:
                # print("    ERROR in emit: App is not available on the widget!")
                return

            # The core call to update the UI thread-safely
            # print("    Calling app.call_soon...")
            self.rich_log_widget.app.call_soon(self.rich_log_widget.write, message)
            # print("    app.call_soon completed.")

        except Exception as e:
            print(f"!!!!!!!! ERROR within RichLogHandler.emit !!!!!!!!!!")
            traceback.print_exc() # Print full exception details to console

# --- Helper Function for Sidebar (use updated AVAILABLE_PROVIDERS) ---
def create_settings_sidebar(id_prefix: str) -> ComposeResult:
    """Yields the widgets for a standard settings sidebar with dependent dropdowns."""
    with VerticalScroll(id=f"{id_prefix}-sidebar", classes="sidebar"):
        yield Static("Settings", classes="sidebar-title")
        yield Static("API Provider", classes="sidebar-label")
        # Use updated provider list
        provider_options = [(provider, provider) for provider in AVAILABLE_PROVIDERS]
        default_provider = AVAILABLE_PROVIDERS[0]
        yield Select(
            options=provider_options, prompt="Select Provider...", allow_blank=False,
            id=f"{id_prefix}-api-provider", value=default_provider
        )
        yield Static("Model", classes="sidebar-label")
        # Use updated model map
        initial_models = ALL_API_MODELS.get(default_provider, [])
        model_options = [(model, model) for model in initial_models]
        yield Select(
            options=model_options, prompt="Select Model...", allow_blank=True,
            id=f"{id_prefix}-api-model",
            value=initial_models[0] if initial_models else None
        )
        yield Static("API Key (Set in config/env)", classes="sidebar-label", id=f"{id_prefix}-api-key-placeholder") # Updated label
        yield Static("System prompt", classes="sidebar-label")
        yield TextArea(id=f"{id_prefix}-system-prompt", classes="sidebar-textarea")
        yield Static("Temperature", classes="sidebar-label")
        yield Input(placeholder="e.g., 0.7", id=f"{id_prefix}-temperature", classes="sidebar-input", value="0.7") # Default value
        yield Static("Top-P", classes="sidebar-label")
        yield Input(placeholder="0.0 to 1.0", id=f"{id_prefix}-top-p", classes="sidebar-input", value="0.95") # Default value
        yield Static("Min-P", classes="sidebar-label")
        yield Input(placeholder="0.0 to 1.0", id=f"{id_prefix}-min-p", classes="sidebar-input", value="0.05") # Default value
        # Add Top-K if needed by some models/UI preference
        yield Static("Top-K", classes="sidebar-label")
        yield Input(placeholder="e.g., 50", id=f"{id_prefix}-top-k", classes="sidebar-input", value="50") # Default value

# --- Main App ---
class TabApp(App):
    CSS_PATH = "tab_app.css"
    BINDINGS = [ ("ctrl+q", "quit", "Quit App") ]
    current_tab = reactive(TAB_CHAT)

    # --- on_mount (remains the same - configure logging) ---
    def on_mount(self) -> None:
        # ... (previous implementation - setup RichLogHandler, no print statements needed now unless debugging setup) ...
        print("\n--- on_mount: Starting logging setup ---")
        try:
            log_display_widget = self.query_one("#app-log-display", RichLog)
            widget_handler = RichLogHandler(log_display_widget)
            formatter = logging.Formatter(
                "{asctime} [{levelname:<8}] {name}:{lineno:<4} : {message}",
                style="{", datefmt="%Y-%m-%d %H:%M:%S"
            )
            widget_handler.setFormatter(formatter)
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                if isinstance(handler, RichLogHandler):
                   root_logger.removeHandler(handler)
            root_logger.addHandler(widget_handler)
            widget_handler.setLevel(logging.DEBUG)
            root_logger.setLevel(logging.DEBUG)
            logging.info("Logging configured to redirect to Logs tab.")
        except Exception as e:
             print(f"!!!!!!!! FATAL ERROR in on_mount during logging setup !!!!!!!!")
             traceback.print_exc()
             try: logging.exception("FATAL: Failed to configure RichLogHandler!")
             except Exception: pass
        print("--- on_mount: Logging setup finished ---\n")


    # --- compose methods (remain mostly the same) ---
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
                    classes="-active" if tab_id == self.current_tab else ""
                )

    def compose_content_area(self) -> ComposeResult:
        # Use updated ALL_API_MODELS in sidebar creation if needed
        with Container(id="content"):
            # --- Chat Window ---
            with Container(id=f"{TAB_CHAT}-window", classes="window"):
                yield from create_settings_sidebar(TAB_CHAT) # Sidebar Left
                with Container(id="chat-main-content"): # Main content area (vertical: log above input)
                    yield RichLog(id="chat-log", wrap=True, highlight=True, classes="chat-log")
                    with Horizontal(id="chat-input-area"): # Input area at the bottom
                        yield TextArea(id="chat-input", classes="chat-input")
                        yield Button("Send", id="send-chat", classes="send-button")

            # --- Character Chat Window (Portrait on Right) ---
            with Container(id=f"{TAB_CHARACTER}-window", classes="window hidden"):
                yield from create_settings_sidebar(TAB_CHARACTER) # Sidebar Left
                with Container(id="character-main-content"):
                    with Horizontal(id="character-top-area"): # Top area (horizontal: log + portrait)
                        yield RichLog(id="character-log", wrap=True, highlight=True, classes="chat-log")
                        yield Static(ASCII_PORTRAIT, id="character-portrait")
                    with Horizontal(id="character-input-area"): # Input area at the bottom
                        yield TextArea(id="character-input", classes="chat-input")
                        yield Button("Send", id="send-character", classes="send-button")

            # --- Logs Window (MODIFIED) ---
            with Container(id=f"{TAB_LOGS}-window", classes="window hidden"):
                 yield RichLog(
                    id="app-log-display", # Specific ID for this log display
                    wrap=True, highlight=True, markup=True
                 )
            # --- END Logs Window Modification ---

            # --- Other Placeholder Windows ---
            for tab_id in ALL_TABS:
                if tab_id not in [TAB_CHAT, TAB_CHARACTER, TAB_LOGS]:
                    yield Static(
                        f"{tab_id.replace('_', ' ').capitalize()} Window Placeholder",
                        id=f"{tab_id}-window",
                        classes="window placeholder-window hidden"
                    )


    # --- Reactive Watcher ---
    def watch_current_tab(self, old_tab: str, new_tab: str) -> None:
        logging.debug(f"Switching tab from '{old_tab}' to '{new_tab}'") # Use standard logging
        for btn in self.query(f"#tab-{old_tab}"): btn.remove_class("-active")
        for window in self.query(f"#{old_tab}-window"): window.add_class("hidden")
        for btn in self.query(f"#tab-{new_tab}"): btn.add_class("-active")
        for window in self.query(f"#{new_tab}-window"):
            window.remove_class("hidden")
            # Try to focus input area in non-log tabs
            if new_tab not in [TAB_LOGS]:
                inputs = window.query("TextArea")
                if inputs:
                     try:
                         inputs.first().focus()
                         logging.debug(f"Focused TextArea input in tab {new_tab}")
                     except Exception:
                         # Use standard logging for warnings
                         logging.warning(f"Could not focus TextArea input in tab {new_tab}")


    # --- Event Handlers ---
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle changes in Select widgets, specifically for API provider."""
        select_id = event.control.id
        # Check if the changed Select is an API provider dropdown
        if select_id and select_id.endswith("-api-provider"):
            id_prefix = select_id.removesuffix("-api-provider")
            new_provider = str(event.value)
            logging.info(f"Provider changed for '{id_prefix}': '{new_provider}'")

            # Construct the correct ID for the model select widget
            model_select_id = f"#{id_prefix}-api-model" # e.g., "#chat-api-model"
            logging.debug(f"Attempting to find model select with ID: {model_select_id}")

            try:
                model_select = self.query_one(model_select_id, Select) # Use the corrected ID
            except Exception as e:
                # Log the corrected ID in the error message for clarity
                logging.critical(f"CRITICAL: Could not find model select '{model_select_id}': {e}")
                return # Stop if we can't find the related widget

            # --- Use updated ALL_API_MODELS map ---
            models = ALL_API_MODELS.get(new_provider, [])
            new_model_options = [(model, model) for model in models]

            # Use standard logging
            logging.info(f"Updating models for '{id_prefix}' based on provider '{new_provider}': {models}")
            model_select.set_options(new_model_options)

            # Set default value for the model dropdown
            if models:
                first_model = models[0]
                model_select.value = first_model # Set the actual value
                model_select.prompt = "Select Model..." # Reset prompt if needed
                logging.info(f"Set model value for '{id_prefix}' to default: '{first_model}'") # Use standard logging
            else:
                model_select.value = None # Clear the value
                model_select.prompt = "No models available" # Update prompt
                logging.info(f"No models available for '{id_prefix}', cleared value and updated prompt.") # Use standard logging

            model_select.refresh()
            logging.debug(f"Refreshed model select widget: {model_select.id}")
        else:
             logging.debug(f"Ignoring Select.Changed event from {select_id} (not an api-provider select)")

    # --- MODIFIED on_button_pressed ---
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses for tabs and sending messages."""
        button_id = event.button.id

        # --- Handle Tab Switching ---
        if button_id and button_id.startswith("tab-"):
            new_tab_id = button_id.replace("tab-", "")
            logging.info(f"Tab button pressed: switching to '{new_tab_id}'")
            self.current_tab = new_tab_id
        elif button_id and button_id.startswith("send-"):
            chat_id_part = button_id.replace("send-", "") # e.g., "chat" or "character"
            prefix = chat_id_part # Use this as the prefix for querying widgets
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
                top_k_widget = self.query_one(f"#{prefix}-top-k", Input) # Query Top-K too
            except Exception as e:
                logging.error(f"Could not find required UI widgets for '{prefix}': {e}")
                # Try writing to log widget if possible, otherwise just log
                try:
                    log_widget_for_error = self.query_one(f"#{prefix}-log", RichLog)
                    log_widget_for_error.write("[bold red]Error: Could not find UI elements.[/]")
                except Exception:
                    pass # Cannot even find log widget
                return

            # --- Get Values ---
            message = text_area.text.strip()
            selected_provider = str(provider_widget.value) if provider_widget.value else None
            selected_model = str(model_widget.value) if model_widget.value else None
            system_prompt = system_prompt_widget.text
            try:
                temperature = float(temp_widget.value) if temp_widget.value else 0.7 # Default
            except ValueError:
                logging.warning(f"Invalid temperature value '{temp_widget.value}', using default 0.7")
                temperature = 0.7
            try:
                top_p = float(top_p_widget.value) if top_p_widget.value else 0.95 # Default
            except ValueError:
                logging.warning(f"Invalid top_p value '{top_p_widget.value}', using default 0.95")
                top_p = 0.95
            try:
                min_p = float(min_p_widget.value) if min_p_widget.value else 0.05 # Default
            except ValueError:
                logging.warning(f"Invalid min_p value '{min_p_widget.value}', using default 0.05")
                min_p = 0.05
            try:
                top_k = int(top_k_widget.value) if top_k_widget.value else 50 # Default
            except ValueError:
                logging.warning(f"Invalid top_k value '{top_k_widget.value}', using default 50")
                top_k = 50

            # --- Basic Validation ---
            if not message:
                 logging.debug(f"Empty message submitted in '{prefix}'. Clearing input.")
                 text_area.clear()
                 text_area.focus()
                 return
            if not selected_provider:
                chat_log_widget.write("[bold red]Please select an API Provider.[/]")
                logging.warning(f"Send attempt ('{prefix}') failed: Provider not selected.")
                return
            if not selected_model:
                chat_log_widget.write("[bold red]Please select a Model.[/]")
                logging.warning(f"Send attempt ('{prefix}') failed: Model not selected.")
                return

            # --- Log User Message ---
            chat_log_widget.write(f"You: {message}")
            text_area.clear() # Clear input after getting message

            # --- Prepare and Dispatch API Call ---
            logging.info(f"Initiating API call for '{prefix}': Provider='{selected_provider}', Model='{selected_model}'")
            chat_log_widget.write(f"[dim]AI thinking ({selected_provider} / {selected_model})...[/]") # Indicate processing

            # Find the correct API function from the map
            api_function = API_FUNCTION_MAP.get(selected_provider)

            if api_function is None:
                error_msg = f"Error: No API function configured for provider '{selected_provider}'."
                logging.error(error_msg)
                chat_log_widget.write(f"[bold red]{error_msg}[/]")
                return

            # --- Prepare Arguments for the specific API function ---
            # This needs careful mapping based on your library function signatures
            api_args = {
                "api_key": None, # Let the library handle loading this for now
                "input_data": message,
                "custom_prompt_arg": "", # Assuming chat uses direct message, not transcript + prompt
                "model": selected_model,
                "system_message": system_prompt, # Or 'system_prompt' depending on func
                "temp": temperature,
                "streaming": False, # Hardcoded non-streaming for now
                # --- Map P/K values carefully ---
                # OpenAI uses 'maxp' (top_p), others use 'topp' or 'top_p'
                # Anthropic uses 'topp', 'topk'
                # Cohere uses 'topp', 'topk'
                # Check each function signature in your libs!
                # Example flexible mapping (adjust as needed):
                "topp": top_p,
                "top_p": top_p,
                "maxp": top_p, # For OpenAI if it uses this key
                "topk": top_k,
                "top_k_param": top_k, # Example if a func used a different name
                "minp": min_p,
                "min_p_param": min_p, # Example
                # --- Add other specific args if needed by functions ---
                "api_url": None, # For local models like llama.cpp if needed
            }
            # Filter out None values unless the function specifically handles them
            # Or better: check the signature of the target 'api_function'
            # For simplicity now, we pass them all, library func must handle None
            filtered_api_args = {k: v for k, v in api_args.items()} # Pass all for now

            logging.debug(f"Calling {api_function.__name__} with args: { {k: v for k, v in filtered_api_args.items() if k != 'api_key'} }") # Don't log key

            # --- Run the API call in a worker ---
            self.run_worker(
                self._api_worker, # The function to run in the worker thread
                args=[api_function, filtered_api_args, chat_log_widget], # Pass necessary args to worker
                name=f"API_Call_{prefix}",
                group=f"api_calls",
                exclusive=False, # Allow multiple calls if needed, though maybe True is better?
                thread=True # Crucial: run sync code in a thread
            )

    # --- Worker function to run synchronous API calls ---
    def _api_worker(self, api_func: callable, args: dict, log_widget: RichLog) -> Union[str, Generator[Any, Any, None]]:
        """
        Worker function to execute the synchronous API call.
        Returns the result (string) or potentially a generator if streaming was enabled.
        """
        try:
            # Make sure the function exists (was imported correctly)
            if api_func is None:
                raise ValueError("API function is not available (Import failed?)")

            # Call the actual library function
            # **IMPORTANT**: Adapt the arguments passed based on what the specific
            # api_func actually accepts. The 'args' dict might contain unused keys
            # for a given function. You might need inspect.signature or manual mapping.
            # For now, assume the functions are robust enough to ignore extra kwargs
            # or that the 'filtered_api_args' passed to run_worker was precise.
            logging.info(f"Worker executing: {api_func.__name__}")
            result = api_func(**args)
            logging.info(f"Worker received result from {api_func.__name__}")
            return result
        except Exception as e:
            logging.exception(f"Error during API call in worker ({api_func.__name__}): {e}")
            # Return the error message to be displayed in the UI
            return f"[bold red]API Error ({api_func.__name__}):[/] {str(e)}"

    # --- Handle worker completion ---
    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Called when a worker changes state."""
        worker_name = event.worker.name or "Unknown Worker"
        logging.debug(f"Worker '{worker_name}' state changed to: {event.state}")

        if event.state == WorkerState.SUCCESS:
            if worker_name.startswith("API_Call_"):
                prefix = worker_name.replace("API_Call_", "")
                try:
                    # Get the target log widget based on the worker name prefix
                    chat_log_widget = self.query_one(f"#{prefix}-log", RichLog)
                    result = event.worker.result # Get the return value from _api_worker

                    if isinstance(result, str):
                        logging.info(f"API call ({prefix}) successful. Result: {result[:100]}...")
                        # Remove the "thinking" message before writing result
                        # This is tricky, might need better state management or message IDs
                        # Simple approach: just write the result
                        chat_log_widget.write(f"AI: {result}")
                    elif isinstance(result, Generator):
                        # TODO: Handle streaming generator result
                        logging.warning("Received generator from API worker - Streaming not yet implemented in UI.")
                        chat_log_widget.write("[italic]AI: (Streaming response received, display not implemented)[/]")
                    else:
                        logging.error(f"API call ({prefix}) worker returned unexpected type: {type(result)}")
                        chat_log_widget.write(f"[bold red]Error: Unexpected result type from API.[/]")

                except Exception as e:
                    logging.error(f"Error retrieving result or log widget for worker '{worker_name}': {e}")

        elif event.state == WorkerState.ERROR:
            if worker_name.startswith("API_Call_"):
                prefix = worker_name.replace("API_Call_", "")
                try:
                    chat_log_widget = self.query_one(f"#{prefix}-log", RichLog)
                    # Log the exception from the worker
                    logging.error(f"Worker '{worker_name}' failed: {event.worker.error}")
                    chat_log_widget.write(f"[bold red]Error during API call process.[/]")
                except Exception as e:
                    logging.error(f"Error accessing log widget or worker error for failed worker '{worker_name}': {e}")


# --- Main execution block (remains the same) ---
if __name__ == "__main__":
    # --- CSS definition (Add styles for any new widgets if needed) ---
    css_content = """
    Screen { layout: vertical; }
    Header { dock: top; height: 1; background: $accent-darken-1; }
    Footer { dock: bottom; height: 1; background: $accent-darken-1; }
    #tabs { dock: top; height: 3; background: $background; padding: 0 1; }
    #tabs Button { width: 1fr; height: 100%; border: none; background: $panel; color: $text-muted; }
    #tabs Button:hover { background: $panel-lighten-1; color: $text; }
    #tabs Button.-active { background: $accent; color: $text; text-style: bold; border: none; }
    #content { height: 1fr; width: 100%; }
    .window { height: 100%; width: 100%; layout: horizontal; overflow: hidden; }
    .hidden { display: none; }
    .placeholder-window { align: center middle; background: $panel; }

    /* Sidebar Styling */
    .sidebar { width: 35; background: $boost; padding: 1 2; border-right: thick $background-darken-1; height: 100%; overflow-y: auto; overflow-x: hidden; }
    .sidebar-title { text-style: bold underline; margin-bottom: 1; width: 100%; text-align: center; }
    .sidebar-label { margin-top: 1; text-style: bold; }
    .sidebar-input { width: 100%; margin-bottom: 1; }
    .sidebar-textarea { width: 100%; height: 5; border: round $surface; margin-bottom: 1; }
    .sidebar Select { width: 100%; margin-bottom: 1; }
    #chat-api-key-placeholder, #character-api-key-placeholder { color: $text-muted; text-style: italic; margin-top: 1; }


    /* Base Chat Log Style (for chat and character tabs) */
    .chat-log { height: 1fr; width: 1fr; border: round $surface; padding: 0 1; }

    /* --- Chat Window (Single Pane) specific layouts --- */
    #chat-main-content { layout: vertical; height: 100%; width: 1fr; }
    /* Input area styling (shared) */
    #chat-input-area, #character-input-area { height: auto; max-height: 12; width: 100%; align: left top; padding: 1; border-top: round $surface; }
    /* Input widget styling (shared) */
    .chat-input { width: 1fr; height: auto; max-height: 100%; margin-right: 1; border: round $surface; }
    /* Send button styling (shared) */
    .send-button { width: 10; height: 100%; margin-top: 0; }

    /* --- Character Chat Window specific layouts --- */
    #character-main-content { layout: vertical; height: 100%; width: 1fr; }
    #character-top-area { height: 1fr; width: 100%; layout: horizontal; margin-bottom: 1; }
    #character-top-area > .chat-log { margin: 0 1 0 0; height: 100%; margin-bottom: 0; }
    #character-portrait { width: 25; height: 100%; border: round $surface; padding: 1; margin: 0; overflow: hidden; align: center top; }

    /* --- Styles for Logs Tab --- */
    #logs-window {
         padding: 0; border: none; height: 100%; width: 100%;
         & #logs-content-wrapper { /* If you keep the wrapper */
             layout: vertical; height: 100%; width: 100%;
         }
    }
    #app-log-display {
        border: round $surface; height: 1fr; width: 1fr; margin: 0; padding: 1;
    }
    /* Removed test-button style */

    """

    # --- CSS File Handling (remains the same) ---
    try:
        css_file = Path(TabApp.CSS_PATH)
        if not css_file.is_file():
             css_file.parent.mkdir(parents=True, exist_ok=True)
             with open(css_file, "w") as f:
                 f.write(css_content)
                 print(f"[INFO] Created CSS file: {css_file}")
    except Exception as e:
        print(f"[ERROR] Error handling CSS file '{TabApp.CSS_PATH}': {e}")
        try: logging.error(f"Error handling CSS file '{TabApp.CSS_PATH}': {e}")
        except NameError: pass


    # --- Run the App ---
    print("[INFO] Starting Textual App...")
    TabApp().run()