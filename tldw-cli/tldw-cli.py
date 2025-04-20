import asyncio
import logging  # Standard logging library
import logging.handlers  # For handlers
from pathlib import Path
from textual.app import App, ComposeResult
from textual.widgets import (
    Static, Button, Input, Header, Footer, RichLog, TextArea, Select
)
from textual.containers import Horizontal, Container, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.message import Message
# REMOVED: from textual import log - We will use standard logging

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

API_MODELS_BY_PROVIDER = {
    "OpenAI": [
        "gpt-4-1106-preview", "gpt-4-vision-preview", "gpt-4", "gpt-4-32k",
        "gpt-3.5-turbo-1106", "gpt-3.5-turbo", "gpt-3.5-turbo-16k",
    ],
    "Anthropic": ["claude-2.1", "claude-2.0", "claude-instant-1.2"],
    "Google": ["gemini-pro", "gemini-pro-vision"],
    "MistralAI": ["mistral-7b-instruct", "mixtral-8x7b-instruct"],
    "Meta": ["llama-2-70b-chat", "llama-2-13b-chat", "llama-2-7b-chat"],
    "Custom": ["custom-model-alpha", "custom-model-beta", "legacy-model-v1"]
}
AVAILABLE_PROVIDERS = list(API_MODELS_BY_PROVIDER.keys())

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
            # Use call_soon for thread safety when updating the UI from logging
            self.rich_log_widget.app.call_soon(self.rich_log_widget.write, message)
        except Exception:
            self.handleError(record)


# --- Helper Function for Sidebar ---
def create_settings_sidebar(id_prefix: str) -> ComposeResult:
    """Yields the widgets for a standard settings sidebar with dependent dropdowns."""
    with VerticalScroll(id=f"{id_prefix}-sidebar", classes="sidebar"):
        yield Static("Settings", classes="sidebar-title")
        yield Static("API Provider", classes="sidebar-label")
        provider_options = [(provider, provider) for provider in AVAILABLE_PROVIDERS]
        default_provider = AVAILABLE_PROVIDERS[0]
        yield Select(
            options=provider_options, prompt="Select Provider...", allow_blank=False,
            id=f"{id_prefix}-api-provider", value=default_provider
        )
        yield Static("Model", classes="sidebar-label")
        initial_models = API_MODELS_BY_PROVIDER.get(default_provider, [])
        model_options = [(model, model) for model in initial_models]
        yield Select(
            options=model_options, prompt="Select Model...", allow_blank=True,
            id=f"{id_prefix}-api-model",
            value=initial_models[0] if initial_models else None
        )
        yield Static("API Key set elsewhere.", classes="sidebar-label", id=f"{id_prefix}-api-key-placeholder")
        yield Static("System prompt", classes="sidebar-label")
        yield TextArea(id=f"{id_prefix}-system-prompt", classes="sidebar-textarea")
        yield Static("Temperature", classes="sidebar-label")
        yield Input(placeholder="e.g., 0.7", id=f"{id_prefix}-temperature", classes="sidebar-input")
        yield Static("Top-P", classes="sidebar-label")
        yield Input(placeholder="0.0 to 1.0", id=f"{id_prefix}-top-p", classes="sidebar-input")
        yield Static("Min-P", classes="sidebar-label")
        yield Input(placeholder="0.0 to 1.0", id=f"{id_prefix}-min-p", classes="sidebar-input")


# --- Main App ---
class TabApp(App):
    CSS_PATH = "tab_app.css"
    BINDINGS = [ ("ctrl+q", "quit", "Quit App") ]
    current_tab = reactive(TAB_CHAT)

    def on_mount(self) -> None:
        """Called when the app and widgets are mounted. Configures logging."""
        try:
            log_display_widget = self.query_one("#app-log-display", RichLog)
            widget_handler = RichLogHandler(log_display_widget)

            # Corrected formatter string using standard LogRecord attributes
            formatter = logging.Formatter(
                "{asctime} [{levelname:<8}] {name}:{lineno:<4} : {message}",
                style="{", datefmt="%Y-%m-%d %H:%M:%S"
            )
            widget_handler.setFormatter(formatter)

            # Get the root logger and add our handler
            root_logger = logging.getLogger()
            root_logger.addHandler(widget_handler)

            # Set levels for handler and root logger to ensure messages are processed
            # Set handler level (e.g., DEBUG to capture all levels >= DEBUG)
            widget_handler.setLevel(logging.DEBUG)
            # Ensure root logger level is also permissive enough (e.g., DEBUG)
            # If the root logger's level is higher (e.g., WARNING), DEBUG messages won't even reach the handler.
            root_logger.setLevel(logging.DEBUG) # Set root logger level

            # Use standard logging for messages intended for the Logs tab
            logging.info("Logging configured to redirect to Logs tab.")
            logging.debug("Test Debug Message: App Mounted.")
            logging.info("Test Info Message: App Mounted.")
            logging.warning("Test Warning Message: App Mounted.")

        except Exception as e:
             # Use standard logging to report errors during setup
             logging.exception("FATAL: Failed to configure RichLogHandler!")


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
        with Container(id="content"):
            # --- Chat Window (Single Pane) ---
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

            # --- Logic to update model options (remains the same) ---
            models = API_MODELS_BY_PROVIDER.get(new_provider, [])
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


    # --- on_button_pressed (remains the same as previous good version) ---
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses for tabs and sending messages."""
        button_id = event.button.id
        selected_provider = None
        selected_model = None

        if button_id and button_id.startswith("tab-"):
            new_tab_id = button_id.replace("tab-", "")
            logging.info(f"Tab button pressed: switching to '{new_tab_id}'")
            self.current_tab = new_tab_id
        elif button_id and button_id.startswith("send-"):
            chat_id_part = button_id.replace("send-", "")
            logging.info(f"'Send' button pressed for '{chat_id_part}'")
            try:
                text_area = self.query_one(f"#{chat_id_part}-input", TextArea)
                # IMPORTANT: log_widget here is the CHAT log, not the APP log
                chat_log_widget = self.query_one(f"#{chat_id_part}-log", RichLog)
                prefix = TAB_CHARACTER if chat_id_part == "character" else TAB_CHAT
                try:
                    # Query settings widgets
                    provider_widget = self.query_one(f"#{prefix}-api-provider", Select)
                    model_widget = self.query_one(f"#{prefix}-api-model", Select)
                    selected_provider = provider_widget.value
                    selected_model = model_widget.value
                    api_key_status = "API Key Needed (Not in UI)"

                    api_key_status = "API Key Needed (Not in UI)" # Placeholder

                    # Check if a model is selected BEFORE proceeding
                    if selected_model is None:
                        chat_log_widget.write("[bold red]Please select a model before sending![/]")
                        # Also log this issue to the main application log
                        logging.warning(f"Send attempt ('{chat_id_part}') failed: Model not selected in UI.")
                        return # Stop processing

                    # Log details to the APPLICATION log (Logs tab)
                    logging.info(f"Settings for '{chat_id_part}': Provider='{selected_provider}', Model='{selected_model}', Status='{api_key_status}'")

                except Exception as e:
                    # Log error to the APPLICATION log
                    logging.error(f"Failed to query settings widgets for '{prefix}': {e}")
                    # Show error in the specific CHAT log
                    chat_log_widget.write("[bold red]Error retrieving settings![/]")
                    return # Stop processing
            except Exception as e:
                # Log error finding core chat widgets to the APPLICATION log
                logging.error(f"Could not find input/log widgets for '{chat_id_part}': {e}")
                # Potentially show an error in a more general way if possible,
                # but here we can't even find the chat log widget reliably.
                return # Stop processing

            # --- If settings were retrieved successfully ---
            message = text_area.text.strip()
            if message:
                # Write user message to the specific CHAT log
                chat_log_widget.write(f"You: {message}")

                if selected_model: # Should always be true if we passed the check above
                    # Log the intended action to the APPLICATION log
                    logging.info(f"TODO: Send message '{message[:50]}...' from '{chat_id_part}' using provider '{selected_provider}' model '{selected_model}' to backend (API key required)")
                    # --- Placeholder for actual API call ---
                    # Simulate backend response (replace with actual call)
                    # await asyncio.sleep(1) # Simulate network delay
                    # response_text = f"Backend received: '{message}' using {selected_model}"
                    # chat_log_widget.write(f"AI: {response_text}")
                    # logging.info(f"Simulated response written to '{chat_id_part}-log'") # App log
                    chat_log_widget.write(f"[dim]AI response simulation for {selected_model}...[/]") # Placeholder in chat
                else:
                     # This case should technically not be reachable due to the earlier check
                     chat_log_widget.write("[bold red]Error: Model not selected (unexpected).[/]")
                     logging.error(f"Send attempt ('{chat_id_part}') failed unexpectedly after model check passed.")

                text_area.clear()
                text_area.focus()
            else:
                 logging.debug(f"Empty message submitted in '{chat_id_part}'. Clearing input.")
                 text_area.clear()
                 text_area.focus()


# --- Main execution block ---
if __name__ == "__main__":
    # --- CSS definition (remains the same) ---
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
    .sidebar Select { width: 100%; margin-bottom: 1; } /* Style for Select widgets */
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
    #logs-window { padding: 0; border: none; height: 100%; width: 100%; }
    #app-log-display { border: round $surface; height: 1fr; width: 1fr; margin: 0; padding: 1; }
    """

    # --- CSS File Handling (remains the same) ---
    try:
        css_file = Path(TabApp.CSS_PATH)
        if not css_file.is_file():
             css_file.parent.mkdir(parents=True, exist_ok=True)
             with open(css_file, "w") as f:
                 f.write(css_content)
                 print(f"Created CSS file: {css_file}")
    except Exception as e:
        print(f"Error handling CSS file '{TabApp.CSS_PATH}': {e}")
        # Log error if logging is somehow available, otherwise print is fallback
        try:
            logging.error(f"Error handling CSS file '{TabApp.CSS_PATH}': {e}")
        except NameError: # logging might not be defined if basicConfig was removed and error is early
            pass

    # *** REMOVED logging.basicConfig(...) call ***
    # We want logs to go *only* to the RichLogHandler once the app starts.
    # Pre-app errors (like CSS) will print to stderr by default.

    # Run the app
    TabApp().run()