import asyncio
from textual.app import App, ComposeResult
from textual.widgets import (
    Static, Button, Input, Header, Footer, RichLog, TextArea,
    Select  # <--- Removed RadioSet, RadioButton, Added Select
)
from textual.containers import Horizontal, Container, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget

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

# Example list of 15+ models
AVAILABLE_API_MODELS = [
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
    "gpt-4",
    "gpt-4-32k",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "claude-2.1",
    "claude-2.0",
    "claude-instant-1.2",
    "gemini-pro",
    "gemini-pro-vision",
    "mistral-7b-instruct",
    "mixtral-8x7b-instruct",
    "llama-2-70b-chat",
    "custom-model-alpha",
    "custom-model-beta",
    "legacy-model-v1"
]

# Use raw string for ASCII art
ASCII_PORTRAIT = r"""
  .--./)
 /.''.')
 | \ '/
 W `-'
 \\    '.
  '.    /
    `~~`
"""

# --- Helper Function for Sidebar (Generator) ---
def create_settings_sidebar(id_prefix: str) -> ComposeResult:
    """Yields the widgets for a standard settings sidebar."""
    with VerticalScroll(id=f"{id_prefix}-sidebar", classes="sidebar"):
        yield Static("Settings", classes="sidebar-title")

        yield Static("API Name", classes="sidebar-label")
        # --- MODIFIED: Use Select instead of RadioSet ---
        model_options = [(model, model) for model in AVAILABLE_API_MODELS]
        yield Select(
            options=model_options,
            prompt="Select API Model...",
            allow_blank=False,
            id=f"{id_prefix}-api-name",
            value=AVAILABLE_API_MODELS[0] # Default to first model
        )
        # --- END MODIFICATION ---

        yield Static("API Key", classes="sidebar-label")
        yield Input(
            password=True,
            placeholder="Enter API Key",
            id=f"{id_prefix}-api-key",
            classes="sidebar-input"
        )
        yield Static("System prompt", classes="sidebar-label")
        yield TextArea(
            id=f"{id_prefix}-system-prompt",
            classes="sidebar-textarea"
        )
        yield Static("Temperature", classes="sidebar-label")
        yield Input(
            placeholder="e.g., 0.7",
            id=f"{id_prefix}-temperature",
            classes="sidebar-input"
        )
        yield Static("Top-P", classes="sidebar-label")
        yield Input(
            placeholder="0.0 to 1.0",
            id=f"{id_prefix}-top-p",
            classes="sidebar-input"
        )
        yield Static("Min-P", classes="sidebar-label")
        yield Input(
            placeholder="0.0 to 1.0",
            id=f"{id_prefix}-min-p",
            classes="sidebar-input"
        )


# --- Main App ---
class TabApp(App):
    CSS_PATH = "tab_app.css"
    BINDINGS = [ ("ctrl+q", "quit", "Quit App") ]
    current_tab = reactive(TAB_CHAT)

    def compose(self) -> ComposeResult:
        """Compose the main layout using helper generator methods."""
        yield Header()
        yield from self.compose_tabs()
        yield from self.compose_content_area()
        yield Footer()

    def compose_tabs(self) -> ComposeResult:
        """Yields the tab buttons within a Horizontal container."""
        with Horizontal(id="tabs"):
            for tab_id in ALL_TABS:
                yield Button(
                    tab_id.replace("_", " ").capitalize(),
                    id=f"tab-{tab_id}",
                    classes="-active" if tab_id == self.current_tab else ""
                )

    def compose_content_area(self) -> ComposeResult:
        """Yields the main content area container and its children."""
        with Container(id="content"):
            # --- Chat Window ---
            with Container(id=f"{TAB_CHAT}-window", classes="window"):
                yield from create_settings_sidebar(TAB_CHAT)
                with Container(id="chat-main-content"):
                    with Horizontal(id="chat-panes"):
                        yield RichLog(id="chat1-log", wrap=True, highlight=True, classes="chat-log")
                        yield RichLog(id="chat2-log", wrap=True, highlight=True, classes="chat-log")
                    with Horizontal(id="chat-input-area"):
                        with Vertical(id="chat1-input-container", classes="chat-input-container"):
                             yield TextArea(id="chat1-input", classes="chat-input")
                             yield Button("Send", id="send-chat1", classes="send-button")
                        with Vertical(id="chat2-input-container", classes="chat-input-container"):
                             yield TextArea(id="chat2-input", classes="chat-input")
                             yield Button("Send", id="send-chat2", classes="send-button")

            # --- Character Chat Window (Portrait on Right) ---
            with Container(id=f"{TAB_CHARACTER}-window", classes="window hidden"):
                yield from create_settings_sidebar(TAB_CHARACTER)
                with Container(id="character-main-content"):
                    with Horizontal(id="character-top-area"):
                        yield RichLog(
                            id="character-log",
                            wrap=True,
                            highlight=True,
                            classes="chat-log"
                        )
                        yield Static(ASCII_PORTRAIT, id="character-portrait")
                    with Horizontal(id="character-input-area"):
                        yield TextArea(id="character-input", classes="chat-input")
                        yield Button("Send", id="send-character", classes="send-button")

            # --- Other Placeholder Windows ---
            for tab_id in ALL_TABS:
                if tab_id not in [TAB_CHAT, TAB_CHARACTER]:
                    yield Static(
                        f"{tab_id.replace('_', ' ').capitalize()} Window Placeholder",
                        id=f"{tab_id}-window",
                        classes="window placeholder-window hidden"
                    )

    # --- Reactive Watcher ---
    def watch_current_tab(self, old_tab: str, new_tab: str) -> None:
        """Called when the reactive 'current_tab' changes."""
        for btn in self.query(f"#tab-{old_tab}"): btn.remove_class("-active")
        for window in self.query(f"#{old_tab}-window"): window.add_class("hidden")
        for btn in self.query(f"#tab-{new_tab}"): btn.add_class("-active")
        for window in self.query(f"#{new_tab}-window"):
            window.remove_class("hidden")
            inputs = window.query("TextArea")
            if inputs:
                 try: inputs.first().focus()
                 except Exception: self.log.warning(f"Could not focus input in tab {new_tab}")


    # --- Event Handlers ---
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses for tabs and sending messages."""
        button_id = event.button.id
        selected_model = None # Keep track of selected model for logging

        if button_id and button_id.startswith("tab-"):
            self.current_tab = button_id.replace("tab-", "")
        elif button_id and button_id.startswith("send-"):
            chat_id_part = button_id.replace("send-", "")
            try:
                text_area = self.query_one(f"#{chat_id_part}-input", TextArea)
                if chat_id_part == "character": log = self.query_one("#character-log", RichLog)
                elif chat_id_part == "chat1": log = self.query_one("#chat1-log", RichLog)
                elif chat_id_part == "chat2": log = self.query_one("#chat2-log", RichLog)
                else:
                    self.log.error(f"Unknown chat ID part: {chat_id_part}")
                    return

                # Get settings including the Select value
                prefix = TAB_CHARACTER if chat_id_part == "character" else TAB_CHAT
                try:
                    select_widget = self.query_one(f"#{prefix}-api-name", Select)
                    selected_model = select_widget.value # Get value from Select widget
                    api_key = self.query_one(f"#{prefix}-api-key", Input).value
                    self.log.info(f"Settings for '{chat_id_part}': Model='{selected_model}', Key='{api_key[:4]}...'")
                except Exception as e:
                    self.log.error(f"Failed to query settings for {prefix}: {e}")
                    # Handle error - maybe prevent sending?
                    log.write("[bold red]Error retrieving settings![/]")
                    return # Stop processing if settings fail

            except Exception as e:
                self.log.error(f"Could not find input/log for {chat_id_part}: {e}")
                return

            message = text_area.text.strip()
            if message:
                log.write(f"You: {message}")
                # Use the selected_model retrieved earlier
                if selected_model:
                    self.log.info(f"TODO: Send message '{message}' from {chat_id_part} using model '{selected_model}' to backend")
                else:
                     self.log.warning(f"TODO: Send message '{message}' from {chat_id_part} but model was not selected?")

                text_area.clear()
                text_area.focus()
            else:
                 text_area.clear()
                 text_area.focus()


# --- Main execution block ---
if __name__ == "__main__":
    # Create CSS file if it doesn't exist
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
    .sidebar { width: 35; background: $boost; padding: 1 2; border-right: thick $background-darken-1; height: 100%; overflow-y: auto; overflow-x: hidden; }
    .sidebar-title { text-style: bold underline; margin-bottom: 1; width: 100%; text-align: center; }
    .sidebar-label { margin-top: 1; text-style: bold; }
    .sidebar-input { width: 100%; margin-bottom: 1; }
    .sidebar-textarea { width: 100%; height: 5; border: round $surface; margin-bottom: 1; }
    /* Removed RadioSet/RadioButton CSS */
    .sidebar Select { width: 100%; margin-bottom: 1; } /* Style for Select */
    #chat-main-content { layout: vertical; height: 100%; width: 1fr; }
    #chat-panes { height: 1fr; width: 100%; }
    .chat-log { height: 100%; width: 1fr; border: round $surface; padding: 0 1; }
    #chat-panes > #chat1-log { margin-right: 1; }
    #chat-input-area, #character-input-area { height: 6; width: 100%; align: center bottom; padding: 1; }
    #chat-input-area { border-top: round $surface; }
    .chat-input-container { width: 1fr; height: 100%; }
    #chat1-input-container { margin-right: 1; }
    .chat-input { width: 100%; height: 1fr; border: round $surface; }
    .send-button { width: 100%; margin-top: 1; height: 3; }
    #character-main-content { layout: vertical; height: 100%; width: 1fr; }
    #character-top-area { height: 1fr; width: 100%; }
    #character-top-area > .chat-log { margin: 0 1 1 0; }
    #character-portrait { width: 25; height: auto; border: round $surface; padding: 1; margin: 0 0 1 0; overflow: hidden; }
    #character-input-area { border-top: round $surface; }
    #character-input-area .chat-input { width: 1fr; height: 100%; margin-right: 1; }
    #character-input-area .send-button { width: 10; height: 100%; margin-top: 0; }
    """
    try:
        with open("tab_app.css", "x") as f: f.write(css_content)
    except FileExistsError: pass
    TabApp().run()