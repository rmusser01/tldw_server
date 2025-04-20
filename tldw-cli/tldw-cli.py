import asyncio
from textual.app import App, ComposeResult
from textual.widgets import (
    Static, Button, Input, Header, Footer, RichLog, TextArea, Select
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

AVAILABLE_API_MODELS = [
    "gpt-4-1106-preview", "gpt-4-vision-preview", "gpt-4", "gpt-4-32k",
    "gpt-3.5-turbo-1106", "gpt-3.5-turbo", "gpt-3.5-turbo-16k",
    "claude-2.1", "claude-2.0", "claude-instant-1.2",
    "gemini-pro", "gemini-pro-vision",
    "mistral-7b-instruct", "mixtral-8x7b-instruct",
    "llama-2-70b-chat",
    "custom-model-alpha", "custom-model-beta", "legacy-model-v1"
]

ASCII_PORTRAIT = r"""
  .--./)
 /.''.')
 | \ '/
 W `-'
 \\    '.
  '.    /
    `~~`
"""

# --- Helper Function for Sidebar ---
def create_settings_sidebar(id_prefix: str) -> ComposeResult:
    """Yields the widgets for a standard settings sidebar."""
    with VerticalScroll(id=f"{id_prefix}-sidebar", classes="sidebar"):
        yield Static("Settings", classes="sidebar-title")
        yield Static("API Name", classes="sidebar-label")
        model_options = [(model, model) for model in AVAILABLE_API_MODELS]
        yield Select(
            options=model_options, prompt="Select API Model...", allow_blank=False,
            id=f"{id_prefix}-api-name", value=AVAILABLE_API_MODELS[0]
        )
        yield Static("API Key", classes="sidebar-label")
        yield Input(
            password=True, placeholder="Enter API Key",
            id=f"{id_prefix}-api-key", classes="sidebar-input"
        )
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
    CSS_PATH = "tldw-cli.css"
    BINDINGS = [ ("ctrl+q", "quit", "Quit App") ]
    current_tab = reactive(TAB_CHAT)

    def compose(self) -> ComposeResult:
        yield Header()
        yield from self.compose_tabs()
        yield from self.compose_content_area()
        yield Footer()

    def compose_tabs(self) -> ComposeResult:
        with Horizontal(id="tabs"):
            for tab_id in ALL_TABS:
                yield Button(
                    tab_id.replace("_", " ").capitalize(),
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
        button_id = event.button.id
        selected_model = None
        if button_id and button_id.startswith("tab-"):
            self.current_tab = button_id.replace("tab-", "")
        elif button_id and button_id.startswith("send-"):
            chat_id_part = button_id.replace("send-", "")
            try:
                text_area = self.query_one(f"#{chat_id_part}-input", TextArea)
                log = self.query_one(f"#{chat_id_part}-log", RichLog)
                prefix = TAB_CHARACTER if chat_id_part == "character" else TAB_CHAT
                try:
                    select_widget = self.query_one(f"#{prefix}-api-name", Select)
                    selected_model = select_widget.value
                    api_key = self.query_one(f"#{prefix}-api-key", Input).value
                    self.log.info(f"Settings for '{chat_id_part}': Model='{selected_model}', Key='{api_key[:4]}...'")
                except Exception as e:
                    self.log.error(f"Failed to query settings for {prefix}: {e}")
                    log.write("[bold red]Error retrieving settings![/]")
                    return
            except Exception as e:
                self.log.error(f"Could not find input/log for {chat_id_part}: {e}")
                return

            message = text_area.text.strip()
            if message:
                log.write(f"You: {message}")
                if selected_model: self.log.info(f"TODO: Send message '{message}' from {chat_id_part} using model '{selected_model}' to backend")
                else: self.log.warning(f"TODO: Send message '{message}' from {chat_id_part} but model was not selected?")
                text_area.clear()
                text_area.focus()
            else:
                 text_area.clear()
                 text_area.focus()


# --- Main execution block ---
if __name__ == "__main__":
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

    /* Base Chat Log Style */
    .chat-log {
        height: 1fr; /* Log takes remaining vertical space */
        width: 1fr;
        border: round $surface;
        padding: 0 1;
    }

    /* --- Chat Window (Single Pane) specific layouts --- */
    #chat-main-content {
        layout: vertical;
        height: 100%;
        width: 1fr;
    }
    /* Input area styling (shared by chat and character) */
    #chat-input-area, #character-input-area {
        height: auto;    /* Allow height to adjust */
        max-height: 12;  /* Limit growth */
        width: 100%;
        align: left top; /* Align children to top-left */
        padding: 1;
        border-top: round $surface;
    }
    /* Input widget styling (shared) */
    .chat-input { /* Targets TextArea */
        width: 1fr;
        height: auto;      /* Allow height to adjust */
        max-height: 100%;  /* Don't overflow parent */
        margin-right: 1;
        border: round $surface;
    }
    /* Send button styling (shared) */
    .send-button { /* Targets Button */
        width: 10;
        height: 100%;     /* Stretch vertically */
        margin-top: 0;
        /* align-self removed */
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
    /* Log when next to portrait */
    #character-top-area > .chat-log {
        margin: 0 1 0 0;
        height: 100%;
        margin-bottom: 0; /* Override base margin */
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
    """
    try:
        with open("tldw-cli.css", "x") as f: f.write(css_content)
    except FileExistsError: pass
    TabApp().run()