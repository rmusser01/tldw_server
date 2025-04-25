# tldw_app/widgets/chat_message.py
# Description: This file contains the ChatMessage widget for tldw_cli
#
# Imports
#
# 3rd-party Libraries
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Static, Button, Label # Added Label
from textual.reactive import reactive
#
# Local Imports
#
#######################################################################################################################
#
# Functions:

class ChatMessage(Widget):
    """A widget to display a single chat message with action buttons."""

    DEFAULT_CSS = """
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
    }
    .message-text {
        padding: 1;
        width: 100%;
        height: auto;
    }
    .message-actions {
        height: auto;
        width: 100%;
        padding: 0 1;
        /* margin-top: 1; */ /* Removed top margin */
        border-top: thin $surface-lighten-1;
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

    # Store the raw text content
    message_text = reactive("", repaint=True)
    role = reactive("User", repaint=True) # "User" or "AI"
    generation_complete = reactive(True) # Used for AI messages to show actions

    def __init__(self, message: str, role: str, generation_complete: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.message_text = message
        self.role = role
        self.generation_complete = generation_complete
        self.add_class(f"-{role.lower()}") # Add role-specific class

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"{self.role}", classes="message-header")
            yield Static(self.message_text, classes="message-text")
            # Add '-generating' class if needed, removed later
            actions_class = "message-actions" + (" -generating" if self.role == "AI" and not self.generation_complete else "")
            with Horizontal(classes=actions_class):
                # Common buttons
                yield Button("Edit", classes="action-button edit-button")
                yield Button("📋", classes="action-button copy-button", id="copy") # Emoji for copy
                yield Button("🔊", classes="action-button speak-button", id="speak") # Emoji for speak

                # AI-specific buttons
                if self.role == "AI":
                    yield Button("👍", classes="action-button thumb-up-button", id="thumb-up")
                    yield Button("👎", classes="action-button thumb-down-button", id="thumb-down")
                    yield Button("🔄", classes="action-button regenerate-button", id="regenerate") # Emoji for regenerate

    def update_message_chunk(self, chunk: str):
        """Append a chunk of text to the message (for streaming)."""
        if self.role == "AI":
            self.query_one(".message-text", Static).update(self.message_text + chunk)
            self.message_text += chunk # Update internal state too

    def mark_generation_complete(self):
        """Marks generation as complete and shows AI action buttons."""
        if self.role == "AI":
            self.generation_complete = True
            actions_container = self.query_one(".message-actions")
            actions_container.remove_class("-generating") # Make buttons visible
            actions_container.display = True # Ensure it's displayed if CSS uses 'display: none'

#
#
#######################################################################################################################
