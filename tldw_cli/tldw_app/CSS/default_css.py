# default_css.py
# Description: This file contains the default CSS for the tldw CLI application.
#
# Imports
#
# 3rd-party Libraries
#
# Local Imports
#
#######################################################################################################################
#
# Static declarations:
# Constants
DEFAULT_CSS_CONTENT = """

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

#
# End of default_css.py
#######################################################################################################################
