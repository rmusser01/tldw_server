# settings_sidebar.py
# Description: settings sidebar widget
#
# Imports
#
# 3rd-Party Imports
import logging

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static, Select, TextArea, Input, Collapsible
#
# Local Imports
from ..config import get_providers_and_models
#
#######################################################################################################################
#
# Functions:

# Sidebar visual constants ---------------------------------------------------
SIDEBAR_WIDTH = "30%"

def create_settings_sidebar(id_prefix: str, config: dict) -> ComposeResult:
    """Yield the widgets for the settings sidebar.

    The sidebar is divided into four collapsible groups:
        1. General & Chat Settings  – existing controls
        2. Character Chat Settings – placeholders for character‑specific UI
        3. Media Settings          – placeholders for media configuration
        4. Search & Tools Settings – placeholders for search / tool options
    """

    with VerticalScroll(id=f"{id_prefix}-sidebar", classes="sidebar"):
        # -------------------------------------------------------------------
        # Retrieve defaults / provider information (used in Collapsible #1)
        # -------------------------------------------------------------------
        defaults = config.get(f"{id_prefix}_defaults", config.get("chat_defaults", {}))
        providers_models = get_providers_and_models()
        logging.info(
            "Sidebar %s: Received providers_models. Count: %d. Keys: %s",
            id_prefix,
            len(providers_models),
            list(providers_models.keys()),
        )

        available_providers = list(providers_models.keys())
        default_provider: str = defaults.get(
            "provider", available_providers[0] if available_providers else ""
        )
        default_model: str = defaults.get("model", "")
        default_system_prompt: str = defaults.get("system_prompt", "")
        default_temp = str(defaults.get("temperature", 0.7))
        default_top_p = str(defaults.get("top_p", 0.95))
        default_min_p = str(defaults.get("min_p", 0.05))
        default_top_k = str(defaults.get("top_k", 50))

        # -------------------------------------------------------------------
        # Sidebar title (always visible)
        # -------------------------------------------------------------------
        yield Static("Settings", classes="sidebar-title")

        # ===================================================================
        # 1. General & Chat Settings – existing controls
        # ===================================================================
        with Collapsible(title="General & Chat Settings", collapsed=False):
            yield Static(
                "Inference Endpoints & \nService Providers", classes="sidebar-label"
            )
            provider_options = [(provider, provider) for provider in available_providers]
            yield Select(
                options=provider_options,
                prompt="Select Provider…",
                allow_blank=False,
                id=f"{id_prefix}-api-provider",
                value=default_provider,
            )

            # ----------------------------- Model ---------------------------
            yield Static("Model", classes="sidebar-label")
            initial_models = providers_models.get(default_provider, [])
            model_options = [(model, model) for model in initial_models]
            current_model_value = (
                default_model if default_model in initial_models else (initial_models[0] if initial_models else None)
            )
            yield Select(
                options=model_options,
                prompt="Select Model…",
                allow_blank=True,
                id=f"{id_prefix}-api-model",
                value=current_model_value,
            )

            # ------------------ Remaining numeric / text inputs ------------
            yield Static(
                "API Key (Set in config/env)",
                classes="sidebar-label",
                id=f"{id_prefix}-api-key-placeholder",
            )
            yield Static("System prompt", classes="sidebar-label")
            yield TextArea(
                id=f"{id_prefix}-system-prompt",
                text=default_system_prompt,
                classes="sidebar-textarea",
            )
            yield Static("Temperature", classes="sidebar-label")
            yield Input(
                placeholder="e.g., 0.7",
                id=f"{id_prefix}-temperature",
                value=default_temp,
                classes="sidebar-input",
            )
            yield Static("Top‑P", classes="sidebar-label")
            yield Input(
                placeholder="0.0 to 1.0",
                id=f"{id_prefix}-top-p",
                value=default_top_p,
                classes="sidebar-input",
            )
            yield Static("Min‑P", classes="sidebar-label")
            yield Input(
                placeholder="0.0 to 1.0",
                id=f"{id_prefix}-min-p",
                value=default_min_p,
                classes="sidebar-input",
            )
            yield Static("Top‑K", classes="sidebar-label")
            yield Input(
                placeholder="e.g., 50",
                id=f"{id_prefix}-top-k",
                value=default_top_k,
                classes="sidebar-input",
            )

        # ===================================================================
        # 2. Character Chat Settings – placeholders
        # ===================================================================
        with Collapsible(title="Character Chat Settings", collapsed=True):
            yield Static("Current Character", classes="sidebar-label")
            yield Select(
                options=[("<placeholder>", "placeholder")],
                prompt="Choose character…",
                allow_blank=True,
                id=f"{id_prefix}-character-select",
            )

            yield Static("Your Name", classes="sidebar-label")
            yield Input(
                placeholder="Type your display name…",
                id=f"{id_prefix}-your-name",
                classes="sidebar-input",
            )

            yield Static("Search & Load Chats", classes="sidebar-label")
            yield Input(
                placeholder="Search saved chats…",
                id=f"{id_prefix}-chat-search",
                classes="sidebar-input",
            )
            yield Static(
                "(search results dropdown placeholder)", classes="sidebar-placeholder"
            )

        # ===================================================================
        # 3. Media Settings – placeholders
        # ===================================================================
        with Collapsible(title="Media Settings", collapsed=True):
            yield Static("Media settings will go here (placeholder)", classes="sidebar-placeholder")

        # ===================================================================
        # 4. Search & Tools Settings – placeholders
        # ===================================================================
        with Collapsible(title="Search & Tools Settings", collapsed=True):
            yield Static(
                "Search & tools configuration will go here (placeholder)",
                classes="sidebar-placeholder",
            )

        # ===================================================================
        # 5. System Settings – placeholders
        # ===================================================================
        with Collapsible(title="Partial System Settings", collapsed=True):
            yield Static(
                "some key system settings will go here (placeholder)",
                classes="sidebar-placeholder",
            )

#
# End of settings_sidebar.py
#######################################################################################################################
