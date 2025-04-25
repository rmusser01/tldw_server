# settings_sidebar.py
# Description: settings sidebar widget
#
# Imports
#
# 3rd-Party Imports
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static, Select, TextArea, Input
#
# Local Imports
from ..config import get_providers_and_models
#
#######################################################################################################################
#
# Functions:

def create_settings_sidebar(id_prefix: str, config: dict) -> ComposeResult:
    """Yields the widgets for a settings sidebar, using config for defaults."""
    with VerticalScroll(id=f"{id_prefix}-sidebar", classes="sidebar"):
        defaults = config.get(f"{id_prefix}_defaults", config.get("chat_defaults", {}))
        providers_models = get_providers_and_models() # Get latest from config
        available_providers = list(providers_models.keys())
        default_provider = defaults.get("provider", available_providers[0] if available_providers else "")
        default_model = defaults.get("model", "")
        default_system_prompt = defaults.get("system_prompt", "")
        default_temp = str(defaults.get("temperature", 0.7))
        default_top_p = str(defaults.get("top_p", 0.95))
        default_min_p = str(defaults.get("min_p", 0.05))
        default_top_k = str(defaults.get("top_k", 50))

        yield Static("Settings", classes="sidebar-title")
        yield Static("API Provider", classes="sidebar-label")
        provider_options = [(provider, provider) for provider in available_providers] # Use corrected list
        yield Select(
            options=provider_options, prompt="Select Provider...", allow_blank=False,
            id=f"{id_prefix}-api-provider", value=default_provider
        )
        yield Static("Model", classes="sidebar-label")
        # Use providers_models loaded earlier
        initial_models = providers_models.get(default_provider, [])
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

#
# End of settings_sidebar.py
#######################################################################################################################
