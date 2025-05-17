# File: /tldw_Server_API/app/core/Chat/prompt_template_manager.py
#
# Imports
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
#
# Third-party Libraries
from pydantic import BaseModel, Field
from loguru import logger
#
# Local Imports
#
########################################################################################################################
#
# Constants:
#
PROMPT_TEMPLATES_DIR = Path(__file__).parent / "prompt_templates"
#
#
# Functions:

class PromptTemplatePlaceholders(BaseModel):
    system: Optional[List[str]] = None
    user: Optional[List[str]] = None
    assistant: Optional[List[str]] = None

class PromptTemplate(BaseModel):
    name: str
    description: Optional[str] = None
    system_message_template: Optional[str] = None
    user_message_content_template: str = "{message_content}" # Default passthrough
    assistant_message_content_template: Optional[str] = None
    placeholders: Optional[PromptTemplatePlaceholders] = None

_loaded_templates: Dict[str, PromptTemplate] = {}

def load_template(template_name: str) -> Optional[PromptTemplate]:
    """Loads a single prompt template from a JSON file."""
    if template_name in _loaded_templates:
        return _loaded_templates[template_name]

    template_file = PROMPT_TEMPLATES_DIR / f"{template_name}.json"
    if not template_file.exists():
        logger.warning(f"Prompt template '{template_name}' not found at {template_file}")
        return None
    try:
        with open(template_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            template = PromptTemplate(**data)
            _loaded_templates[template_name] = template
            logger.info(f"Successfully loaded prompt template: {template_name}")
            return template
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON for template: {template_name}")
    except Exception as e:
        logger.error(f"Error loading template {template_name}: {e}", exc_info=True)
    return None

def apply_template_to_string(template_string: Optional[str], data: Dict[str, Any]) -> Optional[str]:
    """
    Applies data to a template string using str.format().
    Missing placeholders will be kept as is (e.g., "{missing_placeholder}").
    """
    if template_string is None:
        return None
    try:
        # Basic protection against formatting errors with missing keys
        # by providing a default value for missing keys (keeps the placeholder)
        class SafeFormatter(dict):
            def __missing__(self, key):
                return f"{{{key}}}" # Return the placeholder itself if key is missing

        return template_string.format_map(SafeFormatter(data))
    except KeyError as e:
        logger.warning(f"Placeholder {e} not found in data for template string: '{template_string}'")
        # Fallback to keep the template string as is or partially formatted
        # This should be handled by SafeFormatter now.
        return template_string
    except Exception as e:
        logger.error(f"Error applying template string '{template_string}': {e}", exc_info=True)
        return template_string # Return original on error

def get_available_templates() -> List[str]:
    """Returns a list of available template names (without .json extension)."""
    if not PROMPT_TEMPLATES_DIR.exists():
        return []
    return [f.stem for f in PROMPT_TEMPLATES_DIR.glob("*.json")]

# Load a default passthrough template on module load for safety
DEFAULT_RAW_PASSTHROUGH_TEMPLATE = PromptTemplate(
    name="raw_passthrough",
    description="Default template that makes no changes to the prompts.",
    system_message_template="{original_system_message_from_request}", # Or a more complex default if needed
    user_message_content_template="{message_content}",
    assistant_message_content_template="{message_content}"
)
_loaded_templates["raw_passthrough"] = DEFAULT_RAW_PASSTHROUGH_TEMPLATE

logger.info(f"Prompt templates directory: {PROMPT_TEMPLATES_DIR.resolve()}")
logger.info(f"Available templates found: {get_available_templates()}")

#
# End of prompt_template_manager.py
#######################################################################################################################
