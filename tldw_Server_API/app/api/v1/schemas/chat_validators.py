# chat_validators.py
# Description: Advanced validators for chat request schemas
#
# Imports
import configparser
import json
import os
import re
import uuid
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.custom_openai_providers import iter_custom_openai_provider_names

#######################################################################################################################
#
# Constants:

# Valid conversation ID pattern (UUID or alphanumeric with hyphens/underscores)
CONVERSATION_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,100}$')

# Valid character ID pattern (numeric or name)
CHARACTER_ID_PATTERN = re.compile(r'^(\d+|[a-zA-Z0-9_\- ]{1,100})$')

# Pre-compiled pattern for redacting base64 image data in request size validation
# Matches data:image URIs and captures the prefix for redaction
DATA_URI_REDACT_PATTERN = re.compile(r'(data:image[^,]*,)[^"\s]+')

# Maximum tool definition size (reduced from 10KB to 5KB for security)
MAX_TOOL_DEFINITION_SIZE = 5000  # characters

# Maximum total request size
def _get_max_request_size() -> int:
    """Resolve the max request size with env/config overrides.

    Precedence:
    - Env var CHAT_REQUEST_MAX_SIZE (integer, characters)
    - Config [Chat-Module] max_request_size_chars
    - Default 1_000_000
    """
    # Env override
    try:
        env_val = os.getenv("CHAT_REQUEST_MAX_SIZE")
        if env_val is not None:
            return max(1, int(env_val))
    except (ValueError, TypeError) as env_err:
        logger.debug(f"Failed to parse CHAT_REQUEST_MAX_SIZE env var: {env_err}")
    # Config override
    try:
        from tldw_Server_API.app.core.config import load_comprehensive_config
        cfg = load_comprehensive_config()
        if cfg and cfg.has_section('Chat-Module'):
            raw = cfg.get('Chat-Module', 'max_request_size_chars', fallback=None)
            if raw is not None:
                return max(1, int(raw))
    except (
        AttributeError,
        FileNotFoundError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        configparser.Error,
    ) as cfg_err:
        logger.debug(f"Failed to load max_request_size from config: {cfg_err}")
    return 1_000_000

MAX_REQUEST_SIZE = _get_max_request_size()

#######################################################################################################################
#
# Validation Functions:

def validate_conversation_id(conversation_id: Optional[str]) -> Optional[str]:
    """
    Validate conversation ID format.

    Args:
        conversation_id: Conversation ID to validate

    Returns:
        Validated conversation ID or None

    Raises:
        TypeError: If conversation_id is not a string
        ValueError: If format is invalid
    """
    if conversation_id is None:
        return None

    # Type guard
    if not isinstance(conversation_id, str):
        raise TypeError(f"Conversation ID must be a string, got {type(conversation_id).__name__}")

    # Check if it's a valid UUID
    try:
        uuid.UUID(conversation_id)
        return conversation_id
    except ValueError:
        pass

    # Check against pattern
    if not CONVERSATION_ID_PATTERN.match(conversation_id):
        # Safe truncation with type guard already applied above
        display_id = conversation_id[:50] if len(conversation_id) > 50 else conversation_id
        raise ValueError(
            f"Invalid conversation_id format. Must be UUID or alphanumeric "
            f"with hyphens/underscores (max 100 chars): {display_id}"
        )

    return conversation_id


def validate_character_id(character_id: Optional[str]) -> Optional[str]:
    """
    Validate character ID format (can be numeric ID or character name).

    Args:
        character_id: Character ID to validate

    Returns:
        Validated character ID or None

    Raises:
        TypeError: If character_id is not a string
        ValueError: If format is invalid
    """
    if character_id is None:
        return None

    # Type guard
    if not isinstance(character_id, str):
        raise TypeError(f"Character ID must be a string, got {type(character_id).__name__}")

    if not CHARACTER_ID_PATTERN.match(character_id):
        # Safe truncation with type guard already applied above
        display_id = character_id[:50] if len(character_id) > 50 else character_id
        raise ValueError(
            f"Invalid character_id format. Must be numeric or valid name "
            f"(alphanumeric with spaces, hyphens, underscores, max 100 chars): {display_id}"
        )

    return character_id


def _is_gemini_native_tool(tool: dict) -> bool:
    if not isinstance(tool, dict):
        return False
    if "function_declarations" in tool or "functionDeclarations" in tool:
        return True
    tool_type = str(tool.get("type") or "").strip().lower()
    return tool_type in {"gemini_native", "gemini-native", "gemini"}


def _validate_gemini_native_tool(tool: dict, idx: int) -> None:
    decls = tool.get("function_declarations")
    if decls is None:
        decls = tool.get("functionDeclarations")
    if decls is None:
        raise ValueError(f"Tool at index {idx} missing 'function_declarations' field")
    if not isinstance(decls, list):
        raise ValueError(f"Tool at index {idx} function_declarations must be a list")


def validate_tool_definitions(tools: Optional[list], provider: Optional[str] = None) -> Optional[list]:
    """
    Validate tool definitions for function calling.

    Args:
        tools: List of tool definitions
        provider: Optional provider name to allow provider-specific tool shapes

    Returns:
        Validated tools or None

    Raises:
        ValueError: If tools are invalid
    """
    if tools is None:
        return None

    if not isinstance(tools, list):
        raise ValueError("Tools must be a list")

    if len(tools) > 128:
        raise ValueError(f"Too many tools defined (max 128, got {len(tools)})")

    provider_key = (provider or "").strip().lower()

    for idx, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(f"Tool at index {idx} must be a dictionary")

        if provider_key == "google" and _is_gemini_native_tool(tool):
            _validate_gemini_native_tool(tool, idx)
            tool_json = json.dumps(tool)
            if len(tool_json) > MAX_TOOL_DEFINITION_SIZE:
                raise ValueError(
                    f"Tool at index {idx} definition too large "
                    f"(max {MAX_TOOL_DEFINITION_SIZE} chars, got {len(tool_json)})"
                )
            continue

        # Check required fields
        if 'type' not in tool:
            raise ValueError(f"Tool at index {idx} missing 'type' field")

        if tool['type'] != 'function':
            raise ValueError(f"Tool at index {idx} has invalid type '{tool['type']}' (only 'function' supported)")

        if 'function' not in tool:
            raise ValueError(f"Tool at index {idx} missing 'function' field")

        func = tool['function']
        if not isinstance(func, dict):
            raise ValueError(f"Tool at index {idx} function must be a dictionary")

        # Validate function definition
        if 'name' not in func:
            raise ValueError(f"Tool at index {idx} function missing 'name' field")

        name = func['name']
        if not isinstance(name, str) or not re.match(r'^[a-zA-Z0-9_-]{1,64}$', name):
            raise ValueError(
                f"Tool at index {idx} function name must be alphanumeric "
                f"with underscores/hyphens (max 64 chars): {name[:50]}"
            )

        # Check size
        tool_json = json.dumps(tool)
        if len(tool_json) > MAX_TOOL_DEFINITION_SIZE:
            raise ValueError(
                f"Tool at index {idx} definition too large "
                f"(max {MAX_TOOL_DEFINITION_SIZE} chars, got {len(tool_json)})"
            )

    return tools


def validate_temperature(temp: Optional[float]) -> Optional[float]:
    """
    Validate temperature parameter.

    Args:
        temp: Temperature value

    Returns:
        Validated temperature or None

    Raises:
        ValueError: If temperature is invalid
    """
    if temp is None:
        return None

    if not isinstance(temp, (int, float)):
        raise ValueError(f"Temperature must be a number, got {type(temp)}")

    if temp < 0.0 or temp > 2.0:
        raise ValueError(f"Temperature must be between 0.0 and 2.0, got {temp}")

    return float(temp)


def validate_max_tokens(max_tokens: Optional[int]) -> Optional[int]:
    """
    Validate max_tokens parameter.

    Args:
        max_tokens: Maximum tokens value

    Returns:
        Validated max_tokens or None

    Raises:
        ValueError: If max_tokens is invalid
    """
    if max_tokens is None:
        return None

    if not isinstance(max_tokens, int):
        try:
            max_tokens = int(max_tokens)
        except (ValueError, TypeError):
            raise ValueError(f"max_tokens must be an integer, got {type(max_tokens)}") from None

    if max_tokens < 1:
        raise ValueError(f"max_tokens must be at least 1, got {max_tokens}")

    if max_tokens > 128000:  # Most models have lower limits, but we'll be generous
        raise ValueError(f"max_tokens too large (max 128000, got {max_tokens})")

    return max_tokens


def _sanitize_value_for_size(value: Any) -> Any:
    """Redact large data:image payloads to avoid penalizing multimodal requests.

    - Replaces base64 content in data URIs with a short '<redacted>' marker.
    - Leaves other values intact.
    """
    try:
        # Redact strings that look like data:image URIs
        if isinstance(value, str) and value.startswith('data:image'):
            try:
                prefix, _rest = value.split(',', 1)
            except ValueError:
                prefix = value
            # Keep mime/metadata, drop base64 bulk
            return f"{prefix},<redacted>"
        # Recurse lists
        if isinstance(value, list):
            return [_sanitize_value_for_size(v) for v in value]
        # Recurse dicts
        if isinstance(value, dict):
            return {k: _sanitize_value_for_size(v) for k, v in value.items()}
        return value
    except (RecursionError, TypeError):
        return value


def validate_request_size(request_data: Any) -> bool:
    """
    Validate total request size.

    Args:
        request_data: Request data object or JSON string

    Returns:
        True if valid

    Raises:
        ValueError: If request is too large
    """
    try:
        # If it's already a JSON string, avoid re-serializing: redact via regex only.
        if isinstance(request_data, str):
            # Use pre-compiled pattern to replace base64 payload with <redacted>
            request_json = DATA_URI_REDACT_PATTERN.sub(r'\1<redacted>', request_data)
        else:
            # Convert to a serializable dict and sanitize recursively
            raw = request_data.model_dump() if hasattr(request_data, 'model_dump') else request_data
            sanitized_obj = _sanitize_value_for_size(raw)
            request_json = json.dumps(sanitized_obj)

        if len(request_json) > MAX_REQUEST_SIZE:
            raise ValueError(
                f"Request too large (max {MAX_REQUEST_SIZE} chars, got {len(request_json)})"
            )

        return True

    except json.JSONDecodeError as e:
        logger.error(f"JSON error validating request size: {e}")
        raise ValueError(f"Invalid JSON in request: {str(e)}") from e
    except (AttributeError, OverflowError, RecursionError, TypeError, ValueError) as e:
        logger.error(f"Error validating request size: {e}")
        raise ValueError(f"Failed to validate request size: {str(e)}") from e


def validate_stop_sequences(stop: Optional[Any]) -> Optional[Any]:
    """
    Validate stop sequences parameter.

    Args:
        stop: Stop sequences (string or list of strings)

    Returns:
        Validated stop sequences or None

    Raises:
        ValueError: If stop sequences are invalid
    """
    if stop is None:
        return None

    if isinstance(stop, str):
        if len(stop) > 500:
            raise ValueError(f"Stop sequence too long (max 500 chars, got {len(stop)})")
        return stop

    if isinstance(stop, list):
        if len(stop) > 4:
            raise ValueError(f"Too many stop sequences (max 4, got {len(stop)})")

        for idx, seq in enumerate(stop):
            if not isinstance(seq, str):
                raise ValueError(f"Stop sequence at index {idx} must be a string")
            if len(seq) > 500:
                raise ValueError(f"Stop sequence at index {idx} too long (max 500 chars)")

        return stop

    raise ValueError(f"Stop sequences must be string or list of strings, got {type(stop)}")


def validate_model_name(model: Optional[str]) -> Optional[str]:
    """
    Validate model name format.

    Model names can contain:
    - Alphanumeric characters
    - Underscores, hyphens, periods
    - Forward slashes (for HuggingFace-style names like 'meta-llama/Llama-2-70b')
    - Spaces (some models have spaces in names)

    Colons are NOT allowed as they can be used for URL/port injection.
    Path traversal patterns (..) are also rejected.

    Args:
        model: Model name

    Returns:
        Validated model name or None

    Raises:
        TypeError: If model name is not a string
        ValueError: If model name is invalid
    """
    if model is None:
        return None

    if not isinstance(model, str):
        raise TypeError(f"Model name must be a string, got {type(model).__name__}")

    # Check for empty string
    if not model or not model.strip():
        raise ValueError("Model name cannot be empty or whitespace-only")

    if len(model) > 100:
        raise ValueError(f"Model name too long (max 100 chars, got {len(model)})")

    # Reject colon character (can be used for URL/port injection)
    if ':' in model:
        raise ValueError("Model name cannot contain colons")

    # Reject path traversal patterns
    if '..' in model:
        raise ValueError("Model name cannot contain path traversal patterns")

    # Basic sanity check for model name - allows alphanumeric, underscore, hyphen, period, slash, space
    # Removed colon from allowed characters for security
    if not re.match(r'^[a-zA-Z0-9_\-./ ]+$', model):
        display_model = model[:50]
        raise ValueError(f"Model name contains invalid characters: {display_model}")

    # Reject names that start or end with slash (potential path issues)
    if model.startswith('/') or model.endswith('/'):
        raise ValueError("Model name cannot start or end with a slash")

    # Return stripped model name to prevent issues with leading/trailing whitespace
    return model.strip()


# Maximum length for provider names
MAX_PROVIDER_NAME_LENGTH = 50

# Allowed providers - frozen set for O(1) lookup
# Includes all supported providers plus 'aphrodite' and potential future providers
ALLOWED_PROVIDERS: frozenset = frozenset([
    "bedrock",
    "anthropic",
    "cohere",
    "deepseek",
    "google",
    "groq",
    "qwen",
    "huggingface",
    "mistral",
    "openai",
    "openrouter",
    "novita",
    "poe",
    "together",
    "llama.cpp",
    "kobold",
    "ollama",
    "ooba",
    "tabbyapi",
    "vllm",
    "local-llm",
    "aphrodite",
    "moonshot",
    "zai",
    *iter_custom_openai_provider_names(),
])


def validate_provider_name(provider: Optional[str], strict: bool = True) -> Optional[str]:
    """
    Validate provider name format.

    Args:
        provider: Provider name
        strict: If True, reject unknown providers. If False, just warn. Defaults to True.

    Returns:
        Validated provider name or None

    Raises:
        TypeError: If provider name is not a string
        ValueError: If provider name is invalid or unknown (when strict=True)
    """
    if provider is None:
        return None

    if not isinstance(provider, str):
        raise TypeError(f"Provider name must be a string, got {type(provider).__name__}")

    # Length check
    if len(provider) > MAX_PROVIDER_NAME_LENGTH:
        raise ValueError(f"Provider name too long (max {MAX_PROVIDER_NAME_LENGTH} chars)")

    # Convert to lowercase for consistency
    provider = provider.lower().strip()

    # Empty string check
    if not provider:
        return None

    # Character validation - only alphanumeric, hyphen, period, underscore
    if not re.match(r'^[a-z0-9_.\-]+$', provider):
        raise ValueError("Provider name contains invalid characters (allowed: alphanumeric, underscore, hyphen, period)")

    if provider not in ALLOWED_PROVIDERS:
        if strict:
            raise ValueError(f"Unknown provider: {provider}. Allowed providers: {', '.join(sorted(ALLOWED_PROVIDERS))}")
        else:
            logger.warning(f"Unknown provider '{provider}', proceeding anyway")

    return provider


#
# End of chat_validators.py
#######################################################################################################################
