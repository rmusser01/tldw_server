# Server_API/app/api/schemas/chat_request_models.py
# Description: This code provides schema models for the /chat API endpoints
#
# Imports
import json
import os
import re
from pathlib import Path
from typing import Any, Literal, Optional, Union

from dotenv import load_dotenv

#
# 3rd-party imports
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from tldw_Server_API.app.core.testing import is_test_mode

#
# Local Imports
from tldw_Server_API.app.core.LLM_Calls.routing.models import RoutingOverride

#
#######################################################################################################################
#
# Functions:

# --- Pydantic Models for OpenAI Chat Completion Request ---
# Based on https://platform.openai.com/docs/api-reference/chat/create

DEFAULT_LLM_PROVIDER = "openai"
model_config = ConfigDict(extra="allow", from_attributes=True)

# Config Loading
# Prefer the canonical Config_Files/.env location, then fall back to local .env variants.
try:
    api_root = Path(__file__).resolve().parents[4]
    candidate_env_paths = [
        api_root / "Config_Files" / ".env",
        api_root / "Config_Files" / ".ENV",
        Path(".env").resolve(),
        Path(".ENV").resolve(),
    ]
    for env_path in candidate_env_paths:
        if env_path.exists():
            load_dotenv(dotenv_path=str(env_path), override=False)
except Exception as dotenv_error:
    # Fall back silently if dotenv loading fails; environment may be pre-populated
    _ = dotenv_error

# Use load_and_log_configs which returns a proper dict
from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.custom_openai_providers import (
    custom_openai_api_key_env_keys,
    custom_openai_provider_number,
    custom_openai_section_name,
    iter_custom_openai_provider_names,
)

_config = load_and_log_configs() or {}


def _config_default_llm_provider(config_data: Optional[dict[str, Any]]) -> Optional[str]:
    if not isinstance(config_data, dict):
        return None
    for section in ("llm_api_settings", "API"):
        section_data = config_data.get(section)
        if isinstance(section_data, dict):
            default_api = section_data.get("default_api")
            if isinstance(default_api, str):
                candidate = default_api.strip()
                if candidate:
                    return candidate
    return None


_cfg_default_provider = _config_default_llm_provider(_config)
_env_default_provider = os.getenv("DEFAULT_LLM_PROVIDER")
_test_mode_enabled = is_test_mode()

if _cfg_default_provider:
    DEFAULT_LLM_PROVIDER = _cfg_default_provider
elif _env_default_provider:
    DEFAULT_LLM_PROVIDER = _env_default_provider
elif _test_mode_enabled:
    DEFAULT_LLM_PROVIDER = "local-llm"


def _get_setting(env_var, section, key, default=""):
    env_value = os.getenv(env_var)
    if env_value is not None:
        return env_value

    # Check for API key in the config dict
    if section == "api_keys":
        # Look for provider-specific API config like 'openai_api'
        provider_api_key = f"{key}_api"
        if provider_api_key in _config:
            api_config = _config[provider_api_key]
            if isinstance(api_config, dict):
                api_key = api_config.get("api_key")
                if api_key:
                    return api_key

    # Fallback to checking section directly
    config_section = _config.get(section)
    if config_section:
        config_value = config_section.get(key) if isinstance(config_section, dict) else None
        if config_value is not None:
            return config_value

    return default


_BASE_SUPPORTED_PROVIDER_NAMES_LIST: list[str] = [
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
]

ALL_SUPPORTED_PROVIDER_NAMES_LIST: list[str] = [
    *_BASE_SUPPORTED_PROVIDER_NAMES_LIST,
    *iter_custom_openai_provider_names(),
]


def get_api_keys() -> dict[str, Optional[str]]:
    """
    Get API keys dynamically to support runtime changes.
    This function reloads config and environment variables each time it's called,
    ensuring that test environment changes are properly reflected.
    """
    # Reload config to get latest values
    current_config = load_and_log_configs() or {}

    def _get_dynamic_setting(
        env_vars: str | tuple[str, ...],
        section: str,
        key: str,
        default: Optional[str] = "",
    ) -> Optional[str]:
        """Resolve an API key from env aliases before config-backed settings."""
        # First check environment variable
        env_candidates = (env_vars,) if isinstance(env_vars, str) else tuple(env_vars or ())
        for env_var in env_candidates:
            env_value = os.getenv(env_var)
            if env_value is not None:
                return env_value

        custom_number = custom_openai_provider_number(key)
        if section == "api_keys" and custom_number is not None:
            custom_section = current_config.get(custom_openai_section_name(custom_number))
            if isinstance(custom_section, dict):
                api_key = custom_section.get("api_key")
                if api_key:
                    return api_key

        # Check for API key in the config dict
        if section == "api_keys":
            # Look for provider-specific API config like 'openai_api'
            provider_api_key = f"{key}_api"
            if provider_api_key in current_config:
                api_config = current_config[provider_api_key]
                if isinstance(api_config, dict):
                    api_key = api_config.get("api_key")
                    if api_key:
                        return api_key

        # Fallback to checking section directly
        config_section = current_config.get(section)
        if config_section:
            config_value = config_section.get(key) if isinstance(config_section, dict) else None
            if config_value is not None:
                return config_value

        return default

    def _provider_env_keys(provider_name: str) -> tuple[str, ...]:
        custom_number = custom_openai_provider_number(provider_name)
        if custom_number is not None:
            return custom_openai_api_key_env_keys(custom_number)
        normalized = provider_name.upper().replace('.', '_').replace('-', '_')
        if normalized.endswith("_API"):
            normalized = normalized[: -len("_API")]
        return (f"{normalized}_API_KEY",)

    return {
        name: _get_dynamic_setting(_provider_env_keys(name), "api_keys", name)
        for name in ALL_SUPPORTED_PROVIDER_NAMES_LIST
    }


# Keep API_KEYS for backward compatibility but make it compute on first access.
# This map is for test overrides and transitional compatibility; new code should
# use get_api_keys() / resolve_provider_api_key.
API_KEYS = get_api_keys()

# Runtime validation below enforces provider ids; keep this Python 3.10-safe.
SUPPORTED_API_ENDPOINTS = str


# --- Tool Definitions ---

# Security limits for function parameters
MAX_PARAMETER_DEPTH = 10  # Maximum nesting depth for JSON Schema parameters
MAX_PARAMETER_SIZE_BYTES = 5000  # Maximum serialized size of parameters (5KB)


def _calculate_json_depth(obj: Any, current_depth: int = 0) -> int:
    """Calculate the maximum nesting depth of a JSON-like structure."""
    if current_depth > MAX_PARAMETER_DEPTH + 1:
        # Early termination if already exceeded
        return current_depth
    if isinstance(obj, dict):
        if not obj:
            return current_depth
        return max(_calculate_json_depth(v, current_depth + 1) for v in obj.values())
    elif isinstance(obj, list):
        if not obj:
            return current_depth
        return max(_calculate_json_depth(item, current_depth + 1) for item in obj)
    return current_depth


def _validate_json_schema_structure(schema: dict[str, Any], path: str = "root") -> None:
    """
    Basic validation that the schema follows JSON Schema conventions.
    Raises ValueError for invalid schema values and TypeError for type mismatches.
    """
    if not isinstance(schema, dict):
        raise ValueError(f"JSON Schema at '{path}' must be an object, got {type(schema).__name__}")

    # Validate type field if present
    schema_type = schema.get("type")
    if schema_type is not None:
        valid_types = {"string", "number", "integer", "boolean", "array", "object", "null"}
        if isinstance(schema_type, str):
            if schema_type not in valid_types:
                raise ValueError(f"Invalid JSON Schema type '{schema_type}' at '{path}'")
        elif isinstance(schema_type, list):
            for t in schema_type:
                if t not in valid_types:
                    raise ValueError(f"Invalid JSON Schema type '{t}' in type array at '{path}'")
        else:
            raise ValueError(f"JSON Schema 'type' must be string or array at '{path}'")

    # Validate properties if present
    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, dict):
            raise ValueError(f"JSON Schema 'properties' must be an object at '{path}'")
        for prop_name, prop_schema in properties.items():
            if not isinstance(prop_name, str):
                raise ValueError(f"Property name must be string at '{path}'")
            if isinstance(prop_schema, dict):
                _validate_json_schema_structure(prop_schema, f"{path}.properties.{prop_name}")

    # Validate items if present (for arrays)
    items = schema.get("items")
    if items is not None and isinstance(items, dict):
        _validate_json_schema_structure(items, f"{path}.items")

    # Validate required if present
    required = schema.get("required")
    if required is not None:
        if not isinstance(required, list):
            raise ValueError(f"JSON Schema 'required' must be an array at '{path}'")
        for req in required:
            if not isinstance(req, str):
                raise ValueError(f"JSON Schema 'required' items must be strings at '{path}'")


class FunctionDefinition(BaseModel):
    """Describes a function available to the model."""

    name: str = Field(
        ...,
        description="The name of the function to be called. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.",
    )
    description: Optional[str] = Field(
        None,
        description="A description of what the function does, used by the model to choose when and how to call the function.",
    )
    parameters: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="The parameters the functions accepts, described as a JSON Schema object. See the guide[1] for examples, and the JSON Schema reference[2] for documentation about the format. Omitting parameters defines a function with an empty parameter list. [1] https://platform.openai.com/docs/guides/function-calling [2] https://json-schema.org/understanding-json-schema/",
    )

    @field_validator("parameters")
    @classmethod
    def validate_parameters_schema(cls, v: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """Validate that parameters follow JSON Schema conventions and security limits."""
        if v is None or not v:
            return v

        # Check serialized size to prevent DoS via large payloads
        try:
            serialized = json.dumps(v, separators=(',', ':'))
            if len(serialized.encode('utf-8')) > MAX_PARAMETER_SIZE_BYTES:
                raise ValueError(
                    f"Function parameters exceed maximum size of {MAX_PARAMETER_SIZE_BYTES} bytes"
                )
        except (TypeError, ValueError) as e:
            if "maximum size" in str(e):
                raise
            raise ValueError(f"Function parameters must be JSON serializable: {e}") from e

        # Check nesting depth to prevent stack overflow / DoS
        depth = _calculate_json_depth(v)
        if depth > MAX_PARAMETER_DEPTH:
            raise ValueError(
                f"Function parameters exceed maximum nesting depth of {MAX_PARAMETER_DEPTH} levels"
            )

        # Validate basic JSON Schema structure
        _validate_json_schema_structure(v)

        return v


class ToolDefinition(BaseModel):
    """Definition of a tool the model can use."""

    type: Literal["function"] = Field(..., description="The type of the tool. Currently, only `function` is supported.")
    function: FunctionDefinition


class GeminiNativeToolDefinition(BaseModel):
    """Gemini-native tool definition (function declarations)."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)
    function_declarations: list[dict[str, Any]] = Field(
        ...,
        alias="functionDeclarations",
        description="Gemini function declarations payload.",
    )
    type: Optional[str] = Field(
        None,
        description="Optional marker for Gemini-native tools (e.g., 'gemini_native').",
    )


class ToolChoiceFunction(BaseModel):
    """Specifies a specific function to be called."""

    name: str = Field(..., description="The name of the function to call.")


class ToolChoiceOption(BaseModel):
    """Specifies a tool the model should use. Use to force the model to call a specific function."""

    type: Literal["function"] = Field(..., description="The type of the tool. Currently, only `function` is supported.")
    function: ToolChoiceFunction


# --- Message Definitions ---

# Maximum length for individual message content (characters, not tokens)
# This provides per-field protection in addition to total request size validation.
# 100,000 chars is approximately 25,000-40,000 tokens depending on language.
MAX_MESSAGE_CONTENT_LENGTH = 100_000

# Maximum length for the name field on messages
MAX_MESSAGE_NAME_LENGTH = 64


class ChatCompletionRequestMessageContentPartText(BaseModel):
    type: Literal["text"]
    text: str

    @field_validator("text")
    @classmethod
    def validate_text_length(cls, v: str) -> str:
        if len(v) > MAX_MESSAGE_CONTENT_LENGTH:
            raise ValueError(
                f"Text content exceeds maximum length ({MAX_MESSAGE_CONTENT_LENGTH} chars)"
            )
        return v


class ChatCompletionRequestMessageContentPartImageURL(BaseModel):
    url: Union[HttpUrl, str] = Field(
        ...,
        description=(
            "Base64-encoded image data as a data URI (e.g., 'data:image/png;base64,...'). "
            "External HTTP/HTTPS URLs are not supported for chat images."
        ),
    )
    detail: Optional[Literal["auto", "low", "high"]] = Field(
        "auto", description="Specifies the detail level of the image."
    )

    @field_validator("url")
    def check_url_or_data(cls, v):
        # Normalize HttpUrl to string and enforce data URI requirement
        if isinstance(v, HttpUrl):
            v = str(v)
        if not isinstance(v, str):
            raise ValueError("url must be a string data URI")
        if not v.startswith("data:image"):
            raise ValueError("url must be a data URI for base64 encoded images")
        return v


class ChatCompletionRequestMessageContentPartImage(BaseModel):
    type: Literal["image_url"]
    image_url: ChatCompletionRequestMessageContentPartImageURL


# Content Part Union
ChatCompletionRequestMessageContentPart = Union[
    ChatCompletionRequestMessageContentPartText, ChatCompletionRequestMessageContentPartImage
]


# Base Message Model
class BaseMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    name: Optional[str] = Field(
        None,
        max_length=MAX_MESSAGE_NAME_LENGTH,
        description="An optional name for the participant. Provides the model information to differentiate between participants of the same role.",
    )

    @field_validator("name")
    @classmethod
    def validate_name_format(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        # Allow alphanumeric, underscore, hyphen, period, space
        if not re.match(r'^[a-zA-Z0-9_\-. ]+$', v):
            raise ValueError("Name contains invalid characters (allowed: alphanumeric, underscore, hyphen, period, space)")
        return v


# Specific Message Types
class ChatCompletionSystemMessageParam(BaseMessage):
    role: Literal["system"]
    content: str

    @field_validator("content")
    @classmethod
    def validate_content_length(cls, v: str) -> str:
        if len(v) > MAX_MESSAGE_CONTENT_LENGTH:
            raise ValueError(
                f"System message content exceeds maximum length ({MAX_MESSAGE_CONTENT_LENGTH} chars)"
            )
        return v


class ChatCompletionUserMessageParam(BaseMessage):
    role: Literal["user"]
    content: Union[str, list[ChatCompletionRequestMessageContentPart]]

    @field_validator("content")
    @classmethod
    def validate_content_length(cls, v: Union[str, list]) -> Union[str, list]:
        if isinstance(v, str) and len(v) > MAX_MESSAGE_CONTENT_LENGTH:
            raise ValueError(
                f"User message content exceeds maximum length ({MAX_MESSAGE_CONTENT_LENGTH} chars)"
            )
        # Note: List content parts are validated by ChatCompletionRequestMessageContentPartText
        return v


# Maximum size for function call arguments (10KB)
MAX_FUNCTION_ARGUMENTS_SIZE = 10000


class FunctionCall(BaseModel):
    """
    Deprecated and replaced by `tool_calls`. The name and arguments of a function that should be called, as generated by the model.
    """

    arguments: str = Field(
        ..., description="The arguments to call the function with, as generated by the model in JSON format."
    )
    name: str = Field(..., description="The name of the function to call.")

    @field_validator("arguments")
    @classmethod
    def validate_arguments_json(cls, v: str) -> str:
        """Validate that arguments is valid JSON and within size limits."""
        if not isinstance(v, str):
            raise ValueError("Arguments must be a string")

        # Check size limit
        if len(v) > MAX_FUNCTION_ARGUMENTS_SIZE:
            raise ValueError(
                f"Function arguments exceed maximum size of {MAX_FUNCTION_ARGUMENTS_SIZE} characters"
            )

        # Validate JSON format (unless empty string)
        if v.strip():
            try:
                json.loads(v)
            except json.JSONDecodeError as e:
                raise ValueError(f"Function arguments must be valid JSON: {e}") from e

        return v


# Tool call payload (assistant tool invocation in message history)
class ToolCallFunctionPayload(BaseModel):
    """Represents a tool invocation payload in assistant message history.

    Accepts both OpenAI-style `arguments` and legacy `parameters` definitions.
    """

    name: str = Field(..., description="The name of the function to call.")
    arguments: Optional[str] = Field(
        None,
        description="Arguments to call the function with, as a JSON string.",
    )
    parameters: Optional[dict[str, Any]] = Field(
        default=None,
        description="Legacy JSON Schema parameters (accepted for backward compatibility).",
    )
    description: Optional[str] = Field(
        None,
        description="Optional function description (legacy compatibility).",
    )

    @field_validator("arguments")
    @classmethod
    def validate_tool_call_arguments_json(cls, v: Optional[str]) -> Optional[str]:
        """Validate that arguments is valid JSON and within size limits."""
        if v is None:
            return v
        if not isinstance(v, str):
            raise ValueError("Arguments must be a string")
        if len(v) > MAX_FUNCTION_ARGUMENTS_SIZE:
            raise ValueError(
                f"Function arguments exceed maximum size of {MAX_FUNCTION_ARGUMENTS_SIZE} characters"
            )
        if v.strip():
            try:
                json.loads(v)
            except json.JSONDecodeError as e:
                raise ValueError(f"Function arguments must be valid JSON: {e}") from e
        return v

    @field_validator("parameters")
    @classmethod
    def validate_tool_call_parameters_schema(cls, v: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """Reuse FunctionDefinition JSON Schema validation for legacy parameters."""
        return FunctionDefinition.validate_parameters_schema(v)


# Maximum length for tool call IDs
MAX_TOOL_CALL_ID_LENGTH = 64


class ChatCompletionMessageToolCallParam(BaseModel):
    id: str = Field(..., max_length=MAX_TOOL_CALL_ID_LENGTH, description="The ID of the tool call.")
    type: Literal["function"] = Field(..., description="The type of the tool. Currently, only `function` is supported.")
    function: ToolCallFunctionPayload = Field(
        ...,
        description="The function invocation payload generated by the model.",
    )

    @field_validator("id")
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        if not re.match(r'^[a-zA-Z0-9_\-]+$', v):
            raise ValueError("Tool call ID contains invalid characters (allowed: alphanumeric, underscore, hyphen)")
        return v


class ChatCompletionAssistantMessageParam(BaseMessage):
    role: Literal["assistant"]
    content: Optional[str] = Field(
        None,
        description="The contents of the assistant message. Required unless tool_calls or function_call is specified.",
    )
    tool_calls: Optional[list[ChatCompletionMessageToolCallParam]] = Field(
        None, description="The tool calls generated by the model, such as function calls."
    )
    # Deprecated function_call - include for compatibility if needed, but prefer tool_calls
    function_call: Optional[FunctionCall] = Field(
        None, deprecated=True, description="Deprecated and replaced by `tool_calls`."
    )

    @field_validator("content")
    @classmethod
    def validate_content_length(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) > MAX_MESSAGE_CONTENT_LENGTH:
            raise ValueError(
                f"Assistant message content exceeds maximum length ({MAX_MESSAGE_CONTENT_LENGTH} chars)"
            )
        return v

    @model_validator(mode="before")
    def check_content_or_tool_call(cls, values):
        content = values.get("content")
        tool_calls = values.get("tool_calls")
        function_call = values.get("function_call")  # Include deprecated field check if necessary
        if content is None and not tool_calls and not function_call:
            raise ValueError("Assistant message must have content or tool_calls (or deprecated function_call)")
        return values


class ChatCompletionToolMessageParam(BaseMessage):
    role: Literal["tool"]
    content: str = Field(..., description="The contents of the tool message.")
    tool_call_id: str = Field(
        ...,
        max_length=MAX_TOOL_CALL_ID_LENGTH,
        description="Tool call that this message is responding to."
    )

    @field_validator("content")
    @classmethod
    def validate_content_length(cls, v: str) -> str:
        if len(v) > MAX_MESSAGE_CONTENT_LENGTH:
            raise ValueError(
                f"Tool message content exceeds maximum length ({MAX_MESSAGE_CONTENT_LENGTH} chars)"
            )
        return v

    @field_validator("tool_call_id")
    @classmethod
    def validate_tool_call_id_format(cls, v: str) -> str:
        if not re.match(r'^[a-zA-Z0-9_\-]+$', v):
            raise ValueError("Tool call ID contains invalid characters (allowed: alphanumeric, underscore, hyphen)")
        return v


# Message Union
ChatCompletionMessageParam = Union[
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
]


# --- Response Format ---
class ResponseFormatJsonSchemaSpec(BaseModel):
    """Schema spec for structured output response format."""

    name: str = Field(..., description="Unique schema name for provider-side schema routing.")
    schema: dict[str, Any] = Field(..., description="JSON Schema object used to validate output.")
    strict: Optional[bool] = Field(None, description="Provider hint to enforce strict schema adherence.")


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"] = Field(
        "text", description="Must be one of `text`, `json_object`, or `json_schema`."
    )
    json_schema: Optional[ResponseFormatJsonSchemaSpec] = Field(
        None,
        description="Required when `type` is `json_schema`; must be omitted otherwise.",
    )

    @model_validator(mode="after")
    def validate_json_schema_requirements(self) -> "ResponseFormat":
        if self.type == "json_schema" and self.json_schema is None:
            raise ValueError("json_schema must be provided when type is 'json_schema'")
        if self.type != "json_schema" and self.json_schema is not None:
            raise ValueError("json_schema is only allowed when type is 'json_schema'")
        return self


# --- Continuation Controls (tldw extension) ---
class TLDWContinuationSpec(BaseModel):
    """Optional continuation controls for anchored append/branch generation."""

    from_message_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Anchor message ID to continue from.",
    )
    mode: Literal["branch", "append"] = Field(
        ...,
        description="Continuation mode: `branch` creates an alternate path; `append` continues at the current tip.",
    )
    assistant_prefill: Optional[str] = Field(
        None,
        max_length=MAX_MESSAGE_CONTENT_LENGTH,
        description="Optional assistant text prefix to prefill before generation.",
    )


class ResearchChatContextOutlineSection(BaseModel):
    """Bounded outline section attached from a completed deep research run."""

    title: str = Field(..., min_length=1, max_length=200)

    model_config = ConfigDict(extra="forbid")


class ResearchChatContextClaim(BaseModel):
    """Bounded key claim attached from a completed deep research run."""

    text: str = Field(..., min_length=1, max_length=1_000)

    model_config = ConfigDict(extra="forbid")


class ResearchChatContextVerificationSummary(BaseModel):
    """Compact verification summary attached from a completed deep research run."""

    unsupported_claim_count: Optional[int] = Field(default=None, ge=0)

    model_config = ConfigDict(extra="forbid")


class ResearchChatContextSourceTrustSummary(BaseModel):
    """Compact source-trust summary attached from a completed deep research run."""

    high_trust_count: Optional[int] = Field(default=None, ge=0)

    model_config = ConfigDict(extra="forbid")


class ResearchChatContext(BaseModel):
    """Bounded deep research context attached to a chat request."""

    run_id: str = Field(..., min_length=1, max_length=255)
    query: str = Field(..., min_length=1, max_length=2_000)
    question: str = Field(..., min_length=1, max_length=2_000)
    outline: list[ResearchChatContextOutlineSection] = Field(default_factory=list, max_length=7)
    key_claims: list[ResearchChatContextClaim] = Field(default_factory=list, max_length=5)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=5)
    verification_summary: Optional[ResearchChatContextVerificationSummary] = None
    source_trust_summary: Optional[ResearchChatContextSourceTrustSummary] = None
    research_url: str = Field(..., min_length=1, max_length=512)

    model_config = ConfigDict(extra="forbid")


class ChatCompletionResponseChoice(BaseModel):
    """One non-streaming chat completion choice in the OpenAI-compatible response."""

    index: Optional[int] = Field(default=None, description="Zero-based choice index.")
    message: Optional[dict[str, Any]] = Field(
        default=None,
        description="Assistant message payload returned by the selected provider.",
    )
    finish_reason: Optional[str] = Field(
        default=None,
        description="Provider-reported completion stop reason.",
    )
    logprobs: Optional[Any] = Field(
        default=None,
        description="Provider-specific log probability payload when requested.",
    )

    model_config = ConfigDict(extra="allow")


class ChatCompletionUsage(BaseModel):
    """Token usage summary for a non-streaming chat completion response."""

    prompt_tokens: Optional[int] = Field(default=None, ge=0)
    completion_tokens: Optional[int] = Field(default=None, ge=0)
    total_tokens: Optional[int] = Field(default=None, ge=0)

    model_config = ConfigDict(extra="allow")


class ChatCompletionResponse(BaseModel):
    """Document the non-streaming OpenAI-compatible chat completion response shape."""

    id: Optional[str] = Field(default=None, description="Provider completion identifier.")
    object: Optional[str] = Field(default=None, description="OpenAI-compatible object type.")
    created: Optional[int] = Field(default=None, description="Unix timestamp for response creation.")
    model: Optional[str] = Field(default=None, description="Model used for generation.")
    choices: list[ChatCompletionResponseChoice] = Field(
        default_factory=list,
        description="Completion choices returned by the selected provider.",
    )
    usage: Optional[ChatCompletionUsage] = Field(
        default=None,
        description="Token usage reported by the selected provider.",
    )
    tldw_conversation_id: Optional[str] = Field(
        default=None,
        description="Conversation identifier persisted by tldw_server when chat history is enabled.",
    )
    meta: Optional[dict[str, Any]] = Field(
        default=None,
        description="tldw_server metadata such as persona diagnostics.",
    )

    model_config = ConfigDict(extra="allow")


# --- Main Request Model ---
class ChatCompletionRequest(BaseModel):
    """
    Model representing the request body for the /v1/chat/completions endpoint.
    Acts as a proxy request, routing to different LLM providers.
    """

    # --- Routing Parameter ---
    api_provider: Optional[SUPPORTED_API_ENDPOINTS] = Field(
        default=None,  # Default is handled server-side
        description=f"[Extension] The target LLM provider (e.g., 'openai', 'anthropic'). If omitted, defaults to the server's configured default ('{DEFAULT_LLM_PROVIDER}').",
    )
    routing: Optional[RoutingOverride] = Field(
        None,
        description=(
            "[Extension] Optional model-router overrides when `model='auto'`, including routing "
            "mode, objective, provider boundary, and failure handling."
        ),
    )

    @field_validator("api_provider")
    @classmethod
    def validate_api_provider(cls, value: Optional[str]) -> Optional[str]:
        """Validate provider ids without Python 3.11-only dynamic Literal syntax."""
        if value is None:
            return value
        if value in ALL_SUPPORTED_PROVIDER_NAMES_LIST:
            return value
        raise ValueError(f"Unsupported api_provider: {value}")

    # --- Standard OpenAI-like Parameters ---
    model: Optional[str] = Field(
        None, description="ID of the model to use. Specific model compatibility depends on the selected `api_provider`."
    )
    messages: list[ChatCompletionMessageParam] = Field(
        ..., description="A list of messages comprising the conversation so far.", min_length=1
    )
    frequency_penalty: Optional[float] = Field(
        None, ge=-2.0, le=2.0, description="Frequency penalty parameter (provider support varies)."
    )
    logit_bias: Optional[dict[str, float]] = Field(None, description="Logit bias parameter (provider support varies).")
    logprobs: Optional[bool] = Field(
        False, description="Whether to return log probabilities (provider support varies)."
    )
    top_logprobs: Optional[int] = Field(
        None,
        ge=0,
        le=20,
        description="Number of top log probabilities to return (provider support varies). `logprobs` must be true.",
    )
    max_tokens: Optional[int] = Field(
        None, ge=1, description="Maximum number of tokens to generate (provider support varies)."
    )
    n: Optional[int] = Field(
        1, ge=1, le=128, description="Number of completions to generate (provider support varies)."
    )
    presence_penalty: Optional[float] = Field(
        None, ge=-2.0, le=2.0, description="Presence penalty parameter (provider support varies)."
    )
    response_format: Optional[ResponseFormat] = Field(
        None, description="Response format specification (e.g., JSON mode, provider support varies)."
    )
    seed: Optional[int] = Field(None, description="Seed for deterministic sampling (provider support varies).")
    stop: Optional[Union[str, list[str]]] = Field(None, description="Stop sequences (provider support varies).")
    stream: Optional[bool] = Field(False, description="Whether to stream the response.")
    temperature: Optional[float] = Field(
        None, ge=0.0, le=2.0, description="Sampling temperature (provider support varies)."
    )
    top_p: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Top-p (nucleus) sampling parameter (provider support varies)."
    )
    tools: Optional[list[Union[ToolDefinition, GeminiNativeToolDefinition]]] = Field(
        None, max_length=128, description="Tools the model may call (provider support varies)."
    )
    tool_choice: Optional[Union[Literal["none", "auto", "required"], ToolChoiceOption]] = Field(
        None,
        description=(
            "Controls tool usage (provider support varies). Only valid when `tools` are provided; "
            "if omitted, providers default to their standard behavior (typically `auto`)."
        ),
    )
    chat_loop_mode: Optional[Literal["legacy", "enabled"]] = Field(
        None,
        description=(
            "[Extension] Optional loop engine routing hint. `enabled` requests server-driven chat loop mode; "
            "`legacy` keeps existing chat execution behavior."
        ),
    )
    user: Optional[str] = Field(None, description="End-user identifier for monitoring.")
    research_context: Optional[ResearchChatContext] = Field(
        None,
        description=(
            "[Extension] Optional bounded deep research context attached from a completed run. "
            "Used as model-side working context and not persisted as transcript content."
        ),
    )

    # --- Slash Commands Injection Override ---
    slash_command_injection_mode: Optional[Literal["system", "preface", "replace"]] = Field(
        None,
        description="[Extension] Override the server's slash command injection behavior for this request."
        " Options: 'system' (default server behavior), 'preface', or 'replace'.",
    )

    # --- Conversation history controls ---
    history_message_limit: Optional[int] = Field(
        None, ge=1, le=500, description="Optional override for the number of previous messages to load into context."
    )
    history_message_order: Optional[Literal["asc", "desc"]] = Field(
        None, description="Optional override for history ordering: 'asc' for oldest first, 'desc' for newest first."
    )

    # --- Bedrock Guardrails Extensions ---
    extra_headers: Optional[dict[str, str]] = Field(
        None,
        description="Provider-specific additional headers to include via the request body (e.g., Bedrock guardrails)."
        " For Bedrock, include keys: 'X-Amzn-Bedrock-GuardrailIdentifier',"
        " 'X-Amzn-Bedrock-GuardrailVersion', and optional 'X-Amzn-Bedrock-Trace'.",
    )
    extra_body: Optional[dict[str, Any]] = Field(
        None,
        description="Provider-specific extra body content. For Bedrock guardrails, include"
        " 'amazon-bedrock-guardrailConfig': { 'tagSuffix': '...'} if needed.",
    )
    billing_prompt_cache_intent: Optional[dict[str, Any]] = Field(
        None,
        description=(
            "[Extension] Explicit billing prompt-cache intent. Disabled unless enabled=true; "
            "provider adapters translate only documented cache controls and report intent separately from usage."
        ),
    )
    thinking_budget_tokens: Optional[int] = Field(
        None,
        ge=0,
        description="[llama.cpp extension] App-level thinking budget in tokens. Only valid when the resolved target provider is llama.cpp.",
    )
    grammar_mode: Optional[Literal["none", "library", "inline"]] = Field(
        None,
        description="[llama.cpp extension] How to resolve the outbound grammar payload.",
    )
    grammar_id: Optional[str] = Field(
        None,
        min_length=1,
        max_length=128,
        description="[llama.cpp extension] Saved grammar identifier when grammar_mode='library'.",
    )
    grammar_inline: Optional[str] = Field(
        None,
        max_length=200_000,
        description="[llama.cpp extension] Inline grammar when grammar_mode='inline'.",
    )
    grammar_override: Optional[str] = Field(
        None,
        max_length=200_000,
        description="[llama.cpp extension] Optional per-request override applied on top of a saved grammar selection.",
    )

    # --- Extended Parameters for chat_api_call ---
    minp: Optional[float] = Field(None, description="[Extension] Minimum probability threshold (provider specific).")
    topk: Optional[int] = Field(None, description="[Extension] Top-K sampling parameter (provider specific).")
    # topp: Optional[float] = Field(None, description="[Extension] Explicit Top-P if needed separately from top_p.") # Uncomment if needed

    # --- Prompt templating ---
    prompt_template_name: Optional[str] = Field(
        None,
        description="Name of the prompt template to apply. Must contain only alphanumeric characters, underscores, and hyphens.",
    )

    # --- Optional Character Chat Parameters ---
    character_id: Optional[str] = Field(None, description="Optional ID of the character to use for context.")
    persona_id: Optional[str] = Field(
        None,
        description=(
            "[Compatibility] Optional persona alias. Accepted only when it can be deterministically resolved to "
            "a character ID. This does not select a Persona Garden persona-backed chat assistant. "
            "Deprecated as of 2026-02-09; planned removal date: 2026-07-01."
        ),
    )
    conversation_id: Optional[str] = Field(None, description="Optional ID of the conversation to use for context.")
    tldw_continuation: Optional[TLDWContinuationSpec] = Field(
        None,
        description=(
            "[Extension] Optional continuation controls for anchored generation. "
            "Requires `conversation_id` when provided."
        ),
    )
    persona_exemplar_budget_tokens: Optional[int] = Field(
        None,
        ge=1,
        le=20_000,
        description="[Extension] Optional override for persona exemplar selection token budget.",
    )
    persona_exemplar_strategy: Optional[Literal["off", "default", "hybrid", "embeddings"]] = Field(
        None,
        description="[Extension] Persona exemplar selection strategy. 'off' disables selection for this request.",
    )
    persona_debug: Optional[bool] = Field(
        False,
        description="[Extension] Include persona selection debug metadata in response payload when available.",
    )
    save_to_db: Optional[bool] = Field(
        None,
        description=(
            "[Extension] If true, persist conversation and messages to the database. "
            "If omitted, the server uses its configured default (see Chat-Module.chat_save_default/default_save_to_db)."
        ),
    )
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Summarize the key points of this project."}],
                "stream": False,
                "api_provider": "openai",
            }
        },
    )

    @field_validator("prompt_template_name")
    @classmethod
    def validate_template_name(cls, v: Optional[str]) -> Optional[str]:
        """Validate prompt template name to prevent path traversal attacks."""
        if v is None:
            return v

        import re

        # Only allow alphanumeric, underscore, and hyphen
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Invalid template name format. Only alphanumeric characters, underscores, and hyphens are allowed."
            )

        # Additional security check for path traversal patterns
        if "/" in v or "\\" in v or ".." in v:
            raise ValueError("Invalid template name. Path traversal patterns are not allowed.")

        return v

    @model_validator(mode="before")
    def check_logprobs(cls, values):
        if isinstance(values, dict):
            tools = values.get("tools")
            tool_choice = values.get("tool_choice")
            if isinstance(tools, list) and len(tools) == 0:
                values["tools"] = None
                if isinstance(tool_choice, str) and tool_choice.strip().lower() == "auto":
                    values["tool_choice"] = "none"
            if values.get("tldw_continuation") and not values.get("conversation_id"):
                raise ValueError("conversation_id is required when tldw_continuation is provided")
        logprobs = values.get("logprobs")
        top_logprobs = values.get("top_logprobs")
        if top_logprobs is not None and not logprobs:
            raise ValueError("If top_logprobs is specified, logprobs must be set to true.")
        return values

    @model_validator(mode="after")
    def validate_llamacpp_grammar_fields(self) -> "ChatCompletionRequest":
        if self.grammar_mode == "library" and not self.grammar_id:
            raise ValueError("grammar_id is required when grammar_mode is 'library'")
        if self.grammar_mode == "inline" and not self.grammar_inline:
            raise ValueError("grammar_inline is required when grammar_mode is 'inline'")
        return self


#
# --- RAG Context Schemas for Citation Persistence ---
# These schemas define the structure for storing RAG context with messages


class RagContextDocument(BaseModel):
    """A single retrieved document within a RAG context."""

    id: Optional[str] = Field(None, description="Document or media ID")
    source_type: Optional[str] = Field(
        None,
        description="Type of source: media_db, notes, characters, chats, kanban"
    )
    title: Optional[str] = Field(None, description="Document title")
    score: Optional[float] = Field(None, description="Relevance score (0-1)")
    chunk_id: Optional[str] = Field(None, description="Chunk identifier if chunk-level retrieval")
    excerpt: Optional[str] = Field(None, description="Relevant excerpt from the document")
    url: Optional[str] = Field(None, description="URL or path to original document")
    page_number: Optional[int] = Field(None, description="Page number if applicable")
    line_range: Optional[list[int]] = Field(
        None,
        description="Line range [start, end] of the excerpt in the source",
        min_length=2,
        max_length=2
    )
    metadata: Optional[dict[str, Any]] = Field(
        None,
        description="Additional metadata from the source"
    )

    model_config = ConfigDict(extra="allow")


class RagContext(BaseModel):
    """
    RAG context to be stored with a message for citation persistence.
    This is stored in message_metadata.extra_json under the 'rag_context' key.
    """

    search_query: str = Field(..., description="The original search query")
    search_mode: Optional[str] = Field(
        "hybrid",
        description="Search mode used: fts, vector, or hybrid"
    )
    settings_snapshot: Optional[dict[str, Any]] = Field(
        None,
        description="Snapshot of key RAG settings used for this search"
    )
    retrieved_documents: list[RagContextDocument] = Field(
        default_factory=list,
        description="List of retrieved documents with their metadata and scores"
    )
    generated_answer: Optional[str] = Field(
        None,
        description="The AI-generated answer (if generation was enabled)"
    )
    citations: Optional[list[dict[str, Any]]] = Field(
        None,
        description="Citation metadata (academic and chunk-level)"
    )
    claims_verified: Optional[list[dict[str, Any]]] = Field(
        None,
        description="Results from claims verification if enabled"
    )
    timestamp: Optional[str] = Field(
        None,
        description="ISO timestamp of when the search was performed"
    )
    feedback_id: Optional[str] = Field(
        None,
        description="Feedback tracking ID for analytics"
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "search_query": "What are the key findings from the Q3 report?",
                "search_mode": "hybrid",
                "settings_snapshot": {
                    "top_k": 10,
                    "enable_reranking": True,
                    "enable_citations": True
                },
                "retrieved_documents": [
                    {
                        "id": "media_123",
                        "source_type": "media_db",
                        "title": "Q3 Financial Report 2025",
                        "score": 0.92,
                        "excerpt": "Revenue increased by 15% compared to Q2...",
                        "page_number": 5
                    }
                ],
                "generated_answer": "The key findings from the Q3 report include...",
                "timestamp": "2025-01-25T12:00:00Z"
            }
        }
    )


#
# End of chat_request_schemas.py
#######################################################################################################################
