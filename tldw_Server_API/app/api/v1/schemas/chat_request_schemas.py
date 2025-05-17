# Server_API/app/api/schemas/chat_request_models.py
# Description: This code provides schema models for the /chat API endpoints
#
# Imports
import os
from typing import Optional, Dict, Any, Literal, Union, List

import toml
from dotenv import load_dotenv
#
# 3rd-party imports
from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator
#
# Local Imports
#
#######################################################################################################################
#
# Functions:

# --- Pydantic Models for OpenAI Chat Completion Request ---
# Based on https://platform.openai.com/docs/api-reference/chat/create

DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openai") # Default if not set

# Config Loading
load_dotenv()
try:
    _config = toml.load("config.toml")
except FileNotFoundError:
    _config = {}

def _get_setting(env_var, section, key, default=None):
    return os.getenv(env_var) or _config.get(section, {}).get(key, default)

API_KEYS = {
    name: _get_setting(
        f"{name.upper().replace('.', '_')}_API_KEY",
        "api_keys",
        name
    )
    for name in [
        "openai",
        "anthropic",
        "cohere",
        "groq",
        "openrouter",
        "deepseek",
        "mistral",
        "google",
        "huggingface",
        "llama.cpp",
        "kobold",
        "ooba",
        "tabbyapi",
        "vllm",
        "local-llm",
        "custom-openai-api",
        "custom-openai-api-2"
    ]
}

SUPPORTED_API_ENDPOINTS = Literal[
    "anthropic",
    "cohere",
    "deepseek",
    "google",
    "groq",
    "huggingface",
    "mistral",
    "openai",
    "openrouter",
    "llama.cpp",
    "kobold",
    "ollama",
    "ooba",
    "tabbyapi",
    "vllm",
    "local-llm",
    "custom-openai-api",
    "custom-openai-api-2"
]


# --- Tool Definitions ---
class FunctionDefinition(BaseModel):
    """Describes a function available to the model."""
    name: str = Field(..., description="The name of the function to be called. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.")
    description: Optional[str] = Field(None, description="A description of what the function does, used by the model to choose when and how to call the function.")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="The parameters the functions accepts, described as a JSON Schema object. See the guide[1] for examples, and the JSON Schema reference[2] for documentation about the format. Omitting parameters defines a function with an empty parameter list. [1] https://platform.openai.com/docs/guides/function-calling [2] https://json-schema.org/understanding-json-schema/")

class ToolDefinition(BaseModel):
    """Definition of a tool the model can use."""
    type: Literal["function"] = Field(..., description="The type of the tool. Currently, only `function` is supported.")
    function: FunctionDefinition

class ToolChoiceFunction(BaseModel):
    """Specifies a specific function to be called."""
    name: str = Field(..., description="The name of the function to call.")

class ToolChoiceOption(BaseModel):
    """Specifies a tool the model should use. Use to force the model to call a specific function."""
    type: Literal["function"] = Field(..., description="The type of the tool. Currently, only `function` is supported.")
    function: ToolChoiceFunction

# --- Message Definitions ---
class ChatCompletionRequestMessageContentPartText(BaseModel):
    type: Literal["text"]
    text: str

class ChatCompletionRequestMessageContentPartImageURL(BaseModel):
    url: Union[HttpUrl, str] = Field(..., description="Either a URL of the image or the base64 encoded image data.")
    detail: Optional[Literal["auto", "low", "high"]] = Field("auto", description="Specifies the detail level of the image.")

    @field_validator('url')
    def check_url_or_data(cls, v):
        if isinstance(v, str) and not v.startswith('data:image'):
             raise ValueError('String url must be a data URI for base64 encoded images')
        return v

class ChatCompletionRequestMessageContentPartImage(BaseModel):
    type: Literal["image_url"]
    image_url: ChatCompletionRequestMessageContentPartImageURL

# Content Part Union
ChatCompletionRequestMessageContentPart = Union[
    ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestMessageContentPartImage
]

# Base Message Model
class BaseMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    name: Optional[str] = Field(None, description="An optional name for the participant. Provides the model information to differentiate between participants of the same role.")

# Specific Message Types
class ChatCompletionSystemMessageParam(BaseMessage):
    role: Literal["system"]
    content: str

class ChatCompletionUserMessageParam(BaseMessage):
    role: Literal["user"]
    content: Union[str, List[ChatCompletionRequestMessageContentPart]]

class FunctionCall(BaseModel):
    """
    Deprecated and replaced by `tool_calls`. The name and arguments of a function that should be called, as generated by the model.
    """
    arguments: str = Field(..., description="The arguments to call the function with, as generated by the model in JSON format.")
    name: str = Field(..., description="The name of the function to call.")

class ChatCompletionMessageToolCallParam(BaseModel):
    id: str = Field(..., description="The ID of the tool call.")
    type: Literal["function"] = Field(..., description="The type of the tool. Currently, only `function` is supported.")
    function: FunctionDefinition = Field(..., description="The function that the model called.")


class ChatCompletionAssistantMessageParam(BaseMessage):
    role: Literal["assistant"]
    content: Optional[str] = Field(None, description="The contents of the assistant message. Required unless tool_calls or function_call is specified.")
    tool_calls: Optional[List[ChatCompletionMessageToolCallParam]] = Field(None, description="The tool calls generated by the model, such as function calls.")
    # Deprecated function_call - include for compatibility if needed, but prefer tool_calls
    function_call: Optional[FunctionCall] = Field(None, deprecated=True, description="Deprecated and replaced by `tool_calls`.")

    @model_validator(mode='before')
    def check_content_or_tool_call(cls, values):
        content = values.get('content')
        tool_calls = values.get('tool_calls')
        function_call = values.get('function_call') # Include deprecated field check if necessary
        if content is None and not tool_calls and not function_call:
            raise ValueError('Assistant message must have content or tool_calls (or deprecated function_call)')
        return values


class ChatCompletionToolMessageParam(BaseMessage):
    role: Literal["tool"]
    content: str = Field(..., description="The contents of the tool message.")
    tool_call_id: str = Field(..., description="Tool call that this message is responding to.")

# Message Union
ChatCompletionMessageParam = Union[
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
]

# --- Response Format ---
class ResponseFormat(BaseModel):
    type: Literal["text", "json_object"] = Field("text", description="Must be one of `text` or `json_object`.")


# --- Main Request Model ---
class ChatCompletionRequest(BaseModel):
    """
    Model representing the request body for the /v1/chat/completions endpoint.
    Acts as a proxy request, routing to different LLM providers.
    """
    # --- Routing Parameter ---
    api_provider: Optional[SUPPORTED_API_ENDPOINTS] = Field(
        default=None, # Default is handled server-side
        description=f"[Extension] The target LLM provider (e.g., 'openai', 'anthropic'). If omitted, defaults to the server's configured default ('{DEFAULT_LLM_PROVIDER}')."
    )

    # --- Standard OpenAI-like Parameters ---
    model: str = Field(None, description="ID of the model to use. Specific model compatibility depends on the selected `api_provider`.")
    messages: List[ChatCompletionMessageParam] = Field(..., description="A list of messages comprising the conversation so far.", min_length=1)
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0, description="Frequency penalty parameter (provider support varies).")
    logit_bias: Optional[Dict[str, float]] = Field(None, description="Logit bias parameter (provider support varies).")
    logprobs: Optional[bool] = Field(False, description="Whether to return log probabilities (provider support varies).")
    top_logprobs: Optional[int] = Field(None, ge=0, le=20, description="Number of top log probabilities to return (provider support varies). `logprobs` must be true.")
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate (provider support varies).")
    n: Optional[int] = Field(1, ge=1, le=128, description="Number of completions to generate (provider support varies).")
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0, description="Presence penalty parameter (provider support varies).")
    response_format: Optional[ResponseFormat] = Field(None, description="Response format specification (e.g., JSON mode, provider support varies).")
    seed: Optional[int] = Field(None, description="Seed for deterministic sampling (provider support varies).")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences (provider support varies).")
    stream: Optional[bool] = Field(False, description="Whether to stream the response.")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature (provider support varies).")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Top-p (nucleus) sampling parameter (provider support varies).")
    tools: Optional[List[ToolDefinition]] = Field(None, max_length=128, description="Tools the model may call (provider support varies).")
    tool_choice: Optional[Union[Literal["none", "auto", "required"], ToolChoiceOption]] = Field("auto", description="Controls tool usage (provider support varies).")
    user: Optional[str] = Field(None, description="End-user identifier for monitoring.")

    # --- Extended Parameters for chat_api_call ---
    minp: Optional[float] = Field(None, description="[Extension] Minimum probability threshold (provider specific).")
    topk: Optional[int] = Field(None, description="[Extension] Top-K sampling parameter (provider specific).")
    # topp: Optional[float] = Field(None, description="[Extension] Explicit Top-P if needed separately from top_p.") # Uncomment if needed

    # --- Prompt templating ---
    prompt_template_name: Optional[str] = Field(None, description="Name of the prompt template to apply.")
    character_id: Optional[str] = Field(None, description="Optional ID of the character to use for context and templating.")

    class Config:
        extra = "allow"

    @model_validator(mode='before')
    def check_logprobs(cls, values):
        logprobs = values.get('logprobs')
        top_logprobs = values.get('top_logprobs')
        if top_logprobs is not None and not logprobs:
            raise ValueError("If top_logprobs is specified, logprobs must be set to true.")
        return values

#
# End of chat_request_schemas.py
#######################################################################################################################
