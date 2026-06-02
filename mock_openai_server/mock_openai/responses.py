"""
Response generation and management for the mock OpenAI server.

Handles loading response templates and generating appropriate responses.
"""

import json
import random
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from string import Template

from .models import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionUsage,
    ChatMessage,
    EmbeddingResponse,
    EmbeddingData,
    EmbeddingUsage,
    CompletionResponse,
    CompletionChoice,
    CompletionUsage,
    ErrorResponse,
    ErrorDetail
)


class ResponseManager:
    """Manages response generation for the mock server."""

    def __init__(self, responses_dir: Path = None):
        """Initialize the response manager."""
        self.responses_dir = responses_dir or Path("responses")
        self.template_vars = {}
        self.response_cache = {}

    def load_response_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load a response from a JSON file."""
        file_path = Path(file_path)

        # Check if path is absolute or relative to responses_dir
        if not file_path.is_absolute():
            file_path = self.responses_dir / file_path

        # Cache responses to avoid repeated file reads
        cache_key = str(file_path)
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]

        if not file_path.exists():
            # Return a default response if file not found
            return self.get_default_chat_response()

        with open(file_path, 'r') as f:
            content = f.read()

            # Substitute template variables
            template = Template(content)
            content = template.safe_substitute(self.get_template_vars())

            response = json.loads(content)
            self.response_cache[cache_key] = response
            return response

    def get_template_vars(self) -> Dict[str, Any]:
        """Get template variables for response substitution."""
        return {
            "timestamp": int(time.time()),
            "request_id": f"req-{uuid.uuid4().hex[:12]}",
            "chat_id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "model": "gpt-4",
            **self.template_vars
        }

    def set_template_var(self, key: str, value: Any):
        """Set a custom template variable."""
        self.template_vars[key] = value

    def get_default_chat_response(self) -> Dict[str, Any]:
        """Get a default chat completion response."""
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a mock response from the OpenAI API mock server."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 15,
                "total_tokens": 25
            }
        }

    def get_default_embedding_response(self) -> Dict[str, Any]:
        """Get a default embedding response."""
        # Generate a random 1536-dimensional embedding (default for text-embedding-ada-002)
        # Mock-only embedding values; not used for secrets or cryptography.
        embedding = [random.random() for _ in range(1536)]  # nosec B311

        return {
            "object": "list",
            "data": [{
                "index": 0,
                "embedding": embedding,
                "object": "embedding"
            }],
            "model": "text-embedding-ada-002",
            "usage": {
                "prompt_tokens": 8,
                "total_tokens": 8
            }
        }

    def get_default_completion_response(self) -> Dict[str, Any]:
        """Get a default completion response."""
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "gpt-3.5-turbo-instruct",
            "choices": [{
                "text": " This is a mock completion response.",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "total_tokens": 15
            }
        }

    def generate_chat_response(
        self,
        request_data: Dict[str, Any],
        response_file: Optional[str] = None
    ) -> ChatCompletionResponse:
        """Generate a chat completion response."""
        if response_file:
            response_data = self.load_response_file(response_file)
        else:
            response_data = self.get_default_chat_response()
            response_data["model"] = request_data.get("model", "gpt-4")

        # Update model from request if not specified in response
        if "model" not in response_data or not response_data["model"]:
            response_data["model"] = request_data.get("model", "gpt-4")

        # Create response object
        choices = []
        for choice_data in response_data.get("choices", []):
            message = ChatMessage(
                role=choice_data["message"]["role"],
                content=choice_data["message"].get("content"),
                function_call=choice_data["message"].get("function_call"),
                tool_calls=choice_data["message"].get("tool_calls")
            )
            choices.append(ChatCompletionResponseChoice(
                index=choice_data["index"],
                message=message,
                finish_reason=choice_data.get("finish_reason", "stop")
            ))

        usage = None
        if "usage" in response_data:
            usage = ChatCompletionUsage(
                prompt_tokens=response_data["usage"]["prompt_tokens"],
                completion_tokens=response_data["usage"]["completion_tokens"],
                total_tokens=response_data["usage"]["total_tokens"]
            )

        return ChatCompletionResponse(
            id=response_data.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}"),
            created=response_data.get("created", int(time.time())),
            model=response_data["model"],
            choices=choices,
            usage=usage,
            system_fingerprint=response_data.get("system_fingerprint")
        )

    def generate_embedding_response(
        self,
        request_data: Dict[str, Any],
        response_file: Optional[str] = None
    ) -> EmbeddingResponse:
        """Generate an embedding response."""
        if response_file:
            response_data = self.load_response_file(response_file)
        else:
            response_data = self.get_default_embedding_response()

        # Update model from request if not specified
        model = response_data.get("model", request_data.get("model", "text-embedding-ada-002"))

        # Handle multiple inputs
        input_data = request_data.get("input", "")
        if isinstance(input_data, str):
            input_list = [input_data]
        elif isinstance(input_data, list):
            input_list = input_data
        else:
            input_list = [str(input_data)]

        # Generate embeddings for each input
        embeddings = []
        fixture_embeddings = response_data.get("data", [])
        for i, _ in enumerate(input_list):
            if i < len(fixture_embeddings):
                embedding_data = fixture_embeddings[i]
            elif fixture_embeddings:
                embedding_data = {
                    **fixture_embeddings[-1],
                    "index": i,
                }
            else:
                # Generate random embedding for additional inputs
                embedding_data = {
                    "embedding": [random.random() for _ in range(1536)],  # nosec B311
                    "index": i
                }

            embeddings.append(EmbeddingData(
                index=embedding_data.get("index", i),
                embedding=embedding_data["embedding"],
                object="embedding"
            ))

        usage = EmbeddingUsage(
            prompt_tokens=response_data.get("usage", {}).get("prompt_tokens", len(input_list) * 8),
            total_tokens=response_data.get("usage", {}).get("total_tokens", len(input_list) * 8)
        )

        return EmbeddingResponse(
            object="list",
            data=embeddings,
            model=model,
            usage=usage
        )

    def generate_completion_response(
        self,
        request_data: Dict[str, Any],
        response_file: Optional[str] = None
    ) -> CompletionResponse:
        """Generate a completion response."""
        if response_file:
            response_data = self.load_response_file(response_file)
        else:
            response_data = self.get_default_completion_response()

        # Update model from request if not specified
        model = response_data.get("model", request_data.get("model", "gpt-3.5-turbo-instruct"))

        choices = []
        for choice_data in response_data.get("choices", []):
            choices.append(CompletionChoice(
                text=choice_data["text"],
                index=choice_data["index"],
                logprobs=choice_data.get("logprobs"),
                finish_reason=choice_data.get("finish_reason", "stop")
            ))

        usage = None
        if "usage" in response_data:
            usage = CompletionUsage(
                prompt_tokens=response_data["usage"]["prompt_tokens"],
                completion_tokens=response_data["usage"]["completion_tokens"],
                total_tokens=response_data["usage"]["total_tokens"]
            )

        return CompletionResponse(
            id=response_data.get("id", f"cmpl-{uuid.uuid4().hex[:12]}"),
            created=response_data.get("created", int(time.time())),
            model=model,
            choices=choices,
            usage=usage,
            system_fingerprint=response_data.get("system_fingerprint")
        )

    def generate_error_response(
        self,
        message: str,
        error_type: str = "invalid_request_error",
        param: Optional[str] = None,
        code: Optional[str] = None
    ) -> ErrorResponse:
        """Generate an error response."""
        return ErrorResponse(
            error=ErrorDetail(
                message=message,
                type=error_type,
                param=param,
                code=code
            )
        )
