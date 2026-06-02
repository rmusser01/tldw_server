"""
FastAPI server implementation for the mock OpenAI API.

Provides OpenAI-compatible endpoints for testing purposes.
"""

import argparse
import asyncio
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from .config import MockConfig, ResponsePattern, load_config, get_config
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    CompletionRequest,
    CompletionResponse,
    ModelsResponse,
    ModelInfo,
    ErrorResponse,
    ErrorDetail
)
from .responses import ResponseManager
from .streaming import StreamingResponseGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Mock OpenAI API Server",
    description="A mock OpenAI API server for testing purposes",
    version="1.0.0"
)

from fastapi import Depends


def get_config_instance() -> MockConfig:
    """Get the current configuration instance."""
    return get_config()


def get_response_manager(config: MockConfig = Depends(get_config_instance)) -> ResponseManager:
    """Get the response manager instance."""
    return ResponseManager(responses_dir=config.response_base_dir)


def get_streaming_generator(config: MockConfig = Depends(get_config_instance)) -> StreamingResponseGenerator:
    """Get the streaming generator instance."""
    return StreamingResponseGenerator(
        chunk_delay_ms=config.streaming.chunk_delay_ms,
        words_per_chunk=config.streaming.words_per_chunk
    )


def reset_scenario_failure_counts(config: MockConfig) -> None:
    """Reset deterministic scenario failure counters."""
    config.scenario_failure_counts.clear()


def maybe_raise_scenario_failure(
    endpoint: str,
    request_data: Dict[str, Any],
    config: MockConfig,
) -> None:
    """Raise a configured deterministic scenario failure for matching requests."""
    for index, failure in enumerate(config.scenario_failures.get(endpoint, [])):
        if not ResponsePattern(match=failure.match, response_file="").matches(request_data):
            continue

        key: Tuple[str, int] = (endpoint, index)
        used = config.scenario_failure_counts.get(key, 0)
        if used >= max(0, failure.times):
            continue

        config.scenario_failure_counts[key] = used + 1
        raise HTTPException(
            status_code=failure.status_code,
            detail={
                "error": {
                    "message": failure.message,
                    "type": failure.error_type,
                    "code": failure.code,
                }
            },
        )


# Add CORS middleware right after app creation
config = get_config_instance()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize the server on startup."""
    config = get_config_instance()
    reset_scenario_failure_counts(config)

    logger.info(f"Mock OpenAI server started on {config.server.host}:{config.server.port}")
    logger.info(f"Streaming enabled: {config.streaming.enabled}")
    logger.info(f"CORS origins: {config.server.cors_origins}")


def validate_api_key(authorization: Optional[str] = Header(None)) -> bool:
    """Validate the API key (mock validation, always returns True)."""
    if not authorization:
        return False

    if not authorization.startswith("Bearer "):
        return False

    # Mock validation - accept any key starting with "sk-"
    api_key = authorization.replace("Bearer ", "")
    if not api_key.startswith("sk-"):
        logger.warning(f"Invalid API key format: {api_key[:10]}...")
        return False

    return True


def should_simulate_error(config: MockConfig) -> bool:
    """Determine if we should simulate an error based on configuration."""
    if not config.server.simulate_errors:
        return False

    return random.random() < config.server.error_rate


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests if enabled."""
    config = get_config_instance()
    if config.server.log_requests:
        logger.info(f"{request.method} {request.url.path}")

        # Log request body for debugging (be careful with sensitive data)
        # Note: This reads the entire body into memory twice, which has a performance cost
        # but is acceptable for a mock server used in testing environments
        if request.method == "POST":
            body = await request.body()
            if body:
                try:
                    body_json = json.loads(body)
                    logger.debug(f"Request body: {json.dumps(body_json, indent=2)}")
                except json.JSONDecodeError:
                    logger.debug(f"Request body (raw): {body[:200]}...")

            # IMPORTANT: We must recreate the request with the body we read
            # because FastAPI endpoints need to read the body again
            from starlette.datastructures import Headers
            from starlette.requests import Request as StarletteRequest

            async def receive():
                return {"type": "http.request", "body": body}

            request = StarletteRequest(request.scope, receive)

    response = await call_next(request)
    return response


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Mock OpenAI API Server",
        "version": "1.0.0",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "embeddings": "/v1/embeddings",
            "models": "/v1/models",
            "completions": "/v1/completions"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
    config: MockConfig = Depends(get_config_instance),
    response_manager: ResponseManager = Depends(get_response_manager),
    streaming_generator: StreamingResponseGenerator = Depends(get_streaming_generator)
):
    """Chat completions endpoint."""
    # Validate API key
    if not validate_api_key(authorization):
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Invalid API key", "type": "authentication_error"}}
        )

    # Convert request to dict for pattern matching
    request_data = request.model_dump()

    maybe_raise_scenario_failure("chat_completions", request_data, config)

    # Simulate errors if configured
    if should_simulate_error(config):
        error_response = response_manager.generate_error_response(
            message="Simulated error for testing",
            error_type="server_error",
            code="simulated_error"
        )
        raise HTTPException(status_code=500, detail=error_response.model_dump())

    # Find matching response file
    response_file = None
    if "chat_completions" in config.responses:
        response_file = config.responses["chat_completions"].find_matching_response(request_data)

    # Handle streaming response
    if request.stream and config.streaming.enabled:
        # Generate response data first
        response = response_manager.generate_chat_response(request_data, response_file)
        response_data = response.model_dump()

        # Convert to streaming response
        async def generate():
            async for chunk in streaming_generator.generate_stream_from_response(response_data):
                yield chunk

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable Nginx buffering
            }
        )

    # Generate non-streaming response
    response = response_manager.generate_chat_response(request_data, response_file)
    return JSONResponse(content=response.model_dump())


@app.post("/v1/embeddings")
async def embeddings(
    request: EmbeddingRequest,
    authorization: Optional[str] = Header(None),
    config: MockConfig = Depends(get_config_instance),
    response_manager: ResponseManager = Depends(get_response_manager)
):
    """Embeddings endpoint."""
    # Validate API key
    if not validate_api_key(authorization):
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Invalid API key", "type": "authentication_error"}}
        )

    # Convert request to dict for pattern matching
    request_data = request.model_dump()

    maybe_raise_scenario_failure("embeddings", request_data, config)

    # Simulate errors if configured
    if should_simulate_error(config):
        error_response = response_manager.generate_error_response(
            message="Simulated error for testing",
            error_type="server_error",
            code="simulated_error"
        )
        raise HTTPException(status_code=500, detail=error_response.model_dump())

    # Find matching response file
    response_file = None
    if "embeddings" in config.responses:
        response_file = config.responses["embeddings"].find_matching_response(request_data)

    # Generate response
    response = response_manager.generate_embedding_response(request_data, response_file)
    return JSONResponse(content=response.model_dump())


@app.post("/v1/completions")
async def completions(
    request: CompletionRequest,
    authorization: Optional[str] = Header(None),
    config: MockConfig = Depends(get_config_instance),
    response_manager: ResponseManager = Depends(get_response_manager)
):
    """Legacy completions endpoint."""
    # Validate API key
    if not validate_api_key(authorization):
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Invalid API key", "type": "authentication_error"}}
        )

    # Simulate errors if configured
    if should_simulate_error(config):
        error_response = response_manager.generate_error_response(
            message="Simulated error for testing",
            error_type="server_error",
            code="simulated_error"
        )
        raise HTTPException(status_code=500, detail=error_response.model_dump())

    # Convert request to dict for pattern matching
    request_data = request.model_dump()

    # Find matching response file
    response_file = None
    if "completions" in config.responses:
        response_file = config.responses["completions"].find_matching_response(request_data)

    # Generate response
    response = response_manager.generate_completion_response(request_data, response_file)
    return JSONResponse(content=response.model_dump())


@app.get("/v1/models")
async def list_models(
    authorization: Optional[str] = Header(None),
    config: MockConfig = Depends(get_config_instance)
):
    """List available models endpoint."""
    # Validate API key
    if not validate_api_key(authorization):
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Invalid API key", "type": "authentication_error"}}
        )

    # Return configured models or defaults
    models = config.models if config.models else [
        {
            "id": "gpt-4",
            "object": "model",
            "owned_by": "openai"
        },
        {
            "id": "gpt-4-turbo",
            "object": "model",
            "owned_by": "openai"
        },
        {
            "id": "gpt-3.5-turbo",
            "object": "model",
            "owned_by": "openai"
        },
        {
            "id": "text-embedding-ada-002",
            "object": "model",
            "owned_by": "openai"
        },
        {
            "id": "text-embedding-3-small",
            "object": "model",
            "owned_by": "openai"
        },
        {
            "id": "text-embedding-3-large",
            "object": "model",
            "owned_by": "openai"
        }
    ]

    model_objects = [ModelInfo(**model) for model in models]
    response = ModelsResponse(data=model_objects)
    return JSONResponse(content=response.model_dump())


def main():
    """Main entry point for the server."""
    parser = argparse.ArgumentParser(description="Mock OpenAI API Server")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (JSON or YAML)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind to (overrides config)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (overrides config)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config()

    # Override with command line arguments
    host = args.host or config.server.host
    port = args.port or config.server.port

    # Run the server
    uvicorn.run(
        "mock_openai.server:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
