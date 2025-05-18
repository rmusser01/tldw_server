# test_text_processing_endpoint.py
# Description: Unit tests for the chunking endpoint of the tldw_Server_API.
# This file contains tests for the FastAPI endpoints that handle text chunking
#
# Imports
import pytest
import json
from typing import List, Optional, Dict, Any, Callable, Generator, Union
from unittest.mock import patch, MagicMock
#
# Third-party Libraries
from fastapi import FastAPI, Body, File, UploadFile, Form, Depends, APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from fastapi.testclient import TestClient
#
# Local Imports
from tldw_Server_API.app.core.Utils.Chunk_Lib import (
    DEFAULT_CHUNK_OPTIONS as default_chunk_options_from_lib_real,  # Use a different alias to avoid clash
    improved_chunking_process as improved_chunking_process_real,
    ChunkingError, InvalidInputError, InvalidChunkingMethodError
)
#
#######################################################################################################################
#
# Functions:

# We will mock general_llm_analyzer and load_server_configs

# --- Mock versions of external dependencies for the endpoint ---
def mock_load_server_configs():
    """Returns mock server configurations, especially API keys."""
    return {
        "llm_api_settings": {"default_api_for_tasks": "mock_llm", "default_api": "mock_llm"},
        "mock_llm_api": {  # Matches default_api_for_tasks
            "api_key": "mock_api_key_for_tests",
            "model": "mock_model_for_tasks",
            "model_for_summarization": "mock_model_for_sum_tasks",  # Example specific model
            "temperature": 0.1,
            "max_tokens_for_summarization_step": 50  # Small for testing
        },
        "openai_api": {  # Example of another provider for completeness
            "api_key": "fake_openai_key", "model": "gpt-test"
        }
        # Add other configs your endpoint might read during its setup phase
    }


mock_general_llm_analyzer_call_history = []


def mock_general_llm_analyzer(payload: Dict[str, Any]) -> Union[str, Generator[str, None, None]]:
    """Mock for Summarization_General_Lib.analyze."""
    global mock_general_llm_analyzer_call_history
    mock_general_llm_analyzer_call_history.append(payload)  # Record the call

    api_name = payload.get("api_name")
    input_text = str(payload.get("input_data", ""))
    system_msg = payload.get("system_message", "")

    # Simple mock response
    if payload.get("streaming", False):
        def stream_gen():
            yield f"[Mock Stream for {api_name}]: Summarizing part of '{input_text[:20]}...' with system: '{system_msg[:20]}...'"

        return stream_gen()
    else:
        return f"[Mock Summary from {api_name}]: Input '{input_text[:30]}...' processed with system: '{system_msg[:30]}...'"


# --- Pydantic models (copy from your endpoint file) ---
# (LLMOptionsForChunkerInternalSteps, ChunkingOptionsRequest, ChunkingTextRequest, etc.)
# For brevity, I'll assume these are defined as in your endpoint code provided before.
# Ensure they are identical or import them if your test structure allows.
class LLMOptionsForChunkerInternalSteps(BaseModel):
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    system_prompt_for_step: Optional[str] = Field(None)
    max_tokens_per_step: Optional[int] = Field(None, gt=0)


class ChunkingOptionsRequest(BaseModel):
    method: Optional[str] = Field(default_chunk_options_from_lib_real.get('method'))
    max_size: Optional[int] = Field(default_chunk_options_from_lib_real.get('max_size'), gt=0)
    overlap: Optional[int] = Field(default_chunk_options_from_lib_real.get('overlap'), ge=0)
    language: Optional[str] = Field(None)
    tokenizer_name_or_path: Optional[str] = Field(
        default_chunk_options_from_lib_real.get('tokenizer_name_or_path', "gpt2"))
    adaptive: Optional[bool] = Field(default_chunk_options_from_lib_real.get('adaptive'))
    multi_level: Optional[bool] = Field(default_chunk_options_from_lib_real.get('multi_level'))
    custom_chapter_pattern: Optional[str] = Field(None)
    semantic_similarity_threshold: Optional[float] = Field(
        default_chunk_options_from_lib_real.get('semantic_similarity_threshold'), ge=0.0, le=1.0)
    semantic_overlap_sentences: Optional[int] = Field(
        default_chunk_options_from_lib_real.get('semantic_overlap_sentences'), ge=0)
    json_chunkable_data_key: Optional[str] = Field(
        default_chunk_options_from_lib_real.get('json_chunkable_data_key', 'data'))
    summarization_detail: Optional[float] = Field(default_chunk_options_from_lib_real.get('summarization_detail'),
                                                  ge=0.0, le=1.0)
    llm_options_for_internal_steps: Optional[LLMOptionsForChunkerInternalSteps] = Field(None)


class ChunkingTextRequest(BaseModel):
    text_content: str
    file_name: Optional[str] = "input_text.txt"
    options: Optional[ChunkingOptionsRequest] = None


class ChunkedContentResponse(BaseModel):
    text: str
    metadata: Dict[str, Any]


class ChunkingResponse(BaseModel):
    chunks: List[ChunkedContentResponse]
    original_file_name: Optional[str]
    applied_options: ChunkingOptionsRequest


# --- Define a minimal app with the router ---
# In a real setup, you import `app` from `main.py`
# For self-contained test:
test_app_router = APIRouter()


# Copy the endpoint implementation here or import it
# This is the code from your `text_processing.py` that we are testing.
# For this example, I'll assume the endpoint functions `process_text_for_chunking_json`
# and `process_file_for_chunking` are defined here or imported.
# To make this truly runnable, you'd copy those functions here,
# and ensure they use `default_chunk_options_from_lib_real` and `improved_chunking_process_real`

# For this example, let's define stubs for the endpoint functions
# and assume the real ones would be used when you integrate this.
# This is NOT how you'd usually do it, you'd import your actual router.
@test_app_router.post("/chunk_text", response_model=ChunkingResponse)
async def process_text_for_chunking_json_stub(request_data: ChunkingTextRequest = Body(...)):
    # This is where your actual endpoint logic would go.
    # For testing, we need to ensure it calls improved_chunking_process correctly.
    # We will patch 'improved_chunking_process_real' when testing specific interactions.
    if request_data.text_content == "error_trigger":
        raise HTTPException(status_code=500, detail="Simulated server error")

    # Simulate a successful call for basic tests
    mock_chunk_data = {"text": "mocked chunk",
                       "metadata": {"method": request_data.options.method if request_data.options else "words"}}
    return ChunkingResponse(
        chunks=[ChunkedContentResponse(**mock_chunk_data)],
        original_file_name=request_data.file_name,
        applied_options=request_data.options if request_data.options else ChunkingOptionsRequest()
    )


@test_app_router.post("/chunk_file", response_model=ChunkingResponse)
async def process_file_for_chunking_stub(file: UploadFile = File(...), method: Optional[str] = Form("words")):
    content = await file.read()
    await file.close()
    if content == b"error_trigger_file":
        raise HTTPException(status_code=500, detail="Simulated file error")

    mock_chunk_data = {"text": "mocked file chunk " + content.decode()[:10], "metadata": {"method": method}}
    return ChunkingResponse(
        chunks=[ChunkedContentResponse(**mock_chunk_data)],
        original_file_name=file.filename,
        applied_options=ChunkingOptionsRequest(method=method)  # Simplified
    )


app_for_testing = FastAPI()
app_for_testing.include_router(test_app_router, prefix="/api/v1/process")

# --- Test Client Fixture ---
@pytest.fixture
def client():
    # Reset call history for each test
    global mock_general_llm_analyzer_call_history
    mock_general_llm_analyzer_call_history = []
    with TestClient(app_for_testing) as c:  # Use the app with overrides
        yield c


# --- Endpoint Test Cases ---

def test_chunk_text_endpoint_simple(client: TestClient):
    payload = {
        "text_content": "This is a simple test.",
        "options": {"method": "words", "max_size": 2}
    }
    # Patch the actual improved_chunking_process that the *real* endpoint would call
    with patch('tldw_Server_API.app.v1.endpoints.text_processing.improved_chunking_process_real',
               # Path to where real endpoint imports it
               return_value=[{'text': 'This is', 'metadata': {'method': 'words'}},
                             {'text': 'a simple', 'metadata': {'method': 'words'}}]) as mock_icp:
        response = client.post("/api/v1/process/chunk_text", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["original_file_name"] == "input_text.txt"
    assert len(data["chunks"]) > 0  # Or == 2 if mock_icp is precise
    assert data["chunks"][0]["text"] == "This is"
    assert data["applied_options"]["method"] == "words"
    mock_icp.assert_called_once()
    # inspect call args to mock_icp to verify options were passed correctly from endpoint


def test_chunk_text_endpoint_rolling_summarize(client: TestClient):
    global mock_general_llm_analyzer_call_history
    payload = {
        "text_content": "This is a test for rolling summarization feature. It needs an LLM.",
        "options": {
            "method": "rolling_summarize",
            "tokenizer_name_or_path": "gpt2",  # Important for rolling_summarize
            "summarization_detail": 0.1,
            "llm_options_for_internal_steps": {  # Client suggestions
                "temperature": 0.5,
                "system_prompt_for_step": "Client suggested system prompt for step.",
                "max_tokens_per_step": 20
            }
        }
    }

    # Mock the core chunking library's improved_chunking_process
    # This allows us to check what the endpoint prepares *for* improved_chunking_process
    # The unit tests for Chunk_Lib already test improved_chunking_process internals.
    mock_chunk_result = [
        {"text": "[Mock Rolling Summary of the input...]", "metadata": {"method": "rolling_summarize"}}]
    with patch('tldw_Server_API.app.v1.endpoints.text_processing.improved_chunking_process_real',
               return_value=mock_chunk_result) as mock_icp:
        response = client.post("/api/v1/process/chunk_text", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["chunks"][0]["text"] == "[Mock Rolling Summary of the input...]"

    # Verify what improved_chunking_process was called with by the endpoint
    mock_icp.assert_called_once()
    args_to_icp, _ = mock_icp.call_args
    passed_text_content, passed_effective_options, passed_tokenizer, passed_llm_func, passed_llm_config = args_to_icp

    assert passed_text_content == payload["text_content"]
    assert passed_effective_options["method"] == "rolling_summarize"
    assert passed_tokenizer == "gpt2"
    assert passed_llm_func is mock_general_llm_analyzer  # Check it's our (overridden) mock

    # Check that llm_api_config_to_use was built correctly by the endpoint (Option A)
    assert passed_llm_config["api_name"] == "mock_llm"  # From mock_load_server_configs
    assert passed_llm_config["model"] == "mock_model_for_sum_tasks"  # Server-determined
    assert passed_llm_config["api_key"] == "mock_api_key_for_tests"  # Server-determined
    assert passed_llm_config["temp"] == 0.5  # Client suggested
    assert passed_llm_config["system_message"] == "Client suggested system prompt for step."
    assert passed_llm_config["max_tokens"] == 20

    # If you wanted to test that mock_general_llm_analyzer was called by the *real* improved_chunking_process,
    # you would NOT patch improved_chunking_process. Instead, you'd let it run and then check
    # mock_general_llm_analyzer_call_history. This makes it a deeper integration test.
    # For this example, we isolated the endpoint's logic of preparing args FOR improved_chunking_process.


def test_chunk_text_endpoint_invalid_method(client: TestClient):
    # To test this properly, improved_chunking_process should raise InvalidChunkingMethodError
    # which the endpoint then converts to a 400.
    # So, we mock improved_chunking_process to raise that error.
    with patch('tldw_Server_API.app.v1.endpoints.text_processing.improved_chunking_process_real',
               side_effect=InvalidChunkingMethodError("Test invalid method")) as mock_icp:
        payload = {"text_content": "test", "options": {"method": "invalid_method"}}
        response = client.post("/api/v1/process/chunk_text", json=payload)

    assert response.status_code == 400
    assert "Test invalid method" in response.json()["detail"]


def test_chunk_file_endpoint_simple(client: TestClient):
    # Create a dummy file for upload
    dummy_file_content = b"Content from uploaded file."
    files = {'file': ('testfile.txt', dummy_file_content, 'text/plain')}
    form_data = {'method': 'sentences', 'max_size': '1'}  # max_size as string, Pydantic should convert

    # Patch the actual improved_chunking_process for the file endpoint
    mock_file_chunk_result = [{'text': 'Content from', 'metadata': {'method': 'sentences'}}]
    with patch('tldw_Server_API.app.v1.endpoints.text_processing.improved_chunking_process_real',
               return_value=mock_file_chunk_result) as mock_icp_file:
        response = client.post("/api/v1/process/chunk_file", files=files, data=form_data)

    assert response.status_code == 200
    data = response.json()
    assert data["original_file_name"] == "testfile.txt"
    assert data["chunks"][0]["text"] == "Content from"
    assert data["applied_options"]["method"] == "sentences"
    mock_icp_file.assert_called_once()
    # Inspect args to mock_icp_file, especially the options dict built from form data

# TODO: Add more tests for:
# - Different chunking methods via endpoint
# - Validation errors (e.g., non-integer max_size if Pydantic doesn't catch it first)
# - Empty text_content
# - File upload with LLM-dependent method like rolling_summarize (similar setup to the JSON one)

#
# End of test_chunking_endpoint.py
#######################################################################################################################
