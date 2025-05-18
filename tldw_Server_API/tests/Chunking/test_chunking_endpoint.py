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
from tldw_Server_API.app.api.v1.endpoints.chunking import chunking_router as actual_chunking_endpoint_router
from tldw_Server_API.app.core.Utils.Chunk_Lib import (
    DEFAULT_CHUNK_OPTIONS as default_chunk_options_from_lib_real,  # Use a different alias to avoid clash
    improved_chunking_process as improved_chunking_process_real,
    ChunkingError, InvalidInputError, InvalidChunkingMethodError
)
#
#######################################################################################################################
#
# Functions:

# --- Mock versions of external dependencies for the endpoint ---
def mock_load_server_configs_for_test():
    return {
        "llm_api_settings": {"default_api_for_tasks": "mock_llm", "default_api": "mock_llm"},
        "mock_llm_api": {
            "api_key": "mock_api_key_for_tests",
            "model": "mock_model_general", # General model for the provider
            "model_for_summarization": "mock_model_for_sum_tasks_server_default", # Specific for this task
            "temperature": 0.15, # Server's default temperature for this provider/task
            "max_tokens_for_summarization_step": 60, # Server's default max tokens for step
            "cap_max_tokens_summarization_step": 100 # Server's hard cap
        },
        "openai_api": {"api_key": "fake_openai_key", "model": "gpt-test-override"},
        "chunking_llm_defaults": { # Example new section for task-specific LLM
            "provider": "mock_llm_task_provider",
            "model": "mock_model_task_specific"
        }
    }

mock_general_llm_analyzer_call_history_for_test = []
def mock_general_llm_analyzer_for_test(payload: Dict[str, Any]) -> Union[str, Generator[str,None,None]]:
    global mock_general_llm_analyzer_call_history_for_test
    mock_general_llm_analyzer_call_history_for_test.append(payload.copy())

    api_name = payload.get("api_name")
    input_text = str(payload.get("input_data", ""))
    system_msg = payload.get("system_message", "")
    streaming = payload.get("streaming", False)
    temp = payload.get("temp")
    max_tok = payload.get("max_tokens")

    response_text = (
        f"[Mock Summary from {api_name}, model {payload.get('model')}]: "
        f"Input '{input_text[:30]}...' processed. "
        f"SysPrompt: '{system_msg[:30]}...'. Temp: {temp}. MaxTokens: {max_tok}"
    )
    if streaming:
        def stream_gen():
            yield response_text
        return stream_gen()
    return response_text


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


# --- FastAPI App with the actual router for testing ---
app_for_testing_with_real_router = FastAPI()
app_for_testing_with_real_router.include_router(actual_chunking_endpoint_router, prefix="/api/v1/chunking")

# Apply dependency overrides to this app instance
app_for_testing_with_real_router.dependency_overrides[server_config_loader_actual] = mock_load_server_configs_for_test
app_for_testing_with_real_router.dependency_overrides[general_llm_analyzer_actual] = mock_general_llm_analyzer_for_test


# --- Test Client Fixture using the app with the real router ---
@pytest.fixture
def client():  # Renamed from client_real_router for consistency
    global mock_general_llm_analyzer_call_history_for_test
    mock_general_llm_analyzer_call_history_for_test = []  # Reset for each test
    with TestClient(app_for_testing_with_real_router) as c:
        yield c


# --- Path for Patching `improved_chunking_process` ---
# This path MUST point to where `improved_chunking_process` is imported and used
# by your ACTUAL endpoint router module (e.g., chunking_router.py).
# Example: if chunking_router.py has `from tldw_Server_API.app.core.Chunk_Lib import improved_chunking_process`
PATH_TO_ICP_IN_ENDPOINT_ROUTER_MODULE = 'tldw_Server_API.app.api.v1.endpoints.chunking_router.improved_chunking_process'


# --- Endpoint Test Cases ---

@patch(PATH_TO_ICP_IN_ENDPOINT_ROUTER_MODULE)
def test_chunk_text_simple_words(mock_icp, client: TestClient):
    mock_icp.return_value = [{'text': 'This is', 'metadata': {'method': 'words'}},
                             {'text': 'a test.', 'metadata': {'method': 'words'}}]
    payload = {"text_content": "This is a test.", "options": {"method": "words", "max_size": 2}}
    response = client.post("/api/v1/chunking/chunk_text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["chunks"]) == 2
    assert data["chunks"][0]["text"] == "This is"
    assert data["applied_options"]["method"] == "words"
    mock_icp.assert_called_once()
    # Verify options passed to the mocked improved_chunking_process
    _, passed_options, _, _, _ = mock_icp.call_args[0]
    assert passed_options["method"] == "words"
    assert passed_options["max_size"] == 2


@patch(PATH_TO_ICP_IN_ENDPOINT_ROUTER_MODULE)
def test_chunk_text_method_sentences(mock_icp, client: TestClient):
    mock_icp.return_value = [{'text': 'Sentence one.', 'metadata': {'method': 'sentences'}}]
    payload = {"text_content": "Sentence one. Sentence two.", "options": {"method": "sentences", "max_size": 1}}
    response = client.post("/api/v1/chunking/chunk_text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["applied_options"]["method"] == "sentences"
    assert data["chunks"][0]["text"] == "Sentence one."
    _, passed_options, _, _, _ = mock_icp.call_args[0]
    assert passed_options["method"] == "sentences"


@patch(PATH_TO_ICP_IN_ENDPOINT_ROUTER_MODULE)
def test_chunk_text_method_paragraphs(mock_icp, client: TestClient):
    mock_icp.return_value = [{'text': 'Paragraph one.\n\nParagraph two.', 'metadata': {'method': 'paragraphs'}}]
    payload = {"text_content": "Paragraph one.\n\nParagraph two.", "options": {"method": "paragraphs", "max_size": 1}}
    response = client.post("/api/v1/chunking/chunk_text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["applied_options"]["method"] == "paragraphs"
    _, passed_options, _, _, _ = mock_icp.call_args[0]
    assert passed_options["method"] == "paragraphs"


# Add similar tests for "semantic", "json", "xml", "ebook_chapters" by mocking icp's return

def test_chunk_text_validation_error_max_size_string(client: TestClient):
    payload = {"text_content": "test", "options": {"method": "words", "max_size": "not-an-int"}}
    response = client.post("/api/v1/chunking/chunk_text", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert any("max_size" in error["loc"] and "integer" in error.get("msg", "").lower() for error in data["detail"])


def test_chunk_text_validation_overlap_too_large(client: TestClient):
    payload = {"text_content": "test", "options": {"method": "words", "max_size": 5, "overlap": 10}}
    response = client.post("/api/v1/chunking/chunk_text", json=payload)
    assert response.status_code == 422
    assert "Overlap (10) must be less than max_size (5)" in response.json()["detail"][0]["msg"]


@patch(PATH_TO_ICP_IN_ENDPOINT_ROUTER_MODULE)
def test_chunk_text_empty_content(mock_icp, client: TestClient):
    mock_icp.return_value = []
    payload = {"text_content": "", "options": {"method": "words"}}
    response = client.post("/api/v1/chunking/chunk_text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["chunks"] == []
    mock_icp.assert_called_once()
    call_args_list = mock_icp.call_args[0]  # Get the positional arguments
    assert call_args_list[0] == ""  # text_content
    assert call_args_list[1]["method"] == "words"  # effective_options


@patch(PATH_TO_ICP_IN_ENDPOINT_ROUTER_MODULE)
def test_chunk_text_rolling_summarize_server_determines_llm(mock_icp, client: TestClient):
    global mock_general_llm_analyzer_call_history_for_test
    mock_icp.return_value = [{"text": "[Mock Rolling Summary From Test]", "metadata": {"method": "rolling_summarize"}}]
    payload = {
        "text_content": "Content for server-determined LLM rolling summarize.",
        "options": {
            "method": "rolling_summarize",
            "tokenizer_name_or_path": "distilgpt2",  # Different tokenizer
            "summarization_detail": 0.2,
            "llm_options_for_internal_steps": {  # Client suggestions for non-provider/model aspects
                "temperature": 0.6,
                "system_prompt_for_step": "Client custom step prompt.",
                "max_tokens_per_step": 25
            }
        }
    }
    response = client.post("/api/v1/chunking/chunk_text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["chunks"][0]["text"] == "[Mock Rolling Summary From Test]"

    mock_icp.assert_called_once()
    args_to_icp, _ = mock_icp.call_args
    passed_text, passed_opts, passed_tokenizer, passed_llm_func, passed_llm_config = args_to_icp

    assert passed_text == payload["text_content"]
    assert passed_opts["method"] == "rolling_summarize"
    assert passed_opts["summarization_detail"] == 0.2
    assert passed_tokenizer == "distilgpt2"
    assert passed_llm_func is general_llm_analyzer_actual  # Checking it's the (overridden) actual analyzer

    # Verify server-determined LLM (Option A) and client suggestions for other params
    assert passed_llm_config["api_name"] == "mock_llm"  # From mock_load_server_configs
    assert passed_llm_config["model"] == "mock_model_for_sum_tasks_server_default"  # Server-determined
    assert passed_llm_config["api_key"] == "mock_api_key_for_tests"  # Server-determined
    assert passed_llm_config["temp"] == 0.6  # Client-suggested
    assert passed_llm_config["system_message"] == "Client custom step prompt."  # Client-suggested
    assert passed_llm_config["max_tokens"] == 25  # Client-suggested (and within server cap if implemented)


@patch(PATH_TO_ICP_IN_ENDPOINT_ROUTER_MODULE)
def test_chunk_file_rolling_summarize(mock_icp_file, client: TestClient):
    global mock_general_llm_analyzer_call_history_for_test
    mock_icp_file.return_value = [{"text": "[Mock File Rolling Summary]", "metadata": {"method": "rolling_summarize"}}]

    dummy_content = b"File content for rolling summarize test via file upload."
    files = {'file': ('test_roll_file.txt', dummy_content, 'text/plain')}
    form_data = {
        'method': 'rolling_summarize',
        'tokenizer_name_or_path': 'gpt2-medium',
        'summarization_detail': '0.3',
        'llm_step_temperature': '0.75',  # Flattened llm_options
        'llm_step_system_prompt': 'File step prompt.',
        'llm_step_max_tokens': '35'
    }
    response = client.post("/api/v1/chunking/chunk_file", files=files, data=form_data)

    assert response.status_code == 200
    data = response.json()
    assert data["chunks"][0]["text"] == "[Mock File Rolling Summary]"
    assert data["original_file_name"] == "test_roll_file.txt"

    mock_icp_file.assert_called_once()
    args_to_icp, _ = mock_icp_file.call_args
    passed_text_file, passed_opts_file, passed_tokenizer_file, passed_llm_func_file, passed_llm_config_file = args_to_icp

    assert passed_text_file == dummy_content.decode()
    assert passed_opts_file["method"] == "rolling_summarize"
    assert passed_opts_file["tokenizer_name_or_path"] == "gpt2-medium"
    assert passed_opts_file["llm_options_for_internal_steps"]["temperature"] == 0.75  # Check nested construction
    assert passed_llm_func_file is general_llm_analyzer_actual

    assert passed_llm_config_file["api_name"] == "mock_llm"
    assert passed_llm_config_file["model"] == "mock_model_for_sum_tasks_server_default"
    assert passed_llm_config_file["temp"] == 0.75
    assert passed_llm_config_file["system_message"] == "File step prompt."
    assert passed_llm_config_file["max_tokens"] == 35


# @patch(PATH_TO_ICP_IN_ENDPOINT_ROUTER_MODULE)
# def test_chunk_text_endpoint_no_options(mock_icp, client: TestClient):
#     # Test when no 'options' are provided in the payload, relying on defaults
#     mock_icp.return_value = [
#         {'text': 'Default chunked', 'metadata': {'method': default_chunk_options_from_lib_actual.get('method')}}]
#     payload = {"text_content": "Text with no options."}  # No "options" key
#
#     response = client.post("/api/v1/chunking/chunk_text", json=payload)
#     assert response.status_code == 200
#     data = response.json()
#     assert data["applied_options"]["method"] == default_chunk_options_from_lib_actual.get(
#         'method')  # Should use library default
#
#     mock_icp.assert_called_once()
#     _, passed_options, _, _, _ = mock_icp.call_args[0]
#     # Check that passed_options reflects the library defaults merged with request (which is empty for options)
#     assert passed_options["method"] == default_chunk_options_from_lib_actual.get('method')
#     assert passed_options["max_size"] == default_chunk_options_from_lib_actual.get('max_size')


def test_chunk_text_endpoint_invalid_payload_missing_text(client: TestClient):
    payload = {"options": {"method": "words"}}  # Missing "text_content"
    response = client.post("/api/v1/chunking/chunk_text", json=payload)
    assert response.status_code == 422  # Pydantic validation for missing required field
    assert "text_content" in response.json()["detail"][0]["loc"]

#
# End of test_chunking_endpoint.py
#######################################################################################################################
