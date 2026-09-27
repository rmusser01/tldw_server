# test_text_processing_endpoint.py
# Description: Unit tests for the chunking endpoint of the tldw_Server_API.
# This file contains tests for the FastAPI endpoints that handle text chunking
#
# Imports
from collections.abc import Generator
from typing import Any, Optional, Union
from unittest.mock import patch

#
# Third-party Libraries
import pytest
from fastapi import (
    FastAPI,
)  # Removed: Body, File, UploadFile, Form, Depends, APIRouter, HTTPException, status (FastAPI itself is enough for app instance)
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

#
# Local Imports
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import try_get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints import chunking as chunking_module
from tldw_Server_API.app.api.v1.endpoints.chunking import (
    chunking_router as actual_chunking_endpoint_router,
)
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.Chunking import (
    DEFAULT_CHUNK_OPTIONS as DEFAULT_CHUNK_OPTIONS_REAL,
)
from tldw_Server_API.app.core.Chunking.strategies.rolling_summarize import (
    LLM_USAGE_SUCCEEDED_KEY,
    LLM_USAGE_TRACKER_KEY,
    RollingSummarizeStrategy,
)

#
#######################################################################################################################
#
# Functions:


# --- Mock versions of external dependencies for the endpoint ---
def mock_load_server_configs_for_test():
    # This mock function will be used with @patch
    return {
        "llm_api_settings": {"default_api_for_tasks": "mock_llm", "default_api": "mock_llm"},
        "mock_llm_api": {
            "api_key": "mock_api_key_for_tests",
            "model": "mock_model_general",
            "model_for_summarization": "mock_model_for_sum_tasks_server_default",
            "temperature": 0.15,
            "max_tokens_for_summarization_step": 60,
            "cap_max_tokens_summarization_step": 100,
        },
        "openai_api": {"api_key": "fake_openai_key", "model": "gpt-test-override"},
        "chunking_llm_defaults": {"provider": "mock_llm_task_provider", "model": "mock_model_task_specific"},
    }


mock_general_llm_analyzer_call_history_for_test = []


def mock_general_llm_analyzer_for_test(payload: dict[str, Any]) -> Union[str, Generator[str, None, None]]:
    # This mock function will be used with @patch
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


# --- Pydantic models (copied from your schema file for test validation) ---
# It's better to import these from your schema module if possible, to avoid duplication and divergence.
# For this fix, we'll keep them as copied.
class LLMOptionsForChunkerInternalSteps(BaseModel):
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    system_prompt_for_step: Optional[str] = Field(None)
    max_tokens_per_step: Optional[int] = Field(None, gt=0)


class ChunkingOptionsRequest(BaseModel):
    method: Optional[str] = Field(DEFAULT_CHUNK_OPTIONS_REAL.get("method"))
    max_size: Optional[int] = Field(DEFAULT_CHUNK_OPTIONS_REAL.get("max_size"), gt=0)
    overlap: Optional[int] = Field(DEFAULT_CHUNK_OPTIONS_REAL.get("overlap"), ge=0)
    language: Optional[str] = Field(None)  # Default to None for auto-detection
    tokenizer_name_or_path: Optional[str] = Field(
        DEFAULT_CHUNK_OPTIONS_REAL.get("tokenizer_name_or_path", "gpt2")
    )
    adaptive: Optional[bool] = Field(DEFAULT_CHUNK_OPTIONS_REAL.get("adaptive"))
    multi_level: Optional[bool] = Field(DEFAULT_CHUNK_OPTIONS_REAL.get("multi_level"))
    custom_chapter_pattern: Optional[str] = Field(DEFAULT_CHUNK_OPTIONS_REAL.get("custom_chapter_pattern"))
    semantic_similarity_threshold: Optional[float] = Field(
        DEFAULT_CHUNK_OPTIONS_REAL.get("semantic_similarity_threshold"), ge=0.0, le=1.0
    )
    semantic_overlap_sentences: Optional[int] = Field(
        DEFAULT_CHUNK_OPTIONS_REAL.get("semantic_overlap_sentences"), ge=0
    )
    json_chunkable_data_key: Optional[str] = Field(
        DEFAULT_CHUNK_OPTIONS_REAL.get("json_chunkable_data_key", "data")
    )
    summarization_detail: Optional[float] = Field(
        DEFAULT_CHUNK_OPTIONS_REAL.get("summarization_detail"), ge=0.0, le=1.0
    )
    llm_options_for_internal_steps: Optional[LLMOptionsForChunkerInternalSteps] = Field(None)


class ChunkingTextRequest(BaseModel):
    text_content: str
    file_name: Optional[str] = "input_text.txt"
    options: Optional[ChunkingOptionsRequest] = None


class ChunkedContentResponse(BaseModel):
    text: str
    metadata: dict[str, Any]


class ChunkingResponse(BaseModel):
    chunks: list[ChunkedContentResponse]
    original_file_name: Optional[str]
    applied_options: ChunkingOptionsRequest


# --- FastAPI App with the actual router for testing ---
app_for_testing_with_real_router = FastAPI()
app_for_testing_with_real_router.include_router(actual_chunking_endpoint_router, prefix="/api/v1/chunking")


# REMOVED: app.dependency_overrides for load_server_configs and general_llm_analyzer
# These will be mocked using @patch in individual tests where needed,
# as they are not FastAPI 'Depends' style dependencies in the endpoint.


# --- Test Client Fixture using the app with the real router ---
@pytest.fixture
def client():
    global mock_general_llm_analyzer_call_history_for_test
    mock_general_llm_analyzer_call_history_for_test = []  # Reset for each test
    settings = get_settings()
    default_headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
    with TestClient(app_for_testing_with_real_router, headers=default_headers) as c:
        yield c


@pytest.fixture(autouse=True)
def isolate_provider_override_store(monkeypatch):
    """Keep endpoint unit tests independent from the app-lifespan AuthNZ store."""

    class EmptyOverrideSnapshot:
        def enforce(self, _model):
            return None

        def ensure_healthy(self):
            return None

        def server_fallback(self, base_fallback=None):
            return base_fallback

    snapshot = EmptyOverrideSnapshot()
    monkeypatch.setattr(
        chunking_module,
        "capture_provider_override_call_snapshot",
        lambda _provider: snapshot,
    )


# --- Path for Patching `improved_chunking_process` and other direct calls ---
# This path MUST point to where the function is imported and used
# by your ACTUAL endpoint module (e.g., chunking.py).
PATH_TO_ICP_IN_ENDPOINT_MODULE = "tldw_Server_API.app.api.v1.endpoints.chunking.improved_chunking_process"
PATH_TO_LOAD_CONFIGS_IN_ENDPOINT_MODULE = "tldw_Server_API.app.api.v1.endpoints.chunking.load_server_configs"
PATH_TO_LLM_ANALYZER_IN_ENDPOINT_MODULE = "tldw_Server_API.app.api.v1.endpoints.chunking.general_llm_analyzer"
def _install_chunking_snapshot_rotation(monkeypatch, initial_key):
    """Install an A→B (or absent→B) static credential source for one request."""
    initial_config = {"ollama_api": {"base_url": "http://snapshot-a.invalid"}}
    state = {
        "key": initial_key,
        "app_config": initial_config,
    }
    static_calls = []
    lifecycle = []

    class FakeRuntime:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            assert "fallback_resolver" not in kwargs
            assert kwargs["server_config_snapshot"] == _snapshot_chunking_config()
            self.api_key = state["key"]
            self.app_config = {
                section: dict(config)
                for section, config in state["app_config"].items()
            }
            lifecycle.append(("init", kwargs))

        async def resolve(self, provider, *, model=None):
            lifecycle.append(("resolve", provider, model))
            static_calls.append(provider)
            return type(
                "Handle",
                (),
                {
                    "api_key": self.api_key,
                    "app_config": self.app_config,
                    "credentials_resolved": True,
                },
            )()

        async def mark_used(self, _handle):
            lifecycle.append("mark_used")

        async def close(self):
            lifecycle.append("close")

    async def forbidden_low_level_resolver(*_args, **_kwargs):
        raise AssertionError("chunking bypassed ProviderCredentialRuntime")

    monkeypatch.setattr(chunking_module, "ProviderCredentialRuntime", FakeRuntime)
    monkeypatch.setattr(
        chunking_module,
        "resolve_byok_credentials",
        forbidden_low_level_resolver,
        raising=False,
    )
    return state, initial_config, static_calls, lifecycle


def _snapshot_chunking_config():
    return {
        "llm_api_settings": {
            "default_api": "ollama",
            "default_api_for_tasks": "ollama",
        },
        "ollama_api": {
            "api_key": "legacy-bare-key-must-not-win",
            "model": "snapshot-model",
            "model_for_summarization": "snapshot-summary-model",
            "max_tokens_for_summarization_step": 60,
        },
    }


def _invoke_rolling_analyzer(llm_func, llm_config, boundary_calls):
    def boundary_spy(*args, **kwargs):
        boundary_calls.append((args, kwargs))
        return "snapshot summary"

    strategy = RollingSummarizeStrategy(
        llm_call_func=boundary_spy if llm_func is not None else llm_func,
        llm_config=llm_config,
    )
    assert strategy._call_llm("deferred rolling prompt") == "snapshot summary"


def _assert_chunking_boundary_snapshot(
    boundary_calls,
    *,
    expected_key,
    expected_config,
    static_calls,
    lifecycle,
):
    assert len(boundary_calls) == 1
    args, kwargs = boundary_calls[0]
    assert args[3] == expected_key
    assert kwargs["app_config"] == expected_config
    assert kwargs["credentials_resolved"] is True
    assert kwargs["model_override"] == "snapshot-summary-model"
    assert static_calls == ["ollama"]
    assert lifecycle[1:] == [
        ("resolve", "ollama", "snapshot-summary-model"),
        "mark_used",
        "close",
    ]


# --- Endpoint Test Cases ---


@patch(PATH_TO_ICP_IN_ENDPOINT_MODULE)
def test_chunk_text_simple_words(mock_icp, client: TestClient):
    mock_icp.return_value = [
        {"text": "This is", "metadata": {"method": "words", "chunk_index": 1, "total_chunks": 2}},
        {"text": "a test.", "metadata": {"method": "words", "chunk_index": 2, "total_chunks": 2}},
    ]
    # CORRECTED PAYLOAD
    payload = {"text_content": "This is a test.", "options": {"method": "words", "max_size": 2, "overlap": 0}}
    response = client.post("/api/v1/chunking/chunk_text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["chunks"]) == 2
    assert data["chunks"][0]["text"] == "This is"
    assert data["applied_options"]["method"] == "words"
    assert data["applied_options"]["max_size"] == 2
    assert data["applied_options"]["overlap"] == 0  # Check applied options
    mock_icp.assert_called_once()
    passed_text, passed_options, passed_tokenizer, passed_llm_func, passed_llm_config = mock_icp.call_args[0]
    assert passed_text == "This is a test."
    assert passed_options["method"] == "words"
    assert passed_options["max_size"] == 2
    assert passed_options["overlap"] == 0


@patch(PATH_TO_ICP_IN_ENDPOINT_MODULE)
def test_chunk_text_method_sentences(mock_icp, client: TestClient):
    mock_icp.return_value = [
        {"text": "Sentence one.", "metadata": {"method": "sentences", "chunk_index": 1, "total_chunks": 1}}
    ]
    # CORRECTED PAYLOAD
    payload = {
        "text_content": "Sentence one. Sentence two.",
        "options": {"method": "sentences", "max_size": 1, "overlap": 0},
    }
    response = client.post("/api/v1/chunking/chunk_text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["applied_options"]["method"] == "sentences"
    assert data["applied_options"]["max_size"] == 1
    assert data["applied_options"]["overlap"] == 0
    assert data["chunks"][0]["text"] == "Sentence one."
    _, passed_options, _, _, _ = mock_icp.call_args[0]
    assert passed_options["method"] == "sentences"
    assert passed_options["max_size"] == 1
    assert passed_options["overlap"] == 0


@patch(PATH_TO_ICP_IN_ENDPOINT_MODULE)
def test_chunk_text_method_paragraphs(mock_icp, client: TestClient):
    mock_icp.return_value = [
        {
            "text": "Paragraph one.\n\nParagraph two.",
            "metadata": {"method": "paragraphs", "chunk_index": 1, "total_chunks": 1},
        }
    ]
    # CORRECTED PAYLOAD
    payload = {
        "text_content": "Paragraph one.\n\nParagraph two.",
        "options": {"method": "paragraphs", "max_size": 1, "overlap": 0},
    }
    response = client.post("/api/v1/chunking/chunk_text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["applied_options"]["method"] == "paragraphs"
    assert data["applied_options"]["max_size"] == 1
    assert data["applied_options"]["overlap"] == 0
    _, passed_options, _, _, _ = mock_icp.call_args[0]
    assert passed_options["method"] == "paragraphs"
    assert passed_options["max_size"] == 1
    assert passed_options["overlap"] == 0


def test_chunk_text_validation_error_max_size_string(client: TestClient):
    payload = {"text_content": "test", "options": {"method": "words", "max_size": "not-an-int"}}
    response = client.post("/api/v1/chunking/chunk_text", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert any("max_size" in error["loc"] and "integer" in error.get("msg", "").lower() for error in data["detail"])


def test_chunk_text_validation_overlap_too_large(client: TestClient):
    payload = {"text_content": "test", "options": {"method": "words", "max_size": 5, "overlap": 10}}
    response = client.post("/api/v1/chunking/chunk_text", json=payload)
    assert response.status_code == 422  # This should be a Pydantic validation error caught by the model_validator
    # The schema chunking_schema.py has a model_validator:
    # @model_validator(mode='after')
    # def check_overlap_less_than_max_size(cls, values: 'ChunkingOptionsRequest') -> 'ChunkingOptionsRequest':
    # If this validator raises ValueError, FastAPI converts it to a 422.
    assert "Overlap (10) must be less than max_size (5)" in response.json()["detail"][0]["msg"]


@patch(PATH_TO_ICP_IN_ENDPOINT_MODULE)
def test_chunk_text_empty_content(mock_icp, client: TestClient):
    mock_icp.return_value = []  # Simulate no chunks produced
    payload = {"text_content": "", "options": {"method": "words"}}
    response = client.post("/api/v1/chunking/chunk_text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["chunks"] == []
    mock_icp.assert_called_once()
    call_args_list = mock_icp.call_args[0]
    assert call_args_list[0] == ""  # text_content
    assert call_args_list[1]["method"] == "words"  # effective_options


@patch(PATH_TO_ICP_IN_ENDPOINT_MODULE)
def test_chunk_text_sanitizes_unexpected_error(mock_icp, client: TestClient):
    mock_icp.side_effect = RuntimeError("chunking backend exploded")

    payload = {"text_content": "This should fail.", "options": {"method": "words", "max_size": 2, "overlap": 0}}
    response = client.post("/api/v1/chunking/chunk_text", json=payload)

    assert response.status_code == 500
    assert response.json() == {"detail": "An internal error occurred during text chunking"}


# For rolling_summarize tests, we need to patch load_server_configs and general_llm_analyzer
# where they are used in the endpoint (chunking.py)
@patch(PATH_TO_LOAD_CONFIGS_IN_ENDPOINT_MODULE, new=mock_load_server_configs_for_test)
@patch(PATH_TO_LLM_ANALYZER_IN_ENDPOINT_MODULE, new=mock_general_llm_analyzer_for_test)
@patch(PATH_TO_ICP_IN_ENDPOINT_MODULE)
def test_chunk_text_rolling_summarize_server_determines_llm(
    mock_icp, client: TestClient
):  # Order of args for mocks is inner-to-outer patch
    global mock_general_llm_analyzer_call_history_for_test
    mock_general_llm_analyzer_call_history_for_test = []  # Reset history

    def verified_chunking(*args):
        args[4][LLM_USAGE_TRACKER_KEY][LLM_USAGE_SUCCEEDED_KEY] = True
        return [
            {
                "text": "[Mock Rolling Summary From Test]",
                "metadata": {"method": "rolling_summarize", "chunk_index": 1, "total_chunks": 1},
            }
        ]

    mock_icp.side_effect = verified_chunking
    payload = {
        "text_content": "Content for server-determined LLM rolling summarize.",
        "options": {
            "method": "rolling_summarize",
            "tokenizer_name_or_path": "distilgpt2",
            "summarization_detail": 0.2,
            "llm_options_for_internal_steps": {
                "temperature": 0.6,
                "system_prompt_for_step": "Client custom step prompt.",
                "max_tokens_per_step": 25,
            },
        },
    }
    response = client.post("/api/v1/chunking/chunk_text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["chunks"][0]["text"] == "[Mock Rolling Summary From Test]"

    mock_icp.assert_called_once()
    passed_text, passed_opts, passed_tokenizer, passed_llm_func, passed_llm_config = mock_icp.call_args[0]

    assert passed_text == payload["text_content"]
    assert passed_opts["method"] == "rolling_summarize"
    assert passed_opts["summarization_detail"] == 0.2
    assert passed_tokenizer == "distilgpt2"

    # Crucially, check that the endpoint passed our *mocked* general_llm_analyzer
    assert passed_llm_func is mock_general_llm_analyzer_for_test

    # Verify server-determined LLM config (from mock_load_server_configs_for_test)
    # and client suggestions for other params
    assert passed_llm_config["api_name"] == "mock_llm"
    assert passed_llm_config["model"] == "mock_model_for_sum_tasks_server_default"
    assert passed_llm_config["api_key"] == "mock_api_key_for_tests"
    assert passed_llm_config["temp"] == 0.6
    assert passed_llm_config["system_message"] == "Client custom step prompt."
    assert passed_llm_config["max_tokens"] == 25


@patch(PATH_TO_LOAD_CONFIGS_IN_ENDPOINT_MODULE, new=mock_load_server_configs_for_test)
@patch(PATH_TO_LLM_ANALYZER_IN_ENDPOINT_MODULE, new=mock_general_llm_analyzer_for_test)
@patch(PATH_TO_ICP_IN_ENDPOINT_MODULE)
def test_chunk_file_rolling_summarize(mock_icp_file, client: TestClient):
    global mock_general_llm_analyzer_call_history_for_test
    mock_general_llm_analyzer_call_history_for_test = []

    def verified_file_chunking(*args):
        args[4][LLM_USAGE_TRACKER_KEY][LLM_USAGE_SUCCEEDED_KEY] = True
        return [
            {
                "text": "[Mock File Rolling Summary]",
                "metadata": {"method": "rolling_summarize", "chunk_index": 1, "total_chunks": 1},
            }
        ]

    mock_icp_file.side_effect = verified_file_chunking

    dummy_content = b"File content for rolling summarize test via file upload."
    files = {"file": ("test_roll_file.txt", dummy_content, "text/plain")}
    form_data = {
        "method": "rolling_summarize",
        "tokenizer_name_or_path": "gpt2-medium",
        "summarization_detail": "0.3",
        "llm_step_temperature": "0.75",
        "llm_step_system_prompt": "File step prompt.",
        "llm_step_max_tokens": "35",
    }
    response = client.post("/api/v1/chunking/chunk_file", files=files, data=form_data)

    assert response.status_code == 200
    data = response.json()
    assert data["chunks"][0]["text"] == "[Mock File Rolling Summary]"
    assert data["original_file_name"] == "test_roll_file.txt"

    mock_icp_file.assert_called_once()
    passed_text_file, passed_opts_file, passed_tokenizer_file, passed_llm_func_file, passed_llm_config_file = (
        mock_icp_file.call_args[0]
    )

    assert passed_text_file == dummy_content.decode()
    assert passed_opts_file["method"] == "rolling_summarize"
    assert passed_opts_file["tokenizer_name_or_path"] == "gpt2-medium"
    assert passed_opts_file["llm_options_for_internal_steps"]["temperature"] == 0.75
    assert passed_llm_func_file is mock_general_llm_analyzer_for_test

    assert passed_llm_config_file["api_name"] == "mock_llm"
    assert passed_llm_config_file["model"] == "mock_model_for_sum_tasks_server_default"
    assert passed_llm_config_file["temp"] == 0.75
    assert passed_llm_config_file["system_message"] == "File step prompt."
    assert passed_llm_config_file["max_tokens"] == 35


@pytest.mark.parametrize(
    "initial_key",
    ["sk-snapshot-a", None],
    ids=["A-to-B", "absent-to-B"],
)
def test_chunk_text_rolling_summarize_keeps_static_snapshot_at_analyzer_boundary(
    monkeypatch,
    client: TestClient,
    initial_key,
):
    state, initial_config, static_calls, lifecycle = _install_chunking_snapshot_rotation(
        monkeypatch,
        initial_key,
    )
    boundary_calls = []

    def fake_improved_chunking_process(text, options, tokenizer, llm_func, llm_config):
        state["key"] = "sk-snapshot-b"
        state["app_config"] = {"ollama_api": {"base_url": "http://snapshot-b.invalid"}}
        _invoke_rolling_analyzer(llm_func, llm_config, boundary_calls)
        return [
            {
                "text": "snapshot chunk",
                "metadata": {"method": "rolling_summarize", "chunk_index": 1, "total_chunks": 1},
            }
        ]

    monkeypatch.setattr(chunking_module, "load_server_configs", _snapshot_chunking_config)
    monkeypatch.setattr(chunking_module, "improved_chunking_process", fake_improved_chunking_process)

    response = client.post(
        "/api/v1/chunking/chunk_text",
        json={
            "text_content": "ordinary rolling snapshot input",
            "options": {"method": "rolling_summarize", "max_size": 10, "overlap": 0},
        },
    )

    assert response.status_code == 200
    _assert_chunking_boundary_snapshot(
        boundary_calls,
        expected_key=initial_key,
        expected_config=initial_config,
        static_calls=static_calls,
        lifecycle=lifecycle,
    )


@pytest.mark.parametrize(
    "initial_key",
    ["sk-snapshot-a", None],
    ids=["A-to-B", "absent-to-B"],
)
def test_chunk_file_rolling_summarize_keeps_static_snapshot_at_analyzer_boundary(
    monkeypatch,
    client: TestClient,
    initial_key,
):
    state, initial_config, static_calls, lifecycle = _install_chunking_snapshot_rotation(
        monkeypatch,
        initial_key,
    )
    boundary_calls = []

    def fake_improved_chunking_process(text, options, tokenizer, llm_func, llm_config):
        state["key"] = "sk-snapshot-b"
        state["app_config"] = {"ollama_api": {"base_url": "http://snapshot-b.invalid"}}
        _invoke_rolling_analyzer(llm_func, llm_config, boundary_calls)
        return [
            {
                "text": "snapshot file chunk",
                "metadata": {"method": "rolling_summarize", "chunk_index": 1, "total_chunks": 1},
            }
        ]

    monkeypatch.setattr(chunking_module, "load_server_configs", _snapshot_chunking_config)
    monkeypatch.setattr(chunking_module, "improved_chunking_process", fake_improved_chunking_process)

    response = client.post(
        "/api/v1/chunking/chunk_file",
        files={"file": ("snapshot.txt", b"file rolling snapshot input", "text/plain")},
        data={"method": "rolling_summarize", "max_size": "10", "overlap": "0"},
    )

    assert response.status_code == 200
    _assert_chunking_boundary_snapshot(
        boundary_calls,
        expected_key=initial_key,
        expected_config=initial_config,
        static_calls=static_calls,
        lifecycle=lifecycle,
    )


@pytest.mark.parametrize(
    "initial_key",
    ["sk-snapshot-a", None],
    ids=["A-to-B", "absent-to-B"],
)
def test_chunk_template_rolling_summarize_keeps_static_snapshot_at_analyzer_boundary(
    monkeypatch,
    client: TestClient,
    initial_key,
):
    from tldw_Server_API.app.core.Chunking.templates import TemplateProcessor

    state, initial_config, static_calls, lifecycle = _install_chunking_snapshot_rotation(
        monkeypatch,
        initial_key,
    )
    boundary_calls = []

    class TemplateDatabase:
        def get_chunking_template(self, *, name):
            assert name == "snapshot-template"
            return {
                "name": name,
                "description": "snapshot template",
                "template_json": (
                    '{"chunking":{"method":"rolling_summarize",'
                    '"config":{"max_size":10,"overlap":0}}}'
                ),
                "tags": [],
                "version": 1,
            }

    def fake_process_template(self, text, template, **options):
        state["key"] = "sk-snapshot-b"
        state["app_config"] = {"ollama_api": {"base_url": "http://snapshot-b.invalid"}}
        _invoke_rolling_analyzer(
            self._chunker.llm_call_func,
            self._chunker.llm_config,
            boundary_calls,
        )
        return ["snapshot template chunk"]

    monkeypatch.setattr(chunking_module, "load_server_configs", _snapshot_chunking_config)
    monkeypatch.setattr(TemplateProcessor, "process_template", fake_process_template)
    app_for_testing_with_real_router.dependency_overrides[try_get_media_db_for_user] = TemplateDatabase
    try:
        response = client.post(
            "/api/v1/chunking/chunk_text",
            json={
                "text_content": "template rolling snapshot input",
                "options": {"template_name": "snapshot-template"},
            },
        )
    finally:
        app_for_testing_with_real_router.dependency_overrides.pop(try_get_media_db_for_user, None)

    assert response.status_code == 200
    _assert_chunking_boundary_snapshot(
        boundary_calls,
        expected_key=initial_key,
        expected_config=initial_config,
        static_calls=static_calls,
        lifecycle=lifecycle,
    )


@patch(PATH_TO_ICP_IN_ENDPOINT_MODULE)
def test_chunk_text_endpoint_no_options(mock_icp, client: TestClient):
    mock_icp.return_value = [
        {
            "text": "Default chunked",
            "metadata": {
                "method": DEFAULT_CHUNK_OPTIONS_REAL.get("method"),
                "chunk_index": 1,
                "total_chunks": 1,
            },
        }
    ]
    payload = {"text_content": "Text with no options."}

    response = client.post("/api/v1/chunking/chunk_text", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Check applied_options in response reflect defaults
    assert data["applied_options"]["method"] == DEFAULT_CHUNK_OPTIONS_REAL.get("method")
    assert data["applied_options"]["max_size"] == DEFAULT_CHUNK_OPTIONS_REAL.get("max_size")

    mock_icp.assert_called_once()
    # Check options passed to improved_chunking_process also reflect defaults
    _, passed_options_to_icp, _, _, _ = mock_icp.call_args[0]
    assert passed_options_to_icp["method"] == DEFAULT_CHUNK_OPTIONS_REAL.get("method")
    assert passed_options_to_icp["max_size"] == DEFAULT_CHUNK_OPTIONS_REAL.get("max_size")


@patch(PATH_TO_ICP_IN_ENDPOINT_MODULE)
def test_chunk_file_sanitizes_unexpected_error(mock_icp, client: TestClient):
    mock_icp.side_effect = RuntimeError("chunking backend exploded")

    files = {"file": ("test.txt", b"Chunk this content", "text/plain")}
    form_data = {
        "method": "words",
        "max_size": "10",
        "overlap": "0",
    }

    response = client.post("/api/v1/chunking/chunk_file", files=files, data=form_data)

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal error during file chunking"}


def test_chunk_text_endpoint_invalid_payload_missing_text(client: TestClient):
    payload = {"options": {"method": "words"}}
    response = client.post("/api/v1/chunking/chunk_text", json=payload)
    assert response.status_code == 422
    assert "text_content" in response.json()["detail"][0]["loc"]


#
# End of test_chunking_endpoint.py
#######################################################################################################################
