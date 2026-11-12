"""Regressions for call sites migrated onto core/Utils/coercion.py (TASK-13322)."""

import contextlib

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends import (
    dolphin_ocr,
    dots_ocr,
    hunyuan_ocr,
)
from tldw_Server_API.app.core.LLM_Calls.providers import google_adapter
from tldw_Server_API.app.core.RAG.rag_service import request_resolution


@pytest.mark.unit
def test_dots_vllm_data_url_survives_trailing_whitespace(monkeypatch):
    """DOTS_VLLM_USE_DATA_URL="true " (compose environment: list) must still inline bytes,
    not send a server-local file path to a remote vLLM."""
    seen = {}

    @contextlib.contextmanager
    def fake_payload(image_bytes, *, use_data_url):
        seen["use_data_url"] = use_data_url
        yield {"type": "image_url"}

    monkeypatch.setattr(dots_ocr, "image_payload", fake_payload)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.http_client.fetch_json",
        lambda **_: {"choices": [{"message": {"content": "ok"}}]},
    )
    monkeypatch.setenv("DOTS_VLLM_URL", "http://vllm.example:8000/v1/chat/completions")
    monkeypatch.setenv("DOTS_VLLM_USE_DATA_URL", "true ")
    assert dots_ocr._ocr_via_vllm(b"img", "prompt") == "ok"
    assert seen["use_data_url"] is True


@pytest.mark.unit
def test_hunyuan_and_dolphin_flags_are_stripped(monkeypatch):
    monkeypatch.setenv("HUNYUAN_CLEAN_REPEATS", "Yes ")
    assert hunyuan_ocr._should_clean_repeats() is True
    monkeypatch.setenv("DOLPHIN_USE_DATA_URL", " on")
    assert dolphin_ocr._bool_env("DOLPHIN_USE_DATA_URL", False) is True


@pytest.mark.unit
@pytest.mark.parametrize("token", ["n", "disabled", "nope"])
def test_gemini_url_beta_flags_do_not_fail_open(monkeypatch, token):
    monkeypatch.setenv("LLM_ADAPTERS_GEMINI_IMAGE_URLS_BETA", token)
    assert google_adapter._env_flag("LLM_ADAPTERS_GEMINI_IMAGE_URLS_BETA") is False


@pytest.mark.unit
def test_search_agent_flags_accept_y_like_the_rest_of_the_request():
    assert request_resolution._is_truthy_value("y") is True
    assert request_resolution._is_truthy_value("n") is False
