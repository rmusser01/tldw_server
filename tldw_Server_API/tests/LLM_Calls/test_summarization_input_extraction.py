"""Input extraction for ``analyze()`` (TASK-13288, supersedes the TASK-2425 note).

``Summarization_General_Lib`` used to define ``extract_text_from_input`` twice. The
first copy read any string that named an existing file; the second, which won at
runtime, did not. That shadowing was the only thing keeping a caller-controlled
path from being read through ``analyze()``. These tests pin the security property
directly and cover the input shapes the surviving copy used to drop.
"""

from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as sgl


@pytest.fixture
def dispatched(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Capture the text analyze() hands to the provider."""
    seen: list[str] = []

    def dispatch(text: str, *_args: Any, **_kwargs: Any) -> str:
        seen.append(text)
        return "summary"

    monkeypatch.setattr(sgl, "_dispatch_to_api", dispatch)
    return seen


def test_single_module_level_definition() -> None:
    source = open(sgl.__file__, encoding="utf-8").read()
    assert source.count("\ndef extract_text_from_input(") == 1
    assert "F811" not in source


@pytest.mark.parametrize("key", ["text", "content"])
def test_dict_text_and_content_keys_reach_the_provider(dispatched: list[str], key: str) -> None:
    result = sgl.analyze("openai", {key: "the body"}, None)

    assert result == "summary", result
    assert dispatched == ["the body"]


def test_filesystem_path_is_not_read(dispatched: list[str], tmp_path: Any) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET-FILE-CONTENTS", encoding="utf-8")

    sgl.analyze("openai", str(secret), None)

    assert dispatched == [str(secret)], "the path must be treated as literal text"
    assert "TOP-SECRET-FILE-CONTENTS" not in "".join(dispatched)


def test_extractor_never_opens_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    secret = tmp_path / "secret.json"
    secret.write_text('{"text": "leaked"}', encoding="utf-8")

    def _no_open(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("extract_text_from_input must not open files")

    monkeypatch.setattr("builtins.open", _no_open)
    assert sgl.extract_text_from_input(str(secret)) == str(secret)


@pytest.mark.parametrize("raw", ["123", "true", "null", '"quoted"', "[1, 2]"])
def test_scalar_or_non_object_json_is_literal_text(dispatched: list[str], raw: str) -> None:
    result = sgl.analyze("openai", raw, None)

    assert result == "summary", result
    assert dispatched == [raw]


def test_known_media_fields_are_still_labelled() -> None:
    text = sgl.extract_text_from_input(
        {"title": "T", "description": "D", "transcription": [{"Text": "a"}, {"Text": "b"}]}
    )
    assert text == "Title: T\n\nDescription: D\n\nTranscription: a b"
