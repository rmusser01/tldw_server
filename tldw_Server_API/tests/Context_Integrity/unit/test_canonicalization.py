from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.unit


def test_filesystem_digest_is_stable_for_sorted_paths() -> None:
    from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
        canonical_filesystem_digest,
    )

    first = canonical_filesystem_digest(
        source_type="skill_file",
        asset_id="skill:user:1/demo",
        files={
            "SKILL.md": b"hello\r\n",
            "refs/notes.md": b"reference",
        },
        metadata={"context": "inline"},
    )
    second = canonical_filesystem_digest(
        source_type="skill_file",
        asset_id="skill:user:1/demo",
        files={
            "refs/notes.md": b"reference",
            "SKILL.md": b"hello\r\n",
        },
        metadata={"context": "inline"},
    )

    assert first == second
    assert first.startswith("sha256:")


def test_filesystem_digest_detects_formatting_edits() -> None:
    from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
        canonical_filesystem_digest,
    )

    original = canonical_filesystem_digest(
        source_type="prompt_file",
        asset_id="prompt_file:rag.prompts.yaml",
        files={"rag.prompts.yaml": b"answer: one\n"},
    )
    edited = canonical_filesystem_digest(
        source_type="prompt_file",
        asset_id="prompt_file:rag.prompts.yaml",
        files={"rag.prompts.yaml": b"answer: one\n# changed\n"},
    )

    assert original != edited


def test_db_prompt_digest_normalizes_unicode_and_line_endings() -> None:
    from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
        canonical_db_prompt_digest,
    )

    composed = canonical_db_prompt_digest(
        {
            "uuid": "prompt-1",
            "version": 3,
            "name": "Cafe",
            "system": "caf\u00e9\r\nline",
            "user": "body",
            "structured": {"b": 2, "a": 1},
        }
    )
    decomposed = canonical_db_prompt_digest(
        {
            "structured": {"a": 1, "b": 2},
            "user": "body",
            "system": "cafe\u0301\nline",
            "name": "Cafe",
            "version": 3,
            "uuid": "prompt-1",
        }
    )

    assert composed == decomposed
    payload = json.loads(composed.canonical_json)
    assert payload["system"] == "caf\u00e9\nline"
