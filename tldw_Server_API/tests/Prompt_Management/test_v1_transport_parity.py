"""Cross-language schema-v1 transport normalization contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.prompts_db_helpers import (
    parse_stored_prompt_definition,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "Docs" / "fixtures" / "single-text-recipes" / "v1-transport-cases.json"
)
V1_TRANSPORT_CASES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
V1_INTEGER_SYNC_CASES = json.loads(FIXTURE_PATH.with_name("v1-integer-sync-cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", V1_TRANSPORT_CASES, ids=lambda case: case["name"])
def test_python_v1_parser_matches_shared_transport_normalization(case: dict) -> None:
    payload = copy.deepcopy(case["input"])
    before = copy.deepcopy(payload)

    parsed = parse_stored_prompt_definition(payload, schema_version=1)

    assert parsed.model_dump() == case["expected"]
    assert payload == before


@pytest.mark.parametrize("case", V1_INTEGER_SYNC_CASES, ids=lambda case: case["name"])
def test_python_v1_parser_preserves_shared_integer_fixture_exactly(case: dict) -> None:
    """Python accepts wider v1 ints; sync eligibility is a cross-runtime rule."""
    payload = {
        "schema_version": 1,
        "format": "structured",
        "variables": [{"name": "topic"}],
        "blocks": [
            {
                "id": "task",
                "name": "Task",
                "role": "user",
                "content": "Explain the topic.",
                "order": 10,
            }
        ],
    }
    if case["field"] == "order":
        payload["blocks"][0]["order"] = case["input_value"]
    else:
        payload["variables"][0]["max_length"] = case["input_value"]
    before = copy.deepcopy(payload)

    parsed = parse_stored_prompt_definition(payload, schema_version=1)
    parsed_value = parsed.blocks[0].order if case["field"] == "order" else parsed.variables[0].max_length

    assert parsed_value == case["python_value"]
    assert payload == before
