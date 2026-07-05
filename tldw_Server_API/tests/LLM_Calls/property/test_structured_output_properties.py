"""Property tests for structured-output JSON extraction (RA4, triage pick).

``parse_structured_output`` extracts a JSON value from arbitrary model text
(bare, fenced, or with surrounding prose). Its invariants:

* totality — for any input it either returns a JSON-compatible value or raises
  ``StructuredOutputNoPayloadError``; a raw ``JSONDecodeError`` never leaks
* passthrough — an already-parsed dict/list is returned unchanged
* serializable — any returned value round-trips through ``json.dumps``
* round-trip — a JSON value embedded (bare or fenced) in text is recovered

``extract_items`` normalizes parsed output into a list of item dicts; its
lenient mode never raises on a parsed value and always yields a list.
"""
from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.exceptions import StructuredOutputSchemaError
from tldw_Server_API.app.core.LLM_Calls.structured_output import (
    StructuredOutputNoPayloadError,
    extract_items,
    parse_structured_output,
)

pytestmark = [pytest.mark.unit, pytest.mark.property]

_COMMON = hyp_settings(max_examples=200, deadline=None)

# JSON values (objects/arrays at the top level are what the extractor targets).
# Backticks are excluded from generated text: a "```" inside a JSON string would
# collide with the ```json fence in the round-trip test and spuriously fail —
# the parser is not expected to recover JSON from a malformed markdown fence.
def _safe_text(min_size: int = 0, max_size: int = 20) -> st.SearchStrategy[str]:
    """Text without backticks (which would corrupt a ```json fence)."""
    return st.text(
        alphabet=st.characters(blacklist_characters="`"), min_size=min_size, max_size=max_size
    )


_json_scalars = st.none() | st.booleans() | st.integers() | _safe_text(max_size=20)
_json_values = st.recursive(
    _json_scalars,
    lambda children: st.lists(children, max_size=4)
    | st.dictionaries(_safe_text(min_size=1, max_size=8), children, max_size=4),
    max_leaves=12,
)
_json_containers = st.one_of(
    st.lists(_json_values, max_size=5),
    st.dictionaries(_safe_text(min_size=1, max_size=8), _json_values, max_size=5),
)


class TestParseStructuredOutput:
    @_COMMON
    @given(payload=st.text(max_size=80))
    def test_totality_on_arbitrary_text(self, payload: str) -> None:
        """Only StructuredOutputNoPayloadError may be raised; no raw JSON error."""
        try:
            result = parse_structured_output(payload)
        except StructuredOutputNoPayloadError:
            return
        # a returned value must always be JSON-serializable
        json.dumps(result)

    @_COMMON
    @given(payload=st.none() | st.text(alphabet=" \t\n", max_size=10))
    def test_empty_payloads_raise_no_payload(self, payload: Any) -> None:
        """None / whitespace-only payloads raise the typed no-payload error."""
        with pytest.raises(StructuredOutputNoPayloadError):
            parse_structured_output(payload)

    @_COMMON
    @given(value=_json_containers)
    def test_dict_or_list_input_is_returned_unchanged(self, value: Any) -> None:
        """An already-parsed dict/list passes through untouched."""
        assert parse_structured_output(value) == value

    @_COMMON
    @given(value=_json_containers, fenced=st.booleans())
    def test_round_trip_of_embedded_json(self, value: Any, fenced: bool) -> None:
        """A JSON container serialized into text (bare or ```json fenced) is
        recovered by the parser."""
        as_text = json.dumps(value)
        payload = f"```json\n{as_text}\n```" if fenced else as_text
        parsed = parse_structured_output(payload)
        assert parsed == value


class TestExtractItems:
    @_COMMON
    @given(value=_json_containers)
    def test_lenient_extract_returns_list_or_typed_schema_error(self, value: Any) -> None:
        """extract_items either yields a list of item dicts or raises the typed
        StructuredOutputSchemaError (e.g. a top-level list of non-objects) — it
        never leaks an untyped exception."""
        try:
            result = extract_items(value)
        except StructuredOutputSchemaError:
            return
        assert isinstance(result, list)
        assert all(isinstance(item, dict) for item in result)

    @_COMMON
    @given(
        items=st.lists(
            st.dictionaries(_safe_text(min_size=1, max_size=6), _json_scalars, max_size=4),
            max_size=6,
        )
    )
    def test_top_level_list_of_objects_is_preserved(self, items: list[dict]) -> None:
        """A top-level list of objects is returned as-is when allowed."""
        result = extract_items(items, allow_top_level_list=True)
        assert result == items

    @_COMMON
    @given(
        wrapper=_safe_text(min_size=1, max_size=8),
        items=st.lists(
            st.dictionaries(_safe_text(min_size=1, max_size=6), _json_scalars, max_size=3),
            max_size=5,
        ),
    )
    def test_wrapped_items_are_unwrapped(self, wrapper: str, items: list[dict]) -> None:
        """Items under the wrapper key are unwrapped to a bare list."""
        result = extract_items({wrapper: items}, wrapper_key=wrapper)
        assert result == items
