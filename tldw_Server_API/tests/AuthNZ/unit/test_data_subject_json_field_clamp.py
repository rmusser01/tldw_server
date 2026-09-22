"""JSON-blob coercion must clamp to the container the caller asked for.

Four of the five AuthNZ repo copies end with
    return dict(parsed) if isinstance(parsed, dict) else {}
clamping the parsed value to the expected container. data_subject_requests_repo's copy
returned json.loads(...) raw, so a column holding "[]", "null", "123" or "\\"x\\"" yielded
a list / None / int / str where the caller's `fallback` promised a dict.

That matters here because coverage_metadata is declared `dict[str, Any]` on the response
models (api/v1/schemas/admin_schemas.py:1027, :1082), so an unclamped value becomes a
validation error on a GDPR data-subject-request read rather than a benign default.
"""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.AuthNZ.repos.data_subject_requests_repo import (
    AuthnzDataSubjectRequestsRepo,
)

parse = AuthnzDataSubjectRequestsRepo._parse_json_field


@pytest.mark.parametrize("stored", ['"[]"', "null", "123", '"x"', "true"])
def test_non_dict_json_clamps_to_dict_fallback(stored: str) -> None:
    got = parse(stored, fallback={})
    assert isinstance(got, dict), f"stored {stored!r} produced {type(got).__name__}, not dict"


@pytest.mark.parametrize("stored", ["{}", '{"a": 1}', "null", "123"])
def test_non_list_json_clamps_to_list_fallback(stored: str) -> None:
    got = parse(stored, fallback=[])
    assert isinstance(got, list), f"stored {stored!r} produced {type(got).__name__}, not list"


def test_well_formed_values_still_pass_through() -> None:
    assert parse('{"a": 1}', fallback={}) == {"a": 1}
    assert parse('["a", "b"]', fallback=[]) == ["a", "b"]


def test_already_parsed_containers_pass_through() -> None:
    assert parse({"a": 1}, fallback={}) == {"a": 1}
    assert parse(["a"], fallback=[]) == ["a"]


def test_none_and_malformed_return_the_fallback() -> None:
    assert parse(None, fallback={}) == {}
    assert parse("{not json", fallback={}) == {}
    assert parse(None, fallback=[]) == []
