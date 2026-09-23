"""Shared AuthNZ row adapter and JSON-blob coercer (repos/_dual_backend.py).

These replaced 19 per-repo ``_row_to_dict`` copies and five JSON coercers that had
drifted apart: nine adapters had no None guard (``dict(None)`` raised TypeError), and
one coercer (data_subject_requests) did not clamp to the expected container, so a
column holding "[]", "null" or "123" produced a non-dict where the response model
declares ``dict[str, Any]``. None of the coercers decoded bytes.
"""

from __future__ import annotations

import sqlite3

import pytest

from tldw_Server_API.app.core.AuthNZ.repos._dual_backend import load_json, row_dict
from tldw_Server_API.app.core.AuthNZ.repos.data_subject_requests_repo import (
    AuthnzDataSubjectRequestsRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.identity_provider_repo import IdentityProviderRepo
from tldw_Server_API.app.core.AuthNZ.repos.managed_secret_refs_repo import (
    ManagedSecretRefsRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
    AuthnzOrgProviderSecretsRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
    AuthnzUserProviderSecretsRepo,
)

pytestmark = pytest.mark.unit


# --- row_dict -------------------------------------------------------------------


def test_row_dict_none_is_empty() -> None:
    assert row_dict(None) == {}


def test_row_dict_copies_dicts() -> None:
    src = {"a": 1}
    out = row_dict(src)
    assert out == src
    assert out is not src


def test_row_dict_materializes_sqlite_row() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT 1 AS id, 'x' AS name").fetchone()
    assert row_dict(row) == {"id": 1, "name": "x"}


def test_row_dict_unmaterializable_row_raises() -> None:
    with pytest.raises(TypeError):
        row_dict(object())


@pytest.mark.parametrize(
    "adapter",
    [
        IdentityProviderRepo._row_to_dict,
        AuthnzOrgProviderSecretsRepo._row_to_dict,
        AuthnzUserProviderSecretsRepo._row_to_dict,
    ],
)
def test_formerly_unguarded_repo_adapters_accept_none(adapter) -> None:
    assert adapter(None) == {}


# --- load_json ------------------------------------------------------------------


@pytest.mark.parametrize("stored", ['"[]"', "[]", "null", "123", '"x"', "true"])
def test_wrong_shaped_json_clamps_to_dict(stored: str) -> None:
    assert load_json(stored, dict) == {}


@pytest.mark.parametrize("stored", ["{}", '{"a": 1}', "null", "123"])
def test_wrong_shaped_json_clamps_to_list(stored: str) -> None:
    assert load_json(stored, list) == []


@pytest.mark.parametrize("stored", [None, "", "   ", "{not json", b"\xff\xfe", 5, object()])
def test_none_malformed_and_foreign_values_are_empty(stored) -> None:
    assert load_json(stored, dict) == {}
    assert load_json(stored, list) == []


def test_well_formed_values_pass_through() -> None:
    assert load_json('{"a": 1}', dict) == {"a": 1}
    assert load_json('["a", "b"]', list) == ["a", "b"]


def test_bytes_are_decoded() -> None:
    assert load_json(b'{"a": 1}', dict) == {"a": 1}
    assert load_json(bytearray(b'["a"]'), list) == ["a"]


def test_already_decoded_containers_are_copied() -> None:
    src = {"a": 1}
    out = load_json(src, dict)
    assert out == src
    assert out is not src
    assert load_json(["a"], list) == ["a"]


def test_managed_secret_ref_decodes_bytes_blob() -> None:
    row = ManagedSecretRefsRepo._row_to_dict({"metadata_json": b'{"owner": "admin"}'})
    assert row["metadata"] == {"owner": "admin"}


@pytest.mark.parametrize("stored", ['"[]"', "null", "123", '"x"'])
def test_data_subject_request_coverage_metadata_is_always_a_dict(stored: str) -> None:
    record = AuthnzDataSubjectRequestsRepo._normalize_record(
        {"coverage_metadata": stored, "selected_categories": stored, "preview_summary": stored}
    )
    assert record["coverage_metadata"] == {}
    assert record["selected_categories"] == []
    assert record["preview_summary"] == []
