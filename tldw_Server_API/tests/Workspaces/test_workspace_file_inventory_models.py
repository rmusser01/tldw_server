from __future__ import annotations

import base64
import json

import pytest

from tldw_Server_API.app.core.Workspaces.file_inventory_models import (
    INVENTORY_COUNT_KEYS,
    MAX_INVENTORY_DIAGNOSTICS,
    bounded_inventory_diagnostics,
    decode_inventory_cursor,
    encode_inventory_cursor,
    normalize_durable_inventory_state,
    normalize_inventory_counts,
    normalize_inventory_state,
    redact_inventory_path_hint,
    sort_inventory_relative_paths,
)


def test_inventory_state_normalization_keeps_projected_states_and_fails_closed() -> None:
    assert normalize_inventory_state("queued") == "queued"
    assert normalize_inventory_state(" STALE ") == "stale"
    assert normalize_inventory_state("not_started") == "not_started"
    assert normalize_inventory_state("unexpected") == "failed"
    assert normalize_inventory_state(None) == "failed"


def test_durable_inventory_state_rejects_projected_only_states() -> None:
    assert normalize_durable_inventory_state("queued") == "queued"
    assert normalize_durable_inventory_state("current") == "current"
    assert normalize_durable_inventory_state("stale") == "failed"
    assert normalize_durable_inventory_state("not_started") == "failed"


def test_inventory_counts_default_missing_keys_and_clamp_invalid_values() -> None:
    counts = normalize_inventory_counts(
        {
            "files": "3",
            "directories": 2,
            "ignored": True,
            "diagnostics": -4,
            "total_entries": "7",
        }
    )

    assert set(counts) == set(INVENTORY_COUNT_KEYS)
    assert counts["files"] == 3
    assert counts["directories"] == 2
    assert counts["ignored"] == 0
    assert counts["diagnostics"] == 0
    assert counts["symlinks"] == 0
    assert counts["indexing_candidates"] == 0
    assert counts["total_entries"] == 7


def test_diagnostics_are_bounded_and_absolute_paths_are_redacted() -> None:
    raw = [
        {
            "code": "permission_denied",
            "path_hint": f"/home/alice/private/project/secret-{index}.txt",
            "message": f"Permission denied for /home/alice/private/project/secret-{index}.txt",
        }
        for index in range(MAX_INVENTORY_DIAGNOSTICS + 5)
    ]

    diagnostics = bounded_inventory_diagnostics(raw)

    assert len(diagnostics) == MAX_INVENTORY_DIAGNOSTICS
    assert diagnostics[0]["code"] == "permission_denied"
    assert diagnostics[0]["path_hint"] == "secret-0.txt"
    assert "/home/alice" not in diagnostics[0]["message"]


def test_diagnostics_never_expose_absolute_path_hints() -> None:
    diagnostics = bounded_inventory_diagnostics(
        [{"path_hint": "/home/alice/project/secrets/token.txt"}],
        root_relative_only=False,
    )

    assert diagnostics[0]["path_hint"] == "token.txt"


def test_diagnostics_drop_malformed_entries_and_default_missing_fields() -> None:
    diagnostics = bounded_inventory_diagnostics([None, {"message": ""}, {"path_hint": "../outside/.env"}])

    assert diagnostics == [
        {
            "code": "scan_diagnostic",
            "message": "A path could not be inspected.",
        },
        {
            "code": "scan_diagnostic",
            "path_hint": "outside/.env",
            "message": "A path could not be inspected.",
        },
    ]


def test_path_hint_redaction_preserves_safe_relative_hints() -> None:
    assert redact_inventory_path_hint("src\\client\\app.tsx") == "src/client/app.tsx"
    assert redact_inventory_path_hint("/Users/alice/project/src/private.py") == "private.py"
    assert redact_inventory_path_hint("../outside/secrets.env") == "outside/secrets.env"
    assert redact_inventory_path_hint("") is None


def test_inventory_cursors_are_opaque_and_round_trip_relative_paths() -> None:
    cursor = encode_inventory_cursor("src/main.py")

    assert cursor != "src/main.py"
    assert decode_inventory_cursor(cursor) == "src/main.py"


@pytest.mark.parametrize("relative_path", ["", "/tmp/secret.txt", "../secret.txt", "src/../../secret.txt"])
def test_inventory_cursor_encoding_rejects_unsafe_paths(relative_path: str) -> None:
    with pytest.raises(ValueError):
        encode_inventory_cursor(relative_path)


def test_inventory_cursor_decoding_rejects_invalid_or_unsafe_payloads() -> None:
    unsafe_payload = base64.urlsafe_b64encode(
        json.dumps({"v": 1, "relative_path": "/tmp/secret.txt"}).encode("utf-8")
    ).decode("ascii").rstrip("=")

    with pytest.raises(ValueError):
        decode_inventory_cursor("not-base64")
    with pytest.raises(ValueError):
        decode_inventory_cursor(unsafe_payload)


def test_inventory_relative_paths_sort_for_cursor_pagination() -> None:
    assert sort_inventory_relative_paths(["z/readme.md", "a/main.py", "a/README.md"]) == [
        "a/README.md",
        "a/main.py",
        "z/readme.md",
    ]
