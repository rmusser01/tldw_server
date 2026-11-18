"""Saved citation checks are bounded historical data, never access authority."""

from copy import deepcopy

import pytest

from tldw_Server_API.app.core.Chat.workspace_references import validate_workspace_assistant_metadata

pytestmark = pytest.mark.unit


def report():
    return {
        "version": 1, "text_sha256": "a" * 64, "verdict": "unverified",
        "provider": None, "model": None,
        "coverage": {"total": 0, "evaluated": 0, "supported": 0, "uncovered": 0, "complete": False},
        "findings": [], "evidence": {}, "limits": {},
        "context": {"unavailable_reason": "Checker unavailable."},
    }


def test_historical_unavailable_report_is_preserved_without_normalization():
    raw = {"claim_verification": report(), "claim_verification_imported": True}
    original = deepcopy(raw)
    assert validate_workspace_assistant_metadata(raw) == original == raw


@pytest.mark.parametrize("raw", [
    None, {}, {"unknown": "not allowed"},
    {"claim_verification_imported": True},
    {"claim_verification": report(), "claim_verification_imported": False},
    {"claim_verification": report(), "claim_verification_imported": 1},
    {"claim_verification": {"verdict": "supported"}},
])
def test_unknown_or_unbound_import_metadata_is_refused(raw):
    with pytest.raises(ValueError):
        validate_workspace_assistant_metadata(raw)
