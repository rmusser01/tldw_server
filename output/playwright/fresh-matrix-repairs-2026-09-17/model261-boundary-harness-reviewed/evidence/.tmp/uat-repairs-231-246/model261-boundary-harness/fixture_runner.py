"""Temporary UAT261 fixture runner; copy under Character_Chat only to execute."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_HARNESS_PATH = Path(__file__).parents[3] / ".tmp/uat-repairs-231-246/model261-boundary-harness/boundary_harness.py"
_SPEC = importlib.util.spec_from_file_location("uat261_boundary_harness", _HARNESS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_HARNESS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HARNESS)


@pytest.mark.asyncio
async def test_uat261_boundary_projection_uses_existing_character_chat_fixture_seam(
    monkeypatch,
    tmp_path,
    healthy_absent_provider_override_snapshot,
    character_provider_adapter_boundary,
):
    _provider_calls, bind_provider_call = character_provider_adapter_boundary
    evidence = await _HARNESS.run_fixture_harness(monkeypatch, tmp_path, bind_provider_call)

    assert [call["call_label"] for call in evidence["calls"]] == ["A", "B"]
    assert evidence["calls"][0]["message_fingerprint"] == evidence["calls"][1]["message_fingerprint"]
    assert evidence["calls"][1]["provider_response"]["usage"] is None
