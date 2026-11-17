"""Egress checks must not relax under pytest/TESTING/TEST_MODE (TASK-13950.19)."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Files
from tldw_Server_API.app.core.Ingestion_Media_Processing.Video import Video_DL_Ingestion_Lib
from tldw_Server_API.app.core.Utils.egress import evaluate_url_policy


@pytest.fixture(autouse=True)
def _test_context(monkeypatch):
    # PYTEST_CURRENT_TEST is set by pytest itself.
    monkeypatch.setenv("TESTING", "1")
    monkeypatch.setenv("TLDW_TEST_MODE", "y")
    monkeypatch.setenv("WORKFLOWS_EGRESS_ALLOWED_PORTS", "80,443")


def test_loopback_hosts_keep_the_port_policy_under_pytest(monkeypatch):
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "false")
    assert evaluate_url_policy("http://localhost:11434/x").reason_code == "port_not_allowed"


def test_audio_url_validation_blocks_private_addresses_under_pytest(monkeypatch):
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")
    assert Audio_Files._validate_outbound_url("http://10.0.0.5/a.mp3")


def test_video_url_validation_blocks_private_addresses_under_pytest(monkeypatch):
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")
    with pytest.raises(ValueError, match="blocked by security policy"):
        Video_DL_Ingestion_Lib._raise_if_url_blocked("http://10.0.0.5/v.mp4")
