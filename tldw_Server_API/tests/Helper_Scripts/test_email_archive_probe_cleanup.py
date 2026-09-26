"""The disposable archive probe cleanup must reject unrelated resource names."""

import runpy
from pathlib import Path

import pytest


@pytest.fixture
def validate_manifest(monkeypatch, tmp_path):
    monkeypatch.setenv("EMAIL_PROBE_PG_MANIFEST", str(tmp_path / "manifest.json"))
    monkeypatch.setenv("EMAIL_PROBE_PG_CONTAINER", "unused-test-container")
    script = Path(__file__).resolve().parents[3] / "Docs/Operations/probes/email_archive_probe_databases_2026_09_25.py"
    return runpy.run_path(str(script))["validate_manifest"]


def test_generated_probe_resources_are_accepted(validate_manifest):
    manifest = {
        "host": "127.0.0.1",
        "port": 5434,
        "role": "email_probe_012345abcd",
        "auth_db": "email_auth_012345abcd",
        "content_db": "email_content_012345abcd",
    }
    validate_manifest(manifest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("role", "postgres"),
        ("auth_db", "production"),
        ("content_db", "email_content_9999999999"),
        ("host", "192.0.2.1"),
        ("port", 5432),
    ],
)
def test_cleanup_rejects_unrelated_targets(validate_manifest, field, value):
    manifest = {
        "host": "127.0.0.1",
        "port": 5434,
        "role": "email_probe_012345abcd",
        "auth_db": "email_auth_012345abcd",
        "content_db": "email_content_012345abcd",
    }
    manifest[field] = value
    with pytest.raises(ValueError):
        validate_manifest(manifest)
