"""The disposable archive probe cleanup must reject unrelated resource names."""

import json
import runpy
from pathlib import Path
from secrets import token_urlsafe

import pytest


@pytest.fixture
def validate_manifest(monkeypatch, tmp_path):
    monkeypatch.delenv("EMAIL_PROBE_PG_PORT", raising=False)
    monkeypatch.setenv("EMAIL_PROBE_PG_MANIFEST", str(tmp_path / "manifest.json"))
    monkeypatch.setenv("EMAIL_PROBE_PG_CONTAINER", "unused-test-container")
    script = Path(__file__).resolve().parents[3] / "Docs/Operations/probes/email_archive_probe_databases_2026_09_25.py"
    monkeypatch.syspath_prepend(str(script.parent))
    return runpy.run_path(str(script))["validate_manifest"]


def _load_provisioner(monkeypatch, tmp_path, port=None):
    script = Path(__file__).resolve().parents[3] / "Docs/Operations/probes/email_archive_probe_databases_2026_09_25.py"
    monkeypatch.syspath_prepend(str(script.parent))
    monkeypatch.setenv("EMAIL_PROBE_PG_MANIFEST", str(tmp_path / "manifest.json"))
    monkeypatch.setenv("EMAIL_PROBE_PG_CONTAINER", "unused-test-container")
    if port is None:
        monkeypatch.delenv("EMAIL_PROBE_PG_PORT", raising=False)
    else:
        monkeypatch.setenv("EMAIL_PROBE_PG_PORT", port)
    return runpy.run_path(str(script), run_name="email_provisioner_port_test")


def _manifest(port):
    return {
        "host": "127.0.0.1", "port": port, "role": "email_probe_012345abcd",
        "auth_db": "email_auth_012345abcd", "content_db": "email_content_012345abcd",
        "password": token_urlsafe(24),
    }


def test_cleanup_accepts_explicit_alternate_port(monkeypatch, tmp_path):
    provisioner = _load_provisioner(monkeypatch, tmp_path, "5435")
    provisioner["validate_manifest"](_manifest(5435))


@pytest.mark.parametrize("configured,manifest_port", [(None, 5435), ("5435", 5434), ("1", True)])
def test_cleanup_rejects_mismatched_or_boolean_port_before_connecting(monkeypatch, tmp_path, configured, manifest_port):
    provisioner = _load_provisioner(monkeypatch, tmp_path, configured)
    with pytest.raises(ValueError):
        provisioner["validate_manifest"](_manifest(manifest_port))


@pytest.mark.parametrize("port", ["", "true", "false", "5435.0", "invalid", "0", "65536", "-1"])
def test_provisioner_rejects_invalid_configuration_before_database_access(monkeypatch, tmp_path, port):
    with pytest.raises(ValueError):
        _load_provisioner(monkeypatch, tmp_path, port)


@pytest.mark.asyncio
async def test_setup_and_cleanup_use_matching_configured_port(monkeypatch, tmp_path):
    """The real provisioner writes the target and removes that validated manifest."""
    provisioner = _load_provisioner(monkeypatch, tmp_path, "5435")
    statements = []

    class LocalAdmin:
        async def execute(self, statement):
            statements.append(statement)

        async def fetchval(self, _statement, databases, role):
            assert databases == ["email_auth_012345abcd", "email_content_012345abcd"]
            assert role == "email_probe_012345abcd"
            return 0

        async def close(self):
            pass

    async def admin_connection():
        return LocalAdmin()

    namespace = provisioner["setup"].__globals__
    monkeypatch.setitem(namespace, "admin_connection", admin_connection)
    # Fix only the resource suffix, retaining the real random credential path.
    original_token = namespace["token_hex"]
    monkeypatch.setitem(namespace, "token_hex", lambda size: "012345abcd" if size == 5 else original_token(size))
    await provisioner["setup"]()
    path = tmp_path / "manifest.json"
    created = json.loads(path.read_text())
    assert created["port"] == 5435
    assert path.stat().st_mode & 0o077 == 0
    await provisioner["cleanup"]()
    assert not path.exists()
    assert statements[-3:] == [
        'DROP DATABASE IF EXISTS "email_auth_012345abcd" WITH (FORCE)',
        'DROP DATABASE IF EXISTS "email_content_012345abcd" WITH (FORCE)',
        'DROP ROLE IF EXISTS "email_probe_012345abcd"',
    ]


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
