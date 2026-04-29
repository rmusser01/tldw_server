from __future__ import annotations

import builtins
import json
import zipfile
from typing import Any


class _LoggerStub:
    def __init__(self, *, records: list[dict[str, Any]] | None = None, extra: dict[str, Any] | None = None) -> None:
        self.records = records if records is not None else []
        self.extra = extra or {}

    def bind(self, **kwargs: Any) -> "_LoggerStub":
        return _LoggerStub(records=self.records, extra={**self.extra, **kwargs})

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.records.append(
            {
                "message": message,
                "args": args,
                "kwargs": kwargs,
                "extra": self.extra,
            }
        )


def test_app_version_fallback_logs_do_not_include_raw_exceptions(monkeypatch):
    from importlib import metadata

    from tldw_Server_API.app.services import admin_bundle_service as service

    secret_path = "/private/admin/tokened-pyproject.toml"
    secret_token = "bundle-secret-token"

    def _raise_metadata_version(_package_name: str) -> str:
        raise RuntimeError(f"metadata failed at {secret_path} with {secret_token}")

    def _raise_pyproject_open(*_args, **_kwargs):
        raise RuntimeError(f"pyproject read failed at {secret_path} with {secret_token}")

    monkeypatch.setattr(metadata, "version", _raise_metadata_version)
    monkeypatch.setattr(service.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(builtins, "open", _raise_pyproject_open)
    logger_stub = _LoggerStub()
    monkeypatch.setattr(service, "logger", logger_stub)

    assert service._get_app_version() is None

    log_output = "\n".join(record["message"] for record in logger_stub.records)
    assert "Failed to resolve app version from package metadata" in log_output
    assert "Failed to resolve app version from pyproject" in log_output
    assert secret_path not in log_output
    assert secret_token not in log_output
    assert "metadata failed" not in log_output
    assert "pyproject read failed" not in log_output
    assert [record["extra"] for record in logger_stub.records] == [
        {"error_type": "RuntimeError"},
        {"error_type": "RuntimeError"},
    ]
    assert all(record["args"] == () for record in logger_stub.records)
    assert all("exc_info" not in record["kwargs"] for record in logger_stub.records)


def test_sidecar_manifest_fallback_log_does_not_include_raw_exception(
    monkeypatch,
    tmp_path,
):
    from tldw_Server_API.app.services import admin_bundle_service as service

    secret_path = "/private/admin/tokened-sidecar.manifest.json"
    secret_token = "bundle-sidecar-secret"
    zip_path = tmp_path / "bundle.zip"
    sidecar_path = str(zip_path) + ".manifest.json"
    manifest = {
        "manifest_version": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "user_id": None,
        "datasets": ["authnz"],
        "files": {},
        "schema_versions": {},
    }

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
    tmp_path.joinpath("bundle.zip.manifest.json").write_text("{}", encoding="utf-8")

    original_open = builtins.open

    def _raise_sidecar_open(path, *args, **kwargs):
        if str(path) == sidecar_path:
            raise RuntimeError(f"sidecar failed at {secret_path} with {secret_token}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _raise_sidecar_open)
    logger_stub = _LoggerStub()
    monkeypatch.setattr(service, "logger", logger_stub)

    assert service._read_manifest_cached(str(zip_path)) == manifest

    log_output = "\n".join(record["message"] for record in logger_stub.records)
    assert "Failed to read bundle sidecar manifest; falling back to ZIP" in log_output
    assert secret_path not in log_output
    assert secret_token not in log_output
    assert "sidecar failed" not in log_output
    assert [record["extra"] for record in logger_stub.records] == [
        {"error_type": "RuntimeError"},
    ]
    assert all(record["args"] == () for record in logger_stub.records)
    assert all("exc_info" not in record["kwargs"] for record in logger_stub.records)
