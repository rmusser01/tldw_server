"""Integrity and diagnostics at the VN storage replay boundary."""

import hashlib
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.exceptions import StorageError
from tldw_Server_API.app.core.Storage import generated_file_helpers
from tldw_Server_API.app.core.Storage.file_integrity import generated_file_bytes_match
from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService

pytestmark = pytest.mark.unit


def test_file_integrity_propagates_transient_stat_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not use existence helpers that suppress I/O failures on Python 3.14."""
    def unavailable(_path: Path, **_kwargs: Any) -> Any:
        """Simulate an inaccessible filesystem independently of helper behavior."""
        raise PermissionError("temporary filesystem outage")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "is_file", lambda _path: False)
        patch.setattr(Path, "stat", unavailable)
        with pytest.raises(PermissionError, match="temporary filesystem outage"):
            generated_file_bytes_match(tmp_path / "asset.png", expected_size=8, checksum=None)


@pytest.mark.parametrize("checksum_kind,expected", [
    ("valid", True), ("upper", True), ("legacy", True), ("blank", True),
    ("wrong", False), ("malformed", False), ("empty", False),
])
def test_file_integrity_checksum_compatibility(
    tmp_path: Path, checksum_kind: str, expected: bool,
) -> None:
    """Validate checksums while preserving explicitly checksumless legacy records."""
    image = tmp_path / "image.png"
    content = b"recorded-image"
    image.write_bytes(b"" if checksum_kind == "empty" else content)
    digest = hashlib.sha256(content).hexdigest()
    checksum = {
        "valid": digest, "upper": digest.upper(), "legacy": None, "blank": "",
        "wrong": hashlib.sha256(b"wrong-image").hexdigest(), "malformed": "not-sha256", "empty": digest,
    }[checksum_kind]
    assert generated_file_bytes_match(image, expected_size=len(content), checksum=checksum) is expected


@pytest.mark.asyncio
async def test_storage_replay_rejects_same_length_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A size match must not override a persisted SHA-256 mismatch."""
    image = tmp_path / "asset.png"
    image.write_bytes(b"changed!")
    record = {"id": 7, "storage_path": "asset.png", "file_size_bytes": 8,
              "checksum": hashlib.sha256(b"original").hexdigest()}

    class Files:
        """Supply a persisted VN registration without an external database."""

        async def get_file_by_source_ref(self, **_kwargs: Any) -> dict[str, Any]:
            """Return the deliberately corrupted registration."""
            return record

    async def get_repo() -> Files:
        """Return the test repository."""
        return Files()

    service = StorageQuotaService()
    monkeypatch.setattr(service, "get_generated_files_repo", get_repo)
    monkeypatch.setattr(generated_file_helpers.DatabasePaths, "get_user_outputs_dir",
                        staticmethod(lambda _user_id: tmp_path))
    with pytest.raises(StorageError, match="missing or invalid bytes"):
        await service.get_vn_generated_file(user_id=1, source_ref="vn_asset_item:7")


@pytest.mark.asyncio
async def test_cleanup_failure_keeps_context_traceback_without_error_secrets(
    tmp_path: Path,
) -> None:
    """Best-effort cleanup records correlation and traceback, not error secrets."""
    entries: list[dict[str, Any]] = []

    class BrokenService:
        """Fail the lookup with a canary sensitive exception message."""

        async def get_generated_files_repo(self) -> None:
            """Simulate an unavailable database while cleanup is attempted."""
            secret = "secret-cleanup-canary"
            raise RuntimeError(secret)

    sink = logger.add(lambda message: entries.append(message.record), diagnose=False)
    try:
        await generated_file_helpers._cleanup_unregistered_vn_file(
            BrokenService(), user_id=1, source_ref="vn_asset_item:7",
            storage_path="asset.png", file_path=tmp_path / "asset.png",
        )
    finally:
        logger.remove(sink)
    entry = entries[-1]
    expected = {
        "operation": "cleanup_unregistered_vn_file", "user_id": 1,
        "source_ref": "vn_asset_item:7", "storage_path": "asset.png",
        "error_type": "RuntimeError",
    }
    assert {key: entry["extra"][key] for key in expected} == expected
    assert "Traceback (most recent call last)" in entry["message"]
    assert "get_generated_files_repo" in entry["message"]
    assert "secret-cleanup-canary" not in entry["message"]
