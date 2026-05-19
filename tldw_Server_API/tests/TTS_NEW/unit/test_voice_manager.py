"""
Unit tests for VoiceManager upload/delete lifecycle and duration handling.

These tests exercise VoiceManager without relying on external ffmpeg/ffprobe
executables by patching the internal duration/processing helpers.
"""

from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.TTS import voice_manager as voice_manager_module
from tldw_Server_API.app.core.TTS.voice_manager import (
    VoiceManager,
    VoiceUploadRequest,
    VoiceDurationError,
    VoiceInfo,
    PROVIDER_REQUIREMENTS,
    VOICE_RATE_LIMITS,
    VoiceReferenceMetadata,
)


@pytest.mark.asyncio
async def test_upload_and_delete_voice_happy_path(tmp_path, monkeypatch):
    """
    Upload a valid voice sample and then delete it, ensuring files and
    registry entries are cleaned up correctly for a provider profile.
    """
    manager = VoiceManager()

    # Ensure VoiceManager writes into a temporary user DB base dir
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    voices_root = tmp_path / "voices"

    def _fake_user_db_base_dir(*, allow_legacy_alias: bool = False):
        return tmp_path

    def _fake_user_voices_dir(user_id):
        voices_root.mkdir(parents=True, exist_ok=True)
        (voices_root / "uploads").mkdir(parents=True, exist_ok=True)
        (voices_root / "processed").mkdir(parents=True, exist_ok=True)
        (voices_root / "temp").mkdir(parents=True, exist_ok=True)
        return voices_root

    monkeypatch.setattr(DatabasePaths, "get_user_db_base_dir", _fake_user_db_base_dir, raising=True)
    monkeypatch.setattr(DatabasePaths, "get_user_voices_dir", _fake_user_voices_dir, raising=True)

    # Patch duration and processing helpers to be deterministic and fast
    async def fake_duration(path: Path) -> float:  # type: ignore[override]
        # Choose a duration within the recommended range for vibevoice
        reqs = PROVIDER_REQUIREMENTS.get("vibevoice", {})
        return float(reqs.get("duration", {}).get("min", 3.0)) + 0.5

    async def fake_process_for_provider(input_path: Path, output_path: Path, provider: str) -> Path:  # type: ignore[override]
        output_path = output_path.with_suffix(".wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(input_path.read_bytes())
        return output_path

    monkeypatch.setattr(manager, "_get_audio_duration", fake_duration, raising=False)
    monkeypatch.setattr(manager, "_process_for_provider", fake_process_for_provider, raising=False)

    # Build a small WAV-like payload; provider is vibevoice
    file_bytes = b"RIFF" + b"\x00" * 1000
    request = VoiceUploadRequest(name="test-voice", description="unit test", provider="vibevoice")

    # Upload
    resp = await manager.upload_voice(
        user_id=1,
        file_content=file_bytes,
        filename="sample.wav",
        request=request,
    )

    assert resp.voice_id
    assert resp.provider_compatible is True
    assert resp.warnings == []
    assert Path(resp.file_path).exists()

    # Delete and ensure files are removed
    deleted = await manager.delete_voice(user_id=1, voice_id=resp.voice_id)
    assert deleted

    voices_root = manager.get_user_voices_path(1)
    processed_dir = voices_root / "processed"
    uploads_dir = voices_root / "uploads"
    # No processed file with this voice_id should remain
    remaining: List[Path] = list(processed_dir.glob(f"{resp.voice_id}*")) if processed_dir.exists() else []
    assert not remaining
    leftovers: List[Path] = list(uploads_dir.glob(f"{resp.voice_id}_*")) if uploads_dir.exists() else []
    assert not leftovers


@pytest.mark.asyncio
async def test_upload_voice_short_duration_warning_and_strict_mode(tmp_path, monkeypatch):
    """
    When duration is outside provider requirements, uploads should:
    - succeed with warnings by default
    - raise VoiceDurationError when TTS_VOICE_STRICT_DURATION is true.
    """
    manager = VoiceManager()

    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    voices_root = tmp_path / "voices"

    def _fake_user_db_base_dir(*, allow_legacy_alias: bool = False):
        return tmp_path

    def _fake_user_voices_dir(user_id):
        voices_root.mkdir(parents=True, exist_ok=True)
        (voices_root / "uploads").mkdir(parents=True, exist_ok=True)
        (voices_root / "processed").mkdir(parents=True, exist_ok=True)
        (voices_root / "temp").mkdir(parents=True, exist_ok=True)
        return voices_root

    monkeypatch.setattr(DatabasePaths, "get_user_db_base_dir", _fake_user_db_base_dir, raising=True)
    monkeypatch.setattr(DatabasePaths, "get_user_voices_dir", _fake_user_voices_dir, raising=True)

    # Force a duration that is too short for higgs
    async def fake_short_duration(path: Path) -> float:  # type: ignore[override]
        return 0.5

    async def fake_process_for_provider(input_path: Path, output_path: Path, provider: str) -> Path:  # type: ignore[override]
        output_path = output_path.with_suffix(".wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(input_path.read_bytes())
        return output_path

    monkeypatch.setattr(manager, "_get_audio_duration", fake_short_duration, raising=False)
    monkeypatch.setattr(manager, "_process_for_provider", fake_process_for_provider, raising=False)

    mock_storage = AsyncMock()
    mock_storage.register_generated_file = AsyncMock(return_value={"id": 1})

    async def _get_storage_service():
        return mock_storage

    monkeypatch.setattr(voice_manager_module, "get_storage_service", _get_storage_service)

    file_bytes = b"RIFF" + b"\x00" * 1000
    request = VoiceUploadRequest(name="short-voice", description=None, provider="higgs")

    # Non-strict mode (default: env not set) -> accepted with warning
    monkeypatch.delenv("TTS_VOICE_STRICT_DURATION", raising=False)
    resp = await manager.upload_voice(
        user_id=42,
        file_content=file_bytes,
        filename="short.wav",
        request=request,
    )
    assert resp.provider_compatible is False
    assert any("less than recommended" in w for w in resp.warnings)

    # Strict mode -> raises VoiceDurationError
    # Clean up quota tracking for a fresh upload
    manager.user_upload_counts.clear()
    monkeypatch.setenv("TTS_VOICE_STRICT_DURATION", "true")

    with pytest.raises(VoiceDurationError):
        await manager.upload_voice(
            user_id=42,
            file_content=file_bytes,
            filename="short_again.wav",
            request=request,
        )


@pytest.mark.asyncio
async def test_delete_voice_rejects_path_traversal(tmp_path, monkeypatch):
    """Delete should refuse to remove files outside the voices directory."""
    manager = VoiceManager()

    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    voices_root = tmp_path / "voices"

    def _fake_user_db_base_dir(*, allow_legacy_alias: bool = False):
        return tmp_path

    def _fake_user_voices_dir(user_id):
        voices_root.mkdir(parents=True, exist_ok=True)
        (voices_root / "uploads").mkdir(parents=True, exist_ok=True)
        (voices_root / "processed").mkdir(parents=True, exist_ok=True)
        (voices_root / "temp").mkdir(parents=True, exist_ok=True)
        return voices_root

    monkeypatch.setattr(DatabasePaths, "get_user_db_base_dir", _fake_user_db_base_dir, raising=True)
    monkeypatch.setattr(DatabasePaths, "get_user_voices_dir", _fake_user_voices_dir, raising=True)

    outside_file = tmp_path / "outside.wav"
    outside_file.write_bytes(b"not a voice sample")

    voice_info = VoiceInfo(
        voice_id="malicious-id",
        name="malicious",
        description=None,
        file_path="../outside.wav",
        format="wav",
        duration=1.0,
        sample_rate=None,
        size_bytes=outside_file.stat().st_size,
        provider="vibevoice",
        created_at=datetime.utcnow(),
        file_hash="",
    )

    await manager.registry.register_voice(user_id=7, voice_info=voice_info)

    deleted = await manager.delete_voice(user_id=7, voice_id="malicious-id")
    assert deleted is False
    assert outside_file.exists()


@pytest.mark.asyncio
async def test_reference_metadata_round_trip(tmp_path, monkeypatch):
    manager = VoiceManager()

    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    voices_root = tmp_path / "voices"

    def _fake_user_db_base_dir(*, allow_legacy_alias: bool = False):
        return tmp_path

    def _fake_user_voices_dir(user_id):
        voices_root.mkdir(parents=True, exist_ok=True)
        (voices_root / "uploads").mkdir(parents=True, exist_ok=True)
        (voices_root / "processed").mkdir(parents=True, exist_ok=True)
        (voices_root / "temp").mkdir(parents=True, exist_ok=True)
        return voices_root

    monkeypatch.setattr(DatabasePaths, "get_user_db_base_dir", _fake_user_db_base_dir, raising=True)
    monkeypatch.setattr(DatabasePaths, "get_user_voices_dir", _fake_user_voices_dir, raising=True)

    metadata = VoiceReferenceMetadata(
        voice_id="voice-123",
        reference_text="Reference text",
        provider_artifacts={"neutts": {"ref_codes": [1, 2, 3]}},
    )
    await manager.save_reference_metadata(user_id=1, metadata=metadata)

    loaded = await manager.load_reference_metadata(user_id=1, voice_id="voice-123")
    assert loaded is not None
    assert loaded.voice_id == metadata.voice_id
    assert loaded.reference_text == metadata.reference_text
    assert loaded.provider_artifacts["neutts"]["ref_codes"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_encode_voice_reference_stores_artifacts(tmp_path, monkeypatch):
    manager = VoiceManager()

    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    voices_root = tmp_path / "voices"

    def _fake_user_db_base_dir(*, allow_legacy_alias: bool = False):
        return tmp_path

    def _fake_user_voices_dir(user_id):
        voices_root.mkdir(parents=True, exist_ok=True)
        (voices_root / "uploads").mkdir(parents=True, exist_ok=True)
        (voices_root / "processed").mkdir(parents=True, exist_ok=True)
        (voices_root / "temp").mkdir(parents=True, exist_ok=True)
        return voices_root

    monkeypatch.setattr(DatabasePaths, "get_user_db_base_dir", _fake_user_db_base_dir, raising=True)
    monkeypatch.setattr(DatabasePaths, "get_user_voices_dir", _fake_user_voices_dir, raising=True)

    voice_id = "voice-encode"
    processed_path = voices_root / "processed" / f"{voice_id}.wav"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.write_bytes(b"RIFF" + b"\x00" * 1000)

    voice_info = VoiceInfo(
        voice_id=voice_id,
        name="encode-voice",
        description=None,
        file_path=str(processed_path.relative_to(voices_root)),
        format="wav",
        duration=3.5,
        sample_rate=16000,
        size_bytes=processed_path.stat().st_size,
        provider="neutts",
        created_at=datetime.utcnow(),
        file_hash="",
    )

    await manager.registry.register_voice(user_id=1, voice_info=voice_info)

    async def _fake_encode(audio_path: Path) -> List[int]:
        assert audio_path == processed_path
        return [10, 20, 30]

    monkeypatch.setattr(manager, "_encode_neutts_reference", _fake_encode, raising=False)

    result = await manager.encode_voice_reference(
        user_id=1,
        voice_id=voice_id,
        provider="neutts",
        reference_text="Hello there",
    )
    assert result.cached is False
    assert result.ref_codes_len == 3
    assert result.reference_text == "Hello there"

    cached = await manager.encode_voice_reference(
        user_id=1,
        voice_id=voice_id,
        provider="neutts",
    )
    assert cached.cached is True
    assert cached.ref_codes_len == 3


@pytest.mark.asyncio
async def test_encode_voice_reference_for_omnivoice_is_metadata_only(tmp_path, monkeypatch):
    manager = VoiceManager()

    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    voices_root = tmp_path / "voices"

    def _fake_user_db_base_dir(*, allow_legacy_alias: bool = False):
        return tmp_path

    def _fake_user_voices_dir(user_id):
        voices_root.mkdir(parents=True, exist_ok=True)
        (voices_root / "uploads").mkdir(parents=True, exist_ok=True)
        (voices_root / "processed").mkdir(parents=True, exist_ok=True)
        (voices_root / "temp").mkdir(parents=True, exist_ok=True)
        (voices_root / "metadata").mkdir(parents=True, exist_ok=True)
        return voices_root

    monkeypatch.setattr(DatabasePaths, "get_user_db_base_dir", _fake_user_db_base_dir, raising=True)
    monkeypatch.setattr(DatabasePaths, "get_user_voices_dir", _fake_user_voices_dir, raising=True)

    voice_id = "voice-omnivoice"
    processed_path = voices_root / "processed" / f"{voice_id}.wav"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.write_bytes(b"RIFF" + b"\x00" * 1000)

    voice_info = VoiceInfo(
        voice_id=voice_id,
        name="omnivoice-voice",
        description=None,
        file_path=str(processed_path.relative_to(voices_root)),
        format="wav",
        duration=4.0,
        sample_rate=24000,
        size_bytes=processed_path.stat().st_size,
        provider="omnivoice",
        created_at=datetime.utcnow(),
        file_hash="",
    )

    await manager.registry.register_voice(user_id=1, voice_info=voice_info)

    result = await manager.encode_voice_reference(
        user_id=1,
        voice_id=voice_id,
        provider="omnivoice",
        reference_text="Stored transcript",
    )

    assert result.cached is False
    assert result.ref_codes_len is None
    assert result.reference_text == "Stored transcript"

    metadata = await manager.load_reference_metadata(user_id=1, voice_id=voice_id)
    assert metadata is not None
    assert metadata.provider_artifacts["omnivoice"]["reference_text"] == "Stored transcript"

    cached = await manager.encode_voice_reference(
        user_id=1,
        voice_id=voice_id,
        provider="omnivoice",
    )

    assert cached.cached is True
    assert cached.ref_codes_len is None
    assert cached.reference_text == "Stored transcript"


@pytest.mark.asyncio
async def test_list_user_voices_syncs_after_external_filesystem_changes(tmp_path, monkeypatch):
    """Registry views should refresh when processed files are added/removed externally."""
    manager = VoiceManager()

    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    voices_root = tmp_path / "voices"

    def _fake_user_db_base_dir(*, allow_legacy_alias: bool = False):
        return tmp_path

    def _fake_user_voices_dir(user_id):
        voices_root.mkdir(parents=True, exist_ok=True)
        (voices_root / "uploads").mkdir(parents=True, exist_ok=True)
        (voices_root / "processed").mkdir(parents=True, exist_ok=True)
        (voices_root / "temp").mkdir(parents=True, exist_ok=True)
        return voices_root

    monkeypatch.setattr(DatabasePaths, "get_user_db_base_dir", _fake_user_db_base_dir, raising=True)
    monkeypatch.setattr(DatabasePaths, "get_user_voices_dir", _fake_user_voices_dir, raising=True)

    async def _no_default_voice(user_id: int):
        return None

    async def _fake_duration(path: Path) -> float:  # type: ignore[override]
        return 1.25

    monkeypatch.setattr(manager, "ensure_default_voice", _no_default_voice, raising=False)
    monkeypatch.setattr(manager, "_get_audio_duration", _fake_duration, raising=False)

    before = await manager.list_user_voices(user_id=5)
    assert before == []

    external_file = voices_root / "processed" / "external-voice.wav"
    external_file.write_bytes(b"RIFF" + b"\x00" * 128)

    after_add = await manager.list_user_voices(user_id=5)
    assert any(v.voice_id == "external-voice" for v in after_add)

    external_file.unlink()
    after_delete = await manager.list_user_voices(user_id=5)
    assert all(v.voice_id != "external-voice" for v in after_delete)


@pytest.mark.asyncio
async def test_delete_voice_recovers_from_cold_registry_via_filesystem_sync(tmp_path, monkeypatch):
    """Delete should succeed even when the current process has no in-memory voice entry."""
    manager = VoiceManager()

    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    voices_root = tmp_path / "voices"

    def _fake_user_db_base_dir(*, allow_legacy_alias: bool = False):
        return tmp_path

    def _fake_user_voices_dir(user_id):
        voices_root.mkdir(parents=True, exist_ok=True)
        (voices_root / "uploads").mkdir(parents=True, exist_ok=True)
        (voices_root / "processed").mkdir(parents=True, exist_ok=True)
        (voices_root / "temp").mkdir(parents=True, exist_ok=True)
        return voices_root

    monkeypatch.setattr(DatabasePaths, "get_user_db_base_dir", _fake_user_db_base_dir, raising=True)
    monkeypatch.setattr(DatabasePaths, "get_user_voices_dir", _fake_user_voices_dir, raising=True)

    async def _no_default_voice(user_id: int):
        return None

    async def _fake_duration(path: Path) -> float:  # type: ignore[override]
        return 3.0

    monkeypatch.setattr(manager, "ensure_default_voice", _no_default_voice, raising=False)
    monkeypatch.setattr(manager, "_get_audio_duration", _fake_duration, raising=False)

    voice_id = "cold-registry-voice"
    processed_file = voices_root / "processed" / f"{voice_id}.wav"
    upload_file = voices_root / "uploads" / f"{voice_id}_original.wav"
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    upload_file.parent.mkdir(parents=True, exist_ok=True)
    processed_file.write_bytes(b"RIFF" + b"\x00" * 128)
    upload_file.write_bytes(b"RIFF" + b"\x00" * 64)

    deleted = await manager.delete_voice(user_id=7, voice_id=voice_id)
    assert deleted is True
    assert not processed_file.exists()
    assert not upload_file.exists()


@pytest.mark.asyncio
async def test_check_rate_limits_uses_filesystem_voice_count(tmp_path, monkeypatch):
    """Max-voices limit should be enforced from processed files, not just in-memory registry."""
    manager = VoiceManager()

    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    voices_root = tmp_path / "voices"

    def _fake_user_db_base_dir(*, allow_legacy_alias: bool = False):
        return tmp_path

    def _fake_user_voices_dir(user_id):
        voices_root.mkdir(parents=True, exist_ok=True)
        (voices_root / "uploads").mkdir(parents=True, exist_ok=True)
        (voices_root / "processed").mkdir(parents=True, exist_ok=True)
        (voices_root / "temp").mkdir(parents=True, exist_ok=True)
        return voices_root

    monkeypatch.setattr(DatabasePaths, "get_user_db_base_dir", _fake_user_db_base_dir, raising=True)
    monkeypatch.setattr(DatabasePaths, "get_user_voices_dir", _fake_user_voices_dir, raising=True)

    processed_file = voices_root / "processed" / "voice-a.wav"
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    processed_file.write_bytes(b"RIFF" + b"\x00" * 16)
    monkeypatch.setitem(VOICE_RATE_LIMITS, "max_voices_per_user", 1)

    ok, msg = await manager.check_rate_limits(user_id=12)
    assert ok is False
    assert "Maximum voice limit reached" in msg


@pytest.mark.asyncio
async def test_background_task_lifecycle_idempotent():
    """Starting/stopping cleanup worker repeatedly should be safe."""
    manager = VoiceManager()
    manager.cleanup_interval = 60

    await manager.start_background_tasks()
    first_task = manager._cleanup_task
    assert first_task is not None

    await manager.start_background_tasks()
    assert manager._cleanup_task is first_task

    await manager.stop_background_tasks()
    assert manager._cleanup_task is None

    # Second stop should be a no-op
    await manager.stop_background_tasks()


@pytest.mark.asyncio
async def test_persistent_registry_shared_across_manager_instances(tmp_path, monkeypatch):
    """Voices uploaded by one manager should resolve from DB in another manager instance."""
    manager_a = VoiceManager()
    manager_b = VoiceManager()

    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    voices_root = tmp_path / "voices"

    def _fake_user_db_base_dir(*, allow_legacy_alias: bool = False):
        return tmp_path

    def _fake_user_voices_dir(user_id):
        voices_root.mkdir(parents=True, exist_ok=True)
        (voices_root / "uploads").mkdir(parents=True, exist_ok=True)
        (voices_root / "processed").mkdir(parents=True, exist_ok=True)
        (voices_root / "temp").mkdir(parents=True, exist_ok=True)
        (voices_root / "metadata").mkdir(parents=True, exist_ok=True)
        return voices_root

    monkeypatch.setattr(DatabasePaths, "get_user_db_base_dir", _fake_user_db_base_dir, raising=True)
    monkeypatch.setattr(DatabasePaths, "get_user_voices_dir", _fake_user_voices_dir, raising=True)

    async def _fake_duration(path: Path) -> float:  # type: ignore[override]
        return 4.2

    async def _fake_process_for_provider(input_path: Path, output_path: Path, provider: str) -> Path:  # type: ignore[override]
        output_path = output_path.with_suffix(".wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(input_path.read_bytes())
        return output_path

    monkeypatch.setattr(manager_a, "_get_audio_duration", _fake_duration, raising=False)
    monkeypatch.setattr(manager_a, "_process_for_provider", _fake_process_for_provider, raising=False)
    monkeypatch.setattr(manager_b, "_get_audio_duration", _fake_duration, raising=False)

    mock_storage = AsyncMock()
    mock_storage.register_generated_file = AsyncMock(return_value={"id": 1})

    async def _get_storage_service():
        return mock_storage

    monkeypatch.setattr(voice_manager_module, "get_storage_service", _get_storage_service)

    request = VoiceUploadRequest(name="persistent-voice", description="db-backed", provider="vibevoice")
    upload = await manager_a.upload_voice(
        user_id=77,
        file_content=b"RIFF" + b"\x00" * 1024,
        filename="persistent.wav",
        request=request,
    )

    # Simulate warm snapshot state so manager_b should read from the persistent DB
    # without forcing a filesystem scan.
    manager_b._registry_snapshots[77] = manager_b._get_processed_snapshot(77)

    async def _fail_scan(_user_id: int):
        raise AssertionError("filesystem scan should not run")

    monkeypatch.setattr(manager_b, "_scan_user_voices", _fail_scan, raising=False)

    resolved = await manager_b.get_voice(user_id=77, voice_id=upload.voice_id)
    assert resolved is not None
    assert resolved.voice_id == upload.voice_id
    assert resolved.name == "persistent-voice"


@pytest.mark.asyncio
async def test_sync_prunes_stale_persisted_voice_records(tmp_path, monkeypatch):
    """A persisted entry whose file disappears should be removed during sync."""
    manager = VoiceManager()

    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    voices_root = tmp_path / "voices"

    def _fake_user_db_base_dir(*, allow_legacy_alias: bool = False):
        return tmp_path

    def _fake_user_voices_dir(user_id):
        voices_root.mkdir(parents=True, exist_ok=True)
        (voices_root / "uploads").mkdir(parents=True, exist_ok=True)
        (voices_root / "processed").mkdir(parents=True, exist_ok=True)
        (voices_root / "temp").mkdir(parents=True, exist_ok=True)
        (voices_root / "metadata").mkdir(parents=True, exist_ok=True)
        return voices_root

    monkeypatch.setattr(DatabasePaths, "get_user_db_base_dir", _fake_user_db_base_dir, raising=True)
    monkeypatch.setattr(DatabasePaths, "get_user_voices_dir", _fake_user_voices_dir, raising=True)

    async def _no_default_voice(user_id: int):
        return None

    monkeypatch.setattr(manager, "ensure_default_voice", _no_default_voice, raising=False)

    stale_voice = VoiceInfo(
        voice_id="stale",
        name="stale",
        description=None,
        file_path="processed/stale.wav",
        format="wav",
        duration=2.0,
        sample_rate=22050,
        size_bytes=128,
        provider="vibevoice",
        created_at=datetime.utcnow(),
        file_hash="hash",
    )
    await manager._upsert_persisted_voice(user_id=55, voice=stale_voice)
    manager._registry_snapshots[55] = manager._get_processed_snapshot(55)

    voices = await manager.list_user_voices(user_id=55)
    assert voices == []
    assert await manager._get_persisted_voice(55, "stale") is None


@pytest.mark.asyncio
async def test_voice_registry_persistence_can_be_disabled(tmp_path, monkeypatch):
    """When TTS_VOICE_REGISTRY_ENABLED=false, voice listing should stay runtime/filesystem only."""
    monkeypatch.setenv("TTS_VOICE_REGISTRY_ENABLED", "false")
    manager = VoiceManager()

    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    voices_root = tmp_path / "voices"

    def _fake_user_db_base_dir(*, allow_legacy_alias: bool = False):
        return tmp_path

    def _fake_user_voices_dir(user_id):
        voices_root.mkdir(parents=True, exist_ok=True)
        (voices_root / "uploads").mkdir(parents=True, exist_ok=True)
        (voices_root / "processed").mkdir(parents=True, exist_ok=True)
        (voices_root / "temp").mkdir(parents=True, exist_ok=True)
        (voices_root / "metadata").mkdir(parents=True, exist_ok=True)
        return voices_root

    monkeypatch.setattr(DatabasePaths, "get_user_db_base_dir", _fake_user_db_base_dir, raising=True)
    monkeypatch.setattr(DatabasePaths, "get_user_voices_dir", _fake_user_voices_dir, raising=True)

    async def _no_default_voice(user_id: int):
        return None

    async def _fake_duration(path: Path) -> float:  # type: ignore[override]
        return 1.0

    monkeypatch.setattr(manager, "ensure_default_voice", _no_default_voice, raising=False)
    monkeypatch.setattr(manager, "_get_audio_duration", _fake_duration, raising=False)

    processed = voices_root / "processed" / "runtime-only.wav"
    processed.parent.mkdir(parents=True, exist_ok=True)
    processed.write_bytes(b"RIFF" + b"\x00" * 64)

    listed = await manager.list_user_voices(user_id=99)
    assert any(v.voice_id == "runtime-only" for v in listed)
    assert not manager.get_user_voice_registry_db_path(99).exists()
