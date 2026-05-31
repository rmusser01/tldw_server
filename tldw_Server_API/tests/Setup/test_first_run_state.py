import json
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Setup.first_run_state import (
    REQUIRED_FIRST_RUN_STEPS,
    FirstRunStateStore,
    FirstRunStatus,
    InvalidFirstRunTransition,
)


def _persist_required_first_run_step_data(
    store: FirstRunStateStore,
    *,
    acknowledged: bool = True,
) -> None:
    step_payloads = {
        "setup_path": {"acknowledged": acknowledged, "setup_path_key": "local"},
        "privacy_security": {"acknowledged": acknowledged, "local_only": True},
        "providers": {"acknowledged": acknowledged, "default_provider": "openai"},
        "ingest_defaults": {
            "acknowledged": acknowledged,
            "allow_local_file_ingest": False,
            "chunking_profile": "balanced",
            "metadata_mode": "automatic",
        },
        "audio_defaults": {"acknowledged": acknowledged, "mode": "skip"},
        "optional_advanced": {
            "acknowledged": acknowledged,
            "rag": "defer",
            "storage_paths": "defer",
        },
    }
    for step, payload in step_payloads.items():
        store.update_step(step, payload)


def test_new_store_defaults_to_not_started(tmp_path: Path):
    store = FirstRunStateStore(tmp_path / "first_run_state.json")

    state = store.load()

    assert state.status == FirstRunStatus.NOT_STARTED
    assert state.completed_at is None
    assert state.first_chat.completed is False


def test_records_step_and_persists_across_store_instances(tmp_path: Path):
    path = tmp_path / "first_run_state.json"
    store = FirstRunStateStore(path)

    store.update_step("providers", {"default_provider": "openai"})

    reloaded = FirstRunStateStore(path).load()
    assert reloaded.status == FirstRunStatus.IN_PROGRESS
    assert reloaded.current_step == "providers"
    assert reloaded.step_data["providers"]["default_provider"] == "openai"


def test_step_data_secrets_are_redacted_before_persistence(tmp_path: Path):
    path = tmp_path / "first_run_state.json"
    store = FirstRunStateStore(path)

    store.update_step(
        "providers",
        {
            "api_key": "sk-secret-raw",
            "private_key": "private-key-raw",
            "access_key": "access-key-raw",
            "nested": {"refresh_token": "tok-raw", "auth_key": "auth-key-raw"},
            "default_provider": "openai",
            "acknowledged": True,
        },
    )

    state = FirstRunStateStore(path).load()
    serialized_step_data = json.dumps(state.step_data)
    assert "sk-secret-raw" not in serialized_step_data
    assert "tok-raw" not in serialized_step_data
    assert "private-key-raw" not in serialized_step_data
    assert "access-key-raw" not in serialized_step_data
    assert "auth-key-raw" not in serialized_step_data
    assert state.step_data["providers"]["api_key"] == "********"
    assert state.step_data["providers"]["private_key"] == "********"
    assert state.step_data["providers"]["access_key"] == "********"
    assert state.step_data["providers"]["nested"]["refresh_token"] == "********"
    assert state.step_data["providers"]["nested"]["auth_key"] == "********"
    assert state.step_data["providers"]["default_provider"] == "openai"
    assert state.step_data["providers"]["acknowledged"] is True


def test_complete_requires_first_chat_success(tmp_path: Path):
    store = FirstRunStateStore(tmp_path / "first_run_state.json")

    with pytest.raises(InvalidFirstRunTransition):
        store.mark_completed()


def test_complete_requires_required_step_acknowledgements(tmp_path: Path):
    store = FirstRunStateStore(tmp_path / "first_run_state.json")

    _persist_required_first_run_step_data(store, acknowledged=False)
    store.record_first_chat_success(
        provider="openai",
        model="gpt-4.1-mini",
        response_id="chatcmpl-test",
    )

    with pytest.raises(InvalidFirstRunTransition) as excinfo:
        store.mark_completed()

    assert "required_steps_missing" in str(excinfo.value)


def test_later_unacknowledged_step_blocks_completion(tmp_path: Path):
    store = FirstRunStateStore(tmp_path / "first_run_state.json")

    _persist_required_first_run_step_data(store)
    store.update_step("providers", {"acknowledged": False, "default_provider": "openai"})
    store.record_first_chat_success(
        provider="openai",
        model="gpt-4.1-mini",
        response_id="chatcmpl-test",
    )

    with pytest.raises(InvalidFirstRunTransition) as excinfo:
        store.mark_completed()

    assert "required_steps_missing:providers" in str(excinfo.value)


def test_required_step_completion_is_revoked_when_acknowledgement_is_removed(tmp_path: Path):
    store = FirstRunStateStore(tmp_path / "first_run_state.json")

    store.update_step("providers", {"acknowledged": True})
    state = store.update_step("providers", {"acknowledged": False})

    assert "providers" not in state.acknowledged_steps
    assert "providers" not in state.completed_steps


def test_required_step_without_data_revokes_acknowledgement_for_completion(tmp_path: Path):
    store = FirstRunStateStore(tmp_path / "first_run_state.json")

    _persist_required_first_run_step_data(store)
    state = store.update_step("providers")
    store.record_first_chat_success(
        provider="openai",
        model="gpt-4.1-mini",
        response_id="chatcmpl-test",
    )

    assert "providers" not in state.acknowledged_steps
    assert "providers" not in state.completed_steps
    with pytest.raises(InvalidFirstRunTransition) as excinfo:
        store.mark_completed()

    assert "required_steps_missing:providers" in str(excinfo.value)


def test_acknowledgement_without_required_step_data_blocks_completion(tmp_path: Path):
    store = FirstRunStateStore(tmp_path / "first_run_state.json")

    for step in REQUIRED_FIRST_RUN_STEPS:
        store.update_step(step, {"acknowledged": True})
    store.record_first_chat_success(
        provider="openai",
        model="gpt-4.1-mini",
        response_id="chatcmpl-test",
    )

    with pytest.raises(InvalidFirstRunTransition) as excinfo:
        store.mark_completed()

    assert "required_steps_missing" in str(excinfo.value)


def test_blocked_state_rejects_normal_mutations(tmp_path: Path):
    path = tmp_path / "first_run_state.json"
    path.write_text("{invalid-json", encoding="utf-8")
    store = FirstRunStateStore(path)
    assert store.load().status == FirstRunStatus.BLOCKED

    with pytest.raises(InvalidFirstRunTransition, match="state_blocked"):
        store.update_step("providers", {"acknowledged": True})
    with pytest.raises(InvalidFirstRunTransition, match="state_blocked"):
        store.record_first_chat_success(
            provider="openai",
            model="gpt-4.1-mini",
            response_id="chatcmpl-test",
        )
    with pytest.raises(InvalidFirstRunTransition, match="state_blocked"):
        store.mark_completed()
    with pytest.raises(InvalidFirstRunTransition, match="state_blocked"):
        store.mark_skipped(reason="user_skip")


def test_skipped_state_rejects_normal_mutations_but_skip_is_idempotent(tmp_path: Path):
    store = FirstRunStateStore(tmp_path / "first_run_state.json")
    store.mark_skipped(reason="user_skip")

    with pytest.raises(InvalidFirstRunTransition, match="state_skipped"):
        store.update_step("providers", {"acknowledged": True})
    with pytest.raises(InvalidFirstRunTransition, match="state_skipped"):
        store.record_first_chat_success(
            provider="openai",
            model="gpt-4.1-mini",
            response_id="chatcmpl-test",
        )
    with pytest.raises(InvalidFirstRunTransition, match="state_skipped"):
        store.mark_completed()

    state = store.mark_skipped(reason="ignored_later_reason")
    assert state.status == FirstRunStatus.SKIPPED
    assert state.skip_reason == "user_skip"


def test_completed_state_rejects_normal_mutations(tmp_path: Path):
    store = FirstRunStateStore(tmp_path / "first_run_state.json")

    _persist_required_first_run_step_data(store)
    store.record_first_chat_success(
        provider="openai",
        model="gpt-4.1-mini",
        response_id="chatcmpl-test",
    )
    store.mark_completed()

    with pytest.raises(InvalidFirstRunTransition, match="setup_already_completed"):
        store.update_step("providers", {"acknowledged": False})
    with pytest.raises(InvalidFirstRunTransition, match="setup_already_completed"):
        store.record_first_chat_success(
            provider="openai",
            model="gpt-4.1-mini",
            response_id="chatcmpl-test-2",
        )
    with pytest.raises(InvalidFirstRunTransition, match="setup_already_completed"):
        store.mark_completed()
    with pytest.raises(InvalidFirstRunTransition, match="setup_already_completed"):
        store.mark_skipped(reason="user_skip")


def test_mutating_methods_use_state_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    @contextmanager
    def capture_lock(self):
        calls.append(str(self.path))
        yield

    monkeypatch.setattr(FirstRunStateStore, "_state_lock", capture_lock, raising=False)
    store = FirstRunStateStore(tmp_path / "first_run_state.json")

    store.update_step("providers", {"default_provider": "openai"})

    assert calls == [str(tmp_path / "first_run_state.json")]


def test_stale_lock_file_is_recovered_before_mutation(tmp_path: Path):
    path = tmp_path / "first_run_state.json"
    lock_path = path.with_name(f"{path.name}.lock")
    old_timestamp = datetime.now(timezone.utc) - timedelta(hours=1)
    lock_path.write_text(
        json.dumps({"pid": 999999, "created_at": old_timestamp.isoformat()}),
        encoding="utf-8",
    )
    store = FirstRunStateStore(path, lock_timeout_seconds=0.1, stale_lock_seconds=0.01)

    state = store.update_step("providers", {"default_provider": "openai"})

    assert state.step_data["providers"]["default_provider"] == "openai"
    assert not lock_path.exists()


def test_stale_empty_lock_file_is_recovered_before_mutation(tmp_path: Path):
    path = tmp_path / "first_run_state.json"
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.touch()
    old_timestamp = (datetime.now(timezone.utc) - timedelta(hours=1)).timestamp()
    os.utime(lock_path, (old_timestamp, old_timestamp))
    store = FirstRunStateStore(path, lock_timeout_seconds=0.01, stale_lock_seconds=0.001)

    state = store.update_step("providers", {"default_provider": "openai"})

    assert state.step_data["providers"]["default_provider"] == "openai"
    assert not lock_path.exists()


def test_stale_malformed_lock_file_is_recovered_before_mutation(tmp_path: Path):
    path = tmp_path / "first_run_state.json"
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.write_text("not-json-not-pid", encoding="utf-8")
    old_timestamp = (datetime.now(timezone.utc) - timedelta(hours=1)).timestamp()
    os.utime(lock_path, (old_timestamp, old_timestamp))
    store = FirstRunStateStore(path, lock_timeout_seconds=0.01, stale_lock_seconds=0.001)

    state = store.update_step("providers", {"default_provider": "openai"})

    assert state.step_data["providers"]["default_provider"] == "openai"
    assert not lock_path.exists()


def test_corrupt_state_file_is_quarantined_and_loads_blocked_recovery_state(tmp_path: Path):
    path = tmp_path / "first_run_state.json"
    path.write_text("{invalid-json", encoding="utf-8")
    store = FirstRunStateStore(path)

    state = store.load()

    assert state.status == FirstRunStatus.BLOCKED
    assert state.current_step == "state_recovery"
    assert state.skip_reason == "state_file_recovery"
    assert state.step_data["state_recovery"]["reason"] == "invalid_json"
    assert state.step_data["state_recovery"]["quarantined"] is True
    assert path.exists()
    assert list(tmp_path.glob("first_run_state.json.corrupt-*"))


def test_invalid_state_schema_is_quarantined_and_loads_blocked_recovery_state(tmp_path: Path):
    path = tmp_path / "first_run_state.json"
    path.write_text(
        '{"status":"unknown","created_at":"2026-05-31T00:00:00Z","updated_at":"2026-05-31T00:00:00Z"}',
        encoding="utf-8",
    )
    store = FirstRunStateStore(path)

    state = store.load()

    assert state.status == FirstRunStatus.BLOCKED
    assert state.current_step == "state_recovery"
    assert state.step_data["state_recovery"]["reason"] == "invalid_schema"
    assert state.step_data["state_recovery"]["quarantined"] is True
    assert path.exists()
    assert list(tmp_path.glob("first_run_state.json.corrupt-*"))


def test_skip_records_skipped_not_completed(tmp_path: Path):
    store = FirstRunStateStore(tmp_path / "first_run_state.json")

    state = store.mark_skipped(reason="user_skip")

    assert state.status == FirstRunStatus.SKIPPED
    assert state.completed_at is None
    assert state.skip_reason == "user_skip"


def test_first_chat_success_allows_completion(tmp_path: Path):
    store = FirstRunStateStore(tmp_path / "first_run_state.json")

    _persist_required_first_run_step_data(store)
    store.record_first_chat_success(
        provider="openai",
        model="gpt-4.1-mini",
        response_id="chatcmpl-test",
    )
    state = store.mark_completed()

    assert state.status == FirstRunStatus.COMPLETED
    assert state.completed_at is not None
    assert state.first_chat.completed is True


def test_validate_completion_ready_does_not_mark_completed(tmp_path: Path):
    store = FirstRunStateStore(tmp_path / "first_run_state.json")

    _persist_required_first_run_step_data(store)
    store.record_first_chat_success(
        provider="openai",
        model="gpt-4.1-mini",
        response_id="chatcmpl-test",
    )
    state = store.validate_completion_ready()

    assert state.status == FirstRunStatus.FIRST_CHAT_COMPLETE
    assert state.completed_at is None
    assert state.first_chat.completed is True
