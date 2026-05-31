from pathlib import Path

import pytest

from tldw_Server_API.app.core.Setup.first_run_state import (
    REQUIRED_FIRST_RUN_STEPS,
    FirstRunStateStore,
    FirstRunStatus,
    InvalidFirstRunTransition,
)


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


def test_complete_requires_first_chat_success(tmp_path: Path):
    store = FirstRunStateStore(tmp_path / "first_run_state.json")

    with pytest.raises(InvalidFirstRunTransition):
        store.mark_completed()


def test_complete_requires_required_step_acknowledgements(tmp_path: Path):
    store = FirstRunStateStore(tmp_path / "first_run_state.json")

    for step in REQUIRED_FIRST_RUN_STEPS:
        store.update_step(step, {"acknowledged": False})
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

    for step in REQUIRED_FIRST_RUN_STEPS:
        store.update_step(step, {"acknowledged": True})
    store.update_step("providers", {"acknowledged": False})
    store.record_first_chat_success(
        provider="openai",
        model="gpt-4.1-mini",
        response_id="chatcmpl-test",
    )

    with pytest.raises(InvalidFirstRunTransition) as excinfo:
        store.mark_completed()

    assert "required_steps_missing:providers" in str(excinfo.value)


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

    for step in REQUIRED_FIRST_RUN_STEPS:
        store.update_step(step, {"acknowledged": True})
    store.record_first_chat_success(
        provider="openai",
        model="gpt-4.1-mini",
        response_id="chatcmpl-test",
    )
    state = store.mark_completed()

    assert state.status == FirstRunStatus.COMPLETED
    assert state.completed_at is not None
    assert state.first_chat.completed is True
