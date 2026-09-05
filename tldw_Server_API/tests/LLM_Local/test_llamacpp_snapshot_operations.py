"""Single-dispatch, durable completion and launch fencing integration tests."""

import asyncio
import errno
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Local_LLM import llamacpp_snapshot_operations as ops
from tldw_Server_API.app.core.Local_LLM.llamacpp_runtime_models import (
    LlamaCppProfile,
    LlamaCppRuntime,
    LlamaCppRuntimeState,
)
from tldw_Server_API.app.core.Local_LLM.llamacpp_snapshot_models import SnapshotRequest
from tldw_Server_API.app.core.Local_LLM.llamacpp_snapshot_store import SnapshotStore


class Runner:
    def __init__(self, model, executable):
        self.snapshot_generation = "generation1"
        self.snapshot_working = None
        self.snapshot_process = SimpleNamespace(returncode=None)
        self.runtime = LlamaCppRuntime(
            profile_id="p1",
            state=LlamaCppRuntimeState.RUNNING,
            endpoint="http://127.0.0.1:8080",
            model_path=str(model),
            launch_generation="generation1",
        )
        self.snapshot_options = {"ctx_size": 2048}
        self.snapshot_executable = executable
        self.snapshot_fingerprint = ops.build_fingerprint(
            model=model, executable=executable, effective_options=self.snapshot_options, adapters=[]
        )

    def status(self):
        return self.runtime


class Transport:
    def __init__(self, runner):
        self.runner = runner
        self.calls = []
        self.failure = None
        self.block = None
        self.sent = asyncio.Event()

    async def __call__(self, **kwargs):
        if kwargs["method"] == "GET":
            # Source-derived shape, NOT a live capture: llama.cpp commit
            # 4d9176092d00586775af140581bb0b558ddc4389 server-context.cpp:686-719.
            return [
                {
                    "id": 0,
                    "is_processing": False,
                    "n_ctx": 2048,
                    "speculative": False,
                    "id_task": 12,
                    "n_prompt_tokens": 4,
                    "n_prompt_tokens_processed": 4,
                    "n_prompt_tokens_cache": 0,
                    "params": {},
                    "next_token": [{"has_next_token": False, "has_new_line": False, "n_remain": 0, "n_decoded": 0}],
                    "prompt": "must never escape",
                    "generated": "must never escape",
                }
            ]
        self.calls.append(kwargs)
        self.sent.set()
        if self.block:
            await self.block.wait()
        if self.failure:
            if isinstance(self.failure, BaseException):
                raise self.failure
            return self.failure
        filename = kwargs["json"]["filename"]
        action = kwargs["params"]["action"]
        if action == "save":
            (self.runner.snapshot_working / filename).write_bytes(b"cache")
        return {
            "id_slot": 0,
            "filename": filename,
            "n_saved" if action == "save" else "n_restored": 4,
            "n_written" if action == "save" else "n_read": 5,
        }


@pytest.fixture
def setup(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    executable = tmp_path / "llama-server"
    executable.write_bytes(b"verified-test-fixture")
    runner = Runner(model, executable)
    store = SnapshotStore(tmp_path / "snapshots")
    runner.snapshot_working = store.launch_directory("p1", "generation1")
    profile = LlamaCppProfile(profile_id="p1", name="one", model_path=str(model), snapshots_enabled=True)
    transport = Transport(runner)
    service = ops.SnapshotOperations(store, transport=transport, supported_builds={ops.hash_file_stable(executable)})
    yield service, store, profile, runner, transport
    store.close()


async def submit(setup, **changes):
    service, _, profile, runner, _ = setup
    request = SnapshotRequest(
        slot_id=0, expected_launch_generation="generation1", request_id=service.issue_token("p1"), **changes
    )
    receipt = await service.admit(profile, runner, request, "admin", "save")
    return request, receipt


async def finish(service):
    await asyncio.gather(*list(service.tasks.values()))


async def test_full_cache_option_is_admissible_without_model_name_policy(setup):
    service, _, profile, runner, _ = setup
    profile.server_args = {"swa_full": True}
    assert await service.fingerprint(profile, runner) == runner.snapshot_fingerprint


async def test_identical_token_dispatches_once_and_changed_input_conflicts(setup):
    service, store, profile, runner, transport = setup
    request, first = await submit(setup)
    second = await service.admit(profile, runner, request, "admin", "save")
    assert second.operation_id == first.operation_id
    with pytest.raises(ops.SnapshotOperationError) as caught:
        await service.admit(profile, runner, request.model_copy(update={"slot_id": 1}), "admin", "save")
    assert caught.value.status_code == 409
    await finish(service)
    receipt = store.read_receipt("p1", first.operation_id)
    assert receipt.state == "complete"
    assert len(transport.calls) == 1
    assert len(store.list("p1")) == 1


@pytest.mark.parametrize("failure", [TimeoutError(), ConnectionError(), {"id_slot": 8}])
async def test_uncertain_save_never_commits_or_retries_and_quarantines(setup, failure):
    service, store, profile, runner, transport = setup
    transport.failure = failure
    _, receipt = await submit(setup)
    await finish(service)
    assert store.read_receipt("p1", receipt.operation_id).state == "outcome_unknown"
    assert store.list("p1") == []
    assert len(transport.calls) == 1
    with pytest.raises(ops.SnapshotOperationError):
        await submit(setup)
    service.guard_lifecycle("p1", stop=True)
    with pytest.raises(ops.SnapshotOperationError):
        service.guard_lifecycle("p1")


@pytest.mark.parametrize("change", ["generation", "process"])
async def test_changed_generation_or_owner_prevents_dispatch(setup, change):
    service, _, profile, runner, transport = setup
    if change == "generation":
        runner.snapshot_generation = "new"
    else:
        runner.snapshot_process = None
    with pytest.raises(ops.SnapshotOperationError):
        await submit(setup)
    assert transport.calls == []


async def test_busy_operation_blocks_lifecycle_and_shutdown_records_unknown(setup):
    service, store, _, _, transport = setup
    transport.block = asyncio.Event()
    _, receipt = await submit(setup)
    await transport.sent.wait()
    with pytest.raises(ops.SnapshotOperationError):
        service.guard_lifecycle("p1", stop=True)
    await service.drain(timeout=0.01)
    assert store.read_receipt("p1", receipt.operation_id).state == "outcome_unknown"
    assert store.list("p1") == []


async def test_restore_verifies_hash_and_requires_confirmation(setup):
    service, store, profile, runner, transport = setup
    await submit(setup)
    await finish(service)
    saved = store.list("p1")[0]
    request = SnapshotRequest(slot_id=0, expected_launch_generation="generation1", request_id=service.issue_token("p1"))
    with pytest.raises(ops.SnapshotOperationError):
        await service.admit(profile, runner, request, "admin", "restore", saved.snapshot_id)
    request.replace_confirmed = True
    receipt = await service.admit(profile, runner, request, "admin", "restore", saved.snapshot_id)
    await finish(service)
    assert store.read_receipt("p1", receipt.operation_id).state == "complete"
    assert len(transport.calls) == 2


def test_private_tokens_persist_and_reject_expired_future_and_other_profile(setup):
    service, store, _, _, _ = setup
    token = service.issue_token("p1")
    another = ops.SnapshotOperations(store)
    assert another.validate_token("p1", token)
    for profile, changed in [("p2", token), ("p1", token + "x")]:
        with pytest.raises(ops.SnapshotOperationError):
            service.validate_token(profile, changed)
    with pytest.raises(ops.SnapshotOperationError):
        service.validate_token("p1", token, now=service.clock() + 31 * 86400)
    with pytest.raises(ops.SnapshotOperationError):
        service.validate_token("p1", token, now=service.clock() - 5)


def test_monotonic_sequence_survives_deleting_all_entries(setup):
    _, store, _, _, _ = setup
    assert store.allocate_sequence("p1") == 1
    assert store.allocate_sequence("p1") == 2
    assert store.list("p1") == []


def test_replace_closes_source_descriptor_if_target_open_fails(tmp_path, monkeypatch):
    original = SnapshotStore._open_directory_fd
    opened = []

    def injected(path, **kwargs):
        if path.name == "absent":
            raise OSError("injected target error")
        fd = original(path, **kwargs)
        opened.append(fd)
        return fd

    monkeypatch.setattr(SnapshotStore, "_open_directory_fd", injected)
    with pytest.raises(OSError):
        SnapshotStore._replace(tmp_path / "source", tmp_path / "absent" / "target")
    with pytest.raises(OSError):
        os.fstat(opened[0])


async def test_disk_full_after_send_preserves_previous_snapshot_without_pruning(setup, monkeypatch):
    service, store, _, _, transport = setup
    await submit(setup)
    await finish(service)
    original = store.list("p1")[0]

    def disk_full(boundary):
        if boundary == "copy":
            raise OSError(errno.ENOSPC, "full")

    monkeypatch.setattr(store, "_checkpoint", disk_full)
    _, receipt = await submit(setup)
    await finish(service)
    assert store.read_receipt("p1", receipt.operation_id).state == "outcome_unknown"
    assert [item.snapshot_id for item in store.list("p1")] == [original.snapshot_id]
    assert len(transport.calls) == 2


async def test_startup_recovers_dispatch_marker_without_replaying(setup):
    service, store, profile, runner, transport = setup
    from tldw_Server_API.app.core.Local_LLM.llamacpp_snapshot_models import OperationReceipt

    receipt = OperationReceipt(
        profile_id="p1",
        operation_id="crashed",
        launch_generation="old",
        request_digest="a" * 64,
        kind="restore",
        state="restoring",
        dispatched=True,
    )
    store.write_receipt(receipt)
    result = await service.slots(profile.model_copy(update={"snapshots_enabled": False}), None)
    assert result["latest_operation_id"] == "crashed"
    assert store.read_receipt("p1", "crashed").state == "outcome_unknown"
    assert transport.calls == []


async def test_store_failure_is_not_an_empty_catalog(setup, monkeypatch):
    service, store, profile, runner, _ = setup
    from tldw_Server_API.app.core.Local_LLM.llamacpp_snapshot_store import SnapshotStorageUnavailableError

    def fail(*args):
        raise SnapshotStorageUnavailableError("injected")

    monkeypatch.setattr(store, "list", fail)
    with pytest.raises(SnapshotStorageUnavailableError):
        await service.catalog(profile, runner, 0, 25)


async def test_blocking_disk_work_retains_reservation_during_cancellation(setup, monkeypatch):
    service, store, _, _, transport = setup
    started = threading.Event()
    release = threading.Event()
    original = store.write_receipt

    def slow_dispatch(receipt):
        if receipt.dispatched and receipt.state == "saving":
            started.set()
            release.wait(5)
        return original(receipt)

    monkeypatch.setattr(store, "write_receipt", slow_dispatch)
    _, receipt = await submit(setup)
    assert await asyncio.to_thread(started.wait, 2)
    operation = service.tasks[receipt.operation_id]
    operation.cancel()
    await asyncio.sleep(0)
    assert "p1" in service.active
    release.set()
    await finish(service)
    assert store.read_receipt("p1", receipt.operation_id).state == "outcome_unknown"
    assert store.read_receipt("p1", receipt.operation_id).dispatched
    assert transport.calls == []


async def test_acknowledged_save_partial_prune_failure_stays_complete(setup, monkeypatch):
    service, store, _, _, _ = setup
    monkeypatch.setattr(store, "prune", lambda *args: ["old"])
    _, receipt = await submit(setup)
    await finish(service)
    saved = store.read_receipt("p1", receipt.operation_id)
    assert saved.state == "complete"
    assert saved.warning_code == "retention_partial_failure"


async def test_unverified_build_fails_closed_without_probing(setup):
    _, store, profile, runner, _ = setup

    async def forbidden_transport(**kwargs):
        raise AssertionError("unverified build must not be probed")

    service = ops.SnapshotOperations(store, transport=forbidden_transport)
    result = await service.slots(profile, runner)
    assert result["capability"] == "unsupported"
    assert result["reason"] == "unsupported_build"


async def test_checked_transport_pins_origin_and_disables_retry_redirects_and_body_logging(monkeypatch):
    captured = {}

    async def fetch(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(status_code=200, json=lambda: [])

    monkeypatch.setattr(ops.http_client, "afetch", fetch)
    assert (
        await ops.checked_transport(
            method="POST",
            url="http://127.0.0.1:8080/slots/0",
            origin="http://127.0.0.1:8080",
            remaining=40,
            json={"filename": "safe.bin"},
        )
        == []
    )
    assert captured["configured_endpoint"].matches("http://127.0.0.1:8080/slots/0")
    assert not captured["configured_endpoint"].matches("http://127.0.0.1:8081/slots/0")
    assert captured["allow_redirects"] is False
    assert captured["retry"].attempts == 1
    assert captured["sensitive_observability"] is True
    assert captured["max_response_bytes"] == 262144


async def test_changed_origin_during_dispatch_receipt_prevents_send(setup, monkeypatch):
    service, store, _, runner, transport = setup
    original = store.write_receipt

    def replace_origin(receipt):
        original(receipt)
        if receipt.dispatched:
            runner.runtime = runner.runtime.model_copy(update={"endpoint": "http://127.0.0.1:8181"})

    monkeypatch.setattr(store, "write_receipt", replace_origin)
    _, receipt = await submit(setup)
    await finish(service)
    assert transport.calls == []
    assert store.read_receipt("p1", receipt.operation_id).state == "outcome_unknown"


async def test_two_profiles_respect_server_concurrency(setup):
    service, store, profile, runner, transport = setup
    transport.block = asyncio.Event()
    _, receipt = await submit(setup)
    await transport.sent.wait()
    second_profile = profile.model_copy(update={"profile_id": "p2"})
    second_runner = Runner(Path(runner.runtime.model_path), runner.snapshot_executable)
    second_runner.snapshot_working = store.launch_directory("p2", "generation1")
    second_runner.runtime = second_runner.runtime.model_copy(update={"profile_id": "p2"})
    request = SnapshotRequest(slot_id=0, expected_launch_generation="generation1", request_id=service.issue_token("p2"))
    second = await service.admit(second_profile, second_runner, request, "admin", "save")
    await asyncio.sleep(0)
    assert len(transport.calls) == 1
    # Stop both blocked/queued tasks: neither can publish an unacknowledged artifact.
    await service.drain(timeout=0.01)
    assert store.read_receipt("p1", receipt.operation_id).state == "outcome_unknown"
    assert store.read_receipt("p2", second.operation_id).state == "failed"


def test_inherited_store_descriptor_does_not_grant_other_process_ownership(setup, monkeypatch):
    _, store, _, _, _ = setup
    from tldw_Server_API.app.core.Local_LLM.llamacpp_snapshot_store import SnapshotStoreError

    actual_pid = os.getpid()
    monkeypatch.setattr(os, "getpid", lambda: actual_pid + 1)
    with pytest.raises(SnapshotStoreError):
        store.list("p1")


async def test_shutdown_persists_unknown_before_cancelling_sent_transport(setup):
    service, store, _, _, transport = setup
    sent = asyncio.Event()
    cancellation_states = []
    original = transport.__call__

    async def block(**kwargs):
        if kwargs["method"] == "GET":
            return await original(**kwargs)
        sent.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_states.append(store.list_receipts("p1")[-1].state)
            raise

    service.transport = block
    await submit(setup)
    await sent.wait()
    await service.drain(timeout=0.01)
    assert cancellation_states == ["outcome_unknown"]


async def test_cancelled_admission_does_not_leave_unrecoverable_validating_receipt(setup, monkeypatch):
    service, store, _, _, transport = setup
    started, release = threading.Event(), threading.Event()
    original = store.write_receipt

    def slow_initial(receipt):
        if receipt.state == "validating":
            started.set()
            release.wait(5)
        return original(receipt)

    monkeypatch.setattr(store, "write_receipt", slow_initial)
    admission = asyncio.create_task(submit(setup))
    assert await asyncio.to_thread(started.wait, 2)
    admission.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await admission
    assert store.list_receipts("p1")[-1].state == "failed"
    assert transport.calls == []


async def test_model_replaced_after_launch_is_unsupported_even_before_first_save(setup):
    _, _, _, runner, transport = setup
    Path(runner.runtime.model_path).write_bytes(b"changed after child loaded model")
    with pytest.raises(ops.SnapshotOperationError) as caught:
        await submit(setup)
    assert caught.value.code == "runtime_identity_changed"
    assert transport.calls == []


async def test_save_acknowledgement_must_match_observed_token_count(setup):
    service, store, _, runner, transport = setup
    original = transport.__call__

    async def mismatched(**kwargs):
        result = await original(**kwargs)
        if kwargs["method"] == "POST":
            result["n_saved"] = 99
        return result

    service.transport = mismatched
    _, receipt = await submit(setup)
    await finish(service)
    assert store.read_receipt("p1", receipt.operation_id).state == "outcome_unknown"
    assert store.list("p1") == []


async def test_restore_receipt_retains_audit_actor(setup):
    service, store, profile, runner, _ = setup
    await submit(setup)
    await finish(service)
    source = store.list("p1")[0]
    receipt = await service.admit(
        profile,
        runner,
        SnapshotRequest(
            slot_id=0,
            expected_launch_generation="generation1",
            request_id=service.issue_token("p1"),
            replace_confirmed=True,
        ),
        "restoring-admin",
        "restore",
        source.snapshot_id,
    )
    await finish(service)
    assert store.read_receipt("p1", receipt.operation_id).actor_id == "restoring-admin"


async def test_repeated_verified_save_restore_keeps_working_bytes_bounded(setup):
    service, store, profile, runner, _ = setup
    profile.snapshot_retention = 1
    for _ in range(3):
        _, receipt = await submit(setup)
        await finish(service)
        assert store.read_receipt("p1", receipt.operation_id).state == "complete"
        source = store.list("p1")[0]
        await service.admit(
            profile,
            runner,
            SnapshotRequest(
                slot_id=0,
                expected_launch_generation="generation1",
                request_id=service.issue_token("p1"),
                replace_confirmed=True,
            ),
            "admin",
            "restore",
            source.snapshot_id,
        )
        await finish(service)
        assert list(runner.snapshot_working.iterdir()) == []
        assert len(store.list("p1")) == 1


@pytest.mark.parametrize("tokens", [None, -1, True, "4", 1.5])
async def test_source_derived_slot_shape_rejects_malformed_counts(setup, tokens):
    service, _, profile, runner, transport = setup
    original = transport.__call__

    async def malformed(**kwargs):
        payload = await original(**kwargs)
        payload[0]["n_prompt_tokens"] = tokens
        return payload

    service.transport = malformed
    result = await service.slots(profile, runner)
    assert result["reason"] == "invalid_slot_response"


async def test_source_derived_fresh_idle_slot_without_task_has_zero_tokens(setup):
    service, _, profile, runner, _ = setup

    async def fresh(**kwargs):
        return [{"id": 0, "n_ctx": 2048, "speculative": False, "is_processing": False}]

    service.transport = fresh
    result = await service.slots(profile, runner)
    assert result["capability"] == "ready"
    assert result["slots"] == [{"slot_id": 0, "busy": False, "token_count": 0}]


async def test_source_derived_busy_slot_is_reported_and_mutation_is_rejected(setup):
    service, store, profile, runner, transport = setup
    original = transport.__call__

    async def busy(**kwargs):
        payload = await original(**kwargs)
        payload[0]["is_processing"] = True
        return payload

    service.transport = busy
    slots = await service.slots(profile, runner)
    assert slots["slots"] == [{"slot_id": 0, "busy": True, "token_count": 4}]
    _, receipt = await submit(setup)
    await finish(service)
    assert store.read_receipt("p1", receipt.operation_id).error_code == "slot_busy"
    assert transport.calls == []


async def test_unknown_save_preserves_its_working_file(setup):
    service, store, _, runner, transport = setup
    original = transport.__call__

    async def wrote_then_disconnected(**kwargs):
        result = await original(**kwargs)
        if kwargs["method"] == "POST":
            raise ConnectionError("acknowledgement lost")
        return result

    service.transport = wrote_then_disconnected
    _, receipt = await submit(setup)
    await finish(service)
    assert store.read_receipt("p1", receipt.operation_id).state == "outcome_unknown"
    assert [path.read_bytes() for path in runner.snapshot_working.iterdir()] == [b"cache"]


async def test_verified_cleanup_failure_keeps_success_and_reports_warning(setup, monkeypatch):
    service, store, _, _, _ = setup

    def unavailable(*args):
        raise OSError(errno.EIO, "working cleanup unavailable")

    monkeypatch.setattr(store, "remove_working_file", unavailable, raising=False)
    _, receipt = await submit(setup)
    await finish(service)
    result = store.read_receipt("p1", receipt.operation_id)
    assert result.state == "complete"
    assert result.warning_code == "working_cleanup_failed"
    assert service.quarantined == {}
