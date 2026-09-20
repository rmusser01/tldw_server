from __future__ import annotations

import asyncio
import hashlib
import json
import multiprocessing
import os
import signal
import struct
import subprocess
import sys
import textwrap
import threading
import time
import traceback

import pytest

from tldw_Server_API.app.core.Slides import standalone_html_validation_pool as pool_module
from tldw_Server_API.app.core.Slides.standalone_html_contracts import (
    StandaloneHtmlValidationError,
    StandaloneHtmlValidationResult,
)
from tldw_Server_API.app.core.Slides.standalone_html_validation_pool import (
    StandaloneHtmlValidationPool,
)
from tldw_Server_API.app.core.Slides.standalone_html_validator import (
    MAX_DOCUMENT_BYTES,
    validate_standalone_html,
)


def _document(title: str = "Deck") -> str:
    return (
        "<!doctype html><html><head><meta charset=utf-8>"
        f"<title>{title}</title><style>body{{color:#111}}</style></head>"
        '<body><section class="slide"><h1>Ready</h1></section>'
        "<script>document.addEventListener('keydown',()=>{});</script></body></html>"
    )


_INPUT_PREFLIGHT_SECRET = "TOP-SECRET-POOL-PREFLIGHT"


class _ExplodingOversizedText(str):
    def encode(self, *_args: object, **_kwargs: object) -> bytes:
        raise AssertionError(_INPUT_PREFLIGHT_SECRET)

    def __reduce_ex__(self, _protocol: int) -> object:
        raise AssertionError(_INPUT_PREFLIGHT_SECRET)


class _ExplodingOversizedBytes(bytes):
    def __bytes__(self) -> bytes:
        raise AssertionError(_INPUT_PREFLIGHT_SECRET)

    def decode(self, *_args: object, **_kwargs: object) -> str:
        raise AssertionError(_INPUT_PREFLIGHT_SECRET)

    def __reduce_ex__(self, _protocol: int) -> object:
        raise AssertionError(_INPUT_PREFLIGHT_SECRET)


def _rejected_pool_document(kind: str) -> str | bytes:
    if kind == "oversized-text":
        return _ExplodingOversizedText(_INPUT_PREFLIGHT_SECRET + ("x" * MAX_DOCUMENT_BYTES))
    if kind == "oversized-bytes":
        return _ExplodingOversizedBytes((_INPUT_PREFLIGHT_SECRET + ("x" * MAX_DOCUMENT_BYTES)).encode("ascii"))
    if kind == "lone-surrogate":
        return _document(f"{_INPUT_PREFLIGHT_SECRET}\ud800")
    if kind == "invalid-utf8":
        return _INPUT_PREFLIGHT_SECRET.encode("ascii") + b"\xff"
    raise AssertionError("unknown rejected document fixture")


def _slow_validate(document: str | bytes, *, delivery_style: str | None = None):
    time.sleep(0.25)
    return validate_standalone_html(document, delivery_style=delivery_style)


def _very_slow_validate(document: str | bytes, *, delivery_style: str | None = None):
    time.sleep(2)
    return validate_standalone_html(document, delivery_style=delivery_style)


def _hang_on_marker(document: str | bytes, *, delivery_style: str | None = None):
    source = document.decode("utf-8") if isinstance(document, bytes) else document
    if "HANG-VALIDATOR" in source:
        time.sleep(60)
    return validate_standalone_html(document, delivery_style=delivery_style)


def _malformed_title_on_marker(
    document: str | bytes,
    *,
    delivery_style: str | None = None,
) -> StandaloneHtmlValidationResult:
    source = document.decode("utf-8") if isinstance(document, bytes) else document
    if "MALFORMED-RESULT" in source:
        return StandaloneHtmlValidationResult(
            title="\ud800",
            slide_count=1,
            html_bytes=1,
            html_sha256="0" * 64,
            indexable_text="safe",
        )
    return validate_standalone_html(document, delivery_style=delivery_style)


def _diagnostic_error(
    _document_source: str | bytes,
    *,
    delivery_style: str | None = None,
) -> StandaloneHtmlValidationResult:
    del delivery_style
    raise StandaloneHtmlValidationError(
        "standalone_html_invalid_document",
        status_code=422,
        reason="html_parse_error",
        line=7,
        column=11,
    )


_SERIALIZATION_SECRET = "TOP-SECRET-POSTRETURN"


class _ExplodingResult:
    @property
    def title(self) -> str:
        raise RuntimeError(_SERIALIZATION_SECRET)


def _exploding_result_validator(
    _document_source: str | bytes,
    *,
    delivery_style: str | None = None,
) -> _ExplodingResult:
    del delivery_style
    return _ExplodingResult()


def _partial_response_worker_main(
    connection,
    _validator,
    require_isolated_imports: bool = False,
) -> None:
    del _validator, require_isolated_imports
    connection.send((pool_module._IPC_VERSION, "ready", True))
    connection.recv()
    os.write(connection.fileno(), struct.pack("!i", 4_096) + b"partial")
    time.sleep(60)


def test_spawn_worker_ready_handshake_has_no_eager_slides_imports() -> None:
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=True)
    process = context.Process(
        target=pool_module._validator_worker_main,
        args=(child_connection, validate_standalone_html, True),
    )
    process.start()
    child_connection.close()
    try:
        assert parent_connection.poll(10), "spawned validator never became ready"
        assert parent_connection.recv() == (pool_module._IPC_VERSION, "ready", True)
        parent_connection.send((pool_module._IPC_VERSION, "close"))
        process.join(5)
        assert not process.is_alive()
    finally:
        parent_connection.close()
        if process.is_alive():
            process.terminate()
            process.join(2)
        if process.is_alive():
            process.kill()
            process.join(2)


def test_legacy_slides_package_exports_resolve_lazily_to_original_objects() -> None:
    exports = {
        "ConflictError": ("slides_db", "ConflictError"),
        "InputError": ("slides_db", "InputError"),
        "SchemaError": ("slides_db", "SchemaError"),
        "SlidesDatabase": ("slides_db", "SlidesDatabase"),
        "SlidesDatabaseError": ("slides_db", "SlidesDatabaseError"),
        "SlidesGenerator": ("slides_generator", "SlidesGenerator"),
        "export_presentation_bundle": ("slides_export", "export_presentation_bundle"),
        "export_presentation_json": ("slides_export", "export_presentation_json"),
        "export_presentation_markdown": ("slides_export", "export_presentation_markdown"),
        "export_presentation_pdf": ("slides_export", "export_presentation_pdf"),
    }
    probe = textwrap.dedent(
        f"""
        import importlib
        import json
        import sys

        package_name = "tldw_Server_API.app.core.Slides"
        exports = {exports!r}
        package = importlib.import_module(package_name)
        heavy_modules = {{f"{{package_name}}.{{module}}" for module, _ in exports.values()}}
        assert not (heavy_modules & sys.modules.keys())
        assert package.__all__ == {list(exports)!r}
        for name, (module, attribute) in exports.items():
            actual = getattr(package, name)
            expected = getattr(importlib.import_module(f"{{package_name}}.{{module}}"), attribute)
            assert actual is expected
        print(json.dumps({{"ok": True}}))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2_000:]
    assert json.loads(completed.stdout.splitlines()[-1]) == {"ok": True}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "expected_code", "expected_reason"),
    [
        ("oversized-text", "standalone_html_validation_budget_exceeded", "document_bytes"),
        ("oversized-bytes", "standalone_html_validation_budget_exceeded", "document_bytes"),
        ("lone-surrogate", "standalone_html_invalid_document", "document_encoding"),
        ("invalid-utf8", "standalone_html_invalid_document", "document_encoding"),
    ],
)
async def test_rejected_input_never_starts_pool_or_enters_ipc(
    kind: str,
    expected_code: str,
    expected_reason: str,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    pool = StandaloneHtmlValidationPool(max_workers=1, mp_start_method="fork")
    calls: list[str] = []

    def forbidden_spawn(*_args: object, **_kwargs: object) -> object:
        calls.append("spawn")
        raise AssertionError(_INPUT_PREFLIGHT_SECRET)

    def forbidden_rpc(*_args: object, **_kwargs: object) -> object:
        calls.append("rpc")
        raise AssertionError(_INPUT_PREFLIGHT_SECRET)

    monkeypatch.setattr(pool, "_spawn_slot", forbidden_spawn)
    monkeypatch.setattr(pool, "_rpc_sync", forbidden_rpc)
    try:
        with pytest.raises(StandaloneHtmlValidationError) as caught:
            await pool.validate(_rejected_pool_document(kind))

        assert caught.value.code == expected_code
        assert caught.value.reason == expected_reason
        assert caught.value.__context__ is None
        assert calls == []
        assert pool.worker_pids == ()
        assert pool._started is False
        assert pool._request_counter == 0
        assert pool.interactive_waiting == 0
        assert pool.generation_slots_in_use == 0
        assert pool.active_count == 0
        rendered = "".join(traceback.format_exception(caught.value))
        assert _INPUT_PREFLIGHT_SECRET not in rendered
    finally:
        await pool.close()

    captured = capfd.readouterr()
    assert _INPUT_PREFLIGHT_SECRET not in captured.out
    assert _INPUT_PREFLIGHT_SECRET not in captured.err


@pytest.mark.asyncio
async def test_reserved_rejected_input_preserves_capacity_and_never_enters_ipc(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    pool = StandaloneHtmlValidationPool(max_workers=1, mp_start_method="fork")
    reservation = await pool.acquire_generation_reservation()
    original_pids = pool.worker_pids
    rpc_calls: list[int] = []
    original_rpc = pool._rpc_sync

    def forbidden_rpc(slot, job, watchdog_seconds):
        del watchdog_seconds
        rpc_calls.append(job.request_id)
        return (
            pool_module._IPC_VERSION,
            "error",
            slot.epoch,
            job.request_id,
            "standalone_html_validation_budget_exceeded",
            422,
            None,
            "document_bytes",
            None,
            None,
        )

    monkeypatch.setattr(pool, "_rpc_sync", forbidden_rpc)
    try:
        with pytest.raises(StandaloneHtmlValidationError) as caught:
            await asyncio.wait_for(
                reservation.validate(_rejected_pool_document("oversized-text")),
                2,
            )

        assert caught.value.code == "standalone_html_validation_budget_exceeded"
        assert caught.value.reason == "document_bytes"
        assert caught.value.__context__ is None
        assert rpc_calls == []
        assert reservation.consumed is False
        assert pool.generation_slots_in_use == 1
        assert pool.interactive_waiting == 0
        assert pool.active_count == 0
        assert pool.worker_pids == original_pids
        rendered = "".join(traceback.format_exception(caught.value))
        assert _INPUT_PREFLIGHT_SECRET not in rendered
    finally:
        monkeypatch.setattr(pool, "_rpc_sync", original_rpc)
        await reservation.release()
        await pool.close()

    captured = capfd.readouterr()
    assert _INPUT_PREFLIGHT_SECRET not in captured.out
    assert _INPUT_PREFLIGHT_SECRET not in captured.err


@pytest.mark.asyncio
async def test_pool_starts_no_more_than_four_killable_subprocesses() -> None:
    with pytest.raises(ValueError, match="four"):
        StandaloneHtmlValidationPool(max_workers=5)

    pool = StandaloneHtmlValidationPool(max_workers=4, mp_start_method="fork")
    await pool.start()
    pids = pool.worker_pids
    try:
        assert len(pids) == 4
        assert len(set(pids)) == 4
        assert all(pid > 0 for pid in pids)
        assert all(name.startswith("standalone-html-validator-") for name in pool.worker_names)
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_interactive_queue_capacity_is_24_and_saturation_is_redacted_503() -> None:
    pool = StandaloneHtmlValidationPool(
        max_workers=1,
        watchdog_seconds=5,
        validator=_very_slow_validate,
        mp_start_method="fork",
    )
    await pool.start()
    blocker = asyncio.create_task(pool.validate(_document("active")))
    for _ in range(100):
        if pool.active_count == 1:
            break
        await asyncio.sleep(0.005)
    tasks = [blocker]
    tasks.extend(asyncio.create_task(pool.validate(_document(f"Deck {index}"))) for index in range(24))
    try:
        for _ in range(100):
            if pool.interactive_waiting == 24:
                break
            await asyncio.sleep(0.005)
        with pytest.raises(StandaloneHtmlValidationError) as caught:
            await pool.validate(_document("never queued"))
        assert caught.value.code == "standalone_html_validator_busy"
        assert caught.value.status_code == 503
        assert 1 <= (caught.value.retry_after or 0) <= 5
    finally:
        for task in tasks:
            task.cancel()
        _done, pending = await asyncio.wait(tasks, timeout=3)
        assert not pending, [
            (task.get_coro().__qualname__, [frame.f_code.co_name for frame in task.get_stack()]) for task in pending
        ]
        await pool.close()


@pytest.mark.asyncio
async def test_generation_queue_has_eight_reserved_slots_before_provider_dispatch() -> None:
    pool = StandaloneHtmlValidationPool(max_workers=1, mp_start_method="fork")
    reservations = [await pool.acquire_generation_reservation() for _ in range(8)]
    try:
        assert pool.generation_slots_in_use == 8
        assert all(not reservation.consumed for reservation in reservations)
        with pytest.raises(StandaloneHtmlValidationError) as caught:
            await pool.acquire_generation_reservation()
        assert caught.value.code == "standalone_html_validator_busy"
        assert caught.value.status_code == 503
    finally:
        await asyncio.gather(*(reservation.release() for reservation in reservations))
        await pool.close()


@pytest.mark.asyncio
async def test_generation_reservation_is_consumed_when_returned_document_is_queued() -> None:
    pool = StandaloneHtmlValidationPool(max_workers=1, mp_start_method="fork")
    reservation = await pool.acquire_generation_reservation()

    result = await reservation.validate(_document("Generated"))

    assert result.title == "Generated"
    assert reservation.consumed is True
    assert pool.generation_slots_in_use == 0
    with pytest.raises(RuntimeError, match="consumed"):
        await reservation.validate(_document("Twice"))
    await pool.close()


@pytest.mark.asyncio
async def test_weighted_scheduling_serves_both_queues_without_starvation() -> None:
    pool = StandaloneHtmlValidationPool(
        max_workers=1,
        watchdog_seconds=5,
        validator=_slow_validate,
        mp_start_method="fork",
    )
    blocker = asyncio.create_task(pool.validate(_document("Blocker")))
    for _ in range(100):
        if pool.active_count == 1:
            break
        await asyncio.sleep(0.005)

    reservations = [await pool.acquire_generation_reservation() for _ in range(2)]
    completed: list[str] = []

    async def interactive(index: int) -> None:
        await pool.validate(_document(f"I{index}"))
        completed.append(f"I{index}")

    async def generated(index: int) -> None:
        await reservations[index].validate(_document(f"G{index}"))
        completed.append(f"G{index}")

    tasks = [asyncio.create_task(interactive(index)) for index in range(6)]
    tasks += [asyncio.create_task(generated(index)) for index in range(2)]
    try:
        await blocker
        await asyncio.gather(*tasks)
        assert any(item.startswith("G") for item in completed[:4])
        assert any(item.startswith("I") for item in completed[:4])
        assert sorted(completed) == ["G0", "G1", "I0", "I1", "I2", "I3", "I4", "I5"]
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_watchdog_terminates_reaps_and_replaces_hung_worker() -> None:
    pool = StandaloneHtmlValidationPool(
        max_workers=1,
        watchdog_seconds=0.05,
        validator=_hang_on_marker,
        mp_start_method="fork",
    )
    await pool.start()
    old_pid = pool.worker_pids[0]

    with pytest.raises(StandaloneHtmlValidationError) as caught:
        await pool.validate(_document("HANG-VALIDATOR"))

    assert caught.value.code == "standalone_html_validator_timeout"
    assert caught.value.status_code == 503
    assert pool.worker_pids[0] != old_pid
    validate = await pool.validate(_document("Recovered"))
    assert validate.title == "Recovered"
    with pytest.raises(ProcessLookupError):
        os.kill(old_pid, 0)
    await pool.close()


@pytest.mark.asyncio
async def test_caller_cancellation_discards_work_and_replaces_active_worker() -> None:
    pool = StandaloneHtmlValidationPool(
        max_workers=1,
        watchdog_seconds=5,
        validator=_hang_on_marker,
        mp_start_method="fork",
    )
    await pool.start()
    old_pid = pool.worker_pids[0]
    task = asyncio.create_task(pool.validate(_document("HANG-VALIDATOR")))
    for _ in range(100):
        if pool.active_count == 1:
            break
        await asyncio.sleep(0.005)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    for _ in range(100):
        if pool.worker_pids[0] != old_pid:
            break
        await asyncio.sleep(0.005)
    assert pool.worker_pids[0] != old_pid
    assert (await pool.validate(_document("After cancellation"))).title == "After cancellation"
    await pool.close()


@pytest.mark.asyncio
async def test_source_never_appears_in_public_errors_logs_or_process_metadata(capsys) -> None:
    secret = "TOP-SECRET-POOL-SOURCE"
    pool = StandaloneHtmlValidationPool(max_workers=1, mp_start_method="fork")

    with pytest.raises(StandaloneHtmlValidationError) as caught:
        await pool.validate(_document(secret).replace("<h1>", '<h1 onclick="bad()">'))

    captured = capsys.readouterr()
    public = " ".join(
        [
            str(caught.value),
            repr(caught.value),
            captured.out,
            captured.err,
            repr(pool.worker_names),
        ]
    )
    assert secret not in public
    assert caught.value.code == "standalone_html_invalid_document"
    await pool.close()


@pytest.mark.asyncio
async def test_close_terminates_and_reaps_every_worker() -> None:
    pool = StandaloneHtmlValidationPool(max_workers=2, mp_start_method="fork")
    await pool.start()
    pids = pool.worker_pids

    await pool.close()

    assert pool.worker_pids == ()
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.mark.asyncio
async def test_malformed_worker_response_fails_closed_without_stranding_capacity() -> None:
    pool = StandaloneHtmlValidationPool(
        max_workers=1,
        validator=_malformed_title_on_marker,
        mp_start_method="fork",
    )
    try:
        with pytest.raises(StandaloneHtmlValidationError) as caught:
            await asyncio.wait_for(pool.validate(_document("MALFORMED-RESULT")), 2)
        assert caught.value.code == "validator_unavailable"
        assert pool.active_count == 0
        assert (await asyncio.wait_for(pool.validate(_document("Recovered")), 2)).title == "Recovered"
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_worker_error_diagnostics_survive_closed_ipc() -> None:
    pool = StandaloneHtmlValidationPool(
        max_workers=1,
        validator=_diagnostic_error,
        mp_start_method="fork",
    )
    try:
        with pytest.raises(StandaloneHtmlValidationError) as caught:
            await pool.validate(_document())
        assert (caught.value.line, caught.value.column) == (7, 11)
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_generation_admission_repairs_a_dead_worker_before_provider_dispatch() -> None:
    pool = StandaloneHtmlValidationPool(max_workers=1, mp_start_method="fork")
    await pool.start()
    old_pid = pool.worker_pids[0]
    slot = pool._slots[0]
    assert slot is not None
    os.kill(old_pid, signal.SIGKILL)
    slot.process.join(2)
    assert pool.worker_pids == ()
    try:
        reservation = await asyncio.wait_for(pool.acquire_generation_reservation(), 2)
        assert len(pool.worker_pids) == 1
        assert pool.worker_pids[0] != old_pid
        await reservation.release()
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_close_releases_unused_generation_reservations_and_process_handles() -> None:
    pool = StandaloneHtmlValidationPool(max_workers=1, mp_start_method="fork")
    reservation = await pool.acquire_generation_reservation()
    slot = pool._slots[0]
    assert slot is not None
    process = slot.process

    await pool.close()

    assert pool.generation_slots_in_use == 0
    assert reservation._state == "released"
    await reservation.release()
    with pytest.raises(ValueError):
        _ = process.pid


@pytest.mark.asyncio
async def test_double_cancellation_racing_close_finishes_and_reaps() -> None:
    pool = StandaloneHtmlValidationPool(
        max_workers=1,
        watchdog_seconds=5,
        validator=_hang_on_marker,
        mp_start_method="fork",
    )
    await pool.start()
    pid = pool.worker_pids[0]
    task = asyncio.create_task(pool.validate(_document("HANG-VALIDATOR")))
    for _ in range(100):
        if pool.active_count:
            break
        await asyncio.sleep(0.005)

    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    close_task = asyncio.create_task(pool.close())
    results = await asyncio.wait_for(
        asyncio.gather(task, close_task, return_exceptions=True),
        3,
    )

    assert isinstance(results[0], asyncio.CancelledError)
    assert pool.active_count == 0
    assert pool.worker_pids == ()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.asyncio
async def test_default_spawn_pool_recovers_from_dead_worker_and_closes_cleanly() -> None:
    pool = StandaloneHtmlValidationPool(max_workers=1, watchdog_seconds=2)
    await pool.start()
    old_pid = pool.worker_pids[0]
    assert (await asyncio.wait_for(pool.validate(_document("Spawned")), 5)).title == "Spawned"
    slot = pool._slots[0]
    assert slot is not None
    os.kill(old_pid, signal.SIGKILL)
    slot.process.join(2)

    try:
        with pytest.raises(StandaloneHtmlValidationError) as caught:
            await asyncio.wait_for(pool.validate(_document("Repair")), 5)
        assert caught.value.code == "validator_unavailable"
        assert (await asyncio.wait_for(pool.validate(_document("After repair")), 5)).title == "After repair"
        assert pool.worker_pids[0] != old_pid
    finally:
        await asyncio.wait_for(pool.close(), 5)
    with pytest.raises(ProcessLookupError):
        os.kill(old_pid, 0)


@pytest.mark.asyncio
async def test_cancelling_replacement_readiness_reaps_unadmitted_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = StandaloneHtmlValidationPool(max_workers=1, mp_start_method="fork")
    await pool.start()
    original_spawn = pool._spawn_slot
    original_ready = pool._await_ready_sync
    replacement_slots = []
    replacement_pids: list[int] = []
    ready_started = threading.Event()

    def record_spawn(index: int, epoch: int):
        slot = original_spawn(index, epoch)
        replacement_slots.append(slot)
        assert slot.process.pid is not None
        replacement_pids.append(slot.process.pid)
        return slot

    def delayed_ready(slot, require_isolated_imports: bool) -> bool:
        ready_started.set()
        time.sleep(0.25)
        return original_ready(slot, require_isolated_imports)

    monkeypatch.setattr(pool, "_spawn_slot", record_spawn)
    monkeypatch.setattr(pool, "_await_ready_sync", delayed_ready)
    replacement = asyncio.create_task(pool._replace_worker(0))
    try:
        assert await asyncio.to_thread(ready_started.wait, 2)
        replacement.cancel()
        with pytest.raises(asyncio.CancelledError):
            await replacement
        assert replacement_slots
        child = replacement_slots[-1].process
        with pytest.raises(ValueError):
            child.is_alive()
        with pytest.raises(ProcessLookupError):
            os.kill(replacement_pids[-1], 0)
    finally:
        monkeypatch.setattr(pool, "_spawn_slot", original_spawn)
        monkeypatch.setattr(pool, "_await_ready_sync", original_ready)
        for slot in replacement_slots:
            try:
                pool._terminate_slot_sync(slot)
            except ValueError:
                pass
        await pool.close()


@pytest.mark.asyncio
async def test_close_fails_closed_when_worker_cannot_be_confirmed_reaped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = StandaloneHtmlValidationPool(max_workers=1, mp_start_method="fork")
    await pool.start()
    pid = pool.worker_pids[0]
    original_terminate = pool._terminate_slot_sync
    monkeypatch.setattr(pool, "_terminate_slot_sync", lambda _slot: False)

    with pytest.raises(StandaloneHtmlValidationError) as caught:
        await pool.close()

    assert caught.value.code == "validator_unavailable"
    assert pool.worker_pids == (pid,)
    monkeypatch.setattr(pool, "_terminate_slot_sync", original_terminate)
    await pool.close()
    assert pool._closed is True
    assert pool._closing is False
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.asyncio
async def test_watchdog_bounds_blocked_send_and_joins_rpc_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = StandaloneHtmlValidationPool(
        max_workers=1,
        watchdog_seconds=0.05,
        mp_start_method="fork",
    )
    await pool.start()
    old_pid = pool.worker_pids[0]
    rpc_finished = threading.Event()
    original_rpc = pool._rpc_sync

    def tracked_rpc(*args, **kwargs):
        try:
            return original_rpc(*args, **kwargs)
        finally:
            rpc_finished.set()

    monkeypatch.setattr(pool, "_rpc_sync", tracked_rpc)
    os.kill(old_pid, signal.SIGSTOP)
    started = time.monotonic()
    try:
        with pytest.raises(StandaloneHtmlValidationError) as caught:
            await asyncio.wait_for(pool.validate(b"x" * 1_000_000), 1)
        elapsed = time.monotonic() - started
        assert caught.value.code == "standalone_html_validator_timeout"
        assert elapsed < 0.75
        assert rpc_finished.is_set()
        assert pool.worker_pids[0] != old_pid
        assert (await pool.validate(_document("Recovered send"))).title == "Recovered send"
    finally:
        try:
            os.kill(old_pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        await asyncio.wait_for(pool.close(), 3)


@pytest.mark.asyncio
async def test_watchdog_bounds_partial_receive_and_joins_rpc_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_worker_main = pool_module._validator_worker_main
    monkeypatch.setattr(pool_module, "_validator_worker_main", _partial_response_worker_main)
    pool = StandaloneHtmlValidationPool(
        max_workers=1,
        watchdog_seconds=0.05,
        mp_start_method="fork",
    )
    await pool.start()
    old_pid = pool.worker_pids[0]
    monkeypatch.setattr(pool_module, "_validator_worker_main", original_worker_main)
    rpc_finished = threading.Event()
    original_rpc = pool._rpc_sync

    def tracked_rpc(*args, **kwargs):
        try:
            return original_rpc(*args, **kwargs)
        finally:
            rpc_finished.set()

    monkeypatch.setattr(pool, "_rpc_sync", tracked_rpc)
    started = time.monotonic()
    try:
        with pytest.raises(StandaloneHtmlValidationError) as caught:
            await asyncio.wait_for(pool.validate(_document("Partial response")), 1)
        elapsed = time.monotonic() - started
        assert caught.value.code == "standalone_html_validator_timeout"
        assert elapsed < 0.75
        assert rpc_finished.is_set()
        assert pool.worker_pids[0] != old_pid
        assert (await pool.validate(_document("Recovered receive"))).title == "Recovered receive"
    finally:
        monkeypatch.setattr(pool_module, "_validator_worker_main", original_worker_main)
        await asyncio.wait_for(pool.close(), 3)


def test_child_result_projection_failure_is_source_redacted(capfd: pytest.CaptureFixture[str]) -> None:
    context = multiprocessing.get_context("fork")
    parent_connection, child_connection = context.Pipe(duplex=True)
    process = context.Process(
        target=pool_module._validator_worker_main,
        args=(child_connection, _exploding_result_validator, False),
    )
    process.start()
    child_connection.close()
    response: object = None
    try:
        assert parent_connection.poll(2)
        ready = parent_connection.recv()
        assert ready[:2] == (pool_module._IPC_VERSION, "ready")
        assert isinstance(ready[2], bool)
        parent_connection.send((pool_module._IPC_VERSION, "validate", 1, 1, _document(), None))
        if parent_connection.poll(2):
            try:
                response = parent_connection.recv()
            except EOFError:
                pass
        if response is not None:
            parent_connection.send((pool_module._IPC_VERSION, "close"))
        process.join(2)
        assert not process.is_alive()
    finally:
        parent_connection.close()
        if process.is_alive():
            process.terminate()
            process.join(2)
    captured = capfd.readouterr()
    public = f"{response!r} {captured.out} {captured.err}"
    assert _SERIALIZATION_SECRET not in public
    if response is not None:
        assert response[4] == "validator_unavailable"


@pytest.mark.asyncio
async def test_close_cancellation_finishes_terminal_cleanup_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = StandaloneHtmlValidationPool(max_workers=2, mp_start_method="fork")
    await pool.start()
    pids = pool.worker_pids
    started = threading.Event()
    release = threading.Event()
    original_terminate = pool._terminate_slot_sync

    def delayed_terminate(slot):
        started.set()
        release.wait(2)
        return original_terminate(slot)

    monkeypatch.setattr(pool, "_terminate_slot_sync", delayed_terminate)
    close_task = asyncio.create_task(pool.close())
    assert await asyncio.to_thread(started.wait, 2)
    close_task.cancel()
    close_task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(close_task, 3)

    assert pool._closed is True
    assert pool._closing is False
    assert pool._slots == []
    assert pool.worker_pids == ()
    assert pool.active_count == 0
    assert pool.interactive_waiting == 0
    assert pool.generation_slots_in_use == 0
    await pool.close()
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def _malformed_response(kind: str, slot, job) -> tuple[object, ...]:
    document_bytes = job.document if isinstance(job.document, bytes) else job.document.encode("utf-8")
    result = [
        pool_module._IPC_VERSION,
        "result",
        slot.epoch,
        job.request_id,
        "Deck",
        1,
        len(document_bytes),
        hashlib.sha256(document_bytes).hexdigest(),
        "Ready",
    ]
    errors = {
        "invalid_status": (
            pool_module._IPC_VERSION,
            "error",
            slot.epoch,
            job.request_id,
            "standalone_html_invalid_document",
            503,
            None,
            "html_parse_error",
            None,
            None,
        ),
        "invalid_retry": (
            pool_module._IPC_VERSION,
            "error",
            slot.epoch,
            job.request_id,
            "standalone_html_invalid_document",
            422,
            1,
            "html_parse_error",
            None,
            None,
        ),
        "budget_reason": (
            pool_module._IPC_VERSION,
            "error",
            slot.epoch,
            job.request_id,
            "standalone_html_validation_budget_exceeded",
            422,
            None,
            "html_parse_error",
            None,
            None,
        ),
        "unavailable_status": (
            pool_module._IPC_VERSION,
            "error",
            slot.epoch,
            job.request_id,
            "validator_unavailable",
            422,
            None,
            None,
            None,
            None,
        ),
        "unavailable_reason": (
            pool_module._IPC_VERSION,
            "error",
            slot.epoch,
            job.request_id,
            "validator_unavailable",
            503,
            None,
            "html_parse_error",
            None,
            None,
        ),
        "non_parser_location": (
            pool_module._IPC_VERSION,
            "error",
            slot.epoch,
            job.request_id,
            "standalone_html_invalid_document",
            422,
            None,
            "title_blank",
            7,
            11,
        ),
        "parser_line_only": (
            pool_module._IPC_VERSION,
            "error",
            slot.epoch,
            job.request_id,
            "standalone_html_invalid_document",
            422,
            None,
            "html_parse_error",
            7,
            None,
        ),
        "parser_column_only": (
            pool_module._IPC_VERSION,
            "error",
            slot.epoch,
            job.request_id,
            "standalone_html_invalid_document",
            422,
            None,
            "html_parse_error",
            None,
            11,
        ),
        "parser_line_out_of_range": (
            pool_module._IPC_VERSION,
            "error",
            slot.epoch,
            job.request_id,
            "standalone_html_invalid_document",
            422,
            None,
            "html_parse_error",
            0,
            11,
        ),
        "parser_column_out_of_range": (
            pool_module._IPC_VERSION,
            "error",
            slot.epoch,
            job.request_id,
            "standalone_html_invalid_document",
            422,
            None,
            "html_parse_error",
            7,
            1_000_001,
        ),
    }
    if kind in errors:
        return errors[kind]
    if kind in {"valid_result", "oversized_document"}:
        return tuple(result)
    result_index, value = {
        "title_control": (4, "Bad\x00Title"),
        "title_whitespace": (4, " Deck "),
        "title_nfc": (4, "Cafe\u0301"),
        "byte_mismatch": (6, len(document_bytes) - 1),
        "digest_mismatch": (7, "0" * 64),
    }[kind]
    result[result_index] = value
    return tuple(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind",
    [
        "title_control",
        "title_whitespace",
        "title_nfc",
        "byte_mismatch",
        "digest_mismatch",
        "invalid_status",
        "invalid_retry",
        "budget_reason",
        "unavailable_status",
        "unavailable_reason",
        "non_parser_location",
        "parser_line_only",
        "parser_column_only",
        "parser_line_out_of_range",
        "parser_column_out_of_range",
    ],
)
async def test_semantically_malformed_worker_tuples_replace_worker_and_release_capacity(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = StandaloneHtmlValidationPool(max_workers=1, mp_start_method="fork")
    await pool.start()
    old_pid = pool.worker_pids[0]
    original_rpc = pool._rpc_sync

    def malformed_rpc(slot, job, watchdog_seconds):
        del watchdog_seconds
        return _malformed_response(kind, slot, job)

    monkeypatch.setattr(pool, "_rpc_sync", malformed_rpc)
    try:
        document = _document("Malformed semantics")
        with pytest.raises(StandaloneHtmlValidationError) as caught:
            await asyncio.wait_for(pool.validate(document), 2)
        assert caught.value.code == "validator_unavailable"
        assert pool.active_count == 0
        assert pool.interactive_waiting == 0
        assert pool.worker_pids[0] != old_pid
        monkeypatch.setattr(pool, "_rpc_sync", original_rpc)
        assert (await pool.validate(_document("Recovered semantics"))).title == "Recovered semantics"
    finally:
        monkeypatch.setattr(pool, "_rpc_sync", original_rpc)
        await pool.close()


@pytest.mark.asyncio
async def test_internally_consistent_success_at_document_ceiling_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = StandaloneHtmlValidationPool(max_workers=1, mp_start_method="fork")
    await pool.start()
    old_pid = pool.worker_pids[0]
    original_rpc = pool._rpc_sync

    def valid_rpc(slot, job, watchdog_seconds):
        del watchdog_seconds
        return _malformed_response("valid_result", slot, job)

    monkeypatch.setattr(pool, "_rpc_sync", valid_rpc)
    try:
        result = await asyncio.wait_for(pool.validate("x" * MAX_DOCUMENT_BYTES), 2)
        assert result.html_bytes == MAX_DOCUMENT_BYTES
        assert pool.worker_pids == (old_pid,)
        assert pool.active_count == 0
    finally:
        monkeypatch.setattr(pool, "_rpc_sync", original_rpc)
        await pool.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["html_parse_error", "css_parse_error"])
async def test_parser_error_with_bounded_location_pair_is_accepted(
    reason: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = StandaloneHtmlValidationPool(max_workers=1, mp_start_method="fork")
    await pool.start()
    old_pid = pool.worker_pids[0]
    original_rpc = pool._rpc_sync

    def parser_error_rpc(slot, job, watchdog_seconds):
        del watchdog_seconds
        return (
            pool_module._IPC_VERSION,
            "error",
            slot.epoch,
            job.request_id,
            "standalone_html_invalid_document",
            422,
            None,
            reason,
            7,
            11,
        )

    monkeypatch.setattr(pool, "_rpc_sync", parser_error_rpc)
    try:
        with pytest.raises(StandaloneHtmlValidationError) as caught:
            await asyncio.wait_for(pool.validate(_document("Parser location")), 2)
        assert caught.value.code == "standalone_html_invalid_document"
        assert caught.value.reason == reason
        assert (caught.value.line, caught.value.column) == (7, 11)
        assert pool.worker_pids == (old_pid,)
        assert pool.active_count == 0
    finally:
        monkeypatch.setattr(pool, "_rpc_sync", original_rpc)
        await pool.close()
