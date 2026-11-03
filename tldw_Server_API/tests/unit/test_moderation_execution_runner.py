"""Parent ownership tests; hostile expressions live only in isolated probes."""

import asyncio
import io
import threading
import time

import pytest

from tldw_Server_API.app.core.exception_types import ModerationExecutionError
from tldw_Server_API.app.core.Moderation.execution_models import (
    ExecutionBounds,
    ExecutionScope,
    SubstitutionBatchResult,
    WorkerResponse,
    execution_scope,
)
from tldw_Server_API.app.core.Moderation.execution_protocol import (
    encode_response,
    snapshot_substitution_request,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolated_incident_reporting(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Moderation import execution_incidents as incidents

    monkeypatch.setenv("MODERATION_REVIEW_DB_PATH", str(tmp_path / "review.db"))
    incidents.prepare_moderation_incident_store()
    yield
    thread = incidents._report_thread
    if thread is not None:
        thread.join(1)
        assert not thread.is_alive()


def request(bounds=None):
    return snapshot_substitution_request("hello", [], None, bounds or ExecutionBounds())


def scope(seconds=2):
    return ExecutionScope(time.monotonic() + seconds, threading.Event(), "test")


class Child:
    """Controlled Popen boundary, including a genuinely blocked pipe reader."""

    def __init__(self, data=None, *, stalled=False, unreapable=False, code=0):
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(
            data
            if data is not None
            else encode_response(
                WorkerResponse(
                    "custom_substitute",
                    reason="worker_unavailable",
                    resource_limits={"address_space": False, "cpu": False, "core": False},
                )
            )
        )
        self.returncode = None if stalled else code
        self.unreapable = unreapable
        self.killed = threading.Event()
        if stalled:
            owner = self

            class Reader(io.BytesIO):
                def read(self, count=-1):
                    assert count > 0
                    owner.killed.wait(3)
                    return b""

            self.stdout = Reader()

    def kill(self):
        self.killed.set()
        if not self.unreapable:
            self.returncode = -9

    def poll(self):
        return self.returncode


def runner(factory, bounds=None, **kwargs):
    from tldw_Server_API.app.core.Moderation.execution_runner import ModerationExecutionRunner

    return ModerationExecutionRunner(bounds or ExecutionBounds(), process_factory=factory, **kwargs)


def test_fifth_active_worker_is_refused_without_queue():
    entered = threading.Barrier(5)
    release = threading.Event()

    def factory(*args, **kwargs):
        entered.wait(2)
        release.wait(2)
        return Child(stalled=True)

    owner = runner(factory)
    errors = []

    def run():
        try:
            owner.execute(request(), scope())
        except (ModerationExecutionError, asyncio.CancelledError) as error:
            errors.append(getattr(error, "reason", "cancelled"))

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    try:
        entered.wait(2)
        start = time.monotonic()
        with pytest.raises(ModerationExecutionError, match="capacity_exhausted"):
            owner.execute(request(), scope())
        assert time.monotonic() - start < 0.2
    finally:
        owner.close(timeout_s=0)
        release.set()
        for thread in threads:
            thread.join(3)
    assert len(errors) == 4


def test_decision_and_redaction_share_original_deadline():
    now = [10.0]
    owner = runner(lambda *a, **k: Child(), clock=lambda: now[0])
    shared = ExecutionScope(11.0, threading.Event(), "shared")
    with pytest.raises(ModerationExecutionError, match="worker_unavailable"):
        owner.execute(request(), shared)
    now[0] = 11.01
    with pytest.raises(ModerationExecutionError, match="deadline_exceeded"):
        owner.execute(request(), shared)
    owner.close()


def test_cancel_during_spawn_kills_eventual_child():
    entered, release = threading.Event(), threading.Event()
    child = Child(stalled=True)

    def factory(*a, **k):
        entered.set()
        release.wait(2)
        return child

    owner, shared, errors = runner(factory), scope(), []

    def run():
        try:
            owner.execute(request(), shared)
        except (ModerationExecutionError, asyncio.CancelledError) as error:
            errors.append(getattr(error, "reason", "cancelled"))

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(1)
        shared.cancelled.set()
        owner.close(timeout_s=0.01)
        release.set()
        thread.join(2)
        assert child.killed.is_set()
        assert errors == ["cancelled"]
    finally:
        release.set()
        thread.join(3)
        owner.close()


def test_unreaped_child_keeps_slot_quarantined():
    child = Child(stalled=True, unreapable=True)
    owner = runner(lambda *a, **k: child, ExecutionBounds(max_workers=1))
    try:
        with pytest.raises(ModerationExecutionError, match="deadline_exceeded"):
            owner.execute(request(), scope(0.02))
        with pytest.raises(ModerationExecutionError, match="capacity_exhausted"):
            owner.execute(request(), scope())
    finally:
        child.unreapable = False
        child.returncode = -9
        owner.close()


def test_quarantine_stops_pid_wide_admission_even_via_another_runner():
    child = Child(stalled=True, unreapable=True)
    owner = runner(lambda *a, **k: child)
    other = runner(lambda *a, **k: Child())
    try:
        with pytest.raises(ModerationExecutionError, match="deadline_exceeded"):
            owner.execute(request(), scope(0.02))
        with pytest.raises(ModerationExecutionError, match="capacity_exhausted"):
            other.execute(request(), scope())
    finally:
        child.returncode = -9
        owner.close()
        other.close()


def test_oversize_header_is_rejected_before_payload_is_read():
    class HeaderOnly(io.BytesIO):
        def read(self, count=-1):
            assert count == 4
            return b"\xff\xff\xff\xff"

    child = Child()
    child.stdout = HeaderOnly()
    owner = runner(lambda *a, **k: child)
    try:
        with pytest.raises(ModerationExecutionError, match="size_exceeded"):
            owner.execute(request(), scope())
    finally:
        owner.close()


@pytest.mark.parametrize("data", [b"\x00\x00", b"\xff\xff\xff\xff", b"\x00\x00\x00\x02{}extra"])
def test_partial_oversize_and_trailing_frames_are_rejected(data):
    child = Child(data)
    owner = runner(lambda *a, **k: child)
    try:
        with pytest.raises(ModerationExecutionError, match="invalid_response|size_exceeded"):
            owner.execute(request(), scope())
        assert child.killed.is_set()
    finally:
        owner.close()


def test_shutdown_rejects_admission_and_reaps_registry():
    child = Child(stalled=True)
    owner = runner(lambda *a, **k: child)
    owner.close()
    with pytest.raises(ModerationExecutionError, match="worker_unavailable"):
        owner.execute(request(), scope())


def test_nested_scope_does_not_refresh_deadline():
    with execution_scope() as outer:
        with execution_scope(timeout_s=0.01) as inner:
            assert inner is outer


def test_tightened_worker_timeout_is_not_widened_by_scope():
    bounds = ExecutionBounds(timeout_s=0.02)
    child = Child(stalled=True)
    owner = runner(lambda *a, **k: child, bounds)
    start = time.monotonic()
    try:
        with pytest.raises(ModerationExecutionError, match="deadline_exceeded"):
            owner.execute(request(bounds), scope(1))
        assert time.monotonic() - start < 0.3
    finally:
        owner.close()


def test_shutdown_kills_active_child_and_finishes_registry():
    child, entered = Child(stalled=True), threading.Event()
    owner = runner(lambda *a, **k: (entered.set(), child)[1])
    errors = []

    def run():
        try:
            owner.execute(request(), scope())
        except ModerationExecutionError as error:
            errors.append(error.reason)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(1)
        owner.close()
        thread.join(1)
        assert child.poll() == -9
        assert errors == ["worker_unavailable"]
        assert not owner._entries
    finally:
        owner.close()
        thread.join(3)


def test_launch_is_installed_path_isolated_and_environment_allowlisted(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.Moderation.execution_runner import ModerationExecutionRunner

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "secret")
    monkeypatch.setenv("LD_PRELOAD", "secret")
    owner = ModerationExecutionRunner(ExecutionBounds())
    try:
        response = owner.execute(request(), scope())
        assert response.result == SubstitutionBatchResult("hello", 0, ())
        assert response.resource_limits is not None
        assert not owner._entries
    finally:
        owner.close()


@pytest.mark.parametrize("posture", [None, {"address_space": False, "cpu": False, "core": False}])
def test_success_with_missing_or_unsupported_resource_posture_fails_closed(posture):
    import os

    if os.name != "posix" and posture is not None:
        pytest.skip("Windows does not claim stdlib resource backstops")
    child = Child(
        encode_response(
            WorkerResponse(
                "custom_substitute",
                SubstitutionBatchResult("hello", 0, ()),
                resource_limits=posture,
            )
        )
    )
    owner = runner(lambda *a, **k: child)
    try:
        with pytest.raises(ModerationExecutionError, match="worker_unavailable"):
            owner.execute(request(), scope())
    finally:
        owner.close()


def test_predecode_failure_operation_none_is_sanitized():
    child = Child(encode_response(WorkerResponse(None, reason="worker_unavailable")))
    owner = runner(lambda *a, **k: child)
    try:
        with pytest.raises(ModerationExecutionError, match="worker_unavailable"):
            owner.execute(request(), scope())
    finally:
        owner.close()


def test_nonzero_exit_is_worker_crashed():
    owner = runner(lambda *a, **k: Child(code=12))
    try:
        with pytest.raises(ModerationExecutionError, match="worker_crashed"):
            owner.execute(request(), scope())
    finally:
        owner.close()


def test_io_thread_start_failure_is_typed_and_child_is_reaped(monkeypatch):
    original = threading.Thread.start

    def start(thread):
        if thread.name == "moderation-read":
            raise RuntimeError("thread allocation failed")
        return original(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    child = Child(stalled=True)
    owner = runner(lambda *a, **k: child)
    try:
        with pytest.raises(ModerationExecutionError, match="worker_unavailable"):
            owner.execute(request(), scope())
        assert child.poll() == -9
        assert not owner._entries
    finally:
        owner.close()


def test_close_retries_quarantine_if_reaper_thread_cannot_start(monkeypatch):
    original = threading.Thread.start

    def start(thread):
        if thread.name == "moderation-reaper":
            raise RuntimeError("thread allocation failed")
        return original(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    child = Child(stalled=True, unreapable=True)
    owner = runner(lambda *a, **k: child)
    try:
        with pytest.raises(ModerationExecutionError, match="deadline_exceeded"):
            owner.execute(request(), scope(0.02))
        assert owner._entries
        child.returncode = -9
        owner.close()
        assert not owner._entries
    finally:
        child.returncode = -9
        owner.close()


@pytest.mark.timeout(30)
def test_pathological_re_child_is_killed_and_next_safe_request_succeeds():
    import sys
    import tempfile
    from pathlib import Path

    from tldw_Server_API.tests.helpers.moderation_process_harness import run_isolated_probe

    # Re-prove controller containment before this test's adversarial invocation,
    # even when run alone or reordered by a test runner.
    from tldw_Server_API.tests.unit.test_moderation_process_harness import (
        test_outer_harness_timeout_kills_probe_and_grandchild,
        test_probe_hard_exit_and_invoker_disconnect_still_clean_tree,
    )

    with tempfile.TemporaryDirectory() as directory:
        test_outer_harness_timeout_kills_probe_and_grandchild(Path(directory))
        test_probe_hard_exit_and_invoker_disconnect_still_clean_tree(Path(directory))
    probe = Path(__file__).resolve().parents[1] / "helpers" / "moderation_supervisor_probe.py"
    result = run_isolated_probe([sys.executable, "-I", "-S", str(probe), "pathological"])
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith(b"OK pathological")


@pytest.mark.parametrize("mode", ["cancel-spawn", "fork", "nonzero", "posture", "blocked-write", "redact-many"])
def test_independent_real_child_ownership_probes(mode):
    import os
    import sys
    from pathlib import Path

    from tldw_Server_API.tests.helpers.moderation_process_harness import run_isolated_probe

    if mode == "fork" and os.name != "posix":
        pytest.skip("fork is POSIX only")
    probe = Path(__file__).resolve().parents[1] / "helpers" / "moderation_supervisor_probe.py"
    result = run_isolated_probe([sys.executable, "-I", "-S", str(probe), mode])
    assert result.returncode == 0, result.stderr
    if mode == "redact-many":
        assert b"BATCH_PROVEN 102 launches 1" in result.stdout


def test_fork_during_spawn_waits_for_descriptor_publication():
    import os
    import sys
    from pathlib import Path

    from tldw_Server_API.tests.helpers.moderation_process_harness import run_isolated_probe

    if os.name != "posix":
        pytest.skip("fork is POSIX only")
    probe = Path(__file__).resolve().parents[1] / "helpers" / "moderation_supervisor_probe.py"
    result = run_isolated_probe([sys.executable, "-I", "-S", str(probe), "fork-during-spawn"])
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith(b"OK fork-during-spawn")
