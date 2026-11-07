"""Stdlib-only real-child checks; invoke exclusively through the tree harness."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import ModuleType


def bootstrap(root: Path) -> None:
    """Use only the installed minimal graph, without starting the application."""
    for package in (
        "tldw_Server_API",
        "tldw_Server_API.app",
        "tldw_Server_API.app.core",
        "tldw_Server_API.app.core.Moderation",
        "tldw_Server_API.app.core.Web_Scraping",
    ):
        module = ModuleType(package)
        module.__path__ = [str(root.joinpath(*package.split(".")))]
        sys.modules[package] = module


def run(mode: str) -> None:
    from tldw_Server_API.app.core.exception_types import ModerationExecutionError
    from tldw_Server_API.app.core.Moderation.async_execution import _run_moderation_offload
    from tldw_Server_API.app.core.Moderation.execution_models import (
        ExecutionBounds,
        SubstitutionBatchResult,
        SubstitutionRuleSnapshot,
        WorkerResponse,
        execution_scope,
    )
    from tldw_Server_API.app.core.Moderation.execution_protocol import encode_response, snapshot_substitution_request
    from tldw_Server_API.app.core.Moderation.execution_runner import ModerationExecutionRunner

    children = []
    entered = threading.Event()
    publish = threading.Event()

    def spawn(*args, **kwargs):
        if not children and mode in {"nonzero", "posture", "blocked-write"}:
            if mode == "nonzero":
                code = "import os;os.write(2,b'withheld');raise SystemExit(23)"
            elif mode == "blocked-write":
                code = "import time;time.sleep(60)"
            else:
                posture = {"address_space": False, "cpu": False, "core": False} if os.name == "posix" else None
                data = encode_response(
                    WorkerResponse(
                        "custom_substitute",
                        SubstitutionBatchResult("hello", 0, ()),
                        resource_limits=posture,
                    )
                )
                code = f"import os;os.write(1,{data!r})"
            args = ([sys.executable, "-I", "-S", "-c", code],)
        child = subprocess.Popen(*args, **kwargs)
        if mode == "nonzero" and not children:
            child.wait(timeout=1)
        children.append(child)
        entered.set()
        if mode == "fork-during-spawn" and len(children) == 1:
            assert publish.wait(1), "factory publication was not released"
        if mode == "cancel-spawn":
            time.sleep(0.15)
        return child

    bounds = ExecutionBounds()
    owner = ModerationExecutionRunner(bounds, process_factory=spawn)
    safe = snapshot_substitution_request("hello", [], None, bounds)

    def assert_clean():
        assert children
        assert all(child.poll() is not None for child in children)
        assert not owner._entries
        assert all(child.stdin.closed and child.stdout.closed for child in children)

    try:
        if mode == "redact-many":
            import re

            from tldw_Server_API.app.core.Moderation.execution_protocol import snapshot_redaction_batch_request
            from tldw_Server_API.app.core.Moderation.models import ModerationPolicy
            from tldw_Server_API.app.core.Moderation.policy_evaluator import EvaluationLimits

            texts = [" ".join(["secret"] * 100)] + ["secret"] * 100 + [""]
            request = snapshot_redaction_batch_request(
                texts, ModerationPolicy(block_patterns=[re.compile("^secret$")], redact_replacement="[PII]"),
                None, EvaluationLimits(200000, 4096, 800000, 1000), bounds,
            )
            started = time.monotonic()
            with execution_scope() as scope:
                response = owner.execute(request, scope)
            assert response.result == [texts[0]] + ["[PII]"] * 100 + [""]
            assert len(children) == 1
            assert_clean()
            print("BATCH_PROVEN", len(texts), "launches", len(children), "elapsed_ms", round((time.monotonic() - started) * 1000, 3))
        elif mode in {"pathological", "heartbeat"}:
            # Only plain text is handled here. Compilation and adversarial
            # matching occur exclusively inside the owned regex worker child.
            hostile = snapshot_substitution_request(
                "a" * 100000 + "!",
                [SubstitutionRuleSnapshot("(a+)+$", 0, "x")],
                None,
                bounds,
            )
            start = time.monotonic()
            if mode == "pathological":
                with execution_scope() as scope:
                    try:
                        owner.execute(hostile, scope)
                    except ModerationExecutionError as error:
                        assert error.reason == "deadline_exceeded"
                    else:
                        raise AssertionError("hostile request succeeded")
            else:

                async def heartbeat():
                    with execution_scope() as scope:
                        task = asyncio.create_task(
                            _run_moderation_offload(lambda: owner.execute(hostile, scope), scope=scope)
                        )
                        ticks = 0
                        while not task.done():
                            await asyncio.sleep(0.01)
                            ticks += 1
                        try:
                            await task
                        except ModerationExecutionError as error:
                            assert error.reason == "deadline_exceeded"
                        else:
                            raise AssertionError("hostile request succeeded")
                        assert ticks >= 20
                    end = time.monotonic() + 1
                    while owner._entries and time.monotonic() < end:
                        await asyncio.sleep(0.01)

                asyncio.run(heartbeat())
            assert time.monotonic() - start < 3.5
            assert_clean()
        elif mode in {"nonzero", "posture", "blocked-write"}:
            # The large blocked-write frame must reach the pipe, not exhaust
            # an artificial 100 ms scope during serialization on a busy host.
            with execution_scope() as scope:
                submitted = (
                    snapshot_substitution_request("x" * 1000000, [], None, bounds) if mode == "blocked-write" else safe
                )
                try:
                    owner.execute(submitted, scope)
                except ModerationExecutionError as error:
                    assert (
                        error.reason
                        == {
                            "nonzero": "worker_crashed",
                            "posture": "worker_unavailable",
                            "blocked-write": "deadline_exceeded",
                        }[mode]
                    )
                else:
                    raise AssertionError("invalid child accepted")
            assert_clean()
        elif mode == "cancel-spawn":

            async def cancel():
                with execution_scope() as scope:
                    task = asyncio.create_task(_run_moderation_offload(lambda: owner.execute(safe, scope), scope=scope))
                    while not entered.is_set():
                        await asyncio.sleep(0.001)
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    else:
                        raise AssertionError("cancellation was swallowed")
                    assert scope.cancelled.is_set()
                    end = time.monotonic() + 1
                    while owner._entries and time.monotonic() < end:
                        await asyncio.sleep(0.01)

            asyncio.run(cancel())
            assert_clean()
        elif mode == "fork-during-spawn":
            read_fd, write_fd = os.pipe()
            fork_requested = threading.Event()
            results, errors = [], []

            def execute():
                try:
                    with execution_scope() as scope:
                        results.append(owner.execute(safe, scope).result.text)
                except BaseException as error:  # noqa: BLE001 - propagate probe failure to the main owner.
                    errors.append(getattr(error, "reason", type(error).__name__))

            def release_publication():
                assert fork_requested.wait(1)
                time.sleep(0.1)
                publish.set()

            execution = threading.Thread(target=execute)
            publisher = threading.Thread(target=release_publication)
            execution.start()
            assert entered.wait(1)
            publisher.start()
            fork_requested.set()
            pid = os.fork()
            if pid == 0:
                os.close(write_fd)
                try:
                    # The parent must succeed BEFORE this forked API process
                    # exits, despite descriptors inherited while spawn paused.
                    assert os.read(read_fd, 1) == b"x"
                    assert owner._pid == os.getpid()
                    with execution_scope() as scope:
                        assert owner.execute(safe, scope).result.text == "hello"
                    owner.close()
                except BaseException:  # noqa: BLE001 - fixed child failure status inside outer containment.
                    os._exit(1)
                os._exit(0)
            os.close(read_fd)
            try:
                execution.join(3)
                assert not execution.is_alive()
                assert results == ["hello"], errors
            finally:
                publish.set()
                os.write(write_fd, b"x")
                os.close(write_fd)
                publisher.join(1)
                execution.join(1)
                assert os.waitpid(pid, 0)[1] == 0
            assert_clean()
        elif mode == "fork":
            import tldw_Server_API.app.core.Moderation.async_execution as transport

            # Fork with inherited admission/lock state; neither child may use
            # parent-held locks or count the vanished parent's thread leases.
            owner._lock.acquire()
            transport._lock.acquire()
            pid = os.fork()
            if pid == 0:
                try:
                    with execution_scope() as scope:
                        assert owner.execute(safe, scope).result.text == "hello"

                    async def offload():
                        with execution_scope() as scope:
                            assert await _run_moderation_offload(lambda: 42, scope=scope) == 42

                    asyncio.run(offload())
                    owner.close()
                except BaseException:  # noqa: BLE001 - hard-exit with fixed failure status, never an unowned child.
                    os._exit(1)
                os._exit(0)
            transport._lock.release()
            owner._lock.release()
            assert os.waitpid(pid, 0)[1] == 0
        with execution_scope() as scope:
            response = owner.execute(safe, scope)
            assert response.result.text == "hello"
        assert_clean()
        assert "regex" not in sys.modules
        print("OK", mode, sys.platform, sys.version.split()[0], response.resource_limits)
    finally:
        owner.close()


if __name__ == "__main__":
    bootstrap(Path(__file__).resolve().parents[3])
    run(sys.argv[1])
