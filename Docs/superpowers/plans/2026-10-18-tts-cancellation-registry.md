# TTS Cancellation Safety: Engine Locks and Slow Adapter Loads Implementation Plan

Independent review and dry run (2026-10-18): 0 Blocker, 1 Important, 8 Minor; all folded in, plus the fallback-path bound (author's ruling).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A cancelled TTS request (client disconnect, capability-listing timeout) no longer hands a local engine to the next request while its worker-thread call still runs, and no longer cancels, restarts or fails a slow adapter load.

**Architecture:** TASK-13856 lifts PR #3264's completion-released lock (`NeuTTSAdapter._engine_call` / `_release_engine_when_done`) into one async context manager, `hold_tts_engine(lock)` in `core/TTS/utils.py`, and adopts it at every lock held around a thread-offloaded call in the local adapters. TASK-13857 shields `TTSAdapterRegistry.get_adapter`'s initialization from its callers' cancellation, stops the capability listing from marking a timed-out provider failed, bounds every non-requesting caller (`get_all_capabilities`, `list_capabilities`, `find_adapter_for_requirements`, the service's `_get_fallback_adapter`) with `_CAPABILITY_LISTING_TIMEOUT_SECONDS` so none of them waits out another provider's load (Review I1 and the author's fallback ruling), and then moves NeuTTS's vendor import and model load to a worker thread.

**Tech Stack:** Python 3.12 asyncio, anyio cancel scopes (Starlette), pytest + pytest-asyncio, backlog-py.

**Spec:** the two backlog tasks, filed in PR #3264 (`fix/neutts-offload-blocking-calls`, head `1bfd6951af`, OPEN when this plan was written); they land on dev when it merges:
- `backlog/tasks/task-13856 - Local-TTS-adapters-hand-the-engine-to-the-next-request-while-a-cancelled-request-s-call-still-runs.md`
- `backlog/tasks/task-13857 - TTS-capability-listing-cancels-slow-local-model-loads-and-marks-the-provider-failed.md`
- Background: `backlog/tasks/task-13855 - NeuTTS-synthesis-blocks-the-server-s-event-loop.md` (the pattern being lifted, and the anyio finding).

## Base, PR split and sequencing

- **Base: origin/dev after PR #3264 merges.** Both tasks' files, and the NeuTTS code Task A2 and Task B2 edit, exist only on #3264's branch until then. Every task starts by confirming `gh pr view 3264 --json state --jq .state` prints `MERGED`. If #3264 changed after `1bfd6951af`, re-read `neutts_adapter.py` before Task A2 and Task B2.
- **Two PRs, both targeting `dev`:**
  - **PR A, TASK-13856** (branch `fix/tts-engine-lock-cancellation`): Tasks A1-A5. One new helper and 11 adapter files.
  - **PR B, TASK-13857** (branch `fix/tts-registry-slow-load`): Tasks B1-B3. The registry and NeuTTS's load.
  - Why two: they fix independent defects with different blast radius. PR B changes `get_adapter` for every TTS provider and rewrites a pinned registry test; PR A changes 11 local adapters. Neither needs the other to be correct (PR B's NeuTTS load uses `run_tts_blocking_call`, not the new helper), and each gets its own clean review.
- **Merge PR B first** (small, and fixes a user-visible failure: a local provider whose load outlasts 5 s is marked failed for 300 s and real requests get no adapter). Both can be developed in parallel.
- **The one cross-PR hazard is silent, not a text conflict:** both PRs touch `neutts_adapter.py`'s `from ..utils import ...` line differently, in hunks git may merge cleanly. After both merge, the line must read `from ..utils import hold_tts_engine, parse_bool, run_tts_blocking_call`. Task A2 Step 3 and Task B2 Step 3 each say how to get there whichever PR lands first; ruff F821/F401 and the NeuTTS tests catch a miss.

## Measured facts (origin/dev `e1f7770953`; NeuTTS on #3264 head `1bfd6951af`)

Verification of every site TASK-13856 lists, plus the same shape found elsewhere (`grep -rnE "Lock\(\)|async with .*lock|to_thread|run_in_executor" tldw_Server_API/app/core/TTS`):

| Site | Lock held around | Verdict |
|---|---|---|
| `pocket_tts_adapter.py:317-323` (generate) | `asyncio.to_thread(self._engine.generate, ...)` | engine call, fix |
| `pocket_tts_adapter.py:523-536` (stream) | `loop.run_in_executor(None, lambda: next(generator, sentinel))` per chunk, lock held for the whole stream | engine call, fix |
| `luxtts_adapter.py:298-314` | two calls, `encode_prompt` then `generate_speech` (the task said 298-305) | engine call, fix |
| `supertonic_adapter.py:313-316` | `asyncio.to_thread(self._engine, ...)` | engine call, fix |
| `supertonic2_adapter.py:322-325` | `asyncio.to_thread(self._engine, ...)` | engine call, fix |
| `omnivoice_runtime.py:113-115` (`_generate_lock`) | `asyncio.to_thread(model.generate, ...)` | engine call, fix |
| `omnivoice_runtime.py:59-85` (`_load_lock`) | `asyncio.to_thread(omnivoice_cls.from_pretrained, ...)` | model load, fix |
| `qwen3_runtime_mlx.py:133-136` (`_model_lock`) | model load only; generation (`:194`) runs with no lock | model load, fix |
| `qwen3_tts_adapter.py:638-641` (`_pipeline_lock`) | pipeline load only; generation (`_invoke_backend`, `:740`) and the stream worker (`:1020`) run with no lock | model load, fix |
| `vibevoice_adapter.py:963-1059` (`_model_state_lock`), **not listed** | variant reload, input prep, then `run_tts_blocking_call(_generate_outputs)`; a cancelled generate lets the next request's variant reload (`_cleanup_resources`) free the model under the running thread | engine call, fix |
| `echo_tts_adapter.py:479-517` (`_model_lock`), **not listed** | three `asyncio.to_thread` loads | model load, fix |
| `kitten_tts_adapter.py:94-110` (`_runtime_lock`), **not listed** | `download_model_assets` then `KittenRuntime` | model load, fix |
| `kokoro_adapter.py:359-370` (`_model_lock`), **not listed** | `_load_onnx_model` / `_load_pytorch_model`; `asyncio.wait_for(..., init_timeout_sec)` (`:366`, `:389`) abandons its own running load thread at the timeout, and the lock passes to the next caller, which starts a second load. (Not a two-locks race: with lazy init `initialize()` does not load, `:375-382`; without it `generate()` waits on `ensure_initialized()` at `:770` before `_ensure_model_loaded()` at `:776`, which then returns early.) (Review M6) | **follow-up task** (Task A5): the timeout, not a lock swap |
| `adapters/base.py:412` (`TTSAdapter._init_lock` around `initialize()`) | adapters whose `initialize()` offloads the load | covered by TASK-13857: after PR B the registry no longer cancels it |
| `supertonic_adapter.py:101` -> `:147`, `supertonic2_adapter.py:97` -> `:143` (`ensure_initialized` overrides with their own `async with self._init_lock`) | `asyncio.to_thread(vendor.load_text_to_speech, ...)` | same shape and verdict as `base.py:412`: covered by TASK-13857 for registry-built adapters (Review M5) |
| `audio_cpp_adapter.py:177,220` (`_generation_lock`) | HTTP calls to a sidecar, no thread offload | not this shape |
| `index_tts_adapter.py:290,427-437`; `qwen3_runtime_mlx.py:194`; `qwen3_tts_adapter.py:1020` | thread-offloaded engine calls with **no lock** | not this shape; Review Focus |

Registry (TASK-13857): `get_all_capabilities` waits `asyncio.wait_for(self.get_adapter(provider_key), timeout=5.0)` (`adapter_registry.py:1023-1026`) and on timeout calls `_schedule_retry` (`:1031-1034`) when `adapter_failure_retry_seconds` is set (300 in `Config_Files/tts_providers_config.yaml:629`). `get_adapter` awaits `self._base.get_adapter_async(provider_key)` unshielded (`:655`), which holds the base's per-provider `asyncio.Lock` (`Infrastructure/provider_registry.py:520-521`) across `_initialize_adapter` → `adapter.ensure_initialized()`. The timeout cancels that chain: `_initialize_adapter`'s `finally` closes the half-built adapter (`:892-901`), the base lock passes to the next caller, which starts a second load.

Once `get_adapter` is shielded, every other caller that iterates providers would wait out a load instead of skipping a provider the timeout had marked failed (measured with a 3 s load and a 0.2 s listing timeout: `get_all_capabilities()` then `list_capabilities()`, the order of `audio_health.py:516` and `:538`, took `0.20s | 0.00s` on dev and `0.20s | 2.80s` on PR B without a bound) (Review I1):
- `list_capabilities` awaits `get_adapter` unbounded for every ENABLED provider (`adapter_registry.py:1060-1063`); the WebUI polls GET /audio/health every 60 s (`useTldwAudioStatus.tsx:161-174`).
- `find_adapter_for_requirements` (`:1119`, reached through `TTSAdapterFactory.get_best_adapter`, `:1548`) and `TTSServiceV2._get_fallback_adapter` (`tts_service_v2.py:3712`; its provider loop awaits `factory.registry.get_adapter(provider)`) pick a failover provider the same way.
Task B1 bounds all four with `_CAPABILITY_LISTING_TIMEOUT_SECONDS`; on timeout they skip that provider without marking it failed.

Existing tests that pin the old behaviour (both updated by this plan):
- `tests/TTS/test_tts_adapter_registry_wrapper_migration.py::test_registry_closes_adapter_and_reraises_initialization_cancellation` cancels a `get_adapter` caller and expects the adapter closed. After PR B a caller's cancellation no longer reaches the initialization; Task B1 retargets the test at `_initialize_adapter`, which loop shutdown still cancels.
- `tests/TTS/unit/test_local_adapter_async_offload.py:36,49` asserts `vibevoice_adapter.run_tts_blocking_call` exists and that `_stream_audio_vibevoice` references it. After PR A VibeVoice offloads through `hold_tts_engine`; Task A3 updates both lines.

Baseline on unchanged origin/dev (no PyAV, no torch locally); these 7 fail before and after both PRs:
`tests/TTS/adapters/test_luxtts_adapter_mock.py::test_luxtts_generate_non_stream`, `tests/TTS/adapters/test_supertonic_adapter_mock.py::test_supertonic_initialize_and_generate`, `tests/TTS/adapters/test_supertonic2_adapter_mock.py::test_supertonic2_initialize_and_generate`, `tests/TTS/test_tts_module.py::TestStreamingAudioWriter::test_init_valid_formats`, `tests/TTS/adapters/test_higgs_adapter_integration_stub.py::test_voice_reference_integration_message_injection`, `tests/TTS/adapters/test_higgs_adapter_integration_stub.py::test_multispeaker_and_language_hints_in_chatml_stub`, `tests/TTS_NEW/unit/test_tts_service.py::TestTextGeneration::test_chunked_retry_metadata`.

Probed in scratch worktrees (this plan's code, applied to #3264's head): PR A's 24 new cases are RED on dev with `assert 2 == 1` and GREEN after; PR B's 6 new registry-file cases RED (`assert 2 == 1` twice, `failed == enabled` once, and three `TimeoutError` from the tests' own `wait_for` for the three bounded paths) and GREEN; mutations listed in Task A2 and Task B1 each fail a test.

## Global Constraints

- Every PR's final sweep includes `tldw_Server_API/tests/CI/test_test_quality_ratchet.py`: since #3243, dev enforces it, so new tests must not be status-only or tautological.
- `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`. Tests: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q <paths>`.
- Branch from origin/dev after #3264 merges; PRs target `dev`.
- Fix defects test-first: each fix has a test that fails on dev for the stated reason before the code changes.
- Backlog edits only through backlog-py: `PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task ...`; `--append-notes`, never `--notes`; stage with `git add -A backlog/tasks`.
- Commit trailer `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`; never `--no-verify`; never `git stash`.
- No docs change: no document describes the engine locks or the listing timeout (`git grep -n -E "run_tts_blocking|_engine_lock|capabilit.*timeout" -- Docs` finds only `Docs/Product/Completed/TTS_Supertonic_ONNX.md`, a historical spec), so `Docs/Published` is untouched.
- No file this plan touches is listed in `Helper_Scripts/ci/scope_predicate_baseline.txt` (checked: no `core/TTS/` entries).
- Bandit is not installed in the venv; run it as `uvx bandit -q -ll <files>` (do not install it into the venv).

## ADR assessment

No ADR needed for either PR. Both are bug fixes inside the TTS module and change no durable rule. ADR-011 (`Docs/ADR/011-audio-api-semantics.md`) governs adapter initialization failures ("tracked in the provider registry and can be retried after a configured cooldown"); PR B keeps that for real failures and stops recording a still-running initialization as one, which is consistent with it. ADR-021 (services lifecycle) is not touched: the in-process TTS jobs worker calls the same adapters.

## Decisions for the owner

1. **ADR-004 Change summary for PR A and PR B.** Both PRs are AI-authored, and the recorded waiver lists earlier tasks only (through TASK-13848/13849). Recommendation: extend the waiver to TASK-13856 and TASK-13857, as for the other TTS follow-ups; otherwise write the two Change summaries. Merge waits on this.

No product decisions: scope choices (the three unlisted sites in, Kokoro to a follow-up) follow the task's own criterion ("each local adapter that holds a lock around a thread-offloaded engine call").

## Review Focus

Failure modes the tests below do not exercise:

1. **A model load whose holder is cancelled is discarded.** For the load sites (OmniVoice, Qwen3 MLX, Qwen3 pipeline, Echo-TTS, KittenTTS), the next caller waits for the cancelled load to return and then loads again: serial, not concurrent, but a disconnect during a 60 s load costs the next request up to 120 s. Marked with a `ponytail:` comment in the helper; storing the model from the thread is the upgrade.
2. **A stream closed by its consumer (`aclose()`, or garbage collection of an abandoned generator) rather than cancelled.** The lock is released when the `async with` exits at the `yield`, with no thread call running then; an abandoned, unclosed generator still holds the lock until it is finalized, as on dev.
3. **A load that never returns.** After PR B the listing no longer marks a provider failed after 5 s, so requests that name a provider whose load hangs wait on it (as requests that reached it before the listing already did on dev; TASK-13857 AC1: later callers join the running initialization). Callers that only want some ready provider do not wait: `get_all_capabilities`, `list_capabilities` (GET /audio/health, /audio/providers), failover and best-adapter selection (`_get_fallback_adapter`, `find_adapter_for_requirements`) are bounded by `_CAPABILITY_LISTING_TIMEOUT_SECONDS` and skip a still-loading provider without marking it failed, so failover picks a provider that is ready now. A hung load itself is never timed out here; that would be a separate change. (Review I1, author's ruling)
4. **Engine calls that hold no lock at all** (IndexTTS generation and stream, Qwen3 MLX generation, the Qwen3 stream worker that keeps running after cancellation): out of scope here; this plan changes only where a lock exists.
5. **`close_all()` or `unload_provider()` while a shielded initialization still runs.** On dev nothing cancelled such an initialization either; the generation check in `_initialize_adapter` already closes a stale adapter (covered by the existing reregister/unload-during-initialization tests), but `close_all()` does not wait for an in-flight load.
6. **NeuTTS loads that bypass the registry can overlap.** `voice_manager._encode_neutts_reference` (`voice_manager.py:1245-1246`) builds its own `NeuTTSAdapter` and calls `ensure_initialized()` unshielded, and `create_adapter_with_overrides` (`adapter_registry.py:~728`) does the same for per-request override adapters. Before Task B2 a NeuTTS load blocked the loop, so two loads could never overlap; after it they can overlap the registry's shielded load, and a cancelled caller on these paths leaves its load running on the thread. The result can be two NeuTTS models resident at once; each path uses its own engine, so nothing is incorrect. (Pre-existing and out of scope: `voice_manager.py:1247` still calls `encode_reference` on the loop.) (Review M4)

---

## PR A: TASK-13856

### Task A1: The shared helper, adopted by PocketTTS

**Files:**
- Create: `tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py`
- Modify: `tldw_Server_API/app/core/TTS/utils.py:11-20` (imports) and after `:116-125` (`run_tts_blocking_next`)
- Modify: `tldw_Server_API/app/core/TTS/adapters/pocket_tts_adapter.py:30`, `:317-323`, `:520-536`

**Interfaces:**
- Produces: `hold_tts_engine(lock: asyncio.Lock)`, an async context manager yielding `run`, where `await run(func, /, *args, **kwargs)` runs `func(*args, **kwargs)` on a worker thread (through `run_tts_blocking_call`) and returns its result. The lock is released when the `async with` exits, or, if the holder is leaving while its last `run` call still runs on its thread, when that call returns. A stream pulls each chunk with `await run(next, iterator, sentinel)`.
- Produces (tests): `_Gate` (`.started: list[str]`, `.release: threading.Event`, `.enter(name)`), `_assert_cancelled_call_keeps_the_engine(call, gate, *, via_anyio_scope)`, `CANCEL_MODES`, `_WAV_HEADER` in the new test module; Tasks A3 and A4 append to it.

- [ ] **Step 1: Branch and mark the task In Progress**

```bash
gh pr view 3264 --json state --jq .state   # must print MERGED
git fetch origin dev --quiet
git switch -c fix/tts-engine-lock-cancellation origin/dev
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13856 -s "In Progress"
```

- [ ] **Step 2: Write the failing tests**

Create `tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py`:

```python
"""A cancelled request keeps a local TTS engine until its thread call returns (TASK-13856).

asyncio.to_thread cannot stop a thread, so an adapter that guards its engine with a lock must not
hand the engine to the next request while a cancelled request's call still runs. Starlette cancels
a disconnected request through an anyio cancel scope, which re-cancels on every await, so each
site is checked with a plain task cancel and with an anyio scope.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable

import anyio
import numpy as np
import pytest

from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, ProviderStatus, TTSRequest

pytestmark = pytest.mark.unit

CANCEL_MODES = pytest.mark.parametrize("via_anyio_scope", [False, True], ids=["task-cancel", "anyio-scope"])


class _Gate:
    """Engine calls record that they started, then block until released, so a caller can be cancelled mid-call."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.release = threading.Event()

    def enter(self, name: str) -> None:
        self.started.append(name)
        assert self.release.wait(5), "the test never released the engine call"


async def _assert_cancelled_call_keeps_the_engine(
    call: Callable[[], Awaitable[object]],
    gate: _Gate,
    *,
    via_anyio_scope: bool,
) -> None:
    """Cancel one call mid-engine-call while a second waits; the second must not start until the first returns."""
    scopes: list[anyio.CancelScope] = []

    async def _first() -> None:
        if not via_anyio_scope:
            await call()
            return
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            await call()

    first = asyncio.create_task(_first())
    try:
        for _ in range(500):
            if gate.started:
                break
            await asyncio.sleep(0.01)
        assert gate.started, "the first call never reached the engine"
        second = asyncio.create_task(call())
        await asyncio.sleep(0.05)
        if via_anyio_scope:
            scopes[0].cancel()
        else:
            first.cancel()
        await asyncio.sleep(0.1)
        # The cancelled request's engine call is still running, so the next one has not started.
        assert len(gate.started) == 1
    finally:
        gate.release.set()
    # Bounded waits: a lock-leak regression must fail here, not hang the run until pytest-timeout kills it (Review M1).
    await asyncio.wait_for(second, 5)
    assert len(gate.started) == 2
    if via_anyio_scope:
        await asyncio.wait_for(first, 5)
    else:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first, 5)


_WAV_HEADER = b"RIFF\x24\x00\x00\x00WAVEfmt "


def _pocket_adapter(gate: _Gate):
    from tldw_Server_API.app.core.TTS.adapters.pocket_tts_adapter import PocketTTSOnnxAdapter

    class _Engine:
        def generate(self, text, voice, max_frames=500):
            gate.enter("generate")
            return np.zeros(240, dtype=np.float32)

        def stream(self, text, voice, **_kwargs):
            gate.enter("stream")
            yield np.zeros(240, dtype=np.float32)

    adapter = PocketTTSOnnxAdapter({})
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE
    adapter._engine = _Engine()
    return adapter


def _pocket_request(*, stream: bool) -> TTSRequest:
    return TTSRequest(
        text="hello",
        voice="clone",
        format=AudioFormat.PCM,
        stream=stream,
        voice_reference=_WAV_HEADER,
        extra_params={"validate_reference": False, "convert_reference": False},
    )


@pytest.mark.asyncio
@CANCEL_MODES
async def test_pocket_tts_generate_keeps_the_engine_until_its_call_returns(via_anyio_scope):
    gate = _Gate()
    adapter = _pocket_adapter(gate)

    async def call():
        return await adapter.generate(_pocket_request(stream=False))

    await _assert_cancelled_call_keeps_the_engine(call, gate, via_anyio_scope=via_anyio_scope)


@pytest.mark.asyncio
@CANCEL_MODES
async def test_pocket_tts_stream_keeps_the_engine_until_its_call_returns(via_anyio_scope):
    gate = _Gate()
    adapter = _pocket_adapter(gate)

    async def call():
        response = await adapter.generate(_pocket_request(stream=True))
        return [chunk async for chunk in response.audio_stream]

    await _assert_cancelled_call_keeps_the_engine(call, gate, via_anyio_scope=via_anyio_scope)
```

- [ ] **Step 3: Run them to see them fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py`
Expected: `4 failed`, each with `AssertionError: assert 2 == 1` (`['generate', 'generate']` or `['stream', 'stream']`): the cancelled request's `async with self._engine_lock` released the lock while its thread still ran, so the second request's engine call started.

- [ ] **Step 4: Add the helper to `core/TTS/utils.py`**

Replace the stdlib import block at the top (lines 11-20, `import asyncio` through `from collections.abc import Callable, Iterator`) with:

```python
import asyncio
import contextlib
import functools
import hashlib
import hmac
import json
import math
import os
import platform
import re
import unicodedata
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
```

Insert after `run_tts_blocking_next` (after line 125, before `def estimate_max_new_tokens`):

```python
@contextlib.asynccontextmanager
async def hold_tts_engine(lock: asyncio.Lock) -> AsyncIterator[Callable[..., Awaitable[Any]]]:
    """Hold a local TTS engine's lock, yielding ``run``, which makes one engine call on a worker thread.

    ``asyncio.to_thread`` cannot stop a thread, so when the holder is cancelled mid-call (a client
    disconnect), the lock passes to the next caller only once that call returns. Waiting for the
    thread inside the cancelled holder does not work: anyio re-cancels the wait. A stream holds the
    lock across chunks by pulling each one with ``await run(next, iterator, sentinel)``.
    """
    await lock.acquire()
    last_call: asyncio.Future | None = None

    async def run(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        nonlocal last_call
        last_call = asyncio.ensure_future(run_tts_blocking_call(func, *args, **kwargs))
        return await asyncio.shield(last_call)

    try:
        yield run
    finally:
        if last_call is None or last_call.done():
            lock.release()
        else:
            # ponytail: a cancelled model load's result is discarded, so the next holder loads again
            # (after it, not alongside it); store the model from the thread if that cost matters.
            last_call.add_done_callback(functools.partial(_release_engine_lock, lock))


def _release_engine_lock(lock: asyncio.Lock, call: asyncio.Future) -> None:
    """Release a cancelled holder's engine lock once its engine call returns."""
    lock.release()
    if not call.cancelled():
        call.exception()  # its caller is gone; don't let asyncio log the error as never retrieved
```

The `finally` must not `await`: anyio re-cancels a cancelled holder at its next await, which is why TASK-13855 releases from the call's completion.

- [ ] **Step 5: Adopt it in PocketTTS**

In `pocket_tts_adapter.py`, line 30:

```python
from ..utils import hold_tts_engine, parse_bool
```

Lines 317-323, replace `async with self._engine_lock:` / `audio = await asyncio.to_thread(` with:

```python
            async with hold_tts_engine(self._engine_lock) as run:
                audio = await run(
                    self._engine.generate,  # type: ignore[union-attr]
                    request.text,
                    voice_path,
                    self._resolve_max_frames(extras),
                )
```

In `stream()` (lines 520-536): delete `loop = asyncio.get_running_loop()` (line 521), change `async with self._engine_lock:` (line 523) to `async with hold_tts_engine(self._engine_lock) as run:`, and replace the three-line `chunk = await loop.run_in_executor(None, lambda: next(generator, sentinel))` with:

```python
                        chunk = await run(next, generator, sentinel)
```

- [ ] **Step 6: Run the new tests and PocketTTS's own tests**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py tldw_Server_API/tests/TTS/adapters/test_pocket_tts_adapter_mock.py`
Expected: all pass (`4 passed` from the new file; the PocketTTS mock file has no baseline failures).

- [ ] **Step 7: Lint and commit**

```bash
$PY -m ruff check tldw_Server_API/app/core/TTS/utils.py tldw_Server_API/app/core/TTS/adapters/pocket_tts_adapter.py tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py
git add tldw_Server_API/app/core/TTS/utils.py tldw_Server_API/app/core/TTS/adapters/pocket_tts_adapter.py tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py
git add -A backlog/tasks
git commit -m "fix(tts): a cancelled PocketTTS request keeps the engine until its call returns (TASK-13856)

hold_tts_engine (core/TTS/utils.py) releases a local engine's lock from the
thread call's completion when its holder is cancelled mid-call, as NeuTTS
does since TASK-13855.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Expected: ruff `All checks passed!`.

### Task A2: NeuTTS takes the engine through the helper

**Files:**
- Modify: `tldw_Server_API/app/core/TTS/adapters/neutts_adapter.py:34`, `:364-385` (stream), `:411-435` (`_engine_call`, `_release_engine_when_done`); line numbers on #3264's head.

**Interfaces:**
- Consumes: `hold_tts_engine` (Task A1).
- Produces: `NeuTTSAdapter._engine_call(func, *args)` unchanged in signature; `_release_engine_when_done` is deleted.

This is a refactor: #3264's NeuTTS tests already pin the behaviour and must stay green.

- [ ] **Step 1: Record the green baseline**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/TTS/adapters/test_neutts_adapter.py`
Expected: `20 passed`.

- [ ] **Step 2: Replace the stream's hand-rolled lock (lines 364-385)**

Replace from `# The generator drives the engine between chunks, ...` down to `self._release_engine_when_done(call)` with:

```python
                # The generator drives the engine between chunks, so the stream holds the lock until it ends.
                async with hold_tts_engine(self._engine_lock) as run:
                    chunks = self._engine.infer_stream(request.text, ref_codes, ref_text)  # type: ignore
                    done = object()
                    while True:
                        chunk = await run(next, chunks, done)
                        if chunk is done:
                            break
                        if chunk is None or len(chunk) == 0:
                            continue
                        chunk_idx += 1
                        # Normalize float32 -> int16, then encode
                        normalized = self.audio_normalizer.normalize(chunk, target_dtype=np.int16)
                        encoded = writer.write_chunk(normalized)
                        if encoded:
                            yield encoded
```

Replace `_engine_call` and delete `_release_engine_when_done` (lines 411-435) with:

```python
    async def _engine_call(self, func: Callable[..., Any], *args: Any) -> Any:
        """Run one engine call on a worker thread, holding the engine lock until it returns."""
        async with hold_tts_engine(self._engine_lock) as run:
            return await run(func, *args)
```

- [ ] **Step 3: Fix the import line**

Run: `grep -n "run_tts_blocking" tldw_Server_API/app/core/TTS/adapters/neutts_adapter.py`
- If the only hit is line 34 (PR B not merged yet), set line 34 to `from ..utils import hold_tts_engine, parse_bool`.
- If it also prints `self._engine = await run_tts_blocking_call(self._load_engine)` (PR B merged), set it to `from ..utils import hold_tts_engine, parse_bool, run_tts_blocking_call`.

- [ ] **Step 4: Run the NeuTTS tests and lint**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/TTS/adapters/test_neutts_adapter.py && $PY -m ruff check tldw_Server_API/app/core/TTS/adapters/neutts_adapter.py`
Expected: `20 passed` (`21 passed` if PR B has merged) and `All checks passed!`.

- [ ] **Step 5: Confirm the helper is what the tests depend on (mutation check, revert afterwards)**

In `utils.py`, temporarily replace the whole `if last_call is None or last_call.done(): ... else: ...` block inside the helper's `finally` with `lock.release()`; run `test_tts_engine_lock_cancellation.py` and `test_neutts_adapter.py`. Expected: the 4 PocketTTS cases plus `test_neutts_cancelled_request_keeps_the_engine_until_its_call_returns[generate|stream|stream-anyio-scope]` and `test_neutts_cancelled_request_engine_error_is_not_logged_raw` fail. Restore with `git checkout -- tldw_Server_API/app/core/TTS/utils.py` (utils.py was committed in Task A1). Probed results for the full file after Task A4: releasing at cancel fails all 24 new cases and those 4 NeuTTS tests; awaiting the thread in the `finally` and releasing in a `try/finally` (the variant TASK-13855 rejected) fails exactly the 12 `anyio-scope` cases plus the NeuTTS anyio and error-log tests; awaiting the thread in the `finally` and then releasing without a `try/finally` leaks the lock when anyio re-cancels the wait, which the bounded `wait_for` calls in `_assert_cancelled_call_keeps_the_engine` turn into a failure instead of a hang (before them it hung until pytest-timeout killed the process; Review M1); dropping `call.exception()` fails only `test_neutts_cancelled_request_engine_error_is_not_logged_raw`.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/TTS/adapters/neutts_adapter.py
git commit -m "refactor(tts): NeuTTS holds its engine through hold_tts_engine (TASK-13856)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task A3: Engine calls in LuxTTS, Supertonic, Supertonic2, the OmniVoice runtime and VibeVoice

**Files:**
- Modify: `tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py` (imports; append tests)
- Modify: `tldw_Server_API/app/core/TTS/adapters/luxtts_adapter.py:30`, `:298-314`
- Modify: `tldw_Server_API/app/core/TTS/adapters/supertonic_adapter.py:25`, `:313-316`
- Modify: `tldw_Server_API/app/core/TTS/adapters/supertonic2_adapter.py:25`, `:322-325`
- Modify: `tldw_Server_API/app/core/TTS/adapters/omnivoice_runtime.py:12`, `:113-115`
- Modify: `tldw_Server_API/app/core/TTS/adapters/vibevoice_adapter.py:33`, `:963`, `:1059`
- Modify: `tldw_Server_API/tests/TTS/unit/test_local_adapter_async_offload.py:36`, `:49`

**Interfaces:**
- Consumes: `hold_tts_engine` (Task A1); `_Gate`, `_assert_cancelled_call_keeps_the_engine`, `CANCEL_MODES`, `_WAV_HEADER` (Task A1).
- Produces (tests): `_omnivoice_runtime(monkeypatch, tmp_path, gate, *, gate_load: bool) -> OmniVoiceRuntime`, used by Task A4.

- [ ] **Step 1: Write the failing tests**

In the test module, replace the import block from `import asyncio` through the `...adapters.base import ...` line with:

```python
import asyncio
import sys
import threading
import types
from collections.abc import Awaitable, Callable

import anyio
import numpy as np
import pytest

from tldw_Server_API.app.core.TTS.adapters import supertonic2_adapter, supertonic_adapter
from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, ProviderStatus, TTSRequest
```

Append to the end of the module:

```python
@pytest.mark.asyncio
@CANCEL_MODES
async def test_luxtts_keeps_the_engine_until_its_call_returns(via_anyio_scope):
    from tldw_Server_API.app.core.TTS.adapters.luxtts_adapter import LuxTTSAdapter

    gate = _Gate()

    class _Engine:
        def encode_prompt(self, prompt_audio, duration=5, rms=0.001):
            return {"prompt_tokens": [1]}

        def generate_speech(self, text, encode_dict, **_kwargs):
            # The lock spans encode_prompt and generate_speech, so this is the holder's last call.
            gate.enter("generate_speech")
            return np.zeros((1, 3200), dtype=np.float32)

    adapter = LuxTTSAdapter({"validate_reference": False, "convert_reference": False})
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE
    adapter._engine = _Engine()

    async def call():
        return await adapter.generate(
            TTSRequest(text="hello", format=AudioFormat.PCM, voice_reference=_WAV_HEADER, stream=False)
        )

    await _assert_cancelled_call_keeps_the_engine(call, gate, via_anyio_scope=via_anyio_scope)


def _supertonic_adapter(monkeypatch, tmp_path, gate: _Gate, *, version: int):
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    (voices_dir / "M1.json").write_text("{}")

    def _engine(*_args):
        gate.enter("synthesize")
        return np.zeros((1, 2400), dtype=np.float32), np.array([0.1])

    vendor = types.SimpleNamespace(
        load_text_to_speech=lambda _onnx_dir, _use_gpu=False: _engine,
        load_voice_style=lambda _paths, _verbose=False: {"style": "dummy"},
    )
    if version == 1:
        adapter_cls = supertonic_adapter.SupertonicOnnxAdapter
        module_name, voice = "tldw_Server_API.app.core.TTS.vendors.supertonic", "supertonic_m1"
    else:
        adapter_cls = supertonic2_adapter.Supertonic2OnnxAdapter
        module_name, voice = "tldw_Server_API.app.core.TTS.vendors.supertonic2", "supertonic2_m1"
    monkeypatch.setitem(sys.modules, module_name, vendor)
    adapter = adapter_cls(
        {
            "model_path": str(onnx_dir),
            "extra_params": {
                "voice_styles_dir": str(voices_dir),
                "default_voice": voice,
                "voice_files": {voice: "M1.json"},
            },
        }
    )
    return adapter, voice


@pytest.mark.asyncio
@CANCEL_MODES
@pytest.mark.parametrize("version", [1, 2], ids=["supertonic", "supertonic2"])
async def test_supertonic_keeps_the_engine_until_its_call_returns(monkeypatch, tmp_path, version, via_anyio_scope):
    gate = _Gate()
    adapter, voice = _supertonic_adapter(monkeypatch, tmp_path, gate, version=version)

    async def call():
        # WAV, not PCM: Supertonic accepts only MP3 and WAV, and MP3 needs PyAV.
        return await adapter.generate(TTSRequest(text="hello", voice=voice, format=AudioFormat.WAV, stream=False))

    await _assert_cancelled_call_keeps_the_engine(call, gate, via_anyio_scope=via_anyio_scope)


def _omnivoice_runtime(monkeypatch, tmp_path, gate: _Gate, *, gate_load: bool):
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_runtime import OmniVoiceRuntime

    class _OmniVoice:
        @classmethod
        def from_pretrained(cls, _model_path, **_kwargs):
            if gate_load:
                gate.enter("load")
            return cls()

        def generate(self, **_kwargs):
            if not gate_load:
                gate.enter("generate")
            return np.zeros(2400, dtype=np.float32)

    monkeypatch.setitem(sys.modules, "omnivoice", types.SimpleNamespace(OmniVoice=_OmniVoice))
    (tmp_path / "model").mkdir()
    return OmniVoiceRuntime({"model_path": str(tmp_path / "model")})


@pytest.mark.asyncio
@CANCEL_MODES
async def test_omnivoice_runtime_synthesis_keeps_the_model_until_its_call_returns(monkeypatch, tmp_path, via_anyio_scope):
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_protocol import OmniVoiceSynthesizeRequest

    gate = _Gate()
    runtime = _omnivoice_runtime(monkeypatch, tmp_path, gate, gate_load=False)

    async def call():
        return await runtime.synthesize(OmniVoiceSynthesizeRequest(text="hello", mode="auto", language_id="en"))

    await _assert_cancelled_call_keeps_the_engine(call, gate, via_anyio_scope=via_anyio_scope)


@pytest.mark.asyncio
@CANCEL_MODES
async def test_vibevoice_keeps_the_model_until_its_call_returns(monkeypatch, via_anyio_scope):
    """A cancelled generate must also keep a variant reload from freeing the model under the running thread."""
    import contextlib

    from tldw_Server_API.app.core.TTS.adapters import vibevoice_adapter as vibevoice_mod

    gate = _Gate()

    class _Tensor:
        def cpu(self):
            return self

        def numpy(self):
            return np.zeros(2400, dtype=np.float32)

    class _Model:
        def generate(self, **_kwargs):
            gate.enter("generate")
            return types.SimpleNamespace(speech_outputs=[_Tensor()])

    class _Processor:
        tokenizer = object()

        def __call__(self, **_kwargs):
            return {}

    fake_torch = types.SimpleNamespace(no_grad=contextlib.nullcontext, is_tensor=lambda _value: False)
    monkeypatch.setattr(vibevoice_mod, "_get_torch", lambda **_kwargs: fake_torch)
    adapter = vibevoice_mod.VibeVoiceAdapter({"device": "cpu"})
    adapter.model = _Model()
    adapter.processor = _Processor()

    async def call():
        request = TTSRequest(text="Speaker 1: hello", format=AudioFormat.PCM, stream=True)
        return [chunk async for chunk in adapter._stream_audio_vibevoice(request, "speaker_1", 1)]

    await _assert_cancelled_call_keeps_the_engine(call, gate, via_anyio_scope=via_anyio_scope)
```

- [ ] **Step 2: Run them to see them fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py`
Expected: `10 failed, 4 passed`; each failure `AssertionError: assert 2 == 1` (`generate_speech`, `synthesize`, `generate` twice).

- [ ] **Step 3: Adopt the helper**

`luxtts_adapter.py` line 30: `from ..utils import hold_tts_engine, parse_bool`. Lines 298-314: change `async with self._engine_lock:` to `async with hold_tts_engine(self._engine_lock) as run:` and both `await asyncio.to_thread(` to `await run(`, so the block reads:

```python
            async with hold_tts_engine(self._engine_lock) as run:
                encode_dict = await run(
                    self._engine.encode_prompt,  # type: ignore[union-attr]
                    voice_path,
                    duration=prompt_duration,
                    rms=prompt_rms,
                )
                wav = await run(
                    self._engine.generate_speech,  # type: ignore[union-attr]
                    request.text,
                    encode_dict,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    t_shift=t_shift,
                    speed=request.speed,
                    return_smooth=return_smooth,
                )
```

`supertonic_adapter.py`: after `from ..tts_validation import validate_tts_request` (line 25) add `from ..utils import hold_tts_engine`; lines 313-316 become:

```python
            async with hold_tts_engine(self._engine_lock) as run:
                wav, duration = await run(
                    self._engine, request.text, style, total_step, speed
                )
```

`supertonic2_adapter.py`: after `from ..tts_validation import validate_tts_request` (line 25) add `from ..utils import hold_tts_engine`; lines 322-325 become:

```python
            async with hold_tts_engine(self._engine_lock) as run:
                wav, duration = await run(
                    self._engine, request.text, language, style, total_step, speed
                )
```

`omnivoice_runtime.py`: before `from .omnivoice_sidecar_protocol import OmniVoiceSynthesizeRequest` (line 12) add `from ..utils import hold_tts_engine`; lines 113-115 become:

```python
        async with hold_tts_engine(self._generate_lock) as run:
            try:
                generated = await run(model.generate, **generate_kwargs)
```

`vibevoice_adapter.py` line 33: `from ..utils import hold_tts_engine, parse_bool` (it no longer uses `run_tts_blocking_call`). Line 963: `async with hold_tts_engine(self._model_state_lock) as run:`. Line 1059:

```python
                outputs = await run(_generate_outputs)
```

`_ensure_model_variant` (line 1365) keeps its plain `async with self._model_state_lock:`: it makes no thread call, and it now waits for a cancelled generate's thread before reloading.

- [ ] **Step 4: Update the offload test for VibeVoice**

`tests/TTS/unit/test_local_adapter_async_offload.py` line 36:

```python
    assert vibevoice_adapter.hold_tts_engine is tts_utils.hold_tts_engine
```

Line 49:

```python
    assert _code_object_references_name(vibevoice_adapter.VibeVoiceAdapter._stream_audio_vibevoice, "hold_tts_engine")
```

- [ ] **Step 5: Run the tests**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py tldw_Server_API/tests/TTS/unit/test_local_adapter_async_offload.py tldw_Server_API/tests/TTS/adapters/test_luxtts_adapter_mock.py tldw_Server_API/tests/TTS/adapters/test_supertonic_adapter_mock.py tldw_Server_API/tests/TTS/adapters/test_supertonic2_adapter_mock.py tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_runtime.py tldw_Server_API/tests/TTS/test_vibevoice_adapter_unit.py tldw_Server_API/tests/TTS/adapters/test_vibevoice_adapter_mock.py`
Expected: only the 3 baseline failures (`test_luxtts_generate_non_stream`, `test_supertonic_initialize_and_generate`, `test_supertonic2_initialize_and_generate`, all `No module named 'av'`); the new file gives `14 passed`.

- [ ] **Step 6: Lint and commit**

```bash
$PY -m ruff check tldw_Server_API/app/core/TTS/adapters/luxtts_adapter.py tldw_Server_API/app/core/TTS/adapters/supertonic_adapter.py tldw_Server_API/app/core/TTS/adapters/supertonic2_adapter.py tldw_Server_API/app/core/TTS/adapters/omnivoice_runtime.py tldw_Server_API/app/core/TTS/adapters/vibevoice_adapter.py tldw_Server_API/tests/TTS/unit/
git add tldw_Server_API/app/core/TTS/adapters/luxtts_adapter.py tldw_Server_API/app/core/TTS/adapters/supertonic_adapter.py tldw_Server_API/app/core/TTS/adapters/supertonic2_adapter.py tldw_Server_API/app/core/TTS/adapters/omnivoice_runtime.py tldw_Server_API/app/core/TTS/adapters/vibevoice_adapter.py tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py tldw_Server_API/tests/TTS/unit/test_local_adapter_async_offload.py
git commit -m "fix(tts): LuxTTS, Supertonic, OmniVoice and VibeVoice keep the engine through a cancelled call (TASK-13856)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Expected: ruff reports only `omnivoice_runtime.py` B009 (`getattr(omnivoice_module, "OmniVoice")`), present on dev.

### Task A4: Model-load locks in the OmniVoice runtime, Qwen3 MLX, the Qwen3 pipeline, Echo-TTS and KittenTTS

**Files:**
- Modify: `tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py` (imports; append tests)
- Modify: `tldw_Server_API/app/core/TTS/adapters/omnivoice_runtime.py:59`, `:82`
- Modify: `tldw_Server_API/app/core/TTS/adapters/qwen3_runtime_mlx.py:11`, `:133-135`
- Modify: `tldw_Server_API/app/core/TTS/adapters/qwen3_tts_adapter.py:35`, `:638-641`
- Modify: `tldw_Server_API/app/core/TTS/adapters/echo_tts_adapter.py:34`, `:479`, `:502`, `:510`, `:517`
- Modify: `tldw_Server_API/app/core/TTS/adapters/kitten_tts_adapter.py:16`, `:94`, `:103`, `:110`

**Interfaces:**
- Consumes: `hold_tts_engine` (Task A1); `_Gate`, `_assert_cancelled_call_keeps_the_engine`, `CANCEL_MODES` (Task A1); `_omnivoice_runtime` (Task A3).

- [ ] **Step 1: Write the failing tests**

In the test module, replace the stdlib import lines (`import asyncio` through `from collections.abc import Awaitable, Callable`) with:

```python
import asyncio
import platform
import sys
import threading
import types
from collections.abc import Awaitable, Callable
```

Append to the end of the module:

```python
@pytest.mark.asyncio
@CANCEL_MODES
async def test_omnivoice_runtime_load_keeps_the_lock_until_the_load_returns(monkeypatch, tmp_path, via_anyio_scope):
    gate = _Gate()
    runtime = _omnivoice_runtime(monkeypatch, tmp_path, gate, gate_load=True)

    await _assert_cancelled_call_keeps_the_engine(runtime.load, gate, via_anyio_scope=via_anyio_scope)


@pytest.mark.asyncio
@CANCEL_MODES
async def test_qwen3_mlx_model_load_keeps_the_lock_until_the_load_returns(monkeypatch, via_anyio_scope):
    from tldw_Server_API.app.core.TTS.adapters.qwen3_tts_adapter import Qwen3TTSAdapter

    gate = _Gate()

    def load_model(_model_id):
        gate.enter("load_model")
        return object()

    utils_module = types.ModuleType("mlx_audio.tts.utils")
    utils_module.load_model = load_model
    monkeypatch.setitem(sys.modules, "mlx_audio", types.ModuleType("mlx_audio"))
    monkeypatch.setitem(sys.modules, "mlx_audio.tts", types.ModuleType("mlx_audio.tts"))
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.utils", utils_module)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    adapter = Qwen3TTSAdapter({"runtime": "mlx", "device": "mps"})
    await adapter.ensure_initialized()
    runtime = adapter._get_runtime()

    async def call():
        return await runtime._get_model("auto")

    await _assert_cancelled_call_keeps_the_engine(call, gate, via_anyio_scope=via_anyio_scope)


@pytest.mark.asyncio
@CANCEL_MODES
async def test_qwen3_pipeline_load_keeps_the_lock_until_the_load_returns(via_anyio_scope):
    from tldw_Server_API.app.core.TTS.adapters.qwen3_tts_adapter import Qwen3TTSAdapter

    gate = _Gate()

    def build_pipeline(_model_id):
        gate.enter("build_pipeline")
        return object()

    adapter = Qwen3TTSAdapter({})
    adapter._pipeline_builders = [("fake", build_pipeline)]

    async def call():
        return await adapter._get_pipeline("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")

    await _assert_cancelled_call_keeps_the_engine(call, gate, via_anyio_scope=via_anyio_scope)


@pytest.mark.asyncio
@CANCEL_MODES
async def test_echo_tts_model_load_keeps_the_lock_until_the_load_returns(via_anyio_scope):
    from tldw_Server_API.app.core.TTS.adapters.echo_tts_adapter import EchoTTSAdapter

    gate = _Gate()

    def load_model_from_hf(**_kwargs):
        gate.enter("load_model_from_hf")
        return object()

    adapter = EchoTTSAdapter({})
    adapter.device = "cuda"  # initialize() resolves it; this test skips initialize()
    adapter._echo_modules_loaded = True
    adapter._echo_inference = types.SimpleNamespace(
        load_model_from_hf=load_model_from_hf,
        load_fish_ae_from_hf=lambda **_kwargs: object(),
        load_pca_state_from_hf=lambda **_kwargs: object(),
    )
    adapter._import_torch = lambda: types.SimpleNamespace(bfloat16="bf16", float16="f16", float32="f32")

    await _assert_cancelled_call_keeps_the_engine(adapter._ensure_models_loaded, gate, via_anyio_scope=via_anyio_scope)


@pytest.mark.asyncio
@CANCEL_MODES
async def test_kitten_tts_runtime_load_keeps_the_lock_until_the_load_returns(monkeypatch, via_anyio_scope):
    from tldw_Server_API.app.core.TTS.adapters import kitten_tts_adapter as kitten_mod

    gate = _Gate()

    def kitten_runtime(assets):
        gate.enter("KittenRuntime")
        return types.SimpleNamespace(sample_rate=24000)

    monkeypatch.setattr(kitten_mod, "resolve_model_revision", lambda _model, revision: revision)
    monkeypatch.setattr(kitten_mod, "download_model_assets", lambda model, **_kwargs: types.SimpleNamespace(repo_id=model))
    monkeypatch.setattr(kitten_mod, "KittenRuntime", kitten_runtime)
    adapter = kitten_mod.KittenTTSAdapter({})

    async def call():
        return await adapter._load_runtime_for_model("KittenML/kitten-tts-nano-0.8")

    await _assert_cancelled_call_keeps_the_engine(call, gate, via_anyio_scope=via_anyio_scope)
```

- [ ] **Step 2: Run them to see them fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py`
Expected: `10 failed, 14 passed`; each failure `AssertionError: assert 2 == 1` (`load`, `load_model`, `build_pipeline`, `load_model_from_hf`, `KittenRuntime`): the cancelled load released the lock and a second load started alongside it.

- [ ] **Step 3: Adopt the helper**

Line numbers in this step are those before Task A3's and this task's import inserts. Task A3 adds an import at `omnivoice_runtime.py:12`, so by now the load lock is at `:60` and `:83`, and `qwen3_runtime_mlx.py`'s lines 133-135 become 134-136 once its import is added; anchor on the quoted text, not the numbers (Review M2).

`omnivoice_runtime.py` line 59: `async with hold_tts_engine(self._load_lock) as run:`; line 82: `self.model = await run(` (arguments unchanged).

`qwen3_runtime_mlx.py`: after line 11 (`from ..tts_exceptions import ...`) add `from ..utils import hold_tts_engine`; lines 133-135 become:

```python
        async with hold_tts_engine(self._model_lock) as run:
            if self._model is None or self._model_id != backend_model_id:
                self._model = await run(self._load_model, backend_model_id)
```

`qwen3_tts_adapter.py` line 35: `from ..utils import hold_tts_engine, parse_bool, resolve_qwen3_runtime_name`; lines 638-641 become:

```python
        async with hold_tts_engine(self._pipeline_lock) as run:
            if load_id in self._pipelines:
                return self._pipelines[load_id]
            pipeline = await run(self._build_pipeline, load_id)
```

`echo_tts_adapter.py` line 34: `from ..utils import hold_tts_engine, parse_bool`; line 479: `async with hold_tts_engine(self._model_lock) as run:`; lines 502, 510 and 517: `self._model = await run(`, `self._fish_ae = await run(`, `self._pca_state = await run(` (arguments unchanged).

`kitten_tts_adapter.py` line 16: `from ..utils import hold_tts_engine, parse_bool`; line 94: `async with hold_tts_engine(self._runtime_lock) as run:`; line 103: `assets = await run(` (arguments unchanged); line 110:

```python
                runtime = await run(KittenRuntime, assets)
```

- [ ] **Step 4: Run the tests**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_runtime.py tldw_Server_API/tests/TTS_NEW/unit/adapters/ tldw_Server_API/tests/TTS/adapters/test_kittentts_adapter_mock.py tldw_Server_API/tests/TTS_NEW/unit/adapters/test_echo_tts_audio_loading.py`
Expected: all pass; the new file gives `24 passed`. (`test_qwen3_mlx_runtime.py` monkeypatches `asyncio.to_thread` and still sees `["fake_load_model", "_run_generation"]`: the helper offloads through `run_tts_blocking_call`, which calls `asyncio.to_thread`.)

- [ ] **Step 5: Lint and commit**

```bash
$PY -m ruff check tldw_Server_API/app/core/TTS/adapters/omnivoice_runtime.py tldw_Server_API/app/core/TTS/adapters/qwen3_runtime_mlx.py tldw_Server_API/app/core/TTS/adapters/qwen3_tts_adapter.py tldw_Server_API/app/core/TTS/adapters/echo_tts_adapter.py tldw_Server_API/app/core/TTS/adapters/kitten_tts_adapter.py tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py
git add tldw_Server_API/app/core/TTS/adapters/omnivoice_runtime.py tldw_Server_API/app/core/TTS/adapters/qwen3_runtime_mlx.py tldw_Server_API/app/core/TTS/adapters/qwen3_tts_adapter.py tldw_Server_API/app/core/TTS/adapters/echo_tts_adapter.py tldw_Server_API/app/core/TTS/adapters/kitten_tts_adapter.py tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py
git commit -m "fix(tts): a cancelled model load keeps its lock until the load returns (TASK-13856)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Expected: ruff reports only findings present on dev (`kitten_tts_adapter.py` UP045 x3, `omnivoice_runtime.py` B009, `qwen3_runtime_mlx.py` UP037 and B009).

### Task A5: Verify, file the Kokoro follow-up, close TASK-13856, open PR A

**Files:**
- Create: `backlog/tasks/task-<N> - ...md` (through backlog-py)
- Modify: `backlog/tasks/task-13856 - ...md` (through backlog-py)

- [ ] **Step 1: Run every affected suite**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q -n 6 tldw_Server_API/tests/TTS tldw_Server_API/tests/TTS_NEW tldw_Server_API/tests/Docs tldw_Server_API/tests/Persona/test_persona_kitten_model_preparation.py tldw_Server_API/tests/lint/test_private_coercion_ratchet.py`
Expected (probed): `7 failed, 2110 passed, 28 skipped, 1 xfailed, 1 xpassed`; the 7 are exactly the baseline list above. Any other failure: re-run it on unchanged origin/dev before deciding.

- [ ] **Step 2: Stability, lint, Bandit**

```bash
for i in 1 2 3; do TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py tldw_Server_API/tests/TTS/adapters/test_neutts_adapter.py | tail -1; done
uvx bandit -q -ll tldw_Server_API/app/core/TTS/utils.py tldw_Server_API/app/core/TTS/adapters/neutts_adapter.py tldw_Server_API/app/core/TTS/adapters/pocket_tts_adapter.py tldw_Server_API/app/core/TTS/adapters/luxtts_adapter.py tldw_Server_API/app/core/TTS/adapters/supertonic_adapter.py tldw_Server_API/app/core/TTS/adapters/supertonic2_adapter.py tldw_Server_API/app/core/TTS/adapters/omnivoice_runtime.py tldw_Server_API/app/core/TTS/adapters/qwen3_runtime_mlx.py tldw_Server_API/app/core/TTS/adapters/qwen3_tts_adapter.py tldw_Server_API/app/core/TTS/adapters/vibevoice_adapter.py tldw_Server_API/app/core/TTS/adapters/echo_tts_adapter.py tldw_Server_API/app/core/TTS/adapters/kitten_tts_adapter.py
```

Expected: `44 passed` three times (`45` if PR B merged first); Bandit prints no issue (only the existing `nosec encountered (B615)` warning for `vibevoice_adapter.py:610`).

- [ ] **Step 3: File the Kokoro follow-up**

```bash
git fetch origin dev --quiet
HIGHEST=$({ git ls-tree -r --name-only origin/dev backlog/; gh pr list --state open --limit 500 --json files --jq '.[].files[].path'; } | sed -nE 's|^backlog/.*task-([0-9]+)[ .].*|\1|p' | sort -n | tail -1)
NEXT=TASK-$((HIGHEST + 1))
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Kokoro's init_timeout_sec abandons a running model load, so a second load can start alongside it" \
  --id "$NEXT" -l tts,bug \
  -d "KokoroAdapter._ensure_model_loaded holds _model_lock (core/TTS/adapters/kokoro_adapter.py:359) around _load_onnx_model / _load_pytorch_model, which load the model with asyncio.to_thread. asyncio.wait_for(..., timeout=self.init_timeout_sec) (:366, :389; from initialize() and from _ensure_model_loaded()) cancels its own wait at the timeout while the load thread keeps running, and the lock passes to the next caller, which starts a second load. Found while fixing TASK-13856, which gave the other local adapters hold_tts_engine (core/TTS/utils.py); Kokoro needs its timeout reworked before that helper fits. (With lazy init initialize() does not load, :375-382; without it generate() waits on ensure_initialized() at :770 before _ensure_model_loaded() at :776, which then returns early, so no two-locks interleaving is claimed.)" \
  --ac "A Kokoro load that outlasts init_timeout_sec (from initialize() or _ensure_model_loaded()) does not let a second load start while the first thread runs, with a test for a plain cancel and an anyio cancel scope" \
  --ac "The Kokoro load is held through hold_tts_engine, or the filing explains why it cannot be"
echo "$NEXT"
```

Expected: backlog-py prints the new task's path and `echo` prints its id.

- [ ] **Step 4: Close TASK-13856**

Run in the same shell as Step 3 (the notes use `$NEXT`). If Step 1's counts differ from the probed ones below, use yours.

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13856 \
  --append-notes "Fixed: core/TTS/utils.py gains hold_tts_engine(lock), an async context manager that holds a local engine's asyncio lock and yields run(func, *args, **kwargs), which runs one call on a worker thread (through run_tts_blocking_call) under asyncio.shield. On exit it releases the lock at once or, when the holder leaves while its last call still runs on its thread, from that call's completion (TASK-13855's NeuTTS pattern, lifted; its finally never awaits, because anyio re-cancels a cancelled holder at every await). One context manager covers single calls, holds spanning several calls, and streams (await run(next, iterator, sentinel)), so there is no separate stream variant. Adopted at every lock held around a thread-offloaded call in the local adapters: NeuTTS (_engine_call and the stream, replacing _release_engine_when_done), PocketTTS generate (pocket_tts_adapter.py:317) and stream (:523-536), LuxTTS (luxtts_adapter.py:298-314), Supertonic (:313), Supertonic2 (:322), the OmniVoice runtime's synthesis (omnivoice_runtime.py:113) and load (:59-85), the Qwen3 MLX model load (qwen3_runtime_mlx.py:133-136) and the Qwen3 pipeline load (qwen3_tts_adapter.py:638-641). Listed sites verified: qwen3_runtime_mlx.py and qwen3_tts_adapter.py hold their lock around a model load only (their generation runs unlocked); the shape and the fix are the same. Three unlisted sites have the shape and are fixed: VibeVoice (vibevoice_adapter.py:963-1059, where a cancelled generate let the next request's variant reload free the model under the running thread), the Echo-TTS model load (echo_tts_adapter.py:479-517) and the KittenTTS runtime load (kitten_tts_adapter.py:94-110). Kokoro's _model_lock is filed as $NEXT: its own wait_for(init_timeout_sec) abandons a running load thread, so a second load can start. A cancelled model load's result is discarded, so the next caller loads again after it (ponytail comment in the helper)." \
  --append-notes "Tests: tests/TTS/unit/test_tts_engine_lock_cancellation.py, 24 cases (12 sites, each with a plain task cancel and an anyio cancel scope), every one RED on dev with assert 2 == 1 (the next call's engine call started while the cancelled one still ran) and GREEN after. Mutations: releasing at cancel fails all 24 plus 4 NeuTTS tests; awaiting the thread in the holder's finally fails exactly the 12 anyio cases plus the NeuTTS anyio and error-log tests; dropping the exception retrieval fails test_neutts_cancelled_request_engine_error_is_not_logged_raw. test_local_adapter_async_offload.py now checks that VibeVoice offloads through hold_tts_engine. tests/TTS, tests/TTS_NEW, tests/Docs and the two other test files that reference touched modules: 7 failed, 2110 passed, 28 skipped, 1 xfailed, 1 xpassed; the 7 fail on unchanged dev too (No module named 'av'; torch not installed). New file and test_neutts_adapter.py: 44 passed in three runs. Ruff: no new findings. Bandit -ll (uvx): clean. ADR assessment: none needed; a local fix inside the TTS adapters, no durable rule changes (ADR-011 governs adapter init failure retry, untouched). No docs describe these locks." \
  --check-ac 1 --check-ac 2 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "A cancelled TTS request (a client disconnect) no longer hands a local engine to the next request while its call still runs on a worker thread. The completion-released lock NeuTTS got in TASK-13855 is now hold_tts_engine in core/TTS/utils.py, used by NeuTTS, PocketTTS, LuxTTS, Supertonic, Supertonic2, VibeVoice and the OmniVoice runtime for synthesis, and by the OmniVoice runtime, Qwen3 MLX, the Qwen3 pipeline, Echo-TTS and KittenTTS for model loads. Kokoro's init_timeout_sec abandons a running load and is tracked separately." \
  -s Done
git add -A backlog/tasks
git commit -m "docs(backlog): close TASK-13856; file the Kokoro load-lock follow-up

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Push and open PR A**

```bash
git push -u origin fix/tts-engine-lock-cancellation
gh pr create --base dev --title "fix(tts): a cancelled request keeps a local TTS engine until its thread call returns (TASK-13856)" --body "$(cat <<'EOF'
## Summary
- `hold_tts_engine(lock)` (core/TTS/utils.py): holds a local engine's lock and runs engine calls on worker threads; when the holder is cancelled mid-call, the lock is released from the call's completion, not by the cancelled caller (TASK-13855's NeuTTS pattern, lifted).
- Adopted by NeuTTS, PocketTTS (generate and stream), LuxTTS, Supertonic, Supertonic2, VibeVoice and the OmniVoice runtime for synthesis, and by the OmniVoice runtime, Qwen3 MLX, the Qwen3 pipeline, Echo-TTS and KittenTTS for model loads. VibeVoice, Echo-TTS and KittenTTS were not listed in the task but have the same shape.
- Kokoro's `init_timeout_sec` abandoning a running load is filed as a follow-up.

## Test plan
- [x] `tests/TTS/unit/test_tts_engine_lock_cancellation.py`: 24 cases (12 sites x task cancel / anyio cancel scope), RED on dev, GREEN here
- [x] NeuTTS tests unchanged and green; mutation checks in TASK-13856's notes
- [x] tests/TTS, tests/TTS_NEW, tests/Docs: only the 7 baseline failures (PyAV/torch not installed), same on dev
- [x] ruff (no new findings), Bandit -ll clean

## Change summary
(owner, per ADR-004)

## ADR assessment
None needed: a local fix inside the TTS adapters; ADR-011 untouched.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 6: Clean review, then merge per the repo's merge rules**

Run `/superpowers:requesting-code-review` in a clean subagent on the PR diff; address findings before merge. Merge only after the owner's Change summary (Decisions for the owner, 1). Merge through the serial queue (the owner's `pr_queue.sh`), one PR at a time; MERGE_QUEUE is unset.

---

## PR B: TASK-13857

### Task B1: A capability-listing timeout leaves a slow adapter load running

**Files:**
- Modify: `tldw_Server_API/app/core/TTS/adapter_registry.py:54` (constants), `:655` (`get_adapter`), `:1023-1034` (`get_all_capabilities`), `:1060-1063` (`list_capabilities`), `:1119` (`find_adapter_for_requirements`)
- Modify: `tldw_Server_API/app/core/TTS/tts_service_v2.py:31` (import), `_get_fallback_adapter`'s provider loop (`:3712` onward)
- Modify: `tldw_Server_API/tests/TTS/test_tts_adapter_registry_wrapper_migration.py` (imports; new adapter class; retarget one test; five new tests)

**Interfaces:**
- Produces: module constant `adapter_registry._CAPABILITY_LISTING_TIMEOUT_SECONDS = 5.0` (tests monkeypatch it; `tts_service_v2` reads it through the module so the patch reaches it) and module set `adapter_registry._PENDING_INITIALIZATIONS`. `TTSAdapterRegistry.get_adapter` keeps its signature; its initialization now survives its caller's cancellation, and later callers wait for it through the base registry's per-provider lock. `get_all_capabilities`, `list_capabilities`, `find_adapter_for_requirements` and `TTSServiceV2._get_fallback_adapter` wait at most `_CAPABILITY_LISTING_TIMEOUT_SECONDS` for one provider and skip a provider that is still loading, without marking it failed.

- [ ] **Step 1: Branch and mark the task In Progress**

```bash
gh pr view 3264 --json state --jq .state   # must print MERGED
git fetch origin dev --quiet
git switch -c fix/tts-registry-slow-load origin/dev
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13857 -s "In Progress"
```

- [ ] **Step 2: Name the listing timeout (refactor, no behaviour change)**

In `adapter_registry.py`, after `_TTS_REGISTRY_ADAPTER_EXCEPTIONS` (line 54) add:

```python
# How long a caller that only wants some ready provider (the capability listings behind GET /audio/health
# and /audio/providers, failover and best-adapter selection) waits for one provider's adapter.
_CAPABILITY_LISTING_TIMEOUT_SECONDS = 5.0

# Strong references to shielded initializations: the loop holds tasks weakly, and once a caller is
# cancelled nothing else points at the task that is still loading a model.
_PENDING_INITIALIZATIONS: set[asyncio.Future] = set()
```

(The set is Review M8; Step 5 uses it.) At line 1025 replace `timeout=5.0,` with `timeout=_CAPABILITY_LISTING_TIMEOUT_SECONDS,`.

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/TTS/test_tts_adapter_registry_wrapper_migration.py`
Expected: `22 passed`.

- [ ] **Step 3: Write the failing tests**

In `test_tts_adapter_registry_wrapper_migration.py`, replace lines 3-7 (`import asyncio` through the blank line after `import pytest`; keep line 1, `from __future__ import annotations`, which class-body annotations such as `instances: list[_BlockingAdapter] = []` need) with the block below. It ends directly above `from tldw_Server_API.app.core.TTS import adapter_registry` with no blank line, because both lines are first-party imports (a blank line there gives ruff I001) (Review M3):

```python
import asyncio
import threading
from typing import Any
from unittest.mock import AsyncMock

import anyio
import pytest

from tldw_Server_API.app.core.Infrastructure.provider_registry import (
    ProviderStatus as RegistryProviderStatus,
)
```

Then add `from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSServiceV2` directly after the existing `from tldw_Server_API.app.core.TTS.tts_exceptions import TTSError` line.

Before `class _CloseTrackingAdapter` add:

```python
class _SlowLoadAdapter(_MockAdapterV1):
    """Loads its model on a worker thread, as luxtts, index_tts and pocket_tts do, for as long as the test says."""

    loads = 0
    release: threading.Event | None = None

    async def initialize(self) -> bool:
        type(self).loads += 1
        assert await asyncio.to_thread(type(self).release.wait, 5)
        return True
```

In `test_registry_closes_adapter_and_reraises_initialization_cancellation`, replace `task = asyncio.create_task(registry.get_adapter(TTSProvider.MOCK))` with:

```python
    # get_adapter shields initialization from its callers; loop shutdown still cancels it.
    task = asyncio.create_task(registry._initialize_adapter(TTSProvider.MOCK))
```

(The test keeps its purpose: a cancelled initialization closes the half-built adapter. It passes before and after the fix.)

After that test add:

```python
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "via_anyio_scope",
    [False, True],
    # Starlette cancels a disconnected request through an anyio cancel scope.
    ids=["task-cancel", "anyio-scope"],
)
async def test_registry_cancelled_caller_leaves_the_initialization_to_the_next_caller(
    monkeypatch,
    via_anyio_scope,
) -> None:
    """A client disconnect during a first request's model load must not restart the load for the next request."""
    registry = TTSAdapterRegistry(config={"mock_enabled": True}, include_defaults=False)
    _BlockingAdapter.started = asyncio.Event()
    _BlockingAdapter.release = asyncio.Event()
    _BlockingAdapter.instances = []
    registry.register_adapter(TTSProvider.MOCK, _BlockingAdapter)

    async def _get_resource_manager() -> _NonCriticalResourceManager:
        return _NonCriticalResourceManager()

    monkeypatch.setattr(adapter_registry, "get_resource_manager", _get_resource_manager)
    scopes: list[anyio.CancelScope] = []

    async def _request() -> None:
        if not via_anyio_scope:
            await registry.get_adapter(TTSProvider.MOCK)
            return
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            await registry.get_adapter(TTSProvider.MOCK)

    first = asyncio.create_task(_request())
    await _BlockingAdapter.started.wait()
    if via_anyio_scope:
        scopes[0].cancel()
        await first
    else:
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
    second = asyncio.create_task(registry.get_adapter(TTSProvider.MOCK))
    await asyncio.sleep(0.05)
    _BlockingAdapter.release.set()
    adapter = await asyncio.wait_for(second, 5)  # bounded: a regression fails instead of hanging (Review M1)

    assert len(_BlockingAdapter.instances) == 1
    assert adapter is _BlockingAdapter.instances[0]
    assert adapter.close_calls == 0


@pytest.mark.asyncio
async def test_registry_capability_timeout_neither_restarts_nor_fails_a_slow_load(monkeypatch) -> None:
    """GET /audio/health lists capabilities with a timeout; a slow model load must outlive it unharmed."""
    registry = TTSAdapterRegistry(
        config={"mock_enabled": True, "adapter_failure_retry_seconds": 300},
        include_defaults=False,
    )
    _SlowLoadAdapter.loads = 0
    _SlowLoadAdapter.release = threading.Event()
    registry.register_adapter(TTSProvider.MOCK, _SlowLoadAdapter)

    async def _get_resource_manager() -> _NonCriticalResourceManager:
        return _NonCriticalResourceManager()

    monkeypatch.setattr(adapter_registry, "get_resource_manager", _get_resource_manager)
    monkeypatch.setattr(adapter_registry, "_CAPABILITY_LISTING_TIMEOUT_SECONDS", 0.05)
    try:
        assert TTSProvider.MOCK not in await registry.get_all_capabilities()
        assert registry._base.get_status("mock") == RegistryProviderStatus.ENABLED
        # A second poll while the load still runs joins it rather than starting another.
        assert TTSProvider.MOCK not in await registry.get_all_capabilities()
    finally:
        _SlowLoadAdapter.release.set()
    adapter = await registry.get_adapter(TTSProvider.MOCK)

    assert isinstance(adapter, _SlowLoadAdapter)
    assert _SlowLoadAdapter.loads == 1
    assert TTSProvider.MOCK in await registry.get_all_capabilities()


@pytest.mark.asyncio
async def test_registry_list_capabilities_does_not_wait_out_a_slow_load(monkeypatch) -> None:
    """GET /audio/health calls list_capabilities after get_all_capabilities; neither may wait out a load. (Review I1)"""
    registry = TTSAdapterRegistry(
        config={"mock_enabled": True, "adapter_failure_retry_seconds": 300},
        include_defaults=False,
    )
    _SlowLoadAdapter.loads = 0
    _SlowLoadAdapter.release = threading.Event()
    registry.register_adapter(TTSProvider.MOCK, _SlowLoadAdapter)

    async def _get_resource_manager() -> _NonCriticalResourceManager:
        return _NonCriticalResourceManager()

    monkeypatch.setattr(adapter_registry, "get_resource_manager", _get_resource_manager)
    monkeypatch.setattr(adapter_registry, "_CAPABILITY_LISTING_TIMEOUT_SECONDS", 0.05)
    try:
        await registry.get_all_capabilities()
        envelopes = await asyncio.wait_for(registry.list_capabilities(), timeout=1.0)
    finally:
        _SlowLoadAdapter.release.set()

    assert envelopes == [{"provider": "mock", "availability": "enabled", "capabilities": None}]
    assert isinstance(await registry.get_adapter(TTSProvider.MOCK), _SlowLoadAdapter)
    assert _SlowLoadAdapter.loads == 1


@pytest.mark.asyncio
async def test_registry_best_adapter_skips_a_provider_that_is_still_loading(monkeypatch) -> None:
    """Failover picks a provider that is ready now; it does not wait out another provider's model load."""
    registry = TTSAdapterRegistry(
        config={
            "mock_enabled": True,
            "kokoro_enabled": True,
            "provider_priority": ["mock", "kokoro"],
            "adapter_failure_retry_seconds": 300,
        },
        include_defaults=False,
    )
    _SlowLoadAdapter.loads = 0
    _SlowLoadAdapter.release = threading.Event()
    registry.register_adapter(TTSProvider.MOCK, _SlowLoadAdapter)
    registry.register_adapter(TTSProvider.KOKORO, _MockAdapterV1)

    async def _get_resource_manager() -> _NonCriticalResourceManager:
        return _NonCriticalResourceManager()

    monkeypatch.setattr(adapter_registry, "get_resource_manager", _get_resource_manager)
    monkeypatch.setattr(adapter_registry, "_CAPABILITY_LISTING_TIMEOUT_SECONDS", 0.05)
    try:
        adapter = await asyncio.wait_for(registry.find_adapter_for_requirements(language="en"), timeout=1.0)
        # Skipped, not failed: a request that names the loading provider still joins its initialization.
        assert registry._base.get_status("mock") == RegistryProviderStatus.ENABLED
    finally:
        _SlowLoadAdapter.release.set()

    assert isinstance(adapter, _MockAdapterV1) and not isinstance(adapter, _SlowLoadAdapter)
    assert isinstance(await registry.get_adapter(TTSProvider.MOCK), _SlowLoadAdapter)
    assert _SlowLoadAdapter.loads == 1


@pytest.mark.asyncio
async def test_service_fallback_skips_a_provider_that_is_still_loading(monkeypatch) -> None:
    """TTSServiceV2 failover walks the providers itself; a still-loading one is skipped, not waited on."""
    ready = _MockAdapterV1({})
    ready.validate_request = AsyncMock(return_value=(True, None))

    class _Registry:
        _adapter_specs = {TTSProvider.OPENAI: object(), TTSProvider.ELEVENLABS: object()}

        async def get_adapter(self, provider):
            if provider == TTSProvider.OPENAI:
                await asyncio.Event().wait()  # a model load that has not returned
            return ready

    class _Factory:
        registry = _Registry()

        async def get_best_adapter(self, *_, **__):
            return None

    monkeypatch.setattr(adapter_registry, "_CAPABILITY_LISTING_TIMEOUT_SECONDS", 0.05)
    service = TTSServiceV2(_Factory())

    adapter = await asyncio.wait_for(
        service._get_fallback_adapter(TTSRequest(text="hello", format=AudioFormat.MP3)),
        timeout=1.0,
    )

    assert adapter is ready
```

- [ ] **Step 4: Run them to see them fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/TTS/test_tts_adapter_registry_wrapper_migration.py`
Expected: `6 failed, 22 passed`:
- both `test_registry_cancelled_caller_leaves_the_initialization_to_the_next_caller` cases with `assert 2 == 1` (the cancelled caller's initialization was cancelled and the next caller built a second adapter);
- `test_registry_capability_timeout_neither_restarts_nor_fails_a_slow_load` with `assert <ProviderStatus.FAILED: 'failed'> == <ProviderStatus.ENABLED: 'enabled'>`;
- `test_registry_list_capabilities_does_not_wait_out_a_slow_load`, `test_registry_best_adapter_skips_a_provider_that_is_still_loading` and `test_service_fallback_skips_a_provider_that_is_still_loading` with `TimeoutError` from the test's own `wait_for(..., 1.0)`: before the bound, each waits out the load (or, for the service test, a load that never returns). Once Step 5's first half is in but not its second, the first three go green and these three stay red. (Review I1)

- [ ] **Step 5: Shield the initialization, stop failing a provider on timeout, and bound the callers that only want a ready provider**

`get_adapter`, line 655, replace `adapter = await self._base.get_adapter_async(provider_key)` with:

```python
        # Shielded: a caller that stops waiting (the capability listing's timeout, a client
        # disconnect) must not cancel the initialization. An adapter loading its model on a worker
        # thread cannot be stopped, and the next caller would load it again; instead the next
        # caller waits on the base registry's per-provider lock and then shares the adapter.
        initialization = asyncio.ensure_future(self._base.get_adapter_async(provider_key))
        _PENDING_INITIALIZATIONS.add(initialization)
        initialization.add_done_callback(_PENDING_INITIALIZATIONS.discard)
        adapter = await asyncio.shield(initialization)
```

(Review M8: the set keeps the inner task referenced after its caller is cancelled; `asyncio` documents that `shield` does not.)

`get_all_capabilities`, replace the `except asyncio.TimeoutError:` body (lines 1032-1034) with:

```python
            except asyncio.TimeoutError:
                # Still initializing: get_adapter shields the load from this timeout, so it is
                # neither restarted nor marked failed, and the next caller waits for it.
                logger.info("Capabilities for {} are not ready; its adapter is still initializing", provider_key)
```

`list_capabilities`, in the ENABLED branch (`:1060-1063`), replace `adapter = await self.get_adapter(provider_enum or provider_name)` and add a timeout branch ahead of the existing one (Review I1):

```python
                    provider_enum = self.resolve_provider(provider_name)
                    adapter = await asyncio.wait_for(
                        self.get_adapter(provider_enum or provider_name),
                        timeout=_CAPABILITY_LISTING_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    adapter = None  # still initializing; get_adapter shields the load
                except _TTS_REGISTRY_ADAPTER_EXCEPTIONS as exc:
```

`find_adapter_for_requirements` (`:1119`), replace `adapter = await self.get_adapter(provider)` with:

```python
            try:
                adapter = await asyncio.wait_for(
                    self.get_adapter(provider),
                    timeout=_CAPABILITY_LISTING_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                continue  # still initializing: pick a provider that is ready now, without marking it failed
```

`tts_service_v2.py`: add `from . import adapter_registry` directly above `from .adapter_registry import (` (line 31), and in `_get_fallback_adapter`'s provider loop replace `adapter = await factory.registry.get_adapter(provider)` and add a timeout branch ahead of `except TTSProviderNotConfiguredError:`:

```python
                adapter = await asyncio.wait_for(
                    factory.registry.get_adapter(provider),
                    timeout=adapter_registry._CAPABILITY_LISTING_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                # Still initializing (the registry shields the load): fail over to a provider that is ready now.
                logger.debug("Skipping provider {} - its adapter is still initializing", provider.value)
                continue
            except TTSProviderNotConfiguredError:
```

A request that names the loading provider (`get_adapter` called directly, not through these four) still joins its initialization (TASK-13857 AC1).

No per-provider task table is needed: the shielded task keeps holding the base registry's per-provider `asyncio.Lock` (`Infrastructure/provider_registry.py:520-521`), so later callers queue behind it and then read the cache. (The clean-session review later added one: `TTSAdapterRegistry._initializations` holds the one in-flight task per provider, so a still-loading provider no longer adds a task per poll, and the listings skip a load that has already run the bound; see #3272's Review section.) `asyncio.shield` does not retrieve a late inner exception: once its outer caller is cancelled, CPython 3.12 removes the callback that would, so asyncio logs "Task exception was never retrieved". The task's own done-callback must call `task.exception()`. The `except _TTS_REGISTRY_ADAPTER_EXCEPTIONS` branch keeps `_schedule_retry` for real failures (ADR-011).

- [ ] **Step 6: Run the tests**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/TTS/test_tts_adapter_registry_wrapper_migration.py tldw_Server_API/tests/Infrastructure`
Expected: the registry file gives `28 passed`; Infrastructure passes. Mutation checks (probed, revert afterwards): removing `asyncio.shield` fails the 3 original new cases (probed: `assert 2 == 1` x2, `assert 3 == 1`); restoring `_schedule_retry` in the timeout branch fails `test_registry_capability_timeout_neither_restarts_nor_fails_a_slow_load`; removing each `wait_for` fails its own test (`list_capabilities`, `find_adapter_for_requirements`, `_get_fallback_adapter`). Also run the neighbours of the three bounded paths: `tests/TTS/test_tts_adapters.py`, `tests/TTS/test_tts_service_v2.py`, `tests/Setup/test_audio_health_helpers.py`, `tests/Audio/test_tts_runtime_hardening_boundaries.py`: `128 passed, 3 skipped` with the review's bound in `list_capabilities` (probed).

- [ ] **Step 7: Lint and commit**

```bash
$PY -m ruff check tldw_Server_API/app/core/TTS/adapter_registry.py tldw_Server_API/app/core/TTS/tts_service_v2.py tldw_Server_API/tests/TTS/test_tts_adapter_registry_wrapper_migration.py
git add tldw_Server_API/app/core/TTS/adapter_registry.py tldw_Server_API/app/core/TTS/tts_service_v2.py tldw_Server_API/tests/TTS/test_tts_adapter_registry_wrapper_migration.py
git add -A backlog/tasks
git commit -m "fix(tts): a capability-listing timeout no longer cancels or fails a slow adapter load (TASK-13857)

get_adapter shields the initialization from its callers, so the 5 s listing
timeout behind GET /audio/health and a client disconnect leave a model load
running for the next caller; the timeout no longer marks the provider failed.
The listings and failover skip a still-loading provider instead of waiting.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Expected: ruff reports only `adapter_registry.py` C420 (the `dict.fromkeys` suggestion), present on dev (Review M3: any I001 means the import block was replaced with a blank line above the first-party imports).

### Task B2: NeuTTS imports and loads its model on a worker thread

**Files:**
- Modify: `tldw_Server_API/app/core/TTS/adapters/neutts_adapter.py:34`, `:75-95` (`initialize`), new method before `get_capabilities` (`:151`); line numbers on #3264's head.
- Modify: `tldw_Server_API/tests/TTS/adapters/test_neutts_adapter.py` (new test before `test_neutts_runs_engine_calls_off_the_event_loop`)

**Interfaces:**
- Consumes: `run_tts_blocking_call` (`core/TTS/utils.py`); Task B1's registry, which no longer cancels the load.
- Produces: `NeuTTSAdapter._load_engine(self) -> Any`, a blocking method that imports the vendor module and returns the loaded `NeuTTSAir`.

- [ ] **Step 1: Write the failing test**

In `test_neutts_adapter.py`, before `@pytest.mark.asyncio` / `async def test_neutts_runs_engine_calls_off_the_event_loop`, add:

```python
class _LoadThreadRecordingEngine(_FakeNeuTTSEngine):
    """Records the thread its model load runs on (TASK-13857)."""

    load_threads: list[int] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_threads.append(threading.get_ident())


@pytest.mark.asyncio
async def test_neutts_imports_and_loads_its_model_off_the_event_loop(monkeypatch):
    """The vendor import and model load take seconds; on the loop they stall every request and job lease."""
    _install_fake_engine(_LoadThreadRecordingEngine)
    _LoadThreadRecordingEngine.load_threads = []
    import_threads: list[int] = []
    real_import = builtins.__import__

    def recording_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tldw_Server_API.app.core.TTS.vendors.neuttsair.neutts":
            import_threads.append(threading.get_ident())
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", recording_import)
    adapter = NeuTTSAdapter(config={})

    assert await adapter.initialize()
    loop_thread = threading.get_ident()
    assert import_threads and loop_thread not in import_threads
    assert _LoadThreadRecordingEngine.load_threads and loop_thread not in _LoadThreadRecordingEngine.load_threads
```

- [ ] **Step 2: Run it to see it fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/TTS/adapters/test_neutts_adapter.py`
Expected: `1 failed, 20 passed`; `test_neutts_imports_and_loads_its_model_off_the_event_loop` fails on `loop_thread not in import_threads` (the import ran on the loop thread).

- [ ] **Step 3: Move the import and load to a worker thread**

In `initialize()`, replace everything from `# Lazy import of vendored engine ...` through the closing `)` of `self._engine = _NeuTTSAir(...)`, including #3264's `ponytail:` comment, with:

```python
            logger.info(
                f"NeuTTS: loading backbone={self.backbone_repo} codec={self.codec_repo}"
            )
            self._engine = await run_tts_blocking_call(self._load_engine)
```

Before `async def get_capabilities` add:

```python
    def _load_engine(self) -> Any:
        """Import the vendored NeuTTS engine and load its model; blocks, so it runs on a worker thread."""
        # Lazy import of vendored engine to avoid global import failures
        from tldw_Server_API.app.core.TTS.vendors.neuttsair.neutts import NeuTTSAir

        return NeuTTSAir(
            backbone_repo=self.backbone_repo,
            backbone_device=self.backbone_device,
            codec_repo=self.codec_repo,
            codec_device=self.codec_device,
            # Respect auto_download toggle from provider config (default True)
            auto_download=parse_bool(self.config.get("auto_download", True), default=True),
        )
```

Import line 34: before PR A merges it already imports `run_tts_blocking_call` (`from ..utils import parse_bool, run_tts_blocking_call, run_tts_blocking_next`); leave it. If PR A merged first it reads `from ..utils import hold_tts_engine, parse_bool`; change it to `from ..utils import hold_tts_engine, parse_bool, run_tts_blocking_call`.

An `ImportError` raised on the thread still reaches `initialize()`'s `except ImportError` (`test_neutts_import_failure_log_sanitizes_exception_text` keeps passing).

- [ ] **Step 4: Run the tests and lint**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/TTS/adapters/test_neutts_adapter.py && $PY -m ruff check tldw_Server_API/app/core/TTS/adapters/neutts_adapter.py`
Expected: `21 passed` and `All checks passed!`.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/TTS/adapters/neutts_adapter.py tldw_Server_API/tests/TTS/adapters/test_neutts_adapter.py
git commit -m "fix(tts): NeuTTS imports and loads its model on a worker thread (TASK-13857)

The registry no longer cancels a slow load, so the load no longer has to
block the event loop to survive the capability listing's timeout.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task B3: Verify, close TASK-13857, open PR B

- [ ] **Step 1: Run every affected suite**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q -n 6 tldw_Server_API/tests/TTS tldw_Server_API/tests/TTS_NEW tldw_Server_API/tests/Docs tldw_Server_API/tests/Infrastructure tldw_Server_API/tests/Audio/test_audio_voices_endpoint_sanitization.py tldw_Server_API/tests/Audio/test_tts_credential_runtime_boundary.py tldw_Server_API/tests/Audio/test_tts_runtime_hardening_boundaries.py tldw_Server_API/tests/Audio/test_audio_tts_oauth_retry_unit.py tldw_Server_API/tests/LLM_Adapters/unit/test_llm_providers_capabilities_merge.py tldw_Server_API/tests/LLM_Calls/test_adapter_registry_wrapper_migration.py tldw_Server_API/tests/Persona/test_persona_kitten_model_preparation.py tldw_Server_API/tests/Research_Workspace/test_capability_derivation.py tldw_Server_API/tests/Services/test_startup_heavy_init.py tldw_Server_API/tests/Setup/test_audio_health_helpers.py`
(`get_adapter`'s behaviour changes for every provider, so this runs every test that references `TTS.adapter_registry`, `TTSAdapterRegistry`, `get_all_capabilities` or `neutts`: `git grep -l -E "TTS\.adapter_registry|TTS import adapter_registry|TTSAdapterRegistry|get_all_capabilities|neutts" -- tldw_Server_API/tests`. The grep misses `tests/Audio/test_audio_tts_oauth_retry_unit.py`, which hits `/audio/providers`; it is in the list above, 31 tests, passing on PR B in the dry run (Review). Also grep `list_capabilities|_get_fallback_adapter|find_adapter_for_requirements|get_best_adapter` over `tldw_Server_API/tests` and run every hit not already listed.)
Expected: `7 failed, 2611 passed, 28 skipped, 1 xfailed, 1 xpassed`; the 7 are the baseline list. (The dry run measured `2577 passed` for the list without the oauth file; the added 31 tests and this plan's 3 further new registry-file tests give 2611, computed rather than re-run as one sweep. If your count differs by those amounts only, it is fine; any other failure, re-run on unchanged origin/dev.)

- [ ] **Step 2: Stability and Bandit**

```bash
for i in 1 2 3; do TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/TTS/test_tts_adapter_registry_wrapper_migration.py tldw_Server_API/tests/TTS/adapters/test_neutts_adapter.py | tail -1; done
uvx bandit -q -ll tldw_Server_API/app/core/TTS/adapter_registry.py tldw_Server_API/app/core/TTS/adapters/neutts_adapter.py
```

Expected: `49 passed` three times (28 registry + 21 NeuTTS); Bandit prints nothing.

- [ ] **Step 3: Close TASK-13857**

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13857 \
  --append-notes "Fixed: TTSAdapterRegistry.get_adapter awaits asyncio.shield(self._base.get_adapter_async(...)), so a caller that stops waiting (the capability listing's 5 s timeout, now adapter_registry._CAPABILITY_LISTING_TIMEOUT_SECONDS, or a client disconnect) no longer cancels the initialization; the shielded task keeps the base registry's per-provider lock, so later callers wait on it and then share the adapter rather than starting a second load. get_all_capabilities logs a timeout at info and no longer calls _schedule_retry for it; list_capabilities, find_adapter_for_requirements and TTSServiceV2._get_fallback_adapter wait at most that timeout for a provider and skip one still loading, without marking it failed (a request that names the loading provider still joins its initialization); the shielded task is kept referenced in _PENDING_INITIALIZATIONS; real initialization errors still mark the provider failed for adapter_failure_retry_seconds (ADR-011). NeuTTS now imports its vendor module and loads its model in _load_engine on a worker thread (run_tts_blocking_call); #3264's ponytail comment is gone. test_registry_closes_adapter_and_reraises_initialization_cancellation now cancels _initialize_adapter directly: a caller's cancellation no longer reaches the initialization, and loop shutdown still closes the half-built adapter." \
  --append-notes "Tests: test_registry_capability_timeout_neither_restarts_nor_fails_a_slow_load (an adapter that loads on a worker thread for longer than the listing timeout: one load, provider still enabled, adapter available to the next request) RED on dev (provider marked failed); test_registry_list_capabilities_does_not_wait_out_a_slow_load, test_registry_best_adapter_skips_a_provider_that_is_still_loading and test_service_fallback_skips_a_provider_that_is_still_loading RED after the shield alone (TimeoutError from the test's own wait_for: the callers waited out the load), GREEN with the bound; test_registry_cancelled_caller_leaves_the_initialization_to_the_next_caller (task cancel and anyio cancel scope) RED on dev (assert 2 == 1, a second adapter built); test_neutts_imports_and_loads_its_model_off_the_event_loop RED on dev (import on the loop thread). Mutations: dropping the shield fails the 3 original registry cases; keeping _schedule_retry on timeout fails the capability test; removing a wait_for fails its own test. Registry and NeuTTS files: 49 passed in three runs. tests/TTS, tests/TTS_NEW, tests/Docs, tests/Infrastructure and the 10 other files that reference the registry: 7 failed, 2611 passed, 28 skipped, 1 xfailed, 1 xpassed; the 7 fail on unchanged dev too (No module named 'av'; torch not installed). Ruff: no new findings. Bandit -ll (uvx): clean. ADR assessment: none needed; consistent with ADR-011 (a still-running initialization is not a failure). No docs describe the listing timeout." \
  --check-ac 1 --check-ac 2 --check-ac 3 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "A slow local model load now survives the 5 s capability listing behind GET /audio/health and /audio/providers, and a client disconnect: it is neither cancelled, restarted nor marked failed, and the next request gets the adapter once it loads. With that, NeuTTS imports and loads its model on a worker thread instead of blocking the event loop once per process." \
  -s Done
git add -A backlog/tasks
git commit -m "docs(backlog): close TASK-13857

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4: Push and open PR B**

```bash
git push -u origin fix/tts-registry-slow-load
gh pr create --base dev --title "fix(tts): capability listing no longer cancels or fails a slow model load; NeuTTS loads off the loop (TASK-13857)" --body "$(cat <<'EOF'
## Summary
- `TTSAdapterRegistry.get_adapter` shields the adapter initialization from its callers: the 5 s capability-listing timeout (GET /audio/health, /audio/providers) and client disconnects no longer cancel a model load, and later callers wait for it instead of loading again.
- The listing's timeout no longer marks the provider failed for `adapter_failure_retry_seconds` (300 s by default); real initialization errors still do (ADR-011).
- The capability listings and failover (`list_capabilities`, `find_adapter_for_requirements`, `_get_fallback_adapter`) wait at most the same 5 s for one provider and skip one that is still loading, so failover picks a provider that is ready now. Requests that name the loading provider still join its initialization.
- NeuTTS imports its vendor module and loads its model on a worker thread (#3264 had kept it on the loop because of the registry problem).

## Test plan
- [x] Registry: slow load outlasting the listing timeout -> one load, provider enabled, adapter served next; cancelled caller (task cancel / anyio scope) -> one adapter. Both RED on dev.
- [x] GET /audio/health's `list_capabilities`, `find_adapter_for_requirements` and the service's failover skip a still-loading provider within the listing timeout instead of waiting out its load. RED after the shield alone.
- [x] NeuTTS: vendor import and model load run off the loop thread. RED on dev.
- [x] tests/TTS, tests/TTS_NEW, tests/Docs, tests/Infrastructure and every test referencing the registry: only the 7 baseline failures (PyAV/torch not installed), same on dev
- [x] ruff (no new findings), Bandit -ll clean

## Change summary
(owner, per ADR-004)

## ADR assessment
None needed; consistent with ADR-011.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: Clean review, then merge per the repo's merge rules**

Run `/superpowers:requesting-code-review` in a clean subagent on the PR diff; address findings before merge. Merge after the owner's Change summary (Decisions for the owner, 1). Merge through the serial queue (the owner's `pr_queue.sh`), one PR at a time; MERGE_QUEUE is unset. Then, if PR A is still open, rebase it, apply Task A2 Step 3's second branch (the import line gains `run_tts_blocking_call`), and push before it is queued. Do not skip this step: the rebase is textually clean but broken (ruff F821 on `run_tts_blocking_call`, 21 NeuTTS test failures; verified in the dry run) (Review M7).

## Self-review

- TASK-13856 AC1 (helper in utils, releases only when the thread returns, stream variant, NeuTTS uses it): Tasks A1, A2. AC2 (each local adapter with the shape, task-cancel and anyio-scope tests checking the second call waits): Tasks A1, A3, A4; Kokoro, the one site that cannot take the helper as-is, is filed in Task A5.
- TASK-13857 AC1 (timeout neither cancels nor fails; later callers join): Task B1 Step 5. AC2 (one load, adapter available next): Task B1's capability test; the three bound tests keep the listing and failover callers from waiting on it. AC3 (NeuTTS import and load off the loop, with a test): Task B2.
- Names are consistent across tasks: `hold_tts_engine`, `run`, `_release_engine_lock`, `_Gate`, `_assert_cancelled_call_keeps_the_engine`, `CANCEL_MODES`, `_WAV_HEADER`, `_omnivoice_runtime`, `_CAPABILITY_LISTING_TIMEOUT_SECONDS`, `_PENDING_INITIALIZATIONS`, `_SlowLoadAdapter`, `_load_engine`.
- PR B's registry changes (shield, `_PENDING_INITIALIZATIONS`, the three bounds, the service edit) and the three bound tests were assembled in a scratch worktree on origin/dev: RED (`TimeoutError`) with the shield alone, GREEN with the bounds; the registry file, `test_tts_adapters.py` and `test_tts_service_v2.py` gave `103 passed, 3 skipped` plus the one expected failure (the not-yet-retargeted cancellation test); ruff showed only C420.
- The test module's three stages (Task A1, A3, A4 code exactly as written here) were assembled in a scratch worktree: ruff clean at each stage, and `4`, `14`, `24` passed with the fixes applied.
