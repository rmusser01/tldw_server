# Kokoro Model Load Keeps Its Lock (TASK-13860) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Review folded in (2026-10-20).** The independent review (`review_plan_fu_kokoro_load.md`, verdict "ready with fixes": 0 Blocker, 1 Important, 8 Minor) is applied as the coordinator ruled. What changed:
> - **I1.** Option B now states its eager-path cost, and claims to close only the retry case of Review Focus 1.
> - **M1.** The decision says the knob does nothing in a default install.
> - **M2.** Option C's scope and cost are corrected.
> - **M3.** Request-owned adapters are added to the known limits, and the "+300 s" implication is dropped.
> - **M4.** `ensure_initialized` is added to both entry parametrizations.
> - **M5.** The ONNX × bounded duplicates are dropped. One ONNX case now actually pins "ONNX ignores the bound".
> - **M6.** Every command block uses the full python path.
> - **M7.** The test helper removes `KOKORO_INIT_TIMEOUT_SEC` from the environment.
> - **M8.** Taken: requires-python is >= 3.12 and Ruff does not flag it. `_ensure_model_loaded` is now one method using `asyncio.timeout`, and there is no `_load_model_holding_lock`.
>
> All counts were re-measured on origin/dev df7c290fbf. RED: 21 failed, 1 passed (the ONNX pin, which holds on dev too). GREEN: 22 passed.

**Goal:** A Kokoro model load keeps `_model_lock` until its load thread returns. Neither a cancelled request nor an `init_timeout_sec` expiry then lets a second load start beside the running one. This holds on both backends (ONNX, PyTorch) and on every load path: `ensure_initialized()`/`initialize()` (eager) and `_ensure_model_loaded()` (lazy).

**Architecture:**
- Every Kokoro model load goes through `_ensure_model_loaded()`. It holds `_model_lock` with `hold_tts_engine` from `core/TTS/utils.py`, the helper TASK-13856 gave the other local adapters.
- It hands the helper's `run` down to `_initialize_onnx` and `_initialize_pytorch`. Those make their worker-thread calls through `run` instead of `asyncio.to_thread`. A holder that leaves mid-load therefore leaves the lock with the thread.
- `initialize()` loads through `_ensure_model_loaded()`.
- For PyTorch, `init_timeout_sec` becomes an `asyncio.timeout` around the whole hold. It bounds a caller's wait, queueing included, without freeing the lock.

**Tech Stack:** Python 3.12 (`requires-python = ">=3.12"`), asyncio, anyio (Starlette's cancel scopes), pytest + pytest-asyncio, backlog-py.

**Spec:** `backlog/tasks/task-13860 - Kokoro-s-init-timeout-sec-abandons-a-running-model-load-so-a-second-load-can-start-alongside-it.md`.

Background:
- `backlog/tasks/task-13856 - Local-TTS-adapters-hand-the-engine-to-the-next-request-while-a-cancelled-request-s-call-still-runs.md` (the helper).
- `backlog/tasks/task-13857 - TTS-capability-listing-cancels-slow-local-model-loads-and-marks-the-provider-failed.md` (the registry).

## Decisions for the owner

**1. What `init_timeout_sec` means after the fix. Recommendation: A. The plan implements A.**

Scope first: in a default install the knob does nothing.
- The shipped `Config_Files/tts_providers_config.yaml:111-113` selects `use_onnx: true`, and ONNX ignores `init_timeout_sec`, as on dev and after this fix.
- So A, B and C differ only for PyTorch deployments that set `kokoro_init_timeout_sec` or `KOKORO_INIT_TIMEOUT_SEC`.
- The lock fix itself (AC1/AC2) applies to every deployment, whichever option is chosen.

**A (recommended).**
- What it means:
  - `init_timeout_sec` bounds how long one caller waits for the model, including time spent queueing behind another caller's load.
  - A load that outlasts the bound is not stopped. It keeps `_model_lock` until its thread returns. Its model is then discarded, and the next caller loads again after it, not beside it.
  - The knob, its default (off) and its scope (PyTorch only) stay as they are.
- Why:
  - It is the smallest change that meets both ACs, with the same shape and the same documented limit as the other `hold_tts_engine` sites.
  - Every caller's wait stays bounded. On dev that held only through the bug: the holder dropped the lock at the bound, so a queued caller waited about 2× the bound.
  - It keeps failover working on the eager path, which is the default. At the bound, `initialize()` returns False, the registry marks Kokoro failed, and requests fail fast or fall back.
  - A narrower variant would bound only the load, not the queue. There the second caller waits for a hung thread with no limit. The mutation check in Task 2 fails on exactly that variant.
- Cost: a load that always takes longer than the bound never succeeds. That is already true on dev. The fix is to raise the bound or unset it.

**B. Drop the timeout.**
- What it means: remove `kokoro_init_timeout_sec` and `KOKORO_INIT_TIMEOUT_SEC`, and regenerate `Env_Vars.md:861`. Every load runs to completion for whoever waits.
- Cost on the eager path (the default):
  - Every `get_adapter("kokoro")` joins the registry's one shielded in-flight init (`adapter_registry.py:664-675`).
  - `tts_service_v2._get_adapter` (`:3401-3437`) awaits that with no bound.
  - So a hung PyTorch load holds every Kokoro request with no limit, and `fallback=True` never gets a turn, because `_get_adapter` never returns.
- Cost on the lazy path: a request (`generate()`, the Persona voice warm-up) queued behind a hung load waits until the client gives up.
- What it closes: only the retry case of Review Focus 1. Unload/re-register and request-owned adapters remain.
- Plan change if chosen:
  - Task 1 Step 3 drops the bound from `_ensure_model_loaded`.
  - The timeout tests and the ONNX pin go.
  - Task 2 runs `Helper_Scripts/generate_env_vars_md.py`.

**C. A, plus keep a model whose caller left.**
- What it means: the thread parks the raw model in a pending attribute. The next holder, under the lock, finishes the PyTorch device move and the resource-manager registration on the loop. This is the upgrade path in `hold_tts_engine`'s `ponytail:` comment.
- Scope: it saves any lazy load whose caller left, on both backends. That covers a client disconnect in `generate()`, a Persona `voice_stop` and a PyTorch timeout.
- Cost: about one pending attribute and one branch, not a restructure.
- What it does not help: the eager path. After a failed eager init the registry closes and drops that adapter (`_initialize_adapter`'s finally), so the model is lost anyway.

## Measured on origin/dev

Measured at 2b925562e5 and again at df7c290fbf. No TTS, Persona or pyproject file differs between them, or from a6719bf24f.

**The lazy load: `_ensure_model_loaded`** (`kokoro_adapter.py:356-370`).
- It holds `_model_lock` with a plain `async with` (`:359`).
- ONNX (`:362-363`) has no timeout.
- PyTorch: `asyncio.wait_for(self._load_pytorch_model(), timeout=self.init_timeout_sec)` (`:364-369`) wraps only the load. The lock wait stays outside it.

**The eager load: `initialize()`** (`:372-413`).
- It loads at `:384-394` with no `_model_lock` at all.
- Its only guard is `TTSAdapter.ensure_initialized`'s `_init_lock` (`adapters/base.py:412`), and a cancelled caller releases that lock.
- The registry and `generate()` (`:770`) enter through `ensure_initialized()`.
- With lazy init on, `initialize()` does not load (`:375-382`).

**Thread calls in the load.**
- ONNX: `asyncio.to_thread(Kokoro, ...)` at `:511`, `:519` and `:526`. The third is the `TypeError` fallback.
- PyTorch: `asyncio.to_thread(_load_model)` at `:666`.
- These run on the event loop instead, so they block it and cannot overlap another load:
  - the PyTorch sync fallback (`:667-670`);
  - the device move (`:671-683`);
  - the generic `torch.jit.load`/`torch.load` path (`:700-724`).

**The knob: `init_timeout_sec`** (`:309-316`).
- Read from `kokoro_init_timeout_sec`, falling back to `os.getenv("KOKORO_INIT_TIMEOUT_SEC")`. Default `None`; `0` also means off.
- `git grep` finds it only in `kokoro_adapter.py`, the `Env_Vars.md:861` name list and the TASK-13856 plan.
- No test pins it, and no guide or config file describes it.
- The shipped `tts_providers_config.yaml:111-113` runs Kokoro on ONNX, which ignores it.

**The helper: `hold_tts_engine(lock)`** (`core/TTS/utils.py:130-162`).
- It yields `run(func, *args, **kwargs)`, which runs `asyncio.shield(ensure_future(run_tts_blocking_call(...)))`.
- On exit, if the last call is still running, it releases the lock from that call's done-callback.
- Its finally never awaits, so an anyio scope cannot re-cancel it.
- Its `ponytail:` comment says a cancelled load's result is discarded and the next holder loads again.

**Registry (TASK-13857).**
- `get_adapter` (`adapter_registry.py:664-675`) runs one shielded in-flight `get_adapter_async` task per (provider, generation). That task calls `adapter.ensure_initialized()` (`:903`).
- The registry puts no timeout of its own on `initialize()`. Only `_get_adapter_if_ready` (`:701-715`, used by the listings) bounds its wait, at `_CAPABILITY_LISTING_TIMEOUT_SECONDS = 5.0` (`:59`), and it never cancels the load.
- So `init_timeout_sec` is the only thing that ends a Kokoro init early. When it does:
  1. `initialize()` returns False.
  2. `_materialize_adapter_async` raises.
  3. `ProviderRegistryBase._mark_failure_locked` (`Infrastructure/provider_registry.py:347-352`) blocks retries for `adapter_failure_retry_seconds`. That is 300 in `Config_Files/tts_providers_config.yaml:629`, and forever when unset.
  4. The retry builds a new `KokoroAdapter`, with a new `_model_lock`.
- `create_adapter_with_overrides` (`:717-789`) builds a fresh adapter, with its own lock, for each request that carries provider overrides (`tts_service_v2.py:3415-3437`).

**Other callers of the lazy load.**
- `generate()` (`:775-776`).
- The Persona voice warm-up, `core/Persona/live_tts.py:162` (`await adapter._ensure_model_loaded()`), which `voice_stop` cancels.

**Tests that touch the changed signatures** (`git grep` over the repo).
- `tests/TTS/adapters/test_kokoro_adapter_mock.py:233` and `:249` patch `_load_pytorch_model` and `_load_onnx_model` with `AsyncMock`. Their `assert_called_once()` ignores arguments, so they still pass.
- `tests/Persona/test_persona_live_voice_runtime.py:323` calls `adapter._initialize_onnx()` with no arguments, so it has to change.

**Baselines.**
- The 12 test files that reference the Kokoro adapter give `186 passed, 8 skipped` on dev.
- Ruff on the three touched files reports one `I001`, at `test_persona_live_voice_runtime.py:86`. It is there on dev too.
- `Helper_Scripts/ci/scope_predicate_baseline.txt` lists none of the touched files.

## Global Constraints

- Branch `fix/kokoro-load-keeps-lock` from `origin/dev`. One PR, base `dev`.
- Reuse `hold_tts_engine` unchanged, and add no dependency.
- `init_timeout_sec` stays off by default and stays PyTorch-only. ONNX ignores it, as on dev; a test pins that.
- Fix test-first: the new lock tests fail on dev for the right reason before the fix lands.
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never use `--no-verify`.
- Change the backlog only through backlog-py, with `--append-notes`, then `git add -A backlog/tasks`. Never hand-edit task files.
- Use the full path in every command, because shell variables do not survive between an agent's Bash calls:
  - python: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`;
  - tests run with `TLDW_TEST_NO_DOCKER=1`.
- No mirrored doc changes, and no line-baseline regeneration: none of the touched files is listed.

## Review Focus

These are the failure modes the new tests do not exercise, most likely first.

1. **Two adapter instances.** The lock is per adapter instance, like every other `hold_tts_engine` site. A new `KokoroAdapter` gets a fresh `_model_lock`, and its load can run beside a thread from an earlier instance. That happens in three cases:
   - a request carrying provider overrides, through `create_adapter_with_overrides`, which builds and initializes its own adapter per request (pre-existing, shared by every local adapter);
   - an unload or re-registration (a generation bump) while a load thread still runs;
   - a registry retry after `adapter_failure_retry_seconds` (300 s shipped), if the timed-out load's thread is still running by then.

   The close-out notes record all three as known limits. A per-model-path lock would close them all. Option B would close only the third.
2. **Real `kokoro_onnx` / `kokoro.model.KModel`.** The tests use fakes that block on a gate. `_capture_kokoro_repo_warning` swaps `builtins.print` around `await run(_load_model)`. A cancel restores `print` while the thread still runs, so a late repo_id warning prints raw. This is cosmetic and the same on dev.
3. **The PyTorch sync fallback** (`:667-670` on dev). If the load thread raises a non-critical error, the load is run again synchronously on the event loop, under the lock. That blocks the loop for a whole load but cannot overlap another load. Unchanged, and not tested here.
4. **Loop-bound PyTorch work.** The generic `torch.jit.load`/`torch.load` path (used when `kokoro` is not installed) and the device move run on the loop, not through `run`. They block it and cannot be cancelled mid-call, so they cannot overlap. Unchanged.
5. **A hung ONNX load.** ONNX ignores `init_timeout_sec`, as on dev (pinned by `test_kokoro_onnx_load_ignores_init_timeout_sec`). A hung ONNX constructor keeps `_model_lock` for good, and every lazy-load request after it waits with no bound. Applying the bound to ONNX would be a one-condition change, if the owner wants it.

## ADR assessment

No ADR is needed. This is a local concurrency fix inside one adapter, using an existing helper. No durable rule, public API or dependency changes. Docs/ADR/README.md has no ADR on adapter locks or model loads, and ADR-011 (audio API semantics) is untouched.

## Docs

Nothing to change under option A.
- `KOKORO_INIT_TIMEOUT_SEC` appears only as a name in the generated `Env_Vars.md` list (`:861`). No guide describes it.
- The new `_ensure_model_loaded` docstring states the meaning.
- Under option B, regenerate `Env_Vars.md` in Task 2.

---

### Task 1: Kokoro's model load holds `_model_lock` through `hold_tts_engine`

**Files:**
- Test: `tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py` (append after `:411`)
- Modify: `tldw_Server_API/app/core/TTS/adapters/kokoro_adapter.py` at `:13`, `:41`, `:356-370`, `:384-394`, `:415-416`, `:511`, `:519`, `:526`, `:631-632`, `:666` and `:733-737`
- Modify: `tldw_Server_API/tests/Persona/test_persona_live_voice_runtime.py:323`

**Interfaces:**
- Consumes:
  - `hold_tts_engine(lock: asyncio.Lock)` (`core/TTS/utils.py:130`): an async context manager yielding `run(func, /, *args, **kwargs) -> Awaitable[Any]`.
  - From the test file: `_Gate` (`:30-40`), `_assert_cancelled_call_keeps_the_engine(call, gate, *, via_anyio_scope)` (`:43-85`) and `CANCEL_MODES` (`:27`).
- Produces:
  - `KokoroAdapter._ensure_model_loaded() -> bool`: same signature, new meaning.
  - `_load_onnx_model(run)`, `_load_pytorch_model(run)`, `_initialize_onnx(run)` and `_initialize_pytorch(run)`: each now requires `run: Callable[..., Awaitable[Any]]`.
  - `initialize()`: loads through `_ensure_model_loaded()`.

- [ ] **Step 0: Branch**

```bash
git fetch origin
git switch -c fix/kokoro-load-keeps-lock origin/dev
```

- [ ] **Step 1: Write the failing tests**

Append the code below to `tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py`, after `test_kitten_tts_runtime_load_keeps_the_lock_until_the_load_returns`. The file already imports `asyncio`, `sys`, `types` and `pytest`. It already defines `_Gate`, `CANCEL_MODES` and `_assert_cancelled_call_keeps_the_engine`.

```python


def _kokoro_adapter(monkeypatch, tmp_path, gate: _Gate, *, backend: str, init_timeout_sec: float | None = None):
    """A Kokoro adapter whose model load (the ONNX constructor or KModel) blocks on the gate."""
    from tldw_Server_API.app.core.TTS.adapters import kokoro_adapter as kokoro_mod

    async def _resource_manager():
        return types.SimpleNamespace(register_model=lambda **_kwargs: None)

    monkeypatch.setattr(kokoro_mod, "get_resource_manager", _resource_manager)
    # The adapter falls back to this env var; a shell that sets it would bound the "unbounded" cases.
    monkeypatch.delenv("KOKORO_INIT_TIMEOUT_SEC", raising=False)
    config: dict[str, object] = {"kokoro_device": "cpu", "kokoro_lazy_init": False}
    if init_timeout_sec is not None:
        config["kokoro_init_timeout_sec"] = init_timeout_sec
    if backend == "onnx":

        class _Kokoro:
            def __init__(self, *_args, **_kwargs):
                gate.enter("load")

        monkeypatch.setitem(
            sys.modules, "kokoro_onnx", types.SimpleNamespace(Kokoro=_Kokoro, EspeakConfig=lambda **_kwargs: None)
        )
        model, voices = tmp_path / "model.onnx", tmp_path / "voices.bin"
        model.touch()
        voices.touch()
        config.update(kokoro_use_onnx=True, kokoro_model_path=str(model), kokoro_voices_json=str(voices))
    else:

        class _KModel:
            def __init__(self, **_kwargs):
                gate.enter("load")

            def eval(self):
                return self

            def cpu(self):
                return self

        model_module = types.ModuleType("kokoro.model")
        model_module.KModel = _KModel
        monkeypatch.setitem(sys.modules, "kokoro", types.ModuleType("kokoro"))
        monkeypatch.setitem(sys.modules, "kokoro.model", model_module)
        monkeypatch.setattr(kokoro_mod, "safe_import_torch", lambda: types.SimpleNamespace())
        model = tmp_path / "kokoro-v1_0.pth"
        model.touch()
        (tmp_path / "config.json").write_text("{}")
        config.update(kokoro_use_onnx=False, kokoro_model_path=str(model))
    return kokoro_mod.KokoroAdapter(config)


# ensure_initialized() is the production entry (the registry and generate()); initialize() is what it runs.
_KOKORO_ENTRIES = pytest.mark.parametrize("entry", ["ensure_initialized", "initialize", "_ensure_model_loaded"])


@pytest.mark.asyncio
@CANCEL_MODES
@_KOKORO_ENTRIES
@pytest.mark.parametrize(
    ("backend", "init_timeout_sec"),
    [("onnx", None), ("pytorch", None), ("pytorch", 30.0)],
    ids=["onnx", "pytorch-unbounded", "pytorch-bounded"],
)
async def test_kokoro_model_load_keeps_the_lock_until_the_load_returns(
    monkeypatch, tmp_path, backend, init_timeout_sec, entry, via_anyio_scope
):
    """A cancelled Kokoro load keeps the next load waiting for its thread.

    "pytorch-bounded" cancels the caller while it waits inside init_timeout_sec.
    """
    gate = _Gate()
    adapter = _kokoro_adapter(monkeypatch, tmp_path, gate, backend=backend, init_timeout_sec=init_timeout_sec)

    await _assert_cancelled_call_keeps_the_engine(getattr(adapter, entry), gate, via_anyio_scope=via_anyio_scope)


@pytest.mark.asyncio
@_KOKORO_ENTRIES
async def test_kokoro_load_past_init_timeout_keeps_the_lock_until_the_load_returns(monkeypatch, tmp_path, entry):
    """init_timeout_sec bounds each caller's wait; the lock stays with the load until its thread returns."""
    gate = _Gate()
    adapter = _kokoro_adapter(monkeypatch, tmp_path, gate, backend="pytorch", init_timeout_sec=0.1)
    call = getattr(adapter, entry)
    try:
        # The first load outlasts the timeout; its thread keeps running.
        assert await asyncio.wait_for(call(), 5) is False
        # The next caller waits behind that thread, up to its own timeout, and starts no load.
        assert await asyncio.wait_for(call(), 5) is False
        assert gate.started == ["load"]
    finally:
        gate.release.set()
    # Only the stuck load had to outlast the bound; give the reload room on a slow runner.
    adapter.init_timeout_sec = 5.0
    # The timed-out load's model was discarded, so once its thread returns the next caller loads again.
    assert await asyncio.wait_for(call(), 5) is True
    assert gate.started == ["load", "load"]


@pytest.mark.asyncio
async def test_kokoro_onnx_load_ignores_init_timeout_sec(monkeypatch, tmp_path):
    """init_timeout_sec is PyTorch-only: an ONNX load that outlasts it still completes for its caller."""
    gate = _Gate()
    adapter = _kokoro_adapter(monkeypatch, tmp_path, gate, backend="onnx", init_timeout_sec=0.1)
    load = asyncio.create_task(adapter._ensure_model_loaded())
    try:
        await asyncio.sleep(0.3)  # three times the bound
        assert not load.done()
    finally:
        gate.release.set()
    assert await asyncio.wait_for(load, 5) is True
```

That is 22 Kokoro cases:
- the cancel test: 3 loads × 3 entries × 2 cancel modes = 18 cases;
- the timeout test: 3 cases, one per entry;
- the ONNX pin: 1 case.

Notes for the implementer:
- `"kokoro_lazy_init": False` and an explicit `kokoro_model_path` keep `initialize()` loading under pytest. `__init__` turns on lazy init in test mode only when no model path is configured (`kokoro_adapter.py:270-276`).
- `"kokoro_device": "cpu"` skips the device probe.
- The PyTorch fake replaces `safe_import_torch`. torch is only used there for the MPS move.
- `test_kokoro_onnx_load_ignores_init_timeout_sec` pins behaviour that dev already has. It passes before and after the fix, and it fails if the bound is ever applied to ONNX (mutation 3 in Task 2).

- [ ] **Step 2: Run the tests to verify they fail**

```bash
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q -k kokoro tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py 2>&1 | grep -E "^E   AssertionError|passed|failed" | sort | uniq -c
```

Expected (measured on dev):

```
   1 =========== 21 failed, 1 passed, 24 deselected, 30 warnings in 7.95s ===========
   3 E   AssertionError: assert ['load', 'load'] == ['load']
  18 E   AssertionError: assert 2 == 1
```

Both messages mean a second load reached the engine while the first load's thread still ran.
- In the `initialize` cases on dev, the second load starts even before the cancel, because `initialize()` takes no `_model_lock`.
- In the `ensure_initialized` cases, `_init_lock` holds the second load back until the cancel releases it. That is exactly the AC's failure.
- The one pass is `test_kokoro_onnx_load_ignores_init_timeout_sec` (see Step 1).

Any other failure, such as an import error, means the test is failing for the wrong reason. Fix the test before going on.

- [ ] **Step 3: Implement**

All edits are in `tldw_Server_API/app/core/TTS/adapters/kokoro_adapter.py`.

(a) Line 13. Replace:

```python
from collections.abc import AsyncGenerator
```

with:

```python
from collections.abc import AsyncGenerator, Awaitable, Callable
```

(b) Line 41. Replace:

```python
from ..utils import parse_bool, run_tts_blocking_next
```

with:

```python
from ..utils import hold_tts_engine, parse_bool, run_tts_blocking_next
```

(c) Lines 356-370. Replace the whole `_ensure_model_loaded`:

```python
    async def _ensure_model_loaded(self) -> bool:
        if self._model_is_loaded():
            return True
        async with self._model_lock:
            if self._model_is_loaded():
                return True
            if self.use_onnx:
                return await self._load_onnx_model()
            if self.init_timeout_sec:
                try:
                    return await asyncio.wait_for(self._load_pytorch_model(), timeout=self.init_timeout_sec)
                except asyncio.TimeoutError:
                    logger.error(f"{self.provider_name}: Lazy model load timed out after {self.init_timeout_sec}s")
                    return False
            return await self._load_pytorch_model()
```

with:

```python
    async def _ensure_model_loaded(self) -> bool:
        """Load the model once; with init_timeout_sec (PyTorch only), a caller waits at most that long.

        The wait includes queueing behind another caller's load. hold_tts_engine keeps _model_lock
        with a load thread until it returns, even after its caller left, so no second load starts beside it.
        """
        if self._model_is_loaded():
            return True
        bound = None if self.use_onnx else (self.init_timeout_sec or None)
        try:
            async with asyncio.timeout(bound), hold_tts_engine(self._model_lock) as run:
                if self._model_is_loaded():
                    return True
                if self.use_onnx:
                    return await self._load_onnx_model(run)
                return await self._load_pytorch_model(run)
        except TimeoutError:
            if bound is None:
                raise  # raised by the load itself, not by the bound
            logger.error(f"{self.provider_name}: Model load timed out after {bound}s")
            return False
```

Why this is safe:
- `asyncio.timeout(None)` sets no bound.
- The timeout context wraps the hold, so it covers both the queue at `lock.acquire()` and the load.
- At the bound, the cancel lands at `await run(...)`, which is `await asyncio.shield(...)`, so the thread is untouched. The hold's finally then leaves the lock to the thread's done-callback.
- A timeout that fires while a caller is still queued leaves the lock alone.
- An anyio cancel passes through `asyncio.timeout` as a plain `CancelledError`.
- The `bound is None` re-raise keeps today's behaviour when no bound is set: a `TimeoutError` raised by the load itself still propagates. On dev only the PyTorch sync fallback can raise one.

(d) Lines 384-394, inside `initialize()`. Replace:

```python
            if self.use_onnx:
                success = await self._load_onnx_model()
            else:
                if self.init_timeout_sec:
                    try:
                        success = await asyncio.wait_for(self._load_pytorch_model(), timeout=self.init_timeout_sec)
                    except asyncio.TimeoutError:
                        logger.error(f"{self.provider_name}: Initialization timed out after {self.init_timeout_sec}s")
                        success = False
                else:
                    success = await self._load_pytorch_model()
```

with:

```python
            success = await self._ensure_model_loaded()
```

(e) Lines 415-416. Replace:

```python
    async def _initialize_onnx(self) -> bool:
        """Initialize ONNX backend"""
```

with:

```python
    async def _initialize_onnx(self, run: Callable[..., Awaitable[Any]]) -> bool:
        """Initialize ONNX backend, constructing the model through ``run`` from hold_tts_engine."""
```

(f) Lines 511, 519 and 526. In each of the three statements, replace `self.kokoro_instance = await asyncio.to_thread(` with `self.kokoro_instance = await run(`. The argument lists stay as they are. The result:

```python
            if voices_json_arg:
                self.kokoro_instance = await run(
                    Kokoro,
                    self.model_path,
                    voices_json_arg,
                    espeak_config=espeak_config
                )
            else:
                try:
                    self.kokoro_instance = await run(
                        Kokoro,
                        self.model_path,
                        espeak_config=espeak_config
                    )
                except TypeError:
                    # Fallback: pass empty string for voices path if constructor requires it
                    self.kokoro_instance = await run(
                        Kokoro,
                        self.model_path,
                        "",
                        espeak_config=espeak_config
                    )
```

(g) Lines 631-632. Replace:

```python
    async def _initialize_pytorch(self) -> bool:
        """Initialize PyTorch backend"""
```

with:

```python
    async def _initialize_pytorch(self, run: Callable[..., Awaitable[Any]]) -> bool:
        """Initialize PyTorch backend, loading the KModel through ``run`` from hold_tts_engine."""
```

Then, at line 666, replace:

```python
                    self.kokoro_pt_model = await asyncio.to_thread(_load_model)
```

with:

```python
                    self.kokoro_pt_model = await run(_load_model)
```

(h) Lines 733-737. Replace:

```python
    async def _load_onnx_model(self) -> bool:
        return await self._initialize_onnx()

    async def _load_pytorch_model(self) -> bool:
        return await self._initialize_pytorch()
```

with:

```python
    async def _load_onnx_model(self, run: Callable[..., Awaitable[Any]]) -> bool:
        return await self._initialize_onnx(run)

    async def _load_pytorch_model(self, run: Callable[..., Awaitable[Any]]) -> bool:
        return await self._initialize_pytorch(run)
```

Check that no `asyncio.to_thread` is left in the adapter:

```bash
grep -n "to_thread" tldw_Server_API/app/core/TTS/adapters/kokoro_adapter.py
```

Expected: no output.

- [ ] **Step 4: Point the Persona test at the load path**

`tldw_Server_API/tests/Persona/test_persona_live_voice_runtime.py:323` calls `_initialize_onnx()`, which now requires `run`. Load through the real path instead. The test still checks that the constructor runs off the loop thread. Replace:

```python
    assert await adapter._initialize_onnx()
```

with:

```python
    assert await adapter._ensure_model_loaded()
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q -k kokoro tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py
```

Expected: `22 passed, 24 deselected`.

Then run the three touched test files three times. A flaky timing assertion should show up here.

```bash
for i in 1 2 3; do TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py tldw_Server_API/tests/Persona/test_persona_live_voice_runtime.py tldw_Server_API/tests/TTS/adapters/test_kokoro_adapter_mock.py 2>&1 | grep -E "passed|failed"; done
```

Expected: `108 passed` on each run (measured while planning).

- [ ] **Step 6: Run every test file that references the Kokoro adapter**

```bash
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/Persona/test_persona_live_voice_runtime.py \
  tldw_Server_API/tests/TTS_NEW/integration/test_kokoro_runtime_health_envelope.py \
  tldw_Server_API/tests/TTS/adapters/test_kokoro_adapter_integration.py \
  tldw_Server_API/tests/TTS/adapters/test_kokoro_adapter_mock.py \
  tldw_Server_API/tests/TTS/adapters/test_kokoro_alignment.py \
  tldw_Server_API/tests/TTS/test_kokoro_health_and_errors.py \
  tldw_Server_API/tests/TTS/test_phoneme_overrides.py \
  tldw_Server_API/tests/TTS/test_tts_adapters_comprehensive.py \
  tldw_Server_API/tests/TTS/test_tts_adapters.py \
  tldw_Server_API/tests/TTS/test_tts_integration.py \
  tldw_Server_API/tests/TTS/unit/test_local_adapter_async_offload.py \
  tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py 2>&1 | tail -1
```

Expected: `208 passed, 8 skipped`. Dev gives `186 passed, 8 skipped`; the difference is the 22 new cases.

- [ ] **Step 7: Lint and Bandit**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m ruff check tldw_Server_API/app/core/TTS/adapters/kokoro_adapter.py tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py tldw_Server_API/tests/Persona/test_persona_live_voice_runtime.py
uvx bandit -q -ll tldw_Server_API/app/core/TTS/adapters/kokoro_adapter.py
```

Expected:
- Ruff: exactly one finding, `I001` at `test_persona_live_voice_runtime.py:86`. It is there on dev too, so leave it. Ruff does not flag the `asyncio.timeout` form.
- Bandit: no output.

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/TTS/adapters/kokoro_adapter.py tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py tldw_Server_API/tests/Persona/test_persona_live_voice_runtime.py
git commit -m "fix(tts): a Kokoro model load keeps its lock until the load thread returns (TASK-13860)

A cancelled request or an init_timeout_sec expiry freed _model_lock while
the load thread kept running, so the next caller started a second load beside
it; initialize() loaded with no _model_lock at all. Every load now goes
through hold_tts_engine, and init_timeout_sec bounds each caller's wait
without freeing the lock.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Verify, close TASK-13860, open the PR

**Files:**
- Modify (through backlog-py only): `backlog/tasks/task-13860 - Kokoro-s-init-timeout-sec-abandons-a-running-model-load-so-a-second-load-can-start-alongside-it.md`

**Interfaces:**
- Consumes: Task 1's commit, and its RED (Step 2) and GREEN (Steps 5-6) counts.
- Produces: the PR against `dev`.

- [ ] **Step 1: Run the TTS and Persona suites**

```bash
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/TTS tldw_Server_API/tests/TTS_NEW tldw_Server_API/tests/Persona 2>&1 | tail -15
```

Expected: `7 failed, 3010 passed, 28 skipped, 1 xfailed, 1 xpassed`, measured on the folded plan. The run takes about 18-25 minutes.

The 7 failures are environment failures: `ModuleNotFoundError`, because PyAV and torch are not installed. They fail on unchanged origin/dev too; there, the six files below give `7 failed, 70 passed`. The 7 node ids:

```
tldw_Server_API/tests/TTS/adapters/test_higgs_adapter_integration_stub.py::test_voice_reference_integration_message_injection
tldw_Server_API/tests/TTS/adapters/test_higgs_adapter_integration_stub.py::test_multispeaker_and_language_hints_in_chatml_stub
tldw_Server_API/tests/TTS/adapters/test_luxtts_adapter_mock.py::test_luxtts_generate_non_stream
tldw_Server_API/tests/TTS/adapters/test_supertonic2_adapter_mock.py::test_supertonic2_initialize_and_generate
tldw_Server_API/tests/TTS/adapters/test_supertonic_adapter_mock.py::test_supertonic_initialize_and_generate
tldw_Server_API/tests/TTS/test_tts_module.py::TestStreamingAudioWriter::test_init_valid_formats
tldw_Server_API/tests/TTS_NEW/unit/test_tts_service.py::TestTextGeneration::test_chunked_retry_metadata
```

Any other failure is new. Check it on unchanged origin/dev before you call it a baseline failure. Add the node id of each new failure to the pytest line before running it:

```bash
git worktree add --detach ../kokoro-baseline origin/dev
(cd ../kokoro-baseline && TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/TTS/adapters/test_higgs_adapter_integration_stub.py tldw_Server_API/tests/TTS/adapters/test_luxtts_adapter_mock.py tldw_Server_API/tests/TTS/adapters/test_supertonic2_adapter_mock.py tldw_Server_API/tests/TTS/adapters/test_supertonic_adapter_mock.py tldw_Server_API/tests/TTS/test_tts_module.py tldw_Server_API/tests/TTS_NEW/unit/test_tts_service.py)
git worktree remove --force ../kokoro-baseline
```

- [ ] **Step 2: Close TASK-13860**

Planning measured the three mutations named in the notes below. Each one was one edit to Step 3's `_ensure_model_loaded`, run against the 22 Kokoro cases:
- Release the lock at cancel: `async with asyncio.timeout(bound), self._model_lock:` with `run = asyncio.to_thread`. Fails the 21 lock cases.
- Bound only the load, not the queue: the `asyncio.timeout(bound)` moved inside the hold, around `_load_pytorch_model(run)`. Fails exactly the 3 timeout cases.
- Apply the bound to ONNX: `bound = self.init_timeout_sec or None`. Fails exactly `test_kokoro_onnx_load_ignores_init_timeout_sec`.

If your counts differ from the ones below, use yours.

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13860 \
  --append-notes "Fixed: every Kokoro model load goes through KokoroAdapter._ensure_model_loaded, which holds _model_lock with hold_tts_engine (core/TTS/utils.py) and passes run to _load_onnx_model/_load_pytorch_model -> _initialize_onnx/_initialize_pytorch. Those construct the ONNX Kokoro (kokoro_adapter.py:511/519/526 on dev) and the KModel (:666) through run instead of asyncio.to_thread, so a cancelled or timed-out caller leaves the lock with the load thread until it returns. initialize() now loads through _ensure_model_loaded(); on dev it loaded with no _model_lock at all (:384-394), guarded only by ensure_initialized's _init_lock, which a cancelled caller releases. init_timeout_sec (owner decision: option A) is now an asyncio.timeout around the whole hold: it bounds each caller's wait, queueing behind another load included; a load that outlasts it keeps running and keeps the lock, its model is discarded, and the next caller loads again after it. Still PyTorch-only (ONNX, the shipped backend, ignores it) and off by default. Known limits: the lock is per adapter instance, so a load on a new adapter can run beside a thread still running on an earlier one: request-owned adapters from create_adapter_with_overrides (one per request carrying provider overrides, pre-existing and shared by every local adapter), an unload or re-registration, and a registry retry after adapter_failure_retry_seconds (300 s shipped). The PyTorch sync fallback, the generic torch.load fallback and the device move run on the event loop, so they block it but cannot overlap a load." \
  --append-notes "Tests: tests/TTS/unit/test_tts_engine_lock_cancellation.py gains 22 Kokoro cases: 18 = {ONNX, PyTorch unbounded, PyTorch init_timeout_sec 30 s} x {ensure_initialized(), initialize(), _ensure_model_loaded()} x {task cancel, anyio cancel scope}; 3 init_timeout_sec cases (0.1 s, PyTorch, one per entry); and test_kokoro_onnx_load_ignores_init_timeout_sec, which pins ONNX ignoring the bound (passes on dev too). RED on dev: 21 failed (18 assert 2 == 1, 3 assert ['load', 'load'] == ['load']; a second load started while the first thread ran), 1 passed (the ONNX pin). GREEN: 22 passed. Mutations: releasing the lock at cancel fails the 21 lock cases; bounding only the load and not the queue wait fails exactly the 3 timeout cases (the second caller waits on the stuck thread); applying the bound to ONNX fails exactly the ONNX pin. test_persona_live_voice_runtime.py:323 now loads through _ensure_model_loaded(); it called _initialize_onnx() with no run. The 12 test files that reference the Kokoro adapter: 208 passed, 8 skipped (dev: 186 passed, 8 skipped). tests/TTS, tests/TTS_NEW, tests/Persona: 7 failed, 3010 passed, 28 skipped, 1 xfailed, 1 xpassed; the 7 fail on unchanged dev too (ModuleNotFoundError: PyAV/torch not installed). Ruff: no new findings (one pre-existing I001 at test_persona_live_voice_runtime.py:86, also on dev). Bandit -ll (uvx): clean. ADR assessment: none needed; a local fix in one adapter with an existing helper, and ADR-011 is untouched. Docs: none describe init_timeout_sec beyond the Env_Vars.md name list; _ensure_model_loaded's docstring states the meaning. Plan reviewed independently before execution; review folded in." \
  --check-ac 1 --check-ac 2 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "A Kokoro model load now keeps its lock until its load thread returns, through hold_tts_engine, on both backends and on every load path (ensure_initialized/initialize and the lazy load). Neither a cancelled request nor an init_timeout_sec expiry lets a second load start beside a running one on the same adapter. init_timeout_sec now bounds each caller's whole wait; a load that outlasts it keeps the lock and is loaded again after it." \
  -s Done
git add -A backlog/tasks
git commit -m "docs(backlog): close TASK-13860

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 3: Push and open the PR**

```bash
git push -u origin fix/kokoro-load-keeps-lock
gh pr create --base dev --title "fix(tts): a Kokoro model load keeps its lock until the load thread returns (TASK-13860)" --body "$(cat <<'EOF'
## Summary
- Every Kokoro model load goes through `_ensure_model_loaded`, which holds `_model_lock` with `hold_tts_engine` (TASK-13856's helper). The ONNX constructor and the KModel load run through its `run`, so a cancelled or timed-out caller leaves the lock with the load thread until the thread returns.
- `initialize()` (eager) now loads through `_ensure_model_loaded()`; on dev it loaded with no `_model_lock` at all.
- `init_timeout_sec` (owner decision, option A) bounds each caller's whole wait, queueing included. A load that outlasts it keeps running and keeps the lock, and the next caller loads again after it, not beside it. Still PyTorch-only (ONNX, the shipped backend, ignores it) and off by default.

## Test plan
- [x] `tests/TTS/unit/test_tts_engine_lock_cancellation.py`: 22 new Kokoro cases (3 loads x 3 entries incl. `ensure_initialized` x task cancel / anyio scope, 3 timeout cases, 1 ONNX-ignores-the-bound pin); 21 RED on dev, all 22 GREEN here
- [x] The 12 Kokoro test files: 208 passed, 8 skipped (dev: 186 passed, 8 skipped)
- [x] tests/TTS, tests/TTS_NEW, tests/Persona: only the 7 baseline failures (PyAV/torch not installed), same on dev

Known limit: the lock is per adapter instance; request-owned adapters, an unload/re-register, or a registry retry build a new adapter (see TASK-13860's notes).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Review, Change summary, merge**

1. Run `/superpowers:requesting-code-review` in a clean subagent on the PR. Address its findings before anything merges.
2. Ask the owner for the PR's "## Change summary" (ADR-004).
   - This is not CI-enforced.
   - The recorded waivers do not name TASK-13860, which was filed on 2026-10-18 as a TASK-13856 follow-up.
   - So either the owner confirms a waiver, recorded as a full "## Change summary" section, or the owner writes the text.
3. Merge per CLAUDE.md "Merging into dev". Check `gh variable get MERGE_QUEUE` first.
