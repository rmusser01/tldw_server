# TASK-13893: Local TTS model loads share one lock per model, across adapter instances

> **Review folded in (r2, 2026-10-22).** The independent review and dry run (`<scratchpad>/review_plan_fu2_tts_lock.md`) found 0 Blocker, 1 Important and 5 Minor. Every step reproduced verbatim on `e06eeec68c`, and every count matched. r1 is kept as `<scratchpad>/plan_fu2_tts_lock.r1.md`. Changes:
> - **I1.** A fourth follow-up, TASK-<N4> (Task 6 Step 4), covers VibeVoice, Chatterbox, Dia and Higgs, which load (and VibeVoice downloads) on the event loop. The exclusion table and the close-out notes cite it. No code change in this PR.
> - **M1.** Ruff now runs on the 13 touched files, where it reports the 5 pre-existing findings. Re-measured: on the whole `core/TTS` directory plus the test file it reports 65 findings, the same on dev and after the change.
> - **M2.** Review Focus 4 and the PR's known limits now state the multi-model memory peak.
> - **M3.** The `_model_load_lock` rationale is corrected.
> - **M4.** Line numbers are labelled as dev's, with a name anchor for every step whose lines earlier steps shift.
> - **M5.** Review Focus 3 records why the key stays the configured string (no `realpath`).
> - None of the five bears on Decision 1. The review confirmed its framing, so only its evidence line is extended.

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to carry out this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two adapter instances never load the same model at the same time, even after a caller is cancelled or times out mid-load. Kokoro's hung-ONNX bound and its loop-bound PyTorch work are fixed. The dead no-voices ONNX branch goes. The shared test helper checks the waiting caller's state, not elapsed time.

**Architecture:**
- `core/TTS/utils.py` gains `tts_model_load_lock(provider, model_key)`. It returns one `asyncio.Lock` per (event loop, provider, model key), shared by every adapter instance.
  - Entries live in a `weakref.WeakValueDictionary`, so a lock exists only while a holder or waiter keeps it. Request-supplied model names (Kitten) therefore cannot grow the table.
- All 11 in-process sites that load a model on a worker thread hold that lock through the existing `hold_tts_engine`. A cancelled holder therefore leaves the lock with the load thread until the thread returns. That is TASK-13856's mechanism; only the lock's scope changes.
  - Five sites swap their per-instance lock for the shared one: Kokoro, Echo-TTS, KittenTTS, Qwen3 pipeline, Qwen3 MLX.
  - Six sites gain a lock; their `initialize()` loads with a bare `asyncio.to_thread` today: LuxTTS, PocketTTS, NeuTTS, IndexTTS2, Supertonic, Supertonic2.
  - Each key is the adapter's `provider_key` plus the model identity it already gives the resource manager as `model_key`, where it has one.
- Kokoro:
  - `init_timeout_sec` also bounds ONNX callers. That is owner decision 1, recommendation A.
  - The PyTorch device move and the generic `torch.jit.load`/`torch.load` path move into the worker-thread call.
  - The sync fallback, which re-ran a failed load on the event loop, goes.
  - The unreachable no-voices ONNX branch goes.
- `tests/TTS/unit/test_tts_engine_lock_cancellation.py`:
  - Its helper now asserts that, once the cancel has unwound, the second caller is queued on the held lock. Its fixed `sleep(0.05)`/`sleep(0.1)` margins go.
  - A parametrized test runs every one of the 11 sites with two adapter instances.

**Tech Stack:** Python 3.12 (`.venv`), asyncio, anyio cancel scopes, pytest + pytest-asyncio, loguru. No new dependencies.

**Spec:**
- `backlog/tasks/task-13893 - Local-TTS-model-loads-the-per-adapter-instance-lock-lets-a-new-adapter-load-beside-an-orphaned-load-thread.md`: the four ACs.
- Background:
  - `backlog/tasks/task-13860 - Kokoro-s-init-timeout-sec-abandons-a-running-model-load-so-a-second-load-can-start-alongside-it.md`: the notes name the known limits.
  - PR #3291, merged 2026-10-20 as `e1ca22f298`.
  - The PR's clean-session review, Minors 3, 5 and 6: `<scratchpad>/clean3291_review.md`.
  - The TASK-13856 plan, `Docs/superpowers/plans/2026-10-18-tts-cancellation-registry.md` (Measured facts table; Review Focus 4 and 6).

**Branch:** `fix/tts-model-load-lock`, from `origin/dev`. **One PR** into `dev`.

---

## Measured facts (origin/dev `e06eeec68c`, measured 2026-10-22)

### The three paths that build a new adapter beside a running load

All three exist on dev and are not adapter-specific:
1. **Request-owned adapters.** `TTSAdapterRegistry.create_adapter_with_overrides` (`core/TTS/adapter_registry.py:717-789`) builds a non-cached adapter for every request that carries provider overrides (`tts_service_v2.py:3401-3437`). It also builds one for every OmniVoice request. `voice_manager._encode_neutts_reference` (`voice_manager.py:1236-1247`) builds its own `NeuTTSAdapter` per call.
2. **Unload or re-registration.** Both bump the generation (`adapter_registry.py:581-583`, `:1282-1284`). The next `get_adapter` starts a new shielded init with a new adapter (`:664-675`), while the old init's load thread still runs.
3. **Registry retry.** After a failure, `get_adapter_async` builds a new adapter (`Infrastructure/provider_registry.py:499-550`) once `adapter_failure_retry_seconds` has passed (`Config_Files/tts_providers_config.yaml:629`, 300 s). Only Kokoro's `init_timeout_sec` can fail an init while its load thread still runs.

### Every local TTS adapter, and what this plan covers

Measured with `grep -nE "to_thread|run_tts_blocking_call|run_in_executor|hold_tts_engine\(|Lock\(\)|from_pretrained"` over `core/TTS/adapters/`, then by reading each load path.

| Adapter | Model load (dev line) | On a thread? | Load lock on dev | This plan |
|---|---|---|---|---|
| Kokoro | `_ensure_model_loaded` `kokoro_adapter.py:356-378` → ONNX ctor `:509`, KModel `:664` | yes | per instance, `_model_lock` (`:278`, `:368`) | shared lock |
| Echo-TTS | `_ensure_models_loaded` `echo_tts_adapter.py:476-536` | yes | per instance, `_model_lock` (`:150`, `:479`) | shared lock |
| KittenTTS | `_load_runtime_for_model` `kitten_tts_adapter.py:93-123` | yes | per instance, `_runtime_lock` (`:73`, `:94`) | shared lock, keyed per model name |
| Qwen3 (upstream pipeline) | `_get_pipeline` `qwen3_tts_adapter.py:632-657` | yes | per instance, `_pipeline_lock` (`:178`, `:638`) | shared lock |
| Qwen3 (MLX runtime) | `_get_model` `qwen3_runtime_mlx.py:125-138` | yes | per runtime, `_model_lock` (`:27`, `:134`) | shared lock |
| LuxTTS | `initialize` `luxtts_adapter.py:187-192` | yes, `asyncio.to_thread` | **none** | shared lock |
| PocketTTS (ONNX) | `initialize` `pocket_tts_adapter.py:158-166` | yes, `asyncio.to_thread` | **none** | shared lock |
| NeuTTS | `initialize` `neutts_adapter.py:81` | yes, `run_tts_blocking_call` | **none** | shared lock |
| IndexTTS2 | `initialize` `index_tts_adapter.py:133` | yes, `asyncio.to_thread` | **none** | shared lock |
| Supertonic | `initialize` `supertonic_adapter.py:147` | yes, `asyncio.to_thread` | **none** | shared lock |
| Supertonic2 | `initialize` `supertonic2_adapter.py:143-145` | yes, `asyncio.to_thread` | **none** | shared lock |
| OmniVoice | `OmniVoiceRuntime.load` `omnivoice_runtime.py:56-100` | yes | `_load_lock` | **not changed**: the runtime runs in the sidecar process, one per process (`omnivoice_sidecar_server.py:177-180`), and every in-process adapter shares one supervisor (`tts_service_v2.py:3441-3447`), so its lock is already process-wide |
| VibeVoice | `initialize` `vibevoice_adapter.py:416-470` (`from_pretrained`), after `_download_models` (`:576`, `snapshot_download` at `:610`); `_reload_model_for_variant` (`:1341-1359`) re-runs `initialize()` | no, on the event loop | n/a | not changed here: a loop-bound load cannot overlap another one. Loading (and downloading) on the loop is its own defect: follow-up TASK-<N4> |
| Chatterbox | `_get_model`/`_get_vc_model` → `_load_chatterbox_runtime` `chatterbox_adapter.py:1368-1452`, lazily from `generate()` (`:492`, `:1055`) | no, on the event loop | n/a | not changed here; TASK-<N4> |
| Dia | `_load_dia_model` `dia_adapter.py:257`, `:280` | no, on the event loop | n/a | not changed here; TASK-<N4> |
| Higgs | `initialize` `higgs_adapter.py:235` (`HiggsAudioServeEngine(...)`) | no, on the event loop | n/a | not changed here; TASK-<N4> |
| audio_cpp, PocketTTS.cpp, VibeVoice realtime, Qwen3 remote, OpenAI, ElevenLabs, Fish S2, OpenAI-compatible | none in process | n/a | n/a | not applicable |

**The four loop-bound loaders need no lock today, but each stalls the whole server while it loads.**
- This is the defect TASK-13855/13857 fixed for NeuTTS: its load "blocks the server once per process, including the in-process TTS jobs worker's lease renewal".
- No open backlog task covers it. The review scanned every non-Done task.
- Follow-up TASK-<N4> moves those loads off the loop. Each must go through `hold_tts_engine(tts_model_load_lock(...))`, because moving a load to a thread without the shared lock would reintroduce what this plan fixes.

**The six lock-free loads are worse than the task describes:**
- Two concurrent override requests load the same model at once, with no cancellation involved.
- A cancelled `ensure_initialized()` releases `_init_lock` while its thread loads.
- AC1 is not Kokoro-specific, so the plan covers all eleven.

**The multi-model adapters keep correct per-call results without their per-instance lock:**
- These are Kitten (model per request), Qwen3 MLX (model per request) and the Qwen3 pipeline (dict per load id).
- Kitten and Qwen3 return the object they loaded.
- In MLX `_get_model` (`qwen3_runtime_mlx.py:134-138`), nothing awaits between `self._model = await run(...)` and `return self._model, backend_model_id`. Each caller therefore gets its own model.

### Baselines on dev

- **Lock-cancellation file:** `48 passed`.
- **Touched-adapter sweep.** The 41 test files (40 Python files, plus `tests/TTS/adapters/README.md`) that reference a touched adapter or `core.TTS.utils` (`git grep -l` of the module and class names) gave `3 failed, 384 passed, 12 skipped`.
  - The 3 failures are environment failures: `ModuleNotFoundError: No module named 'av'`.
  - Their node ids are `test_luxtts_adapter_mock.py::test_luxtts_generate_non_stream`, `test_supertonic_adapter_mock.py::test_supertonic_initialize_and_generate` and `test_supertonic2_adapter_mock.py::test_supertonic2_initialize_and_generate`.
- **Ruff** on the 13 touched files: 5 pre-existing findings, the same after the change. They are `kitten_tts_adapter.py` UP045 ×3, `qwen3_runtime_mlx.py` UP037 and B009.
  - Run ruff on those 13 files, not on the directory. On the whole `core/TTS` directory plus the test file it reports 65 findings, both on dev and after the change (re-measured in r2).
- **Bandit** `-ll` on the 12 touched product files: 0 findings.
- **No touched file is in `Helper_Scripts/ci/scope_predicate_baseline.txt`.**
- **No test references** the removed attributes `_model_lock` (Kokoro, Echo, Qwen3 MLX), `_runtime_lock` (Kitten) or `_pipeline_lock` (Qwen3): `git grep` over `tldw_Server_API/tests`.

### Planning prototype

The whole plan was run in a scratch worktree, then reverted.
- The final diff is `<scratchpad>/fu2tts_proto_final.patch`. It is a reference only; the steps below are authoritative.
- Every RED, GREEN and mutation count below comes from that run.

---

## Global Constraints

- Branch from `origin/dev`, PR into `dev`. Never `git stash`, `--no-verify` or force-push.
- **No real model weights and no network.** Every test stubs the engine constructor or loader, which blocks on the file's `_Gate`.
- Keep runs focused: the lock file, the touched-adapter sweep, the two ratchets. Never the full suite.
- **Backlog changes only through backlog-py** (ADR-059):
  - Run it as `PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task ...`.
  - Use `--append-notes`, never `--notes`.
  - Stage with `git add -A backlog/tasks`.
- **New task ids** are allocated at execution, under the id lock. The text writes them as `TASK-<N>`.
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`. Tests run as `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q <paths>`.
- `SCRATCH` is the executor's session scratchpad directory (never `/tmp`).
- **Line numbers are origin/dev `e06eeec68c`'s.** Earlier steps shift later ones. Every step also names a function, test or text anchor, so locate the code by that anchor.
  - Task 1 Step 1(a) adds 5 lines to the test file.
  - Task 1 adds 1 net line to `kokoro_adapter.py` above `_initialize_onnx`.
- **The plan file copied into the repo** (Task 6) must have no trailing whitespace.

## Review Focus

Failure modes and inputs the tests do not exercise:

1. **Real runtimes.** Every test stubs the engine.
   - Untested here: `kokoro_onnx`, `kokoro.model.KModel`, torch, and the LuxTTS, PocketTTS, NeuTTS, IndexTTS2 and Supertonic engines.
   - Kokoro's `.cuda()` / `.to(mps)` device move now runs on the worker thread. PyTorch supports that, but no test exercises a real device.
2. **A hung load now stalls every instance of that model, by design (AC1).**
   - On dev, a request-owned adapter whose load hung still let the registry's adapter load beside it, and vice versa. Now each waits for the other.
   - Only Kokoro has a caller bound (`init_timeout_sec`, opt-in, off by default). The other ten wait without limit, as the registry's own shielded init already did on dev.
3. **The lock covers one process and one event loop.**
   - Two uvicorn workers each load their own copy, as each needs one.
   - Production runs one loop per process: the in-process jobs workers are tasks on the server loop (`services/startup_content_jobs_pollers.py:468-469`).
   - Keys are the configured strings. A relative and an absolute path to the same file, or a symlink, make two keys.
     - In practice the registry's adapter and the request-owned ones build their config from the same provider config (`_get_provider_config`). The spellings differ only when a request override supplies a path.
     - So the key stays the configured string. `os.path.realpath` on the five local-path keys (Kokoro, PocketTTS, IndexTTS2, Supertonic, Supertonic2; not the HF repo ids) is the upgrade if that case matters.
4. **Multi-model adapters now serialize per model, not per instance.**
   - This covers KittenTTS, Qwen3 MLX and the Qwen3 pipeline.
   - Two concurrent requests that name different models can load both at once on one instance, as two instances already could.
   - Each caller still gets the object it loaded (see Measured facts).
   - **Memory:** the transient peak per instance is now the cached model plus every model loading. On dev it was the cached model plus one.
     - Kitten: at most the 4 pinned repos (`PINNED_MODEL_REVISIONS`, `vendors/kittentts_compat.py:108-113`) plus the configured one, since `resolve_model_revision` rejects any other.
     - Qwen3 MLX: at most 2 backend ids (`qwen3_runtime_mlx.py:85-97`). That matters most on unified memory: 0.6B + 1.7B bf16 is about 4.6 GB at once.
     - The Qwen3 pipeline already keeps every load in `_pipelines`, so its final residency is unchanged; only the overlap is new.
     - Accepted, not fixed: per-model serialization is what AC1 asks for. A per-instance cap would bring back the cross-model queueing that AC1 does not need. The PR's known limits state the peak.
5. **The helper reads `asyncio.Lock._waiters`** (CPython 3.12 and 3.13). A rename would fail loudly (AttributeError), never pass silently.
   - It relies on FIFO ready-queue order only to queue the second caller before the cancel lands.
   - Either order is checked by the post-cancel state assertion.
6. **Kokoro's sync fallback is gone.**
   - A KModel load that fails on its thread is not retried once on the event loop; the error propagates.
     - Eager path: `initialize()` returns False, and the registry retries after `adapter_failure_retry_seconds`.
     - Lazy path: the request fails.
   - An `ImportError` raised inside the KModel load still falls through to the generic torch path, as on dev.
7. **Lock order is `_init_lock` → model-load lock, never the reverse.**
   - Kokoro, Echo-TTS and the Qwen3 pipeline call `register_model` under the model-load lock.
   - The resource manager's eviction runs other adapters' cleanup callbacks.
   - No cleanup callback reaches a load path. A grep of `_cleanup_resources`/`_drop_pipeline` in the five adapters that have one finds no `initialize`/`_ensure_`/`_load_`/`_get_model`/`tts_model_load_lock` call. So eviction under the lock cannot wait on another model-load lock.
8. **If the owner picks 1A, an operator's `KOKORO_INIT_TIMEOUT_SEC` now also applies to ONNX.** It is unset by default, and the shipped config sets neither it nor `kokoro_init_timeout_sec`.

## Decisions for the owner

**1. The hung-ONNX bound (AC2a). Recommendation: A, apply `init_timeout_sec` to ONNX too. The plan implements A (Task 3).**
- **A (recommended).** Drop the ONNX exemption: one condition in `KokoroAdapter._ensure_model_loaded` (`kokoro_adapter.py:366`).
  - Why:
    - With the shared lock, a hung ONNX load now blocks every Kokoro adapter for that model path, not just one instance. A caller bound is the only way an operator can cap that.
    - The ONNX exemption was never designed. It came from where dev's `wait_for` sat, and TASK-13860 kept it to stay small.
  - Cost:
    - Reverses the "(PyTorch only)" scope that TASK-13860's option A kept.
    - An ONNX deployment that sets `KOKORO_INIT_TIMEOUT_SEC` or `kokoro_init_timeout_sec` below its real load time now fails init and retries after `adapter_failure_retry_seconds`. On dev the setting did nothing for ONNX.
    - Nothing changes in a default install: the shipped config (`tts_providers_config.yaml:111-113`) runs ONNX with no bound set.
      - The review confirmed this. The Kokoro YAML block (`:111-139`) sets no timeout, and `_apply_provider_aliases` maps no timeout key.
      - Nor do `config.txt`, any `.env*`, compose file, Dockerfile, YAML, TOML or shell file set `KOKORO_INIT_TIMEOUT_SEC` / `kokoro_init_timeout_sec`.
- **B. Accept it.** Skip Task 3, keep `test_kokoro_onnx_load_ignores_init_timeout_sec`, and record the owner's acceptance in the TASK-13893 notes. AC2 is then checked as "recorded as accepted by the owner".
  - Cost: a hung ONNX constructor holds the shared load lock for good, and every lazy-load caller for that path waits with no bound.

**2. ADR-004 "## Change summary" for the PR. Recommendation: extend the recorded waiver to TASK-13893.**
- TASK-13893 is the filed follow-up of TASK-13860, whose PR the 2026-10-20 waiver covered. The waiver's task list does not name TASK-13893.
- Cost of not extending it: the owner writes the Change summary. Merge waits on one or the other.

The ACs and the code settle everything else:
- the six lock-free adapters are in scope through AC1, which is not Kokoro-specific;
- AC2b is fixed in code;
- AC3 and AC4 are mechanical.

## ADR assessment

**None needed.**
- This is a process-local concurrency fix inside the TTS adapters. It widens the scope of an existing helper's lock (`hold_tts_engine`, TASK-13856). It sets no durable cross-module rule and changes no API, storage or config contract.
- `Docs/ADR/README.md`: ADR-011 (Audio API semantics) is the nearest decision, and it is untouched.
- No accepted ADR covers adapter model-load concurrency.

## Docs

- No user doc describes the load locks or `init_timeout_sec`'s backend scope.
- `Env_Vars.md:861` lists `KOKORO_INIT_TIMEOUT_SEC` by name only, with no description, so it needs no change and no `Docs/Published` refresh.
- The meaning lives in the `tts_model_load_lock` and `_ensure_model_loaded` docstrings.

---

## Task 1: One model-load lock per model, shared across adapter instances (AC1)

**Files:**
- Modify: `tldw_Server_API/app/core/TTS/utils.py` (imports `:11-23`; new function after `hold_tts_engine`, before `_release_engine_lock` at `:158`)
- Modify:
  - `adapters/kokoro_adapter.py:41,278,356-368`
  - `adapters/echo_tts_adapter.py:34,150,479`
  - `adapters/kitten_tts_adapter.py:16,73,94`
  - `adapters/qwen3_runtime_mlx.py:12,27,134`
  - `adapters/qwen3_tts_adapter.py:35,178,638`
  - `adapters/luxtts_adapter.py:30,187-192`
  - `adapters/pocket_tts_adapter.py:30,158-166`
  - `adapters/neutts_adapter.py:34,81`
  - `adapters/index_tts_adapter.py:31,133`
  - `adapters/supertonic_adapter.py:26,147`
  - `adapters/supertonic2_adapter.py:26,143-145`
  - (all under `tldw_Server_API/app/core/TTS/`)
- Test: `tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py`

**Interfaces:**
- New: `tts_model_load_lock(provider: str, model_key: str) -> asyncio.Lock`, in `core/TTS/utils.py`. Call it on the event loop; it uses `asyncio.get_running_loop()`.
- Removed instance attributes: `KokoroAdapter._model_lock`, `EchoTTSAdapter._model_lock`, `KittenTTSAdapter._runtime_lock`, `Qwen3TTSAdapter._pipeline_lock`, `Qwen3MlxRuntime._model_lock`.

- [ ] **Step 0: Branch and record the task's start**

```bash
git fetch origin dev && git switch -c fix/tts-model-load-lock origin/dev
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13893 -s "In Progress"
```

- [ ] **Step 1: Write the failing tests**

This step uses the file's current helper, unchanged; Task 2 reworks it.

**(a)** In `_supertonic_adapter` (`:179-211`), add a `gate_load` switch so the engine *load* can block on the gate. Make the two `mkdir` calls idempotent, because the cross-instance test builds the adapter twice in one `tmp_path`. Replace the signature and the lines through `vendor = ...` with:

```python
def _supertonic_adapter(monkeypatch, tmp_path, gate: _Gate, *, version: int, gate_load: bool = False):
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir(exist_ok=True)
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir(exist_ok=True)
    (voices_dir / "M1.json").write_text("{}")

    def _engine(*_args):
        gate.enter("synthesize")
        return np.zeros((1, 2400), dtype=np.float32), np.array([0.1])

    def _load_text_to_speech(_onnx_dir, _use_gpu=False):
        if gate_load:
            gate.enter("load")
        return _engine

    vendor = types.SimpleNamespace(
        load_text_to_speech=_load_text_to_speech,
        load_voice_style=lambda _paths, _verbose=False: {"style": "dummy"},
    )
```

The rest of the function is unchanged, from `if version == 1:` on.

**(b)** Replace the four tests at dev `:320-411` (`:325-416` after (a)) with four factories and four slim tests. The four are `test_qwen3_mlx_model_load_...`, `test_qwen3_pipeline_load_...`, `test_echo_tts_model_load_...` and `test_kitten_tts_runtime_load_..._returns`. The cross-instance test reuses the factories, so the setup stops being duplicated:

```python
def _echo_adapter(gate: _Gate):
    from tldw_Server_API.app.core.TTS.adapters.echo_tts_adapter import EchoTTSAdapter

    def load_model_from_hf(**_kwargs):
        gate.enter("load_model_from_hf")
        return object()

    adapter = EchoTTSAdapter({})
    adapter.device = "cuda"  # initialize() resolves it; these tests skip initialize()
    adapter._echo_modules_loaded = True
    adapter._echo_inference = types.SimpleNamespace(
        load_model_from_hf=load_model_from_hf,
        load_fish_ae_from_hf=lambda **_kwargs: object(),
        load_pca_state_from_hf=lambda **_kwargs: object(),
    )
    adapter._import_torch = lambda: types.SimpleNamespace(bfloat16="bf16", float16="f16", float32="f32")
    return adapter


def _kitten_adapter(monkeypatch, gate: _Gate):
    from tldw_Server_API.app.core.TTS.adapters import kitten_tts_adapter as kitten_mod

    def kitten_runtime(assets):
        gate.enter("KittenRuntime")
        return types.SimpleNamespace(sample_rate=24000)

    monkeypatch.setattr(kitten_mod, "resolve_model_revision", lambda _model, revision: revision)
    monkeypatch.setattr(kitten_mod, "download_model_assets", lambda model, **_kwargs: types.SimpleNamespace(repo_id=model))
    monkeypatch.setattr(kitten_mod, "KittenRuntime", kitten_runtime)
    return kitten_mod.KittenTTSAdapter({})


def _qwen3_pipeline_adapter(gate: _Gate):
    from tldw_Server_API.app.core.TTS.adapters.qwen3_tts_adapter import Qwen3TTSAdapter

    def build_pipeline(_model_id):
        gate.enter("build_pipeline")
        return object()

    adapter = Qwen3TTSAdapter({})
    adapter._pipeline_builders = [("fake", build_pipeline)]
    return adapter


async def _qwen3_mlx_runtime(monkeypatch, gate: _Gate):
    from tldw_Server_API.app.core.TTS.adapters.qwen3_tts_adapter import Qwen3TTSAdapter

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
    return adapter._get_runtime()


@pytest.mark.asyncio
@CANCEL_MODES
async def test_qwen3_mlx_model_load_keeps_the_lock_until_the_load_returns(monkeypatch, via_anyio_scope):
    gate = _Gate()
    runtime = await _qwen3_mlx_runtime(monkeypatch, gate)

    async def call():
        return await runtime._get_model("auto")

    await _assert_cancelled_call_keeps_the_engine(call, gate, via_anyio_scope=via_anyio_scope)


@pytest.mark.asyncio
@CANCEL_MODES
async def test_qwen3_pipeline_load_keeps_the_lock_until_the_load_returns(via_anyio_scope):
    gate = _Gate()
    adapter = _qwen3_pipeline_adapter(gate)

    async def call():
        return await adapter._get_pipeline("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")

    await _assert_cancelled_call_keeps_the_engine(call, gate, via_anyio_scope=via_anyio_scope)


@pytest.mark.asyncio
@CANCEL_MODES
async def test_echo_tts_model_load_keeps_the_lock_until_the_load_returns(via_anyio_scope):
    gate = _Gate()
    adapter = _echo_adapter(gate)

    await _assert_cancelled_call_keeps_the_engine(adapter._ensure_models_loaded, gate, via_anyio_scope=via_anyio_scope)


@pytest.mark.asyncio
@CANCEL_MODES
async def test_kitten_tts_runtime_load_keeps_the_lock_until_the_load_returns(monkeypatch, via_anyio_scope):
    gate = _Gate()
    adapter = _kitten_adapter(monkeypatch, gate)

    async def call():
        return await adapter._load_runtime_for_model("KittenML/kitten-tts-nano-0.8")

    await _assert_cancelled_call_keeps_the_engine(call, gate, via_anyio_scope=via_anyio_scope)
```

**(c)** Append at the end of the file. Each factory returns a call that builds a **new** adapter for the same model and runs its load. The helper's two `call()`s therefore reach two instances.

```python
# TASK-13893: the load lock is shared by every adapter instance that loads the same model, so a load on a
# new instance (a request-owned override adapter, a re-registration, a registry retry) waits for another
# instance's load thread, even one whose caller was cancelled. Each factory returns a call that builds a
# NEW adapter for the same model and runs its load.


def _gated_engine_class(gate: _Gate):
    class _Engine:
        SAMPLE_RATE = 24000

        def __init__(self, *_args, **_kwargs):
            gate.enter("load")

    return _Engine


def _new_kokoro_load(monkeypatch, tmp_path, gate, backend):
    return lambda: _kokoro_adapter(monkeypatch, tmp_path, gate, backend=backend).ensure_initialized()


def _new_echo_load(monkeypatch, tmp_path, gate):
    return lambda: _echo_adapter(gate)._ensure_models_loaded()


def _new_kitten_load(monkeypatch, tmp_path, gate):
    return lambda: _kitten_adapter(monkeypatch, gate)._load_runtime_for_model("KittenML/kitten-tts-nano-0.8")


def _new_qwen3_pipeline_load(monkeypatch, tmp_path, gate):
    return lambda: _qwen3_pipeline_adapter(gate)._get_pipeline("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")


def _new_qwen3_mlx_load(monkeypatch, tmp_path, gate):
    async def call():
        return await (await _qwen3_mlx_runtime(monkeypatch, gate))._get_model("auto")

    return call


def _new_luxtts_load(monkeypatch, tmp_path, gate):
    from tldw_Server_API.app.core.TTS.adapters.luxtts_adapter import LuxTTSAdapter

    monkeypatch.setattr(LuxTTSAdapter, "_load_engine_class", lambda _self: _gated_engine_class(gate))
    return lambda: LuxTTSAdapter({}).ensure_initialized()


def _new_pocket_load(monkeypatch, tmp_path, gate):
    from tldw_Server_API.app.core.TTS.adapters.pocket_tts_adapter import PocketTTSOnnxAdapter

    monkeypatch.setattr(PocketTTSOnnxAdapter, "_validate_assets", lambda _self: None)
    monkeypatch.setattr(PocketTTSOnnxAdapter, "_load_engine_class", lambda _self: _gated_engine_class(gate))
    return lambda: PocketTTSOnnxAdapter({}).ensure_initialized()


def _new_neutts_load(monkeypatch, tmp_path, gate):
    from tldw_Server_API.app.core.TTS.adapters.neutts_adapter import NeuTTSAdapter

    def load_engine(_self):
        gate.enter("load")
        return types.SimpleNamespace(_is_quantized_model=False)

    monkeypatch.setattr(NeuTTSAdapter, "_load_engine", load_engine)
    return lambda: NeuTTSAdapter({}).ensure_initialized()


def _new_index_tts_load(monkeypatch, tmp_path, gate):
    from tldw_Server_API.app.core.TTS.adapters.index_tts_adapter import IndexTTS2Adapter

    def create_engine(_self):
        gate.enter("load")
        return object()

    (tmp_path / "config.yaml").write_text("{}")
    monkeypatch.setattr(IndexTTS2Adapter, "_create_engine", create_engine)
    config = {"index_tts_model_dir": str(tmp_path), "index_tts_cfg_path": str(tmp_path / "config.yaml")}
    return lambda: IndexTTS2Adapter(config).ensure_initialized()


def _new_supertonic_load(monkeypatch, tmp_path, gate, version):
    def call():
        adapter, _voice = _supertonic_adapter(monkeypatch, tmp_path, gate, version=version, gate_load=True)
        return adapter.ensure_initialized()

    return call


_NEW_INSTANCE_LOADS = {
    "kokoro-onnx": lambda m, t, g: _new_kokoro_load(m, t, g, "onnx"),
    "kokoro-pytorch": lambda m, t, g: _new_kokoro_load(m, t, g, "pytorch"),
    "echo_tts": _new_echo_load,
    "kitten_tts": _new_kitten_load,
    "qwen3-pipeline": _new_qwen3_pipeline_load,
    "qwen3-mlx": _new_qwen3_mlx_load,
    "lux_tts": _new_luxtts_load,
    "pocket_tts": _new_pocket_load,
    "neutts": _new_neutts_load,
    "index_tts": _new_index_tts_load,
    "supertonic": lambda m, t, g: _new_supertonic_load(m, t, g, 1),
    "supertonic2": lambda m, t, g: _new_supertonic_load(m, t, g, 2),
}


@pytest.mark.asyncio
@CANCEL_MODES
@pytest.mark.parametrize("case", sorted(_NEW_INSTANCE_LOADS))
async def test_a_new_adapter_instance_waits_for_another_instances_load_of_the_same_model(
    monkeypatch, tmp_path, case, via_anyio_scope
):
    """Two adapter instances never load the same model at once, even after the first load's caller is cancelled."""
    gate = _Gate()
    call = _NEW_INSTANCE_LOADS[case](monkeypatch, tmp_path, gate)

    await _assert_cancelled_call_keeps_the_engine(call, gate, via_anyio_scope=via_anyio_scope)


@pytest.mark.asyncio
async def test_kokoro_load_on_a_new_adapter_waits_for_a_timed_out_load(monkeypatch, tmp_path):
    """A registry retry builds a new adapter: its load waits, up to its own bound, for a timed-out load's thread."""
    gate = _Gate()

    def new_adapter(init_timeout_sec):
        return _kokoro_adapter(monkeypatch, tmp_path, gate, backend="pytorch", init_timeout_sec=init_timeout_sec)

    try:
        assert await asyncio.wait_for(new_adapter(0.1).ensure_initialized(), 5) is False
        assert await asyncio.wait_for(new_adapter(0.1).ensure_initialized(), 5) is False
        assert gate.started == ["load"]
    finally:
        gate.release.set()
    assert await asyncio.wait_for(new_adapter(5.0).ensure_initialized(), 5) is True
    assert gate.started == ["load", "load"]
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py`

Expected, as measured on `e06eeec68c`: `25 failed, 48 passed`.
- 24 of the `test_a_new_adapter_instance_...` cases (12 adapters × 2 cancel modes) fail with `AssertionError: assert 2 == 1`: a second instance's load started while the first instance's load thread ran.
- `test_kokoro_load_on_a_new_adapter_waits_for_a_timed_out_load` fails with `assert ['load', 'load'] == ['load']`: the retry's adapter loaded beside the timed-out thread.
- The 48 existing tests pass, the four refactored ones included.

- [ ] **Step 3: Add `tts_model_load_lock` to `core/TTS/utils.py`**

Add `import weakref` after `import unicodedata` (`:21`). Insert this between `hold_tts_engine` and `_release_engine_lock`:

```python
_MODEL_LOAD_LOCKS: weakref.WeakValueDictionary[tuple[asyncio.AbstractEventLoop, str, str], asyncio.Lock] = (
    weakref.WeakValueDictionary()
)


def tts_model_load_lock(provider: str, model_key: str) -> asyncio.Lock:
    """The lock every adapter instance takes, through hold_tts_engine, to load ``model_key`` for ``provider``.

    Shared across instances, so a load on a new adapter (a request-owned override adapter, a
    re-registration, a registry retry) waits for another instance's load thread, even one whose caller
    was cancelled. Keyed by event loop too, as an asyncio.Lock serves one loop. An entry lives only
    while a holder or waiter keeps its lock.
    """
    key = (asyncio.get_running_loop(), provider, model_key)
    lock = _MODEL_LOAD_LOCKS.get(key)
    if lock is None:
        lock = _MODEL_LOAD_LOCKS[key] = asyncio.Lock()
    return lock
```

Why the lock stays alive while a cancelled load runs:
- `hold_tts_engine` registers `functools.partial(_release_engine_lock, lock)` on the load's task.
- That task stays referenced by its pending thread future until the thread returns.
- So the weak entry cannot drop while the lock is held.

- [ ] **Step 4: Hold the shared lock at each of the 11 load sites**

In each file, add `tts_model_load_lock` to the `from ..utils import ...` line. Then make the edits below. All paths are under `tldw_Server_API/app/core/TTS/adapters/`.

**`kokoro_adapter.py`:**
- `:41` becomes `from ..utils import hold_tts_engine, parse_bool, run_tts_blocking_next, tts_model_load_lock`.
- Delete `self._model_lock = asyncio.Lock()` (`:278`).
- Replace the docstring and lines `:357-368` of `_ensure_model_loaded`. The body after `async with` is unchanged.

```python
        """Load the model once, keeping the model's load lock until the load thread returns.

        The lock (tts_model_load_lock) is shared by every Kokoro adapter loading this model path, and
        hold_tts_engine keeps it with a load thread until it returns, even after its caller left, so no
        second load starts beside it on this adapter or another. init_timeout_sec (PyTorch only) bounds
        each caller's wait on the model load, including queueing on that lock; queueing on
        ensure_initialized's _init_lock is not bounded.
        """
        if self._model_is_loaded():
            return True
        bound = None if self.use_onnx else (self.init_timeout_sec or None)
        load_lock = tts_model_load_lock(self.provider_key, f"{'onnx' if self.use_onnx else 'torch'}:{self.model_path}")
        try:
            async with asyncio.timeout(bound) as deadline, hold_tts_engine(load_lock) as run:
```

**`echo_tts_adapter.py`:**
- `:34` becomes `from ..utils import hold_tts_engine, parse_bool, tts_model_load_lock`.
- Delete `self._model_lock = asyncio.Lock()` (`:150`).
- `:479` becomes:

```python
        async with hold_tts_engine(tts_model_load_lock(self.provider_key, str(self.model_repo))) as run:
```

**`kitten_tts_adapter.py`:**
- `:16` becomes `from ..utils import hold_tts_engine, parse_bool, tts_model_load_lock`.
- Delete `self._runtime_lock = asyncio.Lock()` (`:73`).
- `:94` becomes:

```python
        async with hold_tts_engine(tts_model_load_lock(self.provider_key, model_name)) as run:
```

**`qwen3_runtime_mlx.py`:**
- `:12` becomes `from ..utils import hold_tts_engine, tts_model_load_lock`.
- Delete `self._model_lock = asyncio.Lock()` (`:27`).
- `:134` becomes:

```python
        async with hold_tts_engine(tts_model_load_lock(self.adapter.provider_key, f"mlx:{backend_model_id}")) as run:
```

**`qwen3_tts_adapter.py`:**
- `:35` becomes `from ..utils import hold_tts_engine, parse_bool, resolve_qwen3_runtime_name, tts_model_load_lock`.
- Delete `self._pipeline_lock = asyncio.Lock()` (`:178`).
- `:638` becomes:

```python
        async with hold_tts_engine(tts_model_load_lock(self.provider_key, str(load_id))) as run:
```

**`luxtts_adapter.py`:**
- `:30` becomes `from ..utils import hold_tts_engine, parse_bool, tts_model_load_lock`.
- Replace `engine = await asyncio.to_thread(LuxTTS, self.model_path, device=self.device, threads=self.threads)` (`:187-192`, six lines) with:

```python
            async with hold_tts_engine(tts_model_load_lock(self.provider_key, str(self.model_path))) as run:
                engine = await run(LuxTTS, self.model_path, device=self.device, threads=self.threads)
```

**`pocket_tts_adapter.py`:**
- `:30` becomes `from ..utils import hold_tts_engine, parse_bool, tts_model_load_lock`.
- Replace `:158-166` with:

```python
            load_lock = tts_model_load_lock(self.provider_key, f"{self.models_dir}:{self.precision}")
            async with hold_tts_engine(load_lock) as run:
                engine = await run(
                    PocketTTSOnnx,
                    models_dir=str(self.models_dir),
                    tokenizer_path=str(self.tokenizer_path),
                    precision=self.precision,
                    device=self.device,
                    temperature=self.temperature,
                    lsd_steps=self.lsd_steps,
                )
```

**`neutts_adapter.py`:**
- `:34` becomes `from ..utils import hold_tts_engine, parse_bool, tts_model_load_lock`. Nothing else uses `run_tts_blocking_call`; ruff F401 flags it if kept.
- Replace `self._engine = await run_tts_blocking_call(self._load_engine)` (`:81`) with:

```python
            load_lock = tts_model_load_lock(self.provider_key, f"{self.backbone_repo}:{self.codec_repo}")
            async with hold_tts_engine(load_lock) as run:
                self._engine = await run(self._load_engine)
```

**`index_tts_adapter.py`:**
- `:31` becomes `from ..utils import hold_tts_engine, parse_bool, tts_model_load_lock`.
- Replace `engine = await asyncio.to_thread(self._create_engine)` (`:133`) with:

```python
            async with hold_tts_engine(tts_model_load_lock(self.provider_key, str(self.model_dir))) as run:
                engine = await run(self._create_engine)
```

**`supertonic_adapter.py`:**
- `:26` becomes `from ..utils import hold_tts_engine, tts_model_load_lock`.
- Replace `:147` with:

```python
            async with hold_tts_engine(tts_model_load_lock(self.provider_key, str(self.onnx_dir))) as run:
                engine = await run(vendor.load_text_to_speech, str(self.onnx_dir), self.use_gpu)
```

**`supertonic2_adapter.py`:**
- `:26` becomes `from ..utils import hold_tts_engine, tts_model_load_lock`.
- Replace `:143-145` (the three-line `asyncio.to_thread(...)` call) with the same two lines as Supertonic.

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py`
Expected: `73 passed`.

Run ruff on the 13 touched files:

```bash
T=tldw_Server_API/app/core/TTS
$PY -m ruff check $T/utils.py $T/adapters/echo_tts_adapter.py $T/adapters/index_tts_adapter.py $T/adapters/kitten_tts_adapter.py $T/adapters/kokoro_adapter.py $T/adapters/luxtts_adapter.py $T/adapters/neutts_adapter.py $T/adapters/pocket_tts_adapter.py $T/adapters/qwen3_runtime_mlx.py $T/adapters/qwen3_tts_adapter.py $T/adapters/supertonic2_adapter.py $T/adapters/supertonic_adapter.py tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py
```

Expected: `Found 5 errors.`, the 5 pre-existing findings (Measured facts). The whole directory gives 65, the same as dev.

Then confirm the call sites: `git grep -n "tts_model_load_lock(" -- tldw_Server_API/app` gives the definition plus 11 call sites.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/TTS tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py
git commit -m "fix(tts): local TTS model loads share one lock per model across adapter instances (TASK-13893)

A new adapter instance (a request-owned override adapter, an unload or
re-registration, a registry retry) had its own lock, or none, so it could load
a model beside another instance's load thread, including one whose caller had
been cancelled. tts_model_load_lock gives every instance one lock per
(provider, model), held through hold_tts_engine at all 11 in-process sites that
load on a worker thread.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

## Task 2: The helper checks the waiting caller's state, not elapsed time (AC4)

**Files:** Test: `tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py`

**Interfaces:**
- `_assert_cancelled_call_keeps_the_engine(call, gate, *, lock, via_anyio_scope)` gains a required `lock: Callable[[], asyncio.Lock]`, which returns the lock the first call holds.
- New test helpers:
  - `_until(condition, what)`;
  - `_queued_on(lock) -> bool`;
  - `_model_load_lock() -> asyncio.Lock`, the one model-load lock held on the test's loop.

- [ ] **Step 1: Show that the margins let a regression pass**

The "RED" for a test-only change is a mutant that the current helper misses. Commit Task 1 first, then make this temporary edit to `tldw_Server_API/app/core/TTS/utils.py`:
- In `hold_tts_engine`'s `finally`, change `if last_call is None or last_call.done():` to `if True:`. That releases the lock at cancel, the TASK-13856 regression.
- Replace the body of `run_tts_blocking_call` with the code below, which starts every engine thread 0.3 s late, like a loaded runner:

```python
    import time

    def _late(*a, **k):
        time.sleep(0.3)
        return func(*a, **k)

    return await asyncio.to_thread(_late, *args, **kwargs)
```

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q -rf tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py`

Expected (measured): `4 failed, 69 passed`.
- **All 66 helper-based cases pass under the regression.** The second caller takes the released lock, but its thread has not reached the gate within the helper's 0.15 s of sleeps.
- The 4 failures are the non-helper Kokoro timeout tests, whose first load starts after its own 0.1 s bound: `test_kokoro_load_past_init_timeout_...` ×3 and `test_kokoro_load_on_a_new_adapter_...`.

Revert the mutant before Step 2 with `git checkout -- tldw_Server_API/app/core/TTS/utils.py`; Step 4 applies it again.

- [ ] **Step 2: Rewrite the helper**

Add `from tldw_Server_API.app.core.TTS import utils as tts_utils` above the `adapters` import (`:22`).

Replace `_assert_cancelled_call_keeps_the_engine` (`:43-85`) with:

```python
async def _until(condition: Callable[[], object], what: str) -> None:
    """Poll ``condition``; fail after 5 s rather than hang the run."""
    for _ in range(500):
        if condition():
            return
        await asyncio.sleep(0.01)
    pytest.fail(f"timed out waiting for {what}")


def _queued_on(lock: asyncio.Lock) -> bool:
    """Whether a caller is queued on ``lock``: asyncio.Lock keeps each queued acquire's future in _waiters."""
    return any(not waiter.done() for waiter in lock._waiters or ())


def _model_load_lock() -> asyncio.Lock:
    """The one model-load lock (tts_model_load_lock) held on this test's loop."""
    loop = asyncio.get_running_loop()
    (lock,) = [lock for (owner, _provider, _model), lock in tts_utils._MODEL_LOAD_LOCKS.items() if owner is loop]
    return lock


async def _assert_cancelled_call_keeps_the_engine(
    call: Callable[[], Awaitable[object]],
    gate: _Gate,
    *,
    lock: Callable[[], asyncio.Lock],
    via_anyio_scope: bool,
) -> None:
    """Cancel one call mid-engine-call while a second waits; the second must not start until the first returns.

    Checked by state, not by elapsed time (TASK-13893): once the cancel has unwound, the second call must
    end up queued on ``lock`` (the lock the first call holds), not running its own engine call.
    """
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
        await _until(lambda: gate.started, "the first call to reach the engine")
        held = lock()
        # Ready callbacks run in FIFO order, so the second call takes its first step before the cancel lands.
        second = asyncio.create_task(call())
        if via_anyio_scope:
            scopes[0].cancel()
        else:
            first.cancel()
        await asyncio.wait({first}, timeout=5)
        assert first.done(), "the cancelled call never unwound"
        await _until(lambda: _queued_on(held) or len(gate.started) > 1, "the second call to queue on the lock")
        # The cancelled call's engine thread still runs, so the lock is still held and the second call queued on it.
        assert len(gate.started) == 1
        assert _queued_on(held)
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
```

Why this is deterministic:
- **Correct code:** the second caller queues on the held lock. It may first pass through another lock, for example Kokoro's `ensure_initialized` `_init_lock`. It stays queued while the cancelled call's thread runs.
- **Lock released at cancel:**
  - The release resolves the waiter's future before `first` finishes, so `_queued_on` is already False.
  - If the caller arrives after the cancel, it takes the free lock and never queues.
  - Either way the poll ends only when its thread reaches the gate, and `assert len(gate.started) == 1` fails however late the thread starts.
- **No shared lock:** the second thread starts and the same assertion fails.

The pre-cancel `_until` was dropped on purpose. With Kokoro's `ensure_initialized` entry, the second caller first queues on `_init_lock`, not on the held load lock. The post-cancel check covers both orders.

Pass each call site's lock. Where a call is longer than 120 columns, wrap its arguments onto the next line, as in this sample:

```python
    await _assert_cancelled_call_keeps_the_engine(
        call, gate, lock=lambda: adapter._engine_lock, via_anyio_scope=via_anyio_scope
    )
```

The arguments, test by test:

| Test | `lock=` |
|---|---|
| `test_pocket_tts_generate_keeps_the_engine_until_its_call_returns` | `lambda: adapter._engine_lock` |
| `test_pocket_tts_stream_keeps_the_engine_until_its_call_returns` | `lambda: adapter._engine_lock` |
| `test_luxtts_keeps_the_engine_until_its_call_returns` | `lambda: adapter._engine_lock` |
| `test_supertonic_keeps_the_engine_until_its_call_returns` | `lambda: adapter._engine_lock` |
| `test_omnivoice_runtime_synthesis_keeps_the_model_until_its_call_returns` | `lambda: runtime._generate_lock` |
| `test_vibevoice_keeps_the_model_until_its_call_returns` | `lambda: adapter._model_state_lock` |
| `test_omnivoice_runtime_load_keeps_the_lock_until_the_load_returns` | `lambda: runtime._load_lock` |
| `test_qwen3_mlx_model_load_keeps_the_lock_until_the_load_returns` | `_model_load_lock` |
| `test_qwen3_pipeline_load_keeps_the_lock_until_the_load_returns` | `_model_load_lock` |
| `test_echo_tts_model_load_keeps_the_lock_until_the_load_returns` | `_model_load_lock` |
| `test_kitten_tts_runtime_load_keeps_the_lock_until_the_load_returns` | `_model_load_lock` |
| `test_kokoro_model_load_keeps_the_lock_until_the_load_returns` | `_model_load_lock` |
| `test_a_new_adapter_instance_waits_for_another_instances_load_of_the_same_model` | `_model_load_lock` |

`_model_load_lock` unpacks exactly one lock for the test's loop. It runs while only the first call holds a lock, before the second call starts, so the unpack itself never sees two.
- A regression that keyed the lock per instance gives the second call a lock of its own. Its thread then starts, and `assert len(gate.started) == 1` fails.
- The fresh-lock-per-call mutant fails 54 cases that way (measured).

- [ ] **Step 3: Run the tests and confirm they pass**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py`
Expected: `73 passed`. It takes about 5 s; the old helper's fixed sleeps took about 15 s.

- [ ] **Step 4: Re-apply Step 1's mutant and confirm the helper now catches it**

Same command. Expected (measured): `70 failed, 3 passed`.
- All 66 helper-based cases fail with `AssertionError: assert 2 == 1`. The poll waits for the late thread, then the assertion fails.
- So do the 4 non-helper timeout tests.
- The 3 that pass are the ONNX pin, the timeout-error test and the close test.

Then revert: `git checkout -- tldw_Server_API/app/core/TTS/utils.py` and confirm `git status --short` lists only the test file.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py
git commit -m "test(tts): the engine-lock helper checks the waiting caller's state, not elapsed time (TASK-13893)

The helper's fixed sleep(0.05)/sleep(0.1) margins let a lock-released-at-cancel
regression pass whenever the next engine thread started late: with every thread
delayed 0.3 s, all 66 helper-based cases passed against it. The helper now
waits for the cancel to unwind and asserts that the next caller is queued on
the held lock; the same mutant fails all 66.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

## Task 3: Kokoro's init_timeout_sec bounds ONNX callers too (AC2a, owner decision 1, recommendation A)

If the owner picks 1B, skip this task. Task 6 Step 3 then records the acceptance instead.

**Files:**
- Modify: `tldw_Server_API/app/core/TTS/adapters/kokoro_adapter.py` (`_ensure_model_loaded`)
- Test: `tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py`

- [ ] **Step 1: Replace the ONNX pin with a failing test**

Replace `test_kokoro_onnx_load_ignores_init_timeout_sec` (dev `:511-522`; Task 1 moves it, so locate it by name) with:

```python
@pytest.mark.asyncio
async def test_kokoro_onnx_load_honours_init_timeout_sec(monkeypatch, tmp_path):
    """init_timeout_sec bounds an ONNX load's caller as it does a PyTorch one: a hung constructor returns False."""
    gate = _Gate()
    adapter = _kokoro_adapter(monkeypatch, tmp_path, gate, backend="onnx", init_timeout_sec=0.1)
    try:
        assert await asyncio.wait_for(adapter._ensure_model_loaded(), 2) is False
        assert gate.started == ["load"]
    finally:
        gate.release.set()
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py -k honours`

Expected (measured on dev's Kokoro): `1 failed`, with `TimeoutError` from the 2 s `wait_for`. The ONNX caller waited past its 0.1 s bound for the hung constructor.

- [ ] **Step 3: Drop the exemption**

In `_ensure_model_loaded`:
- `bound = None if self.use_onnx else (self.init_timeout_sec or None)` becomes `bound = self.init_timeout_sec or None`.
- In the docstring, the sentence "init_timeout_sec (PyTorch only) bounds each caller's wait …" loses "(PyTorch only)". The docstring then reads:

```python
        """Load the model once, keeping the model's load lock until the load thread returns.

        The lock (tts_model_load_lock) is shared by every Kokoro adapter loading this model path, and
        hold_tts_engine keeps it with a load thread until it returns, even after its caller left, so no
        second load starts beside it on this adapter or another. init_timeout_sec bounds each caller's
        wait on the model load, including queueing on that lock; queueing on ensure_initialized's
        _init_lock is not bounded.
        """
```

- [ ] **Step 4: Run the file and confirm it passes**

Expected: `73 passed`. The mutant that restores the ONNX exemption fails only `test_kokoro_onnx_load_honours_init_timeout_sec` (measured).

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/TTS/adapters/kokoro_adapter.py tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py
git commit -m "fix(tts): Kokoro's init_timeout_sec bounds an ONNX load's caller too (TASK-13893)

With the load lock shared across adapters, a hung ONNX constructor stalls every
Kokoro adapter for that model path; the bound, off by default, now caps each
caller's wait on both backends.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

## Task 4: Kokoro's PyTorch load runs wholly on its worker thread (AC2b)

**Files:**
- Modify: `tldw_Server_API/app/core/TTS/adapters/kokoro_adapter.py` (`_initialize_pytorch`, dev `:660-707`)
- Test: `tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py`

- [ ] **Step 1: Write the failing tests**

Insert both after `test_kokoro_load_timeout_error_is_not_reported_as_the_bound`. Each test asserts on the stub's thread ids together with a real outcome (the raised error, or the returned True), so the test-quality ratchet's collector heuristic does not fire.

```python
@pytest.mark.asyncio
async def test_kokoro_failed_pytorch_load_is_not_retried_on_the_event_loop(monkeypatch, tmp_path):
    """A KModel load that fails on its worker thread is not run again on the event loop."""
    loop_thread = threading.get_ident()
    attempts = []

    class _FailingKModel:
        def __init__(self, **_kwargs):
            attempts.append(threading.get_ident())
            raise RuntimeError("corrupt checkpoint")

    adapter = _kokoro_adapter(monkeypatch, tmp_path, _Gate(), backend="pytorch")
    monkeypatch.setattr(sys.modules["kokoro.model"], "KModel", _FailingKModel)
    with pytest.raises(RuntimeError, match="corrupt checkpoint"):
        await asyncio.wait_for(adapter._ensure_model_loaded(), 5)
    assert len(attempts) == 1
    assert loop_thread not in attempts


@pytest.mark.asyncio
@pytest.mark.parametrize("kokoro_package", [True, False], ids=["kmodel", "generic-torch"])
async def test_kokoro_pytorch_load_runs_off_the_event_loop(monkeypatch, tmp_path, kokoro_package):
    """The whole PyTorch load runs on the worker thread: KModel with its device move, or the generic torch.jit.load."""
    from tldw_Server_API.app.core.TTS.adapters import kokoro_adapter as kokoro_mod

    loop_thread = threading.get_ident()
    threads = []

    class _Model:
        def __init__(self, **_kwargs):
            threads.append(threading.get_ident())

        def eval(self):
            threads.append(threading.get_ident())
            return self

        def cpu(self):
            threads.append(threading.get_ident())
            return self

    adapter = _kokoro_adapter(monkeypatch, tmp_path, _Gate(), backend="pytorch")
    if kokoro_package:
        monkeypatch.setattr(sys.modules["kokoro.model"], "KModel", _Model)
    else:
        monkeypatch.setitem(sys.modules, "kokoro.model", None)  # `from kokoro.model import KModel` raises ImportError
        jit = types.SimpleNamespace(load=lambda *_args, **_kwargs: _Model())
        monkeypatch.setattr(kokoro_mod, "safe_import_torch", lambda: types.SimpleNamespace(jit=jit))
    assert await asyncio.wait_for(adapter._ensure_model_loaded(), 5) is True
    assert threads and loop_thread not in threads
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py -k "not_retried or off_the_event_loop"`

Expected (measured): `3 failed`.
- `not_retried`: `assert 2 == 1`. There were two KModel attempts, and the sync fallback ran the second on the loop thread.
- `kmodel`: the loop thread is in `threads`; `.cpu()` ran on the loop.
- `generic-torch`: every recorded call is on the loop thread; `torch.jit.load` and `.eval()` ran on the loop.

- [ ] **Step 3: Move the work into the thread and delete the sync fallback**

In `_initialize_pytorch`, replace the stretch from `def _load_model():` to the device move's last line (dev `:660-681`, `:661-682` after Task 1) with the code below. The `start = time.time()` above it and the `logger.info(... loaded on {dev} ...)` below it stay.

```python
            dev = str(self.device).lower()

            def _load_model():
                """Load the KModel and move it to its device, on the worker thread."""
                model = KModel(repo_id=repo_id, config=config_path, model=self.model_path).eval()
                if dev.startswith("cuda"):
                    with suppress(_KOKORO_NONCRITICAL_EXCEPTIONS):
                        model = model.cuda()
                    return model
                if dev == "mps":
                    try:
                        return model.to(torch.device("mps"))
                    except _KOKORO_NONCRITICAL_EXCEPTIONS:
                        logger.warning("MPS device not available; using CPU for Kokoro")
                return model.cpu()

            with _capture_kokoro_repo_warning():
                self.kokoro_pt_model = await run(_load_model)
```

In the `except ImportError:` branch (dev `:698-707`, `:699-708` after Task 1), replace the body down to and including the `self.model_pt.eval()` suppression with the code below. The logging, the registration, `return True` and the `except ... raise TTSModelLoadError` stay.

```python
            # Fallback: generic torch.load
            def _load_generic_model():
                """Load a TorchScript (or else pickled) model, on the worker thread."""
                try:
                    model = torch.jit.load(self.model_path, map_location=self.device)
                except _KOKORO_NONCRITICAL_EXCEPTIONS:
                    model = torch.load(self.model_path, map_location=self.device)
                with suppress(_KOKORO_NONCRITICAL_EXCEPTIONS):
                    model.eval()
                return model

            try:
                self.model_pt = await run(_load_generic_model)
```

The device-move semantics are dev's:
- CUDA failure keeps the model where it is.
- MPS failure logs and falls back to `.cpu()`.
- Any other device gets `.cpu()`.

`test_kokoro_load_timeout_error_is_not_reported_as_the_bound` still passes, because the load's own `TimeoutError` now propagates straight from `run`.

- [ ] **Step 4: Run the file and confirm it passes**

Expected: `76 passed`.

Measured mutants, each failing only its own test:
- restore the sync fallback → `not_retried`;
- move `.cpu()` back onto the loop → `kmodel`;
- call `_load_generic_model()` without `run` → `generic-torch`.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/TTS/adapters/kokoro_adapter.py tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py
git commit -m "fix(tts): Kokoro's PyTorch load runs wholly on its worker thread (TASK-13893)

The device move and the generic torch.jit.load/torch.load path ran on the event
loop under the load lock, and a load that failed on its thread was run again
synchronously on the loop. All of it now runs through hold_tts_engine's run; a
failed load raises instead of blocking the loop for a second attempt.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

## Task 5: Delete the unreachable no-voices ONNX branch (AC3)

**Files:** Modify: `tldw_Server_API/app/core/TTS/adapters/kokoro_adapter.py` (`_initialize_onnx`, dev `:427`, `:507-529`)

There is no RED test for this task. The branch cannot run:
- `voices_json_arg` (`:427-447`) is a configured file, or the `voices-v1.0.bin` beside the model; otherwise the method raises `TTSModelNotFoundError` first.
- PR #3291's review proved the branch dead with a surviving mutant.

The 6 ONNX lock cases (the 18-case matrix) and the ONNX cross-instance case exercise the remaining call through its stub.

- [ ] **Step 1: Edit**
  - `voices_json_arg: Optional[str]` (dev `:427`, `:428` after Task 1) becomes `voices_json_arg: str`.
  - Replace dev `:507-529` (`:508-530` after Task 1), from the `# Initialize Kokoro (support constructors that accept either 1 or 2 positional args)` comment through the `TypeError` fallback, with:

```python
            self.kokoro_instance = await run(Kokoro, self.model_path, voices_json_arg, espeak_config=espeak_config)
```

- [ ] **Step 2: Verify**
  - `grep -n 'run(Kokoro' tldw_Server_API/app/core/TTS/adapters/kokoro_adapter.py` gives exactly one line.
  - Run the lock file: `76 passed`.
  - Run the Kokoro test files inside Task 6's sweep.
  - Ruff: `Optional` is still used elsewhere in the module, so no F401.

- [ ] **Step 3: Commit**

```bash
git add tldw_Server_API/app/core/TTS/adapters/kokoro_adapter.py
git commit -m "refactor(tts): drop Kokoro's unreachable no-voices ONNX branch (TASK-13893)

voices_json_arg is always a path or _initialize_onnx raises first, so the
else branch and its TypeError fallback never ran (PR #3291 review, Minor 5).

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

## Task 6: Verify, close out, file follow-ups, open the PR

- [ ] **Step 1: The sweep and the gates**

```bash
git grep -l -E "kokoro_adapter|KokoroAdapter|echo_tts_adapter|EchoTTSAdapter|kitten_tts_adapter|KittenTTSAdapter|qwen3_runtime_mlx|Qwen3MlxRuntime|qwen3_tts_adapter|Qwen3TTSAdapter|luxtts_adapter|LuxTTSAdapter|pocket_tts_adapter|PocketTTSOnnxAdapter|neutts_adapter|NeuTTSAdapter|index_tts_adapter|IndexTTS2Adapter|supertonic_adapter|SupertonicOnnxAdapter|supertonic2_adapter|Supertonic2OnnxAdapter|hold_tts_engine|core.TTS.utils|TTS import utils" -- 'tldw_Server_API/tests/*.py' > "$SCRATCH/tts_lock_files.txt"
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q -rf $(cat "$SCRATCH/tts_lock_files.txt")
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/CI/test_test_quality_ratchet.py tldw_Server_API/tests/lint/test_private_coercion_ratchet.py
T=tldw_Server_API/app/core/TTS
$PY -m ruff check $T/utils.py $T/adapters/echo_tts_adapter.py $T/adapters/index_tts_adapter.py $T/adapters/kitten_tts_adapter.py $T/adapters/kokoro_adapter.py $T/adapters/luxtts_adapter.py $T/adapters/neutts_adapter.py $T/adapters/pocket_tts_adapter.py $T/adapters/qwen3_runtime_mlx.py $T/adapters/qwen3_tts_adapter.py $T/adapters/supertonic2_adapter.py $T/adapters/supertonic_adapter.py tldw_Server_API/tests/TTS/unit/test_tts_engine_lock_cancellation.py
(cd tldw_Server_API/app/core/TTS && uvx bandit -q -ll utils.py adapters/echo_tts_adapter.py adapters/index_tts_adapter.py adapters/kitten_tts_adapter.py adapters/kokoro_adapter.py adapters/luxtts_adapter.py adapters/neutts_adapter.py adapters/pocket_tts_adapter.py adapters/qwen3_runtime_mlx.py adapters/qwen3_tts_adapter.py adapters/supertonic2_adapter.py adapters/supertonic_adapter.py)
```

Expected (measured on the prototype):
- **Sweep:** `3 failed, 412 passed, 12 skipped`, against dev's `3 failed, 384 passed, 12 skipped`. The +28 is the lock file's new cases.
  - Only the three PyAV failures listed in Measured facts may fail.
  - Check any other failure on unchanged origin/dev before you call it a baseline: in a scratch worktree (`git worktree add --detach <scratch> origin/dev`), remove it afterwards.
- **Ratchets:** `5 passed`.
- **Ruff:** `Found 5 errors.` on the 13 files, the 5 pre-existing findings only.
- **Bandit:** no issues.

Stability (measured): the lock file gave `76 passed` in each of 5 sequential runs and in each of 4 concurrent `-n 4` xdist runs.

- [ ] **Step 2: Add the plan to the repo**

```bash
cp <scratchpad>/plan_fu2_tts_lock.md Docs/superpowers/plans/2026-10-22-tts-model-load-lock.md
grep -nE ' +$' Docs/superpowers/plans/2026-10-22-tts-model-load-lock.md && echo "trailing whitespace: fix it" || true
git add Docs/superpowers/plans/2026-10-22-tts-model-load-lock.md
git commit -m "docs(plans): add the TASK-13893 model-load lock plan

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 3: Close TASK-13893**

If your counts differ from the planning measurements below, use yours. If the owner chose 1B:
- drop Task 3's sentences from the notes;
- add "Owner accepted (decision 1B, <date>): ONNX keeps ignoring init_timeout_sec; a hung ONNX load holds the shared load lock with no caller bound";
- keep `--check-ac 2`, since the AC allows "recorded as accepted by the owner".

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13893 \
  --append-notes "Fixed (AC1): core/TTS/utils.py gains tts_model_load_lock(provider, model_key), one asyncio.Lock per (event loop, provider, model key) in a WeakValueDictionary, shared by every adapter instance and held through hold_tts_engine, so a cancelled or timed-out load keeps it until its thread returns. Adopted at all 11 in-process sites that load a model on a worker thread: Kokoro (_ensure_model_loaded), Echo-TTS (_ensure_models_loaded), KittenTTS (_load_runtime_for_model, keyed per model name), Qwen3 pipeline (_get_pipeline) and Qwen3 MLX (_get_model), which swap their per-instance lock; and LuxTTS, PocketTTS, NeuTTS, IndexTTS2, Supertonic and Supertonic2, whose initialize() loaded with a bare asyncio.to_thread and no lock at all (two concurrent override requests loaded the same model at once even without a cancel). Keys are provider_key plus the model identity each adapter gives the resource manager. Not changed: OmniVoice (its runtime lives in the sidecar process, one per process, behind one shared supervisor), and VibeVoice, Chatterbox, Dia and Higgs, which load on the event loop, so no load can overlap; that each of those loads stalls the server is filed as TASK-<N4>. AC2a (owner decision 1A): init_timeout_sec now bounds ONNX callers too. AC2b: Kokoro's PyTorch device move and generic torch.jit.load/torch.load path run on the worker thread, and the sync fallback that re-ran a failed load on the loop is gone. AC3: the unreachable no-voices ONNX branch in _initialize_onnx is deleted. AC4: _assert_cancelled_call_keeps_the_engine takes the held lock and asserts, once the cancel has unwound, that the next caller is queued on it (asyncio.Lock._waiters), instead of sleeping fixed margins." \
  --append-notes "Tests (tests/TTS/unit/test_tts_engine_lock_cancellation.py, measured from origin/dev e06eeec68c): 48 -> 76 passed. New: 24 cross-instance cases (12 sites x task cancel / anyio scope), test_kokoro_load_on_a_new_adapter_waits_for_a_timed_out_load (the registry-retry path), test_kokoro_onnx_load_honours_init_timeout_sec (replaces the ONNX-ignores-the-bound pin), test_kokoro_failed_pytorch_load_is_not_retried_on_the_event_loop, test_kokoro_pytorch_load_runs_off_the_event_loop x2. RED on dev: Task 1 25 failed (24 assert 2 == 1, 1 assert ['load', 'load'] == ['load']); Task 3 1 failed (TimeoutError: the ONNX caller waited past its bound); Task 4 3 failed (two KModel attempts, the second on the loop; .cpu() on the loop; jit.load/eval on the loop). AC4 evidence: with a mutant that releases the lock at cancel and starts every engine thread 0.3 s late, the old helper passed all 66 helper-based cases; the new one fails all 66. Mutations: release at cancel fails 70 of 76; a fresh lock per call (the per-instance lock again) fails 54 (24 cross-instance, the timed-out case, 26 single-instance load cases, 3 past-timeout); each Kokoro mutant fails only its own test. Stable over 5 sequential and 4 concurrent -n 4 runs. The 40 test files that reference a touched adapter: 3 failed, 412 passed, 12 skipped (dev: 3 failed, 384 passed, 12 skipped; the 3 are ModuleNotFoundError: av on both). Test-quality and private-coercion ratchets: 5 passed. Ruff on the 13 touched files: no new findings (5 pre-existing in kitten_tts_adapter.py and qwen3_runtime_mlx.py; the whole core/TTS directory gives 65 on dev and after). Bandit -ll (uvx) on the 12 touched product files: clean. ADR assessment: none needed; a process-local concurrency fix through the existing hold_tts_engine helper, ADR-011 untouched. Docs: none describe the load locks; Env_Vars.md lists KOKORO_INIT_TIMEOUT_SEC by name only. Known limits: the lock is per process and per event loop, keyed on configured strings (a relative and an absolute path to one file are two keys); a hung load now stalls every instance of that model, with a caller bound only for Kokoro (opt-in); a cancelled load's model is still discarded and loaded again after it; one Kitten instance or Qwen3 MLX runtime can now load different models at once (at most the 4 pinned Kitten repos plus the configured one, at most 2 MLX backend ids), so its transient memory peak is the cached model plus every model loading, where dev loaded one at a time per instance. VibeVoice, Chatterbox, Dia and Higgs still load on the event loop (no overlap is possible, but each load stalls the server): TASK-<N4>. Follow-ups: TASK-<N1>, TASK-<N2>, TASK-<N3>, TASK-<N4>." \
  --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "A local TTS model load now holds one lock per model, shared by every adapter instance, until its load thread returns. A request-owned override adapter, a re-registration or a registry retry can no longer load a model beside another instance's running or orphaned load, across all 11 adapters that load on a worker thread (6 of which had no load lock at all). Kokoro's init_timeout_sec now bounds ONNX callers too, its PyTorch load runs wholly off the event loop, and its dead no-voices ONNX branch is gone. The shared test helper checks the waiting caller's state, not elapsed time." \
  -s Done
```

- [ ] **Step 4: File the follow-ups**

Each is filed under the id lock with an explicit `--id` (AGENTS.md "Backlog"). Substitute the ids into Step 3's notes.

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Kokoro's device probe timeout does not bound the adapter constructor" --id TASK-<N1> -l tts,bug \
  -d "KokoroAdapter.__init__ (core/TTS/adapters/kokoro_adapter.py:222-232 on dev e06eeec68c) runs _probe_devices (a torch import and CUDA/MPS probe) in 'with concurrent.futures.ThreadPoolExecutor(max_workers=1)' and waits future.result(timeout=device_probe_timeout_sec). At the timeout the with-exit calls shutdown(wait=True), which waits for the probe anyway, so the timeout changes only the result (CPU), not the wait. Measured while planning TASK-13893: with a probe that takes 2 s and kokoro_device_probe_timeout_sec 0.1, KokoroAdapter() took 2.00 s and logged 'Device probe timed out; defaulting to CPU'. The registry constructs adapters on the event loop (adapter_registry.py _initialize_adapter and create_adapter_with_overrides), so a hung torch import or device probe blocks the server. The probe is skipped in test mode, so no test sees it." \
  --ac "A device probe that outlasts kokoro_device_probe_timeout_sec no longer holds KokoroAdapter() past about that bound, with a test that does not depend on test-mode detection"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Local TTS engine calls that hold no lock: IndexTTS2, Qwen3 MLX generation and the Qwen3 pipeline run concurrently on one engine" --id TASK-<N2> -l tts,bug \
  -d "Thread-offloaded engine calls with no lock, on dev e06eeec68c: index_tts_adapter.py:290 (generate), :427 and :437 (stream); qwen3_runtime_mlx.py:195 (generation); qwen3_tts_adapter.py:740 (_invoke_backend) and :1020 (the stream worker, which keeps running after its consumer is cancelled). Two requests run them concurrently on one engine, and a cancelled request's call keeps running beside the next one. Recorded as out of scope in the TASK-13856 plan (Docs/superpowers/plans/2026-10-18-tts-cancellation-registry.md, Review Focus 4) and never filed; TASK-13893 changed only model loads." \
  --ac "Each listed call is held through hold_tts_engine on a per-engine lock, or the engine is documented safe for concurrent calls with the reason" \
  --ac "Each locked site has a test for a plain task cancel and an anyio cancel scope"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "NeuTTS reference encoding loads a whole NeuTTS model per voice and encodes on the event loop" --id TASK-<N3> -l tts,bug \
  -d "VoiceManager._encode_neutts_reference (core/TTS/voice_manager.py:1236-1257 on dev e06eeec68c) builds a new NeuTTSAdapter for every reference it encodes, awaits ensure_initialized() (a full model load; after TASK-13893 it queues behind any other NeuTTS load of the same repos instead of loading beside it), never closes the adapter, and calls adapter._engine.encode_reference on the event loop (:1247). Recorded in the TASK-13856 plan (Review Focus 6) as pre-existing and out of scope; never filed." \
  --ac "Reference encoding reuses the registry's NeuTTS adapter, or closes the one it builds" \
  --ac "encode_reference runs off the event loop"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "VibeVoice, Chatterbox, Dia and Higgs load their models on the event loop (and VibeVoice downloads them there)" --id TASK-<N4> -l tts,bug \
  -d "Four local adapters load model weights synchronously on the event loop, so each load stalls every request on the server, including the in-process TTS jobs worker's lease renewal. On dev e06eeec68c: VibeVoice initialize() runs from_pretrained (core/TTS/adapters/vibevoice_adapter.py:416-470) after _download_models (:576), whose huggingface_hub.snapshot_download (:610) is a multi-GB download on the loop, and _reload_model_for_variant (:1341-1359) re-runs initialize() under _model_state_lock; Chatterbox loads lazily from generate() (chatterbox_adapter.py:492, :1055) through _get_model/_get_vc_model -> _load_chatterbox_runtime (:631-651, called at :1368, :1395, :1420, :1449), so the first request stalls every request; Dia (_load_dia_model, dia_adapter.py:257, :280) and Higgs (HiggsAudioServeEngine at higgs_adapter.py:235) have the same shape. TASK-13855/13857 fixed this class for NeuTTS (its load 'blocks the server once per process, including the in-process TTS jobs worker's lease renewal'). TASK-13893 left these four unchanged: a loop-bound load cannot be orphaned or overlap another, so they needed no load lock. Moving a load to a worker thread without the shared lock would bring back what TASK-13893 fixed, so each move must hold tts_model_load_lock (core/TTS/utils.py) through hold_tts_engine. Found by the independent review of the TASK-13893 plan." \
  --ac "Each listed load (and VibeVoice's snapshot_download) runs off the event loop through hold_tts_engine(tts_model_load_lock(...)), keyed on the adapter's provider_key and model identity" \
  --ac "Each moved load is a case in tests/TTS/unit/test_tts_engine_lock_cancellation.py's cross-instance matrix (test_a_new_adapter_instance_waits_for_another_instances_load_of_the_same_model), for a plain task cancel and an anyio cancel scope, with stub engines only"
git add -A backlog/tasks
git commit -m "docs(backlog): close TASK-13893; file TASK-<N1>, TASK-<N2>, TASK-<N3>, TASK-<N4>

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Push and open the PR**

```bash
git push -u origin fix/tts-model-load-lock
gh pr create --base dev --title "fix(tts): local TTS model loads share one lock per model across adapter instances (TASK-13893)" --body "$(cat <<'EOF'
## Summary
- `core/TTS/utils.py` gains `tts_model_load_lock(provider, model_key)`: one lock per model, shared by every adapter instance, held through `hold_tts_engine` so a cancelled or timed-out load keeps it until its thread returns. A request-owned override adapter, an unload/re-registration or a registry retry can no longer load a model beside another instance's running load.
- Adopted at all 11 in-process sites that load on a worker thread: Kokoro, Echo-TTS, KittenTTS, Qwen3 pipeline, Qwen3 MLX (per-instance lock swapped), and LuxTTS, PocketTTS, NeuTTS, IndexTTS2, Supertonic, Supertonic2 (no load lock at all before). OmniVoice (sidecar, one runtime per process) and the loop-bound loaders (VibeVoice, Chatterbox, Dia, Higgs) need none.
- Kokoro: `init_timeout_sec` bounds ONNX callers too (owner decision 1A); the PyTorch device move and generic `torch.load` path run on the worker thread, and the sync fallback that re-ran a failed load on the loop is gone; the dead no-voices ONNX branch is deleted.
- The shared test helper asserts the next caller is queued on the held lock once the cancel unwinds, instead of sleeping fixed margins.

## Test plan
- [x] `tests/TTS/unit/test_tts_engine_lock_cancellation.py`: 48 -> 76 passed; 24 cross-instance cases + the registry-retry case RED on dev (25 failed), Kokoro cases RED on dev (4 failed)
- [x] Old vs new helper against a release-at-cancel mutant with late threads: old passed all 66 helper cases, new fails all 66
- [x] 40 test files referencing the touched adapters: 3 failed, 412 passed, 12 skipped (dev: 3 failed, 384 passed; the 3 are PyAV not installed)
- [x] Test-quality ratchets, ruff on the 13 touched files (no new findings), bandit -ll (clean)

Known limits: the lock is per process and per event loop, keyed on configured strings; a hung load stalls every instance of that model (caller bound only for Kokoro, opt-in); one Kitten instance or Qwen3 MLX runtime can now load different models at once, so its transient memory peak is the cached model plus every model loading (dev: one at a time per instance; MLX 0.6B + 1.7B bf16 is about 4.6 GB together). VibeVoice, Chatterbox, Dia and Higgs still load on the event loop: filed as a follow-up. ADR assessment: none needed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 6: Review, Change summary, merge**
  1. Run `/superpowers:requesting-code-review` in a clean subagent on the PR. Address its findings before merging.
  2. ADR-004: apply owner decision 2. Either record the waiver as a full "## Change summary" section, or use the owner's text.
  3. Merge per CLAUDE.md "Merging into dev". Check `gh variable get MERGE_QUEUE` first.
  4. A PR body edit re-triggers the required license status, so wait for it before merging.
