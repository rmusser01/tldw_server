# audio.cpp TTS Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `audio_cpp` as a disabled-by-default TTS provider backed by an external or managed `audiocpp_server`, with tested registry routing, configuration, HTTP client behavior, adapter behavior, sidecar lifecycle, installer scaffolding, documentation, and safe default behavior.

**Architecture:** Keep the existing `/api/v1/audio/speech` path intact. Requests flow through `TTSServiceV2`, the TTS adapter registry, fallback, history, storage, quota, and metrics before reaching a new `AudioCppTTSAdapter`. The adapter uses a small `AudioCppClient` for `/health`, `/v1/models`, and `/v1/audio/speech`. Optional managed mode starts a loopback `audiocpp_server` sidecar, renders upstream server config from provider `extra_params`, waits for health, and reuses the same client path. The first slice does not expose generic `/v1/tasks/run` orchestration.

**Tech Stack:** Python 3, FastAPI TTS stack, Pydantic config models, Loguru, httpx, asyncio subprocess management, existing audio conversion utilities, pytest, Ruff, Bandit

---

## Current Workspace Note

The current checkout is the primary workspace, not a linked worktree, and it contains unrelated untracked files:

- `tldw_Server_API/Config_Files/templates/watchlists/cti_osint_report_markdown.md`
- `tldw_Server_API/Config_Files/templates/watchlists/news_briefing_markdown.md`
- `tldw_Server_API/Databases/system_logs.jsonl`

Before source edits begin, choose one:

- Preferred: create or switch to an isolated worktree/branch for this implementation.
- Acceptable: continue in place while staging only files listed in this plan.

Record that decision in `TASK-12125` before editing source files.

## File Map

- Create: `tldw_Server_API/app/core/TTS/adapters/audio_cpp_client.py`
  Responsibility: HTTP client for health, model listing, speech requests, response decoding, timeout handling, sanitized upstream errors, and injectable test transport.
- Create: `tldw_Server_API/app/core/TTS/adapters/audio_cpp_config.py`
  Responsibility: parse provider `extra_params`, validate loopback/remote policy, validate scratch/model paths, render upstream server JSON, and normalize allowlisted request options.
- Create: `tldw_Server_API/app/core/TTS/adapters/audio_cpp_sidecar_supervisor.py`
  Responsibility: manage optional loopback `audiocpp_server` startup, autoselect port, health wait, backoff, idle shutdown, and sanitized process output.
- Create: `tldw_Server_API/app/core/TTS/adapters/audio_cpp_adapter.py`
  Responsibility: implement the TTS adapter contract, capabilities, request translation, reference-audio staging, format metadata, one-shot streaming compatibility, and provider error mapping.
- Create: `Helper_Scripts/install_tts_audio_cpp.py`
  Responsibility: provide explicit admin helper functions for runtime layout, config patching, optional clone/build commands, and optional model-manager command construction.
- Modify: `tldw_Server_API/app/core/TTS/adapter_registry.py`
  Responsibility: add `TTSProvider.AUDIO_CPP`, aliases, namespaced model aliases, default adapter mapping, and provider metadata without changing bare `pocket-tts` routing.
- Modify: `tldw_Server_API/Config_Files/tts_providers_config.yaml`
  Responsibility: add disabled `audio_cpp` provider config and format preferences using only schema-preserved fields plus `extra_params`.
- Modify: `Docs/STT-TTS/TTS-SETUP-GUIDE.md`
  Responsibility: document external server setup, managed sidecar setup, model-manager use, CUDA-first managed support, memory residency, and security boundaries.
- Create: `tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_registry.py`
  Responsibility: prove provider aliases, namespaced model routing, default adapter registration, and non-regression for bare `pocket-tts`.
- Create: `tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_tts_config.py`
  Responsibility: prove YAML loading preserves `audio_cpp` settings under `extra_params`, disabled defaults, and format preferences.
- Create: `tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_config.py`
  Responsibility: prove base URL policy, remote opt-in, scratch/model path containment, option allowlist filtering, and server JSON rendering.
- Create: `tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_client.py`
  Responsibility: prove health/model/speech HTTP behavior with mocked transports and sanitized failures.
- Create: `tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_adapter.py`
  Responsibility: prove capabilities, text-only synthesis, reference-audio modes, ignored options metadata, conversion handoff, and one-shot streaming compatibility.
- Create: `tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_sidecar_supervisor.py`
  Responsibility: prove loopback-only command construction, port selection, health polling, startup failure backoff, and shutdown behavior with fakes.
- Create: `tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_installer.py`
  Responsibility: prove installer pure functions patch config and construct explicit commands without network or compiler dependencies.
- Create: `tldw_Server_API/tests/TTS_NEW/integration/test_audio_cpp_tts_service.py`
  Responsibility: prove `TTSServiceV2` can select `audio_cpp` by provider/model hint and produce a speech response through a mocked adapter/client path.

## Stage 1: Registry And Config Scaffold

**Goal:** Register `audio_cpp` as a disabled provider with exact routing semantics and schema-safe configuration.

**Success Criteria:** `audio_cpp`, `audio-cpp`, and `audiocpp` resolve to `TTSProvider.AUDIO_CPP`; `audio_cpp:pocket-tts`, `audio-cpp/pocket-tts`, and `audiocpp/pocket-tts` route to `audio_cpp`; bare `pocket-tts` still routes to `pocket_tts`; config loads with all audio.cpp runtime settings preserved under `extra_params`; provider remains disabled.

**Tests:** `test_audio_cpp_registry.py`, `test_audio_cpp_tts_config.py`

**Status:** Complete

- [x] **Step 1: Write failing registry tests**

Add tests like:

```python
def test_audio_cpp_provider_aliases_resolve():
    assert TTSAdapterRegistry.resolve_provider("audio_cpp") == TTSProvider.AUDIO_CPP
    assert TTSAdapterRegistry.resolve_provider("audio-cpp") == TTSProvider.AUDIO_CPP
    assert TTSAdapterRegistry.resolve_provider("audiocpp") == TTSProvider.AUDIO_CPP


def test_audio_cpp_model_aliases_do_not_steal_pocket_tts():
    assert MODEL_PROVIDER_MAP["audio_cpp:pocket-tts"] == TTSProvider.AUDIO_CPP
    assert MODEL_PROVIDER_MAP["audio-cpp/pocket-tts"] == TTSProvider.AUDIO_CPP
    assert MODEL_PROVIDER_MAP["audiocpp/pocket-tts"] == TTSProvider.AUDIO_CPP
    assert MODEL_PROVIDER_MAP["pocket-tts"] == TTSProvider.POCKET_TTS
```

Run:

```powershell
. .\.venv\Scripts\Activate.ps1
python -m pytest -q tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_registry.py
```

Expected result before implementation: failure because `TTSProvider.AUDIO_CPP` is not present.

- [x] **Step 2: Write failing config tests**

Assert that `tts_providers_config.yaml` loads with:

- `providers.audio_cpp.enabled` as `false`
- `providers.audio_cpp.extra_params.managed` as `false`
- `providers.audio_cpp.extra_params.allow_remote_base_url` as `false`
- `providers.audio_cpp.extra_params.external_voice_reference_mode` as `disabled`
- `providers.audio_cpp.extra_params.request_option_allowlist` containing `max_tokens` and `seed`
- no unsupported `ogg`, `webm`, or `ulaw` advertised for `audio_cpp`

Run:

```powershell
. .\.venv\Scripts\Activate.ps1
python -m pytest -q tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_tts_config.py
```

Expected result before implementation: failure because `audio_cpp` config is absent.

- [x] **Step 3: Implement registry and YAML scaffold**

Modify:

- `tldw_Server_API/app/core/TTS/adapter_registry.py`
- `tldw_Server_API/Config_Files/tts_providers_config.yaml`

Implementation requirements:

- Add `TTSProvider.AUDIO_CPP = "audio_cpp"`.
- Add aliases for `audio_cpp`, `audio-cpp`, and `audiocpp`.
- Add default adapter mapping to `tldw_Server_API.app.core.TTS.adapters.audio_cpp_adapter.AudioCppTTSAdapter`.
- Add namespaced model aliases only.
- Keep `pocket-tts` mapped to `TTSProvider.POCKET_TTS`.
- Add a disabled YAML provider block that keeps audio.cpp-specific fields inside `extra_params`.
- Add format preferences only for formats the first pass can return or convert: `wav`, `mp3`, `opus`, `flac`, `aac`, and `pcm`.

- [x] **Step 4: Re-run focused tests**

Run:

```powershell
. .\.venv\Scripts\Activate.ps1
python -m pytest -q `
  tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_registry.py `
  tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_tts_config.py
```

Expected result after implementation: both test files pass.

## Stage 2: HTTP Client And Config Validation

**Goal:** Add reusable, testable support code for speaking to `audiocpp_server` and rendering managed server config.

**Success Criteria:** The client can call `/health`, `/v1/models`, and `/v1/audio/speech`; JSON/base64 and WAV responses decode deterministically; upstream failures map to sanitized TTS exceptions; base URLs are loopback-only by default; remote URLs require explicit admin opt-in; file paths stay under configured roots; option passthrough is allowlisted.

**Tests:** `test_audio_cpp_client.py`, `test_audio_cpp_config.py`

**Status:** Complete

- [x] **Step 1: Write failing client tests**

Use `httpx.MockTransport` or the project equivalent to cover:

- `health()` returns a healthy result for HTTP 200.
- `list_models()` returns model ids from `/v1/models`.
- `speech()` returns WAV bytes when upstream returns `audio/wav`.
- `speech()` decodes base64 audio when upstream returns JSON.
- Upstream 4xx and 5xx do not expose raw request text, full local paths, or response bodies longer than the sanitized limit.

Example shape:

```python
transport = httpx.MockTransport(handler)
async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8080") as http_client:
    client = AudioCppClient(base_url="http://127.0.0.1:8080", http_client=http_client)
    response = await client.speech({"model": "pocket-tts", "input": "hello"})
assert response.audio_bytes.startswith(b"RIFF")
```

Run:

```powershell
. .\.venv\Scripts\Activate.ps1
python -m pytest -q tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_client.py
```

Expected result before implementation: import failure for `audio_cpp_client`.

- [x] **Step 2: Write failing config validation tests**

Cover:

- `http://127.0.0.1:<port>` and `http://localhost:<port>` are accepted by default.
- `http://example.com:8080` is rejected unless `allow_remote_base_url` is true.
- managed sidecar hosts other than `127.0.0.1` or `localhost` are rejected.
- `shared_scratch_dir` and `model.path` resolve under configured roots.
- temp reference names are generated by tldw and are not derived from user filenames.
- non-scalar `extra_params` values are not passed upstream.
- only `max_tokens` and `seed` pass through with the default allowlist.
- server JSON includes a single configured TTS model entry with `id`, `family`, `path`, `task`, `mode`, `load_options`, and `session_options`.

Run:

```powershell
. .\.venv\Scripts\Activate.ps1
python -m pytest -q tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_config.py
```

Expected result before implementation: import failure for `audio_cpp_config`.

- [x] **Step 3: Implement client and config modules**

Create:

- `tldw_Server_API/app/core/TTS/adapters/audio_cpp_client.py`
- `tldw_Server_API/app/core/TTS/adapters/audio_cpp_config.py`

Implementation requirements:

- Prefer injected `httpx.AsyncClient` for tests.
- Use the existing TTS exception hierarchy for provider, network, model, validation, and generation errors.
- Redact request text, full paths, secrets, and arbitrary upstream body details from raised messages.
- Keep config parsing independent from the adapter so sidecar and installer code can reuse it.
- Treat provider `backend` as a setup hint unless upstream config documents a matching server field.

- [x] **Step 4: Re-run focused tests**

Run:

```powershell
. .\.venv\Scripts\Activate.ps1
python -m pytest -q `
  tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_client.py `
  tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_config.py
```

Expected result after implementation: both test files pass.

## Stage 3: TTS Adapter Behavior

**Goal:** Implement `AudioCppTTSAdapter` so public speech requests work through the existing TTS service contract.

**Success Criteria:** The adapter reports accurate capabilities, validates unsupported request shapes early, translates `TTSRequest` into upstream speech JSON, handles reference audio according to managed/external mode, returns full audio bytes so `TTSServiceV2` can perform conversion, and supports `stream=true` as one full chunk with `incremental_streaming=false` metadata.

**Tests:** `test_audio_cpp_adapter.py`, `test_audio_cpp_tts_service.py`

**Status:** Complete

- [x] **Step 1: Write failing adapter tests**

Cover:

- capabilities include `supports_streaming=True` for API compatibility and metadata says incremental streaming is false.
- supported formats exclude `ogg`, `webm`, and `ulaw`.
- text-only request posts `model`, `input`, and allowed options.
- `request.stream=True` still returns `audio_data` so service-level conversion can run before the endpoint streams one chunk.
- `voice_reference` is rejected in external mode when `external_voice_reference_mode` is `disabled`.
- `voice_reference` in managed mode is staged as WAV under `shared_scratch_dir` and passed as `voice_ref`.
- configured voice mappings with `request_field: null` are catalog metadata only and fail clearly when requested without reference audio.
- ignored fields such as unsupported `speed` are recorded in metadata instead of passed blindly.

Example assertion:

```python
response = await adapter.generate(TTSRequest(text="hello", model="audio_cpp:pocket-tts", stream=True))
assert response.audio_data == wav_bytes
assert response.audio_stream is None
assert response.metadata["incremental_streaming"] is False
```

Run:

```powershell
. .\.venv\Scripts\Activate.ps1
python -m pytest -q tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_adapter.py
```

Expected result before implementation: import failure for `audio_cpp_adapter`.

- [x] **Step 2: Write failing service routing test**

Use the TTS factory/service test pattern to prove:

- explicit provider `audio_cpp` selects `AudioCppTTSAdapter`.
- namespaced model `audio-cpp/pocket-tts` selects `audio_cpp`.
- disabled provider does not become the default provider without explicit selection.
- public service call receives audio bytes and metadata from the mocked adapter path.

Run:

```powershell
. .\.venv\Scripts\Activate.ps1
python -m pytest -q tldw_Server_API/tests/TTS_NEW/integration/test_audio_cpp_tts_service.py
```

Expected result before implementation: routing or adapter import failure.

- [x] **Step 3: Implement adapter**

Create `tldw_Server_API/app/core/TTS/adapters/audio_cpp_adapter.py`.

Implementation requirements:

- Use `AudioCppConfig` and `AudioCppClient`.
- For `stream=True`, generate full upstream audio and return `audio_data`, not `audio_stream`, in the first pass so existing conversion code can run.
- Include metadata keys: `provider`, `model`, `managed`, `incremental_streaming`, `voice_reference_mode`, `ignored_options`, and `upstream_response_format`.
- Pass only allowlisted scalar options.
- Pass `voice_ref` only when a managed sidecar can read the scratch path or external `shared_path` mode is explicitly configured.
- Do not pass generic `voice`, `voice_id`, or unverified voice fields.
- Clean request scratch files after generation unless `retain_request_artifacts` is true.

- [x] **Step 4: Re-run adapter and service tests**

Run:

```powershell
. .\.venv\Scripts\Activate.ps1
python -m pytest -q `
  tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_adapter.py `
  tldw_Server_API/tests/TTS_NEW/integration/test_audio_cpp_tts_service.py
```

Expected result after implementation: both test files pass.

## Stage 4: Managed Sidecar, Installer, And Docs

**Goal:** Add optional managed sidecar support plus explicit admin setup helpers and user-facing setup guidance.

**Success Criteria:** Managed mode starts only loopback sidecars, selects or validates a port, renders server config, waits for `/health`, backs off after startup failure, supports idle shutdown, and never exposes arbitrary command args or stderr in user-facing errors. The installer helper can patch config and print explicit clone/build/model commands without doing network work in tests. Docs describe external server and managed sidecar setup truthfully.

**Tests:** `test_audio_cpp_sidecar_supervisor.py`, `test_audio_cpp_installer.py`

**Status:** Complete

- [x] **Step 1: Write failing sidecar tests**

Cover:

- non-loopback managed host raises `TTSValidationError`.
- autoselect port chooses an unused loopback port and updates the derived base URL.
- command construction uses only configured binary path and generated server config path.
- startup health polling succeeds with a fake process plus fake client.
- startup timeout terminates the fake process and stores a backoff deadline.
- idle shutdown stops an idle process after the configured interval.
- sanitized errors omit raw stderr and full config paths.

Run:

```powershell
. .\.venv\Scripts\Activate.ps1
python -m pytest -q tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_sidecar_supervisor.py
```

Expected result before implementation: import failure for `audio_cpp_sidecar_supervisor`.

- [x] **Step 2: Write failing installer tests**

Cover pure functions only:

- runtime layout resolves under repo-local `models/audio_cpp`.
- generated YAML patch sets `providers.audio_cpp.enabled` only when `--enable-provider` is supplied.
- `base_url`, `model_path`, `binary_path`, and `extra_params.server` are written without secrets.
- clone, build, and model-manager commands are constructed explicitly but not executed in unit tests.
- generated provider config keeps runtime-specific settings under `extra_params`.

Run:

```powershell
. .\.venv\Scripts\Activate.ps1
python -m pytest -q tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_installer.py
```

Expected result before implementation: missing installer module or helper failures.

- [x] **Step 3: Implement sidecar supervisor and installer helper**

Create:

- `tldw_Server_API/app/core/TTS/adapters/audio_cpp_sidecar_supervisor.py`
- `Helper_Scripts/install_tts_audio_cpp.py`

Modify adapter initialization so managed mode can call `AudioCppSidecarSupervisor.ensure_started()` before speech generation.

Implementation requirements:

- No arbitrary extra command args from normal speech requests.
- No inherited secret environment values unless existing project process helpers already sanitize them.
- Generated server config lives under configured audio.cpp runtime root.
- Sidecar stdout/stderr may be logged at debug with truncation, but user-facing exceptions stay sanitized.
- Installer network and build operations require explicit CLI flags.

- [x] **Step 4: Update setup documentation**

Modify `Docs/STT-TTS/TTS-SETUP-GUIDE.md` with:

- external `audiocpp_server` mode.
- managed sidecar mode.
- CUDA-first managed support statement.
- model-manager package install guidance.
- no silent model download during normal startup or inference.
- memory residency note for lazy-loaded models and sessions.
- loopback, remote-base-url opt-in, and reference-audio shared-path warnings.
- license boundary: optional external component, no vendored binaries in this implementation.

- [x] **Step 5: Re-run focused tests**

Run:

```powershell
. .\.venv\Scripts\Activate.ps1
python -m pytest -q `
  tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_sidecar_supervisor.py `
  tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_installer.py
```

Expected result after implementation: both test files pass.

## Stage 5: Final Verification And Task Closeout

**Goal:** Verify the complete implementation slice, run security checks, update tracking, and prepare the next Approach C decision.

**Success Criteria:** All focused audio.cpp tests pass; adjacent TTS routing/config tests pass; Ruff passes for touched Python files; Bandit reports no new findings in touched implementation files; Backlog task records verification and remaining limitations; commits are scoped to this task.

**Tests:** All files created in this plan plus adjacent registry, config, and service tests.

**Status:** Complete

- [x] **Step 1: Run focused test suite**

Run:

```powershell
. .\.venv\Scripts\Activate.ps1
python -m pytest -q `
  tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_registry.py `
  tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_tts_config.py `
  tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_config.py `
  tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_client.py `
  tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_adapter.py `
  tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_sidecar_supervisor.py `
  tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_installer.py `
  tldw_Server_API/tests/TTS_NEW/integration/test_audio_cpp_tts_service.py
```

Expected result: all focused tests pass.

- [x] **Step 2: Run adjacent regression tests**

Run:

```powershell
. .\.venv\Scripts\Activate.ps1
python -m pytest -q `
  tldw_Server_API/tests/TTS_NEW/unit/test_fish_s2_registry.py `
  tldw_Server_API/tests/TTS_NEW/unit/adapters/test_pocket_tts_cpp_adapter.py `
  tldw_Server_API/tests/TTS_NEW/unit/test_pocket_tts_cpp_installer.py
```

Expected result: adjacent provider routing and installer tests pass. If any named file is absent in this checkout, record the absent path in `TASK-12125` and run the closest existing adjacent test discovered with `rg --files`.

- [x] **Step 3: Run Ruff on touched Python files**

Run after source files exist:

```powershell
. .\.venv\Scripts\Activate.ps1
python -m ruff check `
  tldw_Server_API/app/core/TTS/adapter_registry.py `
  tldw_Server_API/app/core/TTS/adapters/audio_cpp_client.py `
  tldw_Server_API/app/core/TTS/adapters/audio_cpp_config.py `
  tldw_Server_API/app/core/TTS/adapters/audio_cpp_sidecar_supervisor.py `
  tldw_Server_API/app/core/TTS/adapters/audio_cpp_adapter.py `
  Helper_Scripts/install_tts_audio_cpp.py `
  tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_registry.py `
  tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_tts_config.py `
  tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_config.py `
  tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_client.py `
  tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_adapter.py `
  tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_sidecar_supervisor.py `
  tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_installer.py `
  tldw_Server_API/tests/TTS_NEW/integration/test_audio_cpp_tts_service.py
```

Expected result: Ruff exits 0.

- [x] **Step 4: Run Bandit on touched implementation files**

Run:

```powershell
. .\.venv\Scripts\Activate.ps1
python -m bandit -r `
  tldw_Server_API/app/core/TTS/adapters/audio_cpp_client.py `
  tldw_Server_API/app/core/TTS/adapters/audio_cpp_config.py `
  tldw_Server_API/app/core/TTS/adapters/audio_cpp_sidecar_supervisor.py `
  tldw_Server_API/app/core/TTS/adapters/audio_cpp_adapter.py `
  Helper_Scripts/install_tts_audio_cpp.py `
  -f json -o "$env:TEMP\bandit_audio_cpp_tts.json"
```

Expected result: no new high or medium findings in touched code. Fix new findings before closeout.

- [x] **Step 5: Update task tracking and self-review**

Update `TASK-12125` with:

- implementation decision for worktree versus in-place edits.
- touched files.
- test outputs.
- Ruff result.
- Bandit result path and summary.
- known limitations, including CUDA-first managed server support and no generic `/v1/tasks/run` API in this slice.
- final summary.

Run:

```powershell
git diff --check -- `
  tldw_Server_API/app/core/TTS/adapter_registry.py `
  tldw_Server_API/app/core/TTS/adapters/audio_cpp_client.py `
  tldw_Server_API/app/core/TTS/adapters/audio_cpp_config.py `
  tldw_Server_API/app/core/TTS/adapters/audio_cpp_sidecar_supervisor.py `
  tldw_Server_API/app/core/TTS/adapters/audio_cpp_adapter.py `
  tldw_Server_API/Config_Files/tts_providers_config.yaml `
  Helper_Scripts/install_tts_audio_cpp.py `
  Docs/STT-TTS/TTS-SETUP-GUIDE.md `
  backlog/tasks/task-12125 - Implement-audio.cpp-TTS-provider-and-setup-integration.md `
  docs/superpowers/plans/2026-07-03-audio-cpp-tts-provider-implementation-plan.md
```

Expected result: no whitespace errors.

- [x] **Step 6: Commit scoped slices**

Prefer small commits by stage. Stage explicit paths only, and do not include unrelated untracked files.

Example final commit shape:

```powershell
git add `
  tldw_Server_API/app/core/TTS/adapter_registry.py `
  tldw_Server_API/app/core/TTS/adapters/audio_cpp_client.py `
  tldw_Server_API/app/core/TTS/adapters/audio_cpp_config.py `
  tldw_Server_API/app/core/TTS/adapters/audio_cpp_sidecar_supervisor.py `
  tldw_Server_API/app/core/TTS/adapters/audio_cpp_adapter.py `
  tldw_Server_API/Config_Files/tts_providers_config.yaml `
  Helper_Scripts/install_tts_audio_cpp.py `
  Docs/STT-TTS/TTS-SETUP-GUIDE.md `
  tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_registry.py `
  tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_tts_config.py `
  tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_config.py `
  tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_client.py `
  tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_adapter.py `
  tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_sidecar_supervisor.py `
  tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_installer.py `
  tldw_Server_API/tests/TTS_NEW/integration/test_audio_cpp_tts_service.py `
  backlog/tasks/task-12125 - Implement-audio.cpp-TTS-provider-and-setup-integration.md
git commit -m "feat(tts): add audio_cpp provider integration"
```

Expected result: commit succeeds with only task-scoped files.

## Non-Goals For This Slice

- Do not expose generic Audio Studio `/v1/tasks/run` orchestration.
- Do not auto-download models during server startup or inference.
- Do not vendor `audio.cpp` source or binaries.
- Do not remap bare `pocket-tts` to `audio_cpp`.
- Do not advertise `ogg`, `webm`, or `ulaw` until conversion is verified and tested.
- Do not pass arbitrary server command args, environment variables, or unverified voice fields from normal speech requests.

## Handoff

When executing this plan, use the test-first order in each stage. After each stage, update the stage status, record command output in `TASK-12125`, and commit only scoped files. If a test path named above differs in the current checkout, discover the nearest existing test with `rg --files tldw_Server_API/tests | rg "tts|TTS"` and record the substitution in the task notes.
