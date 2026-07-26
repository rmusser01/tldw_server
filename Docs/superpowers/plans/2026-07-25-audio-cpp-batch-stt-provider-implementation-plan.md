# Dedicated audio.cpp Batch STT Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a disabled-by-default, external-server-only `audio-cpp` batch STT provider that works through ordinary tldw_server transcription paths and the native deterministic STT benchmark.

**Architecture:** Add one focused audio.cpp module for strict configuration, origin/model normalization, bounded upstream contract parsing, WAV validation, discovery caching, and HTTP execution. Register a thin `AudioCppAdapter` that freezes all runtime inputs into the existing STT execution-plan contract, delegates execution to that module, and never falls back, retries, downloads, or manages the audio.cpp process. Reuse the existing secure HTTP client, opaque endpoint observability, safe-path boundary, execution types, benchmark target parser, and documentation publication workflow.

**Tech Stack:** Python 3.10+, standard library (`dataclasses`, `json`, `threading`, `urllib.parse`, `wave`), existing FastAPI/STT adapter types, existing async HTTP client, pytest, Hypothesis where it materially helps parser invariants, Markdown, Backlog.md CLI/MCP workflow.

---

## File Structure

- Create: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_AudioCpp.py`
  - Own strict audio.cpp configuration, canonical origin/model handling, bounded response parsing, PCM WAV validation, discovery cache, and HTTP transcription.
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_execution_contract.py`
  - Add an opt-in, bounded metadata allowlist to planned artifact finalization.
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_provider_adapter.py`
  - Add provider identity/aliases, selector routing, immutable planning, execution delegation, metadata opt-in, and cache reset.
- Modify: `tldw_Server_API/app/core/config.py`
  - Project the four raw audio.cpp settings into canonical `STT_Settings`; validation remains in the dedicated module.
- Modify: `tldw_Server_API/Config_Files/config.txt`
  - Add disabled-by-default documented audio.cpp settings.
- Create: `tldw_Server_API/tests/Audio/fixtures/audio_cpp_http_v1.json`
  - Store upstream-shaped health, model-list, and transcription examples with pinned source provenance.
- Create: `tldw_Server_API/tests/Audio/test_audio_cpp_stt.py`
  - Cover strict config, contract parsing, WAV checks, caching, HTTP behavior, adapter planning/execution, selectors, reset, and failure behavior.
- Modify: `tldw_Server_API/tests/Audio/test_stt_provider_adapter.py`
  - Cover the generic finalizer metadata opt-in without changing existing-provider behavior.
- Modify: `tldw_Server_API/tests/Logging/test_config_loading_sections.py`
  - Assert the canonical STT export contains all four raw audio.cpp settings.
- Modify: `tldw_Server_API/tests/Audio/test_audio_transcriptions_adapter_path.py`
  - Prove the ordinary OpenAI-compatible request selector routes to audio.cpp and does not send the selector prefix upstream.
- Modify: `tldw_Server_API/tests/Benchmarks/test_stt_bench.py`
  - Prove native target preparation, network consent, unresolved identity, and empty-output classification for audio.cpp.
- Modify: `Docs/User_Guides/STT_Benchmark_User_Guide.md`
  - Add audio.cpp setup, target, consent, unresolved identity, and cold/warm instructions.
- Modify: `Docs/Published/User_Guides/STT_Benchmark_User_Guide.md`
  - Publish the operator-guide update through the repository refresh workflow.
- Modify: `Docs/Development/STT_Benchmark_Protocol.md`
  - Record the audio.cpp network-backed target contract and eligibility limits.
- Modify: `Helper_Scripts/benchmarks/README.md`
  - Add a compact audio.cpp target example and required flags.
- Modify: `Docs/Getting_Started/First_Time_Audio_Setup_CPU.md`
  - Link CPU operators to the optional external audio.cpp path without changing the recommended bundled STT defaults.
- Modify: `Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md`
  - Link accelerated operators to the same optional external path.
- Modify through `Helper_Scripts/refresh_docs_published.sh`:
  - `Docs/Published/Getting_Started/First_Time_Audio_Setup_CPU.md`
  - `Docs/Published/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md`
- Modify through the official Backlog.md workflow, using CLI only if MCP is unavailable:
  - `backlog/tasks/task-12987 - Add-dedicated-audio.cpp-batch-STT-provider.md`

## Implementation Constraints

- Work from `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/native-stt-benchmark`.
- Use `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python` for Python, pytest, and Bandit commands.
- Keep `Audio_Transcription_AudioCpp.py` a fresh GPL-compatible implementation. The AGPL tldw_chatbook code is an architecture reference only; do not copy it.
- Do not modify `Helper_Scripts/benchmarks/stt_bench.py` unless a failing test proves the existing generic network-target contract is insufficient.
- Do not add dependencies. Use the standard library for configuration parsing, JSON duplicate detection, locking, and WAV inspection.
- Do not add process supervision, model download, conversion, fallback, streaming, diarization, prompt/hotword forwarding, requested timestamps, redirects, or retries.
- Before each implementation task, update `TASK-12987` with the active stage. After each task, append its test/commit evidence.

## Task 1: Add Bounded Planned-Artifact Metadata Opt-In

**Files:**
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_execution_contract.py:823`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_provider_adapter.py:692-817`
- Test: `tldw_Server_API/tests/Audio/test_stt_provider_adapter.py:865-1045`

- [ ] **Step 1: Write failing finalizer tests**

Add tests proving:

```python
def test_finalize_stt_artifact_omits_metadata_without_allowlist(...):
    finalized = finalize_stt_artifact(
        {"text": "ok", "segments": [], "metadata": {"provider": "audio-cpp"}},
        plan=plan,
        actual=actual,
    )
    assert "metadata" not in finalized


def test_finalize_stt_artifact_preserves_only_allowed_bounded_string_metadata(...):
    finalized = finalize_stt_artifact(
        {
            "text": "ok",
            "segments": [],
            "metadata": {"provider": "audio-cpp", "contract": "audio_cpp_http_v1"},
        },
        plan=plan,
        actual=actual,
        metadata_allowlist=("provider", "contract"),
    )
    assert finalized["metadata"] == {
        "provider": "audio-cpp",
        "contract": "audio_cpp_http_v1",
    }
```

Parameterize rejection of duplicate allowlist names, unknown metadata keys,
non-string values, excessive key counts, and overlong values. Assert
`STTTranscriptionError`, not a raw `TypeError` or `ValueError`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Audio/test_stt_provider_adapter.py \
  -k "finalize_stt_artifact and metadata" -q
```

Expected: new tests fail because `metadata_allowlist` is not accepted and
metadata is not preserved.

- [ ] **Step 3: Implement the smallest generic finalizer extension**

Extend the signature without changing default behavior:

```python
def finalize_stt_artifact(
    artifact: object,
    *,
    plan: SttBatchExecutionPlan,
    actual: SttActualExecution,
    runtime_mismatches: tuple[str, ...] = (),
    metadata_allowlist: tuple[str, ...] = (),
) -> dict[str, Any]:
```

Add private constants for conservative maximum metadata keys and string
length. Validate the allowlist itself, require `artifact["metadata"]` to be a
mapping when the allowlist is non-empty, reject keys outside the allowlist,
and copy only bounded string values. Leave metadata omitted when the allowlist
is empty.

Add this class default:

```python
class SttProviderAdapter(ABC):
    artifact_metadata_allowlist: tuple[str, ...] = ()
```

Pass `self.artifact_metadata_allowlist` from `_run_planned_batch()` to the
finalizer. Do not opt any existing adapter into metadata retention.

- [ ] **Step 4: Run focused and local regression tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Audio/test_stt_provider_adapter.py \
  -k "finalize_stt_artifact" -q
```

Expected: all selected finalizer tests pass and existing metadata remains
omitted by default.

- [ ] **Step 5: Commit Task 1**

```bash
git add \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_execution_contract.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_provider_adapter.py \
  tldw_Server_API/tests/Audio/test_stt_provider_adapter.py
git commit -m "feat(stt): allow bounded planned artifact metadata"
```

## Task 2: Add Strict audio.cpp Configuration and Selector Primitives

**Files:**
- Create: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_AudioCpp.py`
- Modify: `tldw_Server_API/app/core/config.py:4355-4550,5354-5425`
- Modify: `tldw_Server_API/Config_Files/config.txt:1155-1185`
- Test: `tldw_Server_API/tests/Audio/test_audio_cpp_stt.py`
- Test: `tldw_Server_API/tests/Logging/test_config_loading_sections.py:43`

- [ ] **Step 1: Write failing configuration and selector tests**

Cover:

- shipped defaults: disabled, loopback origin, empty default model, 600 seconds;
- all four environment variables override the STT mapping;
- accepted booleans are explicit and case-insensitive;
- invalid/blank booleans do not silently become false;
- timeout is finite and greater than zero;
- origin is only an HTTP(S) origin, with userinfo/path/query/fragment,
  malformed ports, percent-encoded authorities, and ambiguous numeric hosts
  rejected;
- raw non-root and dot-segment paths such as `/api`, `/.`, `/./`, and
  `/a/..` are rejected before URL normalization;
- default model may be empty until an ordinary request needs it;
- `audio-cpp:whisper-small`, `audiocpp:whisper-small`, and
  `audio_cpp:whisper-small` normalize to `whisper-small`;
- exact selectors use the configured default;
- a plain safe model remains unchanged after the audio.cpp adapter has already
  been selected;
- unsafe/control/URL-shaped model IDs fail closed.

Name the route regression
`test_audio_cpp_canonical_origin_routes_have_single_slashes` so the focused
`-k` commands select it.

Use a small pure API:

```python
cfg = audio_cpp.load_audio_cpp_config(
    {
        "audio_cpp_enabled": "true",
        "audio_cpp_base_url": "http://127.0.0.1:8080",
        "audio_cpp_default_model": "whisper-small",
        "audio_cpp_timeout_seconds": "30",
    },
    env={},
)
assert cfg.enabled is True
assert cfg.origin == "http://127.0.0.1:8080"
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Audio/test_audio_cpp_stt.py \
  -k "config or origin or selector" -q
```

Expected: collection/import fails because the module does not exist.

- [ ] **Step 3: Implement strict pure configuration**

Create:

```python
@dataclass(frozen=True)
class AudioCppConfig:
    enabled: bool
    origin: str
    default_model: str | None
    timeout_seconds: float


def load_audio_cpp_config(
    stt_settings: Mapping[str, object],
    *,
    env: Mapping[str, str] | None = None,
) -> AudioCppConfig:
    ...


def normalize_audio_cpp_model(
    model: str | None,
    *,
    default_model: str | None,
) -> str:
    ...
```

Use exact true/false token sets and `math.isfinite()`. Canonicalize the origin
using `urllib.parse` plus the existing `_normalize_audio_endpoint()` authority
rules. Validate the raw parsed path as exactly empty or `/` before any
normalization so dot segments cannot collapse into an accepted root. Rebuild
the canonical origin as scheme plus normalized authority with no trailing
slash; never retain the endpoint helper's normalized `/` root. Derive routes
by appending fixed leading-slash paths only after origin validation.

Keep four explicit environment-name constants in this module. Environment
values override the provided mapping; an explicitly present invalid value
raises `STTExecutionUnsupportedError`.

- [ ] **Step 4: Project raw file settings and document defaults**

In `config.py`, read the four `config.txt` values without permissive
boolean/number conversion and include them in `STT_Settings`. The dedicated
loader remains the only validation point and applies environment precedence.
Extend the canonical STT export test to assert that `audio_cpp_enabled`,
`audio_cpp_base_url`, `audio_cpp_default_model`, and
`audio_cpp_timeout_seconds` are all present with their raw projected values.

Add to `[STT-Settings]`:

```ini
# --- audio.cpp external ASR server (disabled by default) ---
audio_cpp_enabled = false
audio_cpp_base_url = http://127.0.0.1:8080
audio_cpp_default_model =
audio_cpp_timeout_seconds = 600
```

Document the exact environment variable names beside the settings.

- [ ] **Step 5: Run focused configuration tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Audio/test_audio_cpp_stt.py \
  -k "config or origin or selector" -q
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Logging/test_config_loading_sections.py \
  tldw_Server_API/tests/Audio/test_stt_vnext_config_flags.py -q
```

Expected: all selected tests pass; invalid audio.cpp values remain observable
to the strict loader instead of being replaced.

- [ ] **Step 6: Commit Task 2**

```bash
git add \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_AudioCpp.py \
  tldw_Server_API/app/core/config.py \
  tldw_Server_API/Config_Files/config.txt \
  tldw_Server_API/tests/Audio/test_audio_cpp_stt.py \
  tldw_Server_API/tests/Logging/test_config_loading_sections.py
git commit -m "feat(stt): add strict audio cpp configuration"
```

## Task 3: Add Bounded Upstream Contract and WAV Validation

**Files:**
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_AudioCpp.py`
- Create: `tldw_Server_API/tests/Audio/fixtures/audio_cpp_http_v1.json`
- Modify: `tldw_Server_API/tests/Audio/test_audio_cpp_stt.py`

- [ ] **Step 1: Add the pinned fixture**

Create one fixture with:

```json
{
  "_provenance": {
    "repository": "https://github.com/0xShug0/audio.cpp",
    "commit": "10287cb60e71c12177b6bbbc70726950a9c7e29a",
    "contract": "audio_cpp_http_v1"
  },
  "health": {"status": "ok", "backend": "cpu", "models": 1},
  "models": {
    "object": "list",
    "data": [
      {"id": "whisper-small", "family": "whisper", "task": "asr", "mode": "offline"}
    ]
  },
  "transcription": {"text": "fixture transcript", "timing": {"total_ms": 10}}
}
```

Use upstream field names only. Do not add weights hashes or fields audio.cpp
does not expose.

- [ ] **Step 2: Write failing pure contract tests**

Cover:

- valid fixture health/catalog/transcription;
- unknown fields accepted;
- exact model matching only;
- both offline and streaming ASR modes accepted;
- non-ASR, missing, and duplicate model IDs rejected;
- duplicate JSON keys rejected through `object_pairs_hook`;
- invalid UTF-8, top-level types, unsafe identifiers, oversized bodies,
  excessive catalog entries, and overlong text rejected;
- health `status != "ok"`, unsafe backend, and negative model count rejected;
- empty and whitespace-only transcript strings accepted;
- fixture provenance commit is exact.

- [ ] **Step 3: Write failing WAV boundary tests**

Generate WAV data with standard-library `wave`, then assert:

- a valid `.wav` PCM RIFF/WAVE file is accepted;
- uppercase `.WAV` is accepted;
- renamed text/non-WAV bytes, truncated RIFF, compressed/non-PCM WAV, wrong
  suffix, directory, FIFO/special file, symlink/path escape, and missing files
  fail before a network stub is touched;
- a WAV whose header declares more PCM frame bytes than the file contains is
  rejected even when the first frame is readable;
- the accepted upload handle is positioned at byte zero.

Name the special-file and late-truncation regressions with both `wav` and
`audio_cpp` in their test names.

- [ ] **Step 4: Run the contract/WAV tests and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Audio/test_audio_cpp_stt.py \
  -k "fixture or health or catalog or transcription_contract or wav" -q
```

Expected: tests fail because bounded parsers and WAV validation are absent.

- [ ] **Step 5: Implement pure bounded parsing and WAV checks**

Add focused immutable result types:

```python
@dataclass(frozen=True)
class AudioCppDiscovery:
    backend: str
    model_id: str
    family: str
    mode: str
```

Parse raw response bytes with `json.loads(..., object_pairs_hook=...)` so
duplicate keys cannot disappear before validation. Bound bytes before decode,
bound lists/strings after decode, and never include response bodies in raised
messages.

Validate WAV through `open_safe_local_path()` followed by standard-library
`wave.open()`. Before opening, use `lstat()` on the contained candidate to
reject symlinks and non-regular files without blocking on a FIFO; immediately
after safe open, require `stat.S_ISREG(os.fstat(handle.fileno()).st_mode)` and
matching device/inode identity. Require `.wav`, `RIFF` plus `WAVE`,
uncompressed PCM, at least one channel, and a positive frame
rate/sample width. Read the complete declared PCM frame payload in bounded
chunks and verify its exact byte count so truncation near the end is detected
without one unbounded allocation. Rewind the upload file handle to zero after
validation.

- [ ] **Step 6: Run the pure tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Audio/test_audio_cpp_stt.py \
  -k "fixture or health or catalog or transcription_contract or wav" -q
```

Expected: all selected tests pass without network access.

- [ ] **Step 7: Commit Task 3**

```bash
git add \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_AudioCpp.py \
  tldw_Server_API/tests/Audio/fixtures/audio_cpp_http_v1.json \
  tldw_Server_API/tests/Audio/test_audio_cpp_stt.py
git commit -m "feat(stt): validate audio cpp HTTP contract"
```

## Task 4: Add Secure Discovery, Cache, and Multipart Execution

**Files:**
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_AudioCpp.py`
- Modify: `tldw_Server_API/tests/Audio/test_audio_cpp_stt.py`

- [ ] **Step 1: Write failing HTTP and cache tests**

Use a fake `afetch` function/response, not raw `httpx` or `aiohttp` patching.
Cover:

- request order on first use: `/health`, `/v1/models`, transcription;
- warm use skips both discovery requests;
- two concurrent first uses for one endpoint/model perform one discovery
  sequence but two transcription requests;
- concurrent callers from the same event loop and from distinct event loops
  cannot deadlock while discovery is in flight;
- a different endpoint ID or model has a different cache key;
- reset clears the cache;
- health/model contract, transport, unknown-model, and model-unavailable
  failures invalidate the relevant key;
- invalidated failures are not retried within the same call;
- multipart uses only `file=("audio.wav", ..., "audio/wav")`, exact `model`,
  and optional `language`;
- no prompt, hotwords, diarization, timestamps, or response-format field;
- `RetryPolicy(attempts=1)`, `allow_redirects=False`, `verify=True`, frozen
  timeout, and frozen transport reach every call;
- response handles are closed;
- server busy, timeout, non-2xx, malformed JSON, and missing text raise bounded
  typed errors without endpoint/body/transcript leakage.

Name the concurrency regression
`test_audio_cpp_cache_singleflight_is_cross_loop_and_nonblocking` so it is
selected by the focused cache command.

- [ ] **Step 2: Run the HTTP/cache tests and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Audio/test_audio_cpp_stt.py \
  -k "http or multipart or discovery or cache or retry or redirect" -q
```

Expected: new runtime tests fail because no HTTP execution exists.

- [ ] **Step 3: Implement fixed route derivation and plan verification inputs**

Expose one derivation helper:

```python
def audio_cpp_routes(origin: str) -> tuple[str, str, str]:
    return (
        f"{origin}/health",
        f"{origin}/v1/models",
        f"{origin}/v1/audio/transcriptions",
    )
```

Normalize each derived URL through the existing endpoint helper. Runtime
accepts only frozen origin, model, timeout, transport, expected endpoint ID,
and expected egress. It verifies the derived transcription route before any
request.

- [ ] **Step 4: Implement one-attempt HTTP and the process cache**

Use:

```python
RetryPolicy(attempts=1, retry_on_status=(), retry_on_methods=())
```

and:

```python
with opaque_stt_http_observability(endpoint_id):
    response = await afetch(
        method=method,
        url=url,
        timeout=timeout_seconds,
        allow_redirects=False,
        retry=no_retry,
        verify=True,
        transport=transport,
        ...
    )
```

Keep one module-level dictionary keyed by `(endpoint_id, model_id)` and one
module-level `threading.Lock`, but hold the lock only for in-memory dictionary
operations and never across `await`. Keep a second dictionary of
`concurrent.futures.Future[AudioCppDiscovery]` values for cross-loop
single-flight:

```python
with _DISCOVERY_LOCK:
    cached = _DISCOVERY_CACHE.get(key)
    future = _DISCOVERY_INFLIGHT.get(key)
    if cached is None and future is None:
        future = concurrent.futures.Future()
        _DISCOVERY_INFLIGHT[key] = future
        leader = True

if cached is not None:
    return cached
if not leader:
    return await asyncio.wrap_future(future)

try:
    discovered = await _discover(...)
except BaseException as exc:
    future.set_exception(exc)
    raise
else:
    with _DISCOVERY_LOCK:
        _DISCOVERY_CACHE[key] = discovered
    future.set_result(discovered)
    return discovered
finally:
    with _DISCOVERY_LOCK:
        if _DISCOVERY_INFLIGHT.get(key) is future:
            _DISCOVERY_INFLIGHT.pop(key, None)
```

Initialize `leader = False` and capture a cache-generation number so reset
cannot let an already in-flight discovery repopulate a cleared cache. The
leader and followers may run on different event loops; followers await the
wrapped concurrent future without blocking an event-loop thread. Transcription
occurs after single-flight completion and is never serialized.

Add:

```python
def reset_audio_cpp_discovery_cache() -> None:
    ...
```

Use a short `# ponytail:` comment to document that the one short-held
dictionary lock can become sharded only if measured cache bookkeeping
contention warrants it.

- [ ] **Step 5: Implement sync adapter-facing execution**

Provide a synchronous wrapper matching existing STT adapters. Use
`asyncio.run()` when no loop is active and the repository's existing
single-worker-thread pattern when called from a running loop. Keep all cache
single-flight logic inside async helpers so both normal and worker-thread
event loops use the same future-based coordination. Return a
`SttTranscriptionOutcome` with:

```python
{
    "text": text,
    "segments": [],
    "language": language,
    "diarization": {"enabled": False, "speakers": None},
    "usage": {"duration_ms": None, "tokens": None},
    "metadata": {
        "provider": "audio-cpp",
        "contract": "audio_cpp_http_v1",
        "model_id": model_id,
        "model_family": discovery.family,
        "model_mode": discovery.mode,
        "server_backend": discovery.backend,
    },
}
```

Copy typed actual execution from the single planned route, including
transport. Do not use upstream timing for benchmark timing.

- [ ] **Step 6: Run Task 4 tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Audio/test_audio_cpp_stt.py \
  -k "http or multipart or discovery or cache or retry or redirect or empty" -q
```

Expected: selected tests pass; no live server or model is required.

- [ ] **Step 7: Commit Task 4**

```bash
git add \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_AudioCpp.py \
  tldw_Server_API/tests/Audio/test_audio_cpp_stt.py
git commit -m "feat(stt): execute audio cpp batch transcription"
```

## Task 5: Register the Adapter and Enforce Immutable Planning

**Files:**
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_provider_adapter.py:352-375,653-817,2578-2832`
- Modify: `tldw_Server_API/tests/Audio/test_audio_cpp_stt.py`
- Modify: `tldw_Server_API/tests/Audio/test_stt_provider_adapter.py:1050-1325`

- [ ] **Step 1: Write failing registration/routing tests**

Cover:

- `SttProviderName.AUDIO_CPP == "audio-cpp"`;
- canonical lookup plus `audiocpp` and `audio_cpp` aliases;
- capabilities are batch true, streaming/diarization false;
- provider remains discoverable while disabled, but planning fails clearly;
- `resolve_provider_for_model()` handles canonical/aliased prefixes and exact
  selectors before qwen/Whisper heuristics;
- absent model with default provider audio.cpp uses
  `audio_cpp_default_model`;
- strict lookup never returns external/faster-whisper;
- registry reset also clears audio.cpp discovery cache.

- [ ] **Step 2: Write failing planning tests**

Assert planning:

- performs no call to `afetch`;
- supports only `neutral-v1`;
- rejects translation, prompt, hotwords, diarization, word timestamps, and
  unsafe/missing models;
- freezes canonical origin, exact model, timeout, and selected transport in
  `runtime_settings`;
- records one route with provider `audio-cpp`, backend/source
  `audio_cpp_http`, opaque endpoint ID, loopback/remote egress, and transport;
- records `identity_resolved=False`, `artifact_id=None`,
  `local_model_available=False`, and `would_download=False`;
- serializes no origin or endpoint URL;
- still executes with frozen values after configuration mutation;
- rejects runtime endpoint/egress/transport mismatch before HTTP.

- [ ] **Step 3: Run adapter tests and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Audio/test_audio_cpp_stt.py \
  tldw_Server_API/tests/Audio/test_stt_provider_adapter.py \
  -k "audio_cpp or audio_cplusplus" -q
```

Expected: tests fail because the provider/adapter is unregistered.

- [ ] **Step 4: Implement the thin adapter**

Add `AudioCppAdapter` with:

```python
artifact_metadata_allowlist = (
    "provider",
    "contract",
    "model_id",
    "model_family",
    "model_mode",
    "server_backend",
)
```

`plan_batch_execution()` loads and validates config, normalizes the selected
model, resolves the current async transport without I/O, derives the route,
and freezes these runtime keys:

```python
audio_cpp_origin
audio_cpp_model
audio_cpp_timeout
audio_cpp_transport
```

The route carries the same transport. Source modules include the audio.cpp
module, execution contract, adapter, HTTP client, security egress, and opaque
observability context. Dependency distributions contain only the selected
transport distribution.

`_transcribe_planned_batch()` reconstructs only from `runtime_settings`,
delegates to the module, and never rereads config.

`transcribe_batch()` always normalizes canonical/aliased ordinary selectors
before plan validation. With no incoming plan, it creates this same plan and
then calls `_run_planned_batch()`. With an incoming plan, it normalizes the
supplied model before `_run_planned_batch()` so `audio-cpp:<model>` is never
sent upstream or persisted as the upstream model.

- [ ] **Step 5: Register identity, aliases, defaults, and reset**

Add the enum member, adapter registration, aliases, default-model resolution,
and early selector handling in `resolve_provider_for_model()`.

`reset_stt_provider_registry()` imports and calls
`reset_audio_cpp_discovery_cache()` before clearing the registry. Keep the
import local so module import order remains dependency-neutral.

- [ ] **Step 6: Run adapter and execution regressions**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Audio/test_audio_cpp_stt.py \
  tldw_Server_API/tests/Audio/test_stt_provider_adapter.py \
  tldw_Server_API/tests/Audio/test_stt_execution_plan_network.py -q
```

Expected: all selected tests pass, including existing external/vLLM network
plans.

- [ ] **Step 7: Commit Task 5**

```bash
git add \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_provider_adapter.py \
  tldw_Server_API/tests/Audio/test_audio_cpp_stt.py \
  tldw_Server_API/tests/Audio/test_stt_provider_adapter.py
git commit -m "feat(stt): register audio cpp provider"
```

## Task 6: Prove Ordinary API and Benchmark Integration

**Files:**
- Modify: `tldw_Server_API/tests/Audio/test_audio_transcriptions_adapter_path.py`
- Modify: `tldw_Server_API/tests/Benchmarks/test_stt_bench.py`
- Modify if and only if a failing dispatcher test requires the minimal fix:
  - `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Lib.py`
  - relevant Jobs/persistence dispatcher identified by the failure

- [ ] **Step 1: Add the ordinary API selector test**

Post a small generated WAV to `/api/v1/audio/transcriptions` with:

```python
data = {
    "model": "audio-cpp:whisper-small",
    "response_format": "json",
}
```

Use the real registry/adapter and fake the dedicated module transport entry
point. Assert the selected provider is `audio-cpp`, the adapter receives or
normalizes to `whisper-small`, and the returned text is preserved. Assert no
fallback adapter is called.

- [ ] **Step 2: Add benchmark preparation/consent tests**

Using the real `AudioCppAdapter` with config/transport monkeypatched only at
their module boundaries, assert:

```python
preflight_targets(
    ("audio-cpp=whisper-small",),
    mode="neutral-v1",
    allow_network_targets=False,
    common_settings=neutral_settings,
)
```

fails for missing consent, while `allow_network_targets=True` succeeds and
produces an unresolved route. Verify the execution contract contains only the
opaque endpoint ID, not the origin.

Add a worker/artifact classification test showing whitespace-only text becomes
`status="empty"` and is scored as an empty hypothesis rather than a provider
exception.

- [ ] **Step 3: Run integration tests and verify behavior**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Audio/test_audio_transcriptions_adapter_path.py \
  -k "audio_cpp" -q
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Benchmarks/test_stt_bench.py \
  -k "audio_cpp or empty_hypothesis" -q
```

Expected: all selected tests pass. If a dispatcher forwards the original
selector, the adapter-side invariant handles it. Change dispatcher code only
if the test demonstrates routing never reaches the adapter.

- [ ] **Step 4: Run the complete benchmark unit suite**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Benchmarks/test_stt_bench.py \
  tldw_Server_API/tests/Benchmarks/test_stt_bench_integration.py -q
```

Expected: all benchmark tests pass; no live audio.cpp server is contacted.

- [ ] **Step 5: Commit Task 6**

Stage only files actually needed:

```bash
git add \
  tldw_Server_API/tests/Audio/test_audio_transcriptions_adapter_path.py \
  tldw_Server_API/tests/Benchmarks/test_stt_bench.py
git commit -m "test(stt): cover audio cpp API and benchmark paths"
```

If a dispatcher required a minimal production change, stage it in this commit
and record the failing test that justified it in `TASK-12987`.

## Task 7: Document Setup, Benchmarking, and Limitations

**Files:**
- Modify: `Docs/User_Guides/STT_Benchmark_User_Guide.md`
- Modify: `Docs/Development/STT_Benchmark_Protocol.md`
- Modify: `Helper_Scripts/benchmarks/README.md`
- Modify: `Docs/Getting_Started/First_Time_Audio_Setup_CPU.md`
- Modify: `Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md`
- Generate:
  - `Docs/Published/User_Guides/STT_Benchmark_User_Guide.md`
  - `Docs/Published/Getting_Started/First_Time_Audio_Setup_CPU.md`
  - `Docs/Published/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md`

- [ ] **Step 1: Add the operator workflow**

Document:

- audio.cpp is a separately built, configured, and user-managed process;
- tldw_server does not download models or start/restart the server;
- start audio.cpp with an exact ASR model and verify `/health` and
  `/v1/models` yourself;
- the four config keys and environment variables;
- ordinary API forms `audio-cpp:<model>` and exact-selector default behavior;
- only validated uncompressed PCM RIFF/WAVE is accepted in v1;
- benchmark target `--target audio-cpp=<model>`;
- loopback still requires `--allow-network-targets`;
- model identity is descriptive/unresolved and cannot satisfy policy gates;
- cold-first includes adapter discovery, but true server cold start requires an
  operator restart immediately before the run;
- warm timing reuses discovery and the server's loaded model;
- no fallback, retry, redirect, conversion, authentication knob, or automatic
  download.

- [ ] **Step 2: Add compact setup-guide links**

In both first-time audio guides, keep the current local recommendations and
add one optional paragraph pointing external audio.cpp users to the benchmark
guide section. Do not present audio.cpp as a bundled or automatically managed
setup path.

- [ ] **Step 3: Refresh published docs**

Run:

```bash
Helper_Scripts/refresh_docs_published.sh
```

Expected: the three source guides have matching published mirrors. Inspect
`git status --short` and revert no user work; investigate any unexpected
generated diffs before staging.

- [ ] **Step 4: Verify documentation**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Docs/test_docs_published_refresh.py \
  tldw_Server_API/tests/Docs/test_stt_tts_link_hygiene.py \
  tldw_Server_API/tests/Docs/test_stt_tts_guide_roles.py -q
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python \
  Helper_Scripts/benchmarks/stt_bench.py run --help
git diff --check
```

Expected: docs mirror/link tests pass; CLI help still documents
`--allow-network-targets`; no whitespace errors.

- [ ] **Step 5: Commit Task 7**

```bash
git add \
  Docs/User_Guides/STT_Benchmark_User_Guide.md \
  Docs/Published/User_Guides/STT_Benchmark_User_Guide.md \
  Docs/Development/STT_Benchmark_Protocol.md \
  Helper_Scripts/benchmarks/README.md \
  Docs/Getting_Started/First_Time_Audio_Setup_CPU.md \
  Docs/Published/Getting_Started/First_Time_Audio_Setup_CPU.md \
  Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md \
  Docs/Published/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md
git commit -m "docs(stt): document audio cpp provider"
```

## Task 8: Verify, Review, Push, and Update the Existing PR

**Files:**
- Modify through the official Backlog.md workflow:
  - `backlog/tasks/task-12987 - Add-dedicated-audio.cpp-batch-STT-provider.md`
- Remove after every implementation, verification, review, PR-update, and
  Backlog-finalization stage is complete:
  - `Docs/superpowers/plans/2026-07-25-audio-cpp-batch-stt-provider-implementation-plan.md`
    as required by the repository's plan-file lifecycle rule.

- [ ] **Step 1: Run focused feature tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Audio/test_audio_cpp_stt.py \
  tldw_Server_API/tests/Audio/test_stt_provider_adapter.py \
  tldw_Server_API/tests/Audio/test_stt_execution_plan_network.py \
  tldw_Server_API/tests/Audio/test_audio_transcriptions_adapter_path.py \
  tldw_Server_API/tests/Benchmarks/test_stt_bench.py \
  tldw_Server_API/tests/Benchmarks/test_stt_bench_integration.py -q
```

Expected: all selected tests pass with no live server/model.

- [ ] **Step 2: Run changed-file quality checks**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/pre-commit run --files \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_AudioCpp.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_execution_contract.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_provider_adapter.py \
  tldw_Server_API/app/core/config.py \
  tldw_Server_API/tests/Audio/test_audio_cpp_stt.py \
  tldw_Server_API/tests/Audio/test_stt_provider_adapter.py \
  tldw_Server_API/tests/Audio/test_audio_transcriptions_adapter_path.py \
  tldw_Server_API/tests/Benchmarks/test_stt_bench.py
git diff --check
```

Expected: all hooks and diff checks pass.

- [ ] **Step 3: Run Bandit on touched Python production paths**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_AudioCpp.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_execution_contract.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_provider_adapter.py \
  tldw_Server_API/app/core/config.py \
  -f json -o /tmp/bandit_task_12987.json
```

Expected: exit 0 and no new findings. Inspect the JSON even on nonzero exit;
fix findings in changed code rather than suppressing them without rationale.

- [ ] **Step 4: Run an opt-in live smoke only when the operator supplies it**

Do not start/download/manage audio.cpp. If and only if a user-managed server,
model, authorized WAV corpus, and explicit consent are already available, run
the documented golden target. Otherwise record the live test as intentionally
skipped and rely on fake-transport CI coverage.

- [ ] **Step 5: Perform independent code review**

Use `superpowers:requesting-code-review` against the complete diff. Fix
critical/important findings with focused regression tests and rerun Steps
1-3. Re-review until approved or three iterations are exhausted.

- [ ] **Step 6: Finalize Backlog evidence**

Using MCP first and CLI fallback only if MCP remains unavailable:

- check all six acceptance criteria;
- record exact pytest, pre-commit, Bandit, docs, and live-skip results;
- record every commit hash and changed file set;
- set `TASK-12987` to Done only after verification/review passes;
- note that PR merge still requires the requester's human-written
  `Change summary`.

Commit the Backlog update:

```bash
git add "backlog/tasks/task-12987 - Add-dedicated-audio.cpp-batch-STT-provider.md"
git commit -m "chore(backlog): close audio cpp STT provider task"
```

- [ ] **Step 7: Push and update PR #2759**

```bash
git push origin codex/native-stt-benchmark
gh pr view 2759 --repo rmusser01/tldw_server \
  --json baseRefName,headRefName,isDraft,url
gh pr checks 2759 --repo rmusser01/tldw_server
```

Expected: base is `dev`, head is `codex/native-stt-benchmark`, the PR includes
the audio.cpp commits, and CI is running or green. Update the PR body with a
concise AI-authored technical recap, but do not write the required human
`Change summary` for the user.

- [ ] **Step 8: Remove the completed implementation plan**

After all implementation, verification, independent review, Backlog
finalization, and initial PR-update work is complete, remove only this task's
plan as required by `AGENTS.md`:

```bash
git rm Docs/superpowers/plans/2026-07-25-audio-cpp-batch-stt-provider-implementation-plan.md
git commit -m "chore(stt): finish audio cpp implementation plan"
git push origin codex/native-stt-benchmark
```

Expected: no other plan file is removed and PR #2759 receives the closeout
commit.

- [ ] **Step 9: Apply the merge gate**

Do not merge until:

- required CI, including license checks, is green;
- all review threads are resolved;
- the user supplies their own `Change summary` explaining what changed and why
  these implementation choices were made.

After those conditions are satisfied, use
`superpowers:finishing-a-development-branch` and merge into `dev` only if the
user's current instruction still authorizes merge.
