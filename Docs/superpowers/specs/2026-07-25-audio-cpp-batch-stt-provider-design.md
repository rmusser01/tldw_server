# Dedicated audio.cpp Batch STT Provider — Design

**Status:** Approved for planning after independent review
**Backlog task:** `TASK-12987`
**Target branch:** `codex/native-stt-benchmark`
**Upstream contract reviewed:** `0xShug0/audio.cpp` commit
`10287cb60e71c12177b6bbbc70726950a9c7e29a`

## Purpose

Add a first-class `audio-cpp` batch speech-to-text provider to tldw_server.
The provider connects to a user-managed `audiocpp_server`, validates the
server's HTTP contract, and supports both ordinary tldw_server batch
transcription and the native STT benchmark.

This integration is external-server-only. tldw_server does not download,
build, configure, launch, restart, terminate, or otherwise supervise
audio.cpp.

## Requirements

- Register the canonical provider ID `audio-cpp`.
- Accept `audiocpp` and `audio_cpp` as aliases.
- Use the provider/model target form `audio-cpp=<server-model-id>`.
- Keep planning network-free so benchmark network consent is enforced before
  any request leaves the process.
- Validate the audio.cpp health and model catalogs before transcription.
- Require an exact audio.cpp model entry whose task is `asr`.
- Send WAV audio through `POST /v1/audio/transcriptions`.
- Support the benchmark's strict and normalized WER/CER scoring without an
  LLM judge.
- Preserve separate cold-first and warm adapter timings.
- Record descriptive audio.cpp server/model metadata without claiming a
  verified model-weight identity.
- Fail closed without fallback, downloads, redirects, or hidden retries.
- Keep normal CI independent of audio.cpp binaries and model downloads.

## Non-goals

- Streaming benchmark execution.
- Diarization, translation, prompts, hotwords, or requested word timestamps.
- Automatic audio conversion in the first adapter version.
- audio.cpp process management or configuration generation.
- Model installation, download, conversion, or GGUF management.
- Multiple audio.cpp endpoint selection, load balancing, or failover.
- Automatic fallback to the generic external adapter or another STT provider.
- Treating a server model ID as proof of the model weights behind it.

## Architecture

### Provider registration

Extend `SttProviderName` with `AUDIO_CPP = "audio-cpp"` and register an
`AudioCppAdapter` in `SttProviderRegistry.DEFAULT_ADAPTERS`. Add aliases for
`audiocpp` and `audio_cpp`.

The adapter remains visible in capability discovery even when disabled.
Planning or execution fails clearly when `audio_cpp_enabled` is false.
Registry lookup for `audio-cpp` never falls back to `external` or
`faster-whisper`.

### Module boundary

Add a focused
`tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_AudioCpp.py`
module. It owns:

- canonical audio.cpp origin handling;
- bounded parsing of `/health` and `/v1/models`;
- exact ASR model discovery;
- process-scoped discovery caching;
- WAV multipart transcription;
- bounded transcription response validation; and
- conversion of the validated response into `SttTranscriptionOutcome`.

`AudioCppAdapter` remains in `stt_provider_adapter.py` with the other STT
provider registrations. It owns capability reporting, configuration
projection, execution planning, request-semantic enforcement, and delegation
to the audio.cpp transport module.

The implementation reuses tldw_server's existing HTTP client, endpoint
normalization, egress classification, opaque STT observability, safe local
file access, execution-plan types, and artifact finalization.

### tldw_chatbook reuse boundary

tldw_chatbook's audio.cpp adapter is an architectural reference for origin
validation, bounded contract parsing, and safe error classification. Its code
is AGPL-3.0-or-later, while the tldw_server backend is GPL-3.0-only.
Implementation in this repository must therefore be written fresh rather
than copied verbatim. Upstream audio.cpp response shapes may be represented by
new fixtures with explicit provenance.

## Configuration

Add these keys under `[STT-Settings]`:

```ini
audio_cpp_enabled = false
audio_cpp_base_url = http://127.0.0.1:8080
audio_cpp_default_model =
audio_cpp_timeout_seconds = 600
```

Add explicit environment overrides:

- `STT_AUDIO_CPP_ENABLED`
- `STT_AUDIO_CPP_BASE_URL`
- `STT_AUDIO_CPP_DEFAULT_MODEL`
- `STT_AUDIO_CPP_TIMEOUT_SECONDS`

Environment values take precedence over `config.txt`. Invalid boolean,
timeout, origin, or model configuration fails closed rather than silently
using a different provider.

`audio_cpp_base_url` must be an HTTP(S) origin. User information, path,
query, fragment, malformed ports, and ambiguous host forms are rejected.
The adapter derives these paths from the origin:

- `GET /health`
- `GET /v1/models`
- `POST /v1/audio/transcriptions`

HTTPS verification is always enabled. There is no verification-disable knob.
Upstream audio.cpp does not expose authentication; deployments requiring
authentication should use an authenticated reverse proxy or the existing
generic external-provider integration.

`audio_cpp_default_model` is used only when an ordinary tldw_server request
does not supply a model. Benchmark targets must explicitly provide the model
portion of `audio-cpp=<model>`.

Configuration parsing is strict. Boolean values use an explicit accepted
token set, and the timeout must be finite and greater than zero. The new
parser does not use permissive helpers that replace invalid values with
defaults. An empty default model is valid while the provider is disabled or
an ordinary request supplies an explicit audio.cpp model; it fails when an
ordinary request actually needs the default.

### Ordinary API model selection

Ordinary OpenAI-compatible transcription requests select audio.cpp with
`audio-cpp:<server-model-id>`. The `audiocpp:` and `audio_cpp:` prefixes are
accepted as aliases and normalized to `audio-cpp`. Exact selectors
`audio-cpp`, `audiocpp`, and `audio_cpp` use `audio_cpp_default_model`.

When audio.cpp is the configured default provider, an absent request model
also uses `audio_cpp_default_model`. A bare audio.cpp server model ID is never
treated as an audio.cpp selector because it can collide with existing local
provider heuristics such as `qwen3-asr`.

`resolve_provider_for_model()` handles the canonical selector and aliases
before existing model-family heuristics, and
`_resolve_default_model_for_provider()` supplies the configured default.
These paths return the canonical provider and the exact upstream server model
ID. Missing or unsafe model IDs fail closed; they do not fall through to
Whisper or another adapter.

Some existing REST, ingestion, and Jobs dispatchers select the resolved
adapter but still pass the original non-empty request model to
`transcribe_batch()`. The adapter therefore owns the final normalization
invariant at both `plan_batch_execution()` and unplanned
`transcribe_batch()` entry points:

- strip a canonical or aliased `<selector>:` prefix and use the remaining
  exact server model ID;
- resolve an exact canonical or aliased selector through the configured
  default; and
- accept an unprefixed safe model only after the caller has already selected
  `AudioCppAdapter`, as the benchmark does.

Consequently, `audio-cpp:<model>` is never sent upstream verbatim. Updating
dispatchers to pass their resolved `provider_model_name` is optional cleanup,
not a correctness dependency.

## Planning contract

`AudioCppAdapter.plan_batch_execution()` performs no network I/O. It:

1. requires `audio_cpp_enabled=true`;
2. requires a safe explicit model label;
3. canonicalizes the configured origin;
4. derives and normalizes the transcription endpoint;
5. classifies its egress as loopback or remote;
6. rejects unsupported task or decoding semantics;
7. records a single route with backend `audio_cpp_http`;
8. records `source="audio_cpp_http"`;
9. records the opaque endpoint ID;
10. resolves the available async HTTP transport without performing I/O and
    records it on the route;
11. freezes the canonical origin, exact upstream model ID, validated timeout,
    and selected HTTP transport in in-memory `runtime_settings`;
12. sets `local_model_available=false` and `would_download=false`; and
13. keeps `identity_resolved=false` and `artifact_id=None`.

The transcription endpoint, endpoint identity, egress class, and transport
are therefore part of the authorization boundary. Runtime never rereads
audio.cpp configuration. Before any request, it derives the three URLs from
the frozen origin, verifies that the derived transcription endpoint still
matches the planned endpoint ID and egress class, and verifies that the
selected HTTP transport matches both the frozen runtime value and
`route.transport`.

An ordinary call without an incoming execution plan first resolves its
explicit/default audio.cpp model, builds this same immutable plan, and then
executes through the planned path. It does not maintain a second,
configuration-reading execution path.

The native benchmark coordinator uses the route egress to require
`--allow-network-targets` before starting a worker. An API key or loopback
address is not treated as network consent.

The initial benchmark-planning surface supports `neutral-v1`. Ordinary
tldw_server batch requests use the same transport contract outside the
benchmark. `production-v1` is not added until the provider has a separately
specified and tested production-configuration identity contract.

## Runtime data flow

1. Resolve the selected adapter and immutable execution plan.
2. Resolve the WAV path through the existing safe-path boundary.
3. Validate the file as the supported WAV subset before any network call.
4. Look up the discovery cache by normalized endpoint identity and exact model
   ID.
5. On a cache miss, request and validate `/health`.
6. Request and validate `/v1/models`.
7. Require one exact model ID with `task=asr`; record its family and mode.
8. Upload the authorized WAV as multipart form data with:
   - `file` named `audio.wav`;
   - the exact configured `model`;
   - optional benchmark/request `language`; and
   - no prompt, hotwords, diarization, word-timestamp, or streaming fields.
9. Validate a bounded JSON object containing a string `text` field.
10. Accept an empty string as a valid transcript for silence.
11. Return a normalized STT artifact and typed actual-execution record.

The first version accepts only regular files with a case-insensitive `.wav`
suffix that the Python standard library can open as an uncompressed
RIFF/WAVE PCM container. It verifies the RIFF/WAVE header and reads the
container parameters far enough to reject truncated, malformed, compressed,
or renamed non-WAV input. The upload uses a fresh file handle positioned at
byte zero; if validation and upload ever share a handle, the implementation
must explicitly rewind it before the multipart request. This intentionally
narrow subset may reject WAV encodings that audio.cpp itself can decode, but
it makes the pre-network guarantee testable and can be broadened separately.

The artifact metadata contract contains exactly these string keys:

- `provider`: `audio-cpp`;
- `contract`: `audio_cpp_http_v1`;
- `model_id`: the requested server model ID;
- `model_family`: the discovered model family;
- `model_mode`: the discovered model mode; and
- `server_backend`: the discovered audio.cpp server backend.

`finalize_stt_artifact()` currently drops provider `metadata`. Extend it with
an optional metadata allowlist whose default is empty, preserving current
behavior for all existing providers. When an adapter opts in, the finalizer
requires a mapping containing only allowlisted keys and bounded string
values; unknown keys, non-string values, excessive counts, and oversized
values fail closed. `AudioCppAdapter` opts in only for the six fields above,
and the finalized artifact retains them under `metadata`. Raw upstream
objects are never copied into the artifact.

Upstream timing fields are not used for benchmark calculations. The benchmark
continues to use its client-side monotonic adapter timer.

## Discovery cache and concurrency

Discovery is cached per adapter process and keyed by opaque endpoint identity
plus exact model ID. Access is protected by a lock so concurrent first use does
not issue an unbounded discovery burst. On a miss, the lock covers one
health-plus-model-catalog discovery sequence for that key. It does not cover
the transcription request, so independent audio uploads are not serialized.

The cache is invalidated after:

- a health or model-contract failure;
- a transport failure;
- an unknown-model response; or
- a transcription response indicating the configured model is unavailable.

The failed transcription is not retried. Invalidation only allows a later
request to perform fresh discovery. Resetting the STT provider registry also
calls a module reset hook that clears the cache under the same lock.

## Cold and warm timing semantics

The benchmark starts one fresh Python worker per target. For audio.cpp:

- `cold_first_transcription_seconds` includes first-use adapter discovery,
  the multipart transcription request, and any model loading performed by an
  audio.cpp server that was still lazy/cold.
- Warm calls reuse the adapter's discovery cache and the external server's
  already-loaded model/session.

Because audio.cpp is an independently managed process, the adapter cannot
guarantee server cold state. A true audio.cpp cold-start measurement requires
the operator to restart `audiocpp_server` immediately before the benchmark
run and leave audio.cpp lazy loading enabled. If the server is already warm,
the cold-first number measures only fresh tldw_server adapter state against a
warm backend.

## Contract validation

The new contract parser accepts unknown response fields but validates the
stable fields it consumes.

### Health

Require a bounded JSON object with:

- `status == "ok"`;
- a safe string `backend`; and
- a bounded non-negative integer `models`.

### Model catalog

Require a bounded OpenAI-style list with safe model entries. Consume:

- `id`;
- `family`;
- `task`; and
- `mode`.

Reject duplicate JSON keys, duplicate model IDs, unsafe identifiers, excessive
entry counts, malformed encodings, and oversized bodies. Select models using
exact ID equality, never substring or case-folded matching.

Both audio.cpp `offline` and `streaming` ASR entries may be used through the
non-streaming batch HTTP request because audio.cpp returns a complete final
transcript for that request form. The discovered mode remains visible in
metadata.

### Transcription

Require a bounded JSON object with a `text` string. Empty text is valid.
Ignore unknown fields. Do not use upstream timing values to replace or adjust
the benchmark's timing. The adapter returns empty or whitespace-only text as
a valid artifact. The benchmark then applies its existing outcome policy:
whitespace-only text is classified as `status="empty"` and scored as an empty
hypothesis rather than as a provider exception.

Fixtures record the reviewed audio.cpp commit. Contract changes require a new
fixture provenance record and focused compatibility review.

## Failure behavior

Planning raises the existing STT execution exceptions for:

- disabled or incomplete configuration;
- unsafe endpoint or model identifiers;
- unsupported benchmark mode or semantic options; and
- a target that cannot be represented without fallback.

Runtime failures include:

- non-WAV input;
- connection or timeout failure;
- non-success health;
- malformed or oversized metadata;
- missing exact ASR model;
- server-busy response;
- rejected multipart request;
- malformed or oversized transcription JSON; and
- actual/planned route mismatch.

Failures never invoke another provider and never trigger an implicit retry.
The benchmark records them as failures with an empty hypothesis, preserving
the failure penalty in aggregate accuracy.

Errors and logs remain bounded and do not serialize raw endpoint URLs,
response bodies, transcript text, audio paths, or credentials into benchmark
metadata. Planned STT HTTP logging uses the opaque endpoint observability
context.

## Security and privacy

- No network I/O occurs during planning.
- Loopback and remote targets both require explicit benchmark network consent.
- Redirects are disabled.
- HTTPS verification cannot be disabled.
- Local audio paths pass through the existing safe-path boundary.
- Only WAV inputs are sent.
- Audio is sent only to the planned normalized endpoint.
- Endpoint identities in persisted benchmark data are opaque hashes.
- Response metadata and identifiers are size-bounded.
- Raw audio and transcripts follow the benchmark's existing retention policy.
- No provider response or endpoint value is logged verbatim during planned
  execution.

## Testing strategy

Normal tests use fake transports and upstream-shaped fixtures. CI does not
install audio.cpp or download models.

### Configuration tests

- Defaults and disabled behavior.
- Environment precedence.
- Canonical origin handling.
- Invalid booleans, timeouts, schemes, ports, user information, paths,
  queries, fragments, and ambiguous host forms.
- Explicit benchmark model versus ordinary-request default model.
- Canonical and aliased ordinary selectors, exact-selector default use, and
  precedence over existing model-family heuristics.
- Adapter-side stripping of ordinary selector prefixes even when a dispatcher
  forwards the original request model.
- Strict boolean and finite-positive-timeout parsing without silent defaults.

### Registry and planning tests

- Canonical ID and both aliases.
- Default-provider normalization.
- Strict lookup with no fallback.
- Planning performs no HTTP request.
- Loopback and remote egress classification.
- No-download and unresolved-artifact fields.
- Rejection of translation, prompts, hotwords, diarization, word timestamps,
  unsupported modes, and disabled configuration.
- Secret-safe plan serialization.
- Frozen origin, exact model, timeout, and transport despite later config
  mutation.
- Runtime endpoint/egress/transport verification before discovery.

### Contract tests

- Valid health and mixed model catalog.
- Exact ASR model selection.
- Offline and streaming ASR entries.
- Duplicate keys and duplicate IDs.
- Invalid status, task, encoding, identifiers, counts, and body sizes.
- Unknown fields accepted.
- Fixture provenance.

### Execution tests

- Multipart field names, fixed WAV filename, model, and optional language.
- RIFF/WAVE PCM validation rejects renamed, truncated, and compressed inputs
  before network I/O and uploads from byte zero.
- Empty and non-empty transcripts.
- Discovery cached after first success.
- Concurrent first use performs one discovery sequence.
- Cache invalidated after contract, transport, and unknown-model failures.
- No redirect, retry, fallback, or download behavior.
- Non-WAV inputs fail before network I/O.
- Busy, timeout, malformed JSON, missing text, and server errors.
- Actual execution and artifact metadata remain consistent with the plan.
- Finalizer preserves only the audio.cpp metadata allowlist and rejects
  unknown, non-string, or oversized metadata.

### Benchmark tests

- `audio-cpp=<model>` target preparation.
- Explicit network consent enforcement.
- Strict and normalized scoring.
- Cold-first and warm timing classification.
- Unresolved model identity prevents policy-gate eligibility.
- Failure and empty-output scoring.
- Resume behavior preserves the existing recovery contract.

### Opt-in live test

An opt-in golden target may use the existing STT golden-test environment with
an explicit audio.cpp target and network-consent flag. It is skipped by
default and must not download models.

## Documentation

Update:

- `tldw_Server_API/Config_Files/config.txt`;
- the CPU/GPU audio setup guidance as appropriate;
- `Docs/User_Guides/STT_Benchmark_User_Guide.md`;
- `Docs/Development/STT_Benchmark_Protocol.md`;
- `Helper_Scripts/benchmarks/README.md`; and
- the published user-guide mirror.

Documentation must include:

- building and starting `audiocpp_server` is the operator's responsibility;
- the required ASR server configuration and exact model ID;
- the provider configuration and environment variables;
- the benchmark target and network-consent flag;
- WAV-only input;
- descriptive/unresolved model identity;
- restart instructions for a true audio.cpp cold start; and
- no fallback or automatic download behavior.

## Verification

Before completion:

- run focused configuration, registry, contract, execution, and benchmark
  tests;
- run existing STT adapter and benchmark regression suites;
- run changed-file pre-commit checks;
- run Bandit over touched Python implementation paths;
- run `git diff --check`; and
- perform the repository's code-review and verification-before-completion
  gates.

## Rollback

The provider defaults to disabled. Rollback consists of disabling
`audio_cpp_enabled` or reverting the provider registration and configuration.
Existing STT providers and the generic external-provider path remain
unchanged. There is no managed process or downloaded state to clean up.
