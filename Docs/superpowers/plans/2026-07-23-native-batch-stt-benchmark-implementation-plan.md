# Native Batch STT Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone, batch-only benchmark that exercises tldw_server's native STT adapters, deterministically scores strict and normalized WER/CER, separates cold-first from warm adapter timing, and produces crash-safe resumable artifacts without Pipecat or an LLM judge.

**Architecture:** Extend the existing `SttProviderAdapter` boundary with a backward-compatible immutable execution plan and allowlisted actual-execution envelope. Keep the harness in one importable standard-library CLI module under `Helper_Scripts/benchmarks/`; use a coordinator plus one spawned worker per provider/model target, append-only/fsync JSONL, and disposable reports rebuilt from immutable run metadata and result history. Reuse the native registry, provider helpers, centralized transcription-error recognition, pytest, Hypothesis, the existing golden-test marker, and the repository's HTTP client.

**Tech Stack:** Python 3.10+, standard library (`argparse`, `dataclasses`, `hashlib`, `json`, `multiprocessing`, `subprocess`, `unicodedata`), existing STT adapters/loaders, pytest, Hypothesis, optional psutil best-effort metrics, Bandit

## Global Constraints

- The approved source of truth is `Docs/superpowers/specs/2026-07-22-native-batch-stt-benchmark-design.md`.
- Do not use Pipecat, an LLM judge, FastAPI, Jobs, the Evaluations service, a benchmark framework, or a parallel provider registry.
- Preserve all existing callers: `execution_plan` is optional and omitted calls retain current production behavior.
- A benchmark plan is approved before audio is opened, a model is loaded, or a network request is made.
- The supplied plan pins backend, endpoint, fallback policy, semantic settings, device/compute contract, and no-download behavior. Planned execution must not reread material configuration.
- Unknown providers, unknown egress, missing local artifacts, unenforceable no-download behavior, neutral-mode fallback, redirects, and any actual execution route outside the approved ordered routes fail closed. Non-route semantic/result mismatches are retained explicitly and make the result gate-ineligible.
- Only the explicit safe descriptor and `SttActualExecution` envelope may enter benchmark artifacts. Never copy unrestricted adapter metadata.
- `neutral-v1` is the only model-quality mode. `production-v1` compares complete configurations and may mix planned backends only when reported by actual backend.
- Public/private suites and `diagnostic_only` samples stay separate in every headline quality metric and gate.
- Accuracy has one active scored record per sample. Additional warm repetitions are performance-only records.
- All real-model tests remain opt-in; ordinary CI uses fake adapters and generated tiny fixtures.
- Before changing repository files for an implementation slice, search/create its own Backlog.md child task and link this plan and the approved design.
- Work in an isolated `codex/` branch/worktree. The current planning checkout is dirty with unrelated user changes; never stage them.
- Activate `.venv` before every Python, pytest, or Bandit command.
- Use test-first red/green/refactor steps. Do not proceed from a slice while its focused tests fail.
- Run Bandit on every touched Python scope and `git diff --check` before each slice commit.
- At each commit step, stage only the exact paths listed under that task's **Files** plus the exact Backlog task path printed by the CLI. Never use `git add -A` or `git add .`.
- If the single CLI module becomes genuinely difficult to test independently, stop and amend this plan before splitting it; do not create a speculative package hierarchy.

---

## Current-Code Findings That Drive the Plan

1. `SttProviderRegistry.get_adapter()` intentionally falls back to faster-whisper for unknown names. The benchmark needs a separate strict lookup, not a behavioral change to production lookup.
2. Parakeet, Canary, Qwen2Audio, Qwen3-ASR, and VibeVoice can currently fall back or reread configuration inside execution. Benchmark calls must bypass those choices with the approved plan.
3. Faster-whisper can download on first use and can silently fall back from CUDA to CPU/int8.
4. NeMo `from_pretrained`, Parakeet ONNX `snapshot_download`, Parakeet MLX `from_pretrained`, Qwen2Audio `from_pretrained`, and local Transformers loaders need explicit no-download enforcement.
5. Qwen3-ASR and VibeVoice can send audio to configured HTTP services. VibeVoice can then fall back from HTTP to local inference.
6. Qwen3-ASR currently normalizes vLLM results with `source="local"`, VibeVoice metadata currently contains raw hotwords, and external providers currently return `[Error: ...]` strings.
7. `Audio_Transcription_Lib.is_transcription_error_message()` is the central sentinel recognizer and should remain the source of truth.
8. The current golden test has a private ASCII-only normalizer and Levenshtein implementation; both must be removed in favor of the benchmark scorer.
9. `.benchmarks/` is not currently ignored.
10. FFmpeg/ffprobe is already a project prerequisite and is the one duration-probe path that supports the planned WAV, FLAC, and compressed inputs without a new Python dependency.

## Delivery Sequence

1. Shared execution-plan contract and strict registry boundary.
2. Local adapter/loader enforcement.
3. Dynamic and network adapter/loader enforcement.
4. Deterministic scorer.
5. Manifest validation and immutable run identity.
6. Crash-safe persistence and aggregation primitives.
7. Spawned target worker, coordinator, and `run`.
8. `report`, `compare`, regression policy, retention, and security behavior.
9. Golden-test migration, example manifest, protocol docs, and final gates.

Keep these slices sequential. Slices 2 and 3 both modify `stt_provider_adapter.py`; slices 4-8 all modify `stt_bench.py`.

## Planned File Map

- `tldw_Server_API/app/core/exceptions.py`
  - Typed execution-plan/preflight failures and existing `STTTranscriptionError`.
- `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_execution_contract.py`
  - Dependency-neutral frozen routes, plan, actual execution, loaded-runtime, and transcription-outcome types.
- `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_provider_adapter.py`
  - Re-exported contract types, strict lookup, per-adapter planning, plan validation, sentinel enforcement.
- `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Lib.py`
  - Planned faster-whisper/Qwen2/Parakeet execution, no fallback, no download, exact device/compute behavior.
- `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Nemo.py`
  - Explicit local NeMo artifacts and no-download loading.
- `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Parakeet_ONNX.py`
  - `allow_download=False` enforcement.
- `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Parakeet_MLX.py`
  - Explicit local model/settings path with no config reread or fallback.
- `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Qwen3ASR.py`
  - Planned local/vLLM execution, no redirects, corrected actual source.
- `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_VibeVoice.py`
  - Planned local/vLLM/fallback execution, no redirects, no hotword leakage.
- `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_External_Provider.py`
  - Planned immutable provider config and typed sentinel conversion.
- `Helper_Scripts/benchmarks/stt_bench.py`
  - Scorer, manifest, run identity, persistence, worker/coordinator, reports, comparisons, and argparse CLI.
- `Helper_Scripts/benchmarks/stt_benchmark_manifest.example.jsonl`
  - Schema example only; no large corpus.
- `tldw_Server_API/tests/Benchmarks/test_stt_bench.py`
  - Scorer, property, manifest, persistence, runner, report, compare, and CLI tests.
- Existing provider tests under `tldw_Server_API/tests/Audio/`
  - Provider-specific plan enforcement and production-call compatibility.
- `tldw_Server_API/tests/Audio/test_stt_adapters_golden.py`
  - Opt-in real adapters using the regression manifest and shared scorer.
- `Helper_Scripts/Audio/generate_stt_golden.py`
  - Human/public reference or explicit unverified candidate output.
- `tldw_Server_API/tests/Helper_Scripts/test_generate_stt_golden.py`
  - Reference-provenance rules.
- `Docs/Development/STT_Benchmark_Protocol.md`
  - Dataset, annotation, scoring, privacy, timing, and comparison protocol.
- `Docs/Development/STT_Adapter_Golden_Tests.md`
  - Updated opt-in workflow.
- `Helper_Scripts/benchmarks/README.md`
  - CLI quick start and artifact warning.
- `.gitignore`
  - `.benchmarks/stt/`.

---

## Slice 1: Shared Native Execution Contract

### Task 1: Add Frozen Plan Types, Strict Registry Lookup, and Artifact Finalization

**Files:**

- Modify: `tldw_Server_API/app/core/exceptions.py`
- Create: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_execution_contract.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_provider_adapter.py`
- Modify: `tldw_Server_API/tests/Audio/test_stt_provider_adapter.py`

- [ ] **Step 1: Write failing contract tests**

Add tests that prove:

- `SttBatchExecutionPlan` and nested descriptor are frozen and pickleable.
- decoding/runtime setting keys must be unique and stored in lexicographic order; duplicate or non-canonical tuples are rejected.
- secret runtime values do not appear in `repr(plan)` or `descriptor.as_safe_dict()`.
- the safe descriptor contains only its declared fields.
- importing the dependency-neutral contract module and every touched loader in
  either order does not create an adapter/provider circular import.
- `get_adapter_strict("unknown")` raises instead of returning faster-whisper.
- existing `get_adapter("unknown")` still returns the production fallback.
- omitted `execution_plan` keeps each adapter's existing call signature behavior.
- a plan/provider/model/semantic mismatch fails before the mocked provider helper is invoked.
- a recognized `[Error: ...]` artifact text raises `STTTranscriptionError`.
- hostile unrestricted metadata never becomes `actual_execution`.

Use this public contract:

```python
SttPlanScalar = str | int | float | bool | None | tuple[str, ...]

class SttAudioEgress(str, Enum):
    NONE = "none"
    LOOPBACK = "loopback"
    REMOTE = "remote"

@dataclass(frozen=True)
class SttExecutionRoute:
    route_id: str
    provider: str
    model_label: str
    artifact_id: str | None
    identity_resolved: bool
    backend: str
    source: str
    audio_egress: SttAudioEgress
    endpoint_id: str | None
    device: str | None
    compute_type: str | None
    dtype: str | None
    decoding_ids: tuple[str, ...]
    local_model_available: bool
    would_download: bool

@dataclass(frozen=True)
class SttExecutionDescriptor:
    requested_provider: str
    requested_model_label: str
    resolved_provider: str
    resolved_model_label: str
    routes: tuple[SttExecutionRoute, ...]
    honors_task: bool
    honors_language: bool
    honors_prompt_absence: bool
    honors_hotword_absence: bool
    honors_diarization: bool
    honors_word_timestamps: bool
    decoding_settings: tuple[tuple[str, SttPlanScalar], ...]
    source_modules: tuple[str, ...]
    dependency_distributions: tuple[str, ...]

@dataclass(frozen=True)
class SttBatchExecutionPlan:
    descriptor: SttExecutionDescriptor
    task: str
    language: str | None
    prompt: str | None = field(default=None, repr=False)
    hotwords: tuple[str, ...] = field(default=(), repr=False)
    diarization: bool = False
    word_timestamps: bool = False
    runtime_settings: tuple[tuple[str, SttPlanScalar], ...] = field(
        default=(), repr=False
    )

    def runtime_values(self) -> dict[str, SttPlanScalar]:
        return dict(self.runtime_settings)

@dataclass(frozen=True)
class SttActualExecution:
    route_id: str
    provider: str
    model_label: str
    artifact_id: str | None
    backend: str
    audio_egress: SttAudioEgress
    endpoint_id: str | None
    source: str
    device: str | None
    compute_type: str | None
    dtype: str | None
    decoding_ids: tuple[str, ...] = ()

@dataclass(frozen=True)
class SttLoadedRuntime:
    components: tuple[Any, ...] = field(repr=False, compare=False)
    actual_execution: SttActualExecution

@dataclass(frozen=True)
class SttTranscriptionOutcome:
    artifact: dict[str, Any] = field(repr=False, compare=False)
    actual_execution: SttActualExecution
```

`SttExecutionDescriptor.primary_route` returns `routes[0]`; `fallback_allowed` is `len(routes) > 1`; `as_safe_dict()` serializes the ordered route list and declared safe fields only. `SttExecutionRoute.as_safe_dict()` and `SttActualExecution.as_safe_dict()` serialize their exact declared fields and enum values. `SttBatchExecutionPlan.runtime_values()` returns `dict(runtime_settings)`.

`routes` is the complete ordered authorization. `neutral-v1` has exactly one route. A production VibeVoice HTTP→local fallback has two routes with distinct `route_id`, backend/source, egress, endpoint ID, model/artifact, device/compute, and decoder-ID contracts. `finalize_stt_artifact` accepts an actual execution only when it matches one declared route on every non-null material field; an undeclared route is an execution error, not merely a gate-ineligible mismatch.

Do not put endpoint URLs, credentials, headers, prompt/hotword text, arbitrary metadata, absolute local paths, or content-derived speech hashes in any safe dictionary. Safe model labels must pass a conservative provider/model-label validator and must not parse as a path or URL; use `local-model` when the requested identifier is path-like. `artifact_id` is identity-resolved only for a full content SHA-256 or an immutable snapshot commit (not a branch such as `main`); otherwise set `identity_resolved=False`. Endpoint IDs match `sha256:[0-9a-f]{64}`. `source_modules` and `dependency_distributions` let the harness fingerprint the real implementation without hardcoding a second provider map.

Never use `dataclasses.asdict(plan)`, because it would traverse secret runtime fields. Use `__post_init__` validation on the frozen dataclasses to reject duplicate/non-lexicographic setting keys, decoder IDs, and module/distribution lists; duplicate/blank route IDs; blank provider/model/backend/source values; an endpoint ID on `audio_egress=none`; a network egress without a valid endpoint ID; path/URL-like safe labels; a mutable revision claimed as resolved identity; or a fallback route identical to an earlier route. Decoder IDs must be stable identifiers and exactly name the allowlisted effective decoding settings used on that route. Source modules must be valid dotted Python identifiers under `tldw_Server_API` or `Helper_Scripts`; dependency names must match the PEP-503 character set and contain no path separator. Resolve every source-module file beneath the repository root before hashing it.

Keep `stt_execution_contract.py` standard-library-only and independent of
`stt_provider_adapter.py` and provider modules. Loaders, provider helpers, and
the adapter import contract types from that module; the adapter re-exports the
public names for existing callers. If an annotation would otherwise require a
reverse import, use `TYPE_CHECKING` plus a quoted annotation. No provider
module may import the adapter at runtime.

- [ ] **Step 2: Run the tests to verify red**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audio/test_stt_provider_adapter.py -q
```

Expected: FAIL because the plan types and strict lookup do not exist.

- [ ] **Step 3: Add typed failures and the backward-compatible adapter methods**

Add:

```python
class STTExecutionPlanError(BadRequestError):
    """Raised when a planned STT execution cannot be honored."""

class STTExecutionUnsupportedError(STTExecutionPlanError):
    """Raised when an adapter cannot safely expose the benchmark contract."""
```

Add a concrete default `plan_batch_execution` to `SttProviderAdapter` that raises `STTExecutionUnsupportedError`; this lets adapters become benchmark-capable incrementally without breaking production construction.

```python
def plan_batch_execution(
    self,
    *,
    model: str | None,
    language: str | None,
    task: str,
    word_timestamps: bool,
    prompt: str | None,
    hotwords: Sequence[str] | None,
    diarization: bool,
    mode: str,
) -> SttBatchExecutionPlan:
    raise STTExecutionUnsupportedError(
        f"Provider {self.name.value} does not expose enforceable benchmark planning"
    )
```

Extend the abstract and all concrete `transcribe_batch` signatures with:

```python
execution_plan: SttBatchExecutionPlan | None = None
```

Add `SttProviderRegistry.get_adapter_strict(provider_name: str)`. It normalizes aliases but checks the registered adapter directly and raises `STTExecutionPlanError` when absent. Do not change `get_adapter()`.

- [ ] **Step 4: Add one shared finalization boundary**

Implement:

```text
finalize_stt_artifact(
    artifact: object,
    *,
    plan: SttBatchExecutionPlan,
    actual: SttActualExecution,
) -> dict[str, Any]
```

Its implementation must:

1. Require a mapping with string `text` and list `segments`.
2. Reuse `Audio_Transcription_Lib.is_transcription_error_message()` via a lazy import.
3. Raise `STTTranscriptionError` for a recognized sentinel.
4. Discard any provider-supplied `actual_execution` value and replace it with only `actual.as_safe_dict()` at the top-level key.
5. Require the actual execution to equal one declared route on route ID plus
   every non-null material provider, model/artifact, backend/source,
   egress/endpoint, device, compute, dtype, and ordered decoder-ID field.
6. Add a bounded `execution_mismatch` list only for non-route semantic/result
   mismatches; never copy `metadata`. An undeclared actual route raises
   `STTExecutionPlanError`.

All seven benchmark-capable adapters call this helper only in their
`execution_plan is not None` branch. An omitted plan follows the current
production code path and returns its exact legacy artifact without adding an
unverified actual-execution envelope. A supplied plan requires a typed actual
outcome and returns the safe envelope.

- [ ] **Step 5: Re-run focused tests**

Expected: PASS.

- [ ] **Step 6: Run security and diff gates**

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/exceptions.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_execution_contract.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_provider_adapter.py \
  -f json -o /tmp/bandit_stt_bench_contract.json
git diff --check
```

Expected: no new findings and no whitespace errors.

- [ ] **Step 7: Commit Slice 1**

```bash
git add \
  tldw_Server_API/app/core/exceptions.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_execution_contract.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_provider_adapter.py \
  tldw_Server_API/tests/Audio/test_stt_provider_adapter.py
git commit -m "feat: add native STT execution plan contract"
```

Also stage the exact Backlog task path printed by `backlog task create` for this slice.

---

## Slice 2: Local Provider and Loader Enforcement

### Task 2: Make Faster-Whisper, NeMo, Parakeet, and Qwen2Audio Honor No-Download Plans

**Files:**

- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_provider_adapter.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Lib.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Nemo.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Parakeet_ONNX.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Parakeet_MLX.py`
- Modify: `tldw_Server_API/tests/Audio/test_stt_provider_adapter.py`
- Modify: `tldw_Server_API/tests/Audio/test_parakeet_onnx_failfast.py`
- Create: `tldw_Server_API/tests/Audio/test_stt_execution_plan_local.py`

- [ ] **Step 1: Write failing local-plan tests**

Cover:

- faster-whisper rejects an unavailable local model and calls `WhisperModel` with `local_files_only=True` and the planned device/compute type.
- planned CUDA failure is an error; it never retries CPU.
- planned neutral execution bypasses custom-vocabulary prompt injection.
- Qwen2Audio uses a planned local path/revision with `local_files_only=True`, ignores later config mutation, and never falls back to whisper in `neutral-v1`.
- Canary and standard Parakeet require an explicit local `.nemo` artifact for no-download benchmark execution.
- Parakeet ONNX passes `allow_download=False` and never calls `snapshot_download`.
- Parakeet MLX accepts only an already-existing explicit local artifact and never falls back to standard/CPU.
- Parakeet and Qwen2Audio planners record
  `language_contract="fixed:en"`, accept a primary language subtag of `en`,
  and reject non-English manifests because their current native paths do not
  honor arbitrary language hints.
- unsupported variants fail preflight before the audio-open mock.
- actual device/compute/dtype comes from the loaded runtime, not the requested string.

- [ ] **Step 2: Verify red**

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Audio/test_stt_execution_plan_local.py \
  tldw_Server_API/tests/Audio/test_parakeet_onnx_failfast.py \
  -q
```

Expected: FAIL because local helpers do not accept planned settings.

- [ ] **Step 3: Add explicit loader controls without changing default behavior**

Use keyword-only additions:

```text
get_whisper_model(
    model_name: str,
    device: str,
    check_download_status: bool = False,
    *,
    compute_type_override: str | None = None,
    local_files_only: bool = False,
    allow_device_fallback: bool = True,
    execution_route: SttExecutionRoute | None = None,
) -> Any | tuple[None, dict[str, Any]] | SttLoadedRuntime

load_qwen2audio(
    *,
    model_id: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    device_map: str | None = None,
    dtype_name: str | None = None,
    execution_route: SttExecutionRoute | None = None,
) -> tuple[Any, Any] | SttLoadedRuntime

load_canary_model(
    *,
    model_path: str | None = None,
    device: str | None = None,
    dtype_name: str | None = None,
    allow_download: bool = True,
    execution_route: SttExecutionRoute | None = None,
) -> Any | SttLoadedRuntime

load_parakeet_model(
    variant: str = "standard",
    *,
    model_path: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    allow_download: bool = True,
    allow_variant_fallback: bool = True,
    execution_route: SttExecutionRoute | None = None,
) -> Any | SttLoadedRuntime

load_parakeet_onnx_model(
    model_path: str | None = None,
    device: str = "cpu",
    *,
    allow_download: bool = True,
    execution_route: SttExecutionRoute | None = None,
) -> tuple[Any, Any] | SttLoadedRuntime

load_parakeet_mlx_model(
    *,
    force_reload: bool = False,
    model_path: str | None = None,
    cache_dir: str | None = None,
    allow_download: bool = True,
    execution_route: SttExecutionRoute | None = None,
) -> Any | None | SttLoadedRuntime
```

Add the shown annotations while implementing the behavior below.

When `allow_download=False`, require the planned local path and use native offline/local-only options. If a library cannot prove local-only execution for a requested form, raise `STTExecutionUnsupportedError`; do not emulate cache discovery with a network probe.

These are plan-aware overloads, not silent changes to legacy return values.
When `execution_route is None`, retain each exact legacy return. When it is
supplied, validate the route before library entry and return
`SttLoadedRuntime(components=(model,), actual_execution=loaded_actual)`, or the
corresponding processor/model component tuple, populated from the
loaded object's effective device/dtype/compute data. Reject a library result
whose effective values cannot be determined. Standard NeMo and Canary must
receive the pinned device explicitly rather than rereading `nemo_device`;
unsupported dtype/compute requests fail preflight.

- [ ] **Step 4: Thread the immutable plan through native execution**

Add `execution_plan: SttBatchExecutionPlan | None = None` to `speech_to_text` and the focused provider helpers it invokes. When present:

- use only `plan.runtime_settings`;
- do not reread STT configuration;
- do not inject custom vocabulary;
- do not fall back between provider, variant, device, or compute type unless that finite fallback sequence is explicitly captured in a `production-v1` plan;
- use `local_files_only=True`/`allow_download=False`;
- report the loaded actual execution envelope.

Keep omitted-plan behavior exactly as it is.

The planned `speech_to_text`, `speech_to_text_parakeet`,
`speech_to_text_canary`, `speech_to_text_qwen2audio`,
`transcribe_with_canary`, `transcribe_with_parakeet`,
`transcribe_with_parakeet_onnx`, and `transcribe_with_parakeet_mlx` branches
accept the execution plan/selected route and propagate
`SttTranscriptionOutcome(artifact=artifact,
actual_execution=loaded_actual)`. The omitted-plan branches return their exact
legacy tuple/list/string values. The four local adapters branch on
`execution_plan`, require the typed outcome in planned mode, and are the local
callers of `finalize_stt_artifact`; the three dynamic/network adapters are the
remaining planned callers. A planned branch that returns a legacy value is an
execution error because actual execution was not proven. Add
import-order smoke tests for the contract, adapter,
`Audio_Transcription_Lib`, NeMo, ONNX, and MLX modules.

Implement local `plan_batch_execution` overrides in the four adapters. A model directory without a stable revision may run descriptively with `identity_resolved=False`; it is not gate-eligible. Do not hash multi-gigabyte weights during ordinary preflight.

For v1, Faster-Whisper and Canary support `neutral-v1` for the language
semantics their native calls can honor; Parakeet and Qwen2Audio support it only
for English BCP47 tags through the explicit `fixed:en` contract. A local
provider's `production-v1` plan is supported only when its complete
provider/variant/device fallback chain can be frozen with already-local
artifacts; otherwise planning raises `STTExecutionUnsupportedError` rather
than silently benchmarking a stricter configuration than production.

In `neutral-v1`, planners force task=`transcribe`, the manifest language, no prompt/hotwords, no diarization, and no word timestamps. In `production-v1`, each planner snapshots its current production prompt/hotwords and material decoder settings into the secret plan. The safe descriptor records only prompt presence, hotword count, and an opaque user-supplied `configuration_id`; it never records their text or a digest.

- [ ] **Step 5: Re-run new and existing regression tests**

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Audio/test_stt_execution_plan_local.py \
  tldw_Server_API/tests/Audio/test_stt_provider_adapter.py \
  tldw_Server_API/tests/Audio/test_parakeet_onnx_failfast.py \
  tldw_Server_API/tests/Audio/test_stt_provider_registry_wrapper_migration.py \
  -q
```

Expected: PASS, including omitted-plan production compatibility.

- [ ] **Step 6: Run Bandit and commit**

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_provider_adapter.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Lib.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Nemo.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Parakeet_ONNX.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Parakeet_MLX.py \
  -f json -o /tmp/bandit_stt_bench_local.json
git diff --check
```

Expected: no new Bandit findings and no whitespace errors. Stage only the Task 2 paths and commit:

```bash
git commit -m "feat: enforce local STT benchmark execution plans"
```

---

## Slice 3: Dynamic and Network Provider Enforcement

### Task 3: Pin Qwen3-ASR, VibeVoice, and External Provider Execution

**Files:**

- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_provider_adapter.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Qwen3ASR.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_VibeVoice.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_External_Provider.py`
- Modify: `tldw_Server_API/tests/Audio/test_qwen3_asr.py`
- Modify: `tldw_Server_API/tests/Audio/test_vibevoice_transcription.py`
- Create: `tldw_Server_API/tests/Audio/test_stt_execution_plan_network.py`

- [ ] **Step 1: Write failing network-plan tests**

Test:

- `_classify_audio_egress(url)` returns `loopback` only for the literal hostname `localhost` and literal loopback IPs; arbitrary names (including subdomains of localhost) and non-loopback IPs are `remote`; malformed/ambiguous endpoints fail.
- endpoints containing userinfo, query, or fragment values are rejected;
  accepted endpoint IDs are opaque hashes of the complete normalized final
  transcription endpoint and never expose its path.
- Qwen3 local/vLLM and VibeVoice local/vLLM plans pin the chosen backend and exact endpoint.
- config mutation after planning cannot change backend, endpoint, fallback policy, semantic settings, device, dtype, or model.
- Qwen3 vLLM actual source is `vllm_http`, not `local`.
- `httpx.Client(follow_redirects=False)` is used for Qwen3.
- VibeVoice calls `fetch_json` with `allow_redirects=False`.
- neutral VibeVoice never falls back from vLLM to local.
- production VibeVoice follows only the fallback sequence captured in the plan and records actual backend per sample.
- VibeVoice unrestricted metadata may contain hotwords but `actual_execution` and benchmark-facing fields do not.
- External planning freezes the loaded `ExternalProviderConfig`, classifies egress, disables redirects, converts sentinel strings to `STTTranscriptionError`, and never serializes API keys or custom headers.
- prompt/hotword absence and unsupported language/task/timestamp semantics reject `neutral-v1`.
- a fixed-English backend may claim the language contract only for a manifest tag whose primary subtag is `en`; record `language_contract="fixed:en"` in safe decoding settings. A runtime path that drops a planned language or other semantic option becomes an execution mismatch and is not a valid neutral result.

- [ ] **Step 2: Verify red**

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Audio/test_stt_execution_plan_network.py \
  tldw_Server_API/tests/Audio/test_qwen3_asr.py \
  tldw_Server_API/tests/Audio/test_vibevoice_transcription.py \
  -q
```

- [ ] **Step 3: Add plan-aware helper signatures**

Use frozen settings snapshots:

```text
transcribe_with_qwen3_asr(
    audio_path: str,
    *,
    model_path: str | None = None,
    language: str | None = None,
    word_timestamps: bool = False,
    base_dir: Path | None = None,
    cancel_check: Callable[[], bool] | None = None,
    execution_plan: SttBatchExecutionPlan | None = None,
) -> dict[str, Any] | SttTranscriptionOutcome

transcribe_with_vibevoice(
    audio_path: str,
    *,
    model_id: str | None = None,
    language: str | None = None,
    hotwords: Sequence[str] | str | None = None,
    base_dir: Path | None = None,
    cancel_check: Callable[[], bool] | None = None,
    execution_plan: SttBatchExecutionPlan | None = None,
) -> dict[str, Any] | SttTranscriptionOutcome
```

Preserve the current function bodies for omitted plans and add the planned branch described below.

The planned branch must select only `plan.descriptor.routes` in order. Each
attempt returns a typed `SttTranscriptionOutcome` with the exact selected
`route_id`, source/decoder IDs, egress, endpoint ID, backend, device/dtype, and
artifact identity.
It must not synthesize actual execution from provider metadata. If all
authorized routes fail, raise the final provider error without trying any
undeclared route.

For external providers, construct `ExternalProviderConfig` from the plan in the adapter and pass it through the existing explicit `config=` argument. Do not reread environment/config in the helper.

Apply the same planned-outcome rule to the external-provider helper: omitted
plan/config callers retain the legacy artifact, while the planned call returns
`SttTranscriptionOutcome`. Each of the Qwen3, VibeVoice, and external adapters
unwraps that outcome and calls `finalize_stt_artifact`; no dynamic/network
planned branch returns directly. Add import-order smoke tests for Qwen3,
VibeVoice, external-provider, contract, and adapter modules.

- [ ] **Step 4: Implement safe egress and endpoint identity**

Use `urllib.parse.urlparse` plus `ipaddress.ip_address`. Do not perform DNS resolution. Resolve each provider's final transcription endpoint first, reject userinfo/query/fragment values, normalize scheme/host/default port/path, and hash that complete normalized endpoint into the opaque `endpoint_id`. Reject non-HTTP(S), missing host, and ambiguous ports during planning. Keep the full endpoint only in `plan.runtime_settings`.

Require the coordinator's later `--allow-network-targets` consent for both loopback and remote plans. The adapter itself still enforces that it executes only the supplied endpoint and with redirects disabled.

- [ ] **Step 5: Re-run focused and compatibility tests**

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Audio/test_stt_execution_plan_network.py \
  tldw_Server_API/tests/Audio/test_qwen3_asr.py \
  tldw_Server_API/tests/Audio/test_vibevoice_transcription.py \
  tldw_Server_API/tests/Audio/test_audio_transcriptions_adapter_path.py \
  tldw_Server_API/tests/STT/test_audio_transcription_api.py \
  tldw_Server_API/tests/TTS_NEW/integration/test_transcription_auth.py \
  -q
```

Expected: PASS.

- [ ] **Step 6: Run Bandit and commit**

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_provider_adapter.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Qwen3ASR.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_VibeVoice.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_External_Provider.py \
  -f json -o /tmp/bandit_stt_bench_network.json
git diff --check
```

Expected: no new Bandit findings and no whitespace errors. Stage only the Task 3 paths and commit:

```bash
git commit -m "feat: pin network STT benchmark execution"
```

---

## Slice 4: Deterministic Scorer

### Task 4: Implement `stt-score-v1`, Strict/English Normalization, and Aggregation Math

**Files:**

- Create: `Helper_Scripts/benchmarks/stt_bench.py`
- Create: `tldw_Server_API/tests/Benchmarks/test_stt_bench.py`

- [ ] **Step 1: Write failing example and property tests**

Test exact:

- CRLF/bare-CR exact-match normalization only.
- NFC strict whitespace collapse.
- NFKC, apostrophe mapping, casefolding, Unicode punctuation removal, and whitespace collapse for `en-v1`.
- `we're != were` and `can't != cant`.
- non-English letters survive `en-v1`; digits are not expanded.
- empty preprocessed strings return empty sequences.
- substitution/deletion/insertion counts and deterministic tie priority: match, substitution, deletion, insertion.
- WER/CER denominator behavior for empty hypotheses.
- identity, idempotent normalization, deterministic repeated scoring, non-negative internally consistent counts, and pooled-count reconstruction with Hypothesis.
- type-7 percentile interpolation at p50/p90/p95/p99.
- percentile rejects p outside `[0, 1]` and non-finite observations; an empty
  input returns `None`.

Define:

```python
SCORER_VERSION = "stt-score-v1"
STRICT_PROFILE = "strict-v1"
EN_PROFILE = "en-v1"

@dataclass(frozen=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_units: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        return self.errors / max(self.reference_units, 1)

@dataclass(frozen=True)
class TranscriptScore:
    exact_match: bool
    strict_wer: EditCounts
    strict_cer: EditCounts
    normalized_wer: EditCounts
    normalized_cer: EditCounts
```

Use these exact public signatures:

```text
normalize_exact_text(text: str) -> str
normalize_strict_v1(text: str) -> str
normalize_en_v1(text: str) -> str
edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts
score_transcript(
    reference: str,
    hypothesis: str,
    *,
    normalization_profile: str,
) -> TranscriptScore
percentile_type7(values: Sequence[float], p: float) -> float | None
```

For the scorer's direct empty-sequence API, define a zero-reference rate as `errors / max(reference_units, 1)`: both empty is `0.0`; a non-empty hypothesis retains its insertion penalty. Manifest validation rejects any reference that becomes empty under its declared normalization profile, so pooled benchmark denominators remain meaningful.

For a non-English record with no implemented language profile, strict scores remain available but manifest validation must reject the unimplemented normalized profile rather than silently use `en-v1`.

- [ ] **Step 2: Verify red**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Benchmarks/test_stt_bench.py -k "score or normalize or percentile" -q
```

- [ ] **Step 3: Implement the scorer with standard library only**

Use `unicodedata.normalize`, `unicodedata.category`, `str.isspace`, `str.isalnum`, and `str.casefold`. Do not use ASCII-only regular expressions or third-party WER libraries.

Implement the profiles in this exact order:

- exact: replace `\r\n` and bare `\r` with `\n`; change nothing else;
- strict: NFC, collapse maximal `str.isspace()` runs to one ASCII space, trim ASCII spaces;
- English: NFKC; map U+2018/U+2019/U+02BC/U+FF07 to U+0027; casefold; preserve U+0027 only between two `str.isalnum()` characters; replace other apostrophes with spaces; replace every other Unicode category `P*` character with a space; collapse/trim whitespace.

Strict/normalized WER splits only on U+0020 after preprocessing. CER uses Unicode code points including canonical internal spaces.

Implement edit counts with two dynamic-programming rows, where each cell carries distance plus substitution/deletion/insertion totals. Select equal-cost transitions with explicit priority `match`, `substitution`, `deletion`, `insertion`. This yields deterministic counts without retaining an O(reference×hypothesis) backpointer matrix for long-form samples. `percentile_type7` uses `h=(n-1)*p` and linear interpolation between `floor(h)` and `ceil(h)`.

`percentile_type7` returns `None` for an empty sequence, raises `ValueError`
when `p` is outside `[0, 1]`, and rejects NaN/infinite observations rather than
letting them contaminate JSON summaries.

- [ ] **Step 4: Re-run scorer/property tests**

Expected: PASS.

- [ ] **Step 5: Run Bandit/diff and commit**

```bash
source .venv/bin/activate
python -m bandit -r Helper_Scripts/benchmarks/stt_bench.py \
  -f json -o /tmp/bandit_stt_bench_scorer.json
git diff --check
git commit -m "feat: add deterministic STT benchmark scorer"
```

---

## Slice 5: Manifest and Immutable Run Identity

### Task 5: Validate the Hybrid Manifest Before Provider Loading

**Files:**

- Modify: `Helper_Scripts/benchmarks/stt_bench.py`
- Modify: `tldw_Server_API/tests/Benchmarks/test_stt_bench.py`

- [ ] **Step 1: Write failing manifest tests**

Cover all design validations:

- duplicate/empty IDs and references;
- `bcp47-basic-v1` syntax and canonical comparison form;
- known normalization/profile names;
- absolute paths, traversal, symlink escape, missing/non-regular files;
- SHA-256 mismatch;
- ffprobe missing/failure/non-positive duration;
- declared/measured duration tolerance `max(0.100, measured * 0.01)`;
- bounded unique tags;
- stable suite and annotation identifiers;
- public/private visibility consistency within a suite;
- complete dataset/version/license/reference provenance;
- deterministic content hash independent of manifest file location;
- deterministic profile sample order and cold-probe selection.

Use:

```python
@dataclass(frozen=True)
class ManifestSample:
    sample_id: str
    audio_relative: str
    reference: str
    language: str
    normalization_profile: str
    measured_duration_seconds: float
    profiles: tuple[str, ...]
    suite: str
    suite_visibility: str
    annotation_profile: str
    diagnostic_only: bool
    source: tuple[tuple[str, str], ...]
    tags: tuple[str, ...]
    sha256: str
```

Use these exact public signatures:

```text
probe_audio_duration_ffprobe(audio_path: Path) -> float
load_and_validate_manifest(
    manifest_path: Path,
    dataset_root: Path,
    *,
    duration_probe: Callable[[Path], float] = probe_audio_duration_ffprobe,
) -> tuple[tuple[ManifestSample, ...], str]
select_samples(
    samples: Sequence[ManifestSample],
    *,
    profile: str,
    seed: int,
) -> tuple[tuple[ManifestSample, ...], str]
```

- [ ] **Step 2: Verify red**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Benchmarks/test_stt_bench.py -k "manifest or ffprobe or sample_order" -q
```

Expected: FAIL because manifest parsing/validation is absent.

- [ ] **Step 3: Implement containment, hashing, and the single decoder path**

Resolve the dataset root and candidate with `Path.resolve(strict=True)` and
require `candidate.relative_to(root)`. Symlinks are permitted only when their
fully resolved target remains inside the resolved dataset root; reject escapes,
broken links, and non-regular files. Re-resolve and repeat containment plus
SHA-256 verification immediately before scheduling so a post-validation link
swap cannot redirect the worker.

Invoke:

```text
ffprobe -v error -select_streams a:0 \
  -show_entries stream=duration:format=duration -of json AUDIO_PATH
```

Prefer a positive stream duration, then positive format duration. Run this validation outside all provider timing windows.

Compute the manifest content hash from canonical parsed records, including declared duration and audio SHA-256 but excluding the absolute dataset root and the separately measured duration. Store measured duration and the ffprobe version in run metadata; the audio content hash already binds the measured file without making the manifest hash depend on decoder floating-point output.

Choose deterministic order by sorting on `sha256(f"{seed}\0{sample_id}")`; the first selected sample is the shared cold probe.

Use exact validation constants:

```python
BCP47_BASIC_V1 = re.compile(r"[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*")
STABLE_ID_V1 = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
MAX_TAGS_PER_SAMPLE = 32
```

Require `STABLE_ID_V1.fullmatch` for sample, suite, annotation-profile, profile, and tag identifiers. Require unique tags and source keys `dataset`, `version`, `license`, and `reference_provenance`.

- [ ] **Step 4: Add `validate` argparse subcommand**

`main(argv: Sequence[str] | None = None) -> int` prints counts by profile/suite/visibility and the manifest hash. Errors include sample ID and field, never reference text.

- [ ] **Step 5: Re-run tests and a CLI smoke test**

Use monkeypatched duration probes in unit tests. Add one integration-marked ffprobe test that skips only when `shutil.which("ffprobe")` is absent.

- [ ] **Step 6: Run Bandit/diff and commit**

```bash
source .venv/bin/activate
python -m bandit -r Helper_Scripts/benchmarks/stt_bench.py \
  -f json -o /tmp/bandit_stt_bench_manifest.json
git diff --check
git commit -m "feat: validate STT benchmark manifests"
```

Expected: no new Bandit findings, no whitespace errors, and the commit succeeds after staging only Task 5 paths.

---

## Slice 6: Durable Result History and Pure Summary Inputs

### Task 6: Add Owner-Only Artifacts, Fsync JSONL, Resume Reduction, and Suite Aggregates

**Files:**

- Modify: `Helper_Scripts/benchmarks/stt_bench.py`
- Modify: `tldw_Server_API/tests/Benchmarks/test_stt_bench.py`

- [ ] **Step 1: Write failing persistence and aggregation tests**

Cover:

- `0700` run directories and `0600` files where supported;
- atomic `run.json`/`inflight.json` replacement;
- result flush plus `os.fsync` before the next acknowledgement;
- completion-key stability;
- duplicate/non-monotonic attempt-ID rejection;
- invalid in-flight discriminators and operation/result ID combinations;
- coordinator-owned globally monotonic result attempt IDs survive resume;
- highest-attempt reduction;
- ordinary resume skips active successes and failures;
- `--retry-errors` allocates exactly one higher attempt;
- stale in-flight cleanup when its terminal result already exists;
- sample-attributed crash/interrupted/timeout record otherwise;
- truncated final JSONL line is reported and ignored; earlier malformed lines fail;
- status failures score as empty hypotheses;
- `diagnostic_only` exclusion from primary metrics;
- per-suite pooled/mean/percentile/exact/empty/failure/success metrics;
- dataset/tag and actual-backend slices;
- public/private suites never pooled by default;
- one cold probe excluded from warm aggregates;
- `warmup_recovery` and replayed probe excluded from warm aggregates;
- RTF=`processing/audio`, throughput=`audio/processing`.
- non-finite or non-positive processing/audio durations produce unavailable
  performance fields and are never performance-gate eligible.

Define:

```python
RUN_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
SUMMARY_SCHEMA_VERSION = 1
RESULT_STATUSES = frozenset({
    "ok",
    "empty",
    "adapter_error",
    "timeout",
    "worker_crash",
    "interrupted",
    "invalid_artifact",
})
MEASUREMENT_ROLES = frozenset({"accuracy", "performance_repeat"})
TIMING_CLASSES = frozenset({"cold_first", "warmup_recovery", "warm"})
```

Use these exact public signatures:

```text
completion_key(
    manifest_hash: str,
    target_id: str,
    execution_contract_hash: str,
    sample_id: str,
    repetition: int,
) -> str
atomic_write_json(path: Path, payload: Mapping[str, object]) -> None
append_result_record(path: Path, record: Mapping[str, object]) -> None
load_result_history(path: Path) -> tuple[list[dict[str, object]], bool]
reduce_attempts(
    records: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]
aggregate_results(
    run_metadata: Mapping[str, object],
    active_results: Mapping[str, Mapping[str, object]],
) -> dict[str, object]
```

- [ ] **Step 2: Verify red**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Benchmarks/test_stt_bench.py -k "persist or inflight or attempt or aggregate or retention" -q
```

Expected: FAIL because durable result history and reduction are absent.

- [ ] **Step 3: Implement durable writes**

Use same-directory temporary files, `flush()`, `os.fsync()`, `os.replace()`, and parent-directory fsync when supported. Never weaken correctness because a platform lacks POSIX mode bits; record the limitation.

`inflight.json` is a discriminated, allowlisted record containing
`target_id`, globally monotonic `operation_id`, `operation_role`,
`worker_attempt_id`, `sample_id`, optional `completion_key`, optional
`repetition`, and optional `result_attempt_id`.

- `operation_role="result_call"` requires completion key, repetition, and an
  integer result-attempt ID.
- `operation_role="rewarm_probe"` requires the prior probe completion key but
  requires `result_attempt_id=null`; it may not append a result record.

The coordinator owns both `next_operation_id` and `next_attempt_id` in
`run.json`. It increments the operation counter before acknowledging every
adapter call, but increments the result-attempt counter only for a
`result_call`. On recovery, a stale `result_call` without a terminal record
becomes a sample-attributed crash/interrupted result; a stale `rewarm_probe`
only updates that worker attempt's rewarm status. A replay-only cold probe
therefore cannot supersede a prior scored result.
A retry keeps the same completion key and receives a higher global result
attempt ID.

- [ ] **Step 4: Implement scoring and active-attempt reduction**

Score raw text before applying retention. For non-`ok` statuses, use empty hypothesis in the quality aggregate. Treat `empty` as an explicit non-success status and empty hypothesis.

Each record has two independent fields:

- `measurement_role`: `accuracy` or `performance_repeat`;
- `timing_class`: `cold_first`, `warmup_recovery`, or `warm`.

Accuracy aggregation includes only `measurement_role="accuracy"`. Warm performance aggregation includes both roles when `timing_class="warm"`; the accuracy call is the first of the requested warm repetitions for an ordinary non-probe sample. Replayed probe operations live only in worker-attempt metadata and never enter quality or performance JSONL records.

Bound errors with `sanitize_error(exc) -> {"type": str, "message": str}`:
retain only the exception class name; replace control/whitespace runs with one
space; redact case-insensitive `authorization`, `bearer`, `api_key`, `token`,
`secret`, and `sk-` patterns; redact complete URLs including userinfo/query;
redact absolute POSIX, Windows-drive, and UNC paths; and truncate the final
message to 512 Unicode code points. Add hostile tests containing credentials,
signed URLs, local home paths, model cache paths, and multiline control
characters.

Require every result record to carry: schema/run/target IDs; completion key; sample/repetition/attempt/worker-attempt IDs; measurement role and timing class; suite/dataset/tags/diagnostic flag; requested and allowlisted actual execution; mismatch/eligibility reasons; status; retained reference/hypothesis fields; strict and normalized edit-count dictionaries; adapter nanoseconds, measured audio duration, RTF, throughput; and optional allowlisted resource observations. Reject unknown status/role/timing values during report loading.

Compute RTF and throughput only from finite, strictly positive adapter and audio
durations. Otherwise store both as `null`, append a bounded eligibility reason,
and exclude the observation from percentiles and performance gates.

- [ ] **Step 5: Re-run tests**

Expected: PASS.

- [ ] **Step 6: Bandit/diff and commit**

```bash
source .venv/bin/activate
python -m bandit -r Helper_Scripts/benchmarks/stt_bench.py \
  -f json -o /tmp/bandit_stt_bench_persistence.json
git diff --check
git commit -m "feat: persist resumable STT benchmark results"
```

Expected: no new Bandit findings, no whitespace errors, and the commit succeeds after staging only Task 6 paths.

---

## Slice 7: Worker, Coordinator, Preflight, and `run`

### Task 7: Execute One Native Target per Spawned Process

**Files:**

- Modify: `Helper_Scripts/benchmarks/stt_bench.py`
- Modify: `tldw_Server_API/tests/Benchmarks/test_stt_bench.py`

- [ ] **Step 1: Write failing fake-adapter runner tests**

Use top-level pickleable fake adapter/factory functions in the test module and `multiprocessing.get_context("spawn")`.

Cover:

- strict registry lookup and all-target preflight before any worker starts;
- unknown provider and unavailable adapter fail closed;
- any route with `would_download=True`, unresolved network egress, or an
  unenforceable local artifact fails preflight before worker creation;
- duplicate normalized `provider=model` targets are rejected;
- loopback and remote plans both require `--allow-network-targets`;
- no target starts if any matrix preflight fails;
- safe descriptor verification in the worker;
- execution-contract mismatch refuses resume;
- config mutation after planning has no execution effect;
- stable shared cold probe;
- successful cold probe is scored once and excluded from warm metrics;
- resumed worker retranscribes the completed probe only to rewarm and does not add another scored record;
- replayed-probe in-flight state is marked `operation_role="rewarm_probe"`; a crash updates only worker-attempt metadata and cannot replace a prior scored probe success with a newer failure;
- failed cold probe causes the first later success to be `warmup_recovery`;
- later calls become `warm`;
- default one scored transcription per sample;
- additional timing repetitions are `performance_repeat`, not quality inputs;
- worker begin/ack handshake writes `inflight.json` before adapter entry;
- adapter-done acknowledgement disarms the adapter watchdog before scoring and
  fsync; deliberately delayed persistence is not mislabeled as adapter timeout;
- exceptions, sentinels, empty output, malformed artifact, hard `os._exit`, watchdog termination, and parent interrupt have the correct statuses;
- worker attempts are recorded and targets run sequentially in CLI order;
- parent-measured spawn-to-ready time, worker registry/adapter setup time, and total target wall time are recorded separately;
- deterministic seed/order;
- offline-library environment controls are applied only inside the worker;
- no prompt/hotword/credential/URL leaks through malicious fake metadata.
- an absolute local model path is retained only in secret in-memory plan
  settings and never appears in `PreparedTarget` repr, `run.json`, result
  records, summaries, or sanitized failures.

- [ ] **Step 2: Verify red**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Benchmarks/test_stt_bench.py -k "worker or runner or resume or preflight" -q
```

- [ ] **Step 3: Build immutable run/target metadata**

Define:

```python
@dataclass(frozen=True)
class PreparedTarget:
    target_id: str
    provider: str
    model_label: str
    plan: SttBatchExecutionPlan = field(repr=False)
    adapter_factory_path: str = field(repr=False)
    execution_contract_json: str
    execution_contract_hash: str

@dataclass(frozen=True)
class WorkerSettings:
    run_id: str
    results_path: str
    normalization_profile: str
    cold_probe_sample_id: str
    warm_repetitions: int
    timing_sample_ids: tuple[str, ...]
    text_retention: str
    retry_errors: bool
```

Store the contract as canonical JSON text so the frozen/pickleable target does not contain mutable dictionaries. Parse it only when writing the allowlisted contract to `run.json`.

Normalize the provider and derive `model_label` from the plan's validated safe
descriptor. Derive `target_id` as `target-` plus the first 16 lowercase hex
characters of SHA-256 over the canonical safe descriptor and target ordinal.
The raw `provider=model` input may exist only in the local preflight variable
and secret `plan.runtime_settings`; discard it after constructing
`PreparedTarget`. Persist the target matrix using only target ID, safe provider
and model labels, ordered route descriptors, and execution-contract hashes.
Validate `safe_target_settings` through an explicit allowlist before hashing.

Implement:

```text
build_execution_contract(
    *,
    plan: SttBatchExecutionPlan,
    git_commit: str,
    safe_target_settings: Mapping[str, object],
) -> tuple[str, str]

preflight_targets(
    target_specs: Sequence[str],
    *,
    mode: str,
    allow_network_targets: bool,
    common_settings: Mapping[str, object],
    adapter_factory_path: str = PRODUCTION_ADAPTER_FACTORY_PATH,
) -> tuple[PreparedTarget, ...]

_load_native_adapter(provider: str) -> SttProviderAdapter
_resolve_adapter_factory(path: str) -> Callable[[str], SttProviderAdapter]
_worker_main(
    connection: Connection,
    prepared_target: PreparedTarget,
    samples: tuple[ManifestSample, ...],
    settings: WorkerSettings,
) -> None
```

Set:

```python
PRODUCTION_ADAPTER_FACTORY_PATH = (
    "Helper_Scripts.benchmarks.stt_bench:_load_native_adapter"
)
```

`_load_native_adapter` lazily imports and constructs the native registry, then
calls `get_adapter_strict`; `stt_bench.py` must not import the adapter/provider
modules at module import time. Use a quoted/`TYPE_CHECKING` adapter return
annotation so `validate`, scoring tests, and spawned module import stay light.
`_resolve_adapter_factory` accepts only the
`module:top_level_name` format and verifies a callable after `importlib`
resolution. The CLI always uses the production constant; the injectable
argument exists only for direct tests. Spawn tests pass the import path of a
top-level fake factory in the test module, so correctness never depends on a
parent-process monkeypatch surviving `spawn`.

Resolve source files and package versions from the plan descriptor's `source_modules` and `dependency_distributions`. The execution-contract hash includes the safe descriptor, selected benchmark/adapter/provider source content hashes, Git commit, relevant dependency versions, scorer/Unicode versions, and safe target settings. Secret runtime settings remain only in the in-memory plan passed to the worker.

Reject a prepared plan before hashing when it has no routes, any route declares
`would_download=True`, a local route cannot prove its artifact exists, or a
network route lacks both explicit user consent and an opaque validated endpoint
ID. This is an availability result, not permission to repair/download.

Run-level resume identity also includes manifest hash, selected sample IDs, profile, mode, seed/order, repetition/timing subset, text retention, hardware profile, and target matrix.

Collect the environment fingerprint without importing models: Python and `unicodedata.unidata_version`; OS/release/architecture; logical/physical cores and RAM (`psutil` when installed); CPU model; requested device/compute; Git commit/dirty flag; selected dependency versions; ffprobe version; and optional GPU/Apple identity from an already-installed local system tool. Record each best-effort collection method and `unavailable` explicitly. Never serialize the environment mapping or command output wholesale.

- [ ] **Step 4: Implement the worker protocol**

Use a duplex `multiprocessing.Pipe`:

0. Parent records `perf_counter_ns()` immediately before `process.start()`; worker records registry/adapter setup around strict lookup and sends `ready`; parent derives `worker_spawn_to_ready_ns`.
1. Worker sends `begin` with the non-secret in-flight fields.
2. Coordinator allocates the operation/result IDs, atomically writes
   `inflight.json`, arms the adapter watchdog, then sends `begin_ack` with the
   allocated IDs.
3. Worker starts `perf_counter_ns()` immediately before `transcribe_batch(...)` and stops immediately after return/raise.
4. Worker sends `adapter_done` with only bounded status/timing metadata;
   coordinator disarms the adapter watchdog and sends `adapter_done_ack`.
5. Worker scores and appends/fsyncs the terminal result outside the adapter
   timing and watchdog windows.
6. Worker sends `committed`.
7. Coordinator verifies the terminal record, clears `inflight.json`, and sends `committed_ack`.

The parent records total target wall time from process start through exit. The cold probe's adapter duration becomes `cold_first_transcription_seconds`; every warm record retains adapter seconds, RTF, and throughput. Optionally sample process RSS before/after each call and peak RSS when the local platform exposes it, recording the collection method; GPU/VRAM observations remain best-effort and never gate across methods.

The watchdog is an adapter-call timer armed immediately before `begin_ack` and
cleared on `adapter_done`; it terminates the whole worker attempt when one
synchronous adapter call exceeds the configured interval and attributes
`timeout` to persisted in-flight state. A separate parent-side worker-liveness
check handles exit/disconnect during scoring or fsync as `worker_crash`, not
`timeout`. Do not claim reusable warm state after either termination.

For a replayed probe, write `operation_role="rewarm_probe"` and the worker-attempt ID to `inflight.json`, but allocate no result attempt ID. On success, failure, interrupt, or watchdog termination, atomically update that worker attempt's `rewarm_status` and optional duration in `run.json`; never append a second scored probe record. With `--retry-errors`, a previously failed probe instead receives a real higher accuracy attempt, and that same call establishes warm state if it succeeds.

Inside the worker, set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1` before importing provider modules. These are defense in depth; native loader controls remain mandatory.

End the module with:

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

This is required for both direct CLI execution and spawn-safe module import.

- [ ] **Step 5: Implement cold/warm and repetition policy**

Add CLI options:

```text
--seed INT                         default 0
--warm-repetitions INT             default 1
--timing-sample SAMPLE_ID          repeatable; default all non-probe samples
--worker-watchdog-seconds FLOAT    optional target-worker watchdog
--retry-errors
--allow-network-targets
--configuration-id ID
--network-collection-profile ID
--network-client-location LABEL
```

`--warm-repetitions` is the total attempted call count per selected non-probe
sample after the cold probe. Repetition zero is the accuracy call and also the
first warm measurement only when its resulting `timing_class` is `warm`;
higher repetitions are performance-only. A failed probe or recovery can
therefore yield fewer warm observations than requested. Default `1` produces
one accuracy attempt and no extra call. Require `--warm-repetitions >= 3` plus
three successfully matched warm observations per gated suite/target before a
performance gate can be eligible.

Require `--configuration-id` for `production-v1`. Include the opaque ID plus prompt presence and hotword count in safe target settings; never include prompt/hotword values or their digests. Reject `--configuration-id` in `neutral-v1` so it cannot imply hidden tuning.

- [ ] **Step 6: Add `run` command and resume semantics**

Parse repeatable `--target provider=model` by the first `=` only. Reject missing/empty sides. Create `.benchmarks/stt/YYYYMMDDTHHMMSSZ-SHORT_HASH/` by default or resume an explicit `--run`.

Warn when `text-retention=full` and any selected suite is private. Refuse a non-empty incompatible run directory.

- [ ] **Step 7: Re-run tests**

Expected: PASS on fake adapters; no real model/network access.

- [ ] **Step 8: Bandit/diff and commit**

```bash
source .venv/bin/activate
python -m bandit -r Helper_Scripts/benchmarks/stt_bench.py \
  -f json -o /tmp/bandit_stt_bench_runner.json
git diff --check
git commit -m "feat: run isolated native STT benchmarks"
```

Expected: no new Bandit findings, no whitespace errors, and the commit succeeds after staging only Task 7 paths.

---

## Slice 8: Reports, Comparisons, Gates, and Retention

### Task 8: Generate Disposable Reports and Enforce Only Compatible Baselines

**Files:**

- Modify: `Helper_Scripts/benchmarks/stt_bench.py`
- Modify: `tldw_Server_API/tests/Benchmarks/test_stt_bench.py`

- [ ] **Step 1: Write failing report/compare tests**

Cover:

- unsupported run/result/summary schema rejection;
- report regeneration from `run.json` + `results.jsonl`;
- partial-run summaries retain pending counts and never treat missing records as successes;
- summary JSON/Markdown/terminal parity;
- worst samples only when retained text permits it;
- `full`, `errors-only`, and `none` retention;
- scorer/manifest/profile/suite/sample/mode/repetition compatibility;
- exact `unicodedata.unidata_version` compatibility;
- descriptive cross-target comparison permits provider/model differences;
- same-target gates require resolved identity, matching actual backend/compute, safe settings, hardware, and collection method;
- implementation/dependency versions may differ and remain visible;
- hardware mismatch rejects performance gates but allows descriptive quality comparison;
- remote performance is informational unless explicitly opted in and matching network collection profiles;
- production mixed-backend results are split and are not model-quality/performance-gate eligible;
- per-suite absolute/relative pooled normalized WER/CER and failure-rate limits;
- optional exact-match minimum;
- strict metrics remain diagnostic by default;
- paired per-sample deltas are emitted and rankings say `descriptive`.
- target execution order and collection method must match for performance-gate eligibility.

- [ ] **Step 2: Verify red**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Benchmarks/test_stt_bench.py -k "report or compare or policy or compatibility" -q
```

Expected: FAIL because disposable reports and comparison policy are absent.

- [ ] **Step 3: Implement retention before serialization**

```text
retain_text(
    *,
    mode: str,
    status: str,
    reference: str,
    hypothesis: str,
    score: TranscriptScore,
) -> tuple[str | None, str | None]
```

`errors-only` retains text for any non-zero strict or normalized edit count or non-`ok` status. Summaries never reload discarded text from the manifest.

- [ ] **Step 4: Add `report`**

Write `summary.json` atomically with `SUMMARY_SCHEMA_VERSION`; render `summary.md` from that exact dictionary. Report:

- per-suite strict/normalized pooled and mean WER/CER;
- p50/p90/p95/p99;
- exact/success/empty/failure rates;
- dataset/tag/diagnostic slices;
- target-level cold observation;
- per-suite warm latency/RTF/throughput median and IQR;
- actual backend split;
- eligibility/rejection reasons;
- environment/configuration fingerprints;
- worst retained examples.

`summary.json` also keeps the active per-sample IDs, edit counts/rates, statuses, timing classes, and durations required for paired deltas. It applies the run's text-retention policy and never embeds audio paths or discarded text.

- [ ] **Step 5: Add an explicit regression-policy input**

The approved design says baselines contain bounded expectations but does not define where those values come from. Clarify it with optional:

```text
compare --policy /path/to/stt-baseline-policy.json
```

Without `--policy`, comparison is descriptive and never fails on metric movement. A versioned policy has a `suites` object keyed by suite ID with allowlisted fields:

```json
{
  "schema_version": 1,
  "suites": {
    "public-english-v1": {
      "max_normalized_pooled_wer_absolute_regression": 0.01,
      "max_failure_rate_absolute_regression": 0.0,
      "min_exact_match_rate": 0.5,
      "max_warm_rtf_relative_regression": 0.1
    }
  }
}
```

Reject unknown fields, negative bounds, and suites absent from either summary.
If a baseline metric is zero, a relative-regression rule for that metric is ineligible; require an absolute rule instead. Do not divide by zero or silently reinterpret the policy.

- [ ] **Step 6: Add `compare` compatibility and exit codes**

Exit:

- `0`: compatible descriptive comparison or all eligible gates pass;
- `1`: an eligible gate fails;
- `2`: invalid/incompatible artifacts or requested ineligible gate.

Add `--allow-network-performance-gates`; require matching non-secret network profiles and at least three warm repetitions.

- [ ] **Step 7: Re-run tests**

Expected: PASS.

- [ ] **Step 8: Bandit/diff and commit**

```bash
source .venv/bin/activate
python -m bandit -r Helper_Scripts/benchmarks/stt_bench.py \
  -f json -o /tmp/bandit_stt_bench_report.json
git diff --check
git commit -m "feat: report and compare STT benchmark runs"
```

Expected: no new Bandit findings, no whitespace errors, and the commit succeeds after staging only Task 8 paths.

---

## Slice 9: Golden Integration, Examples, Documentation, and Release Gates

### Task 9: Reuse the Scorer and Manifest in Opt-In Golden Tests

**Files:**

- Modify: `tldw_Server_API/tests/Audio/test_stt_adapters_golden.py`
- Modify: `Helper_Scripts/Audio/generate_stt_golden.py`
- Modify: `tldw_Server_API/tests/Helper_Scripts/test_generate_stt_golden.py`
- Modify: `Docs/Development/STT_Adapter_Golden_Tests.md`

- [ ] **Step 1: Write failing golden-helper tests**

Test that:

- generated adapter text defaults to `reference_status="unverified_candidate"`;
- an independently supplied reference requires `--reference` plus `--reference-provenance` of `canonical-dataset` or `human-reviewed`;
- an unverified candidate cannot be emitted as a scored manifest record;
- no provider allowlist duplicates `SttProviderRegistry`;
- `TLDW_STT_GOLDEN_TARGETS` is parsed as a JSON array of `provider=model` strings, avoiding delimiter ambiguity;
- loopback/remote golden targets require `TLDW_STT_GOLDEN_ALLOW_NETWORK=1`;
- the golden test imports `score_transcript` and `load_and_validate_manifest`;
- artifact/segment contract assertions remain.

- [ ] **Step 2: Verify red**

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Helper_Scripts/test_generate_stt_golden.py \
  tldw_Server_API/tests/Audio/test_stt_adapters_golden.py \
  -q
```

The real-model cases should skip unless the existing opt-in environment is enabled.

- [ ] **Step 3: Migrate the golden flow**

Remove `_normalize_text`, `_levenshtein`, and `_token_error_rate`. Load the regression profile from `TLDW_STT_GOLDEN_MANIFEST` plus `TLDW_STT_GOLDEN_AUDIO_DIR`. Require `TLDW_STT_GOLDEN_TARGETS` as a JSON array of `provider=model` strings; resolve each target strictly, create/approve its execution plan, and score with `score_transcript`. Use `TLDW_STT_GOLDEN_MAX_NORMALIZED_WER` (default `0.20`) for the existing per-sample assertion and keep segment-shape assertions. Dynamic network plans also require the separate truthy `TLDW_STT_GOLDEN_ALLOW_NETWORK`.

Keep the golden marker and explicit opt-in. Treat adapter-generated output as a candidate/snapshot only.

- [ ] **Step 4: Re-run tests, Bandit, and commit**

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Helper_Scripts/test_generate_stt_golden.py \
  tldw_Server_API/tests/Audio/test_stt_adapters_golden.py \
  -q
python -m bandit -r \
  Helper_Scripts/Audio/generate_stt_golden.py \
  -f json -o /tmp/bandit_stt_bench_golden.json
git diff --check
git commit -m "test: align STT golden tests with benchmark scoring"
```

Expected: helper tests pass, real-model cases skip unless opted in, Bandit has no new findings, and the commit succeeds after staging only Task 9 paths.

### Task 10: Add the Protocol, Example Manifest, Ignore Rule, and CLI Documentation

**Files:**

- Create: `Helper_Scripts/benchmarks/stt_benchmark_manifest.example.jsonl`
- Create: `Docs/Development/STT_Benchmark_Protocol.md`
- Modify: `Helper_Scripts/benchmarks/README.md`
- Modify: `.gitignore`
- Test: `tldw_Server_API/tests/Benchmarks/test_stt_bench.py`

- [ ] **Step 1: Add a failing example-manifest parse test**

The example must parse as JSONL schema documentation but may point to non-existent illustrative media; validation tests replace the audio/checksum fields in a temp directory. Do not commit corpus audio without explicit license/provenance.

- [ ] **Step 2: Document the exact protocol**

Include:

- no Pipecat/no LLM judge/no model download;
- `strict-v1`, `en-v1`, scorer version, Unicode version;
- `bcp47-basic-v1`;
- regression vs comparison profiles;
- public/private suites and `diagnostic_only`;
- challenge-pack annotation rules for orthography, casing, punctuation, fillers, false starts, partial words, unintelligible/non-speech events, numerals, abbreviations, proper nouns, review, and adjudication;
- manual LibriSpeech/Common Voice acquisition and checksum pinning;
- four CLI commands and target syntax;
- network egress consent and redirect policy;
- cold-first vs warm-adapter definitions;
- RTF/throughput formulas and performance repetition minimum;
- artifact layout, resume/retry behavior, retention/privacy warnings, owner-only permissions;
- comparison eligibility and descriptive-ranking language.

- [ ] **Step 3: Add `.benchmarks/stt/` to `.gitignore`**

Do not ignore the example manifest or documentation.

- [ ] **Step 4: Run doc/example tests and commit**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Benchmarks/test_stt_bench.py -k "example_manifest or cli_help" -q
git diff --check
git commit -m "docs: add native STT benchmark protocol"
```

Expected: PASS, no whitespace errors, and the commit succeeds after staging only Task 10 paths.

### Task 11: Run the Full Verification Matrix and Finalize Tracking

**Files:**

- Modify: slice Backlog tasks
- Remove after all implementation slices are complete: `Docs/superpowers/plans/2026-07-23-native-batch-stt-benchmark-implementation-plan.md`

- [ ] **Step 1: Run focused benchmark and adapter suites**

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Benchmarks/test_stt_bench.py \
  tldw_Server_API/tests/Audio/test_stt_provider_adapter.py \
  tldw_Server_API/tests/Audio/test_stt_execution_plan_local.py \
  tldw_Server_API/tests/Audio/test_stt_execution_plan_network.py \
  tldw_Server_API/tests/Audio/test_qwen3_asr.py \
  tldw_Server_API/tests/Audio/test_vibevoice_transcription.py \
  tldw_Server_API/tests/Audio/test_parakeet_onnx_failfast.py \
  tldw_Server_API/tests/Audio/test_audio_transcriptions_adapter_path.py \
  tldw_Server_API/tests/Audio/test_audio_transcriptions_hotwords.py \
  tldw_Server_API/tests/Audio/test_audio_transcriptions_timed_segments.py \
  tldw_Server_API/tests/STT/test_audio_transcription_api.py \
  tldw_Server_API/tests/STT/test_audio_transcription_translate_task.py \
  tldw_Server_API/tests/TTS_NEW/integration/test_transcription_auth.py \
  tldw_Server_API/tests/Helper_Scripts/test_generate_stt_golden.py \
  -q
```

Expected: PASS; real models/network are not contacted.

- [ ] **Step 2: Run the broader STT regression scope**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audio -q
```

Expected: PASS or documented pre-existing unrelated failures. Do not silently waive new failures.

- [ ] **Step 3: Exercise the CLI with generated fake fixtures**

Run `validate`, fake-adapter `run`, `report`, and `compare` through tests or a temp-directory smoke script. Confirm an interrupt/resume, a worker crash, and a retention-none report without storing transcript text.

- [ ] **Step 4: Run final security/static gates**

```bash
source .venv/bin/activate
python -m compileall -q \
  Helper_Scripts/benchmarks/stt_bench.py \
  Helper_Scripts/Audio/generate_stt_golden.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio
python -m bandit -r \
  Helper_Scripts/benchmarks/stt_bench.py \
  Helper_Scripts/Audio/generate_stt_golden.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_execution_contract.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_provider_adapter.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Lib.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Nemo.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Parakeet_ONNX.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Parakeet_MLX.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Qwen3ASR.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_VibeVoice.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_External_Provider.py \
  -f json -o /tmp/bandit_native_stt_benchmark.json
git diff --check
```

Expected: compilation succeeds, Bandit reports no new findings, and there are no whitespace errors.

- [ ] **Step 5: Verify every acceptance criterion**

Map the approved design's twelve criteria to passing tests/artifacts:

1. two fake native targets complete without Pipecat/LLM;
2. strict/normalized counts reproduce;
3. resume/cold-probe rules hold;
4. cold/warm metrics use documented formulas;
5. suite/slice/failure reports are distinct;
6. unknown targets never hit registry fallback;
7. compatibility rejects hardware-mismatched performance gates;
8. golden tests import the scorer;
9. dynamic egress/no-download/exact-plan rules hold;
10. sentinel/crash/watchdog attribution holds;
11. contract mismatch/unresolved identity/allowlist rules hold;
12. apostrophes/Unicode/annotation protocol hold.

- [ ] **Step 6: Finalize Backlog tasks**

Record exact test counts, Bandit result path, known skips, touched files, final summary, and commit IDs. Mark acceptance criteria and Definition of Done before setting each task to Done.

- [ ] **Step 7: Remove this execution plan only after all slices are complete**

The repository guide requires removing the task-specific implementation plan after its stages are complete. Preserve the approved design spec and Backlog history.

- [ ] **Step 8: Commit final verification/tracking cleanup**

```bash
git commit -m "chore: finalize native STT benchmark rollout"
```

## Out of Scope for This Plan

- Streaming/VAD/TTFS or diarization accuracy.
- Automatic corpus/model download.
- Parallel/distributed provider execution.
- Paired bootstrap confidence intervals.
- Published leaderboard generation.
- Web/API/Evaluations integration.
- Provider lifecycle hooks beyond the existing lazy first call.

Implement those only through a new design amendment and Backlog task.
