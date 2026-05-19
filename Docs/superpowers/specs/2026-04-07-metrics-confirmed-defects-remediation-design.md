# Metrics Confirmed Defects Remediation Design

- Date: 2026-04-07
- Project: tldw_server
- Topic: Remediate confirmed Metrics audit findings only
- Mode: Design for implementation planning

## 1. Objective

Implement the confirmed Metrics defects from the completed audit without broadening into the lower-confidence risk items or a general observability redesign.

The remediation must:

- make every exposed Metrics endpoint truthful about the data it returns
- remove caller-visible semantic breakage in Metrics decorators and tracing helpers
- make telemetry lifecycle behavior restart-safe and cleanup-safe
- eliminate the confirmed registry/export/reset correctness mismatches
- add direct regression coverage for every corrected contract

## 2. Scope

### In Scope

- Metrics API and route wiring:
  - `tldw_Server_API/app/api/v1/endpoints/metrics.py`
  - `tldw_Server_API/app/main.py`
- Chat metrics data source and endpoint support:
  - `tldw_Server_API/app/core/Chat/chat_metrics.py`
- Metrics core behavior:
  - `tldw_Server_API/app/core/Metrics/telemetry.py`
  - `tldw_Server_API/app/core/Metrics/decorators.py`
  - `tldw_Server_API/app/core/Metrics/traces.py`
  - `tldw_Server_API/app/core/Metrics/metrics_manager.py`
  - `tldw_Server_API/app/core/Metrics/metrics_logger.py`
- Targeted scrape-path support code only where needed to preserve current endpoint guarantees:
  - bounded use of `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py`
- Targeted tests in:
  - `tldw_Server_API/tests/Metrics/`
  - `tldw_Server_API/tests/Monitoring/`
  - adjacent `tldw_Server_API/tests/Embeddings/` only if required to lock `/metrics` and `/api/v1/metrics/text` parity
  - `tldw_Server_API/tests/server_e2e_tests/test_llm_provider_workflow.py` as a direct dependent of `/api/v1/metrics/chat` contract if targeted verification needs one consumer-level assertion
- Behavior-tied docs only where the corrected contract would otherwise be misleading:
  - `tldw_Server_API/app/core/Metrics/README.md`
  - `tldw_Server_API/app/core/Chat/README.md` only if the chat metrics endpoint contract or data-source explanation changes materially

### Out of Scope

- speculative middleware timing/status risk fixes not already confirmed by the audit
- logger-config cleanup that is only stylistic or lower-confidence
- broad OpenTelemetry architecture redesign
- broad Metrics docs rewrite beyond behavior that changes in this remediation
- unrelated chat, embeddings, or monitoring refactors

## 3. Confirmed Issues To Fix

1. `/api/v1/metrics/chat` advertises chat metric series but reads from the JSON registry, while `ChatMetricsCollector` records those series through OpenTelemetry meters.
2. `/metrics` and `/api/v1/metrics/text` are separate exporters with different scrape-time behavior, so the public text metrics surface is inconsistent.
3. Telemetry lifecycle is not restart-safe; partial initialization can leave global OTel providers installed while the manager falls back to dummy components and becomes effectively unshuttable.
4. `TelemetryConfig.get_resource_attributes()` can reference OTel constants that do not exist on the import-fallback path.
5. `cache_metrics()` changes decorated function return semantics by treating any 2-tuple as `(result, cache_hit)` and returning only the first item.
6. `trace_method()` does not preserve async-method semantics, and `trace_operation()` records failures twice.
7. Registry cumulative-series cap behavior can make JSON/stat views and Prometheus export disagree about which series exist.
8. Label-key normalization can silently merge distinct logical labels into the same exported series.
9. `reset()` preserves definitions in a way that lets later bridge registration keep stale type/definition state.

## 4. Approved Behavior Changes

### 4.1 Chat Metrics Endpoint Contract

- `/api/v1/metrics/chat` must return chat metrics from the data source that actually owns them.
- The implementation should use a small in-process summary maintained by `ChatMetricsCollector` alongside OTel emission, not attempt to reconstruct endpoint data from exporter internals.
- The endpoint may continue to expose `active_operations` and `token_costs` exactly as it does now.
- The `metrics` object must become truthfully populated after chat metric emission.
- This remediation does not require a broad OTel-to-registry bridge for every subsystem; it only requires a truthful chat metrics endpoint contract.

### 4.2 Public Text Exporter Contract

- `/metrics` and `/api/v1/metrics/text` must share one exporter implementation.
- If embeddings/orchestrator metrics require scrape-time registration or refresh to satisfy current contract expectations, both routes must go through the same path.
- Public route paths remain unchanged.
- Cache-control, content-type, and response body behavior for the text exporter should be consistent across both routes for the same process state.

### 4.3 Telemetry Lifecycle Contract

- `initialize_telemetry()` and `shutdown_telemetry()` must be safe to call repeatedly.
- Shutdown must clear or reset global state enough to allow a clean re-initialization path.
- Partial initialization failure must not leave globally installed OTel providers attached while the manager reports an uninitialized dummy fallback state.
- `TelemetryConfig.get_resource_attributes()` must work safely when OTel imports are unavailable.

### 4.4 Decorator and Tracing Semantics

- `cache_metrics()` must preserve the original return contract of the decorated function.
- Cache-hit classification must stop inferring semantics from generic 2-tuples.
- Minimum-safe rule: only explicit cache metadata, such as a boolean `from_cache` attribute/property on the returned object, may mark a hit; otherwise the decorator must preserve the return value unchanged and avoid tuple reinterpretation.
- `trace_method()` must preserve coroutine-function semantics for async methods.
- `trace_operation()` must record one logical failure per failing operation.

### 4.5 Registry, Export, and Reset Semantics

- Capped cumulative-series behavior must not allow JSON/stat views to claim a series exists when Prometheus export will omit it.
- Overflow label-series must be rejected before mutating either cumulative state or point-sample/state views, so every public Metrics surface agrees on admitted series.
- Label normalization must no longer silently alias distinct logical label keys; ambiguous normalized collisions must be rejected explicitly rather than merged.
- Reset semantics must restore a fresh post-bootstrap registry state so bridge-driven re-registration does not inherit stale type/definition state.
- The remediation should keep the registry model intact rather than replacing it wholesale.

## 5. API and Internal Contract Design

### 5.1 `GET /api/v1/metrics/chat`

Recommended response rule:

- keep the existing top-level shape
- populate `metrics` from an explicit `ChatMetricsCollector` snapshot function backed by in-process summaries updated alongside OTel emission
- if a metric family is unavailable, omit only that family instead of returning a structurally correct but misleadingly empty snapshot for emitted data

Compatibility rule:

- existing consumers reading `active_operations` and `token_costs` should continue to work
- existing consumers reading registry-style metric summary fields, especially `sum`, should continue to work for the metric families currently exposed by this endpoint
- the key contract change is that emitted chat metrics actually appear in `metrics`

### 5.2 `GET /metrics` and `GET /api/v1/metrics/text`

- both routes call one shared helper
- helper is responsible for any registry export, `prometheus_client` merge, and bounded embeddings refresh needed by current contract
- route wrappers should be thin and should not drift in behavior
- targeted tests should be able to compare the two route outputs directly under the same prepared state

### 5.3 Telemetry Global State

- one canonical lifecycle path manages global singleton creation, cleanup, and restart
- cleanup must cover both local manager state and any globally installed providers that this module owns
- fallback initialization must leave the module in a state that is internally consistent and testable
- partial initialization failure must leave the manager in a shutdown-capable or fully rolled-back state rather than an uninitialized object with globally installed providers

### 5.4 Registry Semantics

- series admission and export must agree on what is accepted
- reset must rebuild registry state equivalent to a fresh bootstrap, including any required default registrations, so post-reset bridge registration behaves predictably
- normalization must preserve correctness first by rejecting ambiguous label-key combinations rather than silently merging them

## 6. File Responsibilities

- `metrics.py`
  - own the shared text export helper
  - switch `/api/v1/metrics/chat` to a truthful chat snapshot source
- `main.py`
  - route both public text exporters through the shared helper
  - preserve existing URLs while removing exporter drift
- `chat_metrics.py`
  - expose a stable snapshot method for the chat metrics families that the endpoint promises
  - maintain the small in-process summaries needed for that snapshot alongside OTel emission
- `telemetry.py`
  - implement idempotent init/shutdown/reinit behavior
  - fix import-fallback resource attribute behavior
- `decorators.py`
  - preserve return semantics for `cache_metrics()`
  - stop tuple-based cache-hit inference
- `traces.py`
  - preserve async-method semantics and avoid duplicate failure recording
- `metrics_manager.py`
  - align admitted-series behavior across stats and export
  - reject ambiguous normalized label collisions explicitly
  - make reset behavior reconstruct fresh registry state predictably
- `metrics_logger.py`
  - cooperate with the corrected reset/registration semantics without preserving stale definitions
- targeted tests
  - prove each corrected defect directly

## 7. Testing Strategy

Required regression coverage:

- `/api/v1/metrics/chat` returns populated chat metrics after chat metric emission
- `/api/v1/metrics/chat` preserves the current per-metric summary shape needed by existing consumers, especially `sum`
- adjacent consumer expectations for `/api/v1/metrics/chat` remain satisfied where targeted verification is warranted
- `/metrics` and `/api/v1/metrics/text` produce equivalent output for the shared tested conditions
- telemetry shutdown followed by re-initialization leaves a usable manager
- partial initialization failure does not strand global provider state
- `TelemetryConfig.get_resource_attributes()` works on import-fallback path
- `cache_metrics()` preserves the decorated function’s return value
- tuple returns remain untouched by `cache_metrics()` unless explicit cache metadata is present
- `trace_method()` preserves async semantics for coroutine methods
- `trace_operation()` does not double-record failures
- cumulative-series cap behavior stays internally consistent across stats and export
- ambiguous label-key normalization is rejected rather than silently aliased
- post-reset bridge registration does not retain stale definition/type behavior

Verification should prefer the smallest targeted pytest slices first, then one touched-scope verification pass.
This remediation should not require broad repo-wide monitoring or embeddings suites unless a changed contract explicitly forces that expansion.

## 8. Risk Management

Primary risks:

- breaking downstream expectations for metrics route payload shape
- unintentionally changing exporter contents beyond the confirmed defects
- overcorrecting registry normalization in a way that breaks existing legitimate metric writers
- test contamination from process-global telemetry and registry state

Mitigations:

- preserve route URLs and top-level response shapes wherever possible
- add regression tests before code changes for each defect cluster
- keep normalization/reset fixes explicit and narrowly scoped
- reset global/singleton state between targeted tests

## 9. Success Criteria

The remediation is successful when:

- `/api/v1/metrics/chat` truthfully exposes emitted chat metrics
- `/metrics` and `/api/v1/metrics/text` no longer diverge by implementation path
- telemetry can be shut down and re-initialized without inconsistent global state
- import-fallback telemetry config is safe
- `cache_metrics()`, `trace_method()`, and `trace_operation()` preserve correct caller-visible semantics
- registry/export/reset behavior no longer exhibits the confirmed internal contradictions
- each corrected behavior has direct regression coverage
