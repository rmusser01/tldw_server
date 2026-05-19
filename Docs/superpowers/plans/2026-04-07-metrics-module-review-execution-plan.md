# Metrics Module Review Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved Metrics module audit and deliver one consolidated, evidence-backed review covering correctness, lifecycle, export, operational, test-gap, and documentation-drift issues across the scoped Metrics surface.

**Architecture:** This is a read-first, risk-first review plan. Execution starts by locking the current worktree baseline and report contract, then inspects registry and export semantics, then instrumentation and lifecycle behavior, then endpoint, wiring, and scrape-time operational behavior, and only after that runs focused pytest slices to confirm or weaken candidate findings. No repository source changes are part of execution; the deliverable is the final in-session review output plus a prioritized fix plan.

**Tech Stack:** Python 3, pytest, git, rg, sed, find, Markdown

---

## Scope Lock

Keep these decisions fixed during execution:

- review the current working tree by default, not only `HEAD`
- label any finding that depends on uncommitted local changes
- keep the code scope centered on `tldw_Server_API/app/core/Metrics`, `tldw_Server_API/app/api/v1/endpoints/metrics.py`, and metrics-related wiring in `tldw_Server_API/app/main.py`
- treat `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py` as in-scope only for scrape-time import and gauge-refresh behavior reached from the metrics endpoint
- treat `tldw_Server_API/app/core/Chat/chat_metrics.py` as a direct-edge dependency, not as a separate chat-system review
- prioritize correctness, operational safety, and export integrity over broad observability redesign or stylistic cleanup
- separate `Confirmed finding`, `Probable risk`, and `Improvement` in working notes, even if the final report groups them under severity
- do not modify repository source files during the review itself
- do not run blanket repo-wide or broad module-wide suites; use the smallest targeted verification slices that answer a concrete question
- report documentation drift explicitly, but do not treat stale docs alone as a code defect without code evidence
- keep blind spots explicit instead of implying unreviewed areas are safe

## Review File Map

**No repository source files should be modified during execution.**

**Spec and plan inputs:**
- `Docs/superpowers/specs/2026-04-07-metrics-module-review-design.md`
- `Docs/superpowers/plans/2026-04-07-metrics-module-review-execution-plan.md`

**Primary implementation files to inspect first:**
- `tldw_Server_API/app/core/Metrics/README.md`
- `tldw_Server_API/app/core/Metrics/__init__.py`
- `tldw_Server_API/app/core/Metrics/metrics_manager.py`
- `tldw_Server_API/app/core/Metrics/telemetry.py`
- `tldw_Server_API/app/core/Metrics/decorators.py`
- `tldw_Server_API/app/core/Metrics/traces.py`
- `tldw_Server_API/app/core/Metrics/http_middleware.py`
- `tldw_Server_API/app/core/Metrics/logger_config.py`
- `tldw_Server_API/app/core/Metrics/metrics_logger.py`
- `tldw_Server_API/app/core/Metrics/stt_metrics.py`
- `tldw_Server_API/app/api/v1/endpoints/metrics.py`
- `tldw_Server_API/app/main.py`

**Direct-edge and documentation files to inspect when the active trace requires them:**
- `tldw_Server_API/app/core/Chat/chat_metrics.py`
- `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py`
- `Docs/Design/Metrics.md`
- any Metrics-specific docs directly linked from `tldw_Server_API/app/core/Metrics/README.md`

**High-value tests to inspect and selectively run:**
- `tldw_Server_API/tests/Metrics/test_singleton_thread_safety.py`
- `tldw_Server_API/tests/Metrics/test_metrics_cumulative_series_cap.py`
- `tldw_Server_API/tests/Metrics/test_metrics_logger_registry_bridge.py`
- `tldw_Server_API/tests/Metrics/test_metrics_logger_timestamps.py`
- `tldw_Server_API/tests/Metrics/test_telemetry_config_and_disable_paths.py`
- `tldw_Server_API/tests/Metrics/test_telemetry_import_fallback.py`
- `tldw_Server_API/tests/Metrics/test_tracing_dummy_span.py`
- `tldw_Server_API/tests/Metrics/test_telemetry_trace_context.py`
- `tldw_Server_API/tests/Metrics/test_chat_metrics_reset_safety.py`
- `tldw_Server_API/tests/Monitoring/test_metrics_endpoints.py`
- `tldw_Server_API/tests/Monitoring/test_metrics_autoregistration.py`
- `tldw_Server_API/tests/Monitoring/test_metrics_decorator_exports.py`

**Nearby tests to inspect only if a concrete claim requires them:**
- `tldw_Server_API/tests/AuthNZ/integration/test_monitoring_metrics_summary.py`
- `tldw_Server_API/tests/Embeddings/test_metrics_golden_contract.py`
- `tldw_Server_API/tests/Embeddings/test_embeddings_metrics_endpoint.py`
- `tldw_Server_API/tests/Chat/unit/test_chat_metrics_integration.py`

**Scratch artifacts allowed during execution:**
- `/tmp/metrics_review_notes.md`
- `/tmp/metrics_registry_pytest.log`
- `/tmp/metrics_lifecycle_pytest.log`
- `/tmp/metrics_surface_pytest.log`

## Stage Overview

## Stage 1: Baseline and Report Contract
**Goal:** Lock the worktree baseline, exact scope, and final report structure before deep reading starts.
**Success Criteria:** The review boundary, hotspot order, test inventory, and final output template are fixed before candidate findings are recorded.
**Tests:** No pytest execution in this stage.
**Status:** Not Started

## Stage 2: Registry and Export Semantics Pass
**Goal:** Inspect registry behavior first, including metric registration, normalization, accumulation, export shape, reset semantics, and registry-bridging helpers.
**Success Criteria:** Candidate findings about data loss, duplication, label drift, histogram behavior, or export/reset inconsistencies are captured with exact file references and evidence notes.
**Tests:** Read registry- and export-focused tests after the static pass; defer execution to Stage 5.
**Status:** Not Started

## Stage 3: Instrumentation, Lifecycle, and Fallback Pass
**Goal:** Inspect telemetry initialization, singleton lifecycle, decorators, tracing helpers, HTTP middleware, and metrics logging integration.
**Success Criteria:** Candidate findings about import-time side effects, unsafe fallbacks, lifecycle races, swallowed exceptions, or instrumentation distortions are recorded with clear confidence levels.
**Tests:** Read lifecycle- and tracing-focused tests after the static pass; defer execution to Stage 5.
**Status:** Not Started

## Stage 4: Surface, Wiring, and Direct-Edge Operational Pass
**Goal:** Inspect endpoint behavior, route wiring, scrape-time dependencies, operator-facing docs, and direct-edge state tracking.
**Success Criteria:** Endpoint/export mismatches, scrape-time side effects, auth/reset surprises, docs drift, and route-surface inconsistencies are traced end to end with evidence.
**Tests:** Read endpoint- and monitoring-focused tests after the static pass; defer execution to Stage 5.
**Status:** Not Started

## Stage 5: Targeted Verification and Final Synthesis
**Goal:** Run the selected pytest slices needed to validate or weaken candidate findings and produce the final remediation-oriented review.
**Success Criteria:** Every major claim in the final report is tied to code inspection, test inspection, executed verification, or an explicitly labeled open question.
**Tests:** Only the targeted pytest slices named in this plan plus any directly adjacent test needed to settle a disputed invariant.
**Status:** Not Started

### Task 1: Lock the Baseline and Final Output Contract

**Files:**
- Create: none
- Modify: none
- Inspect: `Docs/superpowers/specs/2026-04-07-metrics-module-review-design.md`
- Inspect: `Docs/superpowers/plans/2026-04-07-metrics-module-review-execution-plan.md`
- Inspect: `tldw_Server_API/app/core/Metrics`
- Inspect: `tldw_Server_API/tests/Metrics`
- Inspect: `tldw_Server_API/tests/Monitoring`
- Test: none

- [ ] **Step 1: Capture the dirty-worktree baseline**

Run:
```bash
git status --short
```

Expected: a list of uncommitted files, including whether scoped Metrics files already differ from committed history.

- [ ] **Step 2: Record the commit baseline used for the review**

Run:
```bash
git rev-parse --short HEAD
```

Expected: one short commit hash to cite when a finding depends on committed behavior rather than only local edits.

- [ ] **Step 3: Enumerate the exact Metrics review surface**

Run:
```bash
find tldw_Server_API/app/core/Metrics -maxdepth 1 -type f | sort
rg --files tldw_Server_API/app/api/v1/endpoints | rg '/metrics\\.py$|embeddings_v5_production_enhanced\\.py$'
```

Expected: a concrete file inventory that anchors the review and prevents accidental scope creep.

- [ ] **Step 4: Enumerate the primary test surface before reading deeply**

Run:
```bash
rg --files tldw_Server_API/tests/Metrics tldw_Server_API/tests/Monitoring | sort
```

Expected: the dedicated Metrics and Monitoring test inventory that will anchor later verification choices.

- [ ] **Step 5: Fix the final response contract before recording findings**

Use this exact final structure:
```markdown
## Findings
- severity-ordered findings with confidence, type, impact, evidence, and file references

## Open Questions / Assumptions
- only unresolved items that materially affect confidence

## Fix Plan
- immediate, near-term, and optional cleanup items with minimum-safe remediation shape and test implications

## Coverage / Docs Gaps
- blind spots, misleading tests, and doc drift that affected confidence

## Verification
- tests run, important files inspected, and what remains unverified
```

### Task 2: Execute the Registry and Export Semantics Pass

**Files:**
- Create: none
- Modify: none
- Inspect: `tldw_Server_API/app/core/Metrics/README.md`
- Inspect: `Docs/Design/Metrics.md`
- Inspect: `tldw_Server_API/app/core/Metrics/metrics_manager.py`
- Inspect: `tldw_Server_API/app/core/Metrics/metrics_logger.py`
- Inspect: `tldw_Server_API/app/core/Metrics/stt_metrics.py`
- Test: `tldw_Server_API/tests/Metrics/test_metrics_cumulative_series_cap.py`
- Test: `tldw_Server_API/tests/Metrics/test_metrics_logger_registry_bridge.py`
- Test: `tldw_Server_API/tests/Metrics/test_metrics_logger_timestamps.py`

- [ ] **Step 1: Read the declared Metrics docs before inspecting implementation**

Run:
```bash
sed -n '1,260p' tldw_Server_API/app/core/Metrics/README.md
sed -n '1,80p' Docs/Design/Metrics.md
```

Expected: the intended operator and contributor model for Metrics behavior, including any obvious docs drift to carry into later evidence notes.

- [ ] **Step 2: Build a landmark map for the registry file**

Run:
```bash
rg -n "class MetricsRegistry|def register_metric|def record\\(|def increment\\(|def observe\\(|def get_metric_stats|def get_all_metrics|def export_prometheus_format|def reset|def normalize_metric_name|def _normalize_labels" tldw_Server_API/app/core/Metrics/metrics_manager.py
```

Expected: the exact line anchors for registration, recording, normalization, export, and reset behavior.

- [ ] **Step 3: Read the registry in semantic slices**

Run:
```bash
sed -n '1,260p' tldw_Server_API/app/core/Metrics/metrics_manager.py
sed -n '1690,2255p' tldw_Server_API/app/core/Metrics/metrics_manager.py
```

Capture:
- duplicate-registration behavior
- label normalization and collision handling
- ring-buffer versus cumulative-series semantics
- histogram bucket and count handling
- Prometheus text export rules
- reset behavior and what state survives it

Expected: a candidate finding list for registry integrity and export correctness risks.

- [ ] **Step 4: Read the bridge helpers and STT-specific metric normalization**

Run:
```bash
rg -n "def _bridge_to_registry|def log_counter|def log_histogram|def log_gauge|def timeit|def log_resource_usage" tldw_Server_API/app/core/Metrics/metrics_logger.py
rg -n "def normalize_|def emit_|def observe_|iter_stt_metric_definitions" tldw_Server_API/app/core/Metrics/stt_metrics.py
sed -n '1,220p' tldw_Server_API/app/core/Metrics/metrics_logger.py
sed -n '233,499p' tldw_Server_API/app/core/Metrics/stt_metrics.py
```

Expected: evidence on whether helper layers preserve or distort the registry’s contract, especially around normalization, timestamping, and label cardinality control.

- [ ] **Step 5: Read the registry- and bridge-focused tests before any execution**

Run:
```bash
sed -n '1,240p' tldw_Server_API/tests/Metrics/test_metrics_cumulative_series_cap.py
sed -n '1,240p' tldw_Server_API/tests/Metrics/test_metrics_logger_registry_bridge.py
sed -n '1,240p' tldw_Server_API/tests/Metrics/test_metrics_logger_timestamps.py
```

Expected: a map of which export and registry invariants are already protected and which risky branches still appear untested.

### Task 3: Execute the Instrumentation, Lifecycle, and Fallback Pass

**Files:**
- Create: none
- Modify: none
- Inspect: `tldw_Server_API/app/core/Metrics/__init__.py`
- Inspect: `tldw_Server_API/app/core/Metrics/telemetry.py`
- Inspect: `tldw_Server_API/app/core/Metrics/decorators.py`
- Inspect: `tldw_Server_API/app/core/Metrics/traces.py`
- Inspect: `tldw_Server_API/app/core/Metrics/http_middleware.py`
- Inspect: `tldw_Server_API/app/core/Metrics/logger_config.py`
- Test: `tldw_Server_API/tests/Metrics/test_singleton_thread_safety.py`
- Test: `tldw_Server_API/tests/Metrics/test_telemetry_config_and_disable_paths.py`
- Test: `tldw_Server_API/tests/Metrics/test_telemetry_import_fallback.py`
- Test: `tldw_Server_API/tests/Metrics/test_tracing_dummy_span.py`
- Test: `tldw_Server_API/tests/Metrics/test_telemetry_trace_context.py`

- [ ] **Step 1: Build a landmark map for lifecycle and instrumentation code**

Run:
```bash
rg -n "class TelemetryConfig|class TelemetryManager|def _initialize|def _initialize_tracing|def _initialize_metrics|def shutdown|def get_telemetry_manager|def instrument_fastapi_app" tldw_Server_API/app/core/Metrics/telemetry.py
rg -n "def track_metrics|def measure_latency|def count_calls|def track_errors|def monitor_resource|def track_llm_usage|def cache_metrics" tldw_Server_API/app/core/Metrics/decorators.py
rg -n "class TracingManager|def trace_operation|def trace_method|def start_span|def start_async_span|def set_span_status|def record_exception" tldw_Server_API/app/core/Metrics/traces.py
rg -n "class HTTPMetricsMiddleware|async def dispatch" tldw_Server_API/app/core/Metrics/http_middleware.py
rg -n "load_and_log_configs|log_metrics_file|def setup_logger|serialize=True" tldw_Server_API/app/core/Metrics/logger_config.py
```

Expected: the exact line anchors for singleton creation, fallback behavior, decorators, tracing, middleware, and metrics-logging setup.

- [ ] **Step 2: Read the lifecycle and middleware files in execution order**

Run:
```bash
sed -n '1,240p' tldw_Server_API/app/core/Metrics/__init__.py
sed -n '1,260p' tldw_Server_API/app/core/Metrics/telemetry.py
sed -n '368,844p' tldw_Server_API/app/core/Metrics/telemetry.py
sed -n '1,320p' tldw_Server_API/app/core/Metrics/decorators.py
sed -n '1,260p' tldw_Server_API/app/core/Metrics/traces.py
sed -n '1,220p' tldw_Server_API/app/core/Metrics/http_middleware.py
sed -n '1,220p' tldw_Server_API/app/core/Metrics/logger_config.py
```

Capture:
- singleton/thread-safety assumptions
- import-time side effects
- optional dependency behavior when OTel pieces are missing
- exception swallowing that could hide instrumentation defects
- whether middleware and decorators preserve application semantics
- whether logger configuration loads or writes state in ways that matter to Metrics behavior

Expected: a candidate finding list for lifecycle, fallback, and instrumentation correctness risks.

- [ ] **Step 3: Read the lifecycle- and tracing-focused tests before running them**

Run:
```bash
sed -n '1,220p' tldw_Server_API/tests/Metrics/test_singleton_thread_safety.py
sed -n '1,260p' tldw_Server_API/tests/Metrics/test_telemetry_config_and_disable_paths.py
sed -n '1,220p' tldw_Server_API/tests/Metrics/test_telemetry_import_fallback.py
sed -n '1,220p' tldw_Server_API/tests/Metrics/test_tracing_dummy_span.py
sed -n '1,240p' tldw_Server_API/tests/Metrics/test_telemetry_trace_context.py
```

Expected: a map of which lifecycle and fallback guarantees are already enforced and which suspected gaps still require confirmation.

### Task 4: Execute the Surface, Wiring, and Direct-Edge Operational Pass

**Files:**
- Create: none
- Modify: none
- Inspect: `tldw_Server_API/app/api/v1/endpoints/metrics.py`
- Inspect: `tldw_Server_API/app/main.py`
- Inspect: `tldw_Server_API/app/core/Chat/chat_metrics.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py`
- Inspect: `Docs/Design/Metrics.md`
- Test: `tldw_Server_API/tests/Monitoring/test_metrics_endpoints.py`
- Test: `tldw_Server_API/tests/Monitoring/test_metrics_autoregistration.py`
- Test: `tldw_Server_API/tests/Monitoring/test_metrics_decorator_exports.py`
- Test: `tldw_Server_API/tests/Metrics/test_chat_metrics_reset_safety.py`

- [ ] **Step 1: Build an endpoint and route-wiring landmark map**

Run:
```bash
rg -n "@router|async def|get_prometheus_metrics|get_json_metrics|health_check_with_metrics|get_chat_metrics_endpoint|reset_metrics|require_roles" tldw_Server_API/app/api/v1/endpoints/metrics.py
rg -n "HTTPMetricsMiddleware|include_router\\(metrics_router|add_api_route\\(\"/metrics\"|add_api_route\\(f\"\\{API_V1_PREFIX\\}/metrics\"" tldw_Server_API/app/main.py
rg -n "_get_redis_client|embedding_stage_flag|paused|drain" tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py
rg -n "track_request|track_streaming|reset_active_metrics|get_active_metrics|active_requests|active_streams|active_transactions" tldw_Server_API/app/core/Chat/chat_metrics.py
```

Expected: the exact entrypoints and wiring lines for endpoint behavior, route exposure, scrape-time dependency behavior, and direct-edge active-state tracking.

- [ ] **Step 2: Read the endpoint, wiring, and direct-edge files in contract order**

Run:
```bash
sed -n '1,260p' tldw_Server_API/app/api/v1/endpoints/metrics.py
sed -n '6528,6537p' tldw_Server_API/app/main.py
sed -n '5738,5875p' tldw_Server_API/app/main.py
sed -n '7321,7330p' tldw_Server_API/app/main.py
sed -n '1,260p' tldw_Server_API/app/core/Chat/chat_metrics.py
sed -n '1,120p' Docs/Design/Metrics.md
```

Expected: evidence on route shape, auth/reset semantics, scrape-time side effects, health and JSON endpoint expectations, and whether docs match the actual monitoring surface.

- [ ] **Step 3: Read only the scrape-time code path from the embeddings endpoint**

Run:
```bash
sed -n '340,620p' tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py
```

Capture:
- what is imported at scrape time
- whether Redis access is triggered during metrics export
- whether failures are swallowed, logged, or surfaced

Expected: a bounded understanding of the metrics endpoint’s external dependency behavior without broadening into an embeddings review.

- [ ] **Step 4: Read the endpoint- and monitoring-focused tests before running them**

Run:
```bash
sed -n '1,240p' tldw_Server_API/tests/Monitoring/test_metrics_endpoints.py
sed -n '1,240p' tldw_Server_API/tests/Monitoring/test_metrics_autoregistration.py
sed -n '1,240p' tldw_Server_API/tests/Monitoring/test_metrics_decorator_exports.py
sed -n '1,220p' tldw_Server_API/tests/Metrics/test_chat_metrics_reset_safety.py
```

Expected: clarity on which endpoint, registration, and reset invariants are protected versus still untested.

- [ ] **Step 5: Write a discrepancy table before test execution**

Create `/tmp/metrics_review_notes.md` with these headings:
```markdown
# Metrics Review Notes

## Confirmed finding candidates

## Probable risks

## Improvements

## Docs or route discrepancies

## Tests to run and why
```

Expected: a stable working-notes structure that prevents mixing confirmed defects with lower-confidence concerns.

### Task 5: Run Targeted Verification and Produce the Final Review

**Files:**
- Create: `/tmp/metrics_registry_pytest.log`
- Create: `/tmp/metrics_lifecycle_pytest.log`
- Create: `/tmp/metrics_surface_pytest.log`
- Modify: `/tmp/metrics_review_notes.md`
- Inspect: all files already named in prior tasks as needed
- Test: `tldw_Server_API/tests/Metrics/test_metrics_cumulative_series_cap.py`
- Test: `tldw_Server_API/tests/Metrics/test_metrics_logger_registry_bridge.py`
- Test: `tldw_Server_API/tests/Metrics/test_metrics_logger_timestamps.py`
- Test: `tldw_Server_API/tests/Metrics/test_singleton_thread_safety.py`
- Test: `tldw_Server_API/tests/Metrics/test_telemetry_config_and_disable_paths.py`
- Test: `tldw_Server_API/tests/Metrics/test_telemetry_import_fallback.py`
- Test: `tldw_Server_API/tests/Metrics/test_tracing_dummy_span.py`
- Test: `tldw_Server_API/tests/Metrics/test_telemetry_trace_context.py`
- Test: `tldw_Server_API/tests/Metrics/test_chat_metrics_reset_safety.py`
- Test: `tldw_Server_API/tests/Monitoring/test_metrics_endpoints.py`
- Test: `tldw_Server_API/tests/Monitoring/test_metrics_autoregistration.py`
- Test: `tldw_Server_API/tests/Monitoring/test_metrics_decorator_exports.py`

- [ ] **Step 1: Run the registry and export verification slice**

Run:
```bash
source .venv/bin/activate && python -m pytest -v \
  tldw_Server_API/tests/Metrics/test_metrics_cumulative_series_cap.py \
  tldw_Server_API/tests/Metrics/test_metrics_logger_registry_bridge.py \
  tldw_Server_API/tests/Metrics/test_metrics_logger_timestamps.py | tee /tmp/metrics_registry_pytest.log
```

Expected: passing tests or concrete failures that either confirm registry/export concerns or narrow them to untested paths.

- [ ] **Step 2: Run the lifecycle and tracing verification slice**

Run:
```bash
source .venv/bin/activate && python -m pytest -v \
  tldw_Server_API/tests/Metrics/test_singleton_thread_safety.py \
  tldw_Server_API/tests/Metrics/test_telemetry_config_and_disable_paths.py \
  tldw_Server_API/tests/Metrics/test_telemetry_import_fallback.py \
  tldw_Server_API/tests/Metrics/test_tracing_dummy_span.py \
  tldw_Server_API/tests/Metrics/test_telemetry_trace_context.py | tee /tmp/metrics_lifecycle_pytest.log
```

Expected: evidence on whether lifecycle, fallback, and tracing contracts already have passing coverage or expose live regressions.

- [ ] **Step 3: Run the endpoint and monitoring verification slice**

Run:
```bash
source .venv/bin/activate && python -m pytest -v \
  tldw_Server_API/tests/Metrics/test_chat_metrics_reset_safety.py \
  tldw_Server_API/tests/Monitoring/test_metrics_endpoints.py \
  tldw_Server_API/tests/Monitoring/test_metrics_autoregistration.py \
  tldw_Server_API/tests/Monitoring/test_metrics_decorator_exports.py | tee /tmp/metrics_surface_pytest.log
```

Expected: evidence on endpoint, reset, and registration behavior without broadening into unrelated modules.

- [ ] **Step 4: Reconcile code-reading evidence against pytest evidence**

For each candidate finding, record:
- whether it is confirmed by code inspection alone
- whether pytest supports, weakens, or contradicts it
- whether the issue depends on current dirty-worktree state
- whether the gap is in implementation, tests, docs, or some combination

Expected: a final evidence map that prevents overstating ambiguous concerns.

- [ ] **Step 5: Deliver the final review using the locked response contract**

Requirements:
- findings first, ordered by severity
- every finding includes confidence, type, impact, reasoning, and file references
- fix plan is grouped into immediate, near-term, and optional cleanup
- explicitly mention tests run and anything not verified
- if no high-severity issues are found, state that directly instead of implying hidden blockers

Expected: one final review response that matches the approved spec and is ready for user action.
