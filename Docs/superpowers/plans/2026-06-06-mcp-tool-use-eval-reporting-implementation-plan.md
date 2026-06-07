# MCP Tool-Use Evaluation Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build metadata-only MCP tool-use event capture, persistence, export, cleanup, and aggregate reporting for the standalone gateway and in-process protocol paths.

**Architecture:** Add a lightweight `mcp_unified.tool_use_reporting` package for event models, sanitization, recorder contracts, store contracts, SQLite persistence, and report aggregation. Wire the no-op/default recorder into `MCPRuntimeDependencies`, record protocol events around `tools/call`, and wrap `GatewayRuntime.call_tool` during gateway bootstrap so HTTP, WebSocket, stdio, profile bridge, and external runtime calls share one capture path.

**Tech Stack:** Python 3.10+, Pydantic, SQLAlchemy Core, asyncio `to_thread`, FastAPI gateway runtime contracts, pytest, Bandit.

---

## References

- Spec: `Docs/superpowers/specs/2026-06-06-mcp-tool-use-eval-reporting-design.md`
- Backlog: `backlog/tasks/task-2264 - Plan-MCP-tool-use-evaluation-reporting-implementation.md`
- Prior observability contract: `Docs/superpowers/plans/2026-06-04-mcp-tool-observability-contract-implementation-plan.md`

## Design Review Decisions

This plan incorporates the review pass before implementation:

1. Keep the event store separate from `SQLiteMCPStore` so profile/external registry schema migrations are not coupled to telemetry retention.
2. Keep package core imports lightweight. `mcp_unified.tool_use_reporting.models`, `recorder`, `builders`, and `reporting` must not import SQLAlchemy.
3. Do not re-export `SQLiteToolUseEventStore` from the package root; callers must import it from `mcp_unified.tool_use_reporting.sqlite` so no-op/default paths avoid SQLAlchemy imports.
4. Use JSON/JSONL CLI output first. The existing gateway CLI is machine-readable JSON, so Markdown/table output is deferred.
5. Treat `profile_id`, `mcp_profile_id`, `gateway_profile_id`, `mode_id`, `mcp_mode_id`, `model_id`, and `mcp_model_id` as candidate safe context keys only after sanitizer allowlists pass. Do not persist user, client, or session ids by default.
6. Add a double-counting context marker before delegating from the gateway wrapper, and make the protocol recorder skip when it sees that marker.
7. Record idempotency cache hits outside the idempotency execution wrapper so replays produce `execution_origin="cached"` with `idempotency_replay=true`.
8. Use UTC timestamp plus integer epoch microseconds for ordering. Do not order by mixed-offset ISO text.
9. Use a bounded recorder timeout and swallow/log recorder failures so tool behavior is unchanged when capture breaks.
10. Keep all event fields metadata-only. Tests must prove raw args, outputs, paths, and exception messages do not persist.
11. Add package-boundary tests so importing the no-op recorder and gateway config does not eagerly import optional storage dependencies.
12. Require a persistent reporting store for standalone CLI `tool-events report/export/cleanup`; explicit in-memory reporting is only for injected tests or same-process embedders.
13. Record protocol `tools/call` failures that return before handler dispatch, including early rate-limit, tool-name validation, and authorization failures.
14. Resolve the recorder defensively with `getattr(..., "tool_use_recorder", NoopToolUseRecorder())` so existing custom dependency bundles and tests that use `SimpleNamespace` do not break.
15. Add a gateway bridge metadata side-channel for `profile.tools.call` because the wrapper cannot reliably infer `effective_tool_name` from the raw `tool_id` argument.
16. Make `ToolUseEvent` immutable/frozen and test mutation rejection.
17. Include bounded-report disclosure fields such as `events_scanned`, `event_limit`, and `truncated` so operators know when aggregates are partial.

## File Structure

Create:

- `mcp_unified/tool_use_reporting/__init__.py`: public package exports for models, recorder contracts, in-memory store, and report service. Do not export SQLite store here.
- `mcp_unified/tool_use_reporting/models.py`: Pydantic models, literal aliases, timestamp normalization, filter/query/result models.
- `mcp_unified/tool_use_reporting/sanitization.py`: bounded string and safe id sanitizers shared by protocol and gateway capture.
- `mcp_unified/tool_use_reporting/builders.py`: event builder helpers, context extraction, result/eval metadata extraction, and exception-to-status mapping.
- `mcp_unified/tool_use_reporting/recorder.py`: `ToolUseRecorder`, `NoopToolUseRecorder`, `StoreBackedToolUseRecorder`, timeout helper.
- `mcp_unified/tool_use_reporting/store.py`: `ToolUseEventStore` protocol and `InMemoryToolUseEventStore`.
- `mcp_unified/tool_use_reporting/sqlite.py`: SQLAlchemy-backed `SQLiteToolUseEventStore`; this is the only tool-use reporting module that imports SQLAlchemy.
- `mcp_unified/tool_use_reporting/reporting.py`: `ToolUseReportService` and aggregate calculations.
- `mcp_unified/gateway/tool_use_reporting.py`: `ToolUseReportingGatewayRuntime` wrapper and gateway config/store factory helpers.
- `tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_models.py`: event model and sanitizer tests.
- `tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_store.py`: in-memory and SQLite store contract tests.
- `tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py`: in-process protocol capture tests.
- `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_use_reporting.py`: gateway wrapper/bootstrap capture tests.

Modify:

- `mcp_unified/interfaces/runtime.py`: add optional/defaulted `tool_use_recorder` dependency at the end of `MCPRuntimeDependencies`.
- `tldw_Server_API/app/core/MCP_unified/interfaces/runtime.py`: re-export any new interface symbols only if implementation exposes them from `mcp_unified.interfaces.runtime`.
- `tldw_Server_API/app/core/MCP_unified/protocol.py`: record metadata-only tool-use events for `tools/call`.
- `mcp_unified/gateway/config.py`: add `GatewayToolUseReportingConfig`, parse config, build event store/recorder, and wrap runtime during bootstrap.
- `mcp_unified/gateway/bootstrap.py`: accept optional gateway runtime wrapper/recorder only if needed after config integration review.
- `mcp_unified/gateway/profile_runtime.py`: attach safe bridge-resolution metadata for `profile.tools.call` so the outer wrapper can report requested and effective tool names without inspecting raw delegated arguments.
- `mcp_unified/gateway/cli.py`: add `tool-events report`, `tool-events export`, and `tool-events cleanup` commands; include config validation payload.
- `mcp_unified/README.md`: document the reporting feature at package overview level.
- `mcp_unified/USER_GUIDE.md`: add operator guide for enabling capture, CLI reports, export, cleanup, and privacy constraints.
- `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py`: add CLI parse/handler coverage.
- `tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py`: preserve compatibility for dependency bundles without `tool_use_recorder`.
- `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`: assert no eager SQLAlchemy import from lightweight reporting imports.

## Implementation Tasks

### Task 1: Event Models And Sanitizers

**Files:**
- Create: `mcp_unified/tool_use_reporting/__init__.py`
- Create: `mcp_unified/tool_use_reporting/models.py`
- Create: `mcp_unified/tool_use_reporting/sanitization.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_models.py`

- [ ] **Step 1: Write failing tests for timestamp normalization and safe metadata**

```python
from datetime import datetime, timezone, timedelta

import pytest

from mcp_unified.tool_use_reporting.models import ToolUseEvent


def test_tool_use_event_normalizes_created_at_to_utc_epoch_ordering():
    event = ToolUseEvent(
        created_at=datetime(2026, 6, 6, 12, 0, tzinfo=timezone(timedelta(hours=-7))),
        runtime_surface="protocol",
        requested_tool_name="git.status",
        status="success",
    )

    assert event.created_at_utc.tzinfo == timezone.utc
    assert event.created_at_utc.isoformat() == "2026-06-06T19:00:00+00:00"
    assert event.created_at_epoch_us > 0


def test_tool_use_event_rejects_or_omits_sensitive_payload_fields():
    event = ToolUseEvent(
        runtime_surface="gateway",
        requested_tool_name="fs.read",
        status="error",
        reason_code="/Users/example/private.txt",
        raw_arguments={"path": "/Users/example/private.txt"},
    )

    dumped = event.model_dump(mode="json")
    assert "raw_arguments" not in dumped
    assert "/Users/example" not in str(dumped)


def test_tool_use_event_is_immutable():
    event = ToolUseEvent(
        runtime_surface="protocol",
        requested_tool_name="git.status",
        status="success",
    )

    with pytest.raises((TypeError, ValueError)):
        event.status = "error"
```

Run: `source .venv/bin/activate` then `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_models.py -q`

Expected: FAIL because the package and model do not exist.

- [ ] **Step 2: Implement minimal models and sanitizers**

Implementation requirements:

- Use Pydantic `BaseModel`.
- Accept `created_at` input as an alias or internal field, normalize to `created_at_utc`.
- Generate `event_id` with `uuid4().hex`.
- Configure the model as immutable/frozen. Use the Pydantic v2 `ConfigDict(frozen=True)` path when available and a v1-compatible fallback if needed.
- Bound string fields to a small constant, for example 128 chars for ids/names and 64 chars for status/reason categories.
- Use literal aliases for:
  - `runtime_surface`: `protocol`, `gateway`
  - `execution_origin`: `executed`, `cached`, `denied`, `unavailable`, `failed_before_execution`
  - `status`: `success`, `error`, `denied`, `approval_required`, `unavailable`, `invalid_params`, `rate_limited`
- Default unknown safe dimensions to `None` or `"unknown"` consistently. Use `None` for optional host context fields and `"unknown"` for required groupable names.
- Set `extra="ignore"` so accidental raw fields such as `raw_arguments` are not persisted by model construction.

- [ ] **Step 3: Add sanitizer tests for allowlisted ids**

```python
from mcp_unified.tool_use_reporting.sanitization import sanitize_safe_id


def test_sanitize_safe_id_allows_bounded_profile_model_mode_ids():
    assert sanitize_safe_id("Architect-01", field="profile_id") == "Architect-01"
    assert sanitize_safe_id("gpt-4.1-mini", field="model_id") == "gpt-4.1-mini"


def test_sanitize_safe_id_drops_paths_emails_and_long_values():
    assert sanitize_safe_id("/Users/me/project", field="profile_id") is None
    assert sanitize_safe_id("person@example.com", field="profile_id") is None
    assert sanitize_safe_id("x" * 512, field="profile_id") is None
```

- [ ] **Step 4: Run model tests**

Run: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_models.py -q`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add mcp_unified/tool_use_reporting tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_models.py
git commit -m "feat: add MCP tool-use reporting event models"
```

### Task 2: Event Builder And Recorder Contracts

**Files:**
- Create: `mcp_unified/tool_use_reporting/builders.py`
- Create: `mcp_unified/tool_use_reporting/recorder.py`
- Modify: `mcp_unified/tool_use_reporting/__init__.py`
- Modify: `mcp_unified/interfaces/runtime.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_models.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`

- [x] **Step 1: Write failing tests for status mapping and default dependency compatibility**

```python
from mcp_unified.interfaces.runtime import MCPRuntimeDependencies
from mcp_unified.tool_use_reporting.builders import classify_tool_use_exception
from mcp_unified.tool_use_reporting.recorder import NoopToolUseRecorder
from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol


class FakeGovernanceDenied(PermissionError):
    governance = {"reason_code": "workspace_out_of_scope"}


def test_classify_tool_use_exception_uses_safe_reason_codes():
    status, reason = classify_tool_use_exception(FakeGovernanceDenied("secret path"))
    assert status == "denied"
    assert reason == "workspace_out_of_scope"


def test_runtime_dependencies_default_to_noop_tool_use_recorder(runtime_dependency_kwargs):
    deps = MCPRuntimeDependencies(**runtime_dependency_kwargs)
    assert isinstance(deps.tool_use_recorder, NoopToolUseRecorder)


def test_protocol_accepts_dependency_bundle_without_tool_use_recorder(runtime_dependency_namespace):
    if hasattr(runtime_dependency_namespace, "tool_use_recorder"):
        delattr(runtime_dependency_namespace, "tool_use_recorder")

    protocol = MCPProtocol(dependencies=runtime_dependency_namespace)

    assert isinstance(protocol._tool_use_recorder, NoopToolUseRecorder)
```

If no reusable `runtime_dependency_kwargs` fixture exists, add a local fixture with the same fake dependencies used by existing runtime boundary tests.
If no namespace fixture exists, add the compatibility test beside `_fake_runtime_dependencies()` in `test_extraction_contracts.py`, which already constructs `SimpleNamespace` dependency bundles.

- [x] **Step 2: Implement builder and recorder contracts**

Implementation requirements:

- `ToolUseRecorder` is an async protocol with `record_tool_use(event: ToolUseEvent) -> None`.
- `NoopToolUseRecorder.record_tool_use()` returns without work.
- `StoreBackedToolUseRecorder` appends to a store and exposes a configurable timeout.
- Use `asyncio.wait_for()` in the recorder helper.
- Log recorder failure by sanitized exception class only. Do not include raw exception messages.
- `classify_tool_use_exception()` must inspect exception class names and safe attributes rather than importing host protocol classes.
- `extract_safe_context_dimensions()` must only read candidate metadata keys:
  - profile: `profile_id`, `mcp_profile_id`, `gateway_profile_id`
  - mode: `mode_id`, `mcp_mode_id`
  - model: `model_id`, `mcp_model_id`
  - correlation: only `correlation_id`, `request_id` when `metadata["mcp_tool_use_safe_correlation_id"] is True`
- Add `MCPRuntimeDependencies.tool_use_recorder` as the final dataclass field with `field(default_factory=NoopToolUseRecorder)`.
- In `MCPProtocol.__init__`, resolve the recorder with `getattr(self.dependencies, "tool_use_recorder", NoopToolUseRecorder())` rather than direct attribute access, because host tests and embedders may pass duck-typed dependency bundles.

- [x] **Step 3: Add import boundary test**

Add or extend `test_runtime_package_boundary.py`:

```python
def test_tool_use_reporting_core_imports_do_not_import_sqlalchemy(monkeypatch):
    import sys

    for name in list(sys.modules):
        if name.startswith("mcp_unified.tool_use_reporting"):
            sys.modules.pop(name)
    sys.modules.pop("sqlalchemy", None)

    import mcp_unified.tool_use_reporting.recorder  # noqa: F401
    import mcp_unified.tool_use_reporting.builders  # noqa: F401

    assert "sqlalchemy" not in sys.modules
```

- [x] **Step 4: Run focused tests**

Run:

```bash
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_models.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add mcp_unified/tool_use_reporting mcp_unified/interfaces/runtime.py tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_models.py tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
git commit -m "feat: add MCP tool-use reporting recorder contract"
```

### Task 3: Store And Aggregate Report Service

**Files:**
- Create: `mcp_unified/tool_use_reporting/store.py`
- Create: `mcp_unified/tool_use_reporting/sqlite.py`
- Create: `mcp_unified/tool_use_reporting/reporting.py`
- Modify: `mcp_unified/tool_use_reporting/__init__.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_store.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`

- [x] **Step 1: Write failing in-memory and SQLite store contract tests**

```python
from datetime import datetime, timezone, timedelta

import pytest

from mcp_unified.tool_use_reporting.models import ToolUseEvent, ToolUseEventQuery
from mcp_unified.tool_use_reporting.store import InMemoryToolUseEventStore
from mcp_unified.tool_use_reporting.sqlite import SQLiteToolUseEventStore


@pytest.mark.parametrize("store_factory", [
    lambda tmp_path: InMemoryToolUseEventStore(),
    lambda tmp_path: SQLiteToolUseEventStore(tmp_path / "tool-use.sqlite3"),
])
async def test_store_queries_events_by_epoch_newest_first(tmp_path, store_factory):
    store = store_factory(tmp_path)
    older = ToolUseEvent(
        created_at=datetime(2026, 6, 6, 10, 0, tzinfo=timezone.utc),
        runtime_surface="protocol",
        requested_tool_name="git.status",
        status="success",
    )
    newer = ToolUseEvent(
        created_at=datetime(2026, 6, 6, 7, 30, tzinfo=timezone(timedelta(hours=-7))),
        runtime_surface="protocol",
        requested_tool_name="git.status",
        status="success",
    )

    await store.append_event(older)
    await store.append_event(newer)

    rows = await store.query_events(ToolUseEventQuery(limit=10))
    assert [row.event_id for row in rows] == [newer.event_id, older.event_id]
```

- [x] **Step 2: Implement store protocol and in-memory store**

Implementation requirements:

- `ToolUseEventStore.append_event(event)`
- `query_events(query)`
- `delete_events_older_than(cutoff)`
- `delete_events_over_limit(max_events)`
- `export_events(query, format)`
- Query filters mirror the spec filters and always apply a maximum limit.
- In-memory store returns copy-isolated Pydantic models.

- [x] **Step 3: Implement SQLAlchemy-backed SQLite store**

Implementation requirements:

- Import SQLAlchemy only in `mcp_unified/tool_use_reporting/sqlite.py`.
- Use SQLAlchemy Core, not `sqlite3`.
- Use an internal `_run_db()` helper with `asyncio.to_thread()`.
- Store scalar columns for common filters and a JSON payload for full event data.
- Include `created_at_epoch_us` and order by `created_at_epoch_us DESC, event_id DESC`.
- Create indexes for time, profile, model, requested/effective tool, prompt, status, runtime surface.
- Provide `close()` and `aclose()`.
- Use bounded limits and cursor pagination. A cursor can be an opaque string encoding `(created_at_epoch_us, event_id)`.

- [x] **Step 4: Write failing report aggregation tests**

```python
from mcp_unified.tool_use_reporting.reporting import ToolUseReportService
from mcp_unified.tool_use_reporting.models import ToolUseReportQuery


async def test_report_groups_by_tool_prompt_with_tool_call_rates():
    store = InMemoryToolUseEventStore()
    await store.append_event(ToolUseEvent(
        runtime_surface="protocol",
        requested_tool_name="fs.read",
        tool_prompt_id="fs.read.default",
        status="success",
    ))
    await store.append_event(ToolUseEvent(
        runtime_surface="protocol",
        requested_tool_name="fs.read",
        tool_prompt_id="fs.read.default",
        status="denied",
        reason_code="permission_denied",
    ))

    report = await ToolUseReportService(store).build_report(
        ToolUseReportQuery(group_by="tool_prompt")
    )

    row = report.rows[0]
    assert row.group_key == "fs.read.default"
    assert row.call_count == 2
    assert row.tool_call_success_rate == 0.5
    assert row.top_reason_codes[0]["reason_code"] == "permission_denied"


async def test_report_discloses_when_event_limit_truncates_aggregates():
    store = InMemoryToolUseEventStore()
    for index in range(5):
        await store.append_event(ToolUseEvent(
            runtime_surface="protocol",
            requested_tool_name=f"tool.{index}",
            status="success",
        ))

    report = await ToolUseReportService(store).build_report(
        ToolUseReportQuery(group_by="tool", event_limit=2)
    )

    assert report.events_scanned == 2
    assert report.event_limit == 2
    assert report.truncated is True
```

- [x] **Step 5: Implement report service**

Implementation requirements:

- Support group dimensions: `profile`, `tool_prompt`, `model`, `tool`.
- Use names from the spec, for example `tool_call_success_rate`, never `task_success_rate`.
- Bound group count and top reason-code count.
- Calculate p50 and p95 duration from bounded rows in memory for first slice.
- Include `events_scanned`, `event_limit`, and `truncated` on the report payload. Set `truncated=true` whenever the query hit the event limit before exhausting the filtered window.
- Return JSON-serializable Pydantic models.

- [x] **Step 6: Run focused tests**

Run:

```bash
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_store.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  -q
```

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add mcp_unified/tool_use_reporting tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_store.py tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
git commit -m "feat: add MCP tool-use reporting stores"
```

### Task 4: Protocol Capture

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/protocol.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_tool_observability.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_and_category.py`

- [x] **Step 1: Write failing protocol success and denial tests**

Use existing protocol test helpers from `test_tool_observability.py` and `support.py`.

```python
async def test_protocol_records_successful_tool_use_event(protocol_with_recorder):
    protocol, recorder = protocol_with_recorder

    response = await protocol._handle_tools_call(
        {"name": "test.read", "arguments": {}},
        _request_context(metadata={"profile_id": "architect", "model_id": "gpt-4.1"}),
    )

    assert response["tool"] == "test.read"
    event = recorder.events[-1]
    assert event.runtime_surface == "protocol"
    assert event.requested_tool_name == "test.read"
    assert event.effective_tool_name == "test.read"
    assert event.profile_id == "architect"
    assert event.model_id == "gpt-4.1"
    assert event.status == "success"


async def test_protocol_records_prepare_denial_without_raw_error(protocol_with_recorder):
    protocol, recorder = protocol_with_recorder

    with pytest.raises(PermissionError):
        await protocol._handle_tools_call(
            {"name": "write.file", "arguments": {"path": "/Users/me/secret.txt"}},
            _request_context(metadata={"allowed_tools": ["read.file"]}),
        )

    event = recorder.events[-1]
    assert event.execution_origin == "failed_before_execution"
    assert event.status == "denied"
    assert event.reason_code == "permission_denied"
    assert "/Users/me" not in event.model_dump_json()


async def test_protocol_records_early_process_request_tool_name_error(protocol_with_recorder):
    protocol, recorder = protocol_with_recorder

    response = await protocol.process_request(
        {
            "jsonrpc": "2.0",
            "id": "req-1",
            "method": "tools/call",
            "params": {"name": "../secret", "arguments": {}},
        },
        _request_context(),
    )

    assert response.error.code == ErrorCode.INTERNAL_ERROR
    event = recorder.events[-1]
    assert event.runtime_surface == "protocol"
    assert event.requested_tool_name == "unknown"
    assert event.status == "invalid_params"
    assert event.execution_origin == "failed_before_execution"
```

- [x] **Step 2: Add recorder helper methods to `MCPProtocol`**

Implementation requirements:

- Add `_tool_use_recorder` from `self.dependencies.tool_use_recorder` or equivalent existing dependency access.
- Add `_should_record_tool_use(context)` that returns false when `context.metadata.get("mcp_tool_use_observed") is True`.
- Add `_record_tool_use_event(event)` with bounded timeout via the recorder helper.
- Add `_record_process_request_tool_use_failure(...)` or an equivalent helper for `process_request()` paths that return JSON-RPC errors before `_handle_tools_call()` is invoked.
- Do not change existing audit events, metrics, or tool response shape.

- [x] **Step 3: Instrument `process_request()` for early `tools/call` failures**

Implementation requirements:

- Detect `request.method == "tools/call"` after request parsing but before each early return.
- Record bounded metadata-only events for:
  - top-level rate-limit failures when the requested tool name is available
  - missing/non-string/regex-invalid tool names
  - authorization failures before handler dispatch
- Preserve the existing JSON-RPC error codes and return payloads. Some legacy paths intentionally map invalid regex names to `INTERNAL_ERROR`; the event status should still be normalized to `invalid_params`.
- Do not record non-`tools/call` method failures in this slice.
- Reuse the same safe context extraction and recorder timeout helper.

- [x] **Step 4: Instrument `_handle_tools_call()` for preparation failures**

Implementation requirements:

- Start a monotonic timer before `prepare_tool_call()`.
- On prepare exception, build a partial event with:
  - `runtime_surface="protocol"`
  - requested tool name if string and safe, otherwise `"unknown"`
  - `execution_origin` from classifier, usually `failed_before_execution`, `denied`, or `unavailable`
  - safe profile/mode/model context
  - status/reason from `classify_tool_use_exception()`
  - duration
- Await recorder, then re-raise the original exception.

- [x] **Step 5: Instrument `execute_prepared_tool_call()` for success, execution errors, and idempotency**

Implementation requirements:

- Record after normal execution returns, using `payload.get("eval")` when present and `prepared.tool_def["metadata"]["eval"]` as fallback.
- In non-idempotent execution, catch exceptions around `_execute_tool_call()`, record sanitized event, then re-raise.
- In idempotent execution, record after `self._idempotency.run()` returns:
  - `execution_origin="cached"` and `idempotency_replay=true` when `from_cache` is true.
  - `execution_origin="executed"` otherwise.
- If `bind_arguments()` or argument fingerprint checks fail, record an invalid params event and re-raise.
- Preserve existing metrics for idempotency hit/miss.

- [x] **Step 6: Add tests for recorder failure and idempotency replay**

```python
async def test_protocol_recorder_failure_does_not_change_tool_response(protocol_with_failing_recorder):
    protocol, recorder = protocol_with_failing_recorder

    response = await protocol._handle_tools_call(
        {"name": "test.read", "arguments": {}},
        _request_context(),
    )

    assert response["tool"] == "test.read"
    assert recorder.called is True


async def test_protocol_records_idempotency_replay(protocol_with_recorder):
    protocol, recorder = protocol_with_recorder

    params = {"name": "test.write", "arguments": {}, "idempotencyKey": "same-key"}
    await protocol._handle_tools_call(params, _request_context())
    await protocol._handle_tools_call(params, _request_context())

    assert recorder.events[-1].execution_origin == "cached"
    assert recorder.events[-1].idempotency_replay is True
```

- [x] **Step 7: Run focused protocol tests**

Run:

```bash
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_tool_observability.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_and_category.py \
  -q
```

Expected: PASS.

- [x] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/MCP_unified/protocol.py tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py tldw_Server_API/app/core/MCP_unified/tests/test_tool_observability.py tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_and_category.py
git commit -m "feat: record MCP protocol tool-use events"
```

### Task 5: Gateway Runtime Wrapper And Config

**Files:**
- Create: `mcp_unified/gateway/tool_use_reporting.py`
- Modify: `mcp_unified/gateway/config.py`
- Modify: `mcp_unified/gateway/bootstrap.py` if needed after config integration
- Modify: `mcp_unified/gateway/profile_runtime.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_use_reporting.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`

- [x] **Step 1: Write failing wrapper tests**

```python
async def test_gateway_wrapper_records_direct_call_with_profile_and_model():
    runtime = ToolUseReportingGatewayRuntime(
        FakeGatewayRuntime(),
        recorder=MemoryToolUseRecorder(),
    )

    await runtime.call_tool(
        "git.status",
        {},
        GatewayRequestContext(
            request_id="req-1",
            metadata={"profile_id": "devops", "model_id": "gpt-4.1"},
        ),
    )

    event = runtime.recorder.events[-1]
    assert event.runtime_surface == "gateway"
    assert event.requested_tool_name == "git.status"
    assert event.effective_tool_name == "git.status"
    assert event.profile_id == "devops"
    assert event.status == "success"


async def test_gateway_wrapper_records_policy_denial():
    runtime = ToolUseReportingGatewayRuntime(DenyingGatewayRuntime(), recorder=MemoryToolUseRecorder())

    with pytest.raises(GatewayPolicyDenied):
        await runtime.call_tool("fs.write", {}, GatewayRequestContext(request_id="req-1"))

    event = runtime.recorder.events[-1]
    assert event.status == "denied"
    assert event.reason_code == "profile_tool_denied"


async def test_gateway_bridge_call_records_effective_tool_name_when_tool_id_differs():
    runtime = await profile_runtime_with_deferred_tool(
        tool_id="git-status",
        tool_name="git.status",
    )
    recorder = MemoryToolUseRecorder()
    wrapped = ToolUseReportingGatewayRuntime(runtime, recorder=recorder)

    await wrapped.call_tool(
        "profile.tools.call",
        {"tool_id": "git-status", "arguments": {}},
        GatewayRequestContext(request_id="req-1"),
    )

    event = recorder.events[-1]
    assert event.requested_tool_name == "profile.tools.call"
    assert event.effective_tool_name == "git.status"
    assert event.source_kind == "bridge"
```

- [x] **Step 2: Implement `ToolUseReportingGatewayRuntime`**

Implementation requirements:

- Delegate every `GatewayRuntime` method except `call_tool` directly.
- In `call_tool`, if `context.metadata["mcp_tool_use_observed"] is True`, delegate without recording.
- Otherwise create a copied `GatewayRequestContext` with:
  - `mcp_tool_use_observed=True`
  - `mcp_tool_use_outer_surface="gateway"`
- Record success, `GatewayPolicyDenied`, and sanitized generic exceptions.
- Detect bridge delegation metadata from safe context/result keys:
  - requested `profile.tools.call`
  - effective target from `mcp_tool_use_effective_tool_name` or result eval metadata.
  - requested bridge id from `mcp_tool_use_requested_tool_id` when present.
- Do not persist raw bridge arguments.

- [x] **Step 3: Add a safe bridge-resolution side-channel in `profile_runtime.py`**

Implementation requirements:

- When `ProfileAwareGatewayRuntime` handles `profile.tools.call`, it currently resolves `arguments["tool_id"]` to `resolved_name` before delegating. Add a helper that copies the `GatewayRequestContext` and attaches bounded metadata keys before `_call_backend_tool_through_policy()` delegates:
  - `mcp_tool_use_bridge_tool_name="profile.tools.call"`
  - `mcp_tool_use_requested_tool_id=<safe tool_id>`
  - `mcp_tool_use_effective_tool_name=<resolved backend tool name>`
  - `mcp_tool_use_source_kind="bridge"`
- The side-channel must not include delegated arguments.
- The gateway wrapper should prefer these metadata keys when building the event. This is required because `tool_id` is not always the backend tool name.

- [x] **Step 4: Add bootstrap config tests**

```python
def test_gateway_config_parses_tool_use_reporting_defaults():
    config = GatewayProfileBootstrapConfig()
    assert config.tool_use_reporting.enabled is False
    assert config.tool_use_reporting.store.kind == "memory"


async def test_bootstrap_wraps_runtime_when_tool_use_reporting_enabled(tmp_path):
    config = GatewayProfileBootstrapConfig(
        tool_use_reporting={
            "enabled": True,
            "store": {"kind": "sqlite", "sqlite_path": str(tmp_path / "events.sqlite3")},
        }
    )

    bootstrap = await bootstrap_profile_gateway_from_config(FakeGatewayRuntime(), config)
    assert isinstance(bootstrap.runtime, ToolUseReportingGatewayRuntime)
```

- [x] **Step 5: Implement `GatewayToolUseReportingConfig` and store factory**

Implementation requirements:

- Add `tool_use_reporting` field to `GatewayProfileBootstrapConfig` with defaults:
  - `enabled=False`
  - `store.kind="memory"` for tests/no persistence only
  - optional `store.sqlite_path`
  - `write_timeout_seconds`
  - `retention_max_age_days`
  - `retention_max_events`
  - `export_default_limit`
  - `report_default_window`
- Reject config that enables SQLite reporting without a path unless a safe default path already exists in gateway config conventions.
- Allow explicit memory reporting for tests and same-process embedders, but document that standalone CLI report/export/cleanup requires SQLite.
- Build `StoreBackedToolUseRecorder` only when enabled.
- Wrap `ProfileAwareGatewayRuntime` after profile runtime construction so all profile policy decisions are visible.
- Keep external runtime manager construction unchanged.

- [x] **Step 6: Run focused gateway tests**

Run:

```bash
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_use_reporting.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  -q
```

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add mcp_unified/gateway/tool_use_reporting.py mcp_unified/gateway/config.py mcp_unified/gateway/bootstrap.py mcp_unified/gateway/profile_runtime.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_use_reporting.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py
git commit -m "feat: add gateway tool-use reporting runtime wrapper"
```

### Task 6: CLI Report, Export, And Cleanup

**Files:**
- Modify: `mcp_unified/gateway/cli.py`
- Modify: `mcp_unified/gateway/config.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py`

- [ ] **Step 1: Write failing CLI parser and command tests**

```python
def test_gateway_cli_parses_tool_events_report():
    parser = _build_parser()
    args = parser.parse_args([
        "tool-events",
        "report",
        "--config",
        "gateway.toml",
        "--group-by",
        "profile",
        "--since",
        "24h",
    ])

    assert args.command == "tool-events"
    assert args.tool_events_command == "report"
    assert args.group_by == "profile"


def test_validate_config_includes_tool_use_reporting_payload(tmp_path, capsys):
    config_path = tmp_path / "gateway.json"
    config_path.write_text('{"tool_use_reporting": {"enabled": false}}')

    assert _handle_validate_config(_args(path=config_path)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tool_use_reporting"]["enabled"] is False
```

- [ ] **Step 2: Add `tool-events` subcommands**

Implementation requirements:

- Add nested subparser `tool-events` with:
  - `report`
  - `export`
  - `cleanup`
- Reuse `_add_profile_config_argument()`.
- Report command options:
  - `--group-by` choices `profile`, `tool_prompt`, `model`, `tool`
  - `--since`
  - filter flags from spec
  - `--limit`
  - JSON output only in first slice
- Export command options:
  - `--format jsonl` default, optional `json`
  - `--output`
  - filters and `--limit`
- Cleanup command options:
  - `--max-age-days`
  - `--max-events`
- Handlers should load the gateway config, build the reporting store, run the service, and `_emit_json()` a deterministic payload.
- If reporting is disabled, return JSON error with `reason_code="tool_use_reporting_disabled"`.
- If reporting is enabled with only an in-memory store, return JSON error with `reason_code="tool_use_reporting_persistent_store_required"` for CLI report/export/cleanup.

- [ ] **Step 3: Include config validation payload**

Add to `_validated_config_payload()`:

```python
"tool_use_reporting": {
    "enabled": config.tool_use_reporting.enabled,
    "store": {
        "kind": config.tool_use_reporting.store.kind,
        "sqlite_path": str(config.tool_use_reporting.store.sqlite_path)
        if config.tool_use_reporting.store.sqlite_path is not None
        else None,
    },
    "retention_max_age_days": config.tool_use_reporting.retention_max_age_days,
    "retention_max_events": config.tool_use_reporting.retention_max_events,
}
```

- [ ] **Step 4: Run focused CLI tests**

Run: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcp_unified/gateway/cli.py mcp_unified/gateway/config.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py
git commit -m "feat: add MCP tool-use reporting CLI"
```

### Task 7: Documentation, Package Boundaries, And Verification

**Files:**
- Modify: `mcp_unified/README.md`
- Modify: `mcp_unified/USER_GUIDE.md`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`
- Modify: `backlog/tasks/task-2264 - Plan-MCP-tool-use-evaluation-reporting-implementation.md` only if this plan task is finalized in the same branch.

- [ ] **Step 1: Update docs**

Document:

- What metadata-only tool-use reporting captures.
- What it deliberately does not capture.
- How to enable reporting in gateway config.
- How to run:
  - `mcp-unified-gateway tool-events report --group-by profile`
  - `mcp-unified-gateway tool-events export --format jsonl --since 7d`
  - `mcp-unified-gateway tool-events cleanup --max-age-days 30 --max-events 100000`
- How profiles, models, modes, and tool prompt ids appear in reports.
- Privacy and retention guidance.
- Relationship to operational metrics, traces, and future evaluator-labeled task outcomes.

- [ ] **Step 2: Run focused verification**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_models.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_store.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_use_reporting.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  -q
```

Expected: PASS.

- [ ] **Step 3: Run Bandit on touched Python scopes**

Run:

```bash
python -m bandit -r \
  mcp_unified/tool_use_reporting \
  mcp_unified/gateway/tool_use_reporting.py \
  mcp_unified/gateway/config.py \
  mcp_unified/gateway/cli.py \
  tldw_Server_API/app/core/MCP_unified/protocol.py \
  -f json \
  -o /tmp/bandit_mcp_tool_use_reporting.json
```

Expected: command exits 0, or any findings are existing/irrelevant and documented with evidence before merge.

- [ ] **Step 4: Run diff whitespace check**

Run: `git diff --check`

Expected: no output.

- [ ] **Step 5: Commit docs and final task updates**

```bash
git add mcp_unified/README.md mcp_unified/USER_GUIDE.md tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
git commit -m "docs: document MCP tool-use reporting"
```

## Execution Notes

- Use TDD for each task: write the failing test first, run it, implement the smallest change, rerun the focused test.
- Keep commits task-sized. Do not bundle protocol, gateway, and CLI changes in one commit.
- Do not import SQLAlchemy from package core modules or gateway config import paths that are used when reporting is disabled.
- Do not add raw payload capture in this slice.
- Do not modify `SQLiteMCPStore` schema for these events.
- If any existing test helper lacks an injectable recorder, add the smallest local fixture rather than broad test harness refactors.
- If subagents are not explicitly authorized for implementation, use `superpowers:executing-plans` inline and preserve the same task checkpoints.

## Final Verification Checklist

- [ ] New event model tests pass.
- [ ] Event model immutability test passes.
- [ ] New store/report tests pass.
- [ ] Report payload exposes `events_scanned`, `event_limit`, and `truncated`.
- [ ] Protocol success, denial, invalid params, not found, early `process_request()` failure, recorder failure, and idempotency replay tests pass.
- [ ] Protocol dependency compatibility test passes for a bundle without `tool_use_recorder`.
- [ ] Gateway direct call, bridge call with `tool_id != tool_name`, policy denial, and double-counting guard tests pass.
- [ ] CLI report/export/cleanup tests pass.
- [ ] Package boundary test proves no eager SQLAlchemy import from lightweight reporting modules.
- [ ] Bandit run is recorded for touched Python scopes.
- [ ] Docs explain privacy boundaries and retention.
- [ ] Backlog task for the implementation PR records touched files, verification, and known skips.
