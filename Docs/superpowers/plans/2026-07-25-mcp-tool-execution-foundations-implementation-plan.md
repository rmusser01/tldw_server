# MCP Tool Execution Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden generic MCP tool execution before model-running tools are added, so expected failures, rate admission, prepared-call integrity, and idempotent execution are fail-closed, independently reviewable, and stable under future tool changes.

**Architecture:** Keep `protocol.py` as a compatibility facade. Move idempotency ownership into `tool_execution/idempotency.py`; represent execution decisions in an immutable, HMAC-bound prepared policy; execute only after two integrity/live-binding checks; and use one generic expected-failure contract whose breaker action is enforced identically by fallback and injected breakers. Persist idempotent successes through a strict canonical, bounded, cancellation-shielded commit before post-execution observers.

**Tech Stack:** Python 3.11+, asyncio, dataclasses, FastAPI/Pydantic configuration, Redis-compatible async client, Loguru, pytest/pytest-asyncio, Ruff, Bandit.

**Backlog:** `TASK-2294.3.1`

**Design:** `Docs/superpowers/specs/2026-07-23-mcp-skills-model-only-runner-design.md`

---

## Scope

In scope:

- Generic expected tool failures and neutral/counting circuit-breaker behavior.
- Immutable prepared execution policy and bounded detached observer snapshots.
- HMAC coverage of every runtime execution control and idempotency relationship.
- Two complete integrity and live registry checks before module invocation.
- Generic fail-closed rate-admission metadata.
- Mechanical extraction, then hardening, of `IdempotencyManager`.
- Bounded contention, sticky backend ownership, strict replay encoding, active-scope partitioning, and cancellation-safe success finalization.
- Safe metrics, audit, hooks, tool-use reporting, and logs.
- Compatibility re-exports and focused documentation.

Out of scope:

- Skills, model, provider, prompt rendering, or credential implementation.
- Tool-name-specific behavior in `protocol.py` or `tool_execution/runtime.py`.
- New public permission subjects or changes to existing JSON-RPC mappings for invalid parameters, permission denial, approval, governance denial, actual rate-limit denial, or idempotency argument conflict.
- Populating model-runner active organization/team scope from request payloads. This task provides the explicit trusted context field and replay partition; the bounded model adapter task supplies its authenticated value.
- Exactly-once execution without a caller-supplied idempotency key.

## Non-Negotiable Invariants

1. An expected failure with breaker action `ignore` is not a success: it does not increment or reset closed-state counters, and in half-open it only releases the acquired probe lease.
2. An expected failure with breaker action `record_failure`, plus every unexpected execution exception, counts normally and reopens a half-open breaker with existing backoff.
3. Expected failures are returned as `tools/call` results with `isError: true`; they are not JSON-RPC request errors and never enter the idempotency result cache.
4. Runtime security decisions come only from an immutable `PreparedExecutionPolicy`, never from observer dictionaries decoded from tool-definition or scope snapshots.
5. HMAC, arguments, context, active scope, key relationship, module identity, operational module ID, and canonical tool-definition digest are checked before rate/idempotency admission and again inside the owner callback immediately before dispatch.
6. A request invokes its idempotency owner callback at most once. Redis fallback is allowed only before ownership and only when the request definitely did not acquire a remote lock.
7. A valid callback result is the caller outcome. Serialization, size, cache, Redis, release, metric, audit, reporting, or post-hook failures cannot cause redispatch or replace it with a tool failure.
8. Generic metadata, headers, session/client IDs, and tool content cannot partition the authenticated active-scope replay domain.
9. New execution controls derived from tool metadata, schema, or configuration must be typed `PreparedExecutionPolicy` fields and HMAC-bound before runtime consumption.
10. Logs in touched breaker/runtime/idempotency paths contain only constant reason codes, exception class names, module/tool IDs, and bounded structural values, never exception messages or reprs.
11. `asyncio.CancelledError` is re-raised before every broad/noncritical exception handler. It is deferred only after a valid idempotent success exists and only for the bounded success-finalization protocol.

## Execution Stages

## Stage 1: Characterize And Extract
**Goal:** Lock current compatibility behavior and move idempotency code out of `protocol.py` without changing behavior.
**Success Criteria:** Existing imports still resolve, `IdempotencyManager.__module__` points at `tool_execution.idempotency`, and focused existing tests remain green.
**Tests:** Extraction AST/import contracts and existing idempotency/category tests.
**Status:** Complete

## Stage 2: Expected Failure And Breaker Contract
**Goal:** Add one generic expected-failure result and identical fallback/injected breaker accounting.
**Success Criteria:** Closed and half-open neutral/counting matrices pass, safe error envelopes are emitted, and unexpected exceptions retain existing behavior.
**Tests:** Outcome validation, fallback/injected breaker tests, runtime envelope/reporting tests, safe-log assertions.
**Status:** Complete

## Stage 3: Prepared Policy And Admission Integrity
**Goal:** Remove mutable tool-definition authority from runtime and make admission fail closed when configured.
**Success Criteria:** Policy and snapshots are immutable/HMAC-bound, stale or tampered calls fail before dispatch, and fail-closed rate metadata is generic.
**Tests:** Canonical encoding, mutation/tamper, live replacement/disable/definition drift, second-check race, and rate-admission tests.
**Status:** Complete

## Stage 4: Idempotency Ownership And Success Commit
**Goal:** Guarantee callback-at-most-once per request with bounded waiting and robust post-success persistence.
**Success Criteria:** Local/Redis contention, ambiguity, persistence, serialization, mutation, cancellation, late completion, and shutdown fault tests pass.
**Tests:** Dedicated manager state-machine tests plus runtime observer-ordering tests.
**Status:** Complete

## Stage 5: Integration And Security Gate
**Goal:** Verify compatibility, documentation, package boundaries, lint, compile, and security scanning.
**Success Criteria:** Focused and MCP-wide tests pass, Ruff/Bandit/diff checks are clean, and Backlog records evidence.
**Tests:** Full MCP Unified test directory, standalone reporting tests in touched scope, compileall, Ruff, Bandit.
**Status:** Not Started

## Planned File Map

Create:

- `tldw_Server_API/app/core/MCP_unified/execution_outcomes.py`
- `tldw_Server_API/app/core/MCP_unified/tool_execution/canonical.py`
- `tldw_Server_API/app/core/MCP_unified/tool_execution/idempotency.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_tool_execution_outcomes.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_prepared_execution_integrity.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_manager.py`

Modify:

- `tldw_Server_API/app/core/MCP_unified/protocol.py`
- `tldw_Server_API/app/core/MCP_unified/protocol_types.py`
- `tldw_Server_API/app/core/MCP_unified/config.py`
- `tldw_Server_API/app/core/MCP_unified/server.py`
- `tldw_Server_API/app/core/MCP_unified/monitoring/metrics.py`
- `tldw_Server_API/app/core/MCP_unified/modules/base.py`
- `tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py`
- `tldw_Server_API/app/core/MCP_unified/tool_execution/__init__.py`
- `tldw_Server_API/app/core/MCP_unified/tool_execution/models.py`
- `tldw_Server_API/app/core/MCP_unified/tool_execution/dependencies.py`
- `tldw_Server_API/app/core/MCP_unified/tool_execution/security.py`
- `tldw_Server_API/app/core/MCP_unified/tool_execution/runtime.py`
- `tldw_Server_API/app/core/MCP_unified/tool_execution/reporting.py`
- `apps/mcp-unified/src/mcp_unified/tool_use_reporting/builders.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_concurrency_and_breaker.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_and_category.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_nested_tool_preparation.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_rate_limit_categories.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_tool_hooks.py`
- `tldw_Server_API/tests/MCP_unified/test_phase3_3_small_core_sanitizers.py`
- `tldw_Server_API/app/core/MCP_unified/README.md`
- `backlog/tasks/task-2294.3.1 - Harden-generic-MCP-tool-execution-foundations.md`

Conditional files should be added to the touched set only if their existing contract requires it. Do not perform unrelated cleanup in these files.

## Task 1: Characterize And Mechanically Extract IdempotencyManager

**Files:**

- Create `tldw_Server_API/app/core/MCP_unified/tool_execution/idempotency.py`
- Modify `tldw_Server_API/app/core/MCP_unified/protocol.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tool_execution/__init__.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_and_category.py`

### Steps

- [x] Add a failing extraction contract proving `protocol.py` no longer defines `class IdempotencyManager` and the compatibility import resolves to the extracted class.
- [x] Add characterization tests for local cache replay, argument binding, lock pruning, Redis-unavailable fallback, and the current runtime replacement seam.
- [x] Run the focused tests and confirm only the new extraction assertions fail.
- [x] Move the class and its private Redis/local helpers verbatim to `tool_execution/idempotency.py`; do not change the state machine in this commit.
- [x] Move only imports used exclusively by the manager (`OrderedDict`, Redis exception fallback, Redis redaction helper, and no-client factory) with it.
- [x] Re-export with `from .tool_execution.idempotency import IdempotencyManager` in `protocol.py` and from `tool_execution.__init__`.
- [x] Preserve `MCPProtocol._idempotency` replacement synchronization and constructor injection exactly.
- [x] Run extraction and characterization tests.
- [x] Commit the mechanical move separately.

Required extraction assertion:

```python
def test_protocol_reexports_extracted_idempotency_manager() -> None:
    import inspect

    from tldw_Server_API.app.core.MCP_unified import protocol
    from tldw_Server_API.app.core.MCP_unified.tool_execution.idempotency import (
        IdempotencyManager,
    )

    assert protocol.IdempotencyManager is IdempotencyManager
    assert inspect.getmodule(protocol.IdempotencyManager).__name__.endswith(
        ".tool_execution.idempotency"
    )
    protocol_source = inspect.getsource(protocol)
    assert "class IdempotencyManager" not in protocol_source
```

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q --tb=short \
  tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_and_category.py
```

Expected before move: the new extraction assertion fails. Expected after move: both files pass with no behavioral changes.

Commit:

```bash
git add tldw_Server_API/app/core/MCP_unified/protocol.py \
  tldw_Server_API/app/core/MCP_unified/tool_execution/__init__.py \
  tldw_Server_API/app/core/MCP_unified/tool_execution/idempotency.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_and_category.py
git commit -m "refactor(mcp): extract idempotency manager"
```

## Task 2: Add Generic Expected Failures And Breaker-Neutral Accounting

**Files:**

- Create `tldw_Server_API/app/core/MCP_unified/execution_outcomes.py`
- Create `tldw_Server_API/app/core/MCP_unified/tests/test_tool_execution_outcomes.py`
- Modify `tldw_Server_API/app/core/MCP_unified/modules/base.py`
- Modify `tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tool_execution/runtime.py`
- Modify `apps/mcp-unified/src/mcp_unified/tool_use_reporting/builders.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_concurrency_and_breaker.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py`

### Contract

Use an enum-backed reason catalog so callers cannot provide arbitrary public text or choose breaker behavior independently of the server-defined reason:

```python
from __future__ import annotations

from enum import Enum


class BreakerAction(str, Enum):
    IGNORE = "ignore"
    RECORD_FAILURE = "record_failure"


class ExpectedToolFailureReason(Enum):
    RATE_LIMIT_UNAVAILABLE = (
        "rate_limit_unavailable",
        "Rate-limit admission is temporarily unavailable.",
        BreakerAction.IGNORE,
    )
    IDEMPOTENCY_IN_PROGRESS = (
        "idempotency_in_progress",
        "A request with this idempotency key is still in progress.",
        BreakerAction.IGNORE,
    )
    IDEMPOTENCY_UNAVAILABLE = (
        "idempotency_unavailable",
        "Idempotent execution is temporarily unavailable.",
        BreakerAction.IGNORE,
    )
    STALE_PREPARED_CALL = (
        "stale_prepared_call",
        "The prepared tool call is no longer valid.",
        BreakerAction.IGNORE,
    )
    DEPENDENCY_UNAVAILABLE = (
        "dependency_unavailable",
        "A required tool dependency is temporarily unavailable.",
        BreakerAction.RECORD_FAILURE,
    )

    def __init__(
        self,
        reason_code: str,
        public_message: str,
        breaker_action: BreakerAction,
    ) -> None:
        self.reason_code = reason_code
        self.public_message = public_message
        self.breaker_action = breaker_action


class ExpectedToolFailure(Exception):
    def __init__(self, reason: ExpectedToolFailureReason) -> None:
        super().__init__(reason.reason_code)
        self.reason = reason
        self.reason_code = reason.reason_code
        self.public_message = reason.public_message
        self.breaker_action = reason.breaker_action
```

At import-time tests, assert every reason matches `^[a-z][a-z0-9_]{0,63}$`, every message is non-empty and at most 200 characters, and enum tuples are unique.

### Breaker Adapter

Do not change the shared infrastructure breaker. Use its existing `expected_exception` filter:

- Add `expected_exception: type | tuple[type[BaseException], ...] = Exception` to `ModuleCircuitBreakerConfig`.
- Map that field in `adapters/tldw_runtime._to_tldw_circuit_breaker_config`.
- Add private `_IgnoredModuleOutcome` and `_CountedModuleOutcome` wrappers in `modules/base.py`.
- Configure both fallback and injected breakers to count only `_CountedModuleOutcome`.
- Inside the operation passed to `call_async`, wrap `ExpectedToolFailure` according to its fixed `breaker_action`, wrap other `Exception` values as counted, and unwrap after breaker processing.
- Update `_DefaultModuleCircuitBreaker.call_async` to honor `config.expected_exception` before calling `record_failure`; nonmatching exceptions are re-raised and the half-open lease is still released by `finally`.
- Do not wrap `ModuleCircuitBreakerOpenError` or the injected `CircuitBreakerOpenError`; they are raised by `call_async` outside the module operation.

Core shape:

```python
async def _breaker_operation() -> T:
    try:
        return await _guarded_operation()
    except ExpectedToolFailure as exc:
        if exc.breaker_action is BreakerAction.IGNORE:
            raise _IgnoredModuleOutcome(exc) from None
        raise _CountedModuleOutcome(exc) from None
    except Exception as exc:
        raise _CountedModuleOutcome(exc) from None


try:
    result = await self._circuit_breaker.call_async(_breaker_operation)
except (_IgnoredModuleOutcome, _CountedModuleOutcome) as wrapped:
    latency_ms = max(0.0, (time.time() - start_time) * 1000.0)
    self._metrics.record_request(False, latency_ms)
    original = wrapped.original
    raise original.with_traceback(original.__traceback__) from None
```

The wrapper stores the original exception only in memory; it must not expose it via `__str__`, `__repr__`, logs, telemetry, or public output.

### Minimal Safe Runtime Mapping

In the same commit, catch `ExpectedToolFailure` outside the idempotency manager callback boundary and return the exact `isError: true` content specified in Task 7. This ordering ensures the callback raises through the manager without being cached, while `MCPProtocol.process_request` receives a normal `tools/call` result rather than an exception. Add shape-based reason-code classification to the standalone reporting builder without importing tldw server modules. Task 7 will complete per-origin metrics, audit, post-hook, and reporting detail, but no intermediate commit may expose an expected failure as a JSON-RPC internal error.

### Required Tests

- [x] Closed fallback breaker: prior failure count remains unchanged after ignored failure.
- [x] Closed injected breaker: same assertion using `create_tldw_circuit_breaker`.
- [x] Half-open fallback: ignored failure releases `_half_open_in_flight`, remains half-open, preserves failure/success counters, and allows the next probe.
- [x] Half-open injected: ignored failure releases `half_open_calls`, remains half-open, and does not advance success threshold.
- [x] Counted expected failure reopens both half-open implementations with normal backoff.
- [x] Unexpected exception counts in both implementations and re-raises the original exception type.
- [x] Success still resets/closes according to existing thresholds.
- [x] Cancellation remains uncounted and propagates.
- [x] Captured logs contain exception family/reason code but not a sentinel secret embedded in the original message.
- [x] Replace all `str(e)`/`repr(e)` interpolation and `exc_info=<exception>` in touched `BaseModule` lifecycle, health, tool-cache, and execution logs with structured module and exception-family fields.
- [x] An expected failure from a write callback bypasses local/Redis result caching and returns a JSON-RPC success whose `tools/call` payload has `isError: true` and the sole safe JSON content object.

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q --tb=short \
  tldw_Server_API/app/core/MCP_unified/tests/test_tool_execution_outcomes.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_concurrency_and_breaker.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py \
  tldw_Server_API/tests/Infrastructure/test_circuit_breaker.py
```

Expected: all fallback and injected matrix cases pass; no sentinel exception message appears in captured logs.

Commit:

```bash
git add tldw_Server_API/app/core/MCP_unified/execution_outcomes.py \
  tldw_Server_API/app/core/MCP_unified/modules/base.py \
  tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py \
  tldw_Server_API/app/core/MCP_unified/tool_execution/runtime.py \
  apps/mcp-unified/src/mcp_unified/tool_use_reporting/builders.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_tool_execution_outcomes.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_concurrency_and_breaker.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py
git commit -m "feat(mcp): add breaker-aware expected failures"
```

## Task 3: Define Strict Canonical Encoding And Immutable Prepared Policy

**Files:**

- Create `tldw_Server_API/app/core/MCP_unified/tool_execution/canonical.py`
- Create `tldw_Server_API/app/core/MCP_unified/tests/test_prepared_execution_integrity.py`
- Modify `tldw_Server_API/app/core/MCP_unified/protocol_types.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tool_execution/models.py`
- Modify `tldw_Server_API/app/core/MCP_unified/config.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tool_execution/security.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tool_execution/runtime.py`
- Modify `tldw_Server_API/app/core/MCP_unified/protocol.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_nested_tool_preparation.py`
- Modify `tldw_Server_API/tests/MCP_unified/test_phase3_3_small_core_sanitizers.py`

### Canonical JSON Primitive

Implement one encoder used by snapshots, HMAC payloads, active-scope fingerprints, and idempotency results:

```python
def canonical_json_bytes(value: JsonValue, *, max_bytes: int) -> bytes:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > max_bytes:
        raise CanonicalJsonTooLarge(max_bytes=max_bytes, actual_bytes=len(encoded))
    return encoded
```

Validate recursively before `json.dumps` so only `None`, booleans, integers, finite floats, strings, lists, and dictionaries with string keys are accepted. Reject tuples, sets, paths, dataclasses, bytes, custom objects, NaN, and infinities. Never use `default=str`. Decoder helpers must validate the expected top-level type and return a new object.

Limits:

- Tool-definition snapshot: `1_000_000` bytes.
- Scope-reporting snapshot: `256_000` bytes.
- Prepared HMAC payload: `64_000` bytes.
- Idempotency result: policy-bound, default `256_000`, hard maximum `1_000_000` bytes.

### Prepared Types

Add immutable types to `tool_execution/models.py` and reference them from `PreparedToolCall`:

```python
@dataclass(frozen=True, slots=True)
class CanonicalJsonSnapshot:
    encoded: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class IdempotencyExecutionPolicy:
    inject_argument: bool
    ttl_seconds: int
    contention_wait_seconds: int
    finalize_seconds: int
    lock_ttl_seconds: int
    max_entries: int
    max_result_bytes: int


@dataclass(frozen=True, slots=True)
class PreparedExecutionPolicy:
    version: Literal[1]
    effect: Literal["read", "write"]
    rate_limit_category: str
    rate_limit_fail_closed: bool
    idempotency: IdempotencyExecutionPolicy
```

`PreparedToolCall` must retain the operational module object but replace mutable authoritative dictionaries with:

- `policy`
- `tool_definition_snapshot`
- `scope_reporting_snapshot`
- `normalized_idempotency_key_digest`
- `idempotency_scope_fingerprint`

The raw normalized key remains only because a prepared write may inject it into an explicitly declared schema field. It is verified against its signed digest before any use. Observer accessors decode a separate fresh private copy for each hook, reporter, audit, or eval consumer; observers never share a mutable decoded dictionary.

For incremental compatibility, expose read-only `tool_def`, `scope_payload`, and `is_write` properties on `PreparedToolCall` that decode a new detached copy or derive from `policy`. Mark them observer/compatibility-only in the docstring. This keeps Task 3 green while Task 4 removes every runtime security decision from those properties.

Add explicit `RequestContext.server_auth_scope: AuthenticatedExecutionScope | None`, where the frozen scope permits only positive non-boolean integer `active_org_id` and `active_team_id`. Do not derive it from `context.metadata`. Its canonical digest is empty for no active scope and `sha256(canonical_json(scope))` otherwise. The no-scope cache-key shape must remain byte-for-byte unchanged; scoped keys insert `|scope:sha256:<64 lowercase hex>` before the unrestricted `|key:` segment so a crafted personal key cannot collide with a scoped replay domain.

Add the canonical server-auth scope object to `fingerprint_request_context` in addition to the separately signed idempotency-scope fingerprint. Generic metadata remains covered by the existing complete context fingerprint, but only `server_auth_scope` can affect the replay-domain suffix.

Configuration fields and bounds:

```python
idempotency_wait_seconds: int = Field(
    default=5,
    ge=1,
    le=30,
    validation_alias="MCP_IDEMPOTENCY_WAIT_SECONDS",
)
idempotency_finalize_seconds: int = Field(
    default=5,
    ge=1,
    le=15,
    validation_alias="MCP_IDEMPOTENCY_FINALIZE_SECONDS",
)
idempotency_result_max_bytes: int = Field(
    default=256_000,
    ge=1,
    le=1_000_000,
    validation_alias="MCP_IDEMPOTENCY_RESULT_MAX_BYTES",
)
```

When preparing the policy, reject rather than silently clamp invalid runtime doubles. Validate TTL, lock TTL, and max entries as positive integers and cap their prepared representations at `604_800`, `604_800`, and `100_000` respectively. Add matching config validation for production settings so oversized values fail during configuration load, not per request.

Derive `lock_ttl_seconds` as `max(ttl_seconds, module_timeout_seconds * 2 + finalize_seconds)` before enforcing the hard maximum. This keeps ownership through the bounded module timeout and success finalization window.

### Policy Derivation

Move final rate-category derivation, effect classification, idempotency injection eligibility, and all bounded config reads into preparation. `rate_limit_fail_closed` is true only when the resolved, server-authored tool metadata value is the JSON boolean `true`; strings and integers do not enable it. Runtime must not read category, network, input schema, fail-closed, timeout, TTL, limits, or effect from `tool_def`, module settings, or live config.

### HMAC Payload

Set an explicit integrity payload version. The strict canonical HMAC payload must include:

```python
{
    "version": 1,
    "tool_name": prepared.tool_name,
    "module_id": prepared.module_id or "",
    "policy": asdict(prepared.policy),
    "idempotency_cache_key": prepared.idempotency_cache_key or "",
    "normalized_idempotency_key_digest": prepared.normalized_idempotency_key_digest,
    "arguments_hash": prepared.arguments_hash or "",
    "context_fingerprint": prepared.context_fingerprint,
    "idempotency_scope_fingerprint": prepared.idempotency_scope_fingerprint,
    "tool_definition_sha256": prepared.tool_definition_snapshot.sha256,
    "scope_reporting_sha256": prepared.scope_reporting_snapshot.sha256,
}
```

Before HMAC comparison, recompute arguments, context, normalized-key digest, authenticated-scope fingerprint, cache-key relationship, and both snapshot digests. Decode each snapshot, strictly re-encode it under its original limit, and require byte-for-byte equality with the stored bytes so a compatibility helper cannot sign an alternate non-canonical representation. Use `hmac.compare_digest` for HMAC and fixed-length digests. Remove permissive `default=str` from argument hashing; invalid public argument JSON must fail preparation as invalid parameters rather than collapse to a coerced hash.

The normalized-key digest is `sha256(normalized_key.encode("utf-8"))` in lowercase hex, or the empty string when no key is present. Preserve the current normalized key acceptance contract in this task; do not silently truncate it. The cache-key relationship check must invoke the same key builder used during preparation and compare the complete string.

Make `ToolExecutionSecurity.make_idempotency_cache_key` the sole authoritative key builder. Remove the duplicate implementation from `ToolExecutionRuntime`; the runtime asks security to verify the prepared relationship and never constructs an alternate key. Keep `MCPProtocol._make_idempotency_cache_key` only as a compatibility delegate so existing tests or embedders that patch the facade still reach the one builder.

### Required Tests

- [x] Canonical output is UTF-8, sorted, compact, Unicode-preserving, and stable.
- [x] Non-string keys, tuples, sets, bytes, paths, custom objects, NaN, and infinities are rejected.
- [x] Exact-limit payloads pass and over-limit payloads fail.
- [x] A manually signed but non-canonical snapshot encoding is rejected even when its stored SHA-256 matches its bytes.
- [x] Mutating source `tool_def` or `scope_payload` after preparation does not alter observer snapshots or policy.
- [x] Mutating policy, snapshots, raw normalized key, cache key, arguments, context, or explicit server-auth scope is detected before admission.
- [x] `metadata["org_id"]` and `metadata["team_id"]` do not change the active-scope fingerprint or cache key.
- [x] Personal/no-scope key shape matches the existing literal string.
- [x] Same user/key/arguments under two explicit active scopes yields different fixed-format digests and no raw IDs.
- [x] Policy rate category and idempotency-injection fields are fixed at preparation even if shared source dictionaries or config objects mutate; Task 4 proves runtime consumption.

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q --tb=short \
  tldw_Server_API/app/core/MCP_unified/tests/test_prepared_execution_integrity.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_nested_tool_preparation.py \
  tldw_Server_API/tests/MCP_unified/test_phase3_3_small_core_sanitizers.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_config_safe_defaults.py
```

Expected: canonical/policy tests pass and existing preparation behavior remains compatible for valid JSON requests.

Commit:

```bash
git add tldw_Server_API/app/core/MCP_unified/tool_execution/canonical.py \
  tldw_Server_API/app/core/MCP_unified/tool_execution/models.py \
  tldw_Server_API/app/core/MCP_unified/protocol_types.py \
  tldw_Server_API/app/core/MCP_unified/config.py \
  tldw_Server_API/app/core/MCP_unified/tool_execution/security.py \
  tldw_Server_API/app/core/MCP_unified/tool_execution/runtime.py \
  tldw_Server_API/app/core/MCP_unified/protocol.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_prepared_execution_integrity.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_nested_tool_preparation.py \
  tldw_Server_API/tests/MCP_unified/test_phase3_3_small_core_sanitizers.py
git commit -m "feat(mcp): bind immutable prepared execution policy"
```

## Task 4: Enforce Live Binding Twice And Fail-Closed Rate Admission

**Files:**

- Modify `tldw_Server_API/app/core/MCP_unified/tool_execution/security.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tool_execution/runtime.py`
- Modify `tldw_Server_API/app/core/MCP_unified/modules/registry.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_prepared_execution_integrity.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_rate_limit_categories.py`

### Live Binding

Add one async `verify_prepared_tool_call(prepared, *, require_live_binding: bool)` method that performs the complete integrity check. When live binding is required it must:

1. Resolve `find_module_for_tool(tool_name)` from the current injected registry.
2. Require object identity with `prepared.module`.
3. Require `get_module_id_for_tool(tool_name)` to equal the prepared operational module ID.
4. Require the module to remain operational through registry resolution and require its current `config.enabled` value not to be false.
5. Resolve and normalize the current tool definition exactly as preparation does.
6. Canonically encode it under the same 1,000,000-byte limit.
7. Require its digest to equal the signed prepared digest.

Any missing, disabled, replaced, renamed, or definition-changed binding raises `ExpectedToolFailure(STALE_PREPARED_CALL)`. Do not dispatch through the retained old module object.

Call the complete check:

- First, at the start of runtime before rate admission or argument binding/ownership.
- Second, inside the idempotency owner callback after rate and contention waits and immediately before `module.execute_with_circuit_breaker`.
- For non-idempotent calls, call the same owner callback directly so the second check is still used.
- For cached replays, repeat the complete check after the idempotency wait and before hit metrics, reporting, or return; a change completed before the current preparation remains replay-compatible.

### Rate Admission

Use only `prepared.policy.rate_limit_category` and `prepared.policy.rate_limit_fail_closed`:

```python
try:
    await self.rate_limiter.check_rate_limit(rate_key, category=policy.rate_limit_category)
except RateLimitExceeded:
    raise
except asyncio.CancelledError:
    raise
except self._noncritical_exceptions as exc:
    logger.warning(
        "MCP tool rate admission unavailable: module={module_id} tool={tool_name} "
        "error_type={error_type} fail_closed={fail_closed}",
        module_id=module_id or "unknown",
        tool_name=tool_name,
        error_type=exc.__class__.__name__,
        fail_closed=policy.rate_limit_fail_closed,
    )
    if policy.rate_limit_fail_closed:
        raise ExpectedToolFailure(
            ExpectedToolFailureReason.RATE_LIMIT_UNAVAILABLE
        ) from None
```

This admission runs before idempotency argument binding or ownership. An actual `RateLimitExceeded` keeps its current JSON-RPC mapping. A backend error for a tool without the flag keeps compatibility and dispatches.

### Required Tests

- [x] First check blocks a prepared call whose module was unregistered, disabled, non-operational, replaced under the same ID, remapped, or definition-mutated.
- [x] A registry/tool-definition mutation while rate admission is waiting is caught by the second check.
- [x] A mutation while idempotency contention is waiting is caught by the second check and the callback/module counter remains zero.
- [x] A cached replay rechecks live binding after its idempotency wait without breaking replay when a change completed before the current preparation.
- [x] Observer snapshot mutation never changes rate category, fail-closed behavior, effect, or idempotency injection.
- [x] Fail-closed backend error returns the generic expected failure before idempotency binding and breaker entry.
- [x] Unflagged backend error preserves dispatch compatibility.
- [x] Actual rate denial remains `RateLimitExceeded` and existing response/status behavior.
- [x] Cancellation during rate admission or cached-replay verification propagates without dispatch, idempotency mutation, or success reporting.
- [x] No branch refers to a Skills, model, provider, or concrete tool name.

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q --tb=short \
  tldw_Server_API/app/core/MCP_unified/tests/test_prepared_execution_integrity.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_rate_limit_categories.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_and_category.py
```

Expected: all stale/tamper/rate cases pass without module dispatch.

Commit:

```bash
git add tldw_Server_API/app/core/MCP_unified/tool_execution/security.py \
  tldw_Server_API/app/core/MCP_unified/tool_execution/runtime.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_prepared_execution_integrity.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_rate_limit_categories.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_and_category.py
git commit -m "fix(mcp): fail closed on stale execution admission"
```

## Task 5: Harden Idempotency Ownership, Contention, And Replay Encoding

**Files:**

- Create `tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_manager.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tool_execution/idempotency.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tool_execution/models.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tool_execution/dependencies.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tool_execution/runtime.py`
- Modify `tldw_Server_API/app/core/MCP_unified/protocol.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_and_category.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_prometheus_idempotency_metrics.py`

### Narrow Interface

Replace the runtime's separate argument-binding and execution calls, plus their free integer arguments, with one signed-policy operation and a typed result. Keeping binding and ownership in one manager call prevents Redis/local backend selection from diverging between two awaits:

```python
@dataclass(frozen=True, slots=True)
class IdempotencyRunResult:
    payload: dict[str, JsonValue]
    from_cache: bool
    persistence: Literal["durable", "local", "none"]


class IdempotencyManager:
    async def execute(
        self,
        cache_key: str,
        arguments_hash: str,
        execute_fn: Callable[[], Awaitable[dict[str, Any]]],
        *,
        policy: IdempotencyExecutionPolicy,
    ) -> IdempotencyRunResult:
        return await self._execute_bound(
            cache_key,
            arguments_hash,
            execute_fn,
            policy=policy,
        )

    async def shutdown(self) -> None:
        await self._drain_finalizers()
```

Keep `bind_arguments` and `run` only if an existing external compatibility test proves they are required; in that case, make them thin deprecated adapters over explicit manager state and do not use them from `ToolExecutionRuntime`. Update the existing replacement-manager test seam to implement `execute`.

Define an `IdempotencyExecutor` `Protocol` in `tool_execution/dependencies.py` with only the `execute` and `shutdown` signatures above, and type `ToolExecutionDependencies.idempotency` with that protocol instead of `Any`. The protocol may import policy/result types under `TYPE_CHECKING` to avoid runtime cycles.

Give `IdempotencyManager` one optional synchronous `on_degraded(stage: str, error_type: str) -> None` callback with a no-op default. Permit only a fixed internal stage catalog such as `serialization`, `result_size`, `redis_result_write`, `redis_release`, `finalize_timeout`, and `finalizer_stuck`. Wire it to `MetricsCollector.record_idempotency_degraded(stage)` in `protocol.py`; do not pass raw exceptions or dynamic backend text. Expose a read-only `remote_degraded` boolean for server health/status composition.

### Local State Machine

- Replace `_local_guard: asyncio.Lock` with a `threading.RLock` used only for fast in-memory dictionary/LRU operations; never await while holding it. Per-key execution ownership remains an `asyncio.Lock`.
- Store the bound argument hash, canonical decoded template, canonical bytes, and timestamp in the bounded LRU.
- Return `copy.deepcopy(template)` on every replay; never return the stored object.
- Check the process-local committed cache and its argument hash before selecting or contacting Redis. This is what makes degraded Redis persistence useful for later requests in the same process.
- Acquire the per-key lock with a monotonic deadline bounded by `contention_wait_seconds`.
- After acquisition, recheck the cache before invoking the callback.
- A waiter timeout raises `ExpectedToolFailure(IDEMPOTENCY_IN_PROGRESS)` and does not cancel the owner.
- Invoke `execute_fn` at most once after ownership.
- If the callback raises or is cancelled before a valid result exists, store no result.
- If canonical serialization succeeds, synchronously install the bounded local replay entry under the short `threading.RLock` before any later await.
- If serialization or size validation fails, return the original valid payload with persistence `none`, emit degraded observability, and do not cache.

### Redis State Machine

Use a Redis client with response decoding disabled for result bytes. Distinguish these phases explicitly:

```text
PRE_OWNER -> REMOTE_OWNER -> CALLBACK_COMPLETE -> LOCAL_COMMITTED -> FINALIZING -> DONE
```

Rules:

- Failure before attempting `SET NX` may fall back locally because this request definitely owns no remote lock.
- If `SET NX` raises, ownership is ambiguous: raise `IDEMPOTENCY_UNAVAILABLE`, do not execute, and do not fall back.
- If `SET NX` returns false, poll the result and retry lock acquisition under the same monotonic contention deadline. A result returns a replay; a later successful acquisition becomes owner; deadline expiry raises `IDEMPOTENCY_IN_PROGRESS`.
- A Redis error while polling or retrying acquisition fails as `IDEMPOTENCY_UNAVAILABLE`; it cannot fall back locally because another process may own or have completed the operation.
- Once `SET NX` returns true, backend choice is sticky. No subsequent exception can enter `_run_local` or call `execute_fn` again.
- Redis stores the exact canonical bytes created once after the callback.
- Result-write or release failure after callback returns the original success, retains the local replay copy when canonicalization succeeded, and marks persistence degraded.
- Lock release remains token checked through Lua.
- Invalid/corrupt remote cached bytes are not returned as success; emit safe degraded observability and continue only through a state transition that cannot re-enter an already-run callback.

Delete the current broad `except RedisError: return await _run_local(...)` path around `_run_redis`. Backend fallback must be chosen by an explicit pre-owner result, not by a broad exception handler.

### Argument Binding And Scope

- Use the same scoped cache key for result, lock, and argument binding, all inside `execute`.
- Treat ambiguous Redis argument-binding writes as `IDEMPOTENCY_UNAVAILABLE`; do not create a conflicting local binding and dispatch.
- Keep the argument fingerprint after failed execution for the configured TTL.
- Preserve the legacy no-scope key literal.
- Assert result/lock/binding Redis keys and local dictionaries contain only the fixed scope digest, never active org/team IDs.

### Required Fault Tests

- [x] Local and Redis replay call the callback once and return equivalent fresh copies.
- [x] Mutating first response or one replay does not alter future replay.
- [x] Concurrent local waiter times out at the configured bound without cancelling the owner.
- [x] Concurrent Redis waiter returns cache if it appears before the bound; otherwise returns `idempotency_in_progress` without dispatch.
- [x] Redis lock `SET NX` stores then raises: callback count zero, `idempotency_unavailable`, no local fallback.
- [x] Redis result `SET` fails after callback: callback count one, original success returned, local replay available, no second backend execution.
- [x] Redis release fails after callback: same callback-at-most-once assertion.
- [x] Argument-binding ambiguity fails closed without callback.
- [x] Non-JSON, NaN, and over-limit valid callback payloads are returned unchanged but not cached.
- [x] Expected and unexpected callback failures are absent from both local and Redis result caches.
- [x] Cancellation before a valid callback result propagates, invokes no fallback path, and stores no result.
- [x] Different active scope digest cannot replay or conflict with another scope; no-scope compatibility remains exact.
- [x] Every degraded path increments one bounded-stage metric, sets remote health degraded when applicable, and logs only stage plus exception family.

Use monotonic-time assertions with generous scheduling tolerance; do not sleep for full production defaults. Construct a policy with a one-second wait and coordinate owners with `asyncio.Event`.

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q --tb=short \
  tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_manager.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_and_category.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_prometheus_idempotency_metrics.py
```

Expected: callback counts remain at most one under every injected fault.

Commit:

```bash
git add tldw_Server_API/app/core/MCP_unified/tool_execution/idempotency.py \
  tldw_Server_API/app/core/MCP_unified/tool_execution/models.py \
  tldw_Server_API/app/core/MCP_unified/tool_execution/dependencies.py \
  tldw_Server_API/app/core/MCP_unified/tool_execution/runtime.py \
  tldw_Server_API/app/core/MCP_unified/protocol.py \
  tldw_Server_API/app/core/MCP_unified/monitoring/metrics.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_manager.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_and_category.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_prometheus_idempotency_metrics.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_prometheus_idempotency_metrics.py
git commit -m "fix(mcp): make idempotency ownership sticky"
```

## Task 6: Make Success Finalization Cancellation-Safe And Observer-Ordered

**Files:**

- Modify `tldw_Server_API/app/core/MCP_unified/tool_execution/idempotency.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tool_execution/runtime.py`
- Modify `tldw_Server_API/app/core/MCP_unified/server.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_manager.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_tool_hooks.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py`

### Finalizer Ownership

After a valid remote-owner callback result:

1. Canonically serialize synchronously.
2. Decode and install the fresh bounded local replay template synchronously.
3. Create one manager-owned task that writes those exact bytes and token-releases the lock.
4. Keep a strong reference in `self._finalizers` until its done callback removes it.
5. Await it under `asyncio.shield` for at most `finalize_seconds`.
6. If the caller is cancelled, defer propagation until finalization completes or reaches the bound.
7. At the bound, request task cancellation and perform a second bounded drain.
8. If the backend ignores cancellation, keep the task owned, mark remote idempotency degraded, and drain it during `shutdown()`.
9. A late finalizer may only write the committed bytes captured at task creation and may only token-release its own lock.

Do not use an unreferenced `create_task`, and do not allow a timeout handler to construct or publish a different payload.

Wire `await protocol.idempotency.shutdown()` (through a narrow protocol shutdown method or runtime dependency) into `MCPServer.shutdown()` before module registry teardown. Shutdown is best effort and logs only finalizer count and exception family.

### Runtime Observer Ordering

Split runtime work into:

- Admission and first integrity check.
- Owner callback: second complete check, schema-approved key injection from policy, module/breaker invocation, response construction only.
- Idempotency success commit.
- Fresh-execution success observers: module metrics, audit, post hooks, and tool-use reporting.
- Replay reporting only, preserving current no-module/no-post-hook replay behavior.

For idempotent fresh success, observer exceptions and observer cancellation occur after cache commit and can never cause callback re-entry. Best-effort observer failures log only exception family. For module failures, run failure metrics/audit/post-hook/reporting once and re-raise so the manager cannot cache them.

### Required Cancellation And Ordering Tests

- [x] Block Redis write, cancel caller after local commit, release finalizer within bound: cancellation propagates only after exact bytes persist and callback count remains one.
- [x] Redis write exceeds bound but honors cancellation: original success remains locally replayable, finalizer set drains, remote health degrades.
- [x] Redis write ignores cancellation: manager retains a strong reference; later completion cannot replace payload; `shutdown()` drains it.
- [x] Post-hook raises after committed success: returned idempotent result remains success and replayable, callback count one.
- [x] Post-hook is cancelled after committed success: cancellation may propagate, and retry replays the committed result without module invocation.
- [x] Audit, metrics, and reporting failures after commit do not redispatch or alter result.
- [x] Cached replay does not execute module success observers but records one cached tool-use event.
- [x] No finalizer task is pending after normal manager/server shutdown tests.
- [x] Finalization timeout/stuck paths publish only the fixed degraded stage and exception family to metrics/logs and set `remote_degraded`.

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q --tb=short \
  tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_manager.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_tool_hooks.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py
```

Expected: persistence precedes observers, cancellation tests leave no unowned tasks, and retries never dispatch twice.

Commit:

```bash
git add tldw_Server_API/app/core/MCP_unified/tool_execution/idempotency.py \
  tldw_Server_API/app/core/MCP_unified/tool_execution/runtime.py \
  tldw_Server_API/app/core/MCP_unified/server.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_manager.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_tool_hooks.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py
git commit -m "fix(mcp): finalize idempotent successes safely"
```

## Task 7: Complete Expected-Failure Observability And Envelope Hardening

**Files:**

- Modify `tldw_Server_API/app/core/MCP_unified/tool_execution/runtime.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tool_execution/reporting.py`
- Modify `apps/mcp-unified/src/mcp_unified/tool_use_reporting/builders.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_tool_execution_outcomes.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_tool_hooks.py`

### Error Result

Use the safe mapper introduced in Task 2 only after `ExpectedToolFailure` has escaped admission, idempotency, or module execution. Reassert and test this exact sole content item while adding complete observer behavior:

```python
failure_content = {
    "status": "failed",
    "reason_code": failure.reason_code,
    "message": failure.public_message,
}
payload = {
    "content": [{"type": "json", "json": failure_content}],
    "isError": True,
    "module": prepared.module_id or getattr(prepared.module, "name", None),
    "tool": prepared.tool_name,
    "eval": execution_eval,
}
```

The outer bookkeeping may contain only the existing safe `module`, `tool`, and `eval` fields in addition to `content` and `isError`. It must not include raw module payload, exception class, traceback, args, context, HMAC, policy, scope, idempotency key, or internal wrapper.

Place `except asyncio.CancelledError: raise` before every touched broad or `_noncritical_exceptions` handler in runtime execution, admission, metrics, audit, hooks, and reporting. Do not map pre-success cancellation to `ExpectedToolFailure` or the generic tool-execution error.

Before returning it:

- Record module failure metrics only if execution reached the module.
- Audit failure with safe exception family/reason.
- Run failure post hooks with private decoded observer snapshots.
- Record one tool-use event with status `error`, the stable reason code, and correct origin (`failed_before_execution` or `executed`).
- Ensure observer failures cannot replace the error payload.
- Ensure expected failure exceptions never reach `MCPProtocol.process_request`; therefore no new Skills or expected-failure branch belongs in `protocol.py`.

Update the standalone shape-based reporting classifier to recognize an exception carrying the validated expected-failure fields without importing tldw server modules. Sanitize `reason_code` through its existing helper; never read exception text.

### Required Tests

- [x] Domain ignored failure produces JSON-RPC success containing a `tools/call` `isError: true` payload.
- [x] Content contains exactly one JSON object with exactly `status`, `reason_code`, and `message`.
- [x] Fail-closed rate, idempotency in-progress/unavailable, stale prepared call, and counted dependency failure share the envelope.
- [x] Expected failures are absent from local/Redis result caches.
- [x] Ignored failure records metrics/audit/hooks/reporting but leaves breaker counters/state unchanged.
- [x] Counted dependency failure records the same observers and increments/reopens the breaker.
- [x] Unexpected exceptions keep the existing generic JSON-RPC/internal-error path and counted breaker behavior.
- [x] Existing invalid params, permission, approval, governance, actual rate limit, and argument conflict tests remain unchanged.
- [x] Cancellation before a valid result propagates as cancellation and produces neither an error envelope nor a cached result.
- [x] Sentinel secrets embedded in internal exception messages do not occur in result, eval, event, telemetry, audit, or captured logs.

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q --tb=short \
  tldw_Server_API/app/core/MCP_unified/tests/test_tool_execution_outcomes.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_tool_hooks.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_nested_tool_preparation.py
```

Expected: all generic expected failures use the safe tool-error envelope and existing protocol errors are unchanged.

Commit:

```bash
git add tldw_Server_API/app/core/MCP_unified/tool_execution/runtime.py \
  tldw_Server_API/app/core/MCP_unified/tool_execution/reporting.py \
  apps/mcp-unified/src/mcp_unified/tool_use_reporting/builders.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_tool_execution_outcomes.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_tool_hooks.py
git commit -m "feat(mcp): return sanitized expected tool failures"
```

## Task 8: Compatibility, Documentation, And Final Security Review

**Files:**

- Modify `tldw_Server_API/app/core/MCP_unified/README.md`
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py`
- Modify `backlog/tasks/task-2294.3.1 - Harden-generic-MCP-tool-execution-foundations.md` through Backlog MCP

### Documentation

Document:

- Expected tool failures are `isError` tool results, not JSON-RPC request failures.
- `rate_limit_fail_closed` is server-authored tool metadata and changes only admission-backend failure behavior.
- Idempotency is owner-, module-, tool-, key-, arguments-, and optional authenticated active-scope-partitioned.
- Personal/no-scope keys retain compatibility; active scopes use only a SHA-256 digest in keys.
- Successful normalized responses can be retained locally or in Redis for the configured TTL and may contain sensitive model/tool output.
- Failed executions are not result cached; argument binding remains after failure.
- A replay is stale-by-design with respect to mutable downstream content until TTL expiry.
- Persistence degradation preserves caller success but weakens cross-process replay guarantees.
- Contention and finalization limits and their environment variables.

### Static Contract Checks

Add/extend AST checks proving:

- `protocol.py` does not define `IdempotencyManager`.
- `protocol.py` contains no Skills/model/provider expected-failure branch.
- `tool_execution` does not import the protocol facade.
- Runtime does not read `metadata.category`, `uses_network`, `rate_limit_fail_closed`, `inputSchema`, idempotency TTL/limits, or effect after preparation.
- Runtime has no `default=str` serialization.
- Touched breaker/runtime/idempotency files do not interpolate `str(exception)`, `repr(exception)`, or Loguru `exception=`/`exc_info=` with raw execution exceptions.
- Compatibility exports remain available.

### Final Verification

- [ ] Update all five stage statuses in this plan as implementation progresses.
- [ ] Update `TASK-2294.3.1` notes with each commit and verification result through Backlog MCP.
- [ ] Review every acceptance criterion 1-17 against a named test.
- [ ] Search for placeholders and temporary skips in touched files.
- [ ] Run focused tests.
- [ ] Run the MCP Unified suite.
- [ ] Run Ruff, compileall, Bandit, and diff checks.
- [ ] Review `git diff --stat` and `git diff` for unrelated edits.
- [ ] Update Backlog acceptance criteria, Definition of Done, modified files, final summary, and PR link when available.
- [ ] After all implementation stages are complete and their evidence is recorded in Backlog, remove only this task's implementation-plan file as required by the repository workflow; remove its Backlog documentation link in the same finalization update and do not remove any other plan.

Commands:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q --tb=short \
  tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_tool_execution_outcomes.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_prepared_execution_integrity.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_manager.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_and_category.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_concurrency_and_breaker.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_rate_limit_categories.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_nested_tool_preparation.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_tool_hooks.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py \
  tldw_Server_API/tests/MCP_unified/test_phase3_3_small_core_sanitizers.py \
  tldw_Server_API/tests/Infrastructure/test_circuit_breaker.py
```

Expected: all focused tests pass.

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q --tb=short tldw_Server_API/app/core/MCP_unified/tests
```

Expected: MCP Unified suite passes. Record any pre-existing unrelated failure with exact test and output; do not silently skip it.

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m ruff check \
  tldw_Server_API/app/core/MCP_unified/execution_outcomes.py \
  tldw_Server_API/app/core/MCP_unified/protocol.py \
  tldw_Server_API/app/core/MCP_unified/protocol_types.py \
  tldw_Server_API/app/core/MCP_unified/config.py \
  tldw_Server_API/app/core/MCP_unified/server.py \
  tldw_Server_API/app/core/MCP_unified/monitoring/metrics.py \
  tldw_Server_API/app/core/MCP_unified/modules/base.py \
  tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py \
  tldw_Server_API/app/core/MCP_unified/tool_execution \
  apps/mcp-unified/src/mcp_unified/tool_use_reporting/builders.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_tool_execution_outcomes.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_prepared_execution_integrity.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_manager.py
python -m compileall -q \
  tldw_Server_API/app/core/MCP_unified \
  apps/mcp-unified/src/mcp_unified/tool_use_reporting
python -m bandit -r \
  tldw_Server_API/app/core/MCP_unified/execution_outcomes.py \
  tldw_Server_API/app/core/MCP_unified/protocol.py \
  tldw_Server_API/app/core/MCP_unified/protocol_types.py \
  tldw_Server_API/app/core/MCP_unified/config.py \
  tldw_Server_API/app/core/MCP_unified/server.py \
  tldw_Server_API/app/core/MCP_unified/monitoring/metrics.py \
  tldw_Server_API/app/core/MCP_unified/modules/base.py \
  tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py \
  tldw_Server_API/app/core/MCP_unified/tool_execution \
  apps/mcp-unified/src/mcp_unified/tool_use_reporting/builders.py \
  -f json -o /tmp/bandit_task_2294_3_1.json
git diff --check
```

Expected: Ruff, compileall, Bandit, and diff checks complete with no new findings. Inspect `/tmp/bandit_task_2294_3_1.json` and record the finding counts in Backlog.

Search gate:

```bash
rg -n "TODO|FIXME|XXX|pass[[:space:]]*#|default=str|str\((exc|e|error)\)|repr\((exc|e|error)\)" \
  tldw_Server_API/app/core/MCP_unified/execution_outcomes.py \
  tldw_Server_API/app/core/MCP_unified/protocol.py \
  tldw_Server_API/app/core/MCP_unified/protocol_types.py \
  tldw_Server_API/app/core/MCP_unified/modules/base.py \
  tldw_Server_API/app/core/MCP_unified/tool_execution
```

Expected: no new placeholders, permissive execution serialization, or raw exception logging in touched execution paths. Review legitimate compatibility parsing uses manually rather than suppressing the gate.

Final commit:

```bash
git rm Docs/superpowers/plans/2026-07-25-mcp-tool-execution-foundations-implementation-plan.md
git add tldw_Server_API/app/core/MCP_unified/README.md \
  tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py \
  "backlog/tasks/task-2294.3.1 - Harden-generic-MCP-tool-execution-foundations.md"
git commit -m "docs(mcp): document hardened tool execution"
```

## Acceptance-Criteria Traceability

| AC | Primary implementation task | Primary verification |
|---|---|---|
| 1 | Task 2 | `test_tool_execution_outcomes.py` reason catalog validation |
| 2 | Task 7 | Exact `isError` protocol-envelope tests |
| 3 | Tasks 2 and 7 | Closed/half-open metrics/audit/hooks/reporting matrix |
| 4 | Tasks 5 and 7 | Local/Redis expected-failure cache absence |
| 5 | Task 2 | Fallback/injected closed/half-open parity matrix |
| 6 | Task 1 | Extraction AST and compatibility re-export tests |
| 7 | Task 4 | Flagged/unflagged admission backend tests |
| 8 | Task 5 | Local/Redis contention deadline tests |
| 9 | Task 5 | Ambiguous lock and post-owner Redis fault injection |
| 10 | Tasks 3 and 5 | Canonical JSON and mutation-isolation tests |
| 11 | Task 5 | Serialization/size/persistence degraded-success tests |
| 12 | Task 6 | Cache-before-observer and post-hook fault tests |
| 13 | Task 8 | Focused/MCP-wide/Ruff/compile/Bandit gates |
| 14 | Tasks 2, 4, 5, and 8 | Sentinel-secret log capture and AST/search gate |
| 15 | Tasks 3 and 4 | Policy/HMAC/snapshot/live-binding/two-check tests |
| 16 | Task 6 | Cancellation/finalizer/late-write/shutdown tests |
| 17 | Tasks 3 and 5 | Explicit scope digest/key/replay-domain tests |

## Implementation Review Checkpoints

After each task:

1. Run the task's focused tests from a clean shell with the project virtual environment active.
2. Review the diff against only the task's listed invariants and files.
3. Confirm failure messages and logs are constant or exception-family-only.
4. Confirm no security decision reads a mutable observer dictionary.
5. Confirm no newly introduced await exists between a valid callback result and the synchronous local canonical commit.
6. Update Backlog notes and the corresponding stage status.
7. Commit before starting the next task.

Tasks are intentionally sequential. Do not implement Tasks 3-7 in parallel because they all change `PreparedToolCall`, runtime sequencing, and idempotency contracts. Reviewer agents may run after each implementation commit, but implementation ownership must remain single-writer for these files.
