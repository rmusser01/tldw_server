# MCP Unified Multi-Revision Stdio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `mcp-unified` 0.2.0 with a reusable, bounded stdio runtime that implements MCP `2026-07-28` and the four approved legacy revisions without changing existing HTTP, WebSocket, or compatibility-stdio behavior.

**Architecture:** Add a strict protocol stack beside the existing `GatewayStdioServer`: immutable revision profiles and public contracts feed bounded schema validation, descriptor/result projection, and cursor pagination; `GatewayProtocolConnection` owns lifecycle, in-flight work, rate limits, cancellation, and serialized output; `GatewayProtocolStdioServer` owns binary stream framing and shutdown. Existing `GatewayRuntime`, `GatewayStdioServer`, `handle_stdio_line`, FastAPI, and WebSocket paths remain on their current dispatch implementation.

**Tech Stack:** Python 3.10-3.13, asyncio, multiprocessing `spawn`, Pydantic, `jsonschema>=4.23,<5`, pytest/pytest-asyncio, Bandit, setuptools/build/twine.

## Global Constraints

- Supported revisions, newest first: `2026-07-28`, `2025-11-25`, `2025-06-18`, `2025-03-26`, `2024-11-05`.
- `2026-07-28` is stateless and rejects batches; legacy revisions initialize first; only initialized `2025-03-26` accepts request batches.
- Modern tool `inputSchema` remains object-rooted; modern `outputSchema` and `structuredContent` may use any finite JSON root; incompatible legacy projection is deterministic text-only and never invents a wrapper.
- `GatewayRuntime`, `GatewayStdioServer`, `handle_stdio_line`, FastAPI, and WebSocket behavior are compatibility surfaces and must not delegate to the new stateful connection.
- Runtime data, permissions, policy, audit, and privacy remain application-owned; protocol diagnostics remain payload-free.
- Exact `GatewayLimits` defaults: input line `1_048_576`, output line `1_048_576`, result `786_432`, JSON depth `64`, in-flight `16`, catalog page default `50`, catalog page max `100`, catalog items `10_000`, batch items `100`, requests/minute `600`, burst `32`, schema bytes `262_144`, schema depth `32`, subschemas `1_024`, refs `256`, pattern chars `4_096`, validation processes `4`, validation timeout `1.0s`, graceful shutdown `5.0s`.
- Schema compilation and validation run in fresh spawned workers. Every success, validation failure, crash, timeout, cancellation, and shutdown path reaps the child before releasing its concurrency permit.
- Modern cache hints are exactly `ttlMs: 0` and `cacheScope: "private"` on discovery, catalog pages, and resource reads; legacy responses omit them.
- Modern results require `resultType: "complete"`; the server never generates `input_required`; legacy projections omit `resultType`.
- Tool execution failures use result metadata key `io.github.rmusser01.mcp-unified/error` with bounded `reasonCode` and `kind`.
- Normative test fixtures are pinned to `modelcontextprotocol/modelcontextprotocol` tag `2026-07-28`, commit `5f5440bb26a62e2cf3440b92da5a667efa03b267`, with source URLs, SHA-256 checksums, and upstream Apache-2.0 notice recorded in a fixture manifest.
- PyPI currently contains only `mcp-unified` `0.1.1`; `0.2.0` remains the intended candidate and must be rechecked immediately before release.
- ADR required: yes. Existing ADR: `Docs/ADR/033-mcp-unified-stdio-contract-hardening.md`; no new ADR is needed because this plan directly implements it.

## File Responsibility Map

- `apps/mcp-unified/src/mcp_unified/gateway/runtime.py`: compatible legacy runtime plus additive JSON aliases, strict core/template protocols, and enriched request context.
- `apps/mcp-unified/src/mcp_unified/gateway/protocol_profiles.py`: sole immutable revision table and version constants.
- `apps/mcp-unified/src/mcp_unified/gateway/protocol_limits.py`: immutable limit validation and bounded JSON/serialization helpers.
- `apps/mcp-unified/src/mcp_unified/gateway/protocol_errors.py`: safe application exceptions and strict wire-error allowlists.
- `apps/mcp-unified/src/mcp_unified/gateway/protocol_cancellation.py`: thread-safe per-request cancellation token.
- `apps/mcp-unified/src/mcp_unified/gateway/protocol_validation.py`: schema preflight and disposable spawned validation workers.
- `apps/mcp-unified/src/mcp_unified/gateway/protocol_projection.py`: descriptor normalization plus revision-specific capability/result projection.
- `apps/mcp-unified/src/mcp_unified/gateway/protocol_pagination.py`: deterministic catalog sorting, fingerprints, authenticated cursors, and pages.
- `apps/mcp-unified/src/mcp_unified/gateway/protocol_connection.py`: strict JSON-RPC lifecycle, dispatch, batching, rate/in-flight accounting, cancellation, and serialized responses.
- `apps/mcp-unified/src/mcp_unified/gateway/protocol_stdio.py`: strict binary stream protocols/adapters, reader loop, shutdown, and `serve_stdio`.
- `apps/mcp-unified/src/mcp_unified/gateway/__init__.py`: additive public exports only.
- `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_contracts.py`: public constants/types/errors/limits/compatibility tests.
- `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_validation.py`: schema bounds, dialect, worker lifecycle, and adversarial-regex tests.
- `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_projection.py`: descriptor/result/pagination/profile vectors.
- `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_connection.py`: lifecycle, methods, IDs, batches, rate/in-flight, and cancellation vectors.
- `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_stdio.py`: binary framing, concurrent reading, output serialization, EOF/cancellation, and native/fallback adapters.
- `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_artifact_consumer.py`: installed wheel/sdist downstream-consumer contract.
- `tldw_Server_API/app/core/MCP_unified/tests/fixtures/mcp_protocol/`: pinned official schemas/examples plus provenance manifest.
- `apps/mcp-unified/{README.md,USER_GUIDE.md}` and packaged copies under `src/mcp_unified/`: public API, version matrix, privacy/limits, and embed examples.
- `apps/mcp-unified/pyproject.toml`, root `pyproject.toml`, `package_metadata.py`, RC/publish workflows and helpers: dependency, artifact, platform, and release gates.

---

### Task 1: Public Contracts, Profiles, Limits, Cancellation, and Safe Errors

**Files:**
- Create: `apps/mcp-unified/src/mcp_unified/gateway/protocol_profiles.py`
- Create: `apps/mcp-unified/src/mcp_unified/gateway/protocol_limits.py`
- Create: `apps/mcp-unified/src/mcp_unified/gateway/protocol_errors.py`
- Create: `apps/mcp-unified/src/mcp_unified/gateway/protocol_cancellation.py`
- Modify: `apps/mcp-unified/src/mcp_unified/gateway/runtime.py`
- Modify: `apps/mcp-unified/src/mcp_unified/gateway/__init__.py`
- Modify: `apps/mcp-unified/pyproject.toml`
- Modify: `pyproject.toml`
- Modify: `apps/mcp-unified/src/mcp_unified/package_metadata.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_contracts.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`

**Interfaces:**
- Consumes: existing `GatewayRuntime` and `GatewayRequestContext` compatibility contract.
- Produces: `GatewayProtocolProfile`, five version constants, `GatewayLimits`, `GatewayCancellationToken`, `GatewayApplicationError` subclasses, `GatewayJSONValue`, `GatewayCoreRuntime`, and `GatewayResourceTemplateRuntime` for all later tasks.

- [ ] **Step 1: Write failing public-contract tests**

  Add literal tests that prove:
  - the five constants are ordered exactly as specified;
  - every profile's era/lifecycle/batch/schema/cache flags match the matrix;
  - `GatewayLimits()` exposes every exact default and rejects booleans, out-of-range values, and invalid cross-field relationships;
  - cancellation changes state once, bounds the reason to 128 code points, wakes `wait()`, and raises `asyncio.CancelledError`;
  - integer `1` and string `"1"` remain unchanged in `GatewayRequestContext`;
  - a minimal `GatewayCoreRuntime` need not implement module aliases;
  - safe application errors accept only the documented message/code/kind/limit shapes;
  - `GatewayRuntime`, `GatewayStdioServer`, and `handle_stdio_line` retain their old signatures and behavior;
  - standalone wheel metadata declares `jsonschema>=4.23,<5` directly.

  Representative assertions:

  ```python
  assert SUPPORTED_PROTOCOL_VERSIONS == (
      "2026-07-28",
      "2025-11-25",
      "2025-06-18",
      "2025-03-26",
      "2024-11-05",
  )
  assert GatewayLimits().default_catalog_page_size == 50
  with pytest.raises(ValueError):
      GatewayLimits(max_in_flight=True)
  assert GatewayRequestContext(request_id=1).request_id == 1
  assert GatewayRequestContext(request_id="1").request_id == "1"
  ```

- [ ] **Step 2: Run the focused tests and verify RED**

  Run:

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -q
  ```

  Expected: failures identify missing public protocol modules/exports, the old string-only request context annotation, and absent standalone `jsonschema` metadata—not unrelated import or fixture errors.

- [ ] **Step 3: Implement immutable profiles and additive runtime types**

  Implement one frozen profile table keyed by exact revision and keep all version decisions there:

  ```python
  @dataclass(frozen=True, slots=True)
  class GatewayProtocolProfile:
      version: str
      era: Literal["modern", "legacy"]
      requires_initialize: bool
      accepts_batches: bool
      requires_result_type: bool
      cache_hints: bool
      supports_titles: bool
      supports_icons: bool
      supports_resource_links: bool
      structured_content_mode: Literal["any", "object", "none"]
      missing_resource_code: int
      schema_dialect: str

  PROTOCOL_PROFILES = MappingProxyType({
      "2026-07-28": GatewayProtocolProfile(
          version="2026-07-28",
          era="modern",
          requires_initialize=False,
          accepts_batches=False,
          requires_result_type=True,
          cache_hints=True,
          supports_titles=True,
          supports_icons=True,
          supports_resource_links=True,
          structured_content_mode="any",
          missing_resource_code=-32602,
          schema_dialect="https://json-schema.org/draft/2020-12/schema",
      ),
      "2025-11-25": GatewayProtocolProfile(
          version="2025-11-25",
          era="legacy",
          requires_initialize=True,
          accepts_batches=False,
          requires_result_type=False,
          cache_hints=False,
          supports_titles=True,
          supports_icons=True,
          supports_resource_links=True,
          structured_content_mode="object",
          missing_resource_code=-32002,
          schema_dialect="https://json-schema.org/draft/2020-12/schema",
      ),
      "2025-06-18": GatewayProtocolProfile(
          version="2025-06-18",
          era="legacy",
          requires_initialize=True,
          accepts_batches=False,
          requires_result_type=False,
          cache_hints=False,
          supports_titles=True,
          supports_icons=False,
          supports_resource_links=True,
          structured_content_mode="object",
          missing_resource_code=-32002,
          schema_dialect="http://json-schema.org/draft-07/schema#",
      ),
      "2025-03-26": GatewayProtocolProfile(
          version="2025-03-26",
          era="legacy",
          requires_initialize=True,
          accepts_batches=True,
          requires_result_type=False,
          cache_hints=False,
          supports_titles=False,
          supports_icons=False,
          supports_resource_links=False,
          structured_content_mode="none",
          missing_resource_code=-32002,
          schema_dialect="http://json-schema.org/draft-07/schema#",
      ),
      "2024-11-05": GatewayProtocolProfile(
          version="2024-11-05",
          era="legacy",
          requires_initialize=True,
          accepts_batches=False,
          requires_result_type=False,
          cache_hints=False,
          supports_titles=False,
          supports_icons=False,
          supports_resource_links=False,
          structured_content_mode="none",
          missing_resource_code=-32002,
          schema_dialect="http://json-schema.org/draft-07/schema#",
      ),
  })
  ```

  In `runtime.py`, add recursive JSON aliases, widen `request_id`, add compatibility-default context fields, and define runtime-checkable narrow core/template protocols. Do not remove or reorder existing `GatewayRuntime` members.

- [ ] **Step 4: Implement limits, cancellation, and errors**

  `GatewayLimits.__post_init__` must use explicit integer/finite-number validators and validate all cross-field relationships. `GatewayCancellationToken` uses a `threading.Lock` plus lazily registered asyncio waiters so `cancel()` is safe from worker threads. Error constructors validate locally and expose only `public_message`, `reason_code`, `kind`, and optional `limit_bytes`.

  ```python
  class GatewayToolExecutionError(GatewayApplicationError):
      def __init__(self, public_message: str, *, reason_code: str) -> None:
          super().__init__(public_message, reason_code=reason_code, kind="tool")

  class GatewayResultTooLarge(GatewayApplicationError):
      def __init__(self, *, limit_bytes: int) -> None:
          super().__init__(
              "Application result exceeds the configured limit",
              reason_code="result_too_large",
              kind="application",
          )
          self.limit_bytes = _positive_int(limit_bytes, "limit_bytes")
  ```

- [ ] **Step 5: Declare and expose the direct validator dependency**

  Add `jsonschema>=4.23,<5` to standalone base dependencies, mirror the compatible bound in the root project dependency, include `jsonschema` in `PROJECT_DEPENDENCIES`, and export every approved public symbol from `mcp_unified.gateway` without eagerly importing FastAPI.

- [ ] **Step 6: Run focused and legacy compatibility tests and verify GREEN**

  Run the Step 2 command plus:

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime_adapter.py -q
  ```

  Expected: all pass; existing runtime adapters remain structurally compatible.

- [ ] **Step 7: Commit Task 1**

  ```bash
  git add apps/mcp-unified/src/mcp_unified/gateway apps/mcp-unified/src/mcp_unified/package_metadata.py apps/mcp-unified/pyproject.toml pyproject.toml tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
  git commit -m "feat(mcp): add strict protocol contracts and limits"
  ```

---

### Task 2: Schema Validation, Descriptor/Result Projection, and Pagination

**Files:**
- Create: `apps/mcp-unified/src/mcp_unified/gateway/protocol_validation.py`
- Create: `apps/mcp-unified/src/mcp_unified/gateway/protocol_projection.py`
- Create: `apps/mcp-unified/src/mcp_unified/gateway/protocol_pagination.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_validation.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_projection.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/fixtures/mcp_protocol/manifest.json`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/fixtures/mcp_protocol/2026-07-28/schema.json`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/fixtures/mcp_protocol/2025-11-25/schema.json`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/fixtures/mcp_protocol/2025-06-18/schema.json`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/fixtures/mcp_protocol/2025-03-26/schema.json`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/fixtures/mcp_protocol/2024-11-05/schema.json`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/fixtures/mcp_protocol/NOTICE.md`

**Interfaces:**
- Consumes: Task 1 profiles, limits, JSON aliases, and application errors.
- Produces: `GatewaySchemaValidationManager.validate(schema, instance)`, `project_descriptor`, `project_tool_result`, `project_resource_result`, `project_prompt_result`, and `GatewayCatalogPaginator.page` for connection dispatch.

- [ ] **Step 1: Pin official schema fixtures and provenance**

  Copy the five dated `schema/<revision>/schema.json` files from official commit `5f5440bb26a62e2cf3440b92da5a667efa03b267`. Record each raw GitHub URL and SHA-256 in `manifest.json`; record that the fixtures are test-only snapshots under the upstream Apache-2.0 license in `NOTICE.md`. Add a test that loads the manifest, hashes each file, and validates representative literal request/result vectors locally without network access.

- [ ] **Step 2: Write failing schema-boundary and worker-lifecycle tests**

  Test valid Draft 2020-12 object/array/scalar/null instances and literal failures for external `$ref`, schema bytes/depth/subschemas/refs/pattern length, instance depth, invalid dialect, and invalid tool input/output. Use a real catastrophic-backtracking schema such as `^(a+)+$` with a non-matching bounded string; prove the call returns `schema_validation_timeout` and a subsequent simple validation succeeds. Instrument real spawned process handles at the manager boundary to prove permits are not released before join/reap on success, validation failure, crash, timeout, cancellation, and manager shutdown.

- [ ] **Step 3: Run validation tests and verify RED**

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_validation.py -q
  ```

  Expected: missing validation manager/functions, while fixture provenance/hash tests already pass.

- [ ] **Step 4: Implement bounded preflight and disposable workers**

  Parent-side preflight serializes with `allow_nan=False`, walks JSON iteratively, counts schema nodes/refs/pattern characters, and rejects network/unresolved external references before spawn. A module-level worker entrypoint constructs `Draft202012Validator`, returns only a bounded success/error tuple through a one-way pipe, and performs no network resolution.

  The concrete manager stores `asyncio.Semaphore(limits.max_schema_validation_processes)` and a set of live `multiprocessing.Process` objects. Its `validate(schema, instance)` method performs preflight, acquires the semaphore, spawns exactly one worker, reads a bounded verdict under `schema_validation_timeout_seconds`, and calls one shared cleanup path. That cleanup terminates then kills when needed, joins the child, removes it from the live set, and only then releases the semaphore. `close()` rejects new work and applies the same cleanup to every live child. No executor thread may outlive `close()`.

  ```python
  SchemaWorkerVerdict: TypeAlias = tuple[Literal["ok", "invalid", "internal"], str]
  _SCHEMA_WORKER_MAX_VERDICT_BYTES = 4_096
  ```

- [ ] **Step 5: Write failing projection and pagination tests**

  Cover all revision profiles with literal descriptors/results: valid names/URIs/roles/content blocks, unsupported-field stripping, reserved `_meta` overwrite, modern arbitrary-root output schemas/results, legacy object-only structured content, deterministic JSON text fallback, modern cache/result fields, legacy absence, typed tool error metadata, version-correct resource-not-found errors, empty catalogs, duplicate identity rejection, stable sorting, first page size 50, cross-method cursor rejection, tamper rejection, and changed-catalog fingerprint rejection.

- [ ] **Step 6: Run projection tests and verify RED**

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_projection.py -q
  ```

  Expected: missing projector/paginator imports and no false fixture failures.

- [ ] **Step 7: Implement projection and authenticated cursor pagination**

  Normalize once, validate before publication, and project from profile flags. Use deterministic UTF-8 JSON (`sort_keys=True`, compact separators, `allow_nan=False`) for text fallback and catalog fingerprints. Generate a connection-local 32-byte cursor secret with `secrets.token_bytes`; encode a bounded payload containing method, version, offset, page size, and SHA-256 catalog fingerprint; authenticate with HMAC-SHA256 and `compare_digest`.

  ```python
  @dataclass(frozen=True, slots=True)
  class GatewayCatalogPage:
      items: list[dict[str, GatewayJSONValue]]
      next_cursor: str | None
  ```

  `GatewayCatalogPaginator.page(*, method, profile, items, cursor)` performs concrete identity extraction, duplicate rejection, sorting, max-item enforcement, fingerprint verification, offset slicing, and cursor emission before returning `GatewayCatalogPage`.

- [ ] **Step 8: Run Tasks 1-2 tests and verify GREEN**

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_validation.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_projection.py -q
  ```

- [ ] **Step 9: Commit Task 2**

  ```bash
  git add apps/mcp-unified/src/mcp_unified/gateway/protocol_validation.py apps/mcp-unified/src/mcp_unified/gateway/protocol_projection.py apps/mcp-unified/src/mcp_unified/gateway/protocol_pagination.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_validation.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_projection.py tldw_Server_API/app/core/MCP_unified/tests/fixtures/mcp_protocol
  git commit -m "feat(mcp): validate and project revisioned protocol data"
  ```

---

### Task 3: Revision-Aware Protocol Connection and Dispatch

**Files:**
- Create: `apps/mcp-unified/src/mcp_unified/gateway/protocol_connection.py`
- Modify: `apps/mcp-unified/src/mcp_unified/gateway/__init__.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_connection.py`

**Interfaces:**
- Consumes: Task 1 public contracts and Task 2 validators/projectors/paginator.
- Produces: `GatewayProtocolConnection.receive(payload)`, `wait_for_idle()`, and `shutdown()` for the stdio server.

- [ ] **Step 1: Write failing lifecycle, method, and ID tests**

  Use a real in-memory core runtime and writer. Cover modern discovery as the first request; modern tool/resource/template/prompt operations; empty catalogs; required modern `_meta`; `-32022` with all five versions; legacy initialization/fallback/initialized notification; pre-initialize rejection; second initialize; era mixing; standard capabilities; no module aliases; string/integer ID distinction; null/boolean rejection; duplicate active IDs; payload-free errors; and reserved server metadata authority.

- [ ] **Step 2: Write failing batching, rate, and cancellation tests**

  Cover initialize-in-batch rejection, empty batches, four-profile batch rejection, post-initialize `2025-03-26` request/mixed/notification-only batches, max batch size before task creation, max in-flight before runtime dispatch, token-bucket burst/minute behavior using an injected monotonic clock, cancellation before dispatch/during work/at writer lock, late-result suppression, EOF-style shutdown, and no server-to-client requests.

- [ ] **Step 3: Run connection tests and verify RED**

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_connection.py -q
  ```

  Expected: missing `GatewayProtocolConnection`; the test runtime and memory writer construct successfully.

- [ ] **Step 4: Implement lifecycle classification and authoritative contexts**

  `receive()` accepts decoded bounded JSON, classifies modern by required reserved `_meta` and legacy by initialize state, rejects ambiguous/mixed eras, and schedules only admitted requests. Use typed keys `(type(request_id), request_id)` internally so `1` and `"1"` never collide. Construct `GatewayRequestContext` with authoritative version/era/client capabilities/cancellation and stripped caller metadata.

  ```python
  class GatewayProtocolConnection:
      async def receive(self, payload: GatewayJSONValue) -> None:
          """Validate, admit, schedule, and eventually serialize one line value."""

      async def wait_for_idle(self) -> None:
          """Wait until every admitted request and validation child is reaped."""

      async def shutdown(self) -> None:
          """Reject new work, cancel tracked requests, and drain bounded cleanup."""
  ```

- [ ] **Step 5: Implement core dispatch and revision projection**

  Dispatch only `ping`, discovery, tools, resources, resource templates, prompts, initialize/initialized, and cancellation. Validate tool arguments against the published descriptor before calling the runtime; validate/project results afterward. Map `GatewayToolExecutionError` to an `isError: true` result and all other safe errors to their exact profile code. Unknown methods/tools/prompts and malformed results use bounded generic protocol errors.

- [ ] **Step 6: Implement batching, admission, rate limits, and cancellation races**

  Reject arrays before creating element tasks except in initialized `2025-03-26`; batch responses preserve request order while concurrent runtime work remains bounded. Check cancellation under the same serialized writer lock immediately before output. A notification never emits a response, and a notification-only batch emits no line.

- [ ] **Step 7: Run connection plus Tasks 1-2 tests and verify GREEN**

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_validation.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_projection.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_connection.py -q
  ```

- [ ] **Step 8: Commit Task 3**

  ```bash
  git add apps/mcp-unified/src/mcp_unified/gateway/protocol_connection.py apps/mcp-unified/src/mcp_unified/gateway/__init__.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_connection.py
  git commit -m "feat(mcp): add revision-aware protocol connection"
  ```

---

### Task 4: Strict Portable Stdio Engine

**Files:**
- Create: `apps/mcp-unified/src/mcp_unified/gateway/protocol_stdio.py`
- Create: `apps/mcp-unified/src/mcp_unified/gateway/protocol_stdio_adapters.py`
- Modify: `apps/mcp-unified/src/mcp_unified/gateway/__init__.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_stdio.py`
- Modify: `.github/workflows/mcp-unified-rc.yml`

**Interfaces:**
- Consumes: `GatewayProtocolConnection` from Task 3.
- Produces: `GatewayAsyncByteReader`, `GatewayAsyncByteWriter`, `GatewayProtocolStdioServer`, and `serve_stdio` public APIs.

- [ ] **Step 1: Write failing injected-stream tests**

  Implement test-owned async byte reader/writer fakes that exercise real server behavior. Assert text streams are rejected before reading; injected streams remain open; blank lines are ignored; each response is one bounded newline-terminated JSON value; oversized/incomplete input fails safely; output overflow emits a safe error when possible; writer drain is bounded by shutdown; EOF waits/cancels/reaps admitted work; serving-task cancellation re-raises after cleanup; and individual protocol errors still return exit code `0`.

- [ ] **Step 2: Write failing concurrency/native/fallback tests**

  Prove the reader admits a second request while the first runtime call is blocked, output writes never interleave, cancellation suppresses a result at the final writer race, and fatal stream failures return `1`. Exercise the native async pipe adapter on POSIX and force the dedicated-thread binary fallback through an injected adapter selector. The fallback must bound reads, serialize writes, propagate cancellation, and join its threads during shutdown.

- [ ] **Step 3: Run stdio tests and verify RED**

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_stdio.py -q
  ```

  Expected: missing strict stdio classes/functions, while compatibility stdio imports remain available.

- [ ] **Step 4: Implement binary adapters and strict server**

  `GatewayProtocolStdioServer` owns one connection and a read loop. Omitted streams wrap `sys.stdin.buffer`/`sys.stdout.buffer` without replacement or close. Native POSIX registration is preferred; unsupported loops select bounded `asyncio.to_thread` calls around binary read/write locks. Limit enforcement happens before JSON decoding and before `write()`.

  ```python
  async def serve_stdio(
      runtime: GatewayCoreRuntime,
      *,
      input_stream: GatewayAsyncByteReader | None = None,
      output_stream: GatewayAsyncByteWriter | None = None,
      limits: GatewayLimits = GatewayLimits(),
      metadata: Mapping[str, Any] | None = None,
  ) -> int:
      server = GatewayProtocolStdioServer(
          runtime=runtime,
          input_stream=input_stream,
          output_stream=output_stream,
          limits=limits,
          metadata=metadata,
      )
      return await server.serve()
  ```

- [ ] **Step 5: Preserve compatibility stdio and add platform CI**

  Do not modify `GatewayStdioServer.handle_line` or `handle_stdio_line` semantics. Add their characterization tests to this suite. Extend the MCP RC workflow with a small `ubuntu-latest`/`windows-latest` matrix that installs the standalone base+dev package and runs protocol contracts, connection, and stdio tests; the Windows job is the real fallback proof behind the OS-independent classifier.

- [ ] **Step 6: Run strict and compatibility suites and verify GREEN**

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_stdio.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_connection.py tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -q
  ```

- [ ] **Step 7: Commit Task 4**

  ```bash
  git add apps/mcp-unified/src/mcp_unified/gateway/protocol_stdio.py apps/mcp-unified/src/mcp_unified/gateway/protocol_stdio_adapters.py apps/mcp-unified/src/mcp_unified/gateway/__init__.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_stdio.py .github/workflows/mcp-unified-rc.yml
  git commit -m "feat(mcp): serve strict protocol over portable stdio"
  ```

  Accepted Task 4 split: the portable native/thread fallback adapters were
  kept private in `protocol_stdio_adapters.py` rather than widening the public
  gateway surface. The accepted Task 4 commit sequence is `fd263514a2`,
  `e0ae834c95`, and `ac8408c5fd`.

---

### Task 5: Artifact Consumer, Documentation, Security, and Release Gate

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_artifact_consumer.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/mcp_unified_artifact_test_utils.py`
- Create: `Helper_Scripts/Testing-related/mcp_official_sdk_stdio_smoke.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_connection.py`
- Modify: `Helper_Scripts/mcp_unified_rc.py`
- Modify: `.github/workflows/mcp-unified-rc.yml`
- Modify: `.github/workflows/mcp-unified-publish.yml`
- Modify: `.github/license-first-paths.json`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_stdio.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`
- Modify: `apps/mcp-unified/pyproject.toml`
- Modify: `apps/mcp-unified/README.md`
- Modify: `apps/mcp-unified/USER_GUIDE.md`
- Modify: `apps/mcp-unified/src/mcp_unified/README.md`
- Modify: `apps/mcp-unified/src/mcp_unified/USER_GUIDE.md`
- Modify: `apps/mcp-unified/src/mcp_unified/package_metadata.py`
- Modify: `apps/mcp-unified/src/mcp_unified/gateway/protocol_connection.py`
- Modify: `apps/mcp-unified/src/mcp_unified/gateway/protocol_validation.py`
- Modify: `backlog/tasks/task-13009 - Implement-MCP-Unified-multi-revision-stdio-protocol.md`

**Interfaces:**
- Consumes: complete strict public API from Tasks 1-4 and existing package RC/publish tooling.
- Produces: installable 0.2.0 artifact evidence and the merge/publish handoff that gates Chatbook TASK-2512.

- [ ] **Step 1: Write the failing installed-artifact consumer test**

  Extend the existing build helper to install each locally built wheel and sdist into a clean temporary virtual environment. Run a synthetic downstream module that imports only public `mcp_unified.gateway` names, supplies tools/resources/templates/prompts (including empty catalogs), distinguishes integer/string IDs, returns object/array/scalar/null results, uses `max_in_flight=1`, paginates, cancels, raises typed errors, and runs `serve_stdio` with injected binary streams. Assert the subprocess exits `0` and emits only its literal success marker.

- [ ] **Step 2: Run artifact test and verify RED**

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_artifact_consumer.py -q
  ```

  Expected: failure identifies missing artifact-gate inclusion or public API behavior, not a checkout-source import leak.

- [ ] **Step 3: Update RC/publish dependency and artifact gates**

  Make the RC helper assert `jsonschema` is a base dependency in wheel and sdist metadata, run the five protocol suites from the installed artifact, and record the normative fixture commit/checksums. Keep `wheel>=0.41.0` in the development extra so the no-isolation wheel-and-sdist artifact gate has an explicitly declared, consistently available wheel builder. Confine fixture staging to the manifest, NOTICE, and five exact pinned regular non-symlink schema files; recursively sanitize the complete evidence payload. Every package-only RC invocation must use the explicit mirrored tree outside the checkout package hierarchy, including the downstream artifact consumer. Install the exact CI-only official Tier 1 Python SDK pin `mcp==2.0.0` in each wheel and sdist environment and run a bounded stdio smoke covering automatic `2026-07-28` negotiation, tool discovery, and a tool call. Record the official SDK index, release tag, and tag commit `6f69a37`; do not add the SDK to runtime dependencies or claim the URL-oriented conformance server harness passed. Expand the protected portable-stdio workflow to exactly five jobs (Ubuntu Python 3.10-3.13 and Windows Python 3.11), with each job building and testing both wheel and sdist through the isolated portable gate. Keep the pull-request trigger and license-first admission manifest in exact parity for all protocol suites, fixtures, release helpers, package-status boundary tests, and release workflows. Update publish workflow setup to install `jsonschema>=4.23,<5`; keep live upload behind the existing protected environment, `MCP_UNIFIED_PUBLISH` confirmation, and `MCP_UNIFIED_ALLOW_PUBLISH=1` guard.

  Task 4 divergence note: its accepted two-OS Python 3.11 portable-stdio
  coverage proved the transport implementation. Task 5 intentionally widens
  that protected gate to the bounded five-job OS/Python artifact matrix needed
  for release acceptance; this is release verification, not a protocol
  production change.

  Final branch-review divergence note: clean protected execution exposed that
  direct checkout test paths imported the host `MCP_unified` package before
  pytest could exercise the installed distribution, so Task 5 replaces every
  package-only workflow/consumer invocation with the same explicit mirrored
  release gate. The same review found two concrete strict-protocol defects:
  bounded error replacement now prefers any fitting correlated response before
  a null-ID variant, and legacy arbitrary-root text fallback validates a
  supported schema-declared dialect while published object-root descriptors
  remain constrained to the negotiated profile dialect. These are bounded
  contract corrections in `protocol_connection.py` and
  `protocol_validation.py`, not changes to compatibility HTTP/WebSocket or the
  accepted private stdio-adapter split.

- [ ] **Step 4: Update public documentation and package status**

  Document the five revisions, lifecycle/batch matrix, strict versus compatibility surfaces, embed example, exact default limits, cache/error behavior, application-owned privacy boundary, cancellation/shutdown behavior, no modern HTTP claim, and compatible downstream pin `~=0.2.0`. Set package status to `public-alpha`, retain publishing status `published`, and keep root/package-resource README and USER_GUIDE copies byte-identical.

  Embed example:

  ```python
  from mcp_unified.gateway import GatewayLimits, serve_stdio

  raise SystemExit(
      asyncio.run(
          serve_stdio(
              runtime,
              limits=GatewayLimits(max_in_flight=1),
          )
      )
  )
  ```

- [ ] **Step 5: Run focused, package, and compatibility verification**

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_validation.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_projection.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_connection.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_stdio.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_artifact_consumer.py -q
  python -m pytest tldw_Server_API/app/core/MCP_unified/tests tldw_Server_API/tests/MCP_unified tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py -q
  ```

- [ ] **Step 6: Run static, security, and release-candidate gates**

  ```bash
  source .venv/bin/activate
  python -m compileall -q apps/mcp-unified/src/mcp_unified
  python -m ruff check apps/mcp-unified/src/mcp_unified tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_*.py
  python -m mypy apps/mcp-unified/src/mcp_unified/gateway
  python -m bandit -r apps/mcp-unified/src/mcp_unified/gateway Helper_Scripts/mcp_unified_rc.py Helper_Scripts/Testing-related/mcp_official_sdk_stdio_smoke.py -f json -o /tmp/bandit_task_13009.json
  make mcp-unified-rc
  make mcp-unified-publish-dry-run
  git diff --check
  ```

  Record exact tool availability and any repository-baseline skips; do not claim an unavailable checker passed.

- [ ] **Step 7: Recheck PyPI and complete the task record**

  Fetch `https://pypi.org/pypi/mcp-unified/json` immediately before release. If only `0.1.1` exists, retain `0.2.0`; if `0.2.0` now exists, stop and select the next available minor through a task/plan update before changing metadata. Update TASK-13009 acceptance criteria, implementation notes, touched files, test/Bandit/RC evidence, ADR-033 link, rollback, and known skips, then mark it Done only after every DoD item passes.

- [ ] **Step 8: Commit Task 5**

  ```bash
  git add apps/mcp-unified Helper_Scripts/mcp_unified_rc.py Helper_Scripts/Testing-related/mcp_official_sdk_stdio_smoke.py .github/workflows/mcp-unified-rc.yml .github/workflows/mcp-unified-publish.yml .github/license-first-paths.json tldw_Server_API/app/core/MCP_unified/tests/mcp_unified_artifact_test_utils.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_artifact_consumer.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_connection.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_stdio.py tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py backlog/tasks/task-13009\ -\ Implement-MCP-Unified-multi-revision-stdio-protocol.md Docs/superpowers/plans/2026-08-08-mcp-unified-multi-revision-stdio-implementation.md
  git commit -m "release(mcp): prepare multi-revision stdio 0.2.0"
  ```

- [ ] **Step 9: Merge, publish, and independently verify before Chatbook migration**

  Push the reviewed branch, open the upstream PR, wait for required checks, and follow the controlled `dev` to `main` release workflow. After the protected workflow publishes, install `mcp-unified~=0.2.0` from PyPI in a fresh environment, verify metadata/hash/public imports and a real stdio smoke, attach the release evidence to TASK-13009, and only then unblock Chatbook TASK-2512. If publish verification fails, leave Chatbook on FastMCP and publish a corrective upstream version rather than overwriting an artifact.

## Plan Self-Review

- Spec coverage: Tasks 1-5 cover every public API, revision/lifecycle, method, projection, schema, pagination, error, stdio, interoperability, packaging, security, and release requirement in the approved design.
- Placeholder scan: implementation steps name exact files, symbols, fixtures, limits, commands, and expected failures; the plan contains no deferred placeholders.
- Type consistency: later tasks consume the exact `GatewayLimits`, `GatewayProtocolProfile`, `GatewaySchemaValidationManager`, `GatewayCatalogPaginator`, `GatewayProtocolConnection`, and `serve_stdio` names produced earlier.
- ADR check: implementation directly follows accepted ADR-033; no additional storage, ownership, dependency, or transport decision is introduced.
