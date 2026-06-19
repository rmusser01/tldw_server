# MCP UAT JSON-RPC Transport Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the mounted tldw_server MCP and standalone MCP JSON-RPC UAT blockers so strict smoke clients can validate HTTP, batch HTTP, WebSocket, in-process, and stdio transports reliably.

**Architecture:** Keep protocol business logic intact and fix behavior at transport boundaries. Add small JSON-RPC helpers for mounted response serialization, raw envelope/id-presence parsing, and keepalive recognition; update standalone gateway JSON-RPC helpers only where strict envelope or id semantics require it. Product fixes come before smoke-harness relaxations.

**Tech Stack:** Python, FastAPI, Pydantic v1/v2-compatible models, pytest, Starlette TestClient, `mcp_unified` standalone gateway/smoke package, Bandit.

---

## Spec Reference

Approved design:

- `Docs/superpowers/specs/2026-06-19-mcp-uat-jsonrpc-transport-remediation-design.md`

Backlog:

- `TASK-2393` tracks this implementation plan.
- The follow-on implementation PR should create a separate implementation Backlog task before code edits begin.

## File Structure

Create:

- `tldw_Server_API/app/core/MCP_unified/jsonrpc_transport.py`
  Mounted tldw_server transport-boundary helper only. Responsibilities: serialize `MCPResponse` without invalid optional null fields while preserving `id: null`, parse raw HTTP JSON bodies into JSON-RPC payloads or `MCPResponse` parse errors, identify absent-id notifications versus explicit-null-id requests, validate safe JSON-RPC ids for invalid batch-element correlation, and recognize exact WebSocket keepalive frames.

- `tldw_Server_API/app/core/MCP_unified/tests/test_jsonrpc_transport_helpers.py`
  Unit tests for mounted helper behavior: response serialization, parse errors, invalid ids, absent-id detection, and exact keepalive allowlist.

- `tldw_Server_API/app/core/MCP_unified/tests/test_mounted_jsonrpc_transport_contract.py`
  Endpoint/WebSocket contract tests for mounted HTTP, batch HTTP, and WebSocket JSON-RPC shape, notification behavior, raw parsing, authz status mapping, and explicit-null-id behavior.

- `tldw_Server_API/app/core/MCP_unified/tests/fixtures/smoke_gateway_app.py`
  Tiny standalone FastAPI gateway fixture app for live HTTP/WebSocket smoke validation. Responsibilities: expose `app = create_gateway_app(SmokeFixtureGatewayRuntime(), prefix="/mcp")` without importing tldw_server host runtime code.

Modify:

- `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py`
  Change `/request` and `/request/batch` from Pydantic body binding to raw body parsing after dependencies run. Use the mounted helper for parse/invalid-request responses and response serialization. Keep auth, HTTP security, session header, safe-config, and API-key metadata paths intact. Remove the HTTP 403 conversion for post-protocol JSON-RPC `-32001` only on `/request` and `/request/batch`; leave pre-protocol dependency failures as HTTP 401/403/4xx.

- `tldw_Server_API/app/core/MCP_unified/server.py`
  Use mounted response serialization in WebSocket sends. Add exact inbound keepalive tolerance for `{"type":"ping"}` and `{"type":"pong"}` only. Emit JSON-RPC parse/invalid-request frames for malformed non-keepalive frames where possible. Align WebSocket single-user/test API-key compatibility with HTTP through trusted server-side metadata.

- `tldw_Server_API/app/core/MCP_unified/protocol.py`
  Register or special-case `notifications/initialized` as a no-op notification. Honor trusted compatibility claims for module/tool permission checks only when server-side metadata fields prove the source. Preserve existing fail-closed behavior for real policy resolver failures.

- `tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py`
  Break or localize the `McpHubPolicyResolver` import cycle if this is the source of the policy resolver failure. Keep host-specific imports in adapters, not standalone package modules.

- `mcp_unified/gateway/jsonrpc.py`
  Preserve absent `id` versus explicit `"id": null` before `GatewayJSONRPCRequest` construction. Update `response_to_json()` to omit invalid optional nulls while preserving response `id`, including `id: null`. Tighten id validation so booleans/floats/objects/arrays are not valid correlation ids.

- `mcp_unified/gateway/fastapi.py`
  Ensure standalone HTTP transport returns strict JSON-RPC bodies or HTTP 204 for notifications after any gateway JSON-RPC helper changes.

- `mcp_unified/gateway/stdio.py`
  Verify or update standalone stdio behavior so notifications write no response line and explicit-null-id requests write a response line with `id: null`.

- `mcp_unified/smoke/scenarios.py`
  Relax only valid smoke expectations: ping result can include metadata; mounted unknown tool may be `-32602`; strict JSON-RPC envelope/correlation remains enforced.

- `mcp_unified/smoke/transports.py`
  Let the live WebSocket smoke transport ignore exact outbound keepalive frames `{"type":"ping"}` and `{"type":"pong"}` while waiting for correlated JSON-RPC responses. Keep all other malformed frames as transport failures.

- Existing tests to extend where appropriate:
  - `tldw_Server_API/app/core/MCP_unified/tests/test_http_batch.py`
  - `tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py`
  - `tldw_Server_API/app/core/MCP_unified/tests/test_jsonrpc_notifications.py`
  - `tldw_Server_API/app/core/MCP_unified/tests/test_websocket_smoke.py`
  - `tldw_Server_API/app/core/MCP_unified/tests/test_ws_parse_error_jsonrpc.py`
  - `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py`
  - `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`
  - `tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py`

## Task 1: Mounted JSON-RPC Boundary Helper

**Files:**

- Create: `tldw_Server_API/app/core/MCP_unified/jsonrpc_transport.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_jsonrpc_transport_helpers.py`

- [ ] **Step 1: Write failing helper tests**

Add tests covering:

```python
def test_serialize_success_omits_error_but_preserves_id():
    response = MCPResponse(result={"ok": True}, id="ok-1")
    assert mcp_response_to_json(response) == {
        "jsonrpc": "2.0",
        "id": "ok-1",
        "result": {"ok": True},
    }


def test_serialize_error_omits_null_data_and_preserves_null_id():
    response = MCPResponse(
        error=MCPError(code=-32700, message="Parse error"),
        id=None,
    )
    assert mcp_response_to_json(response) == {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": -32700, "message": "Parse error"},
    }


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"jsonrpc": "2.0", "method": "ping"}, False),
        ({"jsonrpc": "2.0", "method": "ping", "id": None}, True),
    ],
)
def test_has_request_id_preserves_absent_vs_explicit_null(payload, expected):
    assert jsonrpc_payload_has_id(payload) is expected


@pytest.mark.parametrize("value", ["abc", 1, None])
def test_safe_jsonrpc_id_accepts_strings_integers_and_null(value):
    assert safe_jsonrpc_id(value) == value


@pytest.mark.parametrize("value", [True, False, 1.2, [], {}])
def test_safe_jsonrpc_id_rejects_unsafe_values(value):
    assert safe_jsonrpc_id(value) is None


@pytest.mark.parametrize("frame", [{"type": "ping"}, {"type": "pong"}])
def test_exact_keepalive_allowlist(frame):
    assert is_jsonrpc_keepalive(frame) is True


@pytest.mark.parametrize("frame", [{"type": "ping", "id": 1}, {"type": "other"}, {"jsonrpc": "2.0"}])
def test_non_keepalive_frames_are_not_allowlisted(frame):
    assert is_jsonrpc_keepalive(frame) is False
```

- [ ] **Step 2: Run helper tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_jsonrpc_transport_helpers.py -v
```

Expected: FAIL because `jsonrpc_transport.py` does not exist.

- [ ] **Step 3: Implement the helper**

Implement small functions only:

```python
def mcp_response_to_json(response: MCPResponse) -> dict[str, Any]:
    ...

def mcp_responses_to_json(response: MCPResponse | list[MCPResponse]) -> dict[str, Any] | list[dict[str, Any]]:
    ...

def parse_jsonrpc_body(raw_body: bytes) -> Any | MCPResponse:
    ...

def jsonrpc_payload_has_id(payload: Any) -> bool:
    ...

def safe_jsonrpc_id(value: Any) -> str | int | None:
    ...

def invalid_request_response(message: str, request_id: str | int | None = None) -> MCPResponse:
    ...

def parse_error_response(message: str = "Parse error") -> MCPResponse:
    ...

def is_jsonrpc_keepalive(payload: Any) -> bool:
    ...
```

Implementation constraints:

- Use `model_dump(mode="json", exclude_none=True)` when available, but always reinsert `id` for response-producing responses.
- Do not include `error` on success or `result` on error.
- Remove `error.data` when it is `None`.
- Treat JSON booleans as invalid ids even though Python `bool` subclasses `int`.
- Keep the file independent from FastAPI.

- [ ] **Step 4: Run helper tests and verify they pass**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_jsonrpc_transport_helpers.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/MCP_unified/jsonrpc_transport.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_jsonrpc_transport_helpers.py
git commit -m "fix: add mounted MCP JSON-RPC transport helpers"
```

## Task 2: Mounted HTTP JSON-RPC Parsing And Response Shape

**Files:**

- Modify: `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_mounted_jsonrpc_transport_contract.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_http_batch.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py`

- [ ] **Step 1: Write failing mounted HTTP tests**

Add tests for:

- `/api/v1/mcp/request` success omits `error`.
- `/api/v1/mcp/request` invalid JSON body returns HTTP 200 with `-32700` and `id: null`.
- `/api/v1/mcp/request` well-formed invalid envelope returns HTTP 200 with `-32600`.
- `/api/v1/mcp/request` notification `notifications/initialized` returns HTTP 204 and empty body.
- `/api/v1/mcp/request` explicit `"id": null` request returns a JSON-RPC response with `id: null`.
- Post-protocol authorization failure returns HTTP 200 JSON-RPC `-32001`; pre-protocol auth dependency failure still returns HTTP 401/403.

Example test skeleton:

```python
def test_mounted_request_invalid_json_returns_jsonrpc_parse_error():
    with build_mcp_test_client(auth_principal_override=build_mcp_admin_auth_override()) as client:
        response = client.post(
            "/api/v1/mcp/request",
            content=b"{not-json",
            headers={"content-type": "application/json", "Authorization": "Bearer test"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["jsonrpc"] == "2.0"
    assert body["id"] is None
    assert body["error"]["code"] == -32700
    assert "result" not in body
```

Batch tests:

- Non-array payload to `/request/batch` returns one `-32600` object, not a list.
- Empty array returns one `-32600` object, not a list. Update legacy `test_http_batch_empty_returns_invalid_request`.
- Invalid batch element with `id: true` returns `id: null`.
- Invalid batch element with non-integer number id returns `id: null`.
- Mixed notification and request batch omits notification item.
- All-notification batch returns HTTP 204 with empty body.
- Batch success items omit `error`.

- [ ] **Step 2: Run mounted HTTP tests and verify failures**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_mounted_jsonrpc_transport_contract.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_http_batch.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py \
  -v
```

Expected: FAIL on FastAPI 422, null envelope fields, legacy empty-batch shape, or HTTP 403 conversion.

- [ ] **Step 3: Refactor mounted routes to raw JSON-RPC parsing**

In `mcp_unified_endpoint.py`:

- Change `/request` body parameter from `request: MCPRequest` to raw `http_request: Request`.
- Read `await http_request.body()` after dependencies have executed.
- Use `parse_jsonrpc_body()`.
- If parser returns `MCPResponse`, return `JSONResponse(mcp_response_to_json(response))`.
- Validate single-request payload shape before constructing `MCPRequest`.
- Keep session creation for `initialize` by checking raw payload `method`.
- Keep `_attach_api_key_metadata()`, `_get_derived_user_id()`, safe-config parsing, `mcp-session-id`, and `enforce_http_security` exactly in path.
- Return `Response(status_code=204)` for notification responses.
- Return `JSONResponse(mcp_response_to_json(resp_obj))` for all JSON-RPC response objects.
- Remove `/request` and `/request/batch` post-protocol `-32001` to HTTP 403 conversion. Leave direct auth dependencies untouched.

For `/request/batch`:

- Change body from `requests: list[MCPRequest]` to raw body.
- Reject non-array and empty array with one JSON-RPC error object.
- For valid items, construct `MCPRequest` only after retaining whether `id` was present.
- Preserve invalid batch element ids only for strings, integer numbers excluding booleans, and null.
- Return HTTP 204 for all-notification batches.
- Remove `response_model=list[MCPResponse]`; use explicit `responses={200: ..., 204: ...}` documentation.

- [ ] **Step 4: Run mounted HTTP tests and verify pass**

Run the same command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_mounted_jsonrpc_transport_contract.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_http_batch.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py
git commit -m "fix: normalize mounted MCP HTTP JSON-RPC envelopes"
```

## Task 3: Mounted WebSocket JSON-RPC Shape And Keepalives

**Files:**

- Modify: `tldw_Server_API/app/core/MCP_unified/server.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_mounted_jsonrpc_transport_contract.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_websocket_smoke.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_ws_parse_error_jsonrpc.py`

- [ ] **Step 1: Write failing WebSocket tests**

Cover:

- Successful WebSocket `ping` response omits `error`.
- `notifications/initialized` sends no frame and does not break a follow-up `ping`.
- Explicit `"id": null` request produces a response frame with `id: null`.
- Inbound `{"type":"ping"}` and `{"type":"pong"}` produce no response and do not satisfy pending JSON-RPC ids.
- Inbound `{"type":"ping","id":"x"}` returns an invalid-request JSON-RPC error frame with `id: null`.
- Invalid JSON text returns a parse-error JSON-RPC frame with `id: null` before any close behavior.

- [ ] **Step 2: Run WebSocket tests and verify failures**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_mounted_jsonrpc_transport_contract.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_websocket_smoke.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_ws_parse_error_jsonrpc.py \
  -v
```

Expected: FAIL on null fields, `notifications/initialized`, keepalive handling, or malformed frame semantics.

- [ ] **Step 3: Update mounted WebSocket transport**

In `server.py`:

- Before passing parsed JSON to `protocol.process_request()`, check `is_jsonrpc_keepalive(data)` and continue without sending.
- For other parseable non-JSON-RPC non-keepalive objects, send `invalid_request_response()` serialized through `mcp_response_to_json()`.
- For invalid JSON, send `parse_error_response()` serialized through `mcp_response_to_json()`.
- Replace all `response.model_dump()` and `[r.model_dump() for r in response]` WebSocket sends for protocol responses with `mcp_responses_to_json()`.
- Preserve rate-limit and internal-error response ids.

- [ ] **Step 4: Run WebSocket tests and verify pass**

Run the same command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/MCP_unified/server.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_mounted_jsonrpc_transport_contract.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_websocket_smoke.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_ws_parse_error_jsonrpc.py
git commit -m "fix: normalize mounted MCP WebSocket JSON-RPC handling"
```

## Task 4: Notification And Explicit-Null Semantics In Protocol And Standalone

**Files:**

- Modify: `tldw_Server_API/app/core/MCP_unified/protocol.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_jsonrpc_notifications.py`
- Modify: `mcp_unified/gateway/jsonrpc.py`
- Modify: `mcp_unified/gateway/fastapi.py`
- Modify: `mcp_unified/gateway/stdio.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py`

- [ ] **Step 1: Write failing protocol and standalone tests**

Protocol:

- `notifications/initialized` with absent id returns `None`.
- `{"jsonrpc":"2.0","method":"ping","id":None}` returns a response with `id is None`, not `None` response.

Standalone gateway:

- `handle_jsonrpc(runtime, {"jsonrpc":"2.0","method":"ping"})` returns `GatewayNoResponse`.
- `handle_jsonrpc(runtime, {"jsonrpc":"2.0","method":"ping","id":None})` returns success response with `id is None`.
- `response_to_json()` omits success `error`, error `result`, and null `error.data`, while preserving `id`.
- FastAPI HTTP notification returns 204; explicit-null-id request returns 200 JSON body.
- stdio notification writes no line; explicit-null-id request writes one response line.
- Id validation rejects `id: true` and non-integer number ids with `id: null`.

- [ ] **Step 2: Run tests and verify failures**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_jsonrpc_notifications.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py \
  -v
```

Expected: FAIL where absent id and explicit null id are conflated or null fields remain serialized.

- [ ] **Step 3: Implement notification/id-presence behavior**

Mounted protocol:

- Register `notifications/initialized` in the handler table as a no-op or special-case it before method-not-found.
- Preserve existing notification suppression for no-id requests.
- Do not rely only on `MCPRequest.id is None` when the raw transport already identified an explicit-null-id request.

Standalone:

- Add a small raw-envelope helper in `mcp_unified/gateway/jsonrpc.py` to track `has_id = "id" in payload`.
- Treat absent id as notification; explicit null id as request.
- Update `response_to_json()` to omit invalid optional null fields but keep response `id`.
- Ensure `parse_json_payload()` parse-error responses keep `id: null`.
- Keep pydantic v1/v2 compatibility.

- [ ] **Step 4: Run tests and verify pass**

Run the same command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/MCP_unified/protocol.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_jsonrpc_notifications.py \
  mcp_unified/gateway/jsonrpc.py \
  mcp_unified/gateway/fastapi.py \
  mcp_unified/gateway/stdio.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py
git commit -m "fix: preserve MCP notification id semantics"
```

## Task 5: Mounted Single-User/Test Auth Compatibility Without Metadata Bypass

**Files:**

- Modify: `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/server.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/protocol.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_mounted_jsonrpc_transport_contract.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_scope_enforcement.py` or `test_protocol_allowed_tools.py`

- [ ] **Step 1: Write failing auth and metadata tests**

Mounted HTTP:

- Configured `SINGLE_USER_API_KEY` allows default `tools/list` in single-user mode and metadata includes `auth_via="single_user_api_key"`, `trusted_auth_claims=True`, `compat_claims_source="mounted_http"`.
- `SINGLE_USER_TEST_API_KEY` is rejected when `TEST_MODE` is false / production-default config is active.
- `SINGLE_USER_TEST_API_KEY` is accepted only with explicit test guard.

Mounted WebSocket:

- Same `SINGLE_USER_API_KEY` and `SINGLE_USER_TEST_API_KEY` behavior as HTTP.
- Metadata source is `mounted_ws`.

Protocol:

- Caller-provided request metadata with `permissions=["*"]`, `trusted_auth_claims=True`, and `auth_via="single_user_api_key"` does not bypass RBAC unless it came from server-side trusted context.
- Normal MCP JWT and DB-backed RBAC behavior remains unchanged.

- [ ] **Step 2: Run auth tests and verify failures**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_mounted_jsonrpc_transport_contract.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_scope_enforcement.py \
  -v
```

Expected: FAIL where WebSocket lacks compatibility or protocol ignores trusted metadata.

- [ ] **Step 3: Implement trusted compatibility metadata**

HTTP:

- When `_resolve_token_data_compat()` accepts `SINGLE_USER_API_KEY`, attach server-side metadata to request state, for example:

```python
{
    "auth_via": "single_user_api_key",
    "trusted_auth_claims": True,
    "compat_claims_source": "mounted_http",
}
```

- For `SINGLE_USER_TEST_API_KEY`, use `auth_via="single_user_test_api_key"` and the same trust/source pattern.
- `_attach_api_key_metadata()` should reuse this server-side state and should not accept client payload metadata as trusted.

WebSocket:

- Add the equivalent single-user/test compatibility resolution before falling back to `auth_provider.validate_api_key()`.
- Set `compat_claims_source="mounted_ws"`.
- Preserve IP allowlist behavior and test-mode production guard.

Protocol:

- Strip or ignore all client-supplied metadata keys that look like trust controls before protocol permission checks, including `auth_via`, `trusted_auth_claims`, `compat_claims_source`, and any future reserved prefix such as `_server_auth_*`.
- Carry compatibility grants through a server-side-only auth context object or sentinel that is created by the mounted HTTP/WebSocket transport after authentication succeeds. Do not infer trust from ordinary JSON-RPC request metadata.
- Add a private helper such as `_has_trusted_compat_claims(context)` that checks the server-created sentinel plus exact fields and source allowlist.
- Let `_has_module_permission()` and `_has_tool_permission()` honor wildcard/admin claims only when the helper returns true.
- Add a regression test that passes forged client metadata with `auth_via`, `trusted_auth_claims`, and `compat_claims_source` and proves it does not bypass RBAC.
- Allow direct unit tests to inject trust only through an explicit test helper that constructs the same server-side auth context/sentinel the transport would create.

- [ ] **Step 4: Run auth tests and verify pass**

Run the same command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py \
  tldw_Server_API/app/core/MCP_unified/server.py \
  tldw_Server_API/app/core/MCP_unified/protocol.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_mounted_jsonrpc_transport_contract.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_scope_enforcement.py
git commit -m "fix: align mounted MCP compatibility auth"
```

## Task 6: Policy Resolver Import Cycle And Fail-Closed Coverage

**Files:**

- Modify: `tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/adapters/tldw_policy.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/protocol.py` only if resolver wiring needs no-op handling
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_governance_preflight.py`

- [ ] **Step 1: Write failing policy regression tests**

Add tests:

- Policy-enabled safe discovery/tool call does not fail because `McpHubPolicyResolver` import cycle prevents resolver construction.
- A fake resolver that raises at runtime still produces a fail-closed policy-unavailable/governance-denied response.

Use existing patterns from `test_protocol_allowed_tools.py` around `_FailingPolicyResolver`.

- [ ] **Step 2: Run policy tests and verify failures**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_governance_preflight.py \
  -v
```

Expected: FAIL on the import-cycle regression or missing coverage.

- [ ] **Step 3: Break the import cycle minimally**

Preferred order:

1. Move shared low-level data helpers to `tldw_Server_API/app/core/MCP_unified/adapters/tldw_policy.py` if they are currently nested in a module that imports runtime wiring.
2. Localize imports inside `TldwEffectivePolicyResolver.resolve_effective_policy()` so importing `tldw_runtime.py` no longer imports the resolver back through a cycle.
3. Keep fail-closed exception mapping in `protocol.py` unchanged for real runtime exceptions.

Do not move host-specific resolver code into the standalone `mcp_unified` package.

- [ ] **Step 4: Run policy tests and verify pass**

Run the same command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py \
  tldw_Server_API/app/core/MCP_unified/adapters/tldw_policy.py \
  tldw_Server_API/app/core/MCP_unified/protocol.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_governance_preflight.py
git commit -m "fix: stabilize mounted MCP policy resolver"
```

## Task 7: Smoke Harness Contract Alignment

**Files:**

- Create: `tldw_Server_API/app/core/MCP_unified/tests/fixtures/smoke_gateway_app.py`
- Modify: `mcp_unified/smoke/scenarios.py`
- Modify: `mcp_unified/smoke/transports.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py`

- [ ] **Step 1: Write failing smoke tests**

Add or update tests:

- Ping success accepts `{"pong": True, "timestamp": "..."}`.
- Unknown JSON-RPC method still requires `-32601`.
- Known `tools/call` with unknown tool accepts mounted-style `-32602` with a tool-not-found message as well as standalone-style `-32601` where `tools/call` is unavailable.
- Live WebSocket transport ignores exact outbound `{"type":"ping"}` and `{"type":"pong"}` frames while waiting for correlated response.
- Live WebSocket transport still fails on `{"type":"ping","id":"x"}` or any other malformed non-response frame.

- [ ] **Step 2: Run smoke tests and verify failures**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py -v
```

Expected: FAIL on strict ping result, unknown-tool error expectation, or keepalive frames.

- [ ] **Step 3: Update smoke harness only at contract edges**

In `mcp_unified/smoke/scenarios.py`:

- Update `_is_successful_ping_result()` to require `dict` and `result.get("pong") is True`.
- Update `_step_unknown_tool()` to accept:
  - `-32601` for method unavailable.
  - `-32602` only when the message indicates unknown/missing tool.

In `mcp_unified/smoke/transports.py`:

- Add exact keepalive allowlist in `LiveWebSocketTransport._handle_message()`.
- Do not treat keepalives as responses.
- Keep strict failure on all other malformed frames.

Create `tldw_Server_API/app/core/MCP_unified/tests/fixtures/smoke_gateway_app.py`:

```python
"""Standalone FastAPI gateway app used by live smoke validation."""

from mcp_unified.gateway.fastapi import create_gateway_app
from mcp_unified.smoke.fixtures import SmokeFixtureGatewayRuntime

app = create_gateway_app(SmokeFixtureGatewayRuntime(), prefix="/mcp")
```

- [ ] **Step 4: Run smoke tests and verify pass**

Run the same command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcp_unified/smoke/scenarios.py \
  mcp_unified/smoke/transports.py \
  tldw_Server_API/app/core/MCP_unified/tests/fixtures/smoke_gateway_app.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py
git commit -m "fix: align MCP smoke harness with strict transports"
```

## Task 8: Focused Regression Suite

**Files:**

- No new files expected.
- Run focused tests against all changed surfaces.

- [ ] **Step 1: Run mounted JSON-RPC focused tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_jsonrpc_transport_helpers.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_mounted_jsonrpc_transport_contract.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_http_batch.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_jsonrpc_notifications.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_websocket_smoke.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_ws_parse_error_jsonrpc.py \
  -v
```

Expected: PASS.

- [ ] **Step 2: Run standalone gateway and smoke tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py \
  -v
```

Expected: PASS.

- [ ] **Step 3: Run auth and policy tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_scope_enforcement.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_governance_preflight.py \
  -v
```

Expected: PASS.

- [ ] **Step 4: Commit any test-only stabilization if needed**

If tests require only fixture/test stabilization after implementation commits:

```bash
git add tldw_Server_API/app/core/MCP_unified/tests mcp_unified/smoke
git commit -m "test: stabilize MCP UAT regression coverage"
```

If no changes are needed, skip this commit and record the skip in the PR notes.

## Task 9: Full UAT Matrix And Security Validation

**Files:**

- Modify only if validation exposes bugs in touched MCP files.
- Record outputs in PR notes and Backlog task, not in repo artifacts unless requested.

- [ ] **Step 1: Run fixture smoke harness**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py::test_smoke_cli_returns_zero_for_passed_inprocess_baseline -v
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py::test_smoke_cli_returns_zero_for_stdio_fixture_baseline -v
```

Expected: PASS.

- [ ] **Step 2: Run standalone in-process and stdio smoke paths**

Run:

```bash
source .venv/bin/activate
python -m mcp_unified.smoke.cli inprocess --mode strict
python -m mcp_unified.smoke.cli stdio --command python --arg tldw_Server_API/app/core/MCP_unified/tests/fixtures/smoke_stdio_server.py --mode strict
```

Expected: exit code 0 and report `ok: true`.

If the CLI invocation differs in current code, run:

```bash
source .venv/bin/activate
python -m mcp_unified.smoke.cli --help
```

and use the documented equivalent. Record the exact command used.

- [ ] **Step 3: Run standalone live HTTP/WebSocket smoke paths**

Start the standalone fixture gateway in a separate terminal/session:

```bash
source .venv/bin/activate
python -m uvicorn tldw_Server_API.app.core.MCP_unified.tests.fixtures.smoke_gateway_app:app --host 127.0.0.1 --port 8765
```

Then run:

```bash
source .venv/bin/activate
python -m mcp_unified.smoke.cli http \
  --url http://127.0.0.1:8765/mcp/request \
  --mode strict
python -m mcp_unified.smoke.cli websocket \
  --url ws://127.0.0.1:8765/mcp/ws \
  --mode strict
```

Expected: exit code 0 and report `ok: true` for both standalone live transports.

- [ ] **Step 4: Run mounted live HTTP/WebSocket API-key smoke**

Start a local server in a separate terminal/session:

```bash
source .venv/bin/activate
AUTH_MODE=single_user \
SINGLE_USER_API_KEY=test-api-key-1234567890 \
MCP_JWT_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
MCP_API_KEY_SALT=ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss \
python -m uvicorn tldw_Server_API.app.main:app --host 127.0.0.1 --port 8000
```

Then run:

```bash
source .venv/bin/activate
MCP_SMOKE_API_KEY=test-api-key-1234567890 \
python -m mcp_unified.smoke.cli http \
  --url http://127.0.0.1:8000/api/v1/mcp/request \
  --api-key-env MCP_SMOKE_API_KEY \
  --mode strict
MCP_SMOKE_API_KEY=test-api-key-1234567890 \
python -m mcp_unified.smoke.cli websocket \
  --url ws://127.0.0.1:8000/api/v1/mcp/ws?api_key=test-api-key-1234567890 \
  --api-key-env MCP_SMOKE_API_KEY \
  --mode strict
```

Expected: exit code 0 and report `ok: true`.

- [ ] **Step 5: Run mounted live WebSocket JWT smoke**

Keep the mounted tldw server from Step 4 running with the same `MCP_JWT_SECRET`.

Mint an MCP JWT for the smoke run:

```bash
source .venv/bin/activate
export MCP_JWT_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export MCP_API_KEY_SALT=ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
export MCP_SMOKE_BEARER_TOKEN="$(python -c 'from tldw_Server_API.app.core.MCP_unified.auth.jwt_manager import get_jwt_manager; print(get_jwt_manager().create_access_token(subject="smoke-user", username="smoke", roles=["admin"], permissions=["mcp:*"]))')"
```

Then run:

```bash
source .venv/bin/activate
python -m mcp_unified.smoke.cli websocket \
  --url ws://127.0.0.1:8000/api/v1/mcp/ws \
  --bearer-token-env MCP_SMOKE_BEARER_TOKEN \
  --mode strict
```

Expected: exit code 0 and report `ok: true`.

- [ ] **Step 6: Run Bandit on touched MCP scopes**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/MCP_unified \
  tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py \
  mcp_unified \
  -f json -o /tmp/bandit_mcp_uat_remediation.json
```

Expected: no new findings in touched code. If baseline findings exist, document why they are unrelated and fix any new touched-code findings.

- [ ] **Step 7: Final status and commit**

Run:

```bash
git status --short
```

Expected: clean, or only intentional PR-note/backlog edits staged.

If validation fixes were needed:

```bash
git add <changed-files>
git commit -m "fix: close MCP UAT validation gaps"
```

## PR Completion Notes

Before opening or updating the PR:

- Summarize changed JSON-RPC behavior and intentional mounted batch contract update.
- Include exact focused pytest commands and smoke commands run.
- Include Bandit command and result path.
- Mention any skipped live smoke path with reason, if environment setup prevents it.
- Ask the human requester to provide the required AI-generated PR `Change summary` before merge if the PR is materially AI-authored.
