# MCP UAT JSON-RPC And Transport Remediation Design

Date: 2026-06-19

## Status

Draft specification for TASK-2392. Revised after three automated spec-review
passes; the third-pass findings were incorporated without dispatching a fourth
reviewer to respect the workflow iteration cap.

## Goal

Fix the discovered MCP UAT blockers in one cohesive PR so both the mounted
tldw_server MCP surface and the standalone MCP server/package can be run
through the smoke harness with trustworthy results.

The PR should make JSON-RPC behavior consistent across transports without
turning the work into a broad MCP Hub, AuthNZ, profile, or frontend refactor.

## UAT Targets

Mounted tldw_server MCP:

- `POST /api/v1/mcp/request`
- `POST /api/v1/mcp/request/batch`
- `WS /api/v1/mcp/ws`
- single-user API key, MCP JWT, and test-mode compatibility auth paths

Standalone MCP package:

- in-process gateway transport
- standalone FastAPI HTTP transport
- standalone FastAPI WebSocket transport
- stdio subprocess transport

## Confirmed Issues

1. JSON-RPC success responses include `error: null`, and error responses can
   include null fields that strict clients reject.
2. Default mounted tldw single-user/test auth can produce admin or wildcard
   claims, but module and tool checks only consult DB-backed AuthNZ RBAC, so
   the default catalog can be empty in UAT.
3. Policy-enabled safe tool calls fail closed because effective-policy
   resolution hits a circular import around `McpHubPolicyResolver`.
4. `notifications/initialized` returns method-not-found even though it is a
   standard no-response MCP notification.
5. The smoke harness expects unknown `tools/call` to return JSON-RPC
   method-not-found, while the mounted protocol treats known method plus
   unknown tool name as invalid params.
6. The smoke harness rejects valid ping responses that include metadata such as
   a timestamp.
7. Malformed mounted HTTP JSON-RPC requests are rejected by FastAPI body
   validation as HTTP 422 before the protocol can return JSON-RPC `-32600`.
8. Mounted WebSocket API-key auth is inconsistent with HTTP single-user/test
   compatibility, and strict WebSocket clients reject host keepalive frames or
   null JSON-RPC envelope fields.

## Non-Goals

- No redesign of MCP Hub policy semantics.
- No broad AuthNZ seed or role model refactor.
- No profile, preset, or recommendation catalog changes.
- No frontend or admin UI work.
- No weakening of fail-closed policy behavior for real policy resolver
  failures.
- No new transport framework or large abstraction layer.

## Selected Approach

Use one cohesive PR with internally staged commits:

1. lock the current UAT failure matrix into focused regression coverage
2. fix JSON-RPC envelope serialization across mounted HTTP, batch, and
   WebSocket paths
3. fix mounted notification and malformed request semantics
4. align mounted HTTP and WebSocket single-user/test auth compatibility without
   creating a production bypass
5. break the policy resolver import cycle while preserving fail-closed runtime
   behavior
6. adjust the smoke harness only where the harness contract is too strict for
   valid server behavior
7. run the full mounted and standalone UAT matrix

This keeps product fixes ahead of harness loosening. The harness should only
change where it currently rejects valid behavior or conflates distinct
JSON-RPC error classes.

## Architecture

### JSON-RPC Serialization

Add one small serialization helper for mounted MCP response emission. The
helper should accept `MCPResponse` or a list of `MCPResponse` values and return
plain JSON-serializable dictionaries with invalid optional null fields removed.
It must not remove required JSON-RPC fields whose value is intentionally null,
especially `id: null` on parse errors and invalid requests where the server
cannot determine the original id.

Required behavior:

- success response includes `jsonrpc`, `id`, and `result`
- response `id` is always emitted for every response-producing request, even
  when the value is null
- success response omits `error`
- error response includes `jsonrpc`, `id`, and `error`
- error response omits `result`
- `error.data` is omitted when null
- batch responses apply the same rules to every item
- parse-error and invalid-request responses preserve `id: null` when the
  request id is unavailable
- a request object with no `id` member is a notification and produces no
  response; an explicit `"id": null` is still a response-producing request
- request parsing must preserve whether the `id` member was absent or present
  with a null value; do not rely on a parsed model default alone to distinguish
  notifications from explicit-null-id requests

Use this helper in:

- mounted `/request`
- mounted `/request/batch`
- mounted `/ws`

Keep protocol internals returning `MCPResponse` so existing unit tests and
internal call sites do not churn.

The standalone gateway transports are also part of the UAT matrix. They must
be verified against the same strict-envelope contract. If they already satisfy
it, leave them unchanged and record the verification; if they fail, fix the
transport-specific serialization boundary with the same optional-null omission
rules instead of changing runtime business logic.

### Notification Semantics

Support `notifications/initialized` as a known no-op notification. It should
produce no response when the request id is absent. A notification is a request
object with no `id` member; an explicit `"id": null` is not a notification.

The protocol already suppresses responses for known-method notifications after
handler execution and for many exception paths. The gap is that an unknown
notification method returns method-not-found before reaching the no-response
branch. The fix should either register a no-op handler for
`notifications/initialized` or special-case known MCP notifications before
method-not-found emission.

This PR does not need to redesign all unknown notification behavior, but it
must not bless a JSON-RPC error response for a request object that has no `id`.
If unknown-notification behavior is touched, use the JSON-RPC notification
rule: no response for any notification, including an unknown notification.

Mandatory mounted transport outcomes for `notifications/initialized`:

- single mounted HTTP notification returns HTTP 204 with no response body
- mounted WebSocket notification sends no response frame
- mixed batch requests omit notification items from the response array
- all-notification batches return HTTP 204 with no response body

Standalone notification expectations:

- standalone in-process gateway returns no JSON-RPC response object
- standalone HTTP returns HTTP 204 with no response body
- standalone WebSocket sends no response frame
- standalone stdio writes no response line

The standalone paths are verification-only if they already comply. If a
standalone transport fails the notification no-response contract, fix that
transport boundary in the same PR because it is part of the UAT target matrix.

### Mounted HTTP Request Parsing

Mounted HTTP routes should preserve all existing auth, security, session, and
safe-config behavior, but avoid FastAPI rejecting malformed JSON-RPC envelopes
before the protocol can respond.

Route parsing rules:

- JSON-RPC error bodies return HTTP 200 unless the route is intentionally
  returning HTTP 204 for notifications
- invalid JSON body returns one JSON-RPC parse-error object with code `-32700`
  and `id: null`
- non-object single request returns one JSON-RPC invalid-request object with
  code `-32600`
- object missing required JSON-RPC fields returns one invalid-request object
  with code `-32600`
- valid object flows into the existing `MCPRequest` and server/protocol path
- `POST /request/batch` requires a JSON array; a non-array payload returns one
  invalid-request object with code `-32600`, not an array
- empty batch returns one invalid-request object with code `-32600`, not an
  array
- invalid elements inside a batch return per-element invalid-request responses;
  preserve the element id when safely available, otherwise use `id: null`
- mixed valid, invalid, and notification batch items return responses only for
  response-producing items, preserving response correlation with each item id
- notification-only batches return HTTP 204 with no response body, matching
  the standalone gateway notification contract
- HTTP, batch, and WebSocket parsing tests must cover absent `id` notifications
  separately from explicit `"id": null` requests, and explicit-null-id
  requests must receive a response with `id: null`

The route should not silently bypass dependencies such as authentication,
HTTP security enforcement, API-key metadata attachment, session header
handling, or safe-config parsing.

### Mounted Auth And RBAC Compatibility

Make mounted HTTP and WebSocket authentication agree for single-user and
test-mode UAT. The compatibility path must be narrow.

Required behavior:

- HTTP continues to accept configured `SINGLE_USER_API_KEY` and, in safe
  test contexts, `SINGLE_USER_TEST_API_KEY`.
- WebSocket API-key auth accepts the same single-user/test compatibility keys
  through a shared helper or equivalent injected auth-provider behavior.
- Protocol module/tool permission checks may honor wildcard or admin claims
  only when metadata marks them as trusted compatibility claims from the
  server-side auth path.
- Arbitrary caller-supplied metadata must not be able to set
  `permissions=["*"]` and bypass DB-backed RBAC.
- `SINGLE_USER_TEST_API_KEY` is accepted only when explicit test-mode
  configuration is active, such as the repository's existing `TEST_MODE=true`
  test guard or a dedicated MCP test-auth guard; the presence of the key alone
  must not enable it.
- `SINGLE_USER_TEST_API_KEY` is rejected by both mounted HTTP and WebSocket
  under production-default configuration.
- invalid API keys fail consistently for mounted HTTP and WebSocket.
- MCP JWT and DB-backed RBAC paths remain unchanged and get regression
  coverage so compatibility claims do not replace normal authorization.

This should unblock UAT defaults without weakening production AuthNZ behavior.

### Policy Resolver Import Cycle

The policy-enabled safe tool failure is not a policy decision failure. It is an
import boundary failure that the protocol correctly converts into a fail-closed
policy-unavailable state.

Fix direction:

- identify the circular import involving `McpHubPolicyResolver`
- move shared helpers or result models to a lower-level module, or localize
  imports so resolver construction no longer imports a module that imports the
  resolver back
- preserve fail-closed behavior for real resolver exceptions
- add a focused regression test for resolving policy during a safe discovery
  tool call in the mounted tldw context
- add a negative regression test proving that a real policy resolver/runtime
  failure still fails closed after the import-cycle fix

### Smoke Harness Contract Alignment

The smoke harness should remain strict about JSON-RPC envelope shape and
correlation, but it should not reject valid server-specific payload metadata.

Adjustments:

- ping success accepts any dictionary with `pong is True`
- unknown JSON-RPC method remains expected `-32601`
- known `tools/call` with an unknown tool may return `-32602` with a tool-not
  found message for the mounted tldw protocol
- standalone transports can continue returning `-32601` where `tools/call` is
  unavailable
- the smoke WebSocket client can ignore known outbound host keepalive frames
  such as `{"type": "ping"}` that arrive while waiting for the correlated
  JSON-RPC response
- the mounted WebSocket server may tolerate inbound known keepalive frames with
  the same explicit shapes, but must not produce a JSON-RPC response for them
  or treat them as satisfying an outstanding request id
- malformed non-keepalive WebSocket objects remain strict protocol failures

## Sequential Implementation Plan

### Stage 1: Baseline And Regression Tests

Add focused failing tests or smoke assertions for:

- mounted HTTP success response omits `error`
- mounted batch success responses omit null fields
- mounted WebSocket success response is a strict JSON-RPC response
- `notifications/initialized` returns HTTP 204 for mounted HTTP, sends no
  mounted WebSocket frame, is omitted from mixed batch responses, and returns
  HTTP 204 for all-notification batches
- standalone in-process, HTTP, WebSocket, and stdio notification paths either
  already satisfy the no-response contract or get fixed
- absent `id` notifications and explicit `"id": null` requests are distinct
  for mounted HTTP, batch, and WebSocket
- malformed mounted HTTP single request returns JSON-RPC `-32600`, not HTTP
  422
- mounted batch non-array payload returns one `-32600` object with HTTP 200
- mounted batch empty array returns one `-32600` object with HTTP 200
- mounted batch invalid elements preserve a safe element id or use `id: null`
- mounted mixed batch omits notification items and preserves request-response
  correlation
- mounted all-notification batch returns HTTP 204 with no body
- mounted HTTP and WebSocket accept single-user/test API-key auth consistently
- mounted HTTP and WebSocket reject invalid API keys consistently
- mounted HTTP and WebSocket reject `SINGLE_USER_TEST_API_KEY` under
  production-default configuration and accept it only when the explicit test
  guard is active
- caller-supplied request metadata cannot grant wildcard/admin permissions
- JWT and DB-backed RBAC authorization remains unchanged
- policy-enabled safe discovery call does not fail from resolver import cycle
- real policy resolver/runtime failures still fail closed
- smoke ping predicate accepts metadata
- smoke unknown-tool behavior distinguishes unknown methods from unknown tools

### Stage 2: JSON-RPC Envelope Serialization

Implement the shared serialization helper and wire it through mounted HTTP,
batch, and WebSocket response paths. Verify strict clients no longer see
`error: null` on success or `data: null` on errors.

### Stage 3: Notification And Request Parsing

Add `notifications/initialized` support and update mounted HTTP parsing so
invalid JSON-RPC envelopes are mapped into JSON-RPC errors. Preserve the
existing auth/session/security path.

### Stage 4: Mounted Auth/RBAC Consistency

Unify single-user/test API-key compatibility for mounted HTTP and WebSocket.
Add trusted-claim metadata and ensure protocol RBAC compatibility honors it
only from trusted server-side auth paths.

### Stage 5: Policy Resolver Import Cycle

Break the circular import around effective-policy resolution. Verify
policy-enabled safe tool calls do not fail closed unless the resolver genuinely
fails at runtime.

### Stage 6: Smoke Harness Alignment

Update smoke predicates for valid ping metadata, unknown-tool semantics, and
recognized host WebSocket keepalive frames. Keep strict JSON-RPC envelope and
response-correlation checks.

### Stage 7: Full UAT Validation

Run and record:

- fixture smoke harness
- standalone in-process gateway smoke
- standalone FastAPI HTTP smoke
- standalone FastAPI WebSocket smoke
- stdio subprocess smoke
- mounted tldw FastAPI smoke
- mounted tldw live HTTP smoke
- mounted tldw live WebSocket API-key smoke
- mounted tldw live WebSocket JWT smoke
- focused unit tests for protocol, HTTP, WebSocket, auth, policy, and smoke
  changes
- Bandit on touched MCP files

## Testing Strategy

Use TDD for behavior changes. Prefer focused tests near the affected boundary:

- protocol tests for notification and unknown-tool semantics
- endpoint tests for mounted HTTP malformed request handling and envelope shape
- WebSocket tests for envelope shape, API-key compatibility, and keepalive
  frame tolerance
- policy resolver tests for import-cycle regression
- smoke client tests for ping and unknown-tool contract alignment

Program-level completion requires the mounted and standalone UAT surfaces to
pass their intended smoke paths.

## Risks And Mitigations

Risk: changing FastAPI request typing could weaken docs or validation.
Mitigation: keep parsing localized to MCP JSON-RPC routes, preserve auth and
security dependencies, and convert valid payloads back into `MCPRequest`.

Risk: wildcard permission compatibility could become a production bypass.
Mitigation: honor wildcard/admin claims only when server-side auth metadata
marks them as trusted compatibility claims.

Risk: smoke harness loosening could mask real protocol bugs.
Mitigation: loosen only payload-level metadata expectations, not JSON-RPC
envelope validity, response ids, batch correlation, or transport errors.

Risk: policy import-cycle fix could spread service helper imports.
Mitigation: move shared types/helpers downward or localize imports; avoid
adding new service-to-service cycles.

Risk: WebSocket keepalive tolerance could hide malformed frames.
Mitigation: accept only explicit known keepalive shapes and keep strict failure
for other non-JSON-RPC objects.

## Success Criteria

- Mounted tldw MCP HTTP, batch HTTP, and WebSocket emit strict JSON-RPC
  envelopes.
- Mounted tldw MCP and standalone MCP smoke surfaces pass their intended UAT
  matrix.
- `notifications/initialized` behaves as a no-response notification.
- Mounted malformed JSON-RPC requests return JSON-RPC errors instead of FastAPI
  422 responses.
- Single-user/test API-key behavior is consistent between mounted HTTP and
  WebSocket.
- Policy-enabled safe discovery calls no longer fail due to resolver import
  cycles.
- Smoke harness distinguishes valid metadata from actual protocol failures.
- Targeted tests, smoke runs, and Bandit results are recorded in TASK-2392.
