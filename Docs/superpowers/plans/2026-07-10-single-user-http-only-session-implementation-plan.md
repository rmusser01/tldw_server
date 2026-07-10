# Single-User HttpOnly Session Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace browser-visible runtime API-key provisioning in the runtime-enabled loopback quickstart WebUI with a persistent, revocable HttpOnly session that also authenticates first-party WebSockets.

**Architecture:** Reuse `SessionManager` and the existing AuthNZ `sessions` table for random opaque session tokens. A shared single-user session validator serves HTTP principal resolution, CSRF binding, WebSocket fallback, bootstrap reuse, and logout; Next.js exchanges its server-side API key for the cookie without returning a secret to JavaScript. The capability remains disabled until HTTP, CSRF, standalone WebSocket proxying, and first-party WebSocket callers pass together.

**Tech Stack:** FastAPI/Starlette, Pydantic settings, SQLite/PostgreSQL AuthNZ repositories, Next.js Pages API routes, TypeScript, Vitest, Pytest, Playwright.

## Global Constraints

- Scope automatic provisioning to `AUTH_MODE=single_user`, `NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=quickstart`, loopback Host/peer, no forwarding headers, and `TLDW_WEBUI_EXPOSE_RUNTIME_AUTH=1`.
- Never return, log, persist, serialize, or place the runtime API key in a browser URL, body, page state, storage record, diagnostic, screenshot, or test artifact.
- Session cookie defaults: `tldw_single_user_session`, HttpOnly, host-only, `Path=/api`, `SameSite=Lax`, 30-day bounded lifetime; `Secure=false` only for explicit loopback HTTP quickstart.
- Cookie-session minting fails closed when effective CSRF protection is disabled.
- Explicit `Authorization` and `X-API-KEY` credentials retain precedence and their existing CSRF exemption.
- Cookie WebSockets require a non-null exact trusted Origin before `accept()` and never use query-string secrets in cookie-session mode.
- Add no dependency and no database migration.
- TASK-12108 owns this plan; TASK-12106 executes only after this plan is complete.

---

### Task 1: Opaque Single-User Session Primitive

**Files:**
- Create: `tldw_Server_API/app/core/AuthNZ/single_user_session.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/settings.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/sessions_repo.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_single_user_session.py`

**Interfaces:**
- Consumes: `SessionManager.create_session()`, `SessionManager.validate_session()`, `Settings.SINGLE_USER_FIXED_ID`.
- Produces: `SingleUserSessionIdentity`, `MintedSingleUserSession`, `mint_single_user_session()`, `validate_single_user_session()`, `set_single_user_session_cookie()`, `clear_single_user_session_cookie()`.

- [ ] **Step 1: Write failing primitive tests**

```python
@pytest.mark.asyncio
async def test_mint_uses_random_tokens_constant_type_and_bounded_expiry(monkeypatch):
    manager = AsyncMock()
    manager.create_session.return_value = {
        "session_id": 7,
        "user_id": 1,
        "expires_at": "2026-08-09T00:00:00+00:00",
    }
    created = await mint_single_user_session(_request(), manager)
    kwargs = manager.create_session.await_args.kwargs
    assert kwargs["access_token"] != kwargs["refresh_token"]
    assert len(kwargs["access_token"]) >= 43
    assert kwargs["device_id"] == "single-user-cookie:v1"
    assert created.identity.session_id == 7
    assert created.cookie_token == kwargs["access_token"]


@pytest.mark.asyncio
async def test_validate_rejects_wrong_type_and_accepts_cookie_session(monkeypatch):
    manager = AsyncMock()
    manager.validate_session.side_effect = [
        {"id": 3, "user_id": 1, "device_id": "browser"},
        {"id": 4, "user_id": 1, "device_id": "single-user-cookie:v1", "expires_at": "2026-08-09T00:00:00+00:00"},
    ]
    assert await validate_single_user_session(_request(cookie="opaque"), manager) is None
    identity = await validate_single_user_session(_request(cookie="opaque"), manager)
    assert identity is not None
    assert identity.session_id == 4
```

- [ ] **Step 2: Run the tests and confirm the red state**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_single_user_session.py -q`

Expected: collection fails because `single_user_session` does not exist.

- [ ] **Step 3: Add settings and the minimal primitive**

```python
# settings.py
SINGLE_USER_SESSION_COOKIE_NAME: str = Field(default="tldw_single_user_session")
SINGLE_USER_SESSION_EXPIRE_DAYS: int = Field(default=30, ge=1, le=365)

# single_user_session.py
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import secrets

from fastapi import Request, Response

SESSION_DEVICE_ID = "single-user-cookie:v1"
SESSION_COOKIE_PATH = "/api"

@dataclass(frozen=True)
class SingleUserSessionIdentity:
    session_id: int
    user_id: int
    expires_at: datetime

@dataclass(frozen=True)
class MintedSingleUserSession:
    identity: SingleUserSessionIdentity
    cookie_token: str

async def mint_single_user_session(request: Request, manager) -> MintedSingleUserSession:
    settings = get_settings()
    expires_at = datetime.now(timezone.utc) + timedelta(days=settings.SINGLE_USER_SESSION_EXPIRE_DAYS)
    access_token = secrets.token_urlsafe(32)
    result = await manager.create_session(
        user_id=int(settings.SINGLE_USER_FIXED_ID),
        access_token=access_token,
        refresh_token=secrets.token_urlsafe(32),
        ip_address=resolve_client_ip(request, settings),
        user_agent=request.headers.get("user-agent"),
        device_id=SESSION_DEVICE_ID,
        expires_at_override=expires_at,
        refresh_expires_at_override=expires_at,
    )
    identity = SingleUserSessionIdentity(int(result["session_id"]), int(result["user_id"]), expires_at)
    return MintedSingleUserSession(identity=identity, cookie_token=access_token)

async def validate_single_user_session(request, manager=None) -> SingleUserSessionIdentity | None:
    settings = get_settings()
    if settings.AUTH_MODE != "single_user":
        return None
    token = request.cookies.get(settings.SINGLE_USER_SESSION_COOKIE_NAME)
    if not token:
        return None
    session_manager = manager or await get_session_manager()
    row = await session_manager.validate_session(token)
    if not row or row.get("device_id") != SESSION_DEVICE_ID:
        return None
    if int(row.get("user_id", 0)) != int(settings.SINGLE_USER_FIXED_ID):
        return None
    expires_at = _as_aware_datetime(row["expires_at"])
    return SingleUserSessionIdentity(int(row["id"]), int(row["user_id"]), expires_at)
```

Add `s.device_id` to both validation SELECT projections and returned mappings in `sessions_repo.py`. Implement cookie set/delete helpers with the exact attributes from Global Constraints and no `Domain` attribute.

- [ ] **Step 4: Run focused tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_single_user_session.py tldw_Server_API/tests/AuthNZ/unit/test_session_manager_configured_key.py tldw_Server_API/tests/AuthNZ/unit/test_session_manager_token_metadata.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit the primitive**

```bash
git add tldw_Server_API/app/core/AuthNZ/single_user_session.py tldw_Server_API/app/core/AuthNZ/settings.py tldw_Server_API/app/core/AuthNZ/repos/sessions_repo.py tldw_Server_API/tests/AuthNZ/unit/test_single_user_session.py
git commit -m "feat(auth): add opaque single-user sessions"
```

### Task 2: HTTP Principal, CSRF, Mint, and Logout

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/csrf_protection.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/auth.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_auth_principal_service_and_single_user_tokens.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_csrf_binding.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_single_user_cookie_session.py`

**Interfaces:**
- Consumes: Task 1 session identity and cookie helpers.
- Produces: `POST/DELETE /api/v1/auth/single-user/session`, cookie fallback in `get_auth_principal()`, `resolve_effective_csrf_enabled()`.

- [ ] **Step 1: Write failing HTTP and CSRF tests**

```python
def test_mint_returns_no_token_and_sets_exact_cookie(client, single_user_key):
    response = client.post(
        "/api/v1/auth/single-user/session",
        headers={"X-API-KEY": single_user_key},
    )
    assert response.status_code == 200
    assert response.json().keys() == {"authenticated", "expires_at"}
    cookie = response.headers["set-cookie"]
    assert "tldw_single_user_session=" in cookie
    assert "HttpOnly" in cookie and "SameSite=lax" in cookie and "Path=/api" in cookie
    assert single_user_key not in response.text + cookie

def test_cookie_mutation_requires_csrf_but_api_key_does_not(client, single_user_key):
    mint = client.post("/api/v1/auth/single-user/session", headers={"X-API-KEY": single_user_key})
    client.cookies.update(mint.cookies)
    assert client.delete("/api/v1/auth/single-user/session").status_code == 403
    assert client.post("/api/v1/auth/single-user/session", headers={"X-API-KEY": single_user_key}).status_code == 200

def test_csrf_disabled_refuses_cookie_mint(client, single_user_key, monkeypatch):
    monkeypatch.setenv("CSRF_ENABLED", "0")
    assert client.post("/api/v1/auth/single-user/session", headers={"X-API-KEY": single_user_key}).status_code == 503
```

- [ ] **Step 2: Run focused tests and confirm failures**

Run: `source .venv/bin/activate && CSRF_ENABLED=1 python -m pytest tldw_Server_API/tests/AuthNZ/integration/test_single_user_cookie_session.py tldw_Server_API/tests/AuthNZ/unit/test_csrf_binding.py -q`

Expected: session routes are 404 and cookie principal cases fail.

- [ ] **Step 3: Implement one effective-CSRF resolver and HTTP fallback**

```python
# csrf_protection.py
def resolve_effective_csrf_enabled() -> bool:
    configured = global_settings.get("CSRF_ENABLED")
    raw = os.getenv("CSRF_ENABLED")
    if raw is not None:
        configured = bool(is_truthy(raw.strip().lower()))
    if configured is not None:
        return bool(configured)
    if is_test_mode() or "pytest" in sys.modules:
        return False
    return get_settings().AUTH_MODE in {"single_user", "multi_user"}

# auth_principal_resolver.py, before the missing-credentials 401
if not token and not api_key:
    identity = await validate_single_user_session(request)
    if identity is not None:
        user = User_DB_Handling.get_single_user_instance()
        principal = _build_principal_from_user(
            user=user,
            kind="user",
            request=request,
            token_type="single_user_session",
            subject="single_user",
        )
        request.state.single_user_session_id = identity.session_id
        request.state.user_id = identity.user_id
        request.state.auth = _build_context(principal, request)
        request.state._auth_user = user
        return principal
```

Use `resolve_effective_csrf_enabled()` in `add_csrf_protection()`, protect requests only when the session cookie is present in single-user mode, and use the shared session validator inside `_resolve_user_id()` when binding is enabled. Do not add the session path to `excluded_paths`.

Implement POST with an explicit `X-API-KEY` presence check plus `Depends(get_auth_principal)`, reuse a valid cookie session, and refuse with 503 when effective CSRF is false. Implement DELETE with cookie principal resolution, exact `session_id` revocation, and cookie deletion.

- [ ] **Step 4: Run HTTP/AuthNZ regression tests**

Run: `source .venv/bin/activate && CSRF_ENABLED=1 python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_auth_principal_service_and_single_user_tokens.py tldw_Server_API/tests/AuthNZ/unit/test_csrf_binding.py tldw_Server_API/tests/AuthNZ/integration/test_single_user_cookie_session.py -q`

Expected: all selected tests pass, including bound-CSRF and exact logout cases.

- [ ] **Step 5: Commit HTTP cookie authentication**

```bash
git add tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py tldw_Server_API/app/core/AuthNZ/csrf_protection.py tldw_Server_API/app/api/v1/endpoints/auth.py tldw_Server_API/tests/AuthNZ
git commit -m "feat(auth): authenticate single-user cookie sessions"
```

### Task 3: Shared Cookie WebSocket Authentication

**Files:**
- Create: `tldw_Server_API/app/core/AuthNZ/websocket_session_auth.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/acp_multiplex.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/persona.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/workflows.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/meetings.py`
- Modify: `tldw_Server_API/app/api/v1/API_Deps/Meetings_DB_Deps.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_websocket.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/sandbox.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/voice_assistant.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py`
- Modify: `tldw_Server_API/app/core/Audio/streaming_service.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_websocket_session_auth.py`
- Test: `tldw_Server_API/tests/AuthNZ/test_websocket_cookie_route_contract.py`

**Interfaces:**
- Consumes: `validate_single_user_session()` and runtime allowed origins.
- Produces: `resolve_single_user_cookie_websocket(websocket) -> SingleUserSessionIdentity | None`.

- [ ] **Step 1: Write failing Origin and route-contract tests**

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("origin", [None, "null", "https://evil.example", "not-a-url"])
async def test_cookie_websocket_rejects_untrusted_origin(origin, monkeypatch):
    websocket = fake_websocket(origin=origin, cookie="opaque")
    assert await resolve_single_user_cookie_websocket(websocket) is None
    assert websocket.state.single_user_session_id is None

@pytest.mark.asyncio
async def test_cookie_websocket_accepts_exact_loopback_origin(monkeypatch):
    websocket = fake_websocket(origin="http://127.0.0.1:3000", cookie="opaque")
    monkeypatch.setattr(session_auth, "trusted_webui_origins", lambda: {"http://127.0.0.1:3000"})
    monkeypatch.setattr(session_auth, "validate_single_user_session", AsyncMock(return_value=identity(9, 1)))
    assert (await resolve_single_user_cookie_websocket(websocket)).session_id == 9

def test_first_party_websocket_routes_import_shared_cookie_resolver():
    for path in FIRST_PARTY_WEBSOCKET_AUTH_FILES:
        assert "resolve_single_user_cookie_websocket" in Path(path).read_text()
```

- [ ] **Step 2: Run tests and confirm failures**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_websocket_session_auth.py tldw_Server_API/tests/AuthNZ/test_websocket_cookie_route_contract.py -q`

Expected: resolver import and route contract fail.

- [ ] **Step 3: Implement the shared fallback and adopt it everywhere**

```python
async def resolve_single_user_cookie_websocket(websocket: WebSocket):
    settings = get_settings()
    if settings.AUTH_MODE != "single_user":
        return None
    raw_origin = websocket.headers.get("origin")
    origin = normalize_http_origin(raw_origin)
    allowed = trusted_webui_origins()
    if origin is None or origin not in allowed or "*" in allowed:
        return None
    identity = await validate_single_user_session(websocket)
    if identity is None:
        return None
    websocket.state.single_user_session_id = identity.session_id
    websocket.state.user_id = identity.user_id
    return identity
```

In each endpoint's existing auth helper, preserve explicit header/subprotocol/query authentication first, then call the shared resolver only when no explicit credential was supplied. Run endpoint-specific ownership/scope checks after resolving `identity.user_id`. Close 4401 for invalid session and 4403 for untrusted Origin before `accept()`. The route-contract fixture enumerates every `@router.websocket`, `@ws_router.websocket`, and mounted MCP WebSocket entry point intended for the WebUI and maps it to the shared resolver call.

- [ ] **Step 4: Run WebSocket/AuthNZ tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_websocket_session_auth.py tldw_Server_API/tests/AuthNZ/test_websocket_cookie_route_contract.py tldw_Server_API/app/core/MCP_unified/tests/test_security_hardening.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit shared WebSocket authentication**

```bash
git add tldw_Server_API/app/core/AuthNZ/websocket_session_auth.py tldw_Server_API/app/api/v1 tldw_Server_API/app/core/Audio/streaming_service.py tldw_Server_API/tests/AuthNZ
git commit -m "feat(auth): support cookie-authenticated websockets"
```

### Task 4: Non-Secret Runtime Capability and Next Bootstrap

**Files:**
- Create: `apps/tldw-frontend/pages/api/_tldw-webui/runtime-auth-policy.ts`
- Create: `apps/tldw-frontend/pages/api/_tldw-webui/session.ts`
- Modify: `apps/tldw-frontend/pages/api/_tldw-webui/runtime-config.ts`
- Test: `apps/tldw-frontend/__tests__/pages/api/runtime-config.test.ts`
- Test: `apps/tldw-frontend/__tests__/pages/api/runtime-session.test.ts`
- Test: `apps/tldw-frontend/__tests__/frontend-quickstart-networking.test.ts`

**Interfaces:**
- Consumes: backend POST session endpoint and current loopback runtime-auth checks.
- Produces: `runtimeAuth.transport="cookie-session"`, same-origin `POST /api/_tldw-webui/session`.

- [ ] **Step 1: Write failing route tests**

```ts
it("never serializes the runtime API key", async () => {
  const { req, res } = makeApiRequest({ method: "GET", host: "127.0.0.1:3000" })
  await runtimeConfig(req, res)
  expect(res.body.runtimeAuth).toEqual({
    available: true,
    authMode: "single-user",
    transport: "cookie-session"
  })
  expect(JSON.stringify(res.body)).not.toContain(process.env.SINGLE_USER_API_KEY)
})

it("forwards separate auth and csrf cookies from the fixed internal target", async () => {
  mockFetch.mockResolvedValue(backendResponseWithCookies([
    "tldw_single_user_session=opaque; Path=/api; HttpOnly; SameSite=Lax",
    "csrf_token=csrf; Path=/; SameSite=Lax"
  ]))
  await sessionRoute(sameOriginPost(), res)
  expect(mockFetch).toHaveBeenCalledWith(
    "http://app:8000/api/v1/auth/single-user/session",
    expect.objectContaining({ headers: expect.objectContaining({ "X-API-KEY": expect.any(String) }) })
  )
  expect(res.headers["Set-Cookie"]).toHaveLength(2)
})
```

- [ ] **Step 2: Run route tests and confirm failures**

Run: `cd apps && bunx vitest run tldw-frontend/__tests__/pages/api/runtime-config.test.ts tldw-frontend/__tests__/pages/api/runtime-session.test.ts`

Expected: runtime config still contains `apiKey` and session route is missing.

- [ ] **Step 3: Extract the guard and implement bootstrap**

```ts
export type RuntimeAuthPolicy =
  | { available: true; apiKey: string; internalApiOrigin: string }
  | { available: false; reason: string }

export const resolveRuntimeAuthPolicy = (req: NextApiRequest): RuntimeAuthPolicy => {
  if (process.env.AUTH_MODE !== "single_user") return { available: false, reason: "auth-mode" }
  if (process.env.TLDW_WEBUI_EXPOSE_RUNTIME_AUTH !== "1") return { available: false, reason: "disabled" }
  if (deploymentMode() !== "quickstart") return { available: false, reason: "deployment-mode" }
  if (!isLoopbackHost(req.headers.host)) return { available: false, reason: "host" }
  if (!isTrustedLocalPeer(req.socket.remoteAddress)) return { available: false, reason: "peer" }
  if (hasForwardingHeaders(req)) return { available: false, reason: "forwarded" }
  if (!isUsableApiKey(process.env.SINGLE_USER_API_KEY)) return { available: false, reason: "api-key" }
  return { available: true, apiKey: process.env.SINGLE_USER_API_KEY, internalApiOrigin: validatedInternalOrigin() }
}
```

The GET route returns only capability metadata. The POST route requires exact same-origin `Origin`, rejects cross-site Fetch Metadata, forwards only Cookie/User-Agent plus server-side `X-API-KEY`, reads cookies with `response.headers.getSetCookie()`, filters by the two allowed names, and returns no secret body.

- [ ] **Step 4: Run route and networking tests**

Run: `cd apps && bunx vitest run tldw-frontend/__tests__/pages/api/runtime-config.test.ts tldw-frontend/__tests__/pages/api/runtime-session.test.ts tldw-frontend/__tests__/frontend-quickstart-networking.test.ts`

Expected: all selected tests pass.

- [ ] **Step 5: Commit server-side bootstrap**

```bash
git add apps/tldw-frontend/pages/api/_tldw-webui apps/tldw-frontend/__tests__/pages/api apps/tldw-frontend/__tests__/frontend-quickstart-networking.test.ts
git commit -m "feat(web): bootstrap HttpOnly single-user sessions"
```

### Task 5: Web Client Cookie Mode and Secret-Free WebSockets

**Files:**
- Modify: `apps/tldw-frontend/extension/shims/runtime-bootstrap.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/services/tldw/request-core.ts`
- Modify: `apps/packages/ui/src/services/persona-stream.ts`
- Modify: `apps/packages/ui/src/services/watchlists-stream.ts`
- Modify: `apps/packages/ui/src/services/prompt-studio-stream.ts`
- Modify: `apps/packages/ui/src/services/acp/connection.ts`
- Modify: `apps/packages/ui/src/hooks/useACPSession.tsx`
- Modify: `apps/packages/ui/src/services/acp/client.ts`
- Modify: `apps/packages/ui/src/services/tldw/voice-conversation.ts`
- Modify: `apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx`
- Modify: `apps/packages/ui/src/entries/background.ts`
- Test: `apps/tldw-frontend/__tests__/extension/runtime-bootstrap.test.ts`
- Test: `apps/packages/ui/src/services/tldw/__tests__/request-core.hosted.test.ts`
- Test: `apps/packages/ui/src/services/tldw/__tests__/request-core.quickstart.test.ts`
- Test: `apps/packages/ui/src/services/__tests__/persona-stream.test.ts`
- Test: `apps/packages/ui/src/services/__tests__/watchlists-stream.test.ts`
- Test: `apps/packages/ui/src/services/__tests__/prompt-studio-stream.test.ts`
- Test: `apps/packages/ui/src/services/acp/__tests__/client.test.ts`
- Test: `apps/packages/ui/src/services/__tests__/voice-conversation.test.ts`

**Interfaces:**
- Consumes: runtime capability and bootstrap route from Task 4.
- Produces: `authSource: "cookie-session"`, same-origin credentials/CSRF requests, secret-free same-origin WebSocket URLs.

- [ ] **Step 1: Write failing client tests**

```ts
it("bootstraps cookie mode without persisting an api key", async () => {
  runtimeConfig({ available: true, authMode: "single-user", transport: "cookie-session" })
  await bootstrapRuntimeAuth()
  expect(fetch).toHaveBeenCalledWith("/api/_tldw-webui/session", expect.objectContaining({ method: "POST", credentials: "include" }))
  expect(readStoredValue("tldwConfig")).not.toHaveProperty("apiKey")
})

it("uses cookies and csrf for a same-origin mutation", async () => {
  document.cookie = "csrf_token=csrf-123"
  await tldwRequest({ path: "/api/v1/notes", method: "POST", body: {} }, cookieRuntime)
  const init = fetchMock.mock.calls[0][1]
  expect(init.credentials).toBe("same-origin")
  expect(new Headers(init.headers).get("X-CSRF-Token")).toBe("csrf-123")
  expect(new Headers(init.headers).get("X-API-KEY")).toBeNull()
})

it("omits query credentials from same-origin persona websocket", () => {
  expect(buildPersonaWebSocketUrl(cookieConfig)).toBe("ws://127.0.0.1:3000/api/v1/persona/stream")
})
```

- [ ] **Step 2: Run client tests and confirm failures**

Run: `cd apps && bunx vitest run tldw-frontend/__tests__/extension/runtime-bootstrap.test.ts packages/ui/src/services/tldw/__tests__ packages/ui/src/services/__tests__/persona-stream.test.ts packages/ui/src/services/__tests__/watchlists-stream.test.ts packages/ui/src/services/__tests__/prompt-studio-stream.test.ts`

Expected: cookie transport is unknown and builders require a key.

- [ ] **Step 3: Add cookie auth source and update builders**

```ts
export interface TldwConfig {
  serverUrl: string
  apiKey?: string
  accessToken?: string
  refreshToken?: string
  orgId?: number
  authMode: "single-user" | "multi-user"
  authSource?: "manual" | "cookie-session"
}

const cookieSession = config?.authSource === "cookie-session" && transport.kind === "same-origin"
if (cookieSession) {
  requestInit.credentials = "same-origin"
  const csrf = readCookie("csrf_token")
  if (isMutation(method) && csrf) headers["X-CSRF-Token"] = csrf
} else {
  attachExistingHeaderAuth(headers, config)
}
```

Update every listed WebSocket builder to branch on `authSource === "cookie-session"`: use `resolveBrowserWebSocketBase()` for the page origin and omit auth query parameters. Preserve existing remote/extension and multi-user branches byte-for-byte where possible. Scrub ambiguous legacy runtime keys only after the cookie-authenticated profile probe succeeds.

- [ ] **Step 4: Run WebUI unit tests**

Run: `cd apps && bunx vitest run tldw-frontend/__tests__/extension/runtime-bootstrap.test.ts packages/ui/src/services/tldw/__tests__ packages/ui/src/services/__tests__/persona-stream.test.ts packages/ui/src/services/__tests__/watchlists-stream.test.ts packages/ui/src/services/__tests__/prompt-studio-stream.test.ts packages/ui/src/services/acp/__tests__/client.test.ts`

Expected: all selected tests pass and same-origin URLs contain no API key/token.

- [ ] **Step 5: Commit client cookie mode**

```bash
git add apps/tldw-frontend/extension/shims/runtime-bootstrap.ts apps/packages/ui/src/services apps/packages/ui/src/hooks/useACPSession.tsx apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx apps/packages/ui/src/entries/background.ts apps/tldw-frontend/__tests__
git commit -m "feat(web): use cookie auth for quickstart requests"
```

### Task 6: Lifecycle, Deployment, and Security Verification

**Files:**
- Modify: `Dockerfiles/docker-compose.webui.yml`
- Modify: `Dockerfiles/README.md`
- Modify: `apps/tldw-frontend/e2e/login.spec.ts`
- Create: `apps/tldw-frontend/e2e/single-user-cookie-lifecycle.spec.ts`
- Modify: `backlog/tasks/task-12108 - Add-persistent-HttpOnly-sessions-for-same-origin-single-user-WebUI-auth.md`

**Interfaces:**
- Consumes: completed backend and client cookie-session flow.
- Produces: executable hard-reload/relaunch/WebSocket evidence and documented loopback cookie settings.

- [ ] **Step 1: Add failing lifecycle assertions**

```ts
test("cookie session survives a process relaunch without browser-readable key", async () => {
  const profile = test.info().outputPath("profile")
  await withPersistentBrowser(profile, async page => {
    await page.goto(webUiUrl)
    await expect(page.getByText("Connected securely through this WebUI.")).toBeVisible()
    expect(await browserSecretInventory(page)).toEqual([])
  })
  await withPersistentBrowser(profile, async page => {
    await page.goto(webUiUrl)
    await expect(page.getByText("Connected securely through this WebUI.")).toBeVisible()
  })
})
```

Add hard reload, exact logout revocation, API-key rotation after backend restart, representative persona/ACP/audio WebSockets, and cross-Origin WebSocket rejection.

- [ ] **Step 2: Run lifecycle test and confirm the pre-configuration failure**

Run: `cd apps/tldw-frontend && bunx playwright test e2e/single-user-cookie-lifecycle.spec.ts --reporter=line`

Expected: the test fails until the E2E quickstart starts with cookie-session CSRF and loopback cookie settings.

- [ ] **Step 3: Wire explicit loopback deployment settings and documentation**

```yaml
# docker-compose.webui.yml, API service environment for the local quickstart profile
SESSION_COOKIE_SECURE: ${SESSION_COOKIE_SECURE:-0}
CSRF_ENABLED: ${CSRF_ENABLED:-1}
```

Document that these defaults apply only to loopback HTTP quickstart; TLS/non-loopback deployments keep Secure cookies and are not automatically provisioned by this release.

- [ ] **Step 4: Run complete verification**

Run backend: `source .venv/bin/activate && CSRF_ENABLED=1 python -m pytest tldw_Server_API/tests/AuthNZ -q`

Run frontend: `cd apps && bunx vitest run tldw-frontend/__tests__/pages/api/runtime-config.test.ts tldw-frontend/__tests__/pages/api/runtime-session.test.ts tldw-frontend/__tests__/extension/runtime-bootstrap.test.ts packages/ui/src/services/tldw/__tests__ packages/ui/src/services/__tests__/persona-stream.test.ts packages/ui/src/services/__tests__/watchlists-stream.test.ts packages/ui/src/services/__tests__/prompt-studio-stream.test.ts`

Run lifecycle: `cd apps/tldw-frontend && bunx playwright test e2e/single-user-cookie-lifecycle.spec.ts --reporter=line`

Run security scan: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/AuthNZ/single_user_session.py tldw_Server_API/app/core/AuthNZ/websocket_session_auth.py tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py tldw_Server_API/app/core/AuthNZ/csrf_protection.py tldw_Server_API/app/core/AuthNZ/repos/sessions_repo.py tldw_Server_API/app/api/v1/endpoints/auth.py tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py tldw_Server_API/app/api/v1/endpoints/acp_multiplex.py tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/app/api/v1/endpoints/watchlists.py tldw_Server_API/app/api/v1/endpoints/workflows.py tldw_Server_API/app/api/v1/endpoints/meetings.py tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_websocket.py tldw_Server_API/app/api/v1/endpoints/sandbox.py tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py tldw_Server_API/app/api/v1/endpoints/voice_assistant.py tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py tldw_Server_API/app/core/Audio/streaming_service.py -f json -o /tmp/bandit_task_12108.json`

Expected: all tests pass; Bandit reports no new findings in changed code.

- [ ] **Step 5: Finalize and commit TASK-12108**

Record exact commands/results, checked acceptance criteria, security scan result, known skips, and summary in Backlog.md. Then run:

```bash
git add Dockerfiles apps/tldw-frontend/e2e "backlog/tasks/task-12108 - Add-persistent-HttpOnly-sessions-for-same-origin-single-user-WebUI-auth.md"
git commit -m "test(auth): cover single-user cookie lifecycle"
```
