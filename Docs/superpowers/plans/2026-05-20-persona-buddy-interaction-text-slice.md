# Persona Buddy Interaction Text Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first text-capable Persona Buddy interaction slice: backend live-control summaries/focus/create/stop, shared frontend control state, and a compact Buddy popover that can start/resume a session and send text through the existing Persona Live stream.

**Architecture:** Add a backend live-control facade over the existing Persona session DB and in-memory `SessionManager`, persisting focus metadata in existing session preferences to avoid a schema migration and adding a minimal in-memory stream-presence registry updated by the existing Persona WebSocket handler. Add a shared UI service/controller that consumes the new REST control endpoints for session state while using the existing `/api/v1/persona/stream` WebSocket for actual text-turn delivery. Extend the existing Buddy shell components to consume that controller without changing `VisualPackEditor` or Persona Visual pack workflows.

**Tech Stack:** FastAPI, Pydantic, existing `CharactersRAGDB` persona session APIs, existing Persona WebSocket stream, React, Zustand-adjacent hooks, Vitest, pytest, Playwright.

---

## Scope Boundary

This plan implements the first PRD slice from `Docs/superpowers/specs/2026-05-20-persona-buddy-interaction-prd-design.md`.

In scope:

1. Backend live-control summary/focus/create/stop endpoints.
2. Backend lifecycle, capability, redaction, and stale-focus behavior.
3. Shared frontend live-control service and hook.
4. Buddy dock/popover text controls: status, session switcher, start/stop, composer, urgent routing, Choose/Change Buddy link.
5. Existing WebSocket delivery for text turns, with `client_message_id` included in sent payloads for forward compatibility.
6. Focused backend, frontend, and browser tests for the first slice.

Out of scope for this plan:

1. No Persona Garden Live migration.
2. No browser microphone/listening controls.
3. No REST `/messages` endpoint for full turn processing yet.
4. No inline approve/reject in the compact Buddy popover.
5. No `VisualPackEditor` behavior changes.
6. No visual pack manifest or asset changes.

Important implementation note: the current real Persona turn processor lives inside the WebSocket handler in `tldw_Server_API/app/api/v1/endpoints/persona.py`. Do not add a fake REST message endpoint that only acknowledges a message without producing a real assistant response. The first text slice should use the new REST control endpoints for session state and the existing WebSocket stream for text delivery. A later slice can extract the turn processor and add REST `/messages` once REST delivery can produce the same runtime behavior as the stream.

## File Structure

Backend:

1. Modify `tldw_Server_API/app/api/v1/schemas/persona.py`
   - Add request/response schemas for live-control summaries, focus, create/resume, stop, and capability fields.
2. Create `tldw_Server_API/app/core/Persona/session_materialization.py`
   - Extract the existing endpoint-local session creation/resume materialization path so `/persona/session` and live control create/resume produce the same DB rows, scope snapshots, default preferences, activity surface, and in-memory session synchronization.
3. Create `tldw_Server_API/app/core/Persona/live_control.py`
   - Own live-control summary derivation, stream-presence tracking, focus preference helpers, create/resume policy, stop behavior, lifecycle mapping, capability flags, and redaction.
4. Modify `tldw_Server_API/app/api/v1/endpoints/persona.py`
   - Replace the endpoint-local session materialization block in `/session` with the extracted helper.
   - Wire the new `/live/sessions` endpoints to the service.
   - Keep existing `/session`, `/sessions`, and `/stream` behavior intact.
   - Include `client_message_id` from WebSocket `user_message` payloads in safe turn metadata where appropriate.
5. Create `tldw_Server_API/tests/Persona/test_persona_live_control_api.py`
   - Endpoint and service behavior tests for ownership, focus, lifecycle, create/resume, stop, stale focus, and redaction.

Frontend service/controller:

1. Create `apps/packages/ui/src/services/persona-live-control.ts`
   - Typed API helpers and normalizers for the backend control endpoints.
2. Create `apps/packages/ui/src/services/__tests__/persona-live-control.test.ts`
   - Unit tests for normalizers and request helpers.
3. Modify `apps/packages/ui/src/services/tldw/openapi-guard.ts`
   - Add new endpoint paths used by the UI.
4. Modify `apps/packages/ui/src/services/tldw/server-capabilities.ts`
   - Treat `/api/v1/persona/live/sessions` as a Persona Live control capability when present.
5. Create `apps/packages/ui/src/hooks/usePersonaLiveControl.tsx`
   - Shared controller hook for loading summaries, focusing, starting/resuming, stopping, maintaining one Buddy stream socket, and sending text.
6. Create `apps/packages/ui/src/hooks/__tests__/usePersonaLiveControl.test.tsx`
   - Hook tests with mocked REST calls and fake WebSocket.

Buddy shell UI:

1. Modify `apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellHost.tsx`
   - Resolve Buddy shell target from backend-focused live control when available, falling back to existing render context.
   - Pass live-control props to dock/popover.
2. Modify `apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellDock.tsx`
   - Add compact lifecycle/status and urgent badge rendering without breaking visual diagnostics.
3. Modify `apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellPopover.tsx`
   - Add session switcher, start/stop controls, text composer, Open Full Live View link, and approval-needed deep-link behavior.
4. Modify `apps/packages/ui/src/types/persona-buddy.ts`
   - Add optional live-control surface fields needed by Buddy shell components.
5. Modify `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx`
   - Cover backend focus, text-only controls, stale visual fallback, movement state plus urgent badge, and no Visuals bypass.
6. Create `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/BuddyShellPopover.test.tsx`
   - Focused popover behavior tests.

Browser/E2E:

1. Create `apps/tldw-frontend/e2e/workflows/persona-buddy-interaction.spec.ts`
   - Mock API and WebSocket enough to verify the desktop Buddy flow.
2. Do not modify `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`.
3. Run existing `VisualPackEditor` tests as a guardrail.

## Task 1: Backend Live-Control API

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/persona.py`
- Create: `tldw_Server_API/app/core/Persona/session_materialization.py`
- Create: `tldw_Server_API/app/core/Persona/live_control.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/persona.py`
- Test: `tldw_Server_API/tests/Persona/test_persona_live_control_api.py`
- Guardrail Test: `tldw_Server_API/tests/Persona/test_persona_sessions.py`

- [ ] **Step 1: Write failing backend live-control tests**

Create `tldw_Server_API/tests/Persona/test_persona_live_control_api.py` using the same app/dependency override pattern as `test_persona_sessions.py`.

Include tests named:

```python
def test_live_sessions_requires_auth(): ...
def test_live_sessions_list_returns_owned_session_summaries(): ...
def test_live_session_create_resume_compatible_reuses_active_session(): ...
def test_live_session_create_new_honors_idempotency_key(): ...
def test_live_session_create_uses_existing_session_materialization(): ...
def test_live_session_created_by_live_control_can_resume_existing_session_endpoint(): ...
def test_live_session_focus_last_write_wins(): ...
def test_live_session_focus_a_then_b_only_marks_b_focused(): ...
def test_live_session_focus_rejects_other_user_session(): ...
def test_live_session_stop_marks_closed_and_clears_focus(): ...
def test_live_sessions_ignore_stale_focused_closed_session(): ...
def test_live_session_rest_created_without_stream_is_idle(): ...
def test_live_session_active_stream_presence_is_connected(): ...
def test_live_session_terminal_status_excludes_send_text_action(): ...
def test_live_session_summary_redacts_sensitive_preferences(): ...
```

Expected first failure: imports or `/api/v1/persona/live/sessions` routes do not exist.

- [ ] **Step 2: Run the failing backend tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Persona/test_persona_live_control_api.py -q
```

Expected: FAIL because the new module/routes/schemas are missing.

- [ ] **Step 3: Add live-control schemas**

In `tldw_Server_API/app/api/v1/schemas/persona.py`, add Pydantic models near the existing `PersonaSession*` models:

```python
PersonaLiveLifecycle = Literal[
    "idle",
    "connecting",
    "connected",
    "recovering",
    "stopping",
    "stopped",
    "error",
]

PersonaLiveReusePolicy = Literal["resume_compatible", "create_new"]

class PersonaLiveSessionCreateRequest(BaseModel):
    persona_id: str
    reuse_policy: PersonaLiveReusePolicy = "resume_compatible"
    idempotency_key: str | None = Field(default=None, max_length=128)
    surface: str | None = Field(default=None, max_length=120)

class PersonaLiveSessionSummary(BaseModel):
    session_id: str
    persona_id: str
    persona_name: str
    lifecycle: PersonaLiveLifecycle
    status: PersonaSessionStatus | None = None
    is_focused: bool = False
    focused_at: str | None = None
    focus_generation: int | None = None
    last_activity_at: str | None = None
    pending_approval_count: int = 0
    active_tool_name: str | None = None
    error_state: str | None = None
    recovery_hint: str | None = None
    suggested_visual_state: str | None = None
    allowed_actions: list[str] = Field(default_factory=list)
    capabilities: dict[str, bool] = Field(default_factory=dict)

class PersonaLiveSessionListResponse(BaseModel):
    sessions: list[PersonaLiveSessionSummary] = Field(default_factory=list)
    focused_session_id: str | None = None

class PersonaLiveSessionFocusResponse(BaseModel):
    session: PersonaLiveSessionSummary

class PersonaLiveSessionStopResponse(BaseModel):
    session: PersonaLiveSessionSummary
```

Add validators for `persona_id`, `idempotency_key`, and `surface` that trim whitespace and reject empty required `persona_id`. Keep max lengths bounded.

- [ ] **Step 4: Implement shared session materialization and `live_control.py` service**

Create `tldw_Server_API/app/core/Persona/session_materialization.py` with a module docstring. Move the existing session materialization responsibilities out of `tldw_Server_API/app/api/v1/endpoints/persona.py` and into this helper module before adding live-control create/resume.

At minimum, extract or wrap the endpoint-local logic currently represented by:

```python
_ensure_default_persona_profile(...)
_build_scope_snapshot(...)
_default_persisted_persona_session_preferences(...)
_get_session_preferences_with_activity_surface(...)
```

Expose a focused helper such as:

```python
@dataclass(frozen=True)
class MaterializedPersonaSession:
    session_id: str
    persona_id: str
    profile: dict[str, Any]
    session_row: dict[str, Any]
    created_new_session: bool
    scope_audit: dict[str, object]
    activity_surface: str

def materialize_persona_session(
    db: CharactersRAGDB,
    *,
    session_manager: SessionManager,
    user_id: str,
    persona_id: str | None,
    resume_session_id: str | None = None,
    project_id: str | None = None,
    surface: str | None = None,
) -> MaterializedPersonaSession: ...
```

Rules:

1. Update the existing `/api/v1/persona/session` endpoint to call `materialize_persona_session()` instead of keeping a second local creation path.
2. `create_or_resume_live_session()` must call the same helper when creating a new session or idempotently materializing a requested session.
3. Do not import from `api/v1/endpoints/persona.py` inside core code; extract helpers downward to avoid a circular dependency.
4. Keep `/persona/session` response shape and telemetry behavior unchanged.
5. Add tests that a live-control-created session has a scope snapshot, default persisted preferences, activity surface, and can be passed back to `/api/v1/persona/session` through `resume_session_id`.

Then create `tldw_Server_API/app/core/Persona/live_control.py` with a module docstring.

Implement these focused helpers:

```python
LIVE_CONTROL_PREFS_KEY = "persona_live_control"

class PersonaLiveStreamRegistry:
    def mark_connected(...): ...
    def mark_disconnected(...): ...
    def is_connected(...): ...

def build_live_session_summary(...): ...
def list_live_session_summaries(...): ...
def create_or_resume_live_session(...): ...
def focus_live_session(...): ...
def stop_live_session(...): ...
```

Storage rules:

1. Store focus metadata inside the existing session `preferences_json` under `persona_live_control.focus`.
2. Store create idempotency keys under `persona_live_control.create_idempotency_key`.
3. Do not add a DB migration for this slice.
4. Redact unknown preference fields from API summaries.
5. Treat persisted `status` values `closed` and `archived` as terminal.
6. `resume_compatible` can reuse the most recent non-terminal session for the same user/persona/surface; if no reusable session exists, materialize the new session through `materialize_persona_session()`.
7. `create_new` creates a new session through `materialize_persona_session()` unless an existing session has the same idempotency key.
8. `focus_generation` can be `int(time.time() * 1000)` or another monotonic-ish integer; tests should assert ordering rather than exact value.
9. Enforce one focused session in list responses by deriving the focused record from the highest valid `focus_generation` for the current user; `focus_live_session()` should also clear/demote older focus metadata when practical so stale records do not accumulate.
10. Track active WebSocket stream presence separately from `SessionManager` rows. REST-created sessions and stale in-memory manager rows are not enough to report `connected`.

Summary mapping:

```python
if status in {"closed", "archived"}:
    lifecycle = "stopped"
elif stream_registry.is_connected(user_id=user_id, session_id=session_id):
    lifecycle = "connected"
else:
    lifecycle = "idle"
```

Allowed actions for the first slice:

```python
if status in {"closed", "archived"}:
    allowed_actions = []
else:
    allowed_actions = ["focus", "stop", "send_text_ws"]
```

Do not expose `send_text_ws` for terminal sessions. The frontend must treat a focused stopped session as not sendable and create/resume a compatible session before sending.

Capabilities for the first slice:

```python
{
    "text": True,
    "voice": False,
    "browser_microphone_required": False,
}
```

- [ ] **Step 5: Wire FastAPI routes**

In `tldw_Server_API/app/api/v1/endpoints/persona.py`, import the new schemas and service helpers. Add routes near the existing `/session` and `/sessions` endpoints:

```python
@router.get("/live/sessions", response_model=PersonaLiveSessionListResponse, ...)
async def persona_live_sessions(...): ...

@router.post("/live/sessions", response_model=PersonaLiveSessionFocusResponse, ...)
async def persona_live_session_create(...): ...

@router.post("/live/sessions/{session_id}/focus", response_model=PersonaLiveSessionFocusResponse, ...)
async def persona_live_session_focus(...): ...

@router.post("/live/sessions/{session_id}/stop", response_model=PersonaLiveSessionStopResponse, ...)
async def persona_live_session_stop(...): ...
```

Rules:

1. Use `get_request_user` and `get_chacha_db_for_user`.
2. Use `_require_current_user_id()`.
3. Respect `is_persona_enabled()`.
4. Convert `InputError`, `ConflictError`, and `CharactersRAGDBError` through `_to_http_exception()`.
5. Return 403 for ownership mismatch and 404 for missing sessions.
6. Do not modify the existing `/session`, `/sessions`, or `/stream` response shape.

- [ ] **Step 6: Add WebSocket stream presence and `client_message_id` metadata**

In `persona_stream`, mark stream presence after the connection is accepted and the session is known:

```python
persona_live_stream_registry.mark_connected(
    user_id=current_user_id,
    session_id=session_id,
)
```

In the WebSocket cleanup/finally path, mark every session ID observed on that WebSocket disconnected. Do not only clear the last session ID, because a stream may resume or switch sessions before closing.

```python
for observed_session_id in observed_session_ids:
    persona_live_stream_registry.mark_disconnected(
        user_id=current_user_id,
        session_id=observed_session_id,
    )
```

The registry is process-local and best-effort in this slice. It should not be used for authorization or persistence, and a missing registry entry should map to `idle`, not `connected`.

In `_handle_persona_live_turn`, read:

```python
client_message_id = str(msg.get("client_message_id") or "").strip()[:128] or None
```

Add it to `_record_turn(... metadata={...})` for `user_message` and `voice_commit` turns when present. Do not trust it for authorization or deduplication in this slice. This gives the frontend a stable client-side correlation value without changing stream behavior.

- [ ] **Step 7: Run backend tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Persona/test_persona_live_control_api.py -q
python -m pytest tldw_Server_API/tests/Persona/test_persona_sessions.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit backend slice**

```bash
git add \
  tldw_Server_API/app/api/v1/schemas/persona.py \
  tldw_Server_API/app/core/Persona/session_materialization.py \
  tldw_Server_API/app/core/Persona/live_control.py \
  tldw_Server_API/app/api/v1/endpoints/persona.py \
  tldw_Server_API/tests/Persona/test_persona_live_control_api.py
git commit -m "feat(persona): add Buddy live control API"
```

## Task 2: Frontend Live-Control Service

**Files:**
- Create: `apps/packages/ui/src/services/persona-live-control.ts`
- Create: `apps/packages/ui/src/services/__tests__/persona-live-control.test.ts`
- Modify: `apps/packages/ui/src/services/tldw/openapi-guard.ts`
- Modify: `apps/packages/ui/src/services/tldw/server-capabilities.ts`

- [ ] **Step 1: Write failing service tests**

Create tests for:

```ts
it("normalizes live session summaries with safe defaults", ...)
it("preserves focused session ordering fields", ...)
it("marks text-only capability when voice is absent", ...)
it("calls create with idempotency_key and resume_compatible policy", ...)
it("calls focus and stop endpoints", ...)
```

Expected first failure: module does not exist.

- [ ] **Step 2: Run failing service tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/services/__tests__/persona-live-control.test.ts
```

Expected: FAIL because the service does not exist.

- [ ] **Step 3: Implement service types and normalizers**

Create `apps/packages/ui/src/services/persona-live-control.ts`.

Export:

```ts
export type PersonaLiveLifecycle =
  | "idle"
  | "connecting"
  | "connected"
  | "recovering"
  | "stopping"
  | "stopped"
  | "error"

export type PersonaLiveSessionSummary = {
  sessionId: string
  personaId: string
  personaName: string
  lifecycle: PersonaLiveLifecycle
  isFocused: boolean
  focusedAt: string | null
  focusGeneration: number | null
  lastActivityAt: string | null
  pendingApprovalCount: number
  suggestedVisualState: string | null
  allowedActions: string[]
  capabilities: {
    text: boolean
    voice: boolean
    browserMicrophoneRequired: boolean
  }
}
```

Export helpers:

```ts
export async function listPersonaLiveSessions(): Promise<PersonaLiveSessionList>
export async function createPersonaLiveSession(input: CreatePersonaLiveSessionInput): Promise<PersonaLiveSessionSummary>
export async function focusPersonaLiveSession(sessionId: string): Promise<PersonaLiveSessionSummary>
export async function stopPersonaLiveSession(sessionId: string): Promise<PersonaLiveSessionSummary>
```

Use `tldwClient.fetchWithAuth()` and `toAllowedPath()` rather than hard-casting `as any` in new service code.

- [ ] **Step 4: Update OpenAPI guard and capability detection**

Add paths to `ClientPath`:

```ts
| "/api/v1/persona/live/sessions"
| "/api/v1/persona/live/sessions/{session_id}/focus"
| "/api/v1/persona/live/sessions/{session_id}/stop"
```

In `server-capabilities.ts`, treat `/api/v1/persona/live/sessions` as enabling a `hasPersonaLiveControl` or equivalent existing capability field. If no typed field exists, add a narrow field and tests only where the capability object is already normalized.

- [ ] **Step 5: Run service tests**

Run:

```bash
bunx vitest run src/services/__tests__/persona-live-control.test.ts
```

Expected: PASS.

- [ ] **Step 6: Commit frontend service slice**

```bash
git add \
  apps/packages/ui/src/services/persona-live-control.ts \
  apps/packages/ui/src/services/__tests__/persona-live-control.test.ts \
  apps/packages/ui/src/services/tldw/openapi-guard.ts \
  apps/packages/ui/src/services/tldw/server-capabilities.ts
git commit -m "feat(ui): add Persona live control service"
```

## Task 3: Shared Frontend Live-Control Hook

**Files:**
- Create: `apps/packages/ui/src/hooks/usePersonaLiveControl.tsx`
- Create: `apps/packages/ui/src/hooks/__tests__/usePersonaLiveControl.test.tsx`
- Modify: `apps/packages/ui/src/services/persona-stream.ts` only if a tiny helper is needed; otherwise leave it unchanged.

- [ ] **Step 1: Write failing hook tests**

Create tests for:

```ts
it("loads sessions and chooses backend-focused session", ...)
it("focuses a session with optimistic pending state then backend result", ...)
it("starts a text session with an idempotency key", ...)
it("stops the focused session and refreshes summaries", ...)
it("opens a WebSocket and sends text with client_message_id", ...)
it("creates or resumes before sending when the focused session is stopped", ...)
it("preserves composer text when WebSocket send fails", ...)
it("reuses a caller-provided client_message_id when retrying a failed draft", ...)
it("keeps text available when voice capability is false", ...)
```

Use a fake `WebSocket` class in the test and mocked service functions from `persona-live-control.ts`.

- [ ] **Step 2: Run failing hook tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/hooks/__tests__/usePersonaLiveControl.test.tsx
```

Expected: FAIL because the hook does not exist.

- [ ] **Step 3: Implement `usePersonaLiveControl`**

Create the hook with a small reducer-style state model:

```ts
type PersonaLiveControlState = {
  sessions: PersonaLiveSessionSummary[]
  focusedSessionId: string | null
  loading: boolean
  error: string | null
  streamState: "closed" | "connecting" | "open" | "error"
  pendingFocusSessionId: string | null
}
```

Export:

```ts
export function usePersonaLiveControl(options?: {
  autoLoad?: boolean
  defaultPersonaId?: string | null
  surface?: string | null
}) {
  return {
    sessions,
    focusedSession,
    loading,
    error,
    streamState,
    reload,
    focusSession,
    startTextSession,
    stopSession,
    sendText,
  }
}
```

Rules:

1. `reload()` calls `listPersonaLiveSessions()`.
2. `focusSession(sessionId)` calls the backend and updates local focused state from the response.
3. `startTextSession(personaId)` calls `createPersonaLiveSession({ reusePolicy: "resume_compatible", idempotencyKey, surface })`.
4. `sendText(text, options)` ensures a focused text-capable session, opens `/api/v1/persona/stream` with `buildPersonaWebSocketUrl()`, and sends:

```ts
{
  type: "user_message",
  session_id: focusedSession.sessionId,
  client_message_id: options?.clientMessageId ?? generatedClientMessageId,
  text: trimmed,
}
```

5. `sendText()` returns `{ ok: true, clientMessageId }` only after `ws.send()` succeeds.
6. `sendText()` must not send to summaries that lack `send_text_ws`; if the current focused session is stopped or otherwise not sendable, call `startTextSession()` first and send to the returned session.
7. `sendText()` does not clear caller-owned composer state. The popover clears only after `ok`.
8. Voice controls are not implemented in this hook for this slice; expose capability state only.
9. Close the WebSocket on unmount.

- [ ] **Step 4: Run hook tests**

Run:

```bash
bunx vitest run src/hooks/__tests__/usePersonaLiveControl.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit hook slice**

```bash
git add \
  apps/packages/ui/src/hooks/usePersonaLiveControl.tsx \
  apps/packages/ui/src/hooks/__tests__/usePersonaLiveControl.test.tsx
git commit -m "feat(ui): add shared Persona live control hook"
```

## Task 4: Buddy Shell Text Interaction UI

**Files:**
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellHost.tsx`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellDock.tsx`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellPopover.tsx`
- Modify: `apps/packages/ui/src/types/persona-buddy.ts`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx`
- Create: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/BuddyShellPopover.test.tsx`

- [ ] **Step 1: Write failing Buddy shell tests**

Add tests that verify:

```ts
it("shows a focused live session status in the dock", ...)
it("keeps urgent badge visible while drag movement override is active", ...)
it("opens a compact popover with start stop and composer controls", ...)
it("starts a session before sending text when no focused session exists", ...)
it("preserves composer text when sendText fails", ...)
it("reuses the same client_message_id after a failed send until the draft is edited", ...)
it("routes approval-needed state to full Live view without approve buttons", ...)
it("routes Choose/Change Buddy to Visuals without activation bypass", ...)
```

Mock `usePersonaLiveControl()` so component tests stay deterministic.

- [ ] **Step 2: Run failing Buddy shell tests**

Run:

```bash
bunx vitest run \
  src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx \
  src/components/Common/PersonaBuddy/__tests__/BuddyShellPopover.test.tsx
```

Expected: FAIL until UI props and controls are added.

- [ ] **Step 3: Extend types for live-control display**

In `apps/packages/ui/src/types/persona-buddy.ts`, add small display-only types if needed:

```ts
export interface PersonaBuddyLiveStatusSummary {
  sessionId: string
  personaId: string
  personaName: string
  lifecycle: string
  pendingApprovalCount: number
  canSendText: boolean
  suggestedVisualState?: string | null
}
```

Do not add visual-pack editing fields here.

- [ ] **Step 4: Wire `BuddyShellHost` to `usePersonaLiveControl`**

In `BuddyShellHost.tsx`:

1. Call `usePersonaLiveControl({ autoLoad: true, defaultPersonaId: resolvedPersona.activePersonaId })`.
2. Prefer backend-focused persona/session when available.
3. Keep existing render-context fallback when the live-control API is absent or loading fails.
4. Pass `liveControl` props to `BuddyShellDock`.
5. Map `focusedSession.suggestedVisualState` into `renderContext.visual_state` only when it is valid for `resolvePersonaVisualState()` or can safely fall back.

- [ ] **Step 5: Add dock status and badge**

In `BuddyShellDock.tsx`:

1. Add optional props for `liveStatus`, `urgentCount`, and `streamState`.
2. Render an accessible status line such as `Connected`, `Idle`, `Recovering`, or `Needs approval`.
3. Render `data-testid="persona-buddy-urgent-badge"` when `urgentCount > 0`.
4. Keep visual diagnostics visible and do not hide the badge during drag.

- [ ] **Step 6: Add compact popover controls**

In `BuddyShellPopover.tsx`:

1. Add a session switcher when more than one live session exists.
2. Add Start and Stop buttons.
3. Add a text input/textarea and Send button.
4. On Send:
   - keep a draft-scoped `clientMessageId` in component state;
   - call `sendText(draft, { clientMessageId })`;
   - clear draft only if `{ ok: true }`;
   - clear the draft-scoped `clientMessageId` only after success or when the user edits the draft;
   - leave draft intact and show an error on failure.
5. Add "Open Full Live View" link to `buildPersonaGardenRoute({ personaId, tab: "live" })` or the current live tab key if the route helper exposes a different name.
6. Keep "Choose/Change Buddy" linked to the existing Visuals route.
7. For approval-needed state, render a notice and the full Live link. Do not render approve/reject buttons.

- [ ] **Step 7: Run Buddy shell tests**

Run:

```bash
bunx vitest run \
  src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx \
  src/components/Common/PersonaBuddy/__tests__/BuddyShellPopover.test.tsx
```

Expected: PASS.

- [ ] **Step 8: Run VisualPackEditor guardrail**

Run:

```bash
bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx
```

Expected: PASS. This confirms the PR #1895 sectioned Visual workspace behavior was not disturbed.

- [ ] **Step 9: Commit Buddy shell UI slice**

```bash
git add \
  apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellHost.tsx \
  apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellDock.tsx \
  apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellPopover.tsx \
  apps/packages/ui/src/types/persona-buddy.ts \
  apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx \
  apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/BuddyShellPopover.test.tsx
git commit -m "feat(ui): add Buddy shell text controls"
```

## Task 5: Browser Flow Coverage

**Files:**
- Create: `apps/tldw-frontend/e2e/workflows/persona-buddy-interaction.spec.ts`

- [ ] **Step 1: Write failing Playwright test**

Create an E2E test that:

1. Opens a desktop route where `BuddyShellHost` is active.
2. Mocks `/api/v1/persona/live/sessions` list/create/focus/stop responses.
3. Mocks `/api/v1/persona/visual-packs` or reuses existing visual-pack fixtures if needed.
4. Mocks the Persona stream WebSocket enough to capture the `user_message`.
5. Clicks the Buddy dock.
6. Sends text from the popover.
7. Asserts the WebSocket payload includes `session_id`, `client_message_id`, and `text`.
8. Asserts Choose/Change Buddy routes to Visuals.

- [ ] **Step 2: Run failing Playwright test**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/persona-buddy-interaction.spec.ts --reporter=line
```

Expected: FAIL until the UI is wired.

- [ ] **Step 3: Fix mocks/selectors and run again**

Run:

```bash
bunx playwright test e2e/workflows/persona-buddy-interaction.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 4: Commit E2E slice**

```bash
git add apps/tldw-frontend/e2e/workflows/persona-buddy-interaction.spec.ts
git commit -m "test(e2e): cover Buddy text interaction"
```

## Task 6: Verification, Documentation, And PR Prep

**Files:**
- Modify: `Docs/superpowers/plans/2026-05-20-persona-buddy-interaction-text-slice.md`
- Modify: `backlog/tasks/task-457 - Create-Persona-Buddy-interaction-implementation-plan.md` if executing from this plan
- Optional Modify: `Docs/superpowers/specs/2026-05-20-persona-buddy-interaction-prd-design.md` only if implementation reveals a needed scope clarification.

- [ ] **Step 1: Run focused backend verification**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Persona/test_persona_live_control_api.py \
  tldw_Server_API/tests/Persona/test_persona_sessions.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run focused frontend verification**

Run from `apps/packages/ui`:

```bash
bunx vitest run \
  src/services/__tests__/persona-live-control.test.ts \
  src/hooks/__tests__/usePersonaLiveControl.test.tsx \
  src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx \
  src/components/Common/PersonaBuddy/__tests__/BuddyShellPopover.test.tsx \
  src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx
```

Expected: PASS.

- [ ] **Step 3: Run OpenAPI path verification**

Run from `apps/packages/ui`:

```bash
bun run verify:openapi
```

Expected: PASS. If it fails because the backend OpenAPI snapshot lacks the new routes, regenerate or update the relevant spec snapshot according to the verifier output.

- [ ] **Step 4: Run E2E verification**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/persona-buddy-interaction.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 5: Run Bandit on touched backend code**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/api/v1/endpoints/persona.py \
  tldw_Server_API/app/api/v1/schemas/persona.py \
  tldw_Server_API/app/core/Persona/session_materialization.py \
  tldw_Server_API/app/core/Persona/live_control.py \
  -f json -o /tmp/bandit_persona_buddy_live_control.json
```

Expected: command exits 0 or reports no new findings in touched code. If findings appear, fix them before finalizing.

- [ ] **Step 6: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 7: Update Backlog final summary**

If executing this plan, update the relevant implementation Backlog task with:

1. files touched,
2. verification commands and outcomes,
3. known skips,
4. PR link once opened.

- [ ] **Step 8: Final commit**

```bash
git status --short
git add <remaining docs/task files>
git commit -m "docs: update Buddy interaction plan status"
```

Skip this commit if there are no remaining docs/task changes.

## Review Checklist

Before opening the PR:

1. `VisualPackEditor.tsx` has no behavior changes.
2. No Buddy visual pack activation bypass was introduced.
3. Compact popover does not approve/reject tools.
4. Voice controls are absent or disabled behind clear capability flags.
5. Text send uses existing WebSocket runtime and does not fake assistant responses.
6. Stopped or inaccessible focused sessions do not keep stale Buddy identity in the shell.
7. User/session ownership is tested on backend routes.
8. The Buddy shell still renders a visual fallback when live-control APIs fail.
