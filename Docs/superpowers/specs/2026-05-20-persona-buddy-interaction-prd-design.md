# Persona Buddy Interaction PRD

Date: 2026-05-20
Status: Draft for user review
Backlog: TASK-456
Parent epic: https://github.com/rmusser01/tldw_server/issues/1510
Related PR: https://github.com/rmusser01/tldw_server/pull/1895

## Summary

Make the selected Persona Buddy a controllable live assistant from the web
desktop shell. The Buddy should remain a compact visual companion, but users
need to be able to choose or focus a Buddy, start or resume a Persona Live
session, send text, control listening state, see urgent runtime states, and
switch between active sessions without opening the full Persona Garden surface
for every interaction.

This PRD is about Buddy interaction and Persona Live control. It is not a new
visual pack editor, not a new asset-generation workflow, and not a replacement
for Persona Garden's Visuals tab.

The first implementation should be a controlled vertical slice: backend session
summaries and focus, shared frontend session control state, and a compact Buddy
popover that can start or resume a text-capable live session. Voice controls,
Persona Garden migration, and richer urgent-state handling should follow only
after that path is stable.

## Current Foundations

The repo already has most of the primitives this feature should reuse:

1. `BuddyShellHost` is mounted in the web layout and the sidepanel app root. It
   loads the active Persona visual pack, resolves visual runtime state, renders
   the Buddy dock, and records diagnostics.
2. `BuddyShellDock` supports a draggable compact dock and opens
   `BuddyShellPopover`.
3. `BuddyShellPopover` currently shows identity information and an "Open
   Visuals" path, but no live controls.
4. `personaVisualState` resolves built-in states such as `idle`, `listening`,
   `thinking`, `speaking`, `tool_running`, `approval_needed`, `error`, and
   `offline`, with support for custom manifest states when present.
5. `SpriteFrameRenderer` supports sprite-frame animation and safe fallbacks.
6. Persona Garden and the sidepanel Persona route already have live session,
   voice, wake, and WebSocket code through route-owned hooks.
7. The backend already exposes Persona profile, session, stream, and visual
   pack APIs that should remain compatible during this work.

The main missing product layer is a shared Persona Live control contract that
the Buddy shell and Persona Garden can both consume. Today too much live state
is route-local, so the global Buddy shell cannot reliably become interactive
without duplicating or guessing Persona Garden behavior.

## Open PR And Non-Overlap Review

Live GitHub state checked on 2026-05-20:

1. PR #1895, "Add Persona Visual sectioned workspace", is merged. It modified
   `VisualPackEditor`, its tests, and its Backlog task.
2. Open PRs at the time of review are unrelated to this scope:
   setup connection route-state QA, SummarizePageWorkflow product-state alert,
   Fish Audio S2 provider work, and vLLM managed instances.

This PRD intentionally avoids PR #1895's scope. The Buddy interaction work must
not redesign `VisualPackEditor`, change visual pack authoring semantics, add
auto-activation, change generated-candidate review, or alter import/export,
library, duplicate, manifest validation, or activation behavior.

When users need to choose or change a Buddy visual, this feature should route
them to the existing Visuals flow and preserve draft-first, review-before-
activation semantics.

## Goals

1. Let a new user select or confirm a default Buddy path, then interact with
   that Buddy as a Persona Live assistant from the web desktop shell.
2. Make the Buddy dock a compact control point for a focused Persona Live
   session, not only a passive animation/status widget.
3. Support multiple active or recent Persona Live sessions per user with one
   backend-recorded focused Buddy at a time.
4. Move live-session state ownership toward a shared frontend controller backed
   by a first-class backend Persona Live Control API.
5. Keep Persona Garden Live and the Buddy shell consistent by consuming the
   same control/status model.
6. Surface urgent runtime states such as approval-needed, tool-running, errors,
   recovering/offline, and listening/wake state without taking over the screen.
7. Preserve the existing Persona Visual pack stack, Buddy renderer, setup
   flows, and explicit review/activation model.

## Non-Goals

1. No Visual Pack editor layout changes.
2. No new Persona Visual manifest schema in this feature.
3. No Buddy asset generation, Codex pet import work, or default pack production.
4. No auto-activation of visual packs.
5. No automatic microphone start.
6. No automatic tool approval.
7. No mobile or narrow sidepanel implementation requirement in the first slice,
   beyond avoiding regressions in shared code.
8. No VN/CYOA behavior or routing through VN runtime state.
9. No separate assistant runtime parallel to Persona Live.
10. No proactive autonomous Buddy actions in V1.
11. No approval resolution from the compact Buddy popover in the first slice.
    Approval-needed states should deep-link to the full Live view until a
    separate approval UX is designed.

## User Outcomes

### First usable Buddy

A new user who has not deeply configured Persona Garden should be able to use
an embedded default Buddy by following the existing Buddy/Visuals setup path.
Once a Buddy is active for a persona, the web desktop shell should show it and
offer a clear way to start or resume interaction.

This PRD assumes the default Buddy selection/import/review flow exists and is
usable through Persona Garden. If a user has no active Buddy visual, the shell
may route to that flow, but it must not implement a new starter-pack picker or
activation bypass inside the interaction popover.

### Quick interaction

The user can open the Buddy popover, type a short message, and send it. If no
compatible live session exists, the system can create or resume one before
sending, while preserving the draft text if creation fails.

### Live control

The user can start, stop, listen, stop listening, arm wake behavior when
available, recover from connection failure, and open the full Persona Garden
Live surface for deeper controls.

### Multi-session focus

The user can switch the focused Buddy session when several personas or sessions
are active or recent. Focus is a user-scoped backend preference so Persona
Garden and the web desktop shell converge on the same session. The shell
renders one focused Buddy at a time and uses badges or attention rows for
urgent non-focused sessions.

### Movement feedback

When the Buddy is dragged, the renderer should use `moving_right` or
`moving_left` states when the active pack supports them. It should fall back to
the current state or idle when those states do not exist.

## Product Shape

### One Floating Buddy Shell

The web desktop should continue to render one floating Buddy shell, not one
shell per live session. The shell represents the currently focused Buddy
session. If another session needs attention, it should appear as an urgent badge
or switcher row rather than a second floating companion.

### Compact Dock

The closed dock should show:

1. focused Buddy visual state,
2. connected/recovering/offline indicator,
3. urgent badge when the focused or another active session needs action,
4. drag affordance and persisted desktop position.

During drag, horizontal movement may temporarily override the visual state with
`moving_right` or `moving_left` if the active pack supports those states.
Urgent badges and accessible status text must remain visible while the
transient movement animation is active, and the prior backend-driven visual
state should resume as soon as dragging ends.

### Popover

The popover is a compact control surface. It should include:

1. focused Buddy/persona name,
2. session status and visual pack status,
3. session switcher for active/recent sessions,
4. Start Live and Stop controls,
5. Listen and Stop Listening controls when voice support is available,
6. wake-armed or send-now controls when relevant,
7. text composer that can auto-start or resume before sending,
8. urgent runtime state area,
9. "Open Full Live View" for Persona Garden,
10. "Choose/Change Buddy" link to the existing Visuals flow,
11. diagnostics entry point for pack or live-state failures.

The popover must stay compact. Detailed setup, visual-pack editing, import,
review, activation, generation, and advanced voice configuration remain in
Persona Garden.

## Backend Persona Live Control API

Add a backend-first Persona Live Control API under the Persona module. This API
should summarize and control live sessions without replacing the existing
stream transport on day one.

### Responsibilities

The API should support:

1. listing active and recent Persona Live sessions for the current user,
2. creating or resuming a session for a persona,
3. fetching session status,
4. stopping a live session,
5. focusing a session as the current user's active Buddy target,
6. sending text into a session,
7. requesting or recording voice/listening state transitions,
8. patching session-scoped voice preferences where current behavior supports
   it,
9. exposing urgent status fields for frontend attention surfaces,
10. exposing a suggested visual state based on backend-known runtime state.

### Proposed Endpoints

Initial API shape:

1. `GET /api/v1/persona/live/sessions`
2. `POST /api/v1/persona/live/sessions`
3. `GET /api/v1/persona/live/sessions/{session_id}`
4. `POST /api/v1/persona/live/sessions/{session_id}/focus`
5. `POST /api/v1/persona/live/sessions/{session_id}/stop`
6. `POST /api/v1/persona/live/sessions/{session_id}/messages`
7. `POST /api/v1/persona/live/sessions/{session_id}/voice/control`
8. `PATCH /api/v1/persona/live/sessions/{session_id}/voice/preferences`

The exact routes can be refined during implementation planning, but the model
should stay backend-owned and user-scoped. Avoid frontend-only session lists
derived from scattered route state.

The first API slice should be read/control-light: list summaries, focus a
session, create or resume a compatible text session, stop a session, and send a
text message. Voice control endpoints can be stubbed behind capability flags or
deferred until browser-mediated capture is implemented in the shared
controller.

### Session Summary Fields

Each session summary should include at least:

1. `session_id`,
2. `persona_id`,
3. `persona_name`,
4. `connected_state`,
5. `live_state`,
6. `voice_state`,
7. `wake_state`,
8. `active_tool_name` or redacted tool summary when safe,
9. `pending_approval_count`,
10. `error_state` and recovery hint,
11. `last_activity_at`,
12. `is_focused`,
13. `focused_at`,
14. `focus_generation` or equivalent monotonic value for multi-tab conflict
   resolution,
15. `can_send_text`,
16. `can_start_voice`,
17. `can_stop`,
18. `suggested_visual_state`,
19. `allowed_actions`.

Sensitive prompt text, tool arguments, secrets, and raw provider payloads must
not appear in summaries or diagnostics.

### Focus Semantics

Focus is backend-owned and user-scoped. `POST .../focus` records the session
that should be treated as the current user's focused Buddy target across
participating web surfaces. The frontend may keep an optimistic local focus
while the request is pending, but the backend focus result is authoritative.

If multiple tabs update focus, last successful write wins. Session summaries
must expose enough ordering information, such as `focused_at` and a monotonic
`focus_generation`, for clients to resolve stale focus responses and refresh
their local state. Focus does not stop other sessions, grant approvals, or
change visual-pack activation.

If the focused session is stopped, deleted, expires, or becomes inaccessible,
the next session list response should either select the most recent active
compatible session for that user or return no focused session with a clear
reason. The frontend must not keep rendering a stale focused persona after the
backend says the focus target is gone.

### Create, Resume, And Send Semantics

Session creation and text send need explicit correlation so the Buddy composer
does not duplicate messages during retries:

1. `POST /api/v1/persona/live/sessions` should accept `persona_id`,
   `reuse_policy`, and an `idempotency_key`.
2. `reuse_policy: resume_compatible` should reuse a non-terminal session owned
   by the same user and persona when it can accept the requested input mode.
   Terminal, stopped, deleted, or incompatible sessions must not be reused.
3. `reuse_policy: create_new` should create a new session while still honoring
   the request idempotency key for retry safety.
4. `POST .../messages` should require a `client_message_id` scoped to the
   session. Repeating the same `client_message_id` must return the prior
   acknowledgement instead of enqueueing a duplicate message.
5. Message acknowledgements should return `session_id`, `message_id`,
   `client_message_id`, `status`, and a safe failure reason when rejected.
6. Allowed message statuses should be explicit, for example `accepted`,
   `queued`, `delivered`, and `rejected`.
7. Stream events should echo `client_message_id` or `message_id` so the
   frontend can reconcile composer state, retries, and delivered responses.
8. Ordering is per session. The backend should preserve accepted message order
   through a sequence number, timestamp, or existing session event ordering.

The Buddy composer flow should first create or resume a compatible session when
there is no focused sendable session, then send the text with a stable
`client_message_id`. If either step fails, the frontend keeps the draft text and
shows retry.

### Voice And Browser Authority

The backend may expose voice state, allowed actions, and commands that request
or stop live listening, but browser-side code remains the authority for
microphone permission and audio capture. No backend event may directly start
capture without a user gesture and successful browser permission.

Voice controls should therefore be modeled as a handshake:

1. backend summary exposes whether voice can be requested for the session,
2. user clicks the frontend control,
3. frontend checks browser support and microphone permission,
4. frontend starts or stops capture locally when allowed,
5. frontend reports the resulting state to the backend control endpoint,
6. backend summaries and stream events reflect the resulting state.

Permission denied, unsupported browser, device unavailable, and capture-ended
states should leave text controls usable.

Voice should not block the first Buddy interaction slice. A text-only live
control path is acceptable as long as capability flags and disabled voice
states are explicit.

### Compatibility

The existing Persona session and stream endpoints should remain compatible
while this control API is introduced. The first implementation may adapt
existing session creation and WebSocket behavior internally, then migrate
Persona Garden and Buddy shell callers onto the shared control API.

## Frontend Architecture

### Shared Controller

Introduce a shared Persona Live controller in `apps/packages/ui` that can be
used by both Persona Garden and the Buddy shell.

The controller should own:

1. session list loading and refresh,
2. focused session selection,
3. lifecycle controls,
4. text send,
5. voice control commands,
6. status normalization,
7. urgent-state derivation,
8. WebSocket event merge when a session is connected,
9. degraded/offline state when API or stream calls fail.

Recommended units:

1. `PersonaLiveControlProvider`
2. `usePersonaLiveSessions`
3. `usePersonaLiveSessionControls(sessionId)`
4. `useFocusedPersonaLiveSession`
5. `usePersonaLiveVisualState(sessionId)`

Persona Garden should consume the same controller or a compatibility adapter,
not keep a separate live-state truth once the controller exists.

### State Ownership

Keep these concepts separate:

1. Persona profile: durable assistant identity, defaults, policies, and visual
   configuration.
2. Visual pack: reviewed/active Buddy renderer asset for a persona.
3. Live session: runtime connection, voice/listening state, tools, approvals,
   errors, and stream events.
4. Focused Buddy: backend-recorded user preference deciding which session the
   shell represents across participating surfaces.
5. Buddy shell UI state: popover open/closed, optimistic pending focus, and
   persisted desktop position.

The backend owns live session truth and durable user-scoped Buddy focus. The
frontend owns shell position, popover state, and temporary optimistic focus
while backend focus changes are pending. Persona Garden owns visual editing and
activation workflows.

### Visual State Priority

When rendering the focused Buddy, resolve visual state in this order:

1. local transient drag movement state: `moving_left` or `moving_right`,
2. backend urgent states: approval needed, error, recovering/offline,
3. backend live voice/tool state: listening, thinking, speaking, tool running,
4. valid authored or custom runtime state hints from the active pack,
5. idle or offline fallback.

Missing or invalid custom states must fall back through the current
`personaVisualState` resolver rather than breaking controls.

### Runtime State Lifecycle

The control API should expose a small stable lifecycle vocabulary rather than
forcing clients to infer state from transport details:

1. `idle`: session exists but no current live activity is underway.
2. `connecting`: the backend is creating or resuming the runtime.
3. `connected`: the runtime can accept at least one input mode.
4. `recovering`: stream or provider state is degraded but retry may work.
5. `stopping`: stop has been requested but cleanup is not complete.
6. `stopped`: user or system stopped the session.
7. `error`: user-visible recovery or restart is required.

The exact enum can change during implementation, but the API must distinguish
terminal states from reusable states so create/resume, focus cleanup, and
composer retry behavior are deterministic.

## Error Handling And Recovery

1. If the live-control API cannot load sessions, the dock should still render
   the active visual pack when possible and show a recoverable offline state.
2. If session creation fails, preserve composer text and expose retry.
3. If text sending fails after session creation, show a retry path without
   duplicating the user's text silently.
4. If WebSocket streaming fails, mark the session recovering or offline and
   keep non-stream controls available when the backend reports they are safe.
5. If microphone permission fails, leave text interaction usable and surface a
   clear permission state.
6. If voice controls are unavailable for the browser or current backend, hide
   or disable only those controls, not the whole Buddy interaction surface.
7. If microphone permission is denied, unsupported, or revoked, the backend
   should receive a safe state update and the frontend should keep text
   interaction available.
8. If visual pack load or render fails, keep live controls usable with a safe
   fallback visual state.
9. Urgent badges should remain until the underlying approval, tool, error, or
   recovery condition is resolved, stopped, or explicitly dismissed where
   dismissal is safe.
10. Approval-needed states should not expose inline approve/reject buttons in
    the compact popover for V1. The popover should focus the session and route
    to the full Live view where existing approval context can be shown.

## Safety And Privacy

1. All live-control endpoints must be authenticated and user-scoped.
2. Users must not see or control sessions they do not own.
3. Persona IDs and session IDs must be validated against user ownership.
4. Do not auto-start microphone capture from backend state or stream events.
5. Do not auto-approve tools or hidden actions from Buddy controls.
6. Do not expose raw prompts, provider payloads, secrets, or sensitive tool
   arguments in shell summaries.
7. Stop and cleanup actions should be explicit and idempotent where practical.
8. Visual pack selection remains draft-first and review-first through Persona
   Garden.

## Testing And Acceptance Criteria

### Backend

1. Session list is user-scoped and excludes other users' sessions.
2. Create/resume honors idempotency keys for the same persona and user.
3. Multiple active sessions can exist and be summarized for one user.
4. Focus updates one backend-recorded focused session without deleting other
   sessions, and last-write-wins multi-tab behavior is test-covered.
5. Text send can create or resume a compatible session before enqueueing a
   message.
6. Stop is idempotent and reports stopped/recoverable state clearly.
7. Voice control rejects invalid transitions with safe error messages.
8. Duplicate text send retries with the same `client_message_id` do not enqueue
   duplicate messages.
9. Message acknowledgements and stream events expose correlation IDs.
10. Backend voice state cannot start browser capture without frontend
   permission mediation.
11. Stop-listening is idempotent and safe when capture has already ended.
12. Summaries redact prompts, secrets, raw tool arguments, and provider
   payloads.
13. Existing Persona session and stream tests continue to pass.
14. Stopped, expired, deleted, or inaccessible focused sessions are cleaned up
   or reported as unfocused without stale persona rendering.
15. Contract tests cover the session summary schema, allowed lifecycle values,
   capability flags, and redaction rules.

### Frontend Unit And Integration

1. Shared provider loads sessions and exposes a focused session.
2. Urgent non-focused sessions create a Buddy badge or switcher attention row.
3. Composer preserves text across failed create/send attempts.
4. Start, stop, listen, wake, focus, and send controls call the shared
   controller, not route-local duplicate logic.
5. Permission denied, unsupported browser, and capture-ended voice states leave
   text controls usable and do not report active capture.
6. Composer retries reuse stable `client_message_id` values until the draft is
   accepted or edited.
7. Dragging the Buddy uses `moving_right` and `moving_left` when supported and
   falls back safely when unsupported.
8. Visual pack failure does not remove live controls.
9. Persona Garden Live uses the shared controller or compatibility adapter
   without regressing current live behavior.
10. Existing `VisualPackEditor` sectioned workspace tests remain intact.
11. A text-only capability path works when voice is disabled, unsupported, or
   deferred.
12. Approval-needed states route to the full Live view and do not approve or
   reject from the compact popover.

### Browser / E2E

1. A web desktop user can open the Buddy dock, start or resume Live, send text,
   and stop the session.
2. A user with two active sessions can switch the focused Buddy from the
   popover and see the same focus in Persona Garden after refresh.
3. An approval-needed or error state appears as an urgent badge and can be
   focused.
4. A missing active visual pack routes Choose/Change Buddy to the existing
   Visuals flow without bypassing review/activation.
5. Dragging the dock shows movement-state fallback behavior without visual
   crashes.
6. Browser permission denial for microphone leaves text send available and does
   not mark capture active.
7. Persona Garden Live still works after the shared controller is introduced.

## Implementation Staging Recommendation

This PRD should be implemented in narrow slices after written-spec approval:

1. Backend Persona Live Control API summaries, lifecycle fields, focus
   ownership, and text-capable create/resume/stop/send controls.
2. Shared frontend Persona Live controller consuming the new API for session
   list, focus, lifecycle, text send, retry-safe composer state, and visual
   state hints.
3. Buddy shell popover text interaction: focused session, Start/Stop, composer,
   switcher, urgent badge, Choose/Change routing, and movement-state fallback.
4. Voice capability handshake and browser-mediated listening controls once the
   text path is stable.
5. Persona Garden migration to consume the shared controller or adapter.
6. Browser/E2E coverage for Buddy interaction and Persona Garden compatibility.

Each slice should preserve existing live-session behavior and keep visual-pack
editing untouched unless a separate Visuals issue explicitly scopes it.

Do not merge all slices as one large PR unless implementation proves the shared
controller cannot be reviewed independently. The lowest-risk sequence is API
contract first, controller second, compact Buddy interaction third.

## Exit Criteria

The effort is complete when:

1. The backend exposes a documented, user-scoped Persona Live Control API.
2. The shared frontend controller is the primary live-state source for Buddy
   shell interaction.
3. The Buddy dock/popover can start, stop, focus, switch, send text, and control
   voice/listening state where supported.
4. Urgent runtime states are visible from the compact shell.
5. Missing or broken visual packs do not block live controls.
6. Existing Persona Garden Visual workflows from PR #1895 remain unchanged.
7. Persona Garden Live remains compatible or is migrated onto the shared
   controller without losing current behavior.
8. Backend, frontend, and browser tests cover the accepted workflows.
