# ADR-045: Persona Live pending-plan handoff

**Status:** Accepted
**Date:** 2026-09-05
**Backfilled from:** not backfilled
**Decision owner:** TASK-13180 implementation session
**Related task:** TASK-13180
**Related spec/plan:** Docs/superpowers/specs/2026-05-20-persona-buddy-interaction-prd-design.md

## Decision

Allow the owner-authenticated Persona session detail response to project the latest retained pending plan for an active session so Buddy can route the user to explicit review in that exact full Live session.

## Context

Buddy receives pending tool plans but does not own approval or execution. Existing full Live links identify only a persona, and resuming a session hydrates preferences without its pending plan. A session identifier is sufficient navigation context; plan bodies and tool arguments must not enter URLs, router history, browser storage, session lists, or transcript exports.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Carry the plan in navigation state or browser storage | Copies sensitive arguments outside the authenticated read boundary and becomes stale. |
| Execute or approve from Buddy | Changes the existing full Live approval authority. |
| Re-send the user's request in a fresh session | Can duplicate work and does not resume the original plan. |
| Export all pending plans | Expands disclosure and review scope unnecessarily. |

## Consequences

- Links carry persona_id, tab=live and session_id only. Navigation selects the existing session; the user still presses Connect. No automatic approval, execution, microphone capture, or provider call is introduced.
- An exact Buddy locator may open a transient Live review pane while persona setup is incomplete, only on the Live tab with the matching selected persona. The pane offers Return to setup and does not complete setup, arm setup-test success, or emit detour analytics. It exposes no other setup-gated tabs. Session connection and confirmation remain explicit and authenticated.
- GET /api/v1/persona/sessions/{session_id} may return pending_plan with plan_id and steps (idx, tool, step_type, args, description, why). Only the latest plan is projected after existing runtime-session expiry pruning, with exact user and persona ownership and persisted active-status checks. Reads do not consume plans or extend runtime retention.
- The projection is at most 100 steps and 64 KiB of encoded JSON. An oversized plan is omitted as a whole; no truncated executable plan is presented. Restarted, expired, consumed, closed and archived runtime plans yield null.
- The field is session-detail-only. List and export shapes remain unchanged. Client hydration keeps all steps unselected and presents no prior policy grant. Existing tool confirmation still loads the server-owned plan and revalidates current ownership, session lifecycle, scopes and policy before execution.
- Plans created from a persisted session retain a server-derived requires_persisted_session flag. Confirmation reloads owner-filtered persisted context and rejects disappearance or terminal state before consuming the plan. Legacy runtime-only plans retain their existing path; client payloads cannot opt out of the persisted check.
- The core SessionManager confirmation operation checks runtime ownership/retention and the server-loaded persisted lifecycle snapshot, then consumes under one in-process lock. Rejection does not consume a retained plan. This guarantees single runtime consumption, not atomicity with database lifecycle writes: those use a separate store/transaction boundary, and the persisted flags remain a read snapshot. Execution still applies its existing scope and policy checks.
- Typed user_message events echo the existing bounded client_message_id through task-local turn context in emitted plan, notice, assistant and tool envelopes. The context resets after the turn; delayed child tasks inherit the originating turn. Correlation labels identify feedback only and carry no authority. Legacy voice/confirmation frames remain uncorrelated; fatal typed-turn failures emit a scoped notice before the existing stream shutdown.
- The process-local plan remains ephemeral and may disappear before confirmation. Existing confirmation rejection is authoritative. No new persistence or approval authority is created.

## Follow-up

TASK-13180 targeted route, hydration, owner/terminal/expiry/bounds tests and real Buddy-to-Live UAT.
