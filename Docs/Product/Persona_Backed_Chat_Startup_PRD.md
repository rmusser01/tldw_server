# Persona-backed Chat Startup PRD

Status: Draft

Owner: Persona module

Tracking: #1908, split from #1902

Backlog: TASK-467

## Summary

Make Persona profiles first-class assistants when starting ordinary chat sessions. This is a closeout and hardening PRD, not a greenfield assistant system: the WebUI already has a unified assistant selection model for Characters and Personas, chat session metadata already accepts `assistant_kind`, `assistant_id`, and `persona_memory_mode`, and the chat submit path already has a Persona-backed server chat helper.

The product gap is that Persona-backed ordinary chat is not yet documented as a coherent startup flow with explicit user-facing behavior, compatibility rules, persistence expectations, and validation gates. The work should reuse the current chat, Persona catalog, and conversation contracts instead of creating a parallel Persona chat runtime.

## Problem

Persona Garden is now the advanced workspace for creating, configuring, and live-testing Personas. Ordinary `/chat` startup was intentionally moved out of the completed Persona Garden PRD. Users still need a practical path to say: "start this normal chat as this Persona," then trust that the selected Persona identity, visible assistant label, stored conversation metadata, resumed session, and server-side prompt context remain aligned.

Without a focused PRD, future implementation risks mixing three different concepts:

- Character Chat: role-play and SillyTavern-style character sessions.
- Persona Garden live sessions: advanced Persona runtime, voice, tools, policies, and live timeline.
- Persona-backed ordinary chat: normal chat UI and history, seeded by a Persona identity and memory mode.

## Goals

- Let a user select a Persona as the assistant for a new ordinary chat.
- Preserve Persona selection across the startup, first send, persistence, history list, and resume flow.
- Make Persona-backed chat visibly distinct from Character Chat without forking the normal chat UI.
- Reuse the existing Persona catalog/profile APIs and chat session assistant metadata.
- Define prompt-context semantics for Persona identity, system prompt, greeting behavior, and memory mode.
- Keep the V1 scope backend/WebUI focused; no Buddy animation, Workspace defaults, scheduled work, or design-system backlog work.

## Non-goals

- No Persona Garden replacement inside `/chat`.
- No Buddy animation or expressive avatar runtime.
- No Workspace-scoped default Persona selection; that is a separate future PRD from #1902.
- No scheduled Persona jobs, daily briefs, or background autonomous work.
- No multi-agent or multi-Persona collaboration.
- No broad personalization memory layer beyond the explicit `persona_memory_mode` contract.
- No design-system backlog task migration.

## Current Contract Evidence

- The completed Persona PRD now scopes the current module to Persona Garden and live Persona sessions, and moves ordinary `/chat` Persona startup to future PRDs.
- `apps/packages/ui/src/types/assistant-selection.ts` defines `AssistantSelection` with `kind: "character" | "persona"` and conversion helpers for Characters and Personas.
- `apps/packages/ui/src/components/Sidepanel/Chat/CharacterSelect.tsx` exposes Character and Persona tabs, reads the Persona catalog, and stores Persona selections through the unified assistant selection model.
- `apps/packages/ui/src/hooks/chat/personaServerChat.ts` creates or reuses server chats with `assistant_kind: "persona"`, `assistant_id`, and `persona_memory_mode`.
- `apps/packages/ui/src/hooks/chat/useChatActions.ts` branches on Persona assistant selection and sends normal chat turns through the existing normal chat path with Persona server chat metadata.
- `tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py` accepts Persona assistant metadata on chat session create/response models.
- `tldw_Server_API/app/api/v1/schemas/chat_conversation_schemas.py` exposes assistant metadata on conversation list/detail responses.
- `tldw_Server_API/app/api/v1/endpoints/persona.py` exposes Persona profile and catalog endpoints used by the WebUI.

## Product Shape

V1 should treat Persona-backed ordinary chat as a startup and persistence feature layered onto normal chat:

1. User opens ordinary chat.
2. User opens the assistant selector.
3. User chooses a Persona from the Persona tab.
4. Chat visibly shows the active Persona name/avatar and "Persona" identity.
5. First user send creates or reuses a server chat with `assistant_kind: "persona"`, `assistant_id`, and an explicit `persona_memory_mode`.
6. Resume/history reloads restore the same Persona-backed identity before the next send.
7. Switching away from a Persona follows the same active-chat confirmation semantics as switching Character assistants.

The normal chat composer, queueing, attachments, RAG controls, model controls, and message display should remain the ordinary chat surface.

## Prompt And Context Semantics

The implementation plan should verify and, where needed, complete these semantics:

- Persona profile fields supply assistant identity and default behavior only when a Persona is selected.
- Persona-backed chat must not silently mutate the source Persona profile.
- Existing chat system prompt controls still apply, with Persona identity appended or composed through a deterministic precedence rule.
- Persona greeting behavior should be explicit. V1 can choose one of:
  - no automatic Persona greeting in ordinary chat, or
  - optional first-turn Persona greeting using existing greeting UI semantics.
- Persona memory mode defaults to `read_only` unless the user explicitly opts into `read_write`.
- `read_write` must be visible and should never be inferred from the Persona profile alone in ordinary chat V1.

## UX Requirements

- The assistant selector should clearly show Characters and Personas as sibling assistant sources.
- Persona rows should show enough identity to avoid accidental selection: name, optional avatar, and concise Persona label.
- The active assistant control should show the selected Persona name/avatar.
- Empty and unavailable states should explain when Persona support is absent from the server.
- Switching Personas or switching between Character and Persona should preserve the existing active-chat confirmation behavior.
- A resumed Persona-backed chat should visibly restore the Persona identity before sending.

## Backend Requirements

- Chat session create/update/list/detail responses must preserve `assistant_kind`, `assistant_id`, and `persona_memory_mode`.
- Backend chat execution must resolve Persona context only when conversation metadata or request metadata indicates `assistant_kind: "persona"`.
- Compatibility with legacy Character chats must remain unchanged.
- Unknown/deleted Persona references should degrade predictably:
  - history remains readable,
  - new sends are blocked with a clear recoverable error, or
  - the user is prompted to choose a replacement Persona.
- Telemetry and logs should identify Persona-backed chats without leaking unsafe Persona profile content.

## Staged Delivery

### Stage 1: Contract Audit And Tests

Goal: prove the current substrate and identify the smallest missing pieces.

Deliverables:

- Tests for `AssistantSelection` Persona normalization.
- Tests for Persona selection in the assistant selector.
- Tests for `ensurePersonaServerChat` create/reuse/reset behavior.
- Backend schema/API tests for Persona chat session metadata.

### Stage 2: Startup And Persistence Hardening

Goal: make new Persona-backed ordinary chat startup reliable.

Deliverables:

- Confirm active Persona selection creates a server chat with correct metadata on first send.
- Confirm active-chat switching confirmation applies to Persona changes.
- Confirm no stale Character ID remains after Persona selection.
- Confirm `read_only` memory mode is explicit in created sessions.

### Stage 3: Resume And History Confidence

Goal: make existing Persona-backed chats safe to reopen.

Deliverables:

- Restore Persona identity from conversation/session metadata on load.
- Display degraded state for missing/deleted Persona profiles.
- Keep message history readable even when Persona lookup fails.
- Add regression tests for Persona metadata in history/sidebar rows where supported.

### Stage 4: Prompt Context And Memory Mode Policy

Goal: codify Persona context assembly for ordinary chat.

Deliverables:

- Document precedence between global system prompt, chat system prompt, Persona prompt/defaults, and user message.
- Add tests proving Persona context is included only for Persona-backed chats.
- Keep `read_write` opt-in and visible.
- Add a recoverable error path for unsupported Persona memory writes.

## Risks

- Existing partial implementation may look complete but lack resume and deleted-Persona behavior.
- Persona and Character terminology can blur if UI labels only say "assistant."
- Prompt composition can become non-deterministic if Persona fields are appended in multiple places.
- `read_write` memory mode could surprise users if it becomes implicit.
- Workspace defaults are adjacent but must stay out of V1 to keep scope bounded.

## Open Questions For Implementation Planning

- Should Persona-backed ordinary chat support automatic first-turn greetings in V1, or leave greetings to Character Chat only?
- Should `persona_memory_mode` be selectable in the chat startup UI, or should V1 hardcode `read_only` and defer UI controls?
- Where should degraded missing-Persona state appear: assistant selector, chat header, history row, or all three?
- Does the full-page `/chat` route share enough sidepanel behavior, or does it need its own startup/resume acceptance tests?

## Acceptance Criteria

- A user can start a normal chat with a selected Persona.
- The first sent turn creates or reuses a conversation with `assistant_kind: "persona"` and the selected Persona ID.
- The active Persona identity remains visible during the chat.
- Reloading/resuming the conversation restores the Persona identity or shows a clear degraded state.
- Character Chat behavior is unchanged.
- Persona Garden live-session behavior is unchanged.
- `read_write` Persona memory is never enabled implicitly.
- Tests cover startup, metadata persistence, resume, and missing/deleted Persona degradation.

## Verification Plan

- Unit tests for assistant selection normalization and Persona server chat helper behavior.
- Component tests for Persona tab selection, active Persona display, switching confirmation, and unavailable state.
- API/schema tests for chat session create/list/detail Persona metadata.
- Integration tests for first-send Persona-backed ordinary chat and resume.
- Manual browser verification on ordinary chat for selecting a Persona, sending first turn, reloading, and switching assistants.

## References

- `Docs/Product/Persona_Agent_Design.md`
- `Docs/superpowers/specs/2026-05-21-persona-prd-reconciliation-design.md`
- `apps/packages/ui/src/types/assistant-selection.ts`
- `apps/packages/ui/src/components/Sidepanel/Chat/CharacterSelect.tsx`
- `apps/packages/ui/src/hooks/chat/personaServerChat.ts`
- `apps/packages/ui/src/hooks/chat/useChatActions.ts`
- `tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py`
- `tldw_Server_API/app/api/v1/schemas/chat_conversation_schemas.py`
- `tldw_Server_API/app/api/v1/endpoints/persona.py`
