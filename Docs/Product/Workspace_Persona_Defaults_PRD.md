# Workspace Assistant Defaults PRD

Status: Draft

Owner: Persona module / Workspaces integration

Tracking: #1911, split from #1902

Backlog: TASK-468, TASK-2275

## Summary

Define Workspace-scoped assistant defaults as a broad composition contract over existing Workspaces, Persona profiles, and Workspace-scoped assistant surfaces. This PRD replaces the original Persona PRD's old `project_id` language with current Workspace terminology and keeps the work separate from Persona-backed ordinary global chat startup, scheduled work, Buddy animation, tool administration, and broad personalization memory.

The goal is not to make Workspaces own Persona profiles. The goal is to let a Workspace recommend a default assistant configuration for Workspace-bound surfaces, while preserving explicit user choices and persisting resolved assistant metadata on concrete sessions, runs, or artifacts. Chat Workspace is the first implementation target; Research Workspace, Prompt Studio, writing, audio overview, and agent/tool workflows can adopt the same contract only through later focused stages.

## Problem

Persona Garden now owns Persona profile creation, configuration, and live-session testing. Persona-backed Chat Startup now owns ordinary global chat startup as a Persona. Workspaces still need a separate contract for cases where a research workspace should carry a preferred assistant shape: for example a "literature review" Workspace might default to a careful research Persona, later pair that Persona with a workspace-appropriate voice, and eventually reference a constrained tool policy for that Workspace.

The current code has the pieces but not the product contract:

- Workspace API schemas persist workspace metadata, study-materials policy, banner fields, and audio defaults.
- Workspace chat uses `scope: { type: "workspace", workspaceId }`.
- Chat session schemas already support `scope_type: "workspace"`, `workspace_id`, `assistant_kind`, `assistant_id`, and `persona_memory_mode`.
- Chat Workspace UI already reports a runtime `selectedPersonaLabel`, but there is no documented Workspace default source of truth or precedence model.
- Other Workspace-bound surfaces have no shared assistant-default contract and would otherwise invent incompatible default semantics.

Without a PRD, Workspace defaults risk becoming implicit global state that silently overrides user choices, leaks Persona references across shared Workspaces, or mutates historical chat/run behavior when a Workspace setting changes.

## Goals

- Define a Workspace-level `assistant_defaults` object with Persona-only V1 validation and room for later assistant kinds.
- Preserve explicit session/run/artifact assistant selection over Workspace defaults.
- Keep Persona profiles user-owned and configured in Persona Garden.
- Let Workspace-bound chat and later Workspace surfaces read a Workspace default without duplicating Persona profile data.
- Document precedence between explicit surface choices, Workspace defaults, user/global assistant defaults, and server fallback.
- Define permission-aware stored and effective API response shapes.
- Make Chat Workspace the first implementation target without making later Workspace surfaces V1 blockers.
- Replace `project_id` phrasing with `workspace_id` and current Workspace terminology.

## Non-goals

- No implementation in this PRD slice.
- No Buddy animation or expressive avatar runtime.
- No design-system backlog work.
- No scheduled Persona work or recurring jobs.
- No ordinary global `/chat` Persona startup changes; that is covered by #1908.
- No cross-app semantic personalization memory layer.
- No marketplace-style tool administration.
- No multi-agent or multi-Persona collaboration.
- No voice, style, or tool-policy implementation until their separate contracts are stable.

## Current Contract Evidence

- `Docs/Product/Persona_Agent_Design.md` moves Workspace Persona Defaults out of current Persona Garden completion scope and says old `project_id` language maps to Workspaces.
- `Docs/Product/Persona_Backed_Chat_Startup_PRD.md` keeps Workspace-scoped Persona selection out of ordinary chat startup scope.
- `apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceChatPanel.tsx` creates chat scope as `{ type: "workspace", workspaceId }`.
- `apps/packages/ui/src/components/Option/ChatWorkspace/ChatWorkspacePage.tsx` tracks Workspace runtime state including `selectedPersonaLabel`.
- `apps/packages/ui/src/components/Option/ChatWorkspace/InspectorRail.tsx` displays `Model / Persona` status and falls back to `No persona selected`.
- `apps/packages/ui/src/types/workspace.ts` defines `SavedWorkspace` without Persona defaults today.
- `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py` defines Workspace create/update/response fields and already includes audio defaults.
- `tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py` supports Workspace-scoped chat session creation and Persona assistant metadata.
- `tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py` validates Workspace scope and Persona assistant identity separately.

## Product Shape

Workspace Assistant Defaults should add an explicit optional defaults object to a Workspace. The broad field name is `assistant_defaults`, but V1 accepts only Persona-backed defaults:

```json
{
  "assistant_defaults": {
    "assistant_kind": "persona",
    "assistant_id": "persona-id",
    "persona_memory_mode": "read_only",
    "voice": null,
    "style": null,
    "tool_policy_profile_id": null
  }
}
```

The object must be reference-backed:

- `assistant_kind` is `"persona"` in V1. Other assistant kinds require a future contract.
- `assistant_id` references a Persona profile available to the Workspace owner/editor at save time.
- `persona_memory_mode` defaults to `read_only`.
- `voice` remains `null` in V1. A later voice-default stage must reconcile assistant speaking voice with existing Workspace audio-overview settings before defining fields.
- `style` remains `null` in V1 until a stable style preset/default-prompt contract exists.
- `tool_policy_profile_id` remains `null` in V1 until Persona Tool Administration defines policy profile lifecycle.
- Persona name, prompt, avatar, policy, tool permissions, and other Persona content must not be snapshotted into Workspace defaults.

### Stored And Effective Defaults

Workspace APIs should distinguish the saved reference from the runtime-safe resolved default:

- `assistant_defaults`: stored Workspace setting. Returned only to users allowed to manage or read Workspace settings according to existing Workspace permissions.
- `effective_assistant_default`: permission-filtered runtime view for the current user and surface. It can include the resolved assistant id, display label if visible, memory mode, source `"workspace"`, and degraded status/reason. It must not expose hidden Persona details when the current user lacks access.

Example effective response:

```json
{
  "effective_assistant_default": {
    "status": "available",
    "source": "workspace",
    "assistant_kind": "persona",
    "assistant_id": "persona-id",
    "label": "Literature Review Assistant",
    "persona_memory_mode": "read_only",
    "degraded_reason": null
  }
}
```

Unavailable defaults should keep a stable shape:

```json
{
  "effective_assistant_default": {
    "status": "unavailable",
    "source": "workspace",
    "assistant_kind": null,
    "assistant_id": null,
    "label": null,
    "persona_memory_mode": null,
    "degraded_reason": "permission_denied"
  }
}
```

## Precedence Rules

Workspace-bound surfaces should use deterministic precedence:

1. Explicit per-session, per-run, or per-artifact assistant choice.
2. Workspace assistant default.
3. User/global assistant default.
4. Server fallback.

Chat Workspace V1 applies this as:

1. Existing chat session metadata.
2. Explicit assistant selected before first send.
3. Workspace `effective_assistant_default`.
4. System fallback assistant.

Other surfaces must not adopt Workspace defaults until they can persist the resolved assistant metadata on their own concrete record. That means:

- Prompt Studio runs record the resolved assistant metadata on the run.
- Writing workflows record the resolved assistant metadata on the draft/session.
- Research Workspace chat/RAG records the resolved assistant metadata on the chat or run.
- Audio overview and agent/tool workflows wait until voice/tool policy contracts are stable enough to persist meaningful resolved metadata.

Workspace defaults apply only when the active surface has Workspace scope. They must not silently affect ordinary global `/chat` unless that chat is explicitly started from or bound to a Workspace.

## UX Requirements

- Workspace settings should show the default assistant as a reference to an existing Persona profile, not as copied Persona content.
- Chat Workspace should surface whether a Persona is inherited from Workspace defaults or explicitly selected for the chat.
- Users must be able to clear a Workspace Persona default.
- If the referenced Persona is deleted, inaccessible, or unavailable, the Workspace should show a degraded state and continue to open.
- Applying a Workspace default should not overwrite an active chat session's existing Persona metadata.
- Any `read_write` memory mode default must be explicit, visible before use, and confirmed when saved as a Workspace default.
- Shared Workspace collaborators who cannot access the referenced Persona should see a redacted unavailable state and should not auto-apply the default.
- Users can always choose an explicit assistant for their own session/run where the surface supports explicit selection.

## Surface Behavior

### Chat Workspace: V1 Implementation Target

Chat Workspace is the first applying surface. When a user starts a new Workspace-scoped chat and no assistant is explicitly selected, Chat Workspace resolves `effective_assistant_default` and writes concrete session metadata:

- `assistant_kind: "persona"`
- `assistant_id`
- `persona_memory_mode`
- Workspace scope metadata

After the chat is created, the conversation is independent of later Workspace default edits. Changing the Workspace default does not mutate existing conversations, open conversations with persisted metadata, or historical chat records.

### Later Workspace Surfaces

The PRD defines a shared contract for later adoption, but these are not V1 blockers:

- Research Workspace chat/RAG may inherit the default only for new Workspace-scoped assistant interactions, never for global search or non-Workspace chat.
- Prompt Studio may preselect the default only for Workspace-bound prompt tests/runs and must record the resolved assistant metadata on the run.
- Writing workflows may treat the default as a suggested assistant/style for new Workspace-bound drafts, not as a hidden rewrite of existing drafts.
- Audio overview or media generation may use voice defaults only after this PRD or a follow-up explicitly distinguishes assistant speaking voice from existing Workspace audio overview settings.
- Agent/tool workflows may reference tool-policy defaults only after Persona Tool Administration defines policy profile lifecycle and permission layering.

## Backend Requirements

- Workspace read/update APIs should expose an optional assistant defaults contract.
- The persisted contract should avoid snapshots of Persona name, prompt, avatar, policy, or tool permissions.
- Workspace update should validate references where practical:
  - `assistant_kind` is `"persona"` in V1.
  - Persona exists and is usable by the current owner/editor where practical.
  - memory mode is one of `read_only` or `read_write`.
  - `read_write` saves require explicit confirmation or an equivalent review step.
  - voice/style/tool fields are absent or `null` in V1 unless their contracts have landed.
- Chat session creation should continue to store concrete assistant metadata on the conversation; Workspace defaults are startup hints, not hidden mutable session state.
- If a Workspace default changes, existing conversations, runs, drafts, and artifacts should not silently change assistant identity.
- Runtime clients should prefer `effective_assistant_default` over raw stored defaults.

## Data Model Direction

Preferred storage is an explicit JSON/object field or companion table on Workspace records rather than overloading existing banner/audio fields.

Required V1 fields:

- `assistant_kind: "persona" | null`
- `assistant_id: string | null`
- `persona_memory_mode: "read_only" | "read_write" | null`

Deferred fields:

- voice defaults until assistant voice and Workspace audio-overview semantics are reconciled.
- style preset references until a stable style preset catalog exists.
- tool policy profile references until Persona Tool Administration defines install/config lifecycle.
- personalization memory tuning until the Personalization Memory Layer PRD.

Suggested degraded reason codes:

- `persona_deleted`
- `persona_unavailable`
- `persona_feature_disabled`
- `permission_denied`
- `invalid_default`
- `unsupported_assistant_kind`

## V1 Acceptance Boundary

V1 is complete when:

- Workspace schema/API can store and return `assistant_defaults` with Persona-only validation.
- Workspace API exposes permission-filtered `effective_assistant_default`.
- Workspace settings can select, confirm `read_write`, clear, and show degraded Persona defaults.
- Chat Workspace applies the default only to new Workspace-scoped chats with no explicit assistant.
- Existing sessions remain unchanged after Workspace default edits.

Research Workspace, Prompt Studio, writing, audio overview, and agent/tool adoption are contract-defined but not V1 blockers.

## Staged Delivery

### Stage 1: Contract Audit And Schema/API Design

Goal: define the shared Workspace assistant defaults contract without writing runtime behavior first.

Deliverables:

- Backend schema proposal for Workspace `assistant_defaults`.
- DB migration design for storing Persona references and memory mode.
- Stored-versus-effective API response examples.
- Validation rules for missing/deleted Persona references.
- Degraded reason-code contract.

### Stage 2: Workspace Settings And Read Path

Goal: make defaults visible and editable without changing chat behavior yet.

Deliverables:

- Workspace settings UI for selecting/clearing a default Persona.
- `read_write` confirmation/review step.
- Read-only degraded state for deleted/unavailable/inaccessible Persona.
- Tests for optimistic locking and reference validation.

### Stage 3: Chat Workspace Startup Application

Goal: apply Workspace defaults only when starting a new Workspace-scoped chat without explicit assistant metadata.

Deliverables:

- Workspace chat startup reads Workspace default Persona.
- Explicit chat/session selection overrides Workspace default.
- Existing conversations are unaffected by later Workspace default edits.
- Inspector labels distinguish inherited vs explicit Persona where feasible.

### Stage 4: Surface Adoption Gates

Goal: allow other Workspace surfaces to adopt the shared contract only after they can persist resolved assistant metadata.

Deliverables:

- Research Workspace adoption plan and tests.
- Prompt Studio adoption plan and tests.
- Writing workflow adoption plan and tests.
- Explicit non-adoption behavior for surfaces that cannot persist resolved metadata yet.

### Stage 5: Voice, Style, And Tool Defaults

Goal: integrate adjacent defaults only after Persona identity behavior and the relevant contracts are stable.

Deliverables:

- Reconcile existing Workspace audio fields with assistant speaking voice defaults.
- Add style references only after style presets are stable.
- Add tool policy reference display only after Persona Tool Administration defines policy lifecycle.
- Keep broad tool installation/admin flows out of this PRD.

## Risks

- Workspace defaults could silently override explicit user choices if precedence is not enforced.
- Storing Persona snapshots would drift from Persona Garden profile updates and create privacy risk.
- Reusing existing Workspace audio fields without a structured contract may confuse voice defaults with audio overview generation settings.
- `read_write` memory mode could create unexpected durable memory if it is inherited invisibly.
- Workspace sharing may expose default Persona references in ways that require permission-aware redaction.
- Later surfaces could accidentally treat Workspace defaults as live pointers unless they persist resolved assistant metadata on their own records.

## Open Questions For Implementation Planning

- Which exact Workspace permission level may save or clear `assistant_defaults`?
- Should owners/admins see raw inaccessible Persona references for repair, or should the API return only redacted unavailable state to all clients?
- Should `read_write` confirmation be a frontend-only review step, a backend-required acknowledgement field, or both?
- Should later surfaces share one `effective_assistant_default` response, or request surface-specific resolution with explicit capability filters?

## Acceptance Criteria

- Workspace assistant defaults are documented as Workspace-scoped startup hints, not hidden global assistant state.
- V1 accepts only Persona-backed `assistant_defaults` while preserving the broader field name for future assistant kinds.
- Explicit per-session, per-run, and per-artifact assistant choices override Workspace defaults.
- Existing conversations, runs, drafts, and artifacts do not silently change when Workspace defaults change.
- Persona references remain reference-backed with no Persona content snapshots.
- Missing, deleted, inaccessible, disabled, or invalid Persona references degrade visibly and do not block opening the Workspace.
- Stored defaults and effective runtime defaults are separate API concepts.
- `read_write` memory mode is never inherited invisibly and requires explicit save confirmation.
- Later Workspace surfaces do not adopt defaults until they can persist resolved assistant metadata.
- Old `project_id` terminology is replaced by current Workspace terminology in this feature's contract.

## Verification Plan

- Schema tests for valid, empty, invalid, unsupported assistant kind, and deleted Persona default references.
- Migration tests for adding Workspace Persona defaults without altering existing Workspace records.
- API tests for Workspace read/update optimistic locking, reference validation, stored/effective response shape, and permission redaction.
- Component tests for selecting, confirming `read_write`, clearing, and degraded display of Workspace Persona defaults.
- Integration tests proving Workspace chat startup applies defaults only for new Workspace-scoped chats with no explicit assistant.
- Regression tests proving global chat, existing conversations, and unsupported surfaces are unaffected.
- Later surface adoption tests proving resolved assistant metadata is persisted on runs/drafts/artifacts before defaults are applied.

## References

- `Docs/Product/Persona_Agent_Design.md`
- `Docs/Product/Persona_Backed_Chat_Startup_PRD.md`
- `Docs/superpowers/specs/2026-05-21-persona-prd-reconciliation-design.md`
- `apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceChatPanel.tsx`
- `apps/packages/ui/src/components/Option/ChatWorkspace/ChatWorkspacePage.tsx`
- `apps/packages/ui/src/components/Option/ChatWorkspace/InspectorRail.tsx`
- `apps/packages/ui/src/types/workspace.ts`
- `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
- `tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py`
- `tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py`
