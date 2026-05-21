# Workspace Persona Defaults PRD

Status: Draft

Owner: Persona module / Workspaces integration

Tracking: #1911, split from #1902

Backlog: TASK-468

## Summary

Define Workspace-scoped Persona defaults as a thin composition layer over existing Workspaces, Persona profiles, and Workspace-scoped chat sessions. This PRD replaces the original Persona PRD's old `project_id` language with current Workspace terminology and keeps the work separate from Persona-backed ordinary chat startup, scheduled work, Buddy animation, and broad personalization memory.

The goal is not to make Workspaces own Persona profiles. The goal is to let a Workspace recommend a default assistant identity and related defaults for Workspace-bound surfaces, while preserving explicit user choices and session-level metadata.

## Problem

Persona Garden now owns Persona profile creation, configuration, and live-session testing. Persona-backed Chat Startup now owns ordinary chat startup as a Persona. Workspaces still need a separate contract for cases where a research workspace should carry a preferred assistant shape: for example a "literature review" Workspace might default to a careful research Persona, a specific voice for audio overview workflows, and a limited tool policy appropriate for that workspace.

The current code has the pieces but not the product contract:

- Workspace API schemas persist workspace metadata, study-materials policy, banner fields, and audio defaults.
- Workspace chat uses `scope: { type: "workspace", workspaceId }`.
- Chat session schemas already support `scope_type: "workspace"`, `workspace_id`, `assistant_kind`, `assistant_id`, and `persona_memory_mode`.
- Chat Workspace UI already reports a runtime `selectedPersonaLabel`, but there is no documented Workspace default source of truth or precedence model.

Without a PRD, Workspace defaults risk becoming implicit global state that silently overrides chat choices or Persona profile settings.

## Goals

- Define a Workspace-level defaults object for Persona-related behavior.
- Preserve explicit chat/session Persona selection over Workspace defaults.
- Keep Persona profiles user-owned and configured in Persona Garden.
- Let Workspace-bound chat and later Workspace surfaces read a Workspace default without duplicating Persona profile data.
- Document precedence between global user settings, Workspace defaults, Persona profile defaults, and per-session choices.
- Replace `project_id` phrasing with `workspace_id` and current Workspace terminology.

## Non-goals

- No implementation in this PRD slice.
- No Buddy animation or expressive avatar runtime.
- No design-system backlog work.
- No scheduled Persona work or recurring jobs.
- No ordinary `/chat` Persona startup changes; that is covered by #1908.
- No cross-app semantic personalization memory layer.
- No marketplace-style tool administration.
- No multi-agent or multi-Persona collaboration.

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

Workspace Persona Defaults V1 should add an explicit optional defaults object to a Workspace:

```json
{
  "persona_defaults": {
    "default_persona_id": "persona-id",
    "persona_memory_mode": "read_only",
    "style_preset_id": null,
    "voice": {
      "provider": "kokoro",
      "model": "default",
      "voice": "af_heart",
      "speed": 1.0
    },
    "tool_policy_profile_id": null
  }
}
```

The object should be reference-backed where possible:

- `default_persona_id` references a Persona profile owned by the user.
- `persona_memory_mode` defaults to `read_only`.
- voice defaults can reuse existing Workspace audio fields in V1 or be mirrored into a structured response shape if the existing fields remain the storage source.
- tool policy defaults should be a future reference to Persona policy/scopes, not an embedded copy of tool permissions.

## Precedence Rules

V1 should use deterministic precedence:

1. Per-chat/session explicit assistant selection.
2. Workspace Persona default.
3. User/global chat default.
4. System fallback assistant.

For other Persona-adjacent defaults:

1. Per-session explicit value.
2. Workspace default.
3. Persona profile default.
4. Global user setting.
5. Server fallback.

Workspace defaults should apply only when the active surface has Workspace scope. They must not silently affect ordinary global `/chat` unless that chat is explicitly started from or bound to a Workspace.

## UX Requirements

- Workspace settings should show the default Persona as a reference to an existing Persona profile, not as copied Persona content.
- Chat Workspace should surface whether a Persona is inherited from Workspace defaults or explicitly selected for the chat.
- Users must be able to clear a Workspace Persona default.
- If the referenced Persona is deleted or unavailable, the Workspace should show a degraded state and continue to open.
- Applying a Workspace default should not overwrite an active chat session's existing Persona metadata.
- Any `read_write` memory mode default must be explicit and visible before use.

## Backend Requirements

- Workspace read/update APIs should expose an optional Persona defaults contract.
- The persisted contract should avoid snapshots of Persona name, prompt, avatar, policy, or tool permissions.
- Workspace update should validate references where practical:
  - Persona exists and belongs to the current user.
  - memory mode is one of `read_only` or `read_write`.
  - voice fields follow existing Workspace audio validation once that validation exists.
- Chat session creation should continue to store concrete assistant metadata on the conversation; Workspace defaults are startup hints, not hidden mutable session state.
- If a Workspace default changes, existing conversations should not silently change assistant identity.

## Data Model Direction

Preferred V1 storage is an explicit JSON/object field or companion table on Workspace records rather than overloading existing banner/audio fields.

Required fields:

- `default_persona_id: string | null`
- `persona_memory_mode: "read_only" | "read_write" | null`

Allowed V1-adjacent fields:

- `voice_provider`
- `voice_model`
- `voice_id`
- `voice_speed`

Deferred fields:

- style preset references until a stable style preset catalog exists.
- tool policy profile references until Persona Tool Administration PRD defines install/config lifecycle.
- personalization memory tuning until the Personalization Memory Layer PRD.

## Staged Delivery

### Stage 1: Contract Audit And Schema Design

Goal: define the smallest Workspace defaults contract without writing runtime behavior first.

Deliverables:

- Backend schema proposal for Workspace Persona defaults.
- DB migration design for storing references and memory mode.
- Validation rules for missing/deleted Persona references.
- API response examples for no default, valid default, and degraded default.

### Stage 2: Workspace Settings And Read Path

Goal: make defaults visible and editable without changing chat behavior yet.

Deliverables:

- Workspace settings UI for selecting/clearing a default Persona.
- Read-only degraded state for deleted/unavailable Persona.
- Tests for optimistic locking and reference validation.

### Stage 3: Chat Workspace Startup Application

Goal: apply Workspace defaults only when starting a new Workspace-scoped chat without explicit assistant metadata.

Deliverables:

- Workspace chat startup reads Workspace default Persona.
- Explicit chat/session selection overrides Workspace default.
- Existing conversations are unaffected by later Workspace default edits.
- Inspector labels distinguish inherited vs explicit Persona where feasible.

### Stage 4: Voice And Tool Defaults

Goal: integrate adjacent defaults only after Persona identity behavior is stable.

Deliverables:

- Reconcile existing Workspace audio fields with Persona voice defaults.
- Add tool policy reference display only if the referenced policy contract already exists.
- Keep broad tool installation/admin flows out of this PRD.

## Risks

- Workspace defaults could silently override explicit user choices if precedence is not enforced.
- Storing Persona snapshots would drift from Persona Garden profile updates and create privacy risk.
- Reusing existing Workspace audio fields without a structured contract may confuse voice defaults with audio overview generation settings.
- `read_write` memory mode could create unexpected durable memory if it is inherited invisibly.
- Workspace sharing may expose default Persona references in ways that require permission-aware redaction.

## Open Questions For Implementation Planning

- Should V1 store voice defaults in existing Workspace audio fields or introduce a nested Persona defaults object immediately?
- Should shared Workspace viewers see the owner's default Persona label, a redacted label, or only "Persona default unavailable"?
- Should Workspace defaults apply to Prompt Studio and writing surfaces in V1, or only Chat Workspace?
- Should changing the Workspace default offer to apply to new chats only, or also prompt the user to update open unsent chat state?

## Acceptance Criteria

- Workspace Persona defaults are documented as Workspace-scoped startup hints, not hidden global assistant state.
- Explicit per-session Persona choices override Workspace defaults.
- Existing conversations do not silently change when Workspace defaults change.
- Persona references remain reference-backed with no Persona content snapshots.
- Missing/deleted Persona references degrade visibly and do not block opening the Workspace.
- `read_write` memory mode is never inherited invisibly.
- Old `project_id` terminology is replaced by current Workspace terminology in this feature's contract.

## Verification Plan

- Schema tests for valid, empty, invalid, and deleted Persona default references.
- Migration tests for adding Workspace Persona defaults without altering existing Workspace records.
- API tests for Workspace read/update optimistic locking and reference validation.
- Component tests for selecting, clearing, and degraded display of Workspace Persona defaults.
- Integration tests proving Workspace chat startup applies defaults only for new Workspace-scoped chats with no explicit assistant.
- Regression tests proving global chat and existing conversations are unaffected.

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
