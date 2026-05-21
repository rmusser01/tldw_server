# Persona PRD Reconciliation Design

Date: 2026-05-21

Status: Draft

Tracker: https://github.com/rmusser01/tldw_server/issues/1902

Backlog: TASK-442

## Summary

Reconcile the original Persona module PRD into a current completion contract for Persona Garden and live Persona sessions, while preserving the broader assistant vision as separate future PRDs.

The current Persona PRD should stop acting as a catch-all roadmap. It should define what must be true before the current Persona module is considered coherent and complete. Broader platform work remains valid, but should move into explicit follow-up PRDs tracked by issue #1902.

## Source Documents

- `Docs/Product/Persona_Agent_Design.md`
- `Docs/Plans/2026-03-08-persona-garden-design.md`
- `Docs/Plans/2026-03-08-persona-garden-implementation-plan.md`

## Product Boundary

The current Persona module is the Persona Garden and live Persona session foundation.

It owns:

- Persona profiles derived from Characters and independent afterward.
- Persona Garden as the advanced persona workspace.
- Live text and voice sessions.
- Persona-owned state, memory, scope, policy, tool, and static visual/media context.
- Auditable transcript and tool timelines.

It does not own ordinary chat startup, Workspace-level assistant defaults, scheduled autonomous work, rich avatar rendering, global personalization, full tool marketplace administration, or multi-agent collaboration.

## Keep In Current Persona Completion Scope

### Persona Garden IA And Lifecycle

Keep the separation between `My Chat Identity`, `Characters`, and `Persona Garden`.

Personas remain advanced assistant artifacts created from Characters, with provenance retained for audit and display. After creation, personas evolve independently. Editing the source Character must not silently mutate an existing Persona, and editing a Persona must not mutate the source Character.

### Live Persona Session Runtime

Keep live text and voice sessions as part of current completion. This includes:

- connect and disconnect
- session resume
- websocket user messages
- audio chunks
- partial transcripts
- TTS audio events
- tool plans
- plan confirmation and cancellation
- notices and safe degradation states

The current PRD should mark the stale "full voice protocol not implemented" status as obsolete when reconciling against current code.

### Transcript And Audit Timeline

Keep transcript auditability in current scope.

The Persona Garden live session should show user and assistant turns, voice transcript events where relevant, tool plans, tool calls, tool results, and recovery notices. It should support a minimal downloadable export for a live Persona session in JSON or Markdown.

This export is not a scheduled report system and should not imply daily brief or workflow delivery support.

### Persona Memory Controls

Keep persona-owned memory controls in current scope, but bound them narrowly.

The current module should expose:

- active memory mode
- retrieval toggle
- top-k retrieval setting
- session/persona memory visibility where supported
- archive or clear controls where backend support already exists or is directly adjacent

Move cross-app personalization, semantic memory tuning, automatic memory merge/prune, and broad long-term curation into a separate future PRD.

### Scopes And Policies Editing

Keep Scopes and Policies editing in current scope.

The backend already has persona scope and policy rule storage plus runtime enforcement paths, and Persona Garden already exposes `Scopes` and `Policies` tabs. Those tabs should not remain placeholders in the completion contract.

Current completion should require users to:

- view scope rules
- edit and save scope rules
- recover from scope validation errors
- view policy rules
- edit and save policy rules
- recover from policy validation errors
- understand when a live tool plan is blocked by policy

### Minimal MCP And Tool Capability Discovery

Keep minimal tool discovery and default toolset management in current scope.

Persona Garden should show enough tool information for a user to understand what a persona can use and why a tool may be unavailable. This includes:

- visible available tools or tool categories
- persona default or allowed tools
- blocked or unavailable reason text
- confirmation requirements for impactful tools

Do not expand this into marketplace-style tool installation or admin-level tool lifecycle management in the current PRD.

### Static Visual And Persona Media Context

Keep existing static/state visual pack support as persona-owned media context.

The current PRD should acknowledge visual packs, state mappings, and static persona visual feedback where already integrated. It should explicitly avoid making rich avatar animation, visemes, lip-sync, or 3D rendering blockers for current completion.

### Security And Reliability Completion

Keep security and reliability as acceptance criteria:

- authenticated Persona endpoints and websocket interactions
- capability gating
- session ownership checks
- rate limits and payload bounds
- confirmation for destructive, write, delete, and export actions
- useful validation and recovery errors
- no anonymous Persona interaction
- tests covering the current completion surface

## Move Out To Future PRDs

Each item below should become its own PRD and should not block current Persona module completion. The tracking issue is #1902.

### Persona-backed Chat Startup

Move ordinary `/chat` integration into a future PRD.

Scope:

- selecting Persona profiles as first-class chat assistants
- preserving current Character chat behavior
- migration from deprecated `persona_id` alias semantics
- chat startup state, persistence, and UI selection

### Workspace Persona Defaults

Move workspace-scoped persona selection into a future PRD.

The original PRD's `project_id` language should be updated to current Workspace terminology.

Scope:

- Workspace-level persona defaults
- Workspace-level style, voice, tool, and memory defaults
- interaction with chat, Prompt Studio, writing, and workspace surfaces
- conflict handling between local session choice and Workspace default

### Persona Scheduled Work

Move scheduled jobs and daily briefs into a future PRD.

Scope:

- recurring persona jobs
- daily briefs
- review and approval gates
- delivery channels
- Jobs versus Scheduler integration
- failure/retry/notification behavior

### Persona Expressive Avatar Runtime

Move rich avatar work into a future PRD.

Scope:

- high-frequency animation
- visemes and lip-sync
- expressive 2D or 3D rendering
- advanced runtime synchronization with voice and text

### Personalization Memory Layer

Move broader memory and personalization into a future PRD.

Scope:

- cross-app semantic memory
- memory creation and curation
- merge and prune behavior
- personalization tuning
- user controls for global personalization

### Persona Tool Administration

Move full tool administration into a future PRD.

Scope:

- marketplace-style tool install and configuration
- admin-level tool lifecycle
- broad permissions management
- integration-level setup and health checks

The current PRD keeps only minimal discovery, allowed/default tool visibility, and blocked reason text.

### Persona Collaboration And Multi-agent Workflows

Move multi-agent and multi-persona collaboration into a future PRD.

Scope:

- multiple personas acting concurrently
- persona-to-persona coordination
- shared plans and arbitration
- concurrent tool use and conflict resolution

## Reconciliation Updates For The Current PRD

When updating `Docs/Product/Persona_Agent_Design.md`, apply these changes:

1. Replace stale implementation status with a current status section.
2. Mark voice protocol, persona/session memory, and policy object support as shipped or partially shipped according to current code evidence.
3. Clarify that Persona Garden/live Persona sessions are the current module boundary.
4. Add the current completion scope from this design.
5. Add a future PRD section linking #1902.
6. Replace old `project_id` phrasing with a note that Workspace-scoped persona defaults moved to a future PRD.
7. Keep the original long-term vision visible, but do not let future tracks block current completion.

## Acceptance Criteria

- Current Persona PRD distinguishes current completion scope from future platform scope.
- Current Persona PRD keeps Persona Garden and live Persona sessions as the module boundary.
- Current Persona PRD keeps Scopes and Policies editing, transcript export, minimal tool discovery, memory controls, and reliability/security closeout in scope.
- Current Persona PRD moves ordinary chat integration, Workspace persona defaults, scheduled work, expressive avatars, broad personalization, tool administration, and multi-agent collaboration out to future PRDs.
- Future scope links to GitHub issue #1902.
- No design-system backlog tasks are touched.

## Verification Plan

- Review the updated PRD against this spec.
- Confirm every moved-out feature appears in #1902.
- Confirm the PRD does not describe future PRDs as current completion blockers.
- Confirm no design-system backlog tasks or files are modified.
