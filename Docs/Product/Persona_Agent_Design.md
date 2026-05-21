# Persona Interaction / Persona Garden PRD

Status: Active - current completion scope reconciled

Owner: Core (Persona, LLM, Audio, MCP, AuthNZ, WebUI)

Current scope: Persona Garden and live Persona sessions

Future scope tracker: https://github.com/rmusser01/tldw_server/issues/1902

Last reconciled: 2026-05-21

## Summary

Persona is the advanced assistant profile system for tldw_server. A Persona is derived from a Character, keeps provenance, and then evolves independently with its own live sessions, state, memory, scopes, policies, tools, voice settings, and static visual/media context.

The current Persona module completion target is the Persona Garden and live-session foundation. It is not the whole assistant platform. Ordinary chat startup, Workspace-level assistant defaults, scheduled autonomous work, expressive avatars, global personalization, full tool administration, and multi-agent collaboration are valid future directions, but they are not blockers for declaring the current Persona module complete.

## Current Status

The original scaffold-era status from 2026-02-09 is stale. Current code evidence shows the module has moved beyond the initial catalog/session/websocket scaffold.

Shipped or substantially implemented:

- Authenticated Persona endpoints and websocket interactions.
- Persona Garden route framing, shared WebUI/extension route behavior, and tabs for `Live Session`, `Profiles`, `State Docs`, `Scopes`, and `Policies`.
- `My Chat Identity`, `Characters`, and `Persona Garden` separation in the chat/persona UI.
- Character-to-Persona creation with provenance fields and independence after source Character deletion.
- Live websocket text flow with `user_message`, `tool_plan`, `confirm_plan`, `tool_call`, `tool_result`, notices, and session turn persistence.
- Live voice protocol paths including `audio_chunk`, `partial_transcript`, `voice_commit`, wake activation, and `tts_audio` events.
- Persona session preference persistence, memory retrieval toggles, memory top-k handling, and Persona memory storage.
- Persona scope and policy tables, API endpoints, runtime policy loading, and policy-denial behavior for Persona tool execution.
- Static/state Persona visual-pack support as a Persona-owned media context.

Current completion gaps:

- Minimal live-session transcript export for the selected Persona Garden session.
- Non-placeholder Scopes and Policies editing surfaces in Persona Garden.
- Minimal Persona-local MCP/tool discovery and default selection from already-authorized tools.
- Persona memory visibility controls only where backend support already exists.
- Updated acceptance tests and docs that capture the reconciled current boundary.

Not current completion blockers:

- Ordinary `/chat` startup with Persona profiles.
- Workspace-scoped Persona defaults. The original `project_id` language maps to the old name for Workspaces and is moved to a future PRD.
- Persona scheduled work and daily briefs.
- Rich avatar animation, visemes, lip-sync, 3D rendering, and high-frequency visual runtimes.
- Cross-app personalization and automatic semantic memory curation.
- Marketplace-style MCP/tool installation, configuration, administration, and global permission lifecycle.
- Multi-agent or multi-Persona collaboration.

## Design References

- Reconciliation spec: `Docs/superpowers/specs/2026-05-21-persona-prd-reconciliation-design.md`
- Persona Garden design: `Docs/Plans/2026-03-08-persona-garden-design.md`
- Future PRD tracker: https://github.com/rmusser01/tldw_server/issues/1902

## Product Model

### My Chat Identity

Represents the user in standard chat.

Owns:

- user display name
- user avatar/image
- user-side prompt templates

Does not own:

- Character definitions
- Persona profiles
- Persona memory, state, scope, policy, or tool settings

### Character

Represents the base assistant definition used in standard Character chat.

Owns:

- reusable assistant identity
- greeting, personality, scenario, and system prompt
- base images and related Character metadata

### Persona

Represents an advanced assistant artifact derived from a Character snapshot and then evolved independently.

Owns:

- Persona profile fields
- Character provenance metadata
- Persona-specific system and state overlays
- Persona state docs
- Persona policy rules
- Persona scope rules
- Persona-owned long-term and session memory
- Persona-specific voice settings
- Persona-specific static visual/media context
- Persona live sessions

### Persona Lifecycle

1. User creates or selects a Character.
2. User creates a Persona from that Character.
3. Source Character fields are copied or snapshotted into the Persona seed.
4. Persona Garden opens the newly created Persona.
5. The Persona evolves independently afterward.

Implications:

- Editing the source Character does not automatically mutate existing Personas.
- Editing a Persona does not mutate the source Character.
- Persona memory belongs to the Persona.
- If a source Character is deleted, the derived Persona remains valid.
- Any future `Refresh from source Character` behavior must be explicit, manual, diff-based, and confirmed.

## Goals

- Provide a persistent Persona profile with configurable behavior, voice defaults, memory preferences, scopes, policies, tools, and static visual/media context.
- Preserve Persona Garden as the advanced workspace for Persona setup, configuration, and live testing.
- Support live text and voice Persona sessions using the existing websocket, STT, and TTS stack.
- Enable tool use via MCP with a visible plan, confirmation, execution, and result-review loop.
- Make session history, transcript events, and tool outcomes auditable.
- Keep Persona memory opt-in and bounded by visible controls.
- Ensure RBAC, ownership, policy checks, rate limits, and explicit confirmations for write/delete/export actions.

## Non-Goals For Current Completion

- Ordinary `/chat` Persona startup.
- Workspace-scoped Persona defaults.
- Autonomous scheduled Persona jobs.
- Desktop automation outside browser/server tools.
- Multi-agent collaboration.
- Rich 3D avatars, visemes, lip-sync, or high-frequency animation.
- Marketplace-style MCP/tool administration.
- Cross-app automatic memory curation.

Each of these is tracked as a future PRD bucket in #1902 and is not a current completion blocker.

## Current Completion Scope

### Persona Garden IA And Lifecycle

The current module must clearly distinguish `My Chat Identity`, `Characters`, and `Persona Garden`.

Persona Garden remains the advanced workspace for Persona profiles and live Persona sessions. It should preserve:

- Persona selection
- profile creation and editing
- character-derived provenance display
- live session connect and disconnect
- session resume
- state-doc editing and history restore
- memory and state-context controls
- tool-plan review and confirmation

### Live Persona Session Runtime

The current module includes live text and voice sessions.

The live session runtime should support:

- websocket connection lifecycle
- `user_message`
- `audio_chunk`
- `voice_commit`
- `wake_activation`
- `partial_transcript`
- `tool_plan`
- `confirm_plan`
- `tool_call`
- `tool_result`
- `tts_audio`
- `notice`
- safe text-only degradation when voice/TTS paths fail
- persisted session turns and tool outcomes

### Transcript And Audit Timeline

Persona Garden should expose an auditable live-session timeline containing:

- user messages
- assistant messages
- voice transcript events where relevant
- tool plans
- tool calls
- tool results
- recovery notices and degraded-mode notices

Current completion requires minimal export for the selected live Persona session in deterministic JSON, readable Markdown, or both.

Export hardening requirements:

- Export only sessions owned by the authenticated user.
- Export only the selected session, not all Persona history.
- Omit or redact secrets, auth material, raw binary audio, and large tool payloads.
- Omit or redact hidden system/developer prompts, hidden policy metadata, hidden tool configuration, and tool metadata that was not visible in the live session UI.
- Omit non-selected-session memory records and retrieved source payloads that were not shown to the user during the session.
- Include enough audit metadata to understand the export, such as Persona id, session id, timestamps, event types, and redaction markers.

This export is not a scheduled report system and does not imply daily brief or workflow delivery support.

### Persona Memory Controls

Current completion keeps memory controls bounded to Persona-owned and session-owned behavior.

The module should expose:

- active memory mode
- retrieval enabled/disabled state
- memory top-k
- whether memory was requested and applied for a live turn
- session/persona memory visibility where backend support already exists
- archive or clear controls only where Persona backend support already exists

Missing archive or clear controls are non-blocking follow-up unless they are already supported by the Persona backend.

Move cross-app personalization, semantic memory tuning, automatic memory merge/prune, and broad long-term curation to the future Personalization Memory Layer PRD.

### Scopes And Policies Editing

Current completion keeps Scopes and Policies editing in scope because Persona Garden already exposes these tabs and the backend already has Persona scope/policy storage and runtime enforcement.

Persona Garden should allow users to:

- view scope rules
- edit and save scope rules
- recover from scope validation errors
- view policy rules
- edit and save policy rules
- recover from policy validation errors
- understand when a live tool plan is blocked by policy

Hardening requirements:

- Editing Persona rules must not grant capabilities the authenticated user, server config, or deployment policy does not already allow.
- Destructive changes should use confirmation or a clear review step.
- Validation errors should identify the invalid rule, field, and reason without leaking hidden tool details.
- Saves should preserve rule ownership and avoid cross-Persona writes.
- Concurrent edits should fail predictably or use the existing versioning pattern where available.

### Minimal MCP And Tool Capability Discovery

Current completion includes minimal Persona-local tool discovery and default selection, not full tool administration.

Persona Garden should show:

- available tools or tool categories visible to the authenticated user and deployment
- Persona-local default or allowed tools selected only from already-authorized tools
- blocked or unavailable reason text
- confirmation requirements for impactful tools

Hardening requirements:

- Show only tools visible to the authenticated user and current deployment.
- Avoid exposing hidden/admin-only tool names through blocked reason text.
- Distinguish `not installed`, `disabled by server`, `not allowed for this Persona`, and `requires confirmation` where the backend can truthfully report that distinction.
- Constrain any default-tool save path by Persona policy, server capability checks, and already-authorized user permissions.

Do not expand current completion into marketplace-style tool installation, global tool configuration, admin-level tool lifecycle management, or global permission changes.

### Static Visual And Persona Media Context

Current completion acknowledges already-integrated static/state visual pack support as Persona-owned media context.

The current module may document visual packs, state mappings, and static Persona visual feedback where they already exist. It creates no new animation, viseme, lip-sync, renderer, 3D, or high-frequency visual-runtime requirements.

Rich avatar work belongs to the future Persona Expressive Avatar Runtime PRD.

### Security And Reliability

Current completion requires:

- authenticated Persona HTTP endpoints
- authenticated websocket interaction
- no anonymous Persona sessions
- capability gating when Persona is disabled or unsupported
- session ownership checks
- rate limits and payload bounds
- confirmation for destructive, write, delete, and export actions
- policy enforcement for MCP/tool execution
- useful validation and recovery errors
- tests covering the current completion surface

## API Contract

Base path: `/api/v1/persona`

### Existing Core Endpoints

- `GET /catalog`
- `POST /session`
- `GET /sessions`
- `GET /sessions/{session_id}`
- `WS /stream`

### Session Creation

`POST /session` remains Persona/session scoped.

Any legacy `project_id` field should not be treated as current Workspace-level Persona defaults. Workspace-scoped Persona defaults are moved to a future PRD.

### Websocket Messages

Client to server:

- `user_message`: `{ session_id, text, use_memory_context?, memory_top_k? }`
- `audio_chunk`: `{ session_id, audio_format, bytes_base64 }`
- `voice_commit`: `{ session_id, transcript? }`
- `wake_activation`: `{ session_id, phrase?, source? }`
- `confirm_plan`: `{ session_id, plan_id, approved_steps: [idx...] }`
- `cancel`: `{ session_id, reason? }`

Server to client:

- `assistant_delta`: `{ session_id, text_delta }`
- `partial_transcript`: `{ session_id, text_delta }`
- `tool_plan`: `{ session_id, plan_id, steps: [...] }`
- `tool_call`: `{ session_id, step_idx, tool, args }`
- `tool_result`: `{ session_id, step_idx, ok, output, error? }`
- `tts_audio`: `{ session_id, audio_format, chunk_id }`
- `notice`: `{ session_id, level, message, reason_code? }`

### `tool_result.result` Deprecation Plan

Canonical contract: `tool_result.output`.

Compatibility window: through 2026-06-30, the server may continue emitting both `output` and legacy `result`.

Client behavior during the window:

- Read `output` first.
- Fall back to `result` only for older server compatibility.

Planned removal window: starting 2026-07-01, or the next compatible minor/major release after that date, stop emitting `result` once WebUI/extension compatibility checks are green for output-only payloads.

This is a dated maintenance item, not a current feature gap before the planned removal window.

## Future PRDs

The following product directions are moved out of current completion scope. Each item should get its own PRD before implementation. Each item is not a current completion blocker.

Tracker: https://github.com/rmusser01/tldw_server/issues/1902

### Persona-backed Chat Startup

Not a current completion blocker.

Future scope:

- Make ordinary `/chat` use Persona profiles as first-class assistants.
- Preserve current Character chat behavior.
- Resolve migration from deprecated `persona_id` alias semantics.
- Define chat startup state, persistence, and UI selection.

### Workspace Persona Defaults

Not a current completion blocker.

Future scope:

- Replace old `project_id` terminology with Workspace terminology.
- Define Workspace-level Persona defaults.
- Define Workspace-level style, voice, tool, and memory defaults.
- Handle conflicts between local session choice and Workspace default.
- Integrate with chat, Prompt Studio, writing, and workspace surfaces.

### Persona Scheduled Work

Not a current completion blocker.

Future scope:

- recurring Persona jobs
- daily briefs
- review and approval gates
- delivery channels
- Jobs versus Scheduler integration
- failure, retry, and notification behavior

### Persona Expressive Avatar Runtime

Not a current completion blocker.

Future scope:

- high-frequency animation
- visemes and lip-sync
- expressive 2D or 3D rendering
- advanced runtime synchronization with voice and text

### Personalization Memory Layer

Not a current completion blocker.

Future scope:

- cross-app semantic memory
- memory creation and curation
- merge and prune behavior
- personalization tuning
- user controls for global personalization

### Persona Tool Administration

Not a current completion blocker.

Future scope:

- marketplace-style tool install and configuration
- admin-level tool lifecycle
- broad permissions management
- integration-level setup and health checks

Current completion keeps only minimal discovery, Persona-local defaults from already-authorized tools, and blocked reason text.

### Persona Collaboration And Multi-agent Workflows

Not a current completion blocker.

Future scope:

- multiple Personas acting concurrently
- Persona-to-Persona coordination
- shared plans and arbitration
- concurrent tool use and conflict resolution

## Acceptance Criteria For Current Completion

- Persona Garden clearly separates `My Chat Identity`, `Characters`, and `Persona`.
- Personas can be created from Characters and remain independent afterward.
- Live text/voice sessions can connect, resume, degrade safely, and expose understandable recovery states.
- Transcript timeline includes user/assistant messages, voice transcript events where relevant, tool plans, tool calls/results, and notices.
- Selected-session transcript export is available and follows the privacy/redaction constraints in this PRD.
- Persona memory controls expose mode, retrieval toggle, top-k, and existing backend-supported visibility/archive/clear operations.
- Scopes and Policies tabs are not placeholders; users can view, edit, save, and recover from validation errors for Persona scope and policy rules.
- Scopes and Policies editing cannot grant capabilities the authenticated user/server/deployment does not already allow.
- Persona Garden shows available MCP/tool capabilities, Persona-local defaults from already-authorized tools, and blocked/unavailable reasons without leaking hidden/admin-only tools.
- Static/state visual pack support is documented as current context only, with expressive avatar runtime clearly future scoped.
- Auth, ownership, capability gating, rate limits, destructive-action confirmations, and useful error states are covered by tests.
- Future PRD buckets are linked to #1902 and explicitly labeled as not current completion blockers.

## Verification Plan

- Review this PRD against `Docs/superpowers/specs/2026-05-21-persona-prd-reconciliation-design.md`.
- Confirm each moved-out feature appears in #1902.
- If a moved-out feature is missing from #1902, update #1902 before marking this PRD complete.
- Confirm this PRD does not describe future PRDs as current completion blockers.
- Confirm no design-system backlog tasks or files are modified.

## Evidence Snapshot

Evidence used for this reconciliation included current code and tests for:

- Persona websocket voice and tool events in `tldw_Server_API/app/api/v1/endpoints/persona.py`.
- Persona websocket coverage in `tldw_Server_API/tests/Persona/test_persona_ws.py`.
- Persona session and memory persistence in `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` and `tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py`.
- Persona profile provenance tests in `tldw_Server_API/tests/Persona/test_persona_profiles_api.py` and `tldw_Server_API/tests/ChaChaNotesDB/test_persona_persistence_db.py`.
- Persona Garden route tabs and shared route tests in `apps/packages/ui/src/routes/sidepanel-persona.tsx` and `apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx`.
- Character-to-Persona actions in `apps/packages/ui/src/components/Option/Characters/`.
