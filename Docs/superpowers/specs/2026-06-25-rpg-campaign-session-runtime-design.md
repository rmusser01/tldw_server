# RPG Campaign Session Runtime Design

Date: 2026-06-25
Status: Approved for implementation planning
Backlog task: TASK-12017

## Context

The goal is to add a generic RPG/TTRPG harness to `tldw_server` that can support tabletop roleplaying games as a backend reference, tool, orchestrator, and optional DM assistant. The module must not become a FoundryVTT-style virtual tabletop. It should manage campaign/session state, rules context, dice/check resolution, and future orchestration hooks while leaving canvas maps, token movement, and live tabletop rendering out of scope.

The user selected a full-orchestrator long-term direction, with the first implementation slice focused on the core campaign/session runtime. The design uses Marinara Engine as an external reference for typed game state, rules/check helpers, mode separation, prompt/context building, and agent-ready orchestration boundaries, but adapts those ideas to the existing tldw_server FastAPI, ChaChaNotes, RAG, and MCP architecture.

Relevant local precedents:

- `tldw_Server_API/app/core/VN_Play/` for session runtime, event/state handling, idempotency, and replay-sensitive APIs.
- `tldw_Server_API/app/core/Character_Chat/` for character/world-book/dictionary context and prompt assembly boundaries.
- `tldw_Server_API/app/core/Chat_Workflows/` for a dedicated interactive runtime that sits beside generic Workflows rather than forcing a user-facing flow into the automation engine.
- `tldw_Server_API/app/core/MCP_unified/` for tool registration, RBAC, idempotency, and read/write tool governance.
- `tldw_Server_API/app/api/v1/router_groups/content.py` for route-key gated feature registration.

External references reviewed:

- Marinara Engine: `https://github.com/Pasta-Devs/Marinara-Engine/tree/main`
- D&D SRD: `https://www.dndbeyond.com/srd`
- ORC License: `https://downloads.paizo.com/ORC_License_FINAL.pdf`
- Archives of Nethys license page: `https://2e.aonprd.com/Licenses.aspx`
- Fate SRD: `https://fate-srd.com/`

## Product Decisions

1. The long-term product target is a full RPG orchestrator, not a lightweight dice toolkit.
2. The first implementation slice is the core campaign/session runtime.
3. V1 supports named rules adapters for D&D 5e SRD, Pathfinder 2e, and Fate.
4. V1 may bundle minimal open-content snippets with attribution and must also support user-ingested rules packs through existing retrieval systems.
5. Persistence is hybrid: RPG campaign/session data lives in per-user ChaChaNotes-backed storage, while larger rules-pack content is referenced through existing ingestion/RAG identifiers.
6. RPG sessions are primary domain objects. They may link to chats, but chat transcripts are not the RPG source of truth.
7. V1 snapshots model a play-ready core: scene, party/actors, resources, clocks, notes, recap, quests, NPCs, inventory, locations, factions, linked rules references, and unresolved rulings.
8. V1 exposes both REST APIs and MCP tools.
9. Authority is mixed: deterministic/manual actions may commit directly, while model-derived state changes become proposals unless a session explicitly enables auto-commit.

## Goals

- Provide a replayable, auditable campaign/session runtime.
- Support generic RPG systems without hardcoding d20 assumptions into the core model.
- Ship enough adapter behavior for D&D 5e SRD, Pathfinder 2e, and Fate to prove the adapter boundary.
- Support dice/check resolution, event append, snapshot rebuild, rules lookup, and context assembly.
- Enable MCP agents to inspect and assist sessions without bypassing authority policy.
- Keep rules content legally conservative and source-attributed.
- Make future GM orchestration, NPC tracking, quest tracking, combat assistance, and summarization straightforward without implementing those full agents in the first slice.

## Non-Goals

- No canvas map, battlemap, token movement, lighting, walls, live positioning, or asset-heavy tabletop rendering in v1.
- No multiplayer real-time shared tabletop semantics in v1.
- No full GM loop, combat tracker, map engine, or autonomous quest/NPC agent suite in the first implementation slice.
- No bundled long-form copyrighted rules text.
- No dependence on external LLM calls for core state correctness.
- No rewrite of Character Chat, VN Play, RAG, or MCP Unified.

## Recommended Architecture

Introduce a dedicated backend module:

- Core package: `tldw_Server_API/app/core/RPG/`
- API endpoint: `tldw_Server_API/app/api/v1/endpoints/rpg.py`
- API schemas: `tldw_Server_API/app/api/v1/schemas/rpg_schemas.py`
- Persistence repository: a focused RPG repository integrated with per-user ChaChaNotes DB access.
- Router key: `rpg`, mounted under `/api/v1/rpg`.

The source of truth is an append-only session ledger. Cached snapshots are derived state. The service appends validated typed events transactionally and then runs a deterministic reducer to update the cached snapshot. If event append and snapshot update cannot both succeed, the transaction rolls back.

Primary service components:

- `RPGService`: endpoint-facing orchestration for campaigns, sessions, events, rolls, proposals, snapshots, rules lookup, and context building.
- `RPGRepository`: durable per-user storage for campaigns, sessions, events, snapshots, proposals, idempotency records, and rules-pack links.
- `RuleAdapter`: mechanics and resolver behavior for each system.
- `RuleAdapterRegistry`: discovers bundled adapters and validates adapter/version selection.
- `RuleContentPack`: small bundled snippets with source, license, citation, and attribution metadata.
- `EventReducer`: pure deterministic reducer from prior snapshot plus typed events to next snapshot.
- `AuthorityPolicy`: decides direct commit vs proposal for each source/action/session setting.
- `ProposalQueue`: stores model/client-derived typed event proposals before commit.
- `DiceRoller` / `CheckResolver`: deterministic, testable dice and check helpers with explicit provenance.
- `SessionContextBuilder`: bounded prompt-ready session summary with cited rules excerpts and inclusion/exclusion diagnostics.

The RPG module should reuse existing services rather than duplicating them:

- AuthNZ and API-key scope dependencies for user identity and REST permission checks.
- ChaChaNotes per-user database resolution for local user data.
- RAG/media identifiers for linked user-ingested rules packs.
- MCP Unified for tool discovery, RBAC, idempotency, and execution boundaries.
- Existing logging patterns through Loguru.

## Rules Adapter Contract

The common adapter contract must be broad enough for both d20 and non-d20 games. It must not assume ability scores, proficiency, armor class, DCs, or initiative as universal concepts.

Each `RuleAdapter` exposes:

- `adapter_key`, `adapter_version`, display name, status, and license summary.
- Mechanics schema for actors, resources, tags/aspects, conditions, clocks/tracks, checks, and roll expressions.
- Adapter-specific validation for character/actor state.
- Dice/check resolver functions.
- Supported event type extensions, if any.
- Minimal `RuleContentPack` references for bundled snippets.
- Citation metadata and attribution requirements.

V1 bundled adapters:

- `dnd5e_srd`: D&D 5e SRD-oriented mechanics and citations.
- `pf2e`: Pathfinder 2e-oriented mechanics with conservative content handling. Bundled snippets must be verified as license-compatible before inclusion; when uncertain, ship metadata and citation links rather than text.
- `fate`: Fate-oriented mechanics covering aspects/tags, stress, consequences, approaches/skills where configured, and Fate-style outcomes.

The Fate adapter is a contract test for non-d20 assumptions. Core code must support Fate-style aspects, stress/consequences, and ladder-like outcomes without forcing d20 fields.

## Rules Content And Licensing

Rules content is split into two layers:

1. Mechanics metadata and resolver behavior: structured implementation data needed to run checks and validate state.
2. Snippet packs and rules references: human-readable reference content used for lookup and context.

Bundled snippets must be short, source-attributed, and license-aware. Each snippet or citation object must include:

- `adapter_key`
- `source_title`
- `source_url`
- `license`
- `license_url`
- `attribution`
- `trust_level`
- `content_hash`
- `snippet_id`

User-ingested rules packs are not copied into RPG tables. RPG records store references to existing ingestion/RAG/media identifiers, selected trust levels, and campaign/session binding metadata. Rule lookup must treat retrieved rule text as quoted reference material, not instructions. Context builders must isolate rules excerpts from system/developer instructions and include citations in returned diagnostics.

## Data Model

Campaigns are top-level containers.

Suggested `rpg_campaigns` fields:

- `id`
- `owner_user_id`
- optional future `workspace_id` / `shared_scope`
- `title`
- `description`
- `default_adapter_key`
- `default_adapter_version`
- `settings_json`
- `linked_rules_pack_refs_json`
- `version`
- `status` (`active`, `archived`, `deleted`)
- `created_at`
- `updated_at`

Sessions belong to campaigns and are source-of-truth play records.

Suggested `rpg_sessions` fields:

- `id`
- `campaign_id`
- `owner_user_id`
- optional future `workspace_id` / `shared_scope`
- `title`
- `status` (`active`, `paused`, `completed`, `archived`, `deleted`)
- `adapter_key`
- `adapter_version`
- `authority_settings_json`
- `linked_chat_id`
- `active_rules_pack_refs_json`
- `current_snapshot_version`
- `last_event_sequence`
- `version`
- `created_at`
- `updated_at`

Events are append-only and typed.

Suggested `rpg_session_events` fields:

- `id`
- `session_id`
- `owner_user_id`
- `sequence_number`
- `event_type`
- `event_payload_json`
- `source_type` (`user`, `system`, `mcp`, `model`, `import`)
- `source_actor_id`
- `source_label`
- `idempotency_key`
- `request_payload_hash`
- `event_schema_version`
- `adapter_key`
- `adapter_version`
- optional `proposal_id`
- `created_at`

Uniqueness should include `(owner_user_id, session_id, source_type, idempotency_key)` for mutating operations that provide idempotency keys. Sequence numbers are unique per session.

Snapshots are cached derived state.

Suggested `rpg_session_snapshots` fields:

- `id`
- `session_id`
- `owner_user_id`
- `snapshot_version`
- `last_event_sequence`
- `reducer_version`
- `snapshot_schema_version`
- `snapshot_json`
- `diagnostics_json`
- `created_at`

Snapshot JSON contains the play-ready core:

- scene
- party / actors
- resources
- clocks / tracks
- notes
- recap
- quests
- NPCs
- inventory
- locations
- factions
- rules references
- unresolved rulings

Every mutable entity inside snapshot JSON must have a stable ID. NPCs, quests, actors, locations, factions, inventory items, clocks, and rulings must not be updated by name matching alone.

Proposals hold model/client-derived typed event proposals before commit.

Suggested `rpg_session_proposals` fields:

- `id`
- `session_id`
- `owner_user_id`
- `base_event_sequence`
- `base_snapshot_version`
- `proposed_events_json`
- optional preview `patch_json`
- `rationale`
- `confidence`
- `source_type`
- `source_actor_id`
- `model_metadata_json`
- `status` (`pending`, `applied`, `rejected`, `expired`, `conflicted`)
- `review_notes`
- `created_at`
- `applied_at`
- `rejected_at`

Proposal commits append validated typed events. Proposals may show a patch preview, but no proposal applies arbitrary snapshot patches directly.

Roll/check events require explicit provenance:

- server-rolled
- user-entered
- imported
- model-suggested

Server rolls should store expression, normalized formula, result parts, total, random source metadata where practical, adapter interpretation, rule references, and citation metadata.

## Optimistic Concurrency And Idempotency

Every mutating request must include an idempotency key. REST may accept either a request field or `Idempotency-Key` header, but the service should normalize to one internal value. MCP tools provide the key as an argument.

Every ledger-affecting write must include `expected_last_event_sequence` or an equivalent base snapshot/event version. Stale writes return conflict and do not mutate state.

Idempotent replay with the same payload returns the original result. Reusing the same key with a different normalized payload returns `rpg_idempotency_key_conflict`.

## REST API Surface

All routes are under `/api/v1/rpg`.

Rules routes:

- `GET /rules/adapters`
- `GET /rules/adapters/{adapter_key}`
- `POST /rules/lookup`

Campaign routes:

- `POST /campaigns`
- `GET /campaigns`
- `GET /campaigns/{campaign_id}`
- `PATCH /campaigns/{campaign_id}`
- `DELETE /campaigns/{campaign_id}` for soft delete/archive only

Session routes:

- `POST /campaigns/{campaign_id}/sessions`
- `GET /sessions/{session_id}`
- `PATCH /sessions/{session_id}`
- `GET /sessions/{session_id}/events`
- `POST /sessions/{session_id}/events`
- `GET /sessions/{session_id}/snapshot`
- `POST /sessions/{session_id}/snapshot/rebuild`
- `POST /sessions/{session_id}/rolls`
- `POST /sessions/{session_id}/proposals`
- `POST /sessions/{session_id}/proposals/{proposal_id}/apply`
- `POST /sessions/{session_id}/proposals/{proposal_id}/reject`

Permission families:

- `rpg.read`: read campaigns, sessions, events, snapshots, adapters, and permitted rules lookups.
- `rpg.write`: create/update campaigns and sessions, append deterministic events, commit rolls, create proposals, apply/reject proposals.
- `rpg.admin`: repair/admin operations such as snapshot rebuild and future hard-delete operations.
- `rpg.rules.write`: future rules-pack binding or adapter authoring operations if they become mutable.

`POST /sessions/{session_id}/rolls` supports simulation and commit:

- `commit: false` returns a result without appending an event.
- `commit: true` requires idempotency and expected sequence, then appends a roll/check event.

`POST /rules/lookup` supports explicit source filters:

- `adapter_key`
- `campaign_id`
- `session_id`
- `include_bundled`
- `include_user_packs`
- `trust_levels`
- `max_results`

Rules lookup must not accidentally search unrelated user-ingested content. It only searches bundled snippets and rules packs linked through campaign/session scope unless the request explicitly uses another allowed scope.

`POST /sessions/{session_id}/snapshot/rebuild` is admin/test-repair scoped in v1, not a normal user write path.

Deletes are soft archive/delete with expected version. Hard delete is deferred or admin-only.

## MCP Tool Surface

MCP exposes a smaller, safer tool set over the same service behavior.

Read tools:

- `rpg.list_rules_adapters`
- `rpg.get_session_state`
- `rpg.list_recent_events`
- `rpg.lookup_rule`

Deterministic write tools:

- `rpg.roll_check`
- `rpg.record_event`

Proposal tools:

- `rpg.create_proposal`
- `rpg.apply_proposal`
- `rpg.reject_proposal`

Context tool:

- `rpg.build_session_context`

Tool permissions follow existing MCP naming:

- `tools.execute:rpg.get_session_state`
- `tools.execute:rpg.record_event`
- `tools.execute:rpg.apply_proposal`
- and equivalent per-tool permissions.

Read and write tools are intentionally separate. Read-only tool access must not imply write access. Write tools require per-tool RBAC permission, idempotency key, source metadata, and expected event sequence when they can mutate session state.

`rpg.record_event` must enforce authority policy. If the caller uses `source_type=model`, the tool routes to proposal behavior unless the session explicitly enables auto-commit for that source/action.

`rpg.build_session_context` accepts a token or character budget and returns:

- compact session state summary
- recent event digest
- selected open rulings
- cited bundled and user rules excerpts
- included/excluded diagnostics
- truncation and trust warnings

## Authority Policy

The service owns all write decisions. Endpoint and MCP layers do not decide whether a model-derived change may mutate state.

Default behavior:

- Deterministic/manual events may commit directly when the caller has write permission and concurrency checks pass.
- Server-side committed rolls append direct roll/check events.
- Model-derived changes become proposals by default.
- Auto-commit is disabled by default and may be enabled per session for specific source/action classes.
- Applying a proposal appends normal typed events with proposal lineage.
- Rejected proposals never mutate events or snapshots.

Authority policy must also protect MCP tools. A model-facing tool cannot bypass proposal review simply by calling a direct write tool with model-sourced metadata.

## Service Behavior

The endpoint layer authenticates, validates Pydantic schemas, and translates service errors. The RPG service owns domain behavior.

Service responsibilities:

- Resolve campaign/session ownership and permissions.
- Validate adapter keys and versions.
- Validate event payloads against registered event types.
- Enforce idempotency for every mutation.
- Enforce optimistic concurrency.
- Apply authority policy.
- Append typed events transactionally.
- Run the reducer after successful event append.
- Rebuild snapshots from the ledger for admin/test repair flows.
- Resolve rules lookup through bundled packs and linked user rules packs.
- Build bounded context with citations and diagnostics.

Reducer behavior:

- Deterministic and side-effect-free.
- Accepts prior snapshot plus typed events.
- Returns next snapshot plus diagnostics.
- Does not call external services.
- Maintains stable entity IDs.
- Handles unknown future event versions through explicit compatibility logic or fails with a stable error.

Long campaigns should use checkpointed/cached snapshots. V1 can rebuild from the full ledger for correctness, but the design should preserve a strategy of replaying from the latest valid snapshot checkpoint for performance once ledgers grow large.

## Error Handling

Use stable error codes rather than raw strings.

Suggested codes:

- `rpg_campaign_not_found`
- `rpg_session_not_found`
- `rpg_rules_adapter_not_found`
- `rpg_rules_adapter_version_conflict`
- `rpg_invalid_event_type`
- `rpg_invalid_event_payload`
- `rpg_stale_event_sequence`
- `rpg_idempotency_key_required`
- `rpg_idempotency_key_conflict`
- `rpg_proposal_not_found`
- `rpg_proposal_not_applicable`
- `rpg_authority_policy_requires_proposal`
- `rpg_rules_lookup_scope_invalid`
- `rpg_snapshot_rebuild_forbidden`
- `rpg_snapshot_rebuild_failed`

HTTP mapping:

- `400`: malformed request, invalid event type/payload, invalid lookup scope.
- `401` / `403`: existing auth and permission dependencies.
- `404`: missing campaign/session/proposal in the current user's scope.
- `409`: stale sequence, adapter version conflict, idempotency conflict, proposal conflict, or proposal no longer applicable.
- `422`: schema-valid request with semantically invalid domain fields if existing route style prefers that split.
- `500`: unexpected failures after structured logging with context.

## Testing Strategy

Testing should validate deterministic state behavior and security boundaries, not model quality.

Core unit tests:

- Adapter registry lists `dnd5e_srd`, `pf2e`, and `fate`.
- Each adapter exposes required metadata, license details, mechanics schema, and snippet attribution.
- Bundled snippets fail validation without source title, URL, license, license URL, attribution, trust level, and adapter key.
- D&D/PF2e dice/check helpers work for expected d20-style patterns.
- Fate tests prove aspects/tags, stress, consequences, and non-d20 outcomes work through the common adapter contract.
- Event validation rejects unknown event types and malformed payloads.
- Reducer produces stable snapshots from known event ledgers.
- Reducer property/invariant tests cover determinism, unique stable entity IDs, rebuild equality, proposal isolation, and rejected proposal non-mutation.
- Reducer rebuild from scratch matches incrementally cached snapshots.
- Old snapshot/reducer versions are detected and routed to rebuild/compatibility behavior.
- Proposals cannot mutate snapshots until applied.
- Applying proposals appends typed events and preserves audit/source metadata.
- Authority policy routes model-derived writes to proposals unless auto-commit is enabled.
- Idempotency replay returns the original result and conflicting payloads fail.
- Stale expected sequence fails with conflict.

API tests:

- Campaign/session CRUD is owner-scoped and soft-delete/archive aware.
- Mutating endpoints require idempotency.
- REST permission matrix covers `rpg.read`, `rpg.write`, and `rpg.admin`.
- Event append updates snapshot version and last event sequence.
- Roll simulation does not append an event; committed roll does.
- Rules lookup respects adapter, campaign, session, source, and trust filters.
- Rules lookup uses deterministic fixtures and does not require live embeddings or external content.
- Snapshot rebuild is denied without admin/test repair permission.
- Error codes map to expected HTTP statuses.

MCP tests:

- Read tools do not require write permissions.
- Write tools require per-tool `tools.execute:<tool>` permission.
- Read-only users cannot append events through MCP.
- `record_event` routes model-sourced writes to proposals unless auto-commit is enabled.
- `build_session_context` respects budgets and returns inclusion/exclusion diagnostics.

Security and verification:

- Run focused unit/API/MCP tests for touched RPG paths.
- Run Bandit on touched backend paths before completion.
- Use deterministic fixtures for rules snippets and user rules-pack lookup.
- Do not require live external providers for tests.

## Rollout Plan

Phase 1: Core ledger and adapters

- Add RPG package structure, repository, schemas, service, and route registration.
- Implement campaign/session CRUD.
- Implement adapter registry for D&D 5e SRD, Pathfinder 2e, and Fate.
- Implement typed event append, reducer, cached snapshots, idempotency, and optimistic concurrency.
- Implement dice/check helpers and roll simulation/commit.

Phase 2: Proposals and rules lookup

- Implement authority policy and proposal queue.
- Add bundled snippet packs with license validation.
- Add linked user rules-pack lookup through existing retrieval abstractions.
- Add proposal apply/reject and citation-aware rule lookup tests.

Phase 3: MCP tools and context builder

- Register read/write/proposal/context tools through MCP Unified.
- Add per-tool RBAC tests.
- Implement bounded session context assembly with citations and diagnostics.

Phase 4: Reference docs and examples

- Add API examples and adapter authoring notes.
- Add minimal sample campaigns for all three systems.
- Document non-goals and the distinction from a VTT.

## Future Work

Future specs can build on this runtime with:

- GM orchestration loop.
- Quest/NPC/location/faction tracker agents.
- Combat assistant and initiative/encounter helpers.
- Session summarizer and recap automation.
- Rules-pack import UX.
- Export/import bundles, possibly integrating with Chatbooks.
- Optional WebUI inspection and setup surfaces.

These are intentionally deferred so the first implementation remains focused on the ledger, snapshots, rules adapters, authority policy, and tool/API contract.

## Self-Review Checklist

- No unresolved markers or implementation blockers remain in this spec.
- The first implementation slice is bounded to core campaign/session runtime.
- The design preserves the user's long-term full-orchestrator direction.
- REST and MCP write paths both enforce idempotency, optimistic concurrency, and authority policy.
- Rules content has explicit source/license/attribution constraints.
- Non-goals prevent v1 from drifting into a VTT.
