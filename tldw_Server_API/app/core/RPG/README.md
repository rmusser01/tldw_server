# RPG Runtime

The RPG runtime is a backend harness for tabletop roleplaying sessions. It stores campaigns, sessions, append-only events, cached snapshots, and reviewable state-change proposals in each user's ChaChaNotes database through `RPGRepository`.

It is not a virtual tabletop. It does not manage battle maps, tokens, token movement, lighting, walls, shared canvas rendering, or FoundryVTT-style scene automation. The runtime is intended to help reference rules, maintain structured session state, prepare bounded context for inference, and route state changes through explicit authority rules.

## Storage Model

RPG data lives in the authenticated user's ChaChaNotes database, alongside the rest of the user's local/self-hosted data. `RPGRepository.initialized(db)` ensures the RPG tables exist before use.

The main tables are:

- `rpg_campaigns`: campaign metadata, default rules adapter, settings, and linked rules pack references.
- `rpg_sessions`: session metadata, adapter version, authority settings, optional linked chat id, snapshot cursor, and event cursor.
- `rpg_session_events`: append-only event log with monotonically increasing `sequence_number` per session.
- `rpg_session_snapshots`: deterministic cached snapshots produced from committed events.
- `rpg_session_proposals`: reviewable event batches that are not yet committed.
- `rpg_idempotency_records`: replay records for mutating campaign, session, event, and proposal operations.

Event appends, proposal creation, and proposal application use optimistic sequence checks. Those requests provide the expected current event sequence, and stale requests fail rather than rewriting history. Campaign creation, session creation, proposal rejection, event appends, and proposal application are idempotent. Replaying the same payload returns the stored result, while reusing a key with a different payload raises `idempotency_key_conflict`.

## Events And Snapshots

Committed session state is derived from append-only events. Supported event types include scene, actor, NPC, quest, inventory, location, faction, clock, roll, note, rule reference, and ruling updates. Each event has:

- `event_type`
- `event_payload`
- `source_type`: one of `user`, `system`, `mcp`, `model`, or `import`
- adapter and schema version metadata
- a stable id field inside `event_payload` based on the event type

The reducer applies events into `RPGSnapshotState`, which contains structured session fields such as scene, actors, resources, clocks, rolls, notes, quests, NPCs, inventory, locations, factions, rules references, and unresolved rulings.

## Rules Adapters

The bundled adapter registry currently includes:

- `dnd5e_srd`: D&D 5e SRD 5.1 mechanics metadata and citations.
- `pf2e`: Pathfinder 2e Remaster mechanics metadata and citations.
- `fate`: Fate Core mechanics metadata and citations.

Adapters expose actor/check schemas and check resolution helpers. They do not bundle long-form rules prose. Built-in rules lookup is citation-first: it returns reference metadata and citation diagnostics, not copied rulebook text.

The data model has fields for campaign and session rules pack references, and lookup services accept those references. Current REST create paths initialize those lists empty; attaching and managing user-provided rules packs is a follow-up surface. Current built-in lookup returns citation metadata and reports linked rules pack counts; it does not retrieve user rules-pack prose yet. User-provided rules prose should remain in the existing ingestion/RAG systems and be referenced from RPG records rather than copied into RPG tables.

## Authority And Proposals

Authority is source-aware:

- `user` and `system` events commit directly.
- `import` events commit only when `import_auto_commit` is enabled.
- `mcp` events commit only when `mcp_auto_commit` is enabled.
- `model` events commit only when `model_auto_commit` is enabled and the event type is allowed.

Otherwise, the service creates a pending proposal. Applying a proposal validates the proposal's session, status, and base event sequence, then commits the proposed events atomically. Rejecting a proposal marks it rejected without appending session events.

New sessions default to conservative authority settings with `model_auto_commit` and `mcp_auto_commit` disabled.

## Context Builder

`RPGService.build_context()` creates bounded text context for inference. It combines the session title, adapter key, current snapshot summary, recent notes, open rulings, NPC names, and optional rules citations. The context size is bounded between 1,000 and 24,000 characters, and diagnostics report truncation, returned character count, rules result count, and omitted sections.

This context is intended for reference/orchestration and DM-assistant workflows. It is not a live game renderer.

## REST Surface

The REST router is registered under `/api/v1/rpg`. Current endpoints include:

- `GET /rules/adapters`
- `GET /rules/adapters/{adapter_key}`
- `POST /campaigns`
- `POST /campaigns/{campaign_id}/sessions`
- `POST /sessions/{session_id}/events`
- `POST /sessions/{session_id}/rules/lookup`
- `POST /sessions/{session_id}/context`
- `POST /sessions/{session_id}/proposals/{proposal_id}/apply`
- `POST /sessions/{session_id}/proposals/{proposal_id}/reject`

Mutating REST calls require the `Idempotency-Key` header. Endpoints use RPG-specific permissions, rate limits, and token-scope guards such as `rpg.rules.read`, `rpg.campaigns.manage`, `rpg.sessions.read`, `rpg.sessions.manage`, and `rpg.proposals.review`.

REST request limits include a maximum of 20 events per event-recording request, 64 KiB per event payload after canonical JSON encoding, 500 characters for rules/context queries, 1,000 to 24,000 characters for context output, and 2,000 characters for proposal review notes.

## MCP Surface

The MCP module is optional and disabled by default. Enable it with `MCP_ENABLE_RPG_MODULE`.

Read tools:

- `rpg.adapters.list`
- `rpg.sessions.get`
- `rpg.rules.lookup`
- `rpg.context.build`

Write-classified tools:

- `rpg.events.record`
- `rpg.proposals.apply`
- `rpg.proposals.reject`

Database-backed MCP tools require an authenticated request context with a per-user ChaCha DB path. Write tools are categorized as management operations, require idempotency through `idempotencyKey` or `idempotency_key`, and are marked for approval/governance preflight. Direct module execution validates arguments before opening the database so invalid ids, missing idempotency, bad sequence values, and out-of-bounds context/query inputs fail closed.

MCP write calls enforce a 256-character idempotency key limit. MCP rules/context queries share the 500-character query limit, context requests share the 1,000 to 24,000 character output bound, and proposal review notes share the 2,000-character limit.

## Current Non-Goals

This runtime does not currently provide:

- map, grid, token, wall, lighting, or shared tabletop rendering features
- campaign/session list endpoints
- event list endpoints
- a REST or MCP dice-roll endpoint
- bundled long-form rules prose
- autonomous model state commits by default

Those boundaries are intentional for this backend-first harness. The runtime should remain generic enough to support multiple TTRPG systems while letting tldw_server handle inference, retrieval, session memory, and reviewable orchestration.
