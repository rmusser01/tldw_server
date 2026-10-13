# RPG Core API Surface Completion — Design

- **Date**: 2026-10-11
- **Task**: TASK-13561
- **Status**: Draft (pending review)
- **Governing designs**: `Docs/superpowers/specs/2026-06-25-rpg-campaign-session-runtime-design.md`, `Docs/superpowers/specs/2026-06-25-rpg-rules-pack-attachment-retrieval-design.md`
- **ADR check**: ADR required: **no**. This work is additive REST surface extending the RPG module's established patterns (metadata CAS + idempotency, owner-scoped queries, stable domain-error mapping). It creates no new durable cross-module architecture rule; the two governing design docs above remain authoritative for the module.

## Context

The RPG module (merged via PR #2530) ships create/record/review mutations but almost no read or discovery surface. Today a client can:

- create campaigns and sessions, record events, review proposals, replace rules-pack refs, run rules lookup, and build context —
- but **cannot** list campaigns, sessions, events, or proposals; cannot read a session or its snapshot over REST (MCP `rpg.sessions.get` only); cannot archive anything (the `status` columns on both tables are permanently `'active'`); and **cannot ever change session authority settings**, which are frozen at creation to `{model_auto_commit: false, mcp_auto_commit: false}` — so every `mcp`/`model`/`import` event is forever a proposal.

This is the deliberately descoped portion of the original runtime design (its "Current Non-Goals" list). Phase 1 of the agreed expansion program closes exactly this gap. Subsequent phases (separate tasks): dice/check engine wiring, TASK-12031 MCP answer-mode host controls, MCP discovery-tool parity, and WebUI.

## Goals

1. Full read/discovery surface over REST: list and get campaigns, sessions, events, proposals, and snapshots.
2. Lifecycle: archive/unarchive campaigns and sessions via status transitions.
3. Mutable authority settings with strict validation, making the existing `decide_authority()` knobs reachable.
4. Snapshot rebuild: transactional replay of the full event log through the pure reducer (repair flow).

## Non-goals (Phase 1)

- **No deletion** of campaigns/sessions (archive-only; event history is never destroyed).
- **No new event types, reducer changes, or schema/DDL changes.** `status` columns already exist on `rpg_campaigns` and `rpg_sessions`; authority fields already live in `authority_settings_json`.
- **No MCP tool changes** (parity deferred to the MCP-themed phase).
- No dice/check endpoints (Phase 2), no rules-pack registry, no compaction/pruning of snapshots or idempotency records, no frontend work.
- No changes to commit-vs-proposal semantics; exposing the knobs does not alter `decide_authority()` behavior.

## Core decision: metadata updates, not event sourcing, for config mutations

Status transitions and authority-settings changes are **version-bumped column updates** using the pattern already established by rules-pack refs replacement: `expected_version` CAS inside a single transaction, `Idempotency-Key` header, exact replay of stored responses on retry, `operation_scope` in the house format `{entity}:{id}:{action}`.

Rejected alternative: recording config changes as ledger events (full audit trail) — it blurs the module's deliberate split between gameplay state (events/snapshots) and campaign/session config (rows), and forces reducer + schema version churn. An audit table remains a purely additive future option if ever needed.

## API surface (12 new routes)

All routes are owner-scoped (foreign or missing ids → 404, matching existing behavior), use the existing `_read_dependencies`/`_write_dependencies` guard builders, and map errors through the existing `_map_error` (404 `RPGNotFoundError`, 409 `RPGConflictError`, 400 `RPGValidationError`/`ValueError`).

### Reads

| Route | Purpose | Permission |
|---|---|---|
| `GET /rpg/campaigns` | List owner's campaigns; optional `status` filter; ordered `updated_at DESC` | `rpg.campaigns.read` |
| `GET /rpg/campaigns/{id}` | Campaign detail incl. `version`, `status`, `session_count` | `rpg.campaigns.read` |
| `GET /rpg/campaigns/{id}/sessions` | Sessions within a campaign; optional `status` filter | `rpg.sessions.read` |
| `GET /rpg/sessions` | Flat list of owner's sessions; optional `campaign_id` + `status` filters | `rpg.sessions.read` |
| `GET /rpg/sessions/{id}` | Session detail incl. cursor, authority settings, `pending_proposal_count` | `rpg.sessions.read` |
| `GET /rpg/sessions/{id}/snapshot` | Latest cached snapshot; optional `?version=N` for a historical one | `rpg.sessions.read` |
| `GET /rpg/sessions/{id}/events` | Event log; optional `source_type` + `event_type` filters | `rpg.sessions.read` |
| `GET /rpg/sessions/{id}/proposals` | Proposals; optional `status` filter; ordered `created_at DESC` | `rpg.proposals.review` |

Notes:

- **Pagination.** Entity lists (campaigns, sessions, proposals) use offset pagination: `limit` 1–100 default 25, `offset` ≥ 0, envelope `items` + `total_count` + echoed paging params (align exact field names with the closest existing envelope in `app/api/v1/schemas/` during implementation). Events use cursor pagination: `after_sequence` ≥ 0 (default 0), `limit` 1–100 **default 50**, ascending by `sequence_number`; the envelope reports `next_after_sequence`, and clients stop when a page returns fewer than `limit` items. The 50 default reflects that event payloads are bounded at 64 KiB each (100 events ≈ 6.4 MiB worst case).
- **Ordering.** Entity lists are ordered `updated_at DESC, id DESC` (proposals: `created_at DESC, id DESC`) for deterministic pagination despite same-second timestamps. Omitting the `status` filter returns both active and archived rows. On `GET /rpg/sessions`, a `campaign_id` that is unknown or owned by another user returns 404, consistent with the module's owner-scoping behavior.
- **`GET .../proposals` uses `rpg.proposals.review`** (the review-workflow permission) rather than `rpg.sessions.read`, so the permission grant needed to apply/reject also suffices to discover what to review. A `rpg.sessions.read`-only principal can still see `proposal_id` references on events but cannot enumerate proposals; documented behavior.
- **`pending_proposal_count`** appears on session detail and session list items (filtered `COUNT` served by the existing `idx_rpg_proposals_session(owner, session, status)` index). **`session_count`** appears on campaign detail only, keeping list queries lean.
- The snapshot response reuses the existing `SnapshotResult` shape (`snapshot_version`, `last_event_sequence`, `snapshot` state, `diagnostics`) — `last_event_sequence` doubles as the natural cursor for subsequent event pagination.

### Mutations

All mutations below require the `Idempotency-Key` header, validated exactly like existing header-style routes: a bare header parameter, with the service raising 409 `idempotency_key_required` when empty. (The 8–128 length bound applies only to the refs-replacement body field, which Phase 1 does not use.) Mutations perform the idempotency replay check **before** any CAS check (retry of a succeeded mutation replays the stored response regardless of subsequent version changes, mirroring `commit_events_and_snapshot`), execute as single-transaction conditional writes — never read-modify-write across transactions — and return the updated entity (including its new `version`) as the response body, which is also what replay serves.

| Route | Purpose | Permission | Idempotency scope |
|---|---|---|---|
| `PATCH /rpg/campaigns/{id}` | Update `title`/`description`; status transition | `rpg.campaigns.manage` | `campaign:{id}:update` |
| `PATCH /rpg/sessions/{id}` | Update `title`; status transition | `rpg.sessions.manage` | `session:{id}:update` |
| `PUT /rpg/sessions/{id}/authority-settings` | Full-replace authority settings | `rpg.sessions.manage` | `session:{id}:authority_settings` |
| `POST /rpg/sessions/{id}/snapshot/rebuild` | Replay event log → fresh snapshot | `rpg.sessions.manage` | `session:{id}:snapshot_rebuild` |

- **Status transitions**: `status ∈ {active, archived}`, both directions allowed, no further state machine. Validated at the Pydantic/schema level (the columns carry no SQL CHECK). PATCH bodies are `extra="forbid"`; `owner_user_id`, `version`, timestamps, and adapter fields (`default_adapter_key`/`adapter_key` and versions — deliberately immutable after creation) are never assignable.
- **Field bounds**: `title` 1–200 chars, `description` ≤ 4000 chars, matching `RPGCampaignCreateRequest`.
- **Version semantics (important for clients)**: the session `version` increments on **every event commit** (`RPG_DB.py` `commit_events_and_snapshot`), not only on metadata changes. A PATCH/PUT `expected_version` therefore conflicts (409 `stale_version`) whenever events were recorded since the client last read — correct optimistic behavior already true of refs replacement; clients re-GET and retry.
- **Authority settings contract**: `PUT` accepts exactly the four fields `decide_authority()` reads — `import_auto_commit`, `mcp_auto_commit`, `model_auto_commit` (strict bools) and `model_auto_commit_event_types` (list of event types validated against the registry in `core/RPG/events.py`; unknown types → 400). Full-object replace, `extra="forbid"`. Absent/empty `model_auto_commit_event_types` means **no** model event type auto-commits (existing semantics preserved verbatim). This is a dedicated route, separate from PATCH, because enabling `model_auto_commit` changes what bypasses human review; it stays behind `rpg.sessions.manage`.

### Snapshot rebuild

`POST /rpg/sessions/{id}/snapshot/rebuild`:

1. Idempotency replay check (scope `session:{id}:snapshot_rebuild`).
2. In **one transaction**: load the session row, load the full event log ordered by `sequence_number`, fold from an empty `RPGSnapshotState` through the existing pure `reduce_events`, insert snapshot at `current_snapshot_version + 1` with `last_event_sequence` = last event's sequence, and update the session row (`last_event_sequence` — advancing it forward if it lagged the log, `current_snapshot_version`, `version = version + 1`) with a guarded conditional UPDATE.
3. Diagnostics carry `{"rebuilt": true, "event_count": N, "previous_snapshot_version": v}` so rebuilt snapshots are distinguishable from incremental ones.
4. Zero events → an empty-state snapshot (valid; session creation already seeds v0 empty).
5. **Concurrency**: a rebuild racing an event commit targets the same next snapshot version; the `UNIQUE(owner_user_id, session_id, snapshot_version)` constraint makes the loser fail with `sqlite3.IntegrityError`, mapped to 409 `snapshot_version_conflict` — retry-safe via the idempotency key (the failed attempt wrote nothing).

## Repository and service changes

New repository methods (`RPG_DB.py`): `list_campaigns`, `list_sessions`, `list_events`, `list_proposals` (owner-scoped, filtered, ordered, paged); `update_campaign`/`update_session` (metadata + status CAS updates); `get_snapshot_by_version`; `insert_rebuilt_snapshot` (transactional twin of the commit path's snapshot write, without event append). Existing `get_campaign`, `get_session`, `get_event`, `get_latest_snapshot`, `get_proposal` are reused.

New service methods (`core/RPG/service.py`): thin validated wrappers for the above plus `get_campaign`, `get_session_detail`, `get_snapshot(version=None)`, and rebuild orchestration, mirroring `replace_*_rules_pack_refs` for idempotency handling (payload hashing reuses the canonical-JSON + SHA-256 helpers from `core/RPG/events.py`).

No changes to `models.py` domain dataclasses beyond (if needed) list-item projection dataclasses; no changes to `events.py`, `reducer.py`, `authority.py`, or `constants.py`.

## Schemas

New Pydantic models in `app/api/v1/schemas/rpg_schemas.py`: list envelopes, detail responses (campaign/session/proposal/event), PATCH request bodies, authority-settings PUT body, rebuild response. All `extra="forbid"` with bounds matching existing models (body `expected_version` ≥ 1; `title` 1–200, `description` ≤ 4000). No `privilege_catalog.yaml` change is required: the catalog sync test only asserts that scope ids exist, and all 12 routes reuse existing scopes — only route-level permission tests are extended.

## Error handling

Stable codes via the existing `_map_error`: 404 for foreign/missing ids and unknown snapshot versions; 409 `stale_version` (CAS mismatch), `idempotency_key_conflict` (key reuse with different payload hash), `snapshot_version_conflict` (rebuild race); 400 for validation failures (unknown status, unknown event type in the authority whitelist, malformed paging params).

## CI artifacts and documentation

- **Resource Governor route map**: regenerate/update the RG policy route-map snapshot for the 12 new routes (precedent: TASK-12026's "route snapshot regenerated"); `Helper_Scripts/ci/rg_route_map_lint.py` and the route-auth ratchet must stay clean.
- **Permission catalog sync test** (`tests/RPG/test_rpg_api.py`) updated with the new route/permission rows.
- **RPG README** (`app/core/RPG/README.md`): extend the REST route table; remove the now-implemented items from "Current Non-Goals" (list/read endpoints, authority-settings mutation, snapshot rebuild) while keeping the still-true ones (no dice endpoint until Phase 2, no deletion, no MCP parity yet).
- Frontend OpenAPI type regeneration is Phase 4 prep, not Phase 1.

## Testing

- **Unit (repo/service)**: list filters + pagination edges; CAS mismatch → `stale_version`; idempotency exact replay and payload-mismatch conflict for each mutation; authority-settings validation (strict bools, event-type whitelist, `extra="forbid"` rejection); rebuild correctness — fold of full log equals the incrementally-maintained snapshot for healthy sessions; rebuild repairs a deliberately corrupted/lagged snapshot and advances the cursor; rebuild race → `snapshot_version_conflict`.
- **Integration (API)**: permission matrix for every new route (exact scopes, `media.read` not required); happy paths; 404 on foreign ids; 409 flows; 400 validation; replay semantics over HTTP.
- Full existing RPG suite (~158 tests) stays green.

## Verification

`python -m pytest tests/RPG/ -v` green; Bandit clean on touched paths; RG route-map lint and route-auth ratchet clean.

## Out of scope / future phases

Phase 2: dice/check engine wiring (REST + MCP roll endpoints, richer notation). Phase 3: TASK-12031 MCP answer-mode host controls + MCP discovery-tool parity + documenting `MCP_ENABLE_RPG_MODULE` in `Env_Vars.md`. Phase 4: WebUI. Later candidates: config audit table, snapshot compaction, event-list payload toggle, proposal creation endpoint for restricted sources over REST.
