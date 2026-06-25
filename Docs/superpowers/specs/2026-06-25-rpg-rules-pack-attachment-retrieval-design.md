# RPG Rules-Pack Attachment And Retrieval Design

Date: 2026-06-25
Status: Approved design, pending implementation planning
Backlog task: TASK-12029

## Context

The RPG runtime currently provides citation-only bundled rule references, campaign/session state, deterministic snapshots, REST endpoints, and an optional MCP module. Campaigns and sessions already have JSON rules-pack reference fields, but creation paths initialize them empty and lookup currently returns only bundled adapter citations.

This design adds user-provided rules-pack attachment and retrieval-backed lookup without turning RPG into a virtual tabletop and without copying long-form rules prose into RPG tables. It reuses existing ingestion, Media DB, media collections, and RAG retrieval boundaries where possible.

The approved direction is hybrid:

- Store direct references to existing media items and media collections now.
- Reserve a registry-compatible reference shape for future first-class reusable rules packs.
- Copy campaign references to a session when the session is created, then let the session diverge independently.
- Keep lookup snippet/citation-first, with opt-in generated answer mode.
- If attached retrieval misses, return bundled citation-only references only. Do not search broader user content or the web.

## Goals

- Let users attach, list, replace, and remove campaign/session rules references.
- Keep RPG tables free of user-provided rules prose.
- Blend citation-only bundled adapter references with retrieved user snippets in a clear, attributed response.
- Support opt-in generated answers grounded only in attached user snippets.
- Include retrieved snippets in session context within existing bounds and diagnostics.
- Mirror REST and MCP behavior.
- Preserve AuthNZ, idempotency, optimistic concurrency, privacy, and licensing boundaries.

## Non-Goals

- No first-class reusable rules-pack registry in this task.
- No bundled long-form rules prose for D&D, Pathfinder, Fate, or other systems.
- No automatic web fallback.
- No lookup across un-attached user documents.
- No change to RPG being a backend orchestration/reference harness rather than a FoundryVTT-like tabletop.

## Reference Model

Use the existing JSON fields as the first storage boundary:

- `rpg_campaigns.linked_rules_pack_refs_json` stores campaign defaults.
- `rpg_sessions.active_rules_pack_refs_json` stores session-active references.

Session creation copies the current campaign list into the session. Later campaign edits do not affect existing sessions. Later session edits do not affect the campaign.

Each reference is normalized before persistence:

```json
{
  "source_type": "media_collection",
  "source_id": "7",
  "label": "Fate SRD local copy",
  "enabled": true,
  "adapter_keys": ["fate"],
  "source_policy": "user_provided",
  "snapshot_policy": "live",
  "created_at": "2026-06-25T00:00:00Z",
  "updated_at": "2026-06-25T00:00:00Z"
}
```

Initial `source_type` values:

- `media_item`
- `media_collection`

Reserved future `source_type` value:

- `rules_pack`

`source_id` is stored as a string for future compatibility, but validated according to `source_type`. `adapter_keys` are advisory metadata for filtering and diagnostics, not a hard storage gate. A user can attach any readable rules source to any RPG adapter.

`media_collection` references are live. Lookup resolves ready collection items at query time. Users who need reproducible pinned behavior should attach individual `media_item` references instead.

Removing a reference means replacing the stored list without that reference. There is no separate tombstone table in the first implementation.

## Attachment Writes

The first implementation uses whole-list replacement because it is deterministic and easy to make idempotent.

REST write request shape:

```json
{
  "expected_version": 3,
  "refs": [
    {
      "source_type": "media_item",
      "source_id": "42",
      "label": "House rules",
      "enabled": true,
      "adapter_keys": ["dnd5e_srd"]
    }
  ]
}
```

Write behavior:

- Require `Idempotency-Key`.
- Require `expected_version`.
- Normalize refs before hashing or persistence.
- Reject unsupported source types.
- Reject duplicate active refs with the same `(source_type, source_id)`.
- Enforce a small maximum ref count per object.
- Validate referenced media or collection ownership/readability.
- Update the parent campaign/session `version` and `updated_at` on success.
- Return the updated normalized refs plus the new version.

Idempotency scopes are object-specific:

- `campaign:{campaign_id}:rules_pack_refs`
- `session:{session_id}:rules_pack_refs`

Conflicts and validation:

- `409` for stale `expected_version`.
- `409` for idempotency replay with a different normalized request hash.
- `400` for invalid ref shape, unsupported source type, duplicate refs, or unreadable references visible as invalid input.
- `404` for missing campaign/session, or for fail-closed source visibility cases where the caller should not learn whether the referenced source exists.

## REST Surface

Add these endpoints under `/api/v1/rpg`:

- `GET /campaigns/{campaign_id}/rules-packs`
- `PUT /campaigns/{campaign_id}/rules-packs`
- `GET /sessions/{session_id}/rules-packs`
- `PUT /sessions/{session_id}/rules-packs`

Extend existing lookup:

- `POST /sessions/{session_id}/rules/lookup`

Lookup request gains:

- `mode`: `"lookup"` or `"answer"`, default `"lookup"`

Retrieval bounds remain service-controlled in the first implementation. Caller-tunable lookup limits can be added later if the response contract proves stable.

GET ref responses return:

- normalized `refs`
- the parent campaign or session `version`

Permission mapping:

- Campaign ref read: `rpg.campaigns.read`
- Campaign ref write: `rpg.campaigns.manage`
- Session ref read: `rpg.sessions.read`
- Session ref write: `rpg.sessions.manage`
- Rules lookup and adapter metadata: `rpg.rules.read`

Privilege catalog tests should confirm every endpoint scope used by the RPG endpoint remains cataloged.

## MCP Surface

Extend the RPG MCP module in parallel with REST:

- Add read tools for campaign/session rules-pack refs.
- Add write tools for replacing campaign/session rules-pack refs.
- Extend `rpg.rules.lookup` with optional `mode`.

MCP write tools require:

- `expected_version`
- `idempotencyKey` or `idempotency_key`
- full replacement `refs`

MCP write tool metadata must keep the current governance posture:

- `readOnlyHint: false`
- `is_write: true`
- `mutates_state: true`
- `requires_confirmation: true`
- `agent_write_policy: "approval_required"`
- `governance_preflight_required: true`

Read tools remain `readOnlyHint: true`.

## Retrieval Integration

Add an async injectable RPG rules retrieval adapter. The adapter receives:

- `query`
- normalized enabled refs
- owner/user context
- Media DB and collection access dependencies
- retrieval limits

The adapter returns a canonical evidence list rather than a full RAG answer. It should call existing retrieval components, not the public REST endpoint.

Initial retrieval behavior:

- `media_item`: pass validated IDs as `include_media_ids` or `allowed_media_ids`.
- `media_collection`: resolve ready media IDs using existing collection ownership/readiness semantics, then pass those IDs.
- `sources`: force `["media_db"]`.
- `search_mode`: default to `"hybrid"`, with a small capped `top_k`.
- Web fallback: forced off.
- Broad user-content fallback: forced off.
- Retrieval errors: captured in diagnostics; do not fail lookup unless the session cannot be loaded or request validation fails.

The adapter must be mockable for unit tests so the RPG suite does not require real embeddings, Chroma, or an external model.

## Rules Lookup Response

Lookup always returns ranked, attributed evidence and references.

Response shape should include:

- `query`
- `mode`
- `results`
- `answer`
- `answer_status`
- `answer_citation_ids`
- `diagnostics`

`answer` is present only when `mode="answer"` and a generated answer was successfully grounded in attached user snippets.

Suggested `answer_status` values:

- `not_requested`
- `answered`
- `no_attached_evidence`
- `generation_failed`
- `disabled_by_config`

Result origins:

- `user_provided`: retrieved from attached user sources
- `bundled_citation`: bundled adapter citation metadata only

User-provided result items include:

- `id`, such as `user:media:42:chunk-8`
- `text`
- `score`
- `origin`
- `source_type`
- `source_id`
- `chunk_id`, if present
- `source_title`, if available
- `source_url` or a safe source label, if available
- `license` and `attribution`, if known
- `trust_level: "user_provided"`

Bundled citation items include:

- `id`, such as `bundled:fate:1`
- empty `text`
- `score: 0.0`
- `origin: "bundled_citation"`
- existing adapter citation metadata

Ranking:

1. Qualifying user snippets, sorted by retrieval score.
2. Bundled citation-only references for the active adapter.

Bundled citation scores do not compete with user retrieval scores.

If no attached user snippets are usable, lookup returns bundled citation-only references and diagnostics. It does not search unrelated user content or web sources.

## Generated Answer Mode

`mode="answer"` adds synthesis only when attached user snippets exist.

Generated answers must be grounded in the retrieved snippets:

- Use retrieved snippets as the only rules evidence.
- Include machine-readable `answer_citation_ids`.
- Avoid making official-source claims for bundled citation-only adapters.
- Degrade to snippets only when generation fails, is disabled, or cannot stay grounded.
- Never generate an answer from bundled citation-only references alone.

This mode may use existing LLM generation facilities, but implementation should keep the generator injectable and testable with a fake.

## Context Builder

The session context builder keeps session state primary.

When a context request includes a query:

- Call rules lookup in `mode="lookup"`.
- Include retrieved user snippets, not generated answers.
- Include bundled citation references only if space remains.

Context ordering:

1. Session header and adapter.
2. Scene and core snapshot state.
3. Recent notes and open rulings.
4. `Rules excerpts:` with bounded user snippets and compact citation handles.
5. `Rules references:` with bundled citation-only references if there is room.

Rules excerpts use stable handles matching lookup result IDs, such as:

- `user:media:42:chunk-8`
- `bundled:fate:1`

Context budget behavior:

- Preserve existing `max_chars` bounds.
- Cap each snippet.
- Bound the total rules section after session state has been admitted.
- Preserve source title and citation handle even when snippet text is truncated.

Diagnostics add:

- `rules_user_snippet_count`
- `rules_bundled_citation_count`
- `rules_omitted_count`
- `rules_truncated`
- `rules_retrieval_errors`
- `rules_omission_reasons`, such as `budget_exhausted`, `no_attached_refs`, `retrieval_error`, and `no_results`

Retrieval failures degrade to session state plus diagnostics. Context build should not fail the whole request unless the session itself cannot be loaded or the input is invalid.

## Privacy And Licensing

Bundled adapters remain mechanics metadata plus citation-only references. They must not bundle long-form rules prose in this task.

User snippets are retrieved only from user-attached sources. They are user-provided excerpts, not bundled content. RPG tables persist references and metadata only, not retrieved prose.

User-facing lookup/context responses may include source titles and attribution metadata when available. Logs and low-level diagnostics should avoid snippet text and unnecessary titles. Log counts, source types, opaque IDs, and error categories instead.

Generated answers must expose machine-readable provenance with `answer_citation_ids` and should not imply that user-provided excerpts are official bundled rules content.

## Backward Compatibility

Existing campaigns and sessions with empty or older JSON ref lists must continue to load.

Rules lookup without attached refs keeps the current bundled citation-only behavior, with richer diagnostics and response fields added in a backward-compatible way where possible.

Existing RPG tests for adapter listing, campaign/session creation, event recording, context building, and MCP tool governance should continue to pass.

## Failure Modes

Expected failure modes:

- Invalid source type: `400`
- Duplicate refs: `400`
- Missing or malformed `expected_version`: validation error
- Missing REST idempotency header: request validation error
- Missing MCP/service idempotency key: write validation error before mutation
- Stale campaign/session version: `409`
- Idempotency key reuse with different normalized payload: `409`
- Missing campaign/session: `404`
- Missing, deleted, not-ready, or unreadable source refs: fail closed through validation, using `400` or `404` according to visibility
- Retrieval unavailable or partial failure: lookup/context returns bundled citations and diagnostics
- Generation unavailable or failed: `answer_status="generation_failed"` and snippets still return

## Test And Verification Plan

Repository tests:

- Campaign ref replacement.
- Session ref replacement.
- Optimistic version conflicts.
- Idempotency replay and hash mismatch.
- Duplicate normalization.
- Session creation copying campaign refs.
- Backward compatibility for empty or old JSON lists.

Service tests:

- Lookup with no refs returns bundled citations.
- Lookup with attached media refs returns user snippets plus bundled references.
- Lookup with attached collection refs resolves ready media IDs.
- Lookup does not search broad user content.
- Retrieval errors degrade to diagnostics.
- `mode="answer"` returns no answer without snippets.
- `mode="answer"` includes `answer_citation_ids` when answered.
- Generation failure degrades to snippets only.

Context tests:

- User snippets appear under rules excerpts.
- Bundled citations appear only as references.
- Citation handles match lookup IDs.
- Truncation preserves attribution handles.
- Omission reasons are diagnostic and stable.
- Retrieval failures do not fail context build.

REST tests:

- Read permissions for campaign/session ref listing.
- Write permissions for replacement.
- Request validation.
- Idempotency.
- Stale version conflicts.
- Missing/malformed refs.
- Response schemas and version returns.
- Lookup `mode` behavior.

MCP tests:

- New tool schemas.
- Read/write metadata.
- Required write arguments.
- Optimistic version input validation.
- Idempotency key aliases.
- Execution behavior with fake retrieval.
- Async lookup path works from MCP without event-loop bridging issues.

Retrieval and authorization tests:

- Fake async retriever path.
- Missing media item.
- Deleted media item.
- Missing media collection.
- Collection with no ready items.
- Source not readable by the user.
- Privacy-safe diagnostics without snippet text.

AuthNZ and catalog tests:

- RPG endpoint scope catalog sync.
- No uncataloged newly introduced scopes.

Verification commands:

- Focused RPG pytest suite.
- Relevant AuthNZ privilege catalog tests.
- Bandit on touched RPG/API/MCP/AuthNZ paths. For a design-only change, record that Bandit is not applicable to code behavior if no Python code changed.

## Implementation Slices

Implementation should be split after this design into separate tasks:

1. Repository and schema support for normalized ref replacement and session-copy behavior.
2. REST schemas/endpoints and permission tests.
3. Async retrieval adapter and service lookup response changes.
4. Context builder snippet inclusion and diagnostics.
5. MCP tool surface updates.
6. Documentation and regression verification.
