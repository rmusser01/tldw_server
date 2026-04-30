# Phase 3.2 Pagination Standardization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking. Complete the inventory and compatibility contract before endpoint migrations.

**Goal:** Standardize API v1 pagination request and response semantics while preserving existing clients during a phased migration.

**Architecture:** Create shared pagination request/response helpers and a route-family inventory before changing endpoints. Prefer canonical `limit`/`offset` for offset pagination and canonical opaque cursors for cursor pagination. Existing `page`, `per_page`, `results_per_page`, `next_offset`, and provider-specific forms are accepted as compatibility aliases until each endpoint family is migrated and clients are updated.

**Tech Stack:** FastAPI `Query`, Pydantic, OpenAPI, pytest, shared UI API client, Vitest

---

## Current Inventory

Measured on 2026-04-25:

- `173` schema classes contain pagination-like fields.
- Offset-style fields appear as top-level `limit`/`offset`, nested `pagination.limit`/`pagination.offset`, and ad hoc request fields such as `rows_limit`/`rows_offset`.
- Page-style fields appear as `page`, `per_page`, `results_per_page`, `current_page`, `total_pages`, and `total_items`.
- Cursor-style fields appear as `cursor`, `after`, `next_cursor`, and provider-specific integer cursors.
- `endpoints/_pagination_utils.py` currently only builds RFC5988 `Link` headers and does not normalize request inputs or response metadata.

Representative shapes:

- `media_response_models.PaginationInfo`: `page`, `results_per_page`, `total_pages`, `total_items`
- `prompt_studio_base.PaginationMetadata`: `page`, `per_page`, `total`, `total_pages`
- `chat_conversation_schemas.ConversationListPagination`: `limit`, `offset`, `total`, `has_more`
- `document_references.DocumentReferencesResponse`: `offset`, `limit`, `has_more`, `next_offset`
- `workflows.WorkflowRunListResponse`: `next_offset`, `next_cursor`
- `openai_eval_schemas.ListQueryParams`: `limit`, `after`
- Draft helper contract spec created: `Docs/superpowers/reviews/api-pagination/2026-04-25-helper-contract-spec.md`.

## Canonical Contract Decision

Offset pagination response metadata should converge on:

```json
{
  "pagination": {
    "mode": "offset",
    "limit": 50,
    "offset": 0,
    "total": 123,
    "has_more": true,
    "next_offset": 50
  }
}
```

Cursor pagination response metadata should converge on:

```json
{
  "pagination": {
    "mode": "cursor",
    "limit": 50,
    "cursor": "input-cursor-or-null",
    "next_cursor": "opaque-token-or-null",
    "has_more": true
  }
}
```

Compatibility rules:

- Existing query parameters keep working during migration.
- Existing response fields remain present for migrated route families until the owning frontend clients are updated.
- Provider-specific third-party-compatible endpoints can remain provider-shaped and only adopt shared internal helpers where it does not change their public contract.
- Phase 3.1 decides where `pagination` lives in the standard envelope; Phase 3.2 owns the contents of the pagination object.

## File Structure

- Create: `tldw_Server_API/app/api/v1/schemas/pagination.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/_pagination_utils.py`
- Create: `tldw_Server_API/tests/Utils/test_pagination_contract.py`
- Create: `Docs/superpowers/reviews/api-pagination/2026-04-25-pagination-inventory.md`
- Modify: selected pilot endpoint family after inventory
- Modify: `apps/packages/ui/src/services/tldw/*` pagination parsing only after backend pilot is green

## Task 1: Build The Pagination Inventory

- [x] Generate a route-family inventory of pagination request parameters from endpoint decorators and function signatures. See `Docs/superpowers/reviews/api-pagination/2026-04-25-pagination-inventory.md`.
- [x] Generate a schema inventory of pagination response fields from `app/api/v1/schemas`.
- [x] Categorize each route family as `offset`, `page`, `cursor`, `provider`, `hybrid`, or `not paginated`. See `Docs/superpowers/reviews/api-pagination/2026-04-25-route-family-catalogue.md`.
- [x] Identify frontend callers for the first migration candidates: `skills`, `slides`, and `data_tables`.
- [x] Record route families that should be exempt or deferred because they are third-party-compatible or streaming/file-based at the route-family signal level. Route-by-route exemption confirmation remains part of each migration slice.

## Task 2: Add Shared Pagination Models And Helpers

- [x] Draft the helper, alias-precedence, bounds, metadata, and Link-header contract spec. See `Docs/superpowers/reviews/api-pagination/2026-04-25-helper-contract-spec.md`.
- [ ] Add `OffsetPaginationMeta` and `CursorPaginationMeta` schemas.
- [ ] Add a `PaginationMeta` discriminated union if it keeps generated OpenAPI readable.
- [ ] Add request-normalization helpers for:
  - canonical `limit`/`offset`
  - legacy `page`/`per_page`
  - legacy `page`/`results_per_page`
  - cursor/after inputs
- [ ] Extend `_pagination_utils.py` so metadata and `Link` headers are derived from the same normalized values.
- [ ] Add unit tests for bounds, alias precedence, `has_more`, `next_offset`, cursor handling, and `Link` header output.

## Task 3: Pick A Low-Risk Offset Pilot

Recommended pilot candidates:

- `slides` list/search/style/version endpoints
- `skills` list endpoint
- `data_tables` list/detail row-window endpoints

Selected planning candidate:

- [x] Draft the `skills` pilot execution packet for offset metadata, frontend compatibility, and verification gates. See `Docs/superpowers/reviews/phase3-pilots/2026-04-25-skills-pilot-execution-packet.md`.

Pilot steps:

- [ ] Preserve existing request parameters.
- [ ] Add canonical nested `pagination` metadata where absent or incomplete.
- [ ] Keep legacy top-level pagination fields during the compatibility window.
- [ ] Add backend tests for canonical and legacy aliases.
- [ ] Update the shared UI client parser for the pilot route family.
- [ ] Add focused UI tests for old and new response shapes.

## Task 4: Pick A Cursor Pilot

Recommended pilot candidates:

- `workflows` runs/events
- `audio_history`
- `evaluations` OpenAI-style list endpoints if provider compatibility is preserved

Pilot steps:

- [ ] Confirm cursor opacity and sorting guarantees.
- [ ] Ensure `next_cursor` is null when no more results remain.
- [ ] Ensure `Link` headers agree with response metadata.
- [ ] Add tests for invalid cursors and stable ordering.
- [ ] Update client parsing only for the selected pilot family.

## Task 5: Prepare Route-Family Migration Slices

Recommended slice order:

1. Newer nested-pagination families: chat conversations, slides, kanban, data tables.
2. Top-level offset families: notes, admin lists, storage, skills, flashcards, quizzes.
3. Page-style first-party families: media, prompts, watchlists, privileges.
4. Cursor/hybrid families: workflows, audio history, evaluations.
5. Provider-specific search families only when compatibility rules are explicit.

- [ ] Create one PR per small route-family group.
- [ ] Keep backend and frontend changes in the same PR when response parsing changes.
- [ ] Update OpenAPI verification and fallback schemas as each client path changes.
- [ ] Add a deprecation note for legacy aliases only after all known clients have migrated.

## Verification

Minimum verification before any Phase 3.2 PR:

```bash
python3 -m pytest tldw_Server_API/tests/Utils/test_pagination_contract.py -v
python3 -m pytest <pilot backend test files> -v
cd apps/packages/ui && bunx vitest run <pilot client tests>
cd apps/packages/ui && npm run verify:openapi
```

If Python source files outside schemas/utils are modified, run focused Bandit on touched Python paths.

## Out Of Scope

- Standard success/error envelopes; Phase 3.1 owns the envelope wrapper.
- Changing OpenAI/third-party-compatible pagination shapes without explicit compatibility approval.
- Rewriting data-access pagination algorithms unless a route family has a known correctness bug.
- API versioning policy; Phase 4.5 owns the broader versioning decision.
