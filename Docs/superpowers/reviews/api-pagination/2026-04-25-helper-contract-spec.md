# Phase 3.2 Pagination Helper Contract Spec

**Date:** 2026-04-25

**Status:** Draft contract for implementation after PR #1125 and the Phase 2 bases are accepted stable.

## Purpose

Define shared pagination models and normalization helpers before changing route families. The helpers should centralize bounds, aliases, metadata, and Link-header behavior while preserving existing request and response shapes during migration.

## Current Constraints

- `endpoints/_pagination_utils.py` currently only provides `build_link_header(...)`.
- Existing routes use several public parameter families: `limit`/`offset`, `page`/`per_page`, `page`/`results_per_page`, `rows_limit`/`rows_offset`, `cursor`, and `after`.
- Existing responses include nested `pagination`, top-level `total` fields, OpenAI-style cursor fields, and provider-specific shapes.
- Phase 3.1 owns where pagination metadata lives inside the standard envelope.
- Provider-compatible routes should keep their provider shape unless a route-specific compatibility plan exists.

## Canonical Metadata

Offset metadata:

```json
{
  "mode": "offset",
  "limit": 50,
  "offset": 0,
  "total": 123,
  "has_more": true,
  "next_offset": 50
}
```

Cursor metadata:

```json
{
  "mode": "cursor",
  "limit": 50,
  "cursor": "input-cursor-or-null",
  "next_cursor": "opaque-token-or-null",
  "has_more": true
}
```

Rules:

- `limit` is the canonical page size.
- `offset` is the canonical zero-based offset.
- `cursor` is the canonical input cursor name for first-party routes.
- `after` remains accepted for OpenAI-style and compatibility routes.
- `next_cursor = null` and `has_more = false` mean no next page.
- `next_offset = null` when `has_more = false` or when the next offset cannot be computed safely.

## Proposed Schemas

Create `tldw_Server_API/app/api/v1/schemas/pagination.py`.

Schema names:

- `OffsetPaginationMeta`
- `CursorPaginationMeta`
- `PaginationMode`
- Optional `PaginationMeta` union only if OpenAPI output remains readable.

Field guidance:

- `total` is optional because many routes cannot compute it cheaply.
- `has_more` is optional only at construction boundaries; serialized metadata should prefer a boolean when the route can know.
- `next_offset` and `next_cursor` are nullable.
- Do not force `page` or `total_pages` into canonical metadata. Preserve them as legacy route fields where needed.

## Proposed Helpers

Extend `tldw_Server_API/app/api/v1/endpoints/_pagination_utils.py` or move shared model-free helpers into a nearby utility module if schema imports would create endpoint import cycles.

Helper functions:

- `normalize_offset_pagination(...) -> OffsetPaginationRequest`
- `normalize_cursor_pagination(...) -> CursorPaginationRequest`
- `build_offset_pagination_meta(...) -> OffsetPaginationMeta`
- `build_cursor_pagination_meta(...) -> CursorPaginationMeta`
- `build_pagination_link_header(...) -> str | None`

Internal request value objects may be Pydantic models or frozen dataclasses:

- `OffsetPaginationRequest`: `limit`, `offset`, `source`, `warnings`
- `CursorPaginationRequest`: `limit`, `cursor`, `source`, `warnings`

`build_link_header(...)` should remain available for existing callers during migration. New helpers should derive metadata and Link headers from the same normalized request values.

## Bounds Contract

Do not use one global default for all routes in Phase 3.2. Each route family should pass explicit bounds.

Recommended helper inputs:

- `default_limit`
- `max_limit`
- `min_limit`, default `1`
- `default_offset`, default `0`
- `allow_zero_limit`, default `False`
- `alias_policy`, default `"compat"`

Validation rules:

- Missing `limit` uses `default_limit`.
- Values below `min_limit` raise `400` or `422`, matching the route family's current validation style.
- Values above `max_limit` are clamped only if the existing route already clamps; otherwise reject to avoid hidden behavior changes.
- Negative `offset`, `page`, `rows_offset`, or cursor-window integers are rejected.
- `page` aliases are one-based and convert to zero-based offset.

## Alias Precedence

Recommended conservative precedence:

1. Canonical `limit` and `offset`
2. Row-window aliases `rows_limit` and `rows_offset`
3. Page aliases `page` plus `per_page`
4. Page aliases `page` plus `results_per_page`
5. Route default values

Conflict handling:

- If canonical values are present, use them.
- If legacy aliases are also present and disagree, keep canonical values and record a compatibility warning in the normalized request object.
- Do not reject alias conflicts in the first migration slice unless the route already rejects them.
- Route-family tests should assert the chosen precedence before migration.

Cursor alias precedence:

1. Canonical `cursor`
2. Provider/OpenAI-style `after`
3. Route default `None`

Conflict handling:

- If both `cursor` and `after` are present and differ, use `cursor` for first-party routes.
- For OpenAI-compatible routes, use `after` unless maintainers explicitly approve canonical `cursor`.
- Record a compatibility warning when both are present and differ.

## Link Header Contract

Rules:

- Link headers must agree with serialized pagination metadata.
- Offset `rel="next"` appears only when `has_more = true` and `next_offset` is known.
- Offset `rel="prev"` appears when `offset > 0`.
- Offset `rel="first"` remains best effort and points to `offset=0`.
- Offset `rel="last"` is emitted only when `has_more = false` and total/offset semantics make it safe.
- Cursor `rel="next"` appears only when `next_cursor` is not null.
- Preserve unknown/common query parameters supplied by callers.

`build_link_header(...)` compatibility:

- Keep the current function signature working.
- New helper tests should cover both direct `build_link_header(...)` calls and metadata-driven Link-header generation.

## Route Migration Contract

For each migrated route family:

- Preserve existing request aliases.
- Preserve existing response fields during the compatibility window.
- Add canonical nested `pagination` metadata only where doing so does not conflict with the existing public contract.
- Update frontend parsers in the same PR when they consume changed response metadata.
- Avoid changing database pagination algorithms unless the route has a known correctness bug.

Recommended first offset pilot:

- `skills` if implementation wants the same first pilot as Phase 3.1/3.4.
- `slides` if the team wants richer existing pagination coverage first.

Recommended first cursor pilot:

- Defer until offset helper tests are green.
- Use `workflows` or `audio_history` only after ordering and cursor opacity are documented.

## Test Matrix

Unit tests:

- default limit and offset
- max limit rejection or clamping per configured policy
- negative offset rejection
- `limit`/`offset` canonical input
- `rows_limit`/`rows_offset` alias input
- `page`/`per_page` conversion
- `page`/`results_per_page` conversion
- canonical values win over aliases
- alias conflict warning recorded
- `has_more` true computes `next_offset`
- `has_more` false leaves `next_offset` null
- cursor input
- `after` alias input
- cursor conflict precedence
- Link header generated from offset metadata
- Link header generated from cursor metadata
- existing `build_link_header(...)` examples remain stable

Pilot tests:

- legacy `skills` pagination request still works.
- canonical `skills` pagination request works.
- response keeps legacy fields when present.
- response includes canonical metadata only after the pilot explicitly adopts it.
- UI client handles legacy and canonical metadata for the selected pilot.

## Pending Decisions

- Whether alias conflicts should become hard errors after the compatibility window.
- Whether OpenAPI should expose all aliases for migrated routes or document legacy aliases out of band.
- Whether `total` should be omitted or serialized as null when not computed.
