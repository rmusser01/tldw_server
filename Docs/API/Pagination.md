# API Pagination

This document describes the canonical pagination metadata used by API v1 list,
search, history, runs, jobs, and collection endpoints.

The migration is additive. Existing top-level fields such as `total`, `count`,
`limit`, `offset`, `page`, `page_size`, `per_page`, `has_more`, and
`next_offset` remain available on migrated endpoints for legacy clients. New
clients should prefer the nested `pagination` object when it is present.

Canonical backend schema names are `OffsetPaginationMeta`,
`PagePaginationMeta`, and `CursorPaginationMeta`. Frontend clients expose the
matching `OffsetPaginationMeta`, `PagePaginationMeta`, `CursorPaginationMeta`,
and `ApiPaginationMeta` TypeScript types from `response-envelope.ts`.

When a route opts into the canonical response envelope, the nested envelope
location is `metadata.pagination`. For default legacy-shaped `v1` route bodies,
the additive nested field remains `pagination`.

## Offset Pagination

Offset-based endpoints use `limit` and `offset` request parameters and return
`pagination.mode == "offset"`.

```json
{
  "items": [],
  "total": 42,
  "limit": 25,
  "offset": 0,
  "has_more": true,
  "next_offset": 25,
  "pagination": {
    "mode": "offset",
    "limit": 25,
    "offset": 0,
    "total": 42,
    "has_more": true,
    "next_offset": 25
  }
}
```

Fields:

- `limit`: maximum number of items requested for the current window.
- `offset`: zero-based item offset for the current window.
- `total`: total matching item count when known, otherwise `null`.
- `has_more`: whether another window is available.
- `next_offset`: the next offset to request, or `null` when exhausted.

## Page Pagination

Page-based endpoints use `page` plus `per_page`, `page_size`, `size`, or a
route-specific page-size alias. Their canonical response uses
`pagination.mode == "page"`.

```json
{
  "items": [],
  "page": 2,
  "per_page": 25,
  "total": 60,
  "total_pages": 3,
  "has_more": true,
  "pagination": {
    "mode": "page",
    "page": 2,
    "per_page": 25,
    "total": 60,
    "total_pages": 3,
    "has_more": true
  }
}
```

Fields:

- `page`: one-based page number.
- `per_page`: canonical page size for the current route.
- `total`: total matching item count when known, otherwise `null`.
- `total_pages`: total page count when known, otherwise `null`.
- `has_more`: whether a later page is available.

## Cursor Pagination

Cursor-based endpoints use opaque cursors or provider tokens. Their canonical
response uses `pagination.mode == "cursor"`.

```json
{
  "items": [],
  "next_cursor": "opaque-token",
  "has_more": true,
  "pagination": {
    "mode": "cursor",
    "limit": 100,
    "cursor": null,
    "next_cursor": "opaque-token",
    "has_more": true
  }
}
```

Fields:

- `limit`: maximum number of items requested for the current window.
- `cursor`: input cursor used for the current request, or `null`.
- `next_cursor`: opaque cursor for the next request, or `null` when exhausted.
- `has_more`: whether another cursor request is available.

## Unknown Totals

Some endpoints cannot compute a total cheaply or do not own the upstream count.
Those endpoints should set `pagination.total` or `pagination.total_pages` to
`null` and rely on `has_more`, `next_offset`, or `next_cursor` for continuation.
Clients must treat `null` totals as "unknown", not zero.

## Compatibility Rules

- Preserve legacy top-level pagination aliases on migrated endpoints.
- Prefer `pagination` for new client code when it exists.
- Prefer `metadata.pagination` when consuming canonical envelope responses.
- Do not infer a missing `pagination` object means more data exists.
- Do not alter provider-compatible payloads only to add canonical metadata.
- Do not add response-body pagination to raw-list endpoints without a versioned
  object-envelope route.
- Do not change the default body shape of a `v1` route merely to move
  pagination into an envelope; use a sibling route or `/api/v2/` for
  default-breaking migrations.
- Keep streaming, file export, and binary download continuation in their stream,
  file, header, or query contract rather than inventing a JSON pagination body.

## Exemptions

The canonical matrix is tracked in
`Docs/Design/Pagination_Completion_Matrix.md`. Explicit exemption policy lives
in `Docs/Design/Pagination_Contract_Exemptions.md`.

Current exemption categories:

- Provider-compatible routes: preserve external provider response contracts.
- Raw list routes: defer to API versioning or sibling object-envelope routes.
- Streaming, file-export, and download routes: no JSON list envelope.
- Operation results and aggregate counts: not list pagination.
- Detail and nested subresource routes: only paginate if they expose direct
  list/search/history/job/event semantics.

See [API Versioning Strategy](api-versioning-strategy.md)
for the rule that default-breaking shape changes should move to a sibling route
or `/api/v2/`.
