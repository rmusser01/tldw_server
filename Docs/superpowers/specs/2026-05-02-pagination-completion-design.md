# Pagination Completion Design

## Goal

Finish API v1 pagination standardization after the merged Phase 3.2 offset-pilot work in #1159, without breaking existing clients or collapsing the remaining work into one unreviewable PR.

The completed effort should make every list/search-style route fall into one explicit state:

- It returns canonical nested pagination metadata.
- It is intentionally provider-shaped or raw-list-shaped and documented as exempt.
- It is deferred behind API versioning because adding metadata would be a response-shape change.

## Current Baseline

#1159 established the durable pattern:

- Canonical pagination schemas live in `tldw_Server_API/app/api/v1/schemas/pagination.py`.
- Shared metadata builders live in `tldw_Server_API/app/api/v1/utils/pagination.py`.
- Endpoint-local `_pagination_utils.py` remains as a compatibility/link-header seam.
- Migrated offset/list endpoints preserve legacy top-level fields while adding nested `pagination`.
- Review fixes already covered alias drift, missing totals, page metadata `has_more`, and service-to-endpoint dependency inversion.

That is a strong foundation, but it is not complete coverage. The remaining work should be organized by pagination model, not by whichever endpoint happens to be next.

## Design Principles

1. Preserve response compatibility.
   - Existing top-level fields stay in migrated responses.
   - Raw `list[...]` responses are not wrapped unless the route is versioned or explicitly approved for a breaking change.

2. Prefer canonical metadata over bespoke aliases.
   - Offset metadata uses `OffsetPaginationMeta`.
   - Cursor metadata uses `CursorPaginationMeta`.
   - Page-number metadata uses `PagePaginationMeta`.

3. Use `total=None` when totals are unavailable or expensive.
   - Do not add slow count queries just to fill `total`.
   - Use overfetch (`limit + 1`) for `has_more` when that is cheaper and semantically correct.

4. Require coverage before migrations.
   - Each endpoint-family PR adds red/green tests for the old payload shape plus the new canonical `pagination` object.
   - Count seams need direct unit tests where possible.

5. Keep PRs independently mergeable.
   - One model family or tightly related route family per PR.
   - Do not mix page, cursor, and raw-envelope changes in the same tranche unless the shared helper change requires it.

## Contract Families

### Offset

Offset endpoints use `limit` and `offset`, usually with top-level `total`, `has_more`, and `next_offset` aliases.

Target canonical shape:

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

Remaining offset work is mostly cleanup and coverage: duplicated alias validators, custom envelopes, and route families that were intentionally skipped during #1159.

### Page / Per-Page

Page-number endpoints use `page`, `per_page`, `page_size`, `results_per_page`, `total_pages`, or similar fields.

Target canonical shape:

```json
{
  "pagination": {
    "mode": "page",
    "page": 1,
    "per_page": 25,
    "total": 120,
    "total_pages": 5,
    "has_more": true
  }
}
```

The page family should be handled before cursor/custom envelopes because `PagePaginationMeta` already exists and #1159 hardened `build_page_pagination_meta`.

### Cursor

Cursor endpoints use opaque cursors, `after`, `after_id`, `next_cursor`, or provider-specific token fields.

Target canonical shape:

```json
{
  "pagination": {
    "mode": "cursor",
    "limit": 50,
    "cursor": "input-cursor-or-null",
    "next_cursor": "next-token-or-null",
    "has_more": true
  }
}
```

Cursor migration must be careful about token opacity, stable sort order, invalid cursor handling, and whether existing `Link` headers already encode navigation.

### Custom Legacy Envelopes

Some endpoints use bespoke envelopes that are not clean offset/page/cursor responses. Examples include Kanban, watchlists, sandbox, MCP/admin audit lists, voice/persona lists, chatbooks/jobs, and several raw list returns.

These should not be forced into one pattern blindly. The plan should first classify each route as:

- Additive migration possible now.
- Needs a cheap count or overfetch seam first.
- Must remain exempt until API versioning.

## Architecture

### Inventory Matrix

Create and maintain one route matrix that records:

- Route path and method.
- Endpoint file/function.
- Response model.
- Query pagination params.
- Legacy response pagination fields.
- Canonical `pagination` presence.
- Pagination family: `offset`, `page`, `cursor`, `custom`, `provider`, `raw-list`, or `not-paginated`.
- Count strategy: known total, overfetch, unknown total, provider total, or not applicable.
- Test file coverage.
- Migration status and exemption reason.

The matrix is the program control surface. It prevents repeated rediscovery and gives reviewers a concrete definition of “done.”

The inventory should distinguish confirmed route metadata from inferred metadata. If a route path, method, response model, or pagination family is inferred from static analysis rather than OpenAPI/app route metadata, mark that field as `unknown` or `needs-confirmation` instead of guessing.

### Shared Helpers

Keep helper ownership split:

- `schemas/pagination.py` owns schema models and schema-level alias/default helpers.
- `utils/pagination.py` owns metadata builders.
- `endpoints/_pagination_utils.py` owns endpoint-specific link-header compatibility.

The next helper cleanup should reduce duplicated `_default_offset_pagination_aliases` functions across schema modules and add explicit page/cursor alias default helpers where useful.

### Endpoint Migrations

Each endpoint-family migration should follow this sequence:

1. Add tests proving the current legacy response shape.
2. Add tests expecting canonical nested `pagination`.
3. Add or harden count/overfetch seams only if needed.
4. Use shared builders to populate metadata.
5. Re-run focused endpoint tests and touched-file tests.
6. Run Bandit on touched source paths.

### Frontend Typing

Frontend changes should stay parser-compatible:

- Extend existing response-envelope and unwrap helpers to understand `OffsetPaginationMeta`, `PagePaginationMeta`, and `CursorPaginationMeta`.
- Do not require every route to migrate before frontend helpers can consume canonical metadata.
- Add route-family client tests only when a frontend call path consumes the migrated response.

## Stage Plan

### Stage 0 — Inventory and Exemption Matrix

Build the matrix first. This is a docs/tooling PR with no endpoint behavior changes.

Acceptance criteria:

- Matrix lists all known list/search endpoints and response models.
- Each endpoint has a family, migration status, and exemption/defer reason if not in scope.
- The matrix identifies the first page, cursor, and custom-envelope candidates.
- Unknown or inferred route metadata is explicitly marked for follow-up.

### Stage 1 — Shared Helper Consolidation

Consolidate duplicated schema alias helper code and add guard tests that prevent alias drift.

Acceptance criteria:

- Shared offset alias helper is reused by new migrations.
- Page/cursor alias helpers are available where response schemas need legacy aliases.
- Existing migrated endpoints remain unchanged at the wire level.

### Stage 2 — Page / Per-Page Migration

Migrate page-family endpoints in small PRs:

- Prompt Studio page-list gaps first, without churning routes that already return `PageListResponse`.
- Paper/research page searches next.
- Media/navigation/version-style page lists after coverage is clear.
- Privileges/collections feeds as a final page tranche.

Acceptance criteria:

- Page-family responses preserve legacy fields.
- Nested `pagination.mode == "page"` is present where response models are not raw/provider exempt.
- Tests cover `has_more`, `total_pages`, and legacy aliases.
- Route-specific compatibility aliases, such as top-level `projects`, remain present unless a versioning decision explicitly removes them.

### Stage 3 — Cursor Migration

Migrate cursor-family endpoints:

- TTS/audio history and audio jobs first.
- Workflows run/event cursor paths next.
- Notifications/sync/jobs-admin cursor paths after cursor behavior is confirmed.

Acceptance criteria:

- Cursor tokens remain opaque.
- Invalid cursor paths keep existing error semantics.
- `next_cursor` and `has_more` agree.
- Link headers, when present, agree with canonical metadata.

### Stage 4 — Custom Legacy Envelopes

Handle custom envelopes by domain, but only after the shared/page/cursor helpers are stable.

Acceptance criteria:

- Each custom route family is either migrated additively or explicitly exempted.
- Raw list endpoints are not wrapped without a versioning decision.
- Large modules are split into small PRs with focused tests.

### Stage 5 — Contract Guardrails

Add OpenAPI/test guardrails so new list endpoints cannot silently invent a fourth shape.

Acceptance criteria:

- A test inspects OpenAPI/route metadata and enforces that list/search routes are canonical, exempt, or not paginated.
- The exemption list is explicit and reviewable.
- The matrix and guard test stay in sync.

### Stage 6 — Frontend and Documentation Closeout

Update frontend pagination typing and publish the contract.

Acceptance criteria:

- Frontend helpers can consume offset, page, and cursor metadata.
- API docs describe canonical metadata and legacy compatibility.
- Issue #1116 or its successor has an accurate completion summary.

## Risks and Mitigations

- Risk: Expensive totals cause performance regressions.
  - Mitigation: Allow `total=None` and use overfetch for `has_more`.

- Risk: Provider-compatible endpoints accidentally change public shape.
  - Mitigation: Mark provider-shaped endpoints explicitly and only add internal helper usage when payloads remain unchanged.

- Risk: Raw list endpoints become breaking changes.
  - Mitigation: Exempt raw-list routes until API versioning or add a sibling versioned route.

- Risk: Tightening FastAPI `response_model` filters out legacy compatibility fields.
  - Mitigation: Preserve existing response-model behavior unless a route-specific schema carries those fields explicitly.

- Risk: The matrix drifts from code.
  - Mitigation: Add a lightweight route/OpenAPI contract test in Stage 5.

- Risk: A tranche starts from stale `dev` and reintroduces conflicts or already-fixed pagination behavior.
  - Mitigation: Rebase or recreate each implementation tranche from latest `origin/dev` before source edits.

- Risk: PRs become too large.
  - Mitigation: Split by model family and then by route family; every PR should have focused tests and a small migration list.

## Definition of Done

Pagination completion is done when:

- Every list/search endpoint is classified.
- Every non-exempt paginated endpoint returns canonical nested metadata.
- Legacy fields remain covered where preserved.
- Cursor and page metadata have the same contract discipline as offset metadata.
- OpenAPI/test guardrails prevent new ad hoc list envelopes.
- Remaining exemptions are explicit, justified, and tied to versioning or provider compatibility.
