# Phase 4.5 API Versioning And Phase 3 Alignment

**Date:** 2026-04-25

**Status:** Planning artifact complete; versioning policy decision pending.

## Purpose

Align the existing API versioning strategy with the Phase 3 response-envelope and pagination work before any runtime defaults change. This prevents Phase 3 from accidentally making breaking `v1` contract changes that should belong to a future `v2`.

## Current Versioning Contract

Source: `Docs/API/api-versioning-strategy.md`.

Current policy:

- All current endpoints are under `/api/v1/`.
- Removing or renaming response fields requires a major version bump.
- Changing error response format or status codes requires a major version bump.
- Adding optional request fields and adding response fields are non-breaking.
- Deprecated endpoints continue during a warning period and return `Deprecation` and optional `Sunset` headers.
- When `v2` exists, `v1` and `v2` share database and auth systems, with versioning handled at endpoint/schema boundaries.

## Phase 3 Impact

### Phase 3.1 Response Envelope

Breaking if applied by default in `v1`:

- Wrapping existing response bodies under `data`.
- Moving error bodies from `{"detail": ...}` to `{"success": false, "error": ...}`.
- Removing or renaming existing top-level fields.

Safe in `v1` when opt-in:

- Header opt-in `X-TLDW-Response-Envelope: v1`.
- Optional manual query opt-in if maintainers approve it.
- Keeping legacy payloads as default.
- Adding envelope schemas and helpers without endpoint behavior changes.
- Adding standard error envelopes only for opt-in requests.

Recommendation:

- Keep `v1` default response shapes unchanged.
- Treat the standard envelope as an opt-in compatibility feature in `v1`.
- Make the standard envelope the default only in a future `v2` or after an explicit versioning decision.

### Phase 3.2 Pagination

Breaking if applied by default in `v1`:

- Removing `page`, `per_page`, `results_per_page`, `count`, `total`, `limit`, `offset`, or existing cursor fields from route bodies.
- Rejecting legacy aliases that previously worked.
- Changing provider-compatible cursor semantics.

Safe in `v1` when compatible:

- Adding canonical `pagination` metadata while preserving existing fields.
- Accepting `limit`/`offset` alongside existing page aliases.
- Keeping provider-compatible routes shaped like the provider contract.
- Deriving `Link` headers from the same metadata where existing response behavior remains compatible.

Recommendation:

- Keep legacy pagination fields during `v1`.
- Put canonical pagination metadata in `metadata.pagination` when Phase 3.1 envelope opt-in is active, or in an additive nested `pagination` field for legacy-shaped `v1` responses after route-family approval.
- Reserve legacy alias removal for `v2`.

### Phase 3.4 Auth Dependencies

Breaking if applied by default in `v1`:

- Changing authentication requirements for an endpoint.
- Changing missing/invalid auth status codes or headers.
- Changing TEST_MODE override behavior.
- Changing `require_token_scope(...)` return behavior in ways that affect route dependencies.

Safe in `v1`:

- Adding dependency aliases that preserve behavior.
- Migrating endpoint internals without changing public status codes or headers.
- Keeping `get_current_user`, `get_current_active_user`, and `get_request_user` compatibility shims.

Recommendation:

- Treat Phase 3.4 as internal cleanup only.
- Require route-family status-code regression tests before any endpoint migration.

## Proposed Versioning Decision

For Phase 3:

- `v1` remains legacy-default.
- Standard envelope and canonical pagination metadata are opt-in.
- Header opt-in inside `v1` is transitional and does not replace path-based major versioning.
- Exemptions remain explicit:
  - streaming
  - file downloads
  - `204 No Content`
  - webhooks
  - WebSocket messages
- OpenAI-compatible and provider-compatible payloads unless route-specific approval exists
- Deprecation headers are not emitted for legacy response shapes during the first pilot.
- Shared client/service layers may own opt-in and unwrap behavior, while UI/domain consumers remain legacy-shaped by default.

Draft maintainer decision packet:

- `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-api-versioning-policy-decision-packet.md`

For future `v2`:

- Standard success/error envelope can become default for first-party JSON routes.
- Canonical `pagination` metadata can become the default list metadata.
- Legacy page aliases can be removed or documented as compatibility-only.
- Provider-compatible surfaces should either stay provider-shaped or move under provider-specific versioning rules.

## Required Migration Guide Shape

If maintainers decide to make a `v2`, create:

- `Docs/API/migrations/v1-to-v2.md`

Minimum sections:

- response envelope before/after
- error response before/after
- pagination before/after
- route exemptions
- provider-compatible routes
- client opt-in examples
- deprecation timeline

## Proposed Deprecation Headers

Do not add deprecation headers to all legacy `v1` responses during the pilot.

Only add headers when maintainers approve a deprecation window:

```http
Deprecation: true
Sunset: Sat, 01 Jan 2028 00:00:00 GMT
Link: <https://docs.tldw.example.com/migration/v2>; rel="successor-version"
```

## Open Questions

- Should `response_envelope=v1` ship publicly or remain test-only?
- Should `v2` be path-based only, or can media-type/header versioning be used for envelope defaults?
- Should standard envelopes appear in `v1` OpenAPI as alternate responses, or only in future `v2` OpenAPI?
- What is the earliest acceptable deprecation window for legacy first-party JSON response shapes?
- Should provider-compatible endpoints be permanently exempt from standard envelopes?

## Handoff Checklist

- [ ] Maintainers confirm `v1` remains legacy-default.
- [ ] Maintainers confirm whether Phase 3 opt-in query flag is public or test-only.
- [ ] Frontend owner confirms opt-in client strategy.
- [ ] OpenAPI owner confirms how opt-in response variants should be documented.
- [ ] If `v2` is pulled forward, create `Docs/API/migrations/v1-to-v2.md` before runtime changes.
