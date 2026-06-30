# Phase 4.5 API Versioning Policy Decision Packet

**Date:** 2026-04-25

**Status:** Draft decision packet complete; maintainer approval pending.

## Purpose

Turn the Phase 4.5 versioning analysis into explicit policy decisions that maintainers can accept, reject, or amend before Phase 3 response-envelope and pagination helpers become runtime behavior.

## Recommended Decisions

### Decision 1: Keep `v1` Legacy-Default

Recommendation:

- Keep current `/api/v1/` response shapes, error shapes, pagination aliases, and auth status behavior as default.
- Do not emit deprecation headers for legacy `v1` response shapes during the first Phase 3 pilot.

Reason:

- `Docs/API/api-versioning-strategy.md` treats response-field removal, error-format changes, and auth requirement changes as major-version changes.
- Making envelopes or canonical pagination default in `v1` would be a breaking client contract change.

### Decision 2: Standard Envelope Is Opt-In In `v1`

Recommendation:

- Public opt-in mechanism: `X-TLDW-Response-Envelope: v1`.
- Query opt-in, if implemented, should be test-only or internal until maintainers approve it as public API.
- This header is transitional inside `v1`; it does not replace path-based major versioning.
- Provider-compatible, streaming, file, webhook, WebSocket, and `204 No Content` routes stay exempt unless explicitly approved route-by-route.

Reason:

- Header opt-in keeps existing URLs and clients stable.
- Query opt-in is easy to discover accidentally and can become public API before the contract is ready.

### Decision 3: Canonical Pagination Must Preserve Legacy Fields

Recommendation:

- Keep existing `page`, `per_page`, `results_per_page`, `count`, `total`, `limit`, `offset`, and cursor aliases where routes already expose them.
- Add canonical pagination metadata as `metadata.pagination` for canonical envelope responses during the pilot.
- Add non-envelope nested pagination metadata only after route-family approval.

Reason:

- Removing or rejecting existing aliases in `v1` would break clients.
- Phase 3.2 can still prove helper behavior without forcing default response-shape changes.

### Decision 4: Auth Standardization Is Internal In `v1`

Recommendation:

- Phase 3.4 may add dependency aliases and internal helpers, but endpoint auth requirements, missing-auth status codes, invalid-auth status codes, and TEST_MODE behavior must remain unchanged.
- Require route-family status-code regression tests before migrating each dependency family.

Reason:

- `Docs/API/api-versioning-strategy.md` treats auth requirement changes as breaking.
- Auth helper cleanup is valuable only if it is behavior-preserving.

### Decision 5: Future `v2` Is Path-Based By Default

Recommendation:

- If standard envelopes and canonical pagination become defaults, put those defaults under `/api/v2/`.
- Do not use media-type or header versioning as the primary major-version mechanism unless maintainers explicitly update `Docs/API/api-versioning-strategy.md`.

Reason:

- Current project docs define path-based versioning.
- Path-based versioning keeps Swagger/OpenAPI and client code generation simpler.

### Decision 6: Frontend And Client Boundaries Stay Domain-Shaped By Default

Recommendation:

- Shared client/service layers may send opt-in headers and unwrap canonical
  envelopes or transitional wrappers.
- UI/domain consumers should continue receiving stable domain-shaped data by
  default instead of raw transport envelopes.
- Route-family migration work is not complete until backend docs, OpenAPI
  treatment, and client unwrap/typing behavior are all defined together.

Reason:

- This preserves the Phase 3 boundary that transport concerns live in shared
  client layers, not scattered UI components.
- It avoids accidental frontend lock-in to temporary transport shapes.

## Proposed Policy Text

Suggested addition to `Docs/API/api-versioning-strategy.md` after maintainer approval:

```markdown
## Phase 3 Compatibility Policy

During the Phase 3 API contract cleanup, `/api/v1/` remains legacy-default.
Standard response envelopes and canonical pagination metadata are opt-in for
first-party JSON routes unless a route family explicitly documents additive
default behavior. Existing `v1` response fields, error shapes, pagination
aliases, auth requirements, and status codes must remain compatible.

Provider-compatible, streaming, file download, webhook, WebSocket, and
`204 No Content` routes are exempt from standard envelopes unless a route
family explicitly opts in.
```

## Migration Guide Trigger

Create `Docs/API/migrations/v1-to-v2.md` only when maintainers approve one of these changes:

- standard success/error envelope becomes default
- canonical pagination metadata becomes default and legacy aliases become compatibility-only
- any legacy top-level response fields are removed or renamed
- error response shape changes by default
- auth requirement or status behavior changes

Minimum migration guide sections:

- summary
- response envelope before/after
- error response before/after
- pagination before/after
- route exemptions
- provider-compatible routes
- client opt-in examples
- deprecation timeline

## OpenAPI Decision

Recommendation:

- For `v1`, document legacy default schemas as the primary OpenAPI response.
- Represent opt-in envelope behavior either as explicit alternate responses or as route documentation, after OpenAPI owner decision.
- For future `v2`, make envelope schemas primary for first-party JSON routes.

Open question:

- Should opt-in `v1` envelope variants appear directly in generated OpenAPI, or only in docs until the helper schema names stabilize?

## Owner Review Questions

- Do maintainers accept `v1` legacy-default during Phase 3?
- Is `X-TLDW-Response-Envelope: v1` the public opt-in mechanism?
- Is query opt-in test-only during the pilot?
- Are provider-compatible routes permanently exempt?
- Should a future `v2` be planned now, or deferred until a route family has an approved default-breaking migration?
- Do maintainers accept that client/service layers own opt-in and unwrap behavior while domain consumers remain legacy-shaped by default?

## Handoff Checklist

- [ ] Maintainers accept or amend the five recommended decisions.
- [ ] Maintainers accept or amend the frontend/client boundary decision.
- [ ] `Docs/API/api-versioning-strategy.md` is updated only after acceptance.
- [ ] OpenAPI owner decides how opt-in envelope variants should be documented.
- [ ] Frontend owner confirms whether any client should send the opt-in header during the pilot.
- [ ] Migration guide is created only if a `v2` default change is pulled forward.
