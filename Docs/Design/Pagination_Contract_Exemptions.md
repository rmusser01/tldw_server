# Pagination Contract Exemptions

This document records route families that should not be forced into the
canonical pagination envelope without a separate versioning or contract
decision. The generated matrix in `Docs/Design/Pagination_Completion_Matrix.md`
is the route-level inventory; this document owns the policy categories.

## Exemption Categories

### Provider-Compatible Routes

Do not alter response shapes that intentionally mirror an external provider
contract, such as OpenAI-compatible or Anthropic-compatible APIs. A provider
route can use shared pagination helpers internally only when the public payload
stays provider-compatible.

Route-level matrix status: `exempt-provider`.

### Raw List Routes

Routes returning raw `list[...]` payloads cannot receive a top-level
`pagination` object additively. Defer these until API versioning, or add a
sibling versioned route with an object envelope.

Route-level matrix status: `exempt-raw-list`.

### Streaming, File Export, and Download Routes

Streaming responses, file exports, CSV downloads, and binary downloads are not
normal list envelopes. They should be recorded as not applicable unless a
separate metadata side-channel already exists.

Route-level matrix status: `exempt-not-paginated`.

### Internal Admin and Event Routes With Unknown Totals

Admin/event routes may use canonical metadata with `total=None` when computing a
total would be expensive, provider-owned, or semantically unavailable. Prefer
cursor tokens or overfetch-derived `has_more` over new count queries.

Route-level matrix status: `migration-candidate` with count strategy
`overfetch-or-token` or `needs-confirmation`.

### Detail and Nested Subresource Routes

Parameterized detail routes under plural resources are not pagination targets
unless they expose list/search/history/job/event semantics directly. The
inventory script should mark uncertain cases as `needs-confirmation`; do not
promote them into migration scope without route tests.

Route-level matrix status: `needs-confirmation` or `exempt-not-paginated`.

## Review Rules

- Exemptions must be explicit and narrow; avoid wildcard exemptions in tests.
- Every exempt route needs a reason that is stable enough for future reviewers.
- If a route later gets canonical pagination, remove or update its exemption
  entry in the same PR.
- Raw-list and provider-compatible routes are versioning decisions, not helper
  cleanup tasks.
