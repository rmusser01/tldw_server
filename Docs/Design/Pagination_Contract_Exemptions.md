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

Current pagination-sensitive examples:

- `GET /jobs/list`: raw `list[JobItem]`; keep the existing payload shape unless
  a versioned/object-envelope route is added.
- `GET /api/v1/workflows/runs/{run_id}/events`: raw `list[EventResponse]` with
  `Next-Cursor` and `Link` headers; do not move cursor metadata into the body
  without a versioned route.

### Streaming, File Export, and Download Routes

Streaming responses, file exports, CSV downloads, and binary downloads are not
normal list envelopes. They should be recorded as not applicable unless a
separate metadata side-channel already exists.

Route-level matrix status: `exempt-not-paginated`.

Current pagination-sensitive examples:

- `GET /jobs/events/stream`: SSE event stream using `after_id`; keep pagination
  semantics in the stream cursor/header channel rather than adding a response
  envelope.
- `GET /mcp/hub/events/stream`: SSE replay/live stream. The optional `limit`
  controls stream termination, not a response-body page contract.
- `GET /watchlists/runs/export.csv`: CSV export; keep selection/filter inputs
  separate from response-body pagination metadata.

### Operation Results and Aggregate Counts

Object envelopes that report action results, imports, bulk-create outcomes, or
aggregate counters are not list pagination targets just because they contain a
field named `total`. Classify these separately unless they expose bounded
`items` plus page/cursor/limit inputs.

Route-level matrix status: `exempt-not-paginated`.

Current examples:

- `POST /watchlists/sources/bulk`: bulk-create operation result.
- `POST /watchlists/sources/check-now`: ad hoc source-check operation result.
- `POST /watchlists/sources/import`: OPML import operation result.
- `GET /watchlists/items/smart-counts`: aggregate counts for UI filters.
- `GET /mcp/hub/audit/findings`: generated governance audit snapshot with
  filters but no page, limit, or offset inputs; `total` and `counts` summarize
  the returned snapshot.
- `POST /api/v1/kanban/checklists/{checklist_id}/toggle-all`: checklist
  mutation result; `total_items` is the resulting checklist size.

### Bounded Preview Routes

Preview/test routes with legacy `items` and `total` fields may add canonical
metadata when they already expose a bounded list input such as `limit`. If the
route has no continuation input, the metadata should represent the single
returned window (`offset=0`, `has_more=false`) rather than inventing a new page
contract.

Current examples:

- `POST /watchlists/jobs/{job_id}/preview`
- `POST /watchlists/sources/{source_id}/test`
- `POST /watchlists/sources/test`

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

Current examples:

- `GET /chatbooks/export/jobs/{job_id}` and
  `GET /chatbooks/import/jobs/{job_id}`: job detail responses; `total_items`
  is progress metadata, not a collection total.
- `GET /sandbox/runs/{run_id}/artifacts` and
  `GET /sandbox/sessions/{session_id}/snapshots`: bounded subresource lists
  without pagination inputs.
- `GET /sandbox/runs/{run_id}/{rest:path}`: path guard/fallback route, not a
  collection response.
- `GET /api/v1/kanban/checklists/{checklist_id}` and
  `GET /api/v1/kanban/lists/{list_id}`: detail responses.
- `GET /api/v1/kanban/checklists/{checklist_id}/items`,
  `GET /api/v1/kanban/cards/{card_id}/checklists`,
  `GET /api/v1/kanban/boards/{board_id}/labels`,
  `GET /api/v1/kanban/workflow/boards/{board_id}/statuses`, and
  `GET /api/v1/kanban/workflow/boards/{board_id}/transitions`: small nested
  subresource/catalog responses without pagination inputs.
- `GET /api/v1/kanban/search/status`: status/capability response, not a search
  result page.
- `GET /api/v1/prompt-studio/optimizations/history/{optimization_id}`:
  bounded progress/history snapshot for one optimization with no continuation
  input.
- `GET /api/v1/prompt-studio/prompts/history/{prompt_id}`: bounded version
  history snapshot for one prompt with no pagination inputs.

## Review Rules

- Exemptions must be explicit and narrow; avoid wildcard exemptions in tests.
- Every exempt route needs a reason that is stable enough for future reviewers.
- If a route later gets canonical pagination, remove or update its exemption
  entry in the same PR.
- Raw-list and provider-compatible routes are versioning decisions, not helper
  cleanup tasks.
