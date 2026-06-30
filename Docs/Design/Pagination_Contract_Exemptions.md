# Pagination Contract Exemptions

This document records route families that should not be forced into the
canonical pagination envelope without a separate versioning or contract
decision. The generated matrix in `Docs/Design/Pagination_Completion_Matrix.md`
is the route-level inventory; this document owns the policy categories.

As of the pagination-completion closeout, the matrix should have no
`migration-candidate` or `needs-confirmation` rows. Remaining noncanonical
routes should be explicitly classified as `exempt-provider`,
`exempt-raw-list`, or `exempt-not-paginated`.

## Exemption Categories

### Provider-Compatible Routes

Do not alter response shapes that intentionally mirror an external provider
contract, such as OpenAI-compatible or Anthropic-compatible APIs. A provider
route can use shared pagination helpers internally only when the public payload
stays provider-compatible.

Route-level matrix status: `exempt-provider`.

Current pagination-sensitive examples:

- `GET /api/v1/paper-search/biorxiv/raw/*` and
  `GET /api/v1/paper-search/medrxiv/raw/*`: provider raw passthroughs using
  provider cursor semantics in the upstream response body.
- `GET /api/v1/paper-search/osf/raw`: provider raw passthrough using OSF query
  parameters such as `page[size]`; preserve the provider payload.
- `GET /api/v1/paper-search/pmc-oai/list-identifiers`,
  `GET /api/v1/paper-search/pmc-oai/list-records`, and
  `GET /api/v1/paper-search/pmc-oai/list-sets`: OAI-PMH resumption-token
  contract; do not replace provider continuation metadata with the canonical
  response envelope without a versioned route.

### Raw List Routes

Routes returning raw `list[...]` payloads cannot receive a top-level
`pagination` object additively. Defer these until API versioning, or add a
sibling versioned route with an object envelope.

Route-level matrix status: `exempt-raw-list`.

Current pagination-sensitive examples:

- `GET /jobs/list`: raw `list[JobItem]`; keep the existing payload shape unless
  a versioned/object-envelope route is added.
- `GET /jobs/sla/policies` and `GET /jobs/sla/breaches`: raw admin snapshot
  lists without continuation inputs; adding body pagination would require a
  versioned/object-envelope route.
- `GET /api/v1/workflows/runs/{run_id}/events`: raw `list[EventResponse]` with
  `Next-Cursor` and `Link` headers; do not move cursor metadata into the body
  without a versioned route.
- `GET /api/v1/workflows/step-types`,
  `GET /api/v1/workflows/templates`, and
  `GET /api/v1/workflows/templates/tags`: bounded workflow catalog/introspection
  lists returned as raw arrays.

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
- `GET /research/runs/{session_id}/events/stream`: SSE event stream using
  `after_id`; keep continuation in the stream cursor channel.
- `GET /api/v1/media/ingest/jobs/events/stream`: SSE event stream using
  `after_id`; keep continuation in the stream cursor channel.
- `GET /api/v1/audio/jobs/{job_id}/progress/stream`: SSE event stream using
  `after_id`; keep continuation in the stream cursor channel.
- `GET /api/v1/notifications/stream`: SSE event stream using `after` and
  `Last-Event-ID`; keep continuation in the stream cursor channel.
- `GET /reading/export`: streamed file export. The `page` input selects which
  rows to include in the download; the response body is the exported file, not a
  JSON list envelope.
- `GET /api/v1/admin/llm-usage/export.csv`,
  `GET /api/v1/admin/usage/daily/export.csv`, and
  `GET /api/v1/admin/usage/top/export.csv`: CSV downloads whose `limit` input
  bounds exported rows rather than describing a JSON response-body page.
- `GET /api/v1/admin/audit-log/export` and
  `GET /api/v1/admin/users/export`: file-style export endpoints that can return
  CSV or JSON content directly; keep selection inputs separate from canonical
  list-envelope metadata.
- `GET /api/v1/notes/export.csv`: CSV note export; keep `limit` and `offset`
  as export-window selection inputs, not response-body pagination metadata.
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
- `GET /api/v1/paper-search/biorxiv/reports/summary` and
  `GET /api/v1/paper-search/biorxiv/reports/usage`: aggregate provider
  reports, not paginated result sets.
- `GET /api/v1/audio/jobs/admin/summary` and
  `GET /api/v1/audio/jobs/admin/summary-by-owner`: aggregate job-count
  summaries, not job list pages.
- `POST /api/v1/notes/export`: explicit note-ID selection export with no
  continuation inputs; do not invent a response-body page contract for the
  selected set.

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

Route-level matrix status: `canonical-present-or-custom-pagination` when the
route owns a list envelope and exposes canonical metadata with an unknown total;
otherwise use the narrow `exempt-not-paginated` classification that matches the
route's stream, operation-result, aggregate, or detail semantics.

### Detail and Nested Subresource Routes

Parameterized detail routes under plural resources are not pagination targets
unless they expose list/search/history/job/event semantics directly. The
inventory script may surface uncertain cases during future scans, but closeout
requires converting those rows to a canonical route or a narrow exemption before
merge. Do not promote detail routes into migration scope without route tests.

Route-level matrix status: `exempt-not-paginated` unless the route is a raw list
that requires `exempt-raw-list` or a provider-shaped response that requires
`exempt-provider`.

Current examples:

- `GET /chatbooks/export/jobs/{job_id}` and
  `GET /chatbooks/import/jobs/{job_id}`: job detail responses; `total_items`
  is progress metadata, not a collection total.
- `GET /jobs/archive/meta`: archive metadata detail for one job id.
- `GET /jobs/queue/status`: queue control status for one domain/queue pair.
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
- Paper search `by-id` and `by-doi` routes, such as
  `GET /api/v1/paper-search/arxiv/by-id` and
  `GET /api/v1/paper-search/wiley/by-doi`: single-record provider/detail
  lookups, not collection pages.
- `GET /api/v1/media/{media_id}/annotations` and
  `GET /api/v1/media/{media_id}/figures`: bounded document artifact snapshots
  without pagination inputs.
- `GET /api/v1/media/{media_id}/outline`: document structure response;
  `total_pages` is document metadata, not a pagination contract.
- `GET` and `PUT /api/v1/media/{media_id}/progress`: reading progress state;
  `total_pages` is reading-position metadata, not collection pagination.
- `GET /api/v1/media/keywords`: bounded keyword suggestion list with a limit
  cap but no continuation input.
- `GET /api/v1/audio/history/{history_id}`: single TTS history detail response.
- `GET /api/v1/audio/providers` and `GET /api/v1/audio/voices/catalog`:
  provider capability/catalog responses without pagination inputs.
- `GET /api/v1/audio/voices`: bounded custom voice catalog for a user with no
  pagination inputs.
- `GET /api/v1/connectors/sources/{source_id}/sync`: sync-state detail
  response; `cursor` is provider sync checkpoint state, not response
  pagination.

## Review Rules

- Exemptions must be explicit and narrow; avoid wildcard exemptions in tests.
- Every exempt route needs a reason that is stable enough for future reviewers.
- If a route later gets canonical pagination, remove or update its exemption
  entry in the same PR.
- Raw-list and provider-compatible routes are versioning decisions, not helper
  cleanup tasks.
