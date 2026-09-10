# /api/v1/slides (Presentations)

Purpose: create, generate, search, and export presentations stored per user. The API supports existing structured slide decks and an opt-in `standalone_html` representation stored as untrusted executable text.

## Authentication & Rate Limits
- Auth: API key or JWT (same as other v1 endpoints).
- Rate limits: `rbac_rate_limit("slides.*")` variants per route.
- Mutations require `If-Match` with the latest ETag.

## Data Model (core fields)
- `title` (str, required)
- `description` (str, optional)
- `theme` (Reveal.js theme, default `black`)
- `marp_theme` (optional override for Markdown export; e.g., `default`, `gaia`, `uncover`)
- `settings` (Reveal.js settings allowlist)
- `slides` (array of Slide objects)
- `custom_css` (optional; sanitized)
- `source_type` / `source_ref` / `source_query` (generation provenance)
- `version` (int; used for optimistic locking)

Slide object:
- `order` (int, unique, normalized)
- `layout` (`title|content|two_column|quote|section|blank`)
- `title`, `content`, `speaker_notes`, `metadata`

## CRUD
- `POST /api/v1/slides/presentations`
- `GET /api/v1/slides/presentations`
- `GET /api/v1/slides/presentations/{id}`
- `PUT /api/v1/slides/presentations/{id}`
- `PATCH /api/v1/slides/presentations/{id}`
- `DELETE /api/v1/slides/presentations/{id}` (soft delete)
- `POST /api/v1/slides/presentations/{id}/restore`
- `GET /api/v1/slides/presentations/search?q=...`

### Structured ETag / If-Match
- `GET` returns `ETag: W/"v{version}"` and `Last-Modified`.
- `PUT`, `PATCH`, `DELETE`, `restore` require `If-Match`.
- Missing `If-Match` → 428; mismatch → 412.

These weak ETags remain the compatibility contract for structured routes. Standalone source routes use the strong ETags documented below.

## Structured Generation
- `POST /api/v1/slides/generate` (prompt)
- `POST /api/v1/slides/generate/from-chat`
- `POST /api/v1/slides/generate/from-media`
- `POST /api/v1/slides/generate/from-notes`
- `POST /api/v1/slides/generate/from-rag`

Common params:
- `title_hint`, `theme`, `marp_theme`, `settings`, `custom_css`
- `max_source_tokens` / `max_source_chars`
- `enable_chunking`, `chunk_size_tokens`, `summary_tokens`
- `provider`, `model`, `temperature`, `max_tokens`

RAG generation explicitly searches `media_db`, `notes`, and `chats` sources.
Streaming generation is not yet supported; these existing structured generation endpoints return a full presentation payload synchronously. Standalone generation uses the separate asynchronous Jobs transport below.

## Export
- `GET /api/v1/slides/presentations/{id}/export?format=revealjs`
- `GET /api/v1/slides/presentations/{id}/export?format=markdown`
- `GET /api/v1/slides/presentations/{id}/export?format=json`

Notes:
- Reveal.js ZIP bundles assets from `SLIDES_REVEALJS_ASSETS_DIR` or the default bundled assets under `tldw_Server_API/app/core/Slides/revealjs`.
- Markdown export uses `marp_theme` if set; otherwise uses Reveal→Marp mapping.

## Settings Allowlist (Reveal.js)
Allowed keys:
`transition`, `backgroundTransition`, `slideNumber`, `controls`, `progress`, `hash`, `center`,
`width`, `height`, `margin`, `minScale`, `maxScale`, `viewDistance`, `keyboard`, `touch`,
`loop`, `rtl`, `navigationMode`.

## Errors (common)
- 400: invalid query/format/If-Match syntax.
- 404: not found.
- 412: If-Match precondition failed.
- 413: source too large with chunking disabled.
- 422: validation errors (theme/settings/layouts).
- 429: rate limit.

## Metrics
- `slides_generation_latency_seconds{source_type}`
- `slides_generation_errors_total{source_type,error}`
- `slides_export_latency_seconds{format}`
- `slides_export_errors_total{format,error}`

## Standalone HTML Capability And Negotiation

`GET /api/v1/slides/capabilities` is the source-free readiness endpoint. It reports structured read/edit support separately from standalone read/edit/export and standalone generation. Generation can be disabled while saved standalone projects remain readable.

The response is a closed schema. Its capability fields are:

| Path | Available value | Validator unavailable |
| --- | --- | --- |
| `schema_version` | `1` | `1` |
| `content_kind_request_header` | `X-Slides-Accept-Content-Kinds` | same |
| `content_kinds.structured_slides` | `read: true`, `edit: true` | same |
| `content_kinds.standalone_html.read` | `true` | `true` |
| `content_kinds.standalone_html.edit` | `true` | `false` |
| `content_kinds.standalone_html.export_attachment` | `true` | `false` |
| `content_kinds.standalone_html.draft_attachment` | `true` | `true` |
| `content_kinds.standalone_html.reason` | `null` | `validator_unavailable` |
| `generation_modes.structured_slides` | `enabled: true`, `transport: existing_source_endpoints` | same |
| `generation_modes.standalone_html` | `enabled`, nullable `reason`, `transport: slides_generation_job`, fixed `source_kinds`, nullable target/revision fields, and the effective limit objects below | `enabled: false` with the active generation-blocker reason; `validator_unavailable` appears here only when the validator is that blocker |

`generation_modes.standalone_html.source_kinds` is exactly `prompt`, `chat`,
`media`, `notes`, and `rag`. Its target fields are `provider`, `model`,
`adapter_id`, `endpoint_identity`, and `generation_config_revision`. Disabled
generation returns one bounded reason and null target/revision fields. It does
not change standalone read support. Capability discovery performs no live
provider-health request.

The generation reason enum is exactly `feature_disabled`, `egress_disabled`,
`default_model_not_configured`, `default_model_not_allowed`,
`default_endpoint_not_allowed`, `prompt_asset_unavailable`,
`digest_key_unavailable`, `generation_worker_unavailable`,
`generation_reconciler_overloaded`, or `validator_unavailable`. Reason
precedence matters: for example, a disabled feature reports
`feature_disabled` even when the validator is also absent. The independent
content-kind reason still reports `validator_unavailable` in that state.

The default effective values and fixed V1 ceilings are:

| Capability field | Value | Unit or meaning |
| --- | ---: | --- |
| `input_limits.max_request_bytes` | 4,194,304 | raw generation request bytes |
| `input_limits.max_source_chars` | 200,000 | resolved Unicode scalar values |
| `input_limits.max_source_tokens` | 50,000 | resolved source tokens |
| `input_limits.max_audience_chars` | 500 | Unicode scalar values |
| `input_limits.max_source_identifier_bytes` | 256 | UTF-8 bytes |
| `input_limits.max_note_ids` | 100 | IDs |
| `input_limits.max_rag_query_chars` | 20,000 | Unicode scalar values |
| `input_limits.max_rag_top_k` | 100 | results |
| `output_limits.max_provider_response_bytes` | 8,388,608 | provider success envelope bytes |
| `output_limits.max_document_bytes` | 1,048,576 | validated document bytes |
| `content_kinds.standalone_html.limits.max_document_bytes` | 1,048,576 | stored standalone document bytes |
| `content_kinds.standalone_html.limits.max_source_write_bytes` | 1,048,576 | raw save bytes |
| `content_kinds.standalone_html.limits.max_draft_attachment_bytes` | 1,048,576 | raw recovery attachment bytes |
| `content_kinds.standalone_html.limits.max_slides` | 30 | slides |
| `content_kinds.standalone_html.limits.max_nesting_depth` | 128 | element depth |

Configured source and provider-response limits are clamped downward before
publication. The fixed request, document, write, draft, slide, and depth limits
cannot be raised.

Capabilities require normal Slides authentication and return
`Cache-Control: private, no-store` with `Vary` tokens for `Authorization`,
`X-API-KEY`, and `Cookie`. Exact-ID source-free dispatch is available at:

```text
GET /api/v1/slides/presentations/{id}/metadata
```

The metadata route returns bounded kind/title/provenance/count metadata, a
kind-appropriate ETag, `Last-Modified`, `private, no-store`, and the same
authentication `Vary` tokens. It never returns `html_document`.

REST clients must opt into additive kinds with:

```text
X-Slides-Accept-Content-Kinds: structured_slides,standalone_html
```

Without the header, list/search filter to structured records before pagination, and a direct standalone target returns bounded `406 content_kind_not_accepted`. The header selects a representation; it does not replace authentication or ownership checks. List, search, job, delete, and version-list responses remain source-free.

Standalone detail is a discriminated JSON response with `content_kind: "standalone_html"`, `html_document`, `html_sha256`, `html_bytes`, and `html_slide_count`. Source-bearing JSON responses use `application/json`, `nosniff`, and `private, no-store`. No source-bearing route returns `text/html`.

## Standalone HTML Generation

- `POST /api/v1/slides/generations`
- `GET /api/v1/slides/generations/{generation_id}`

Submission requires `Content-Type: application/json`, absent or `identity`
`Content-Encoding`, a raw body of at most 4,194,304 bytes, a unique
`Idempotency-Key` matching `[A-Za-z0-9._~-]{16,200}`, and
`generation_mode: "standalone_html"`. The
`generation_config_revision` must be the current capability value and must
match `sha256:` followed by exactly 64 lowercase hexadecimal characters. The
closed source union supports only the members below. The `prompt` member is the
user's exact bounded direct material; it is not a placeholder. Every resolved
source must also fit the effective 200,000-scalar and 50,000-token limits.

| `source.kind` | Closed members and limits |
| --- | --- |
| `prompt` | Nonblank `prompt` text. |
| `chat` | Nonblank `conversation_id`, at most 256 UTF-8 bytes. |
| `media` | Integer `media_id` from 1 through 9,223,372,036,854,775,807. |
| `notes` | `note_ids`: 1 through 100 unique, nonblank identifiers, each at most 256 UTF-8 bytes. |
| `rag` | Nonblank `query` of at most 20,000 Unicode scalar values and integer `top_k` from 1 through 100; omitted `top_k` defaults to 8. |

`html_options` is also closed and requires every member:

| Member | Accepted values |
| --- | --- |
| `presentation_type` | `pitch-deck`, `tech-sharing`, `product-launch`, `weekly-report`, `course-module`, `keynote`, `data-report`, `training`, `social-media`, `case-study`, `comparison`, or `roadmap` |
| `audience` | Nonblank text of at most 500 Unicode scalar values |
| `slide_count` | Integer from 1 through 30 |
| `visual_direction` | `auto`, `dark-technical`, `minimal-light`, `editorial`, `corporate`, `soft-pastel`, `bold-creative`, or `neo-brutalist` |
| `delivery_style` | `speaker-led` or `self-guided` |

Public requests cannot override provider, model, adapter, endpoint, proxy,
router, or fallback behavior.

A new queued/running request returns `202`. A completed/failed/cancelled idempotent replay returns its closed owner-scoped receipt with `200`. Status includes only bounded receipt/job metadata and never HTML. Provider execution is at-least-once: a precommit worker crash can repeat an external provider call, while transport replay does not create another committed presentation.

Common generation errors include:

- `400 generation_idempotency_key_required`
- `409 generation_idempotency_conflict`
- `409 generation_configuration_changed`
- `503 generation_digest_key_unavailable`
- `503 generation_reconciler_overloaded`
- stable terminal receipt codes `generation_cancelled`, `generation_expired`,
  `generation_quarantined`, `generation_retry_exhausted`,
  `generation_correlation_mismatch`, `generation_receipt_unresolved`,
  `standalone_html_egress_disabled`, `standalone_html_model_not_allowed`,
  `standalone_html_endpoint_not_allowed`,
  `standalone_html_provider_request_invalid`,
  `standalone_html_provider_response_invalid`,
  `standalone_html_provider_response_too_large`,
  `standalone_html_invalid_document`,
  `standalone_html_validation_budget_exceeded`,
  `standalone_html_validator_failed`, and the redacted fallback
  `generation_failed`

Other bounded provider or Jobs terminal identifiers are opaque diagnostics,
not client branching contracts. Unsafe or malformed identifiers become
`generation_failed`.

Generation input becomes inaccessible and expires at 24 hours. The bounded
running reconciler sweep physically purges an expired input within the next 15
minutes. Terminal receipt metadata is retained for 30 days. Receipts and input
are owner-scoped.

## Standalone Source, Version, And Export Routes

- `PUT /api/v1/slides/presentations/{id}/html-source`
- `GET /api/v1/slides/presentations/{id}/versions`
- `GET /api/v1/slides/presentations/{id}/versions/{version}`
- `POST /api/v1/slides/presentations/{id}/versions/{version}/restore`
- `GET /api/v1/slides/presentations/{id}/export?format=html`
- `GET /api/v1/slides/presentations/{id}/export?format=json`
- `POST /api/v1/slides/presentations/{id}/draft-attachment`

HTML Save accepts the complete raw UTF-8 document as
`application/octet-stream`. During the compatibility transition, the server
accepts exactly one syntactically valid strong `If-Match` or one legacy weak
`If-Match`; current clients send strong tags, and standalone responses always
return the resulting strong ETag. The server derives title, digest, count, and
search text. A stale tag returns `412`; a missing tag returns `428`. Existing
structured routes retain weak `W/"v7"` tags and their current behavior.

Generation JSON, raw HTML source writes, and draft attachments accept only an
absent or `identity` `Content-Encoding`. Source writes and draft attachments
use `application/octet-stream` and the fixed raw-byte limits advertised by
capabilities.

Saved HTML export and draft attachment return exact bytes with this fixed boundary:

```text
Content-Type: application/octet-stream
Content-Disposition: attachment; filename="presentation.html"
X-Content-Type-Options: nosniff
X-Download-Options: noopen
Cache-Control: private, no-store
Referrer-Policy: no-referrer
Cross-Origin-Resource-Policy: same-origin
```

The draft attachment does not persist or validate a deck; it is a bounded authenticated recovery echo. JSON export is an attachment named `presentation.json` and carries source as a JSON string, never as HTML.

Standalone validation rejects invalid encoding, NUL, forbidden C0/C1 and bidi
controls, malformed or over-budget documents, URL/resource-bearing
structures, and documents outside the fixed 1 MiB, 30-slide, or depth-128
ceilings. Tabs, line feeds, and carriage returns remain valid text. Validation
is a persistence rule, not a claim that generated JavaScript is safe. The
WebUI never previews or executes the document.

## Standalone Status And Error Codes

Standalone failures are bounded and never echo source. The stable transport
mapping is:

| HTTP status | Stable codes and use |
| ---: | --- |
| `400` | `generation_idempotency_key_required`, `generation_idempotency_key_invalid`, `invalid_content_kind_header`, `invalid_content_length`, or `invalid_if_match` |
| `404` | `presentation_not_found`, `presentation_version_not_found`, `generation_not_found`, `conversation_not_found`, `conversation_empty`, `notes_not_found`, `notes_empty`, `media_not_found`, `media_content_not_found`, or `rag_no_results` |
| `406` | `content_kind_not_accepted` before loading a source-bearing target |
| `409` | `standalone_html_creation_requires_generation`, `content_kind_immutable`, `operation_not_supported_for_content_kind`, `version_content_kind_mismatch`, `generation_idempotency_conflict`, `generation_configuration_changed`, or `generation_correlation_mismatch` |
| `412` | `presentation_version_conflict` or `precondition_failed` for a stale ETag |
| `413` | `input_too_large`, `standalone_html_storage_limit`, or `generation_job_payload_too_large` |
| `415` | `unsupported_media_type` or `standalone_html_unsupported_encoding` |
| `422` | `source_invalid`, `standalone_html_request_invalid`, `standalone_html_invalid_document`, `standalone_html_validation_budget_exceeded`, `json_structure_too_complex`, or `generation_job_payload_invalid` |
| `500` | fixed redacted `standalone_html_response_invalid` for a source-bearing response validation or serialization failure |
| `503` | any generation capability reason listed above when POST is disabled, plus `validator_unavailable`, `standalone_html_validator_busy`, `standalone_html_validator_timeout`, `source_dependency_unavailable`, `source_tokenizer_unavailable`, `generation_digest_key_unavailable`, `generation_reconciler_overloaded`, `generation_receipt_unresolved`, `generation_job_enqueue_rejected`, `generation_job_payload_unavailable`, or `generation_unavailable`; retryable responses include bounded `Retry-After` where applicable |

A missing `If-Match` returns `428 if_match_required`. Provider output failures
are stored in the closed Jobs/receipt failure response rather than changing an
already accepted generation POST into a synchronous payload error.

## Standalone Response Headers

| Response | Required headers |
| --- | --- |
| Source-bearing detail, save, version content, or restore JSON | `Content-Type: application/json`, `X-Content-Type-Options: nosniff`, `Cache-Control: private, no-store`, one strong `ETag`, `Last-Modified`, and `Vary` tokens for `X-Slides-Accept-Content-Kinds`, `Authorization`, `X-API-KEY`, and `Cookie` |
| Saved HTML or draft attachment | `Content-Type: application/octet-stream`, `Content-Disposition: attachment; filename="presentation.html"`, `X-Content-Type-Options: nosniff`, `X-Download-Options: noopen`, `Cache-Control: private, no-store`, `Referrer-Policy: no-referrer`, `Cross-Origin-Resource-Policy: same-origin`, and the same negotiation/authentication `Vary` tokens; saved export also carries its strong `ETag` and `Last-Modified` |
| Standalone JSON export attachment | `Content-Type: application/json`, `Content-Disposition: attachment; filename="presentation.json"`, `X-Content-Type-Options: nosniff`, `X-Download-Options: noopen`, `Cache-Control: private, no-store`, `Referrer-Policy: no-referrer`, `Cross-Origin-Resource-Policy: same-origin`, strong `ETag`, `Last-Modified`, and the same negotiation/authentication `Vary` tokens |

For allowed separate-origin WebUI requests, CORS middleware additionally
preserves `Vary: Origin` and exposes `Content-Disposition`, `ETag`,
`Last-Modified`, `Retry-After`, `Content-Length`, and the configured request or
trace headers. Credentials never use a wildcard origin.
