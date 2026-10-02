# Reverse-image search tools and provider contracts

Date: 2026-09-05
Backlog: TASK-13409
Recovered from: commit `836d2648b7` (original TASK-13175 now collides with unrelated records; those records are unchanged).
Revision date: 2026-10-01
Status: Approved design and three rounds of review corrections; implementation planning authorized.
Scope: Design and implementation planning. Runtime implementation remains a separate execution step.

## 1. Purpose and approved scope

Expose three independent Unified MCP tools and equivalent authenticated REST operations:

| MCP tool | REST route | Operation |
| --- | --- | --- |
| `web.image_sources` | `POST /api/v1/research/image-sources` | Find copies, modified versions, and pages containing an image. |
| `web.image_similar` | `POST /api/v1/research/image-similar` | Find visually similar images and products. |
| `web.image_identify` | `POST /api/v1/research/image-identify` | Return candidate object, product, or landmark identifications with supporting links. |

All three accept public image URLs, owned upload references, and bounded inline image data. Initial adapters are Google Lens through SerpApi, Bing Visual Search through SerpApi, direct TinEye, and a configurable HTTP backend implementing the contract below. Each call selects one backend; defaults are independent for each operation.

The release includes the minimal durable image-upload extension described below, capability discovery, submission policy, temporary staging when configured, structured failures, documentation, and deterministic tests. It excludes multi-provider fan-out, automatic provider fallback, background searches or monitoring, automatic result ingestion, a frontend redesign, local index construction, a bundled CLIP Retrieval adapter, and an additional LLM synthesis stage. Source matching does not establish original authorship. Identification returns evidence-backed candidates rather than a guaranteed answer.

## 2. Architecture and repository integration

Use one shared core service with thin REST and MCP adapters. This gives other server clients the same behavior as MCP while keeping provider dependencies outside the standalone gateway package.

Alternatives considered: MCP-only implementation is initially smaller but couples reuse to MCP; a separately deployed gateway adds deployment and authentication work without a first-release requirement. The shared in-server service was approved.

Proposed component boundaries, relative to the repository root:

| Component | Responsibility and dependencies |
| --- | --- |
| `tldw_Server_API/app/core/Image_Search/` | Typed contracts, configuration, image resolution, provider selection, normalization, execution bounds, and staging coordination. Depends on shared storage, outbound policy, and injected provider clients. |
| `Image_Search/providers/` within that core package | Focused Lens, Bing, TinEye, and custom HTTP adapters. Translate provider requests/results and declare supported operations and options. |
| `tldw_Server_API/app/api/v1/schemas/image_search_schemas.py` | Public Pydantic requests and responses backed by core contracts. Strict unknown-field rejection and exactly-one-image validation. |
| `tldw_Server_API/app/api/v1/endpoints/image_search.py` | Authentication, authorization, rate limits, HTTP status mapping, and delegation to the core service. |
| `tldw_Server_API/app/core/MCP_unified/modules/implementations/image_search_module.py` | The three tool definitions, per-tool permissions, request context, structured results, and existing web-tool evaluation metadata. |
| Existing ingestion, storage, and DB abstractions | Extend `/media/add` with a file-only `image` branch producing durable original-image records; resolve owned images and persist short-lived staging records where necessary. All new SQL belongs in DB management. |

Reuse patterns from `web_search_module.py`, `web_fetch_module.py`, `web_tool_base.py`, and their tests. Keep the existing text-only `perform_websearch` flow intact. Register module ID `image_search`, department `research`, gated by `MCP_ENABLE_IMAGE_SEARCH_MODULE` or explicit modules YAML configuration, disabled by default. Classify it in the module surface as external-network capable. Module enabling does not grant all three tool permissions to every profile.

The standalone gateway's `apps/mcp-unified/src/mcp_unified/profiles/` integration exposes the new tools only through explicit deployment/profile configuration. Provider clients remain in the server implementation. Extend the shared `profiles/subjects.py` domain-argument key set with `image_url`, so runtime enforcement, policy simulation, and policy explanation all extract the same domain subject. Do not add a second image-specific domain parser. Preserve subject-extraction limits: a URL exceeding the gateway's existing subject length limit must fail safely, not lose its domain check.

The image-search MCP adapter must validate original arguments without `WebToolBase`'s control-character stripping. For example, base64 containing NUL must fail rather than becoming valid after sanitization. Share the strict Pydantic request contract with REST; preserve omitted fields, reject null source fields, boolean integer values, unknown fields, malformed base64, and MIME/source conflicts. Keep this override local to image-search rather than changing all web tools.

REST uses equivalent operation authorization and submission policy so changing transports cannot bypass restrictions.

Existing managed reference-image seams are `core/Image_Generation/reference_images.py`, `/api/v1/files/reference-images`, the media ingestion pipeline, and `core/Storage/`. Reuse their ownership/storage patterns through a small image-search resolver. Any necessary shared extraction is confined to the image lookup/read boundary; image-search limits must not depend on image-generation settings. Streaming reads must enforce limits before loading an entire oversized object. These seams currently provide image lookup, not a complete upload workflow: `MediaType` in `media_request_models.py` excludes `image`, while the reference resolver requires `Media.type='image'` and an original `MediaFiles` row. Closing that gap is an explicit delivery dependency, not assumed existing functionality.

### Durable image-upload prerequisite

Extend `POST /api/v1/media/add` and its form/schema/ingestion dispatch with `media_type=image`, retaining its authenticated user scope, `MEDIA_CREATE` permission, request limits, billing/quota checks, and existing storage/DB abstractions. The v1 image branch accepts multipart `files` only, with `keep_original_file=true` and `perform_analysis=false`; reject URL inputs and incompatible values before persistence. Do not change defaults or processing behavior for other media types. This is a validate-and-store path: no OCR, captions, embeddings, claims extraction, or model/provider calls, including through generic post-ingestion hooks. No additional upload endpoint, MCP tool, or image-search-specific durable store is introduced.

Validate JPEG/PNG/WebP integrity, declared versus detected MIME, single-frame content, and bounded byte/pixel counts before storage. Use the service-wide image-search ceilings from section 3, intersected with existing upload/transport limits; backend-specific submission caps apply later when a backend is selected. Persist the original bytes through the configured storage backend and create an owned `Media` row with type `image` and an original `MediaFiles` row with detected MIME, actual size, and the managed storage path expected by the reference resolver. Metadata-only image ingestion must not be skipped merely because there is no extracted text; do not fabricate text to pass document-processing checks.

Preserve the existing per-item `/media/add` response envelope. Add an explicitly typed optional `original_file_id` to `MediaItemProcessResult`, leaving it unset for unrelated media types; an image item reports success only after its durable original is readable and its DB records exist. Its existing `db_id` is the media ID, while `original_file_id` is `MediaFiles.id`, also returned as `file_id` by reference-image discovery and used as `upload_id` by search. A deduplicated item may return an existing reference only after validating ownership and durable availability. On partial failure, return an ingestion error without a usable file ID and compensate new storage/DB writes through existing cleanup conventions; do not delete a pre-existing deduplicated object. Storage failures must not produce phantom successful references.

The documented client flow is multipart `/media/add` with the three image form values above, then optional `/files/reference-images` discovery to confirm the returned `original_file_id`, then a search using that value as `upload_id`. Persistence requires no configured image-search provider or private-submission permission and makes no search or inference requests; normal configured storage, DB, and authentication infrastructure remain available, including remote storage. Backend grants and submission policy are independently enforced by the subsequent search. The implementation plan must deliver this upload-to-search vertical slice before claiming support for owned uploads.

## 3. Public request contract

Exactly one of these fields must be present, with a non-null value:

| Field | Contract |
| --- | --- |
| `image_url` | Absolute public HTTP(S) URL, at most 8,192 characters. No embedded credentials, local paths, or data URI. |
| `upload_id` | Positive integer identifying a durable managed image's `MediaFiles.id` in the authenticated user's scope. The public name is `upload_id`; existing file discovery returns it as `file_id`. |
| `image_base64` | Strict base64-encoded raster bytes; requires `mime_type`. No data-URI prefix. |

`mime_type` is permitted only with `image_base64`. The initial allowed formats are JPEG, PNG, and WebP, intersected with the selected backend's supported formats. The service verifies actual format and integrity, including MIME mismatch, and rejects animated or multi-frame inputs in v1. Callers upload durable images through the `/media/add` image extension specified in section 2 and obtain a managed reference; searching does not create a second upload store. Inline bytes remain transient.

Common optional fields:

- `backend`: registered backend ID, at most 64 characters. Omission uses that operation's explicitly configured default. Missing or invalid default configuration fails with `backend_not_configured`; a configured backend with a temporary operational failure returns `backend_unavailable`. No implicit fallback or credential-based selection.
- `limit`: integer from 1 to 25. When omitted, resolve to `min(10, effective_backend_max_results)` after backend selection. Preserve omission through schema validation rather than inserting 10 prematurely. An explicit value above the backend's effective maximum returns `unsupported_option`; booleans are invalid.
- `language`: trimmed language code, at most 16 characters. Omission preserves the provider default. An explicitly unsupported language option fails before image submission.
- `query`: trimmed non-empty string, at most 500 characters, permitted only for similarity and identification. Examples: "the lamp" or "manufacturer and model". It is a provider-supported refinement, not a promise of segmentation. Backends that cannot apply it return `unsupported_option` before submission.

Unknown fields and conflicting image sources fail validation. The initial service cap is 3 MiB of image bytes and 16 million decoded pixels; administrators can lower these caps. Intersect these caps with the selected submission method's backend limits, not a single limit shared by all methods. Select the method before image acquisition and enforce its effective byte, pixel, dimension, and format limits during bounded acquisition and integrity validation. Images exceeding effective limits are rejected without silent resizing, cropping, re-encoding, or switching submission methods. Discovery advertises the effective limits by method; errors safely identify the selected method and exceeded limit. A caller can deliberately prepare a smaller image and resubmit.

Transport body caps remain authoritative. Current MCP HTTP and WebSocket defaults are smaller than the full inline-image allowance; discovery/documentation must explain that inline inputs are limited by the transport envelope, including base64 overhead. Clients use `upload_id` for larger images. Do not globally raise MCP body limits to accommodate this feature.

## 4. Public response contract

Every successful response has `ok: true`, `operation` (`sources`, `similar`, or `identify`), `backend`, `provider`, `collection_scope`, `result_count`, `truncated`, `results`, and `warnings`. `collection_scope` is `{kind: "web"}` or `{kind: "private", name: "configured collection name"}`. `result_count` is the number of returned normalized entries. Each warning has a stable `code` and a bounded safe `message`. MCP adds its existing `eval` metadata outside the shared domain result.

Return typed allowlisted fields rather than raw upstream payloads. Optional unavailable values are omitted. Preserve provider order after dropping malformed entries and applying bounds; do not introduce a cross-provider ranking model. Result URLs are returned as references, with scheme validation; thumbnails and matching pages are not automatically downloaded. For private collections, a backend may return an opaque `asset_id` in place of a web link, never a server filesystem path.

| Operation | Per-result fields |
| --- | --- |
| Sources | `title`, `page_url` and/or `asset_id`, optional `image_url`, `thumbnail_url`, `width`, `height`, `match_type`, `provider_match_type`, `first_seen_at`, `first_seen_raw`, and `score`. |
| Similar | `title`, `page_url` and/or `asset_id`, optional `image_url`, `thumbnail_url`, dimensions, `score`, and `product`. Product metadata may contain provider-supplied name, price, currency, and availability. |
| Identify | `name`, `entity_type`, non-empty `evidence`, optional `confidence`. Each evidence entry has a title and a `page_url` or private `asset_id`; at most five entries per candidate. |

Source `match_type` values are `exact`, `modified`, or `unknown`, reflecting the adapter's documented interpretation of provider evidence; omit it when the provider offers no match classification. "Exact" denotes a provider match category, not a cryptographic byte comparison. Preserve a provider's original label in `provider_match_type`. First-seen metadata describes the provider's observation; never relabel it as publication time or the original source. `first_seen_at` is an ISO 8601 timestamp only when the upstream precision and timezone permit conversion; otherwise preserve its bounded string in `first_seen_raw`. Dimensions are positive integers. Scores use finite numeric `value` and a bounded descriptive `scale`; product name, price, currency, and availability are bounded strings preserving upstream representations.

Identification `entity_type` is `object`, `product`, `landmark`, or `unknown`. A candidate must have a provider-supported name and supporting evidence. An ordinary visual-match title alone is insufficient to claim that the entity has been identified. Valid responses without sufficient evidence return an empty list and `insufficient_identification_evidence`. No synthetic confidence is added. Source/similarity scores and identification confidence retain `{value, scale}` with documented provider semantics; no normalization into a universal percentage.

Strings are bounded: names/titles 500 characters, other descriptive text 2,000, URLs 8,192. Results are capped at the requested limit and the serialized public domain response at 256 KiB. Clip descriptive text with a warning; omit entries containing oversized identifiers/URLs instead of cutting links into invalid values. At most 20 warnings are returned. Truncation is true whenever content or results are omitted for size/count bounds or the provider indicates more results. Upstream estimates are optional metadata, not a fabricated total.

An empty valid provider result is a successful search. Malformed JSON or an invalid top-level shape is `provider_response_invalid`. Malformed individual entries can be omitted with a warning, but if all entries in a non-empty relevant result section are malformed, return that error instead of reporting no matches.

## 5. Backend capabilities and initial adapters

Backend IDs identify configured instances; provider names identify implementations. Example IDs are `lens_primary`, `bing_primary`, and `tineye_primary`. Multiple instances may use the same implementation with different settings.

| Adapter | Sources | Similar | Identify | Image submission |
| --- | --- | --- | --- | --- |
| `serpapi_google_lens` | Lens exact-match results | Visual/product results | Disabled pending a substantiated direct identification mapping | Public URL; direct image upload and returned image ID where documented |
| `serpapi_bing_visual` | Pages-with-image results | Related visual results | Explicit identification/looks-like evidence where returned | Public URL; uploaded images use configured staging unless direct-image support is verified |
| `tineye` | Matching copies and modifications | Unsupported | Unsupported | Provider URL or direct byte upload |
| `custom_http` | Declared capability | Declared capability | Declared capability | Declared URL or inline-byte modes |

The Lens adapter uses exact matches for sources, visual matches for similarity without a refinement and the documented refinement-capable visual/product route when applicable. Lens identification is disabled: the reviewed general-result contract does not substantiate a direct name-and-evidence mapping that satisfies section 4 without an AI-overview follow-up. Do not advertise this capability, configure it as the identification default, or return a successful empty identification result in its place. Selecting Lens for identification returns `unsupported_capability` before image acquisition. Enable it only after a documented mapping and sanitized fixture prove a compliant single-search response. The Bing adapter selects the relevant structured sections from its response. TinEye cannot be selected for similarity or identification. Unsupported combinations and options are rejected before fetching or submitting an image.

Provider contracts must be pinned to sanitized response fixtures at implementation time. A capability is enabled only when a documented response mapping and contract tests substantiate it. Identification remains conditional on actual evidence in each response; no extra generative model or automatically followed AI-overview request is introduced. SerpApi and the existing Serper text-search integration are distinct providers and must not share credentials implicitly.

Operational limitation of this initial capability matrix: among the bundled commercial adapters, identification of an owned upload or inline image requires Bing through SerpApi plus permitted, provider-reachable public staging. Lens identification is disabled and TinEye cannot identify. A direct-only deployment needs a compatible custom backend supporting identification from bytes; one is not bundled. Public URL identification through Bing does not require staging. Setup documentation and capability discovery must distinguish operation support from usable private-input methods; do not imply that every operation supports private images with every configured provider or silently relax policy to make identification available.

`GET /api/v1/research/image-backends` exposes only backends authorized for the calling user, with providers, operation/option support, effective `max_results`, resolved default limits, collection scope, and `submission_methods`. Each available method (`public_url`, `direct`, or `staged`) declares compatible caller input types, MIME types, effective `max_image_bytes` and `max_image_pixels`, and any width/height caps. Keep transport-envelope limits separately documented; a backend method cap does not increase them. For example, SerpApi's documented 500 KB direct-upload cap applies to `direct`, not automatically to URL submission. Exclude credentials, grants, internal endpoints, storage paths, and staged URLs; omit inaccessible defaults as well as inaccessible backends. Expose the same sanitized inventory through an MCP resource, `image-search://backends`, preserving the three-tool surface. Discovery is a snapshot, not a live external request on every list call; caller authorization is applied on each request, not cached across users.

## 6. Image access and submission policy

Resolve authenticated identity before accessing any owned image. Missing, deleted, expired, or other-user references return indistinguishable `image_not_found` errors. Neither tool arguments nor custom backends can select another user's storage scope.

Every registered backend has administrator-controlled `allowed_user_ids` and `allowed_roles` grants, defaulting to empty (no caller access). Membership in either grant list permits backend use only when operation authorization and all other policies also allow it; neither module enablement nor default-backend selection bypasses these grants. Use authenticated server identity and trusted AuthNZ roles, never caller-supplied role strings. Apply the same check in REST, MCP, and discovery before image acquisition or provider requests. An inaccessible backend selected explicitly or via a default returns `backend_not_found`, indistinguishable from an unknown ID. Input ownership and permission to submit private bytes do not grant collection-search access.

For v1, each private backend instance represents one collection explicitly shared with its granted users/roles. Its fixed endpoint/credential must be scoped to that collection; do not rely on a display-only collection name to isolate a credential spanning unrelated collections. Separate private collections require separate registered instances and appropriately scoped upstream access. No user identity is forwarded to the backend, and no per-result tenant filtering is promised: grant holders can search the entire registered collection. Recheck grants on each call so revocation is not hidden by cached capabilities.

For URL input, validate and fetch with bounded streaming through existing outbound-policy and HTTP-client mechanisms. Enforce host/IP/port restrictions, connection-time address safety, and policy on each redirect; allow at most five redirects. Where MCP profile domain rules apply, check every source URL and redirect through the existing permission mechanism before fetching, and surface ask/deny decisions rather than bypassing them through internal calls. The shared subject-extractor change protects the initial gateway call only; it does not authorize redirects. Wire a trusted request-scoped permission callback from the actual MCP execution context through the core binary acquisition path. Re-evaluate each hop with the existing domain rules and approval mechanism; `ask` without a matching approval yields `permission_required`, and deny yields `permission_denied`, both before network I/O to that hop. Missing permission context must not implicitly allow a call when profile-domain policy applies. REST uses its own trusted operation/domain policy context. Tests must exercise production wiring, not only a callback injected directly into an isolated module.

Carry the original URL's public-origin provenance through resolution. Do not use `web.fetch`'s text extraction contract as a binary downloader. URLs requiring user credentials or pointing to private infrastructure belong in the upload flow. Reject this deployment's managed-image and staging URLs as public input so they cannot bypass private-image policy.

Backend selection and policy preflight precede image acquisition. Private custom endpoints require explicit administrator configuration and narrowly scoped existing egress authorization. Declaring a backend self-hosted does not broadly exempt it from network policy. Keep endpoint location, collection scope, and permission to receive private bytes as separate settings: a locally hosted proxy can still forward images externally.

`private_submission` has three deployment-controlled values. It governs private-image submission to external services and, independently of recipient location, creation of any publicly reachable staging URL:

| Value | Effect on uploads and inline images |
| --- | --- |
| `disabled` (default) | No external submission of private images and no public staging for any backend. Public URL searches remain available. |
| `direct` | Permit direct byte upload to the selected configured external provider; no public staging for any backend. |
| `staged` | Permit external direct upload and explicitly opt in to temporary public URL staging when required, including for self-hosted recipients. |

Each custom backend additionally declares an administrator-controlled `allow_private_images`, default false. This setting governs submission to that trusted endpoint; capabilities returned by the endpoint cannot grant themselves permission. Deployments must classify externally forwarding proxies as external recipients. Model arguments cannot override submission settings.

A non-forwarding trusted self-hosted backend may receive private bytes directly when `allow_private_images=true` even with `private_submission=disabled`; that does not authorize publication. Every staged private-image request requires `private_submission=staged`, plus backend access grants, `allow_private_images=true` for custom backends, URL-input support, and usable staging infrastructure. For owned uploads or inline input, a self-hosted URL-only backend with `allow_private_images=true` and `private_submission=disabled` or `direct` returns `private_submission_denied` before image acquisition, staged-copy creation, or search submission. Public URL input is not subject to this private-input denial. Conversely, `private_submission=staged` cannot override a custom backend's `allow_private_images=false`. The staging service enforces this gate itself as well as the caller's preflight; a configured base URL or a local recipient is never implicit publication consent.

Select a submission method deterministically from source provenance, configured policy, and adapter capabilities before acquisition. For public URL input, use `public_url` when supported, otherwise `direct` if supported. For owned uploads or inline bytes, prefer permitted `direct`; use `staged` only when direct input is unsupported and staging is permitted and available. If direct input is supported but exceeds its limits, fail rather than silently creating a public staged copy. These are preflight choices within one backend, not retries or provider fallback. Direct means provider-compatible byte submission (multipart upload or inline base64); staged means a TLDW temporary URL submitted through the backend's URL input. Public URL searches pass the validated final public URL. This means the provider may fetch the URL later; validation cannot guarantee that externally hosted content remains unchanged. Return a `provider_refetches_url` warning when that mode is used. When bytes were received through `upload_id` or inline input, never downgrade their private provenance by constructing an internal URL and treating it as public input.

### Temporary staging lifecycle

When staging is required and explicitly permitted by `private_submission=staged` and all recipient-specific checks above, create a transient copy through an injected staging service using configured storage and a provider-reachable HTTPS base URL. Public reachability is an operator prerequisite. Do not expose the original managed-media object or change its ACL.

All instances serving the staging base URL must share both the authoritative token records and the staged bytes, using shared object storage or a shared filesystem namespace. Shared metadata alone is insufficient. Instance-local staging is supported only when the deployment is explicitly restricted to a single serving instance; sticky load-balancer routing is not a substitute. Reject incompatible configured topology as `staging_unavailable` before staging or provider submission. Multi-instance deployment validation must include creating through one instance and reading/revoking through another; concurrent deletion and sweeper cleanup must be idempotent.

Use a random token with at least 128 bits of entropy, expiry at ten minutes, and a storage record associating the staged object with owner, call, and expiry. Serve the transient bytes at `GET /api/v1/research/image-staging/{token}`; unlike the search routes this uses the scoped bearer token rather than user authentication. Token URLs grant access only to that staged image; they are not returned in normal tool results. Serving requires a valid unexpired record and supports repeated provider reads until expiry/revocation. Expired or revoked tokens yield 404 regardless of physical cleanup progress.

Delete/revoke the transient copy after completion, failure, or cancellation. An internal lifecycle sweeper runs on startup and every minute to remove expired objects after crashes; use existing service scheduling conventions. Storage metadata operations go through DB management, and multi-worker deployments share the record store. Cleanup failures produce an operational warning and are retried by the sweeper; they do not erase an otherwise successful search result. Avoid access-log capture of tokens and configure no-store response caching.

Keep staging objects subject to a configured quota and reject staging before provider submission when it is exhausted. Do not introduce a user-facing background-job system for synchronous searches. Provider-owned upload IDs have their own expiry/retention rules; TLDW cleanup only guarantees removal of TLDW-controlled transient copies. It cannot promise deletion of upstream retained data.

## 7. Custom HTTP backend protocol v1

Administrators configure a fixed base URL, optional credential reference, collection scope, permitted operations, backend user/role grants, and private-image policy. Grants and collection isolation follow section 6; a custom capability response cannot grant callers access or broaden the configured collection. Runtime callers supply only the registered backend ID. Transport uses JSON over HTTPS; explicit private-network HTTP may be permitted for configured self-hosted endpoints under the existing egress rules. Credential-bearing API requests do not follow redirects.

`GET {base_url}/v1/capabilities` returns `schema_version: 1`, `operations` (a subset of `sources`, `similar`, `identify`), positive integer `max_results`, `input_limits`, and `options` keyed by operation with boolean `query` and `language` flags. `input_limits` is a non-empty map with keys drawn from `image_url` and `image_base64`; key presence declares support. Each entry supplies non-empty `mime_types`, positive integer `max_image_bytes` and `max_image_pixels`, and optional positive `max_width` and `max_height`. Byte limits describe decoded image-file bytes, not JSON/base64 envelope size. Do not retain ambiguous flat image limits alongside this map. The server derives `public_url` and permitted `staged` methods from `image_url`, and `direct` from `image_base64`, intersecting each with local acquisition, staging, and policy limits. Built-in adapters publish the same effective method model from their provider-specific contracts.

The effective result maximum is `min(25, backend_max_results)`; an omitted caller limit resolves to `min(10, effective_backend_max_results)`. An explicitly requested limit above the effective maximum returns `unsupported_option` before acquisition/submission. For example, a backend maximum of 5 accepts omission as 5 but rejects an explicit 10. Validate capability responses and cache them for five minutes; a missing/invalid/expired discovery response makes that backend unavailable until refreshed. Cache provider facts only, not caller grants. Discovery never downloads credentials or new endpoints from the backend.

`POST {base_url}/v1/{operation}` receives `schema_version: 1`, an opaque `request_id`, exactly one provider-compatible image source, `mime_type` for inline input, and validated `limit`, `language`, and `query` when supported. Do not send internal `upload_id`, user identity, filesystem paths, or unrelated request context. Private images are sent as bytes or via permitted staging, never by giving a backend access to TLDW's authenticated upload API.

Success is HTTP 200 with `{schema_version: 1, ok: true, results: [...], truncated: false, warnings: []}`, using the operation-specific result fields in section 4. The TLDW service supplies trusted backend/provider/collection metadata and revalidates all results. Errors use a non-2xx status and `{schema_version: 1, ok: false, reason_code, message}`; unknown reason codes and raw messages map to safe local errors. No arbitrary request/response template language, downloaded code, or dynamic SDK loading is included.

A privately hosted collection reports its configured collection name in public responses. Such a service searches the images in its index; it does not acquire web-scale coverage merely by being self-hosted. CLIP Retrieval is a concrete later adapter candidate, but its native `/knn-service` is not claimed to implement this protocol.

## 8. Execution, errors, and observability

Each operation has a 60-second end-to-end deadline covering queueing, image acquisition, upload/staging, and provider search. Use cancellable async I/O, bounded connections, and a 2 MiB upstream decoded-response cap. Do not follow pagination, provider-issued follow-up URLs, or queued searches automatically. A provider search is one logical operation; preparatory image upload and capability lookup are separate bounded requests within the same deadline.

Apply per-user request limits and per-backend concurrency limits before chargeable work. Start with 10 calls per minute per user and three concurrent calls per backend instance, configurable downward or upward by administrators according to quotas. Use the existing core Resource Governor directly for one shared REST/MCP admission path, not the MCP wrapper's optional in-memory fallback or a separate semaphore per adapter. Reserve/commit the user's request unit once in the image-search namespace and acquire a concurrency lease scoped to the selected backend instance. Retain the backend lease through image acquisition, preparation, search, and cleanup; release in `finally` on success, errors, timeouts, and cancellation. Lease expiry must cover the full deadline plus bounded cleanup and permit reclamation after process loss. Retrying a safely unsent request stays within the same admission and does not spend a second user unit; ambiguous submitted work is not refunded.

Multiple workers/instances require shared Redis enforcement with `fail_closed`; reject unavailable or non-shared governance before provider work, never silently fall back to memory. An explicitly single-worker deployment may share one application-scoped memory governor between REST and MCP. Configure separate policy scopes for per-user requests and per-backend leases; a global all-backends lease bucket is not equivalent. Existing ingress limits remain independent outer guards, not a second charge against the image-search request bucket. Enforce deployment-wide limits through this shared infrastructure; do not describe process-local counters as global quotas.

Allow at most one retry and only when the transport proves no request was submitted or the provider explicitly guarantees that retry is safe without duplicate charge. Do not retry ambiguous post-send timeouts, disconnects, or generic 5xx responses. Provider rate limits return a safe bounded `retry_after_seconds` when available. Cleanup executes even when the caller disconnects.

| Reason codes | REST status |
| --- | --- |
| `invalid_arguments`, `unsupported_image`, `unsupported_capability`, `unsupported_option`, `image_limit_exceeded` | 422 |
| Request/transport body too large | 413 |
| `image_not_found`, `backend_not_found` | 404 |
| `permission_denied`, `permission_required`, `outbound_policy_denied`, `private_submission_denied` | 403 |
| `rate_limited` | 429 |
| `backend_not_configured`, `backend_unavailable`, `staging_unavailable`, `storage_unavailable` | 503 |
| `image_fetch_failed`, `provider_failed`, `provider_response_invalid` | 502 |
| `deadline_exceeded` | 504 |

The shared failure envelope is `{ok: false, operation, reason_code, message}` with an optional safe backend ID and retry delay. REST retains its native authentication/schema-validation handling and maps domain failures to the table; MCP retains native protocol validation and returns structured tool execution failures through existing conventions. A search failure never masquerades as an empty successful result.

Use Loguru and existing MCP observability. Record request ID, operation, backend ID, latency, outcome, counts, and byte sizes. Do not log image bytes, image hashes, upload identifiers, full input URLs, signed/staged URLs, API keys, or raw provider bodies. Cover protocol hooks, audit storage, exception logging, and HTTP access logs as well as the adapter logger; adding base64 arguments must not introduce persisted image payloads through generic tool reporting. Error messages use allowlisted context. Returned page content and labels are untrusted data, not instructions.

## 9. Verification and delivery criteria

The implementation plan must include behavioral tests with injected clients and sanitized provider fixtures:

1. All three tools and REST routes enforce equivalent contracts, user scope, and policy. Discovery reflects actual adapter capabilities and method-specific submission limits. Module registration stays opt-in; per-tool denies apply independently. Test empty grants, user/role grants, revocation with cached capabilities, inaccessible default selection, cross-user discovery, and a caller who owns an input image but lacks access to the selected private collection; denied calls perform no acquisition or provider requests.
2. Provider fixture tests cover each enabled capability, request option mapping, no-match responses, missing identification evidence, malformed top-level responses, malformed entries, ranking preservation, and provider score semantics. TinEye rejects unsupported operations before image acquisition/submission. Lens identification is absent from capabilities and rejected before acquisition until a compliant mapping is substantiated; it must never masquerade as a successful no-match search.
3. Input tests cover conflicting sources, strict base64, MIME spoofing, corrupt files, animation, pixel and byte caps, streaming truncation, transport limits, deleted references, and another user's upload ID. Test distinct direct/URL limits, deterministic preflight method selection, effective-limit errors, and no silent switch to staging on direct-upload overflow. Property-based tests exercise source selection and normalization bounds.
4. Outbound tests cover local/reserved addresses, redirects, DNS/address changes, source-domain permissions, credential redirects, private backend exceptions, and a denied backend generating no image submission.
5. Privacy/lifecycle tests cover all three private-submission modes, original-object ACL preservation, unguessable expiring staging URLs, revocation, repeated reads, quota exhaustion, cancellation, crashes, cleanup retry, and cross-worker expiry enforcement. Test creation on instance A followed by reads and revocation on instance B using shared records and bytes; reject multi-instance topology with instance-local blobs, and test idempotent concurrent cleanup. Provider-owned upload expiry is documented separately.
6. Execution tests cover deadlines, concurrency/rate limits, safe pre-send retry, no ambiguous retry, upstream/output caps, structured error mapping, and redaction across MCP reporting and provider exceptions.
7. A custom-backend contract fixture verifies version rejection, per-input capability intersection, supported options, authentication, private collection metadata, typed results, and safe failures. Test malformed/missing input-limit maps, independent URL/base64 caps, and backend maxima below/equal to/above 10 with omitted versus explicit limits. Verify that upstream declarations cannot expand local grants or collection scope. It is a test service, not a deployed sample search engine.
8. Opt-in live smoke checks use explicitly configured credentials and benign test images for Lens, Bing, and TinEye. Regular CI makes no external requests or charges. Missing live credentials are reported as unverified live behavior, not a successful smoke test.

9. An integration test uploads an image through `/media/add`, verifies durable owned `Media`/`MediaFiles` records and `original_file_id`, finds the same `file_id` in reference-image discovery, and searches it as `upload_id` through REST and MCP with an injected provider. Cover missing/incompatible retention/analysis form values, rejected image URLs, malformed/oversized images, storage/DB failure compensation, owned deduplicated references, deletion, and another user's discovery/search. Assert zero model, embedding, claims-extraction, or search-provider work during ingestion even when generic hooks are enabled; configured remote storage is exercised through an injected storage backend. Regress existing non-image ingestion behavior.
10. Exercise private input to a non-forwarding self-hosted URL-only backend with `allow_private_images=true` and each `private_submission` value. `disabled` and `direct` must deny with zero image reads, staging writes, or search calls; `staged` permits the normal lifecycle only when the remaining checks pass. Also verify public URL input is not blocked by the private-input gate, trusted direct-byte submission remains possible with external submission disabled, `allow_private_images=false` still denies private input, and calling the staging service directly cannot bypass the publication gate.
11. Test capability discovery and setup examples for private identification: Bing has no usable private-input method without permitted working staging; Lens identification and TinEye identification remain unsupported; a custom direct-byte identification backend remains the documented alternative. A public URL Bing identification request remains available without staging when otherwise authorized.

12. Gateway tests prove `image_url` produces a domain subject for runtime, simulation, and explanation, including deny and ask rules. Production MCP wiring tests follow an allowed source redirecting to a denied/approval-required domain and assert zero I/O to the blocked hop. Missing trusted permission context and over-limit permission subjects fail safely.
13. REST/MCP parity tests send identical malformed original arguments, including base64 with NUL/control bytes, null or multiple sources, boolean integers, unsupported MIME, and unknown fields. Assert rejection before acquisition/provider work and preservation of omitted `limit`; regress unrelated web-tool sanitization.
14. Shared-governor tests interleave REST and MCP for one user's request allowance and two users targeting one backend's concurrency ceiling. Cover separate backend buckets, held leases during provider work/cleanup, release on timeout/cancellation, crash expiry, retry without double-charge, Redis outage fail-closed, startup topology rejection, and the explicitly single-worker shared-memory mode.

Before implementation completion, run focused unit/integration/property tests, relevant ingestion and MCP/gateway regression tests, formatting/linting, and Bandit from the project virtual environment on touched Python code. Update provider setup, the multipart image-upload and file-ID mapping example, REST/MCP usage, custom backend protocol, image limits, universal staging opt-in, and the direct-only private-identification limitation. This specification-only task records document verification; runtime tests and Bandit are not applicable until code changes exist.

## 10. Evidence and review record

Primary provider references consulted during brainstorming (2026-09-04/05):

- [Microsoft Bing Search API retirement](https://learn.microsoft.com/en-au/lifecycle/announcements/bing-search-api-retirement): official APIs retired August 11, 2025. The proposed Bing integration uses a separate provider service.
- [SerpApi Google Lens](https://serpapi.com/google-lens-api): structured search modes and refinement parameters.
- [SerpApi Bing reverse image](https://serpapi.com/bing-reverse-image-api): URL input and structured matching/recognition sections; documented dimension cap is 4,000 pixels per side.
- [SerpApi image upload](https://serpapi.com/image-api): JPEG/PNG/WebP direct upload, documented 500 KB cap, and image IDs expiring after ten minutes. Adapter byte limits must interpret the provider's accepted limit conservatively and verify it in contract/smoke tests.
- [TinEye API](https://services.tineye.com/TinEyeAPI): matching copies/modifications, URL/upload support, and documented 1 MB image cap.
- [CLIP Retrieval](https://github.com/rom1504/clip-retrieval): a self-hostable index and image-query service; requires its own indexed collection and a future adapter.

Provider documentation establishes possible integration paths, not a live integration test. Recheck contracts while implementing, especially identification fields and supported upload modes. If an advertised capability cannot be substantiated, report the gap and revise the capability matrix before claiming completion.

User approvals: separate tools for all three uses; Lens, Bing, TinEye, and configurable/self-hosted backends; URL and private-image inputs; shared-service architecture; public contracts; image handling/provider execution; integration/testing/release scope.

Review corrections approved on 2026-09-05: disable unsubstantiated Lens identification; enforce backend user/role grants and one explicitly shared collection per private instance; require shared staging bytes as well as records across instances; expose and enforce submission-specific limits; cap omitted result limits by the selected backend's maximum. These corrections are reflected in the capability matrix, request/default semantics, discovery, custom protocol, authorization, lifecycle, and acceptance tests above.

Second review corrections approved on 2026-09-05: include a minimal file-only `/media/add` image-ingestion extension rather than assuming the existing type enum supports images; make `private_submission=staged` the explicit publication gate for all recipients, including self-hosted URL-only backends; document the current Bing-plus-staging dependency for bundled private-image identification. These corrections include upload-to-search, no-model-call, failure-cleanup, publication-policy, and capability-discovery tests.

Third review corrections approved on 2026-10-01: add `image_url` to shared gateway permission subject extraction and wire trusted per-hop checks; prevent MCP sanitization from repairing malformed image arguments; reuse shared Resource Governor request accounting and backend concurrency leases with Redis fail-closed in distributed deployments. The specification and linked implementation plan include behavioral regression tests for all three. Recover this document under TASK-13409 because its former tracking ID collides with unrelated tasks in this checkout.

Specification self-review: checked scope against those approvals, required/optional fields, response semantics, ownership, private submission, provider limitations, transport limits, custom protocol, lifecycle, error mapping, and verification. Checked all three review rounds for consistency across their affected sections, including the ingestion prerequisite, trusted permission context, original-argument validation, shared quota enforcement, and separation of trusted local byte submission from public staging. No runtime behavior or provider integration has been implemented or live-tested. Next checkpoint: execute [the five-stage implementation plan](../../IMPLEMENTATION_PLAN_reverse_image_search.md) with TDD, review, and verification.
