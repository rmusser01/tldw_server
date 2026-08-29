# Stale Security PR Reconciliation Design

- **Status:** Approved
- **Date:** 2026-08-29
- **Backlog:** TASK-13013.4
- **Baseline:** `origin/dev` at `41bd5dda336c70259595ebf3ce3fb4a6a5b549db`

## Purpose

Replace five stale, release-blocking security pull requests with one current-`dev`
change set that preserves their security intent without rebasing obsolete code:

- PR #2610: weather-provider outbound egress
- PR #2614: RAG query log redaction
- PR #2622: admin impersonation auditing
- PR #2623: media-processing authorization
- PR #2625: MediaWiki request-user scoping

The replacement is one pull request containing five independently reviewable
boundary commits plus bounded review-correction commits. The stale pull
requests are recorded as superseded by the replacement and are closed only
after the replacement merges.

## Goals

1. Remove raw RAG query text from info logs.
2. Route OpenWeather calls through the central outbound policy without following
   redirects or exposing the API key through observability.
3. Require `media.create` and its RBAC rate limit on every identified media
   processing route.
4. Bind MediaWiki persistence and vector storage to the authenticated request
   user at the HTTP boundary.
5. Make admin impersonation tokens genuinely short-lived, propagate attribution
   into the authenticated principal, and fail closed if the issuance audit event
   cannot be persisted.
6. Produce current-`dev` regression evidence and a recorded disposition for all
   five stale pull requests.

## Non-goals

- Do not rebase, merge, or revive any stale PR branch.
- Do not redesign the repository-wide authorization, audit, RAG, or HTTP client
  frameworks.
- Do not redact unrelated search logs outside the three RAG endpoints named by
  PR #2614.
- Do not remove the MediaWiki core API's legacy fallback for trusted direct-call
  consumers; only the HTTP ingest route must be incapable of falling back.
- Do not install dependencies or modify system files.

## Architecture

### 1. RAG logging

Introduce one private logger helper in `rag_unified.py`. It accepts a label, a
query-like value, and an optional username. It logs only the query length and
the explicitly supplied request context. It does not log raw text or a
deterministic query hash; low-entropy search text should not become recoverable
from a log fingerprint.

Use the helper for unified, simple, and advanced search. Existing topic
monitoring continues to receive the query because it is application behavior,
not logging.

### 2. Weather egress

Replace the provider's direct `httpx.Client` call with the existing synchronous
`http_client.fetch` response API. Configure one attempt, the existing timeout,
no redirects, and sensitive observability. The central egress policy remains
the authority for allow/deny decisions; deployments that enable OpenWeather
must allow `api.openweathermap.org` through `EGRESS_ALLOWLIST`.

Map central egress/network exceptions into the provider's existing sanitized
unavailable response. No raw URL, API key, or exception text is returned.

### 3. Media-processing authorization

Add a small dependency factory in `API_Deps/media_route_deps.py` that returns
the existing `RequirePermission(MEDIA_CREATE)` and
`rbac_rate_limit("media.create")` dependencies. Apply it to the nine stale-PR
route families: audio, code, documents, ebooks, emails, MediaWiki ingest and
ephemeral processing, PDFs, videos, and web scraping.

Keep existing quota, billing, and backpressure dependencies intact. Test the
registered route dependency graph plus one representative denied request.

### 4. MediaWiki request identity

At the ingest HTTP route, resolve both `get_media_db_for_user` and
`get_request_user`. Resolve a media repository from the request-scoped database
using the repository factory's non-null return contract. Thread the writer and
the authenticated user's `id_str` through `_process_mediawiki_dump`,
`import_mediawiki_dump`, `process_single_item`, and vector storage.

Namespace resumable-import checkpoints by a non-reversible digest of the same
request user identity so one user's interrupted import cannot advance another
user's same-wiki import. Preserve the legacy unscoped checkpoint name only for
trusted direct callers that omit request identity.

The ephemeral process route continues to pass neither value and performs no
persistence. Trusted direct callers that omit the new optional arguments retain
the current managed-database and configured-single-user fallbacks.

### 5. Admin impersonation

Allow `JWTService.create_access_token` to accept an internal `timedelta`
expiration override and issue impersonation tokens with the endpoint's
15-minute TTL.

Authentication accepts impersonation attribution only as a consistent pair:
`impersonation` must be a boolean; when it is true, `impersonated_by` must be an
integer; an orphan `impersonated_by` claim is rejected. Store both values on
`AuthPrincipal` and preserve them through the legacy principal adapter.

Token issuance additionally requires an exact integer user actor and rejects
already-impersonated or actorless service principals before repository, token,
or audit work. This prevents chained impersonation from erasing the original
actor and prevents issuance of tokens that strict verification would reject.

After token creation, write a mandatory `admin.impersonation.token.create`
audit event containing actor, target, TTL, and impersonation metadata. Extend
the existing admin audit service with an opt-in fail-closed mode while leaving
all existing best-effort callers unchanged. Return a sanitized 503 when the
mandatory audit write fails.

## Testing

Every boundary follows RED-GREEN-REFACTOR. Tests must fail against the current
`dev` baseline for the intended missing protection before production changes.

- RAG: captured logs contain length/context but never the supplied secret query
  or a deterministic fingerprint.
- Weather: central fetch is used; redirects are false; sensitive observability is
  true; allow and deny outcomes are sanitized; the API key never appears.
- Media authorization: all registered target routes include permission and rate
  limit dependencies; a principal without `media.create` receives 403.
- MediaWiki: HTTP ingest injects the request-scoped writer and vector user;
  checkpoints are user-isolated; direct-call fallback and ephemeral behavior
  remain.
- Impersonation: exact TTL, strict paired claims, principal propagation,
  unambiguous issuer identity, mandatory audit payload, and audit failure 503.

Run the focused modules, touched-file Ruff, touched-code Bandit, Python compile,
and `git diff --check`. CI remains the authority for the full required gate set.

## Delivery and dispositions

The branch contains five security boundary commits, bounded review corrections,
and documentation/task closeout. The replacement PR body lists each stale PR
and its supersession reason. Once the replacement merges:

1. comment on each stale PR with the replacement PR and merge commit;
2. close each stale PR without merging;
3. record those URLs and dispositions on TASK-13013.4;
4. mark acceptance criteria and definition of done only after evidence is
   complete.

## Risk controls

- The two changes to `process_mediawiki.py` (authorization and identity) are
  developed sequentially and verified together.
- Existing best-effort audit callers retain their current behavior because
  fail-closed persistence is opt-in.
- Weather redirects are disabled so an API key cannot be forwarded to another
  host.
- No raw query fingerprint is retained, avoiding dictionary recovery of common
  searches.
