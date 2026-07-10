# Quick Ingest Advanced Transport Fix

## Problem

In an advanced WebUI deployment, `resolveBrowserRequestTransport` can build a
valid absolute API URL from `NEXT_PUBLIC_API_URL`. `tldwRequest` nevertheless
rejects the request when persisted `tldwConfig.serverUrl` is empty. Quick
Ingest uses this direct request path for multipart submission, so the job never
reaches `/api/v1/media/ingest/jobs`.

The resulting HTTP 400 text is also classified as an unsupported input format,
which hides the actual server-configuration problem.

## Design

Keep transport resolution as the single source of truth. For non-absolute,
advanced requests, reject only when the resolved transport URL has no valid
HTTP origin. This matches the existing direct streaming behavior and preserves
the current precedence order: persisted `serverUrl`, then
`NEXT_PUBLIC_API_URL`, with hosted and quickstart modes unchanged.

Add a dedicated configuration error category using the existing `auth`
classification value. It will present a server-configuration message before
the generic HTTP 400 format rule is considered.

No new transport type or abstraction is needed.

## Tests

1. A direct multipart upload with no persisted `serverUrl` uses the advanced
   origin from `NEXT_PUBLIC_API_URL`.
2. An advanced request with neither a persisted nor runtime API origin remains
   rejected before `fetch`.
3. `tldw server not configured` maps to a configuration message rather than a
   file-format message.
4. Existing focused service tests and frontend typecheck remain green.

## Scope

The patch changes only shared frontend request validation, Quick Ingest error
classification, and their focused tests. Backend ingestion and dependency
versions are outside this patch because UAT proved ingestion succeeds with a
current `yt-dlp` installation.
