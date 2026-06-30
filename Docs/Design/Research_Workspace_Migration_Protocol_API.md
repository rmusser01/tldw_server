# Research Workspace Migration Protocol API

## Context

Research Workspace needs a durable backend protocol for migrating local WebUI/browser-extension workspace state into first-class server workspace records. This is separate from the existing `/research` feature. The protocol must not add `/workspace-playground` aliases or redirects.

## Goals

- Provide first-class migration sessions under `/api/v1/workspaces/migrations`.
- Persist migration manifests and per-chunk receipts in the user's ChaChaNotes DB.
- Make migration creation and chunk receipt uploads idempotent.
- Bound chunk payload sizes and recovery diagnostics.
- Return a durable recovery manifest that tells the WebUI what was accepted, what is pending, and whether local deletion is safe.
- Keep client/browser-storage deletion disabled in this slice.

## Non-Goals

- No `/workspace-playground` route aliases, redirects, or compatibility endpoints.
- No automatic deletion of local browser storage.
- No full source ingestion/indexing worker implementation in this slice.
- No changes to `/research`.

## API Shape

- `POST /api/v1/workspaces/migrations`
  Creates or returns an idempotent migration session for a target Research Workspace.

- `GET /api/v1/workspaces/migrations/{migration_id}`
  Returns the current durable session, accepted chunk receipts, recovery manifest, and delete eligibility.

- `PUT /api/v1/workspaces/migrations/{migration_id}/chunks/{chunk_id}`
  Records an accepted chunk receipt. Repeating the same hash and byte count is idempotent. Reusing a chunk id with different content is a conflict.

- `POST /api/v1/workspaces/migrations/{migration_id}/finalize`
  Finalizes only when all declared chunks are accepted. It performs server read-back verification against the persisted declarations and accepted receipts. It returns a recovery manifest and sets `client_delete_eligible=true` only when every declared chunk has a matching accepted receipt and at least one chunk was declared.

- `POST /api/v1/workspaces/migrations/{migration_id}/client-delete-ack`
  Records explicit client acknowledgement only after the migration is finalized, server read-back verification succeeded, `client_delete_eligible=true`, and the acknowledged manifest hash matches the finalized session.

## Persistence

ChaChaNotes owns the protocol state because Research Workspace membership, notes, artifacts, and workspace chat state already live there. The implementation adds:

- `workspace_migration_sessions`
- `workspace_migration_chunks`

Sessions store bounded manifest metadata, diagnostic metadata, target workspace identity, status, timestamps, and finalized receipt JSON. Chunks store declared byte count, SHA-256, accepted timestamp, and optional bounded summary metadata.

## Limits

- Manifest JSON: 256 KiB.
- Diagnostics JSON: 64 KiB.
- Chunk metadata JSON: 64 KiB.
- Individual chunk byte count: 2 MiB.
- Declared chunks per migration: 512.

These limits are intentionally conservative for the browser-storage migration handshake. Actual document/media ingestion should use Media and Jobs APIs rather than sending large source bodies through this protocol.

## Jobs Ownership

This API owns migration session and receipt state only. User-visible ingestion, extraction, chunking, and indexing should be represented by Jobs when source content is later imported into Media DB and RAG indexes. The migration recovery manifest exposes source counts and accepted chunk state so a follow-up can enqueue Jobs without hiding progress from the user.

## Error Handling

- Missing migration: `404`.
- Conflicting idempotency key, manifest hash, or duplicate chunk content: `409`.
- Oversized manifest/chunk/diagnostics or invalid hash: `422`.
- Premature finalize: `409` with missing chunk ids.
- Client delete acknowledgement before eligibility, before finalize, or with a mismatched manifest hash: `409`.
- Finalized zero-chunk or failed read-back verification sessions remain recoverable with `client_delete_eligible=false`.

## Route Ordering

The migration router must be registered before the dynamic `/{workspace_id}` workspace route or mounted as a separate router under the same `/api/v1/workspaces` prefix before the existing workspace router. This prevents `/migrations` from being parsed as a workspace id.
