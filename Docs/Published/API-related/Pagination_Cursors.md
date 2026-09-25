# Pagination Cursors

List endpoints that return a `next_cursor` (or a `Next-Cursor` header) expect the client
to echo it back unchanged. Cursors are opaque: do not parse, build, or edit them.

## Malformed cursor contract

A cursor the server cannot use is rejected with:

- **Status:** `400 Bad Request`
- **Detail:** `"Invalid cursor"`

"Cannot use" covers every case the same way: not base64url, oversized, not the expected
JSON shape, a failed signature, or a signed cursor bound to a different owner or query.
The server never silently ignores a bad cursor and restarts from the first page.

To recover, drop the `cursor` parameter and start again from the first page.

Endpoints on this contract:

| Endpoint | Cursor |
| --- | --- |
| `GET /api/v1/workflows/runs` | `cursor` query parameter |
| `GET /api/v1/workflows/runs/{run_id}/events` | `cursor` query parameter |
| `GET /api/v1/audio/history` (TTS history) | `cursor` query parameter |
| `GET /api/v1/audio/jobs/admin/list` | `cursor` query parameter |
| `GET /api/v1/notes/{note_id}/attachments/canonical` | `cursor` query parameter (signed) |

Notes graph suggestion lists return the same `400` with their structured error body,
`code: "notes_graph_cursor_invalid"`.

The Sync v2 pull token is a protocol token with its own error codes
(`sync_pull_token_invalid`, `sync_pull_token_too_large`) and is not covered here.

## Server side

Opaque cursors are decoded with `decode_opaque_cursor_segment`; signed cursors are read
with `verify_signed_token`, which requires the key and checks the HMAC before returning
the payload. Both live in `tldw_Server_API/app/core/Utils/base64url.py`. Map any
`ValueError` from them to the 400 above.
