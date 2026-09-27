# Email search cursor pagination

Task: TASK-13253. Adds bounded keyset pagination to the existing email query helper
and `GET /api/v1/email/search`, without changing default offset clients.

## Request and response

Omit `cursor` to retain the existing offset request and response. Send
`?cursor=&limit=50` for the first cursor page, then repeat the same `q` with
`cursor=<next_cursor>`. Page size may change between requests. A nonzero offset
with a cursor returns HTTP 400. Invalid or incompatible tokens return HTTP 400;
values over the HTTP parameter's 4096-character limit return HTTP 422.

Cursor responses contain `items`, `pagination` with `mode: "cursor"`, `limit`,
`total`, `has_more`, and `next_cursor`, plus top-level `has_more` and `next_cursor`.
An exhausted traversal has `next_cursor: null` and `has_more: false`.
`total` counts the full matching query at the time of each request, including rows
before the current position. Cursor responses do not contain offset metadata.

The database method retains `(rows, total)` with `cursor=None`; cursor mode
returns `(rows, total, next_cursor)`. One extra row determines whether another
page exists, including when the requested page size is the maximum 500.

## Ordering and scope

Cursor mode orders by `internal_date DESC NULLS LAST, email_message_id DESC`.
The next page seeks strictly below both position values, handling the transition
from dated to undated messages and subsequent undated pages explicitly. New
messages sorting before the cursor cannot cause repeated or skipped existing
rows. This is a live traversal, not a snapshot: inserts below the position can
appear, deleted rows disappear, and changing an existing message's date can
move it across the position. Start a new traversal to see newer messages.

Versioned URL-safe tokens carry the position, initial reference time, and a
SHA-256 fingerprint of resolved tenant, stripped query text, and deleted-row
visibility. Relative date operators use that reference time across all pages.
Tokens are unsigned position hints, not credentials or integrity guarantees.
Every database query independently reapplies tenant, search, deletion and trash
visibility conditions; a token cannot grant access to another tenant's data.

## Validation

Dedicated real SQLite database and HTTP tests cover ties, null dates, inserts,
scope and query mismatches, malformed payloads, relative query windows, deletion
visibility, empty results and unchanged offset compatibility. Property tests vary
insertion order and page size. Tests use only synthetic messages and an outbound
socket guard. Live PostgreSQL validation is outside this offline test run.
