# Email Processing API

This guide documents email support in the media API. It covers the processing-only endpoint (`/process-emails`) and how to ingest emails via `/media/add` for persistence.

## Overview

- Supported format: `.eml` (RFC 822). Attachments are enumerated in metadata; attached `.eml` files can be parsed recursively when enabled.
- The parser extracts plain-text and HTML (HTML is converted to text if no plain-text part exists), headers, addresses, and attachments metadata. No raw attachment bytes are stored.

## Deprecation Headers

Email processing endpoints may emit additive HTTP deprecation metadata when a
contract change is planned. Clients should be prepared to read `Deprecation`,
`Sunset`, and `Link` headers and treat them as advisory migration signals.

## Processing Only

- Endpoint: `POST /api/v1/media/process-emails`
- Purpose: Extract content and metadata from uploaded `.eml` files, optionally chunk and analyze, without writing to the database.

Request (multipart/form-data)
- Field `files`: one or more `.eml` files. When `accept_archives=true` is provided, a `.zip` archive of `.eml` files is also accepted. When `accept_mbox=true` is provided, a `.mbox` mailbox is accepted. When `accept_pst=true` is provided, a `.pst`/`.ost` container is accepted (feature flag; parsing requires external tools).
- Options (form fields):
  - `perform_chunking`: bool (default true)
  - `chunk_method`: one of `sentences|paragraphs|…` (default `sentences`)
  - `chunk_size`: int (default 1000)
  - `chunk_overlap`: int (default 200)
  - `accept_archives`: bool (default false). When true and a `.zip` file is uploaded, member `.eml` files are expanded and processed as individual results.
  - `accept_mbox`: bool (default false). When true and a `.mbox` file is uploaded, messages are expanded and processed as individual results.
  - `accept_pst`: bool (default false). When true and a `.pst`/`.ost` file is uploaded, the request is accepted under a feature flag. When `pypff`/`libpff` is installed, messages are expanded and processed as individual results; otherwise the API returns an informative error.
  - `ingest_attachments`: bool (default false). When true, recursively parse attached `.eml` files and include them in the response under `children` (still no DB persistence in this endpoint).
  - `max_depth`: int (default 2)
  - `title`, `author`, `keywords`, `custom_prompt`, `system_prompt`, `perform_analysis`, `api_name`, `summarize_recursively` (optional analysis controls)

Response (200 or 207)
- `results`: list of per-file result objects. Each includes:
  - `status`: `Success|Warning|Error`
  - `media_type`: `email`
  - `metadata`: includes `title`, `author`, `filename`, `parser_used`, and `email` object with fields:
    - `from`, `to`, `cc`, `bcc`, `subject`, `date`, `message_id`, `format`, `attachments` (name, content_type, size, content_id, disposition), `headers_map`
  - `content`: extracted plain text
  - `chunks`: list when chunking is enabled
  - `children`: present when `ingest_attachments=true` and attached `.eml` are parsed

Examples (curl)
```
curl -X POST \
  -H "Authorization: Bearer $API_BEARER" \
  -F "files=@/path/to/email.eml;type=message/rfc822" \
  -F "perform_chunking=true" \
  -F "ingest_attachments=true" \
  -F "max_depth=2" \
  http://127.0.0.1:8000/api/v1/media/process-emails
```

ZIP of EMLs:

```
curl -X POST \
  -H "Authorization: Bearer $API_BEARER" \
  -F "files=@/path/to/emails.zip;type=application/zip" \
  -F "accept_archives=true" \
  -F "perform_chunking=true" \
  http://127.0.0.1:8000/api/v1/media/process-emails
```

MBOX mailbox:

```
curl -X POST \
  -H "Authorization: Bearer $API_BEARER" \
  -F "files=@/path/to/emails.mbox;type=application/mbox" \
  -F "accept_mbox=true" \
  -F "perform_chunking=true" \
  http://127.0.0.1:8000/api/v1/media/process-emails
```

PST/OST container (feature-flag; accepted with informative error until configured):

```
curl -X POST \
  -H "Authorization: Bearer $API_BEARER" \
  -F "files=@/path/to/mail.pst;type=application/octet-stream" \
  -F "accept_pst=true" \
  -F "perform_chunking=true" \
  http://127.0.0.1:8000/api/v1/media/process-emails
```

## Ingest and Persist

- Endpoint: `POST /api/v1/media/add`
- Behavior: When `media_type=email`, the uploaded `.eml` is processed via the email parser and persisted to the database with versioning, keywords, and chunk metadata.

Parsed `email` metadata (headers, participants, Message-ID, dates and attachment
descriptors) is retained in safe document-version metadata for primary and child
messages. Binary attachment content is not extracted into the search index.

Email dedupe is scoped to tenant, provider and source. Provider message ID takes
precedence over RFC Message-ID; body hash is used only when both IDs are absent.
Distinct messages with identical bodies remain separate. Stored Media URLs use an
opaque `email://identity/` key; source identity is retained in metadata. Upload source
names still derive from filenames/container-member references, so renaming a source
can create a separate import. Existing rows are reused when matching identity
evidence exists; already overwritten historical messages are not automatically
reconstructed.

`overwrite_existing=false` retains saved body and metadata in both Media and email
search. Enabling overwrite replaces both consistently and saves metadata in the
new document version. Reimport also preserves existing normalized metadata when a
legacy document version lacks parsed email fields.

For explicitly model-free file ingestion, set `perform_analysis=false`,
`perform_claims_extraction=false`, `perform_chunking=false`,
`auto_chunking_use_llm=false`, and `generate_embeddings=false`. Analysis otherwise
defaults on and claims can inherit server settings. See the synthetic validation
record for the tested call guards and deployment limitations.

### Search pagination

`GET /api/v1/email/search` retains `limit`/`offset` behavior when `cursor` is omitted.
Start cursor pagination with `cursor=`; pass the returned `next_cursor` on subsequent
requests with the same query and deleted visibility. `next_cursor=null` ends the
traversal. Nonzero offset and malformed or incompatible cursors return an input
error. Cursor responses include the total matching count across the search scope.

Results order by UTC internal date descending (undated messages last), then email
message ID descending. Relative-date queries keep the initial traversal clock.
This is a live traversal: newly inserted older messages may appear, and edits to
existing sort dates may move rows across page boundaries. See
`Docs/Design/email-search-cursor-pagination.md` for the full contract.

Form fields (in addition to common `/media/add` fields)
- `media_type`: `email`
- `accept_archives`: bool (default false). When true, allows uploading a `.zip` of `.eml` files; each child email is expanded and processed.
- `accept_mbox`: bool (default false). When true, allows uploading a `.mbox` mailbox; each message is expanded and processed.
    - `accept_pst`: bool (default false). When true, allows uploading a `.pst`/`.ost` container. With `pypff`/`libpff` installed, messages are expanded and processed; otherwise, the API returns an informative error.
- `ingest_attachments`: bool (default false). When true, attached `.eml` files are parsed recursively and added as separate child media items.
- `max_depth`: int (default 2) - recursion depth for child `.eml` parsing.

Chunking defaults for emails
- If not overridden by the client, `/media/add` aligns chunk size to 1000 and overlap to 150 for emails.
- The processing-only endpoint defaults overlap to 200; this is by design to keep `/add` email behavior slightly more compact.

Parent/child linkage and grouping
- When `ingest_attachments=true` creates children, each child is persisted with a `parent_media_uuid` in safe metadata.
- A grouping keyword `email_group:<Message-ID>` is applied to the parent (and inherited by children via keyword set) to make UI grouping easy.
- The `/media/add` response for the parent includes `child_db_results` summarizing each child’s DB insert (`db_id`, `media_uuid`, `title`).

ZIP archives behavior
- When `accept_archives=true` with a `.zip` upload, each member `.eml` is processed as a separate child email.
- Each child result includes a grouping keyword `email_archive:<zip_file_stem>`.
- The synthetic parent (the ZIP) is not persisted; the response contains `children` (results) and `child_db_results` (DB insert summaries for children).

MBOX behavior
- When `accept_mbox=true` with a `.mbox` upload, each message is processed as a separate child email.
- Each child result includes a grouping keyword `email_mbox:<mbox_file_stem>`.
- The synthetic parent (the MBOX) is not persisted; the response contains `children` (results) and `child_db_results` (DB insert summaries for children).

PST/OST behavior (feature-flag)
- When `accept_pst=true` with a `.pst`/`.ost` upload, the request is accepted under a feature flag.
- With `pypff` (libpff) installed, messages are expanded and processed like MBOX, with guardrails and grouping `email_pst:<pst_file_stem>`.
- Without `pypff`, the API returns a single informative error result; grouping keyword is still included for filtering.

Limitations
- Only `.eml` is supported in this version. `.msg` and `.p7m` are not yet enabled via the API.
- `.zip` uploads are only accepted when `accept_archives=true` and only member `.eml` files are processed (others ignored). Archive limits (max files and uncompressed size) are enforced.
- `.mbox` uploads are only accepted when `accept_mbox=true`; message count and size limits are enforced.
- `.pst`/`.ost` uploads are only accepted when `accept_pst=true`; full parsing requires `pypff`/`libpff`.
- Attachments other than `.eml` are not ingested as separate media by default (metadata only).
- HTML is converted to text using `html2text` when available; otherwise a simple tag strip is used.

Future formats (planned)
- PST/OST: enhanced parsing options (attachments, recipients) and optional `readpst` integration.

## Testing
- All tests: `python -m pytest -v`
- Email tests only: `python -m pytest -v -k "email"`
- Coverage: `python -m pytest --cov=tldw_Server_API --cov-report=term-missing`
