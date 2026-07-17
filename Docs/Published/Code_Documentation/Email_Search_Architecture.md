# Email Search Architecture

Last Updated: 2026-02-10

## Purpose

This document describes the normalized email search architecture in `tldw_Server_API`:

1. Data model and table relationships.
2. Query parser and planner flow.
3. API delegation behavior.
4. Extension guidelines.

## Key Components

### Storage and Search Engine

- Core DB package seam: `tldw_Server_API/app/core/DB_Management/media_db/native_class.py`
- Primary normalized tables:
  - `email_sources`
  - `email_messages`
  - `email_participants`
  - `email_message_participants`
  - `email_labels`
  - `email_message_labels`
  - `email_attachments`
  - `email_sync_state`
  - `email_backfill_state`
- SQLite full-text index:
  - `email_fts` on message search text fields.

### API Surfaces

- Email-native routes:
  - `tldw_Server_API/app/api/v1/endpoints/email.py`
- Media compatibility bridge:
  - `tldw_Server_API/app/api/v1/endpoints/media/listing.py`

## Data Model and Query Path

### Canonical Message Row

`email_messages` is the canonical normalized message record and links to `Media` through `media_id`.

Message detail fanout:

1. Participants via `email_message_participants -> email_participants`.
2. Labels via `email_message_labels -> email_labels`.
3. Attachments via `email_attachments`.

### Core Indexing

Important indexes for search and planner efficiency include:

1. `idx_email_messages_tenant_date_id`
2. `idx_email_messages_tenant_has_attachments_date`
3. `idx_email_message_participants_message_role`
4. `idx_email_participants_tenant_email`
5. `idx_email_labels_tenant_name`

These indexes support tenant-scoped sorting/filtering and role/label join predicates.

## Parser and Planner Flow

### Query Parsing

Parser entrypoint:

- `MediaDatabase._parse_email_operator_query(...)`

Supported operator groups:

1. Participant fields: `from:`, `to:`, `cc:`, `bcc:`
2. Message fields: `subject:`, `label:`
3. Boolean-like unary: `has:attachment`
4. Date windows: `before:`, `after:`, `older_than:`, `newer_than:`
5. Free text and phrase tokens.

Not supported in v1:

1. Parentheses.

### SQL Planning

Planner/executor entrypoint:

- `MediaDatabase.search_email_messages(...)`

Execution outline:

1. Resolve tenant scope.
2. Parse operator tokens into OR groups of AND terms.
3. Build SQL predicates with EXISTS subqueries for labels/participants/attachments.
4. Use FTS fallback for free-text terms on SQLite.
5. Execute count + page query with deterministic sort (`internal_date DESC, id DESC`).

## API Delegation Strategy

`/api/v1/media/search` can delegate to normalized planner based on:

1. Request `email_query_mode`:
   - `operators` forces delegation (with strict validation).
   - `legacy` forces old media planner path.
2. Default cutover mode (`EMAIL_MEDIA_SEARCH_DELEGATION_MODE`):
   - `opt_in` (default)
   - `auto_email` (delegates email-only scope automatically when enabled)
3. Feature gate:
   - `EMAIL_OPERATOR_SEARCH_ENABLED`

Reference implementation:

- `tldw_Server_API/app/api/v1/endpoints/media/listing.py`
- `tldw_Server_API/app/api/v1/schemas/media_request_models.py`

## Retention and Deletion Model

Tenant-safe cleanup is provided by:

1. `MediaDatabase.enforce_email_retention_policy(...)`
2. `MediaDatabase.hard_delete_email_tenant_data(...)`

Behavior:

1. Candidate selection is tenant-scoped from normalized tables.
2. Deletion is applied via linked `Media` records to preserve consistency.
3. Orphan label/participant/source cleanup runs post-delete for target tenant.

## Extension Guidelines

### Adding New Operators

When adding an operator:

1. Extend `_parse_email_operator_query(...)` with explicit token normalization.
2. Add planner SQL branch in `search_email_messages(...)`.
3. Add unit tests in `tldw_Server_API/tests/DB_Management/test_email_native_stage1.py`.
4. Add endpoint/integration tests in `tldw_Server_API/tests/MediaIngestion_NEW/integration/`.
5. Update user guide in `Docs/User_Guides/Server/Email_Operator_Search_Guide.md`.

### Adding New Searchable Fields

1. Add schema column(s) on `email_messages` or related normalized table.
2. Update ingest/upsert path (`upsert_email_message_graph`) to populate value.
3. Update `email_fts` synchronization path if field should participate in free text.
4. Add index(es) for expected query pattern before enabling in planner.
5. Extend benchmark/parity fixtures to prevent silent regressions.

### Performance and Safety

1. Keep all joins tenant-scoped.
2. Keep deterministic ordering for stable pagination.
3. Benchmark with `Helper_Scripts/benchmarks/email_search_bench.py`.
4. Validate legacy parity with `Helper_Scripts/checks/email_search_dual_read_parity.py`.
