# External_Sources

## 1. Descriptive of Current Feature Set

- Purpose: Connect external providers to import and sync content into local media collections.
- Capabilities:
  - OAuth linking, account listing/removal
  - Browsing remote sources (Drive folders/files, OneDrive folders/files, Notion pages/databases, Gmail labels/messages)
  - Reference-manager imports for Zotero collections in v1
  - Creating sources with per-org policy enforcement; queuing import jobs
  - Source-level sync cursors, webhook state, and legacy-import rescan markers
  - Canonical remote-to-media bindings and append-only item sync events
  - Shared file-sync reconciliation for Google Drive and OneDrive:
    - bootstrap import into `Media`
    - delta sync into `DocumentVersions`
    - archive on upstream delete
    - degraded state when a new upstream revision fails ingestion
  - Manual sync status/trigger endpoints and webhook-triggered incremental sync
  - APScheduler bridge that enqueues renewal, replay, and incremental sync jobs
  - Org policy admin: allowed providers, paths, domains, quotas
  - Sync status summaries that surface duplicate and metadata-only reference-manager counts
- Inputs/Outputs:
  - Inputs: OAuth code/state, provider selection, source path/IDs, policy documents
  - Outputs: accounts, sources, sync status, import/sync job descriptors; ingested content in Media DB v2
- Related Endpoints:
  - Connectors API: `tldw_Server_API/app/api/v1/endpoints/connectors.py:46` (providers/catalog, authorize/callback, accounts, sources, jobs, policy)
  - Sync status: `tldw_Server_API/app/api/v1/endpoints/connectors.py:685`
  - Manual sync trigger: `tldw_Server_API/app/api/v1/endpoints/connectors.py:723`
  - Provider webhooks: `tldw_Server_API/app/api/v1/endpoints/connectors.py:753`
- Related Schemas:
  - `tldw_Server_API/app/api/v1/schemas/connectors.py:1`

### Reference-Manager Import Mode (v1)

- Zotero is the only shipped reference-manager provider in v1.
- Reference-manager sources are `collection` sources, not file trees.
- Collection sync is flat in v1; child collections must be linked separately.
- Import mode is one-way and non-destructive by default:
  - newly discovered upstream items can create or bind local records
  - later upstream edits do not rewrite local content or titles automatically
  - upstream deletes do not remove local items automatically
- Sync status surfaces `duplicate_count` and `metadata_only_count` so collection imports distinguish deduped records from metadata-only references.

## 2. Technical Details of Features

- Architecture & Data Flow:
  - Provider connectors implement a small interface; file-hosting providers also implement the sync adapter contract (`list_changes`, `download_or_export`, webhook subscribe/renew/revoke)
  - Reference-manager providers implement a separate collection/item/attachment adapter contract and keep bibliographic identity distinct from attachment identity
  - Service functions write connector account/source/binding state to AuthNZ DB tables via async pool
  - The shared sync coordinator reconciles remote change events into local `Media` and `DocumentVersions`
  - OAuth state is generated on authorize, stored in AuthNZ DB, and consumed once during callback
  - Endpoints enforce org-level policy in multi-user mode; single-user mode bypasses org checks
  - Webhook callbacks only validate, dedupe, and enqueue Jobs; they never perform ingestion inline
  - The recurring scheduler scans sources and enqueues `incremental_sync`, `subscription_renewal`, or `repair_rescan`
  - Reference-manager imports write canonical scholarly `safe_metadata`, persist per-item bindings, and record metadata-only rows when no retrievable attachment exists
- Key Classes/Functions:
  - Connectors service: `core/External_Sources/connectors_service.py` (DDL ensure, policy upsert/get, accounts/sources CRUD)
  - Shared reconciliation: `core/External_Sources/sync_coordinator.py`
  - Worker execution: `app/services/connectors_worker.py`
  - Recurring scheduler: `app/services/connectors_sync_scheduler.py`
  - Providers: `google_drive.py`, `onedrive.py`, `gmail.py`, `notion.py`; registry: `get_connector_by_name`
- Dependencies:
  - Internal: AuthNZ DB pool, policy helpers, Logging context
  - External: Google Drive API, Microsoft Graph, Gmail API, Notion APIs; tokens stored via DB
- Data Models & DB:
  - AuthNZ DB tables: `external_accounts`, `external_sources`, `external_items`, `external_source_sync_state`, `external_item_events`, `org_connector_policy`, `external_oauth_state` (SQLite/PG variants)
  - Reference-manager bindings live in `external_reference_items`, which stores provider item identity, collection identity, dedupe reason, and raw reference metadata
  - `external_items` is the canonical remote binding row; legacy rows without `media_id` are marked for full rescan during migration
  - `external_source_sync_state` stores cursor, webhook, retry, and active-job fence state per source
  - File content versions remain in Media DB v2 `DocumentVersions`; connector tables do not duplicate content bodies
- Configuration:
  - `CONNECTOR_REDIRECT_BASE_URL` for OAuth and webhook callbacks; required outside test mode and must be `https://` unless targeting localhost for local development
  - `CONNECTOR_OAUTH_STATE_TTL_MINUTES` to control OAuth state validity (default: 10)
  - `CONNECTORS_SYNC_SCHEDULER_ENABLED` to start recurring source scans
  - `CONNECTORS_SYNC_SCHEDULER_SCAN_SEC` to control scan cadence (default: 300)
  - `CONNECTORS_SYNC_RENEWAL_LOOKAHEAD_SEC` to renew subscriptions before expiry (default: 3600)
  - `EMAIL_GMAIL_CONNECTOR_ENABLED` to expose the Gmail provider in connector APIs
- Concurrency & Performance:
  - Import/sync jobs are queued through Jobs with source-scoped fencing and idempotent reservation
  - Pagination and cursor-based delta sync are used for large sources
- Error Handling:
  - Consistent HTTP 4xx/5xx; policy evaluation yields 403 with reason; token refresh envelope helpers covered by tests
  - Failed upstream revisions keep the last good local version active and mark the binding `degraded`
- Security:
  - RBAC by role; email/workspace validations; path/domain allow/deny lists; token handling in AuthNZ DB
  - Webhook deliveries are deduped via `external_webhook_receipts`

## 3. Developer-Related/Relevant Information for Contributors

- Folder Structure:
  - `External_Sources/` (provider adapters, policy), service helpers, endpoints under `/api/v1/endpoints/connectors.py`
- Extension Points:
  - Add a provider module and wire into `get_connector_by_name`; implement browse, list, token exchange, and sync adapter methods as needed
  - New scholarly/reference-manager providers should implement the reference-manager adapter contract instead of the file-sync contract unless they truly expose file-style deltas
- Coding Patterns:
  - Async DB via pool; structured logs; keep endpoints thin (service owns DDL and logic)
  - File-hosting providers should keep provider-specific delta/webhook details inside the adapter and let the coordinator own reconciliation semantics
- Tests:
  - `tldw_Server_API/tests/External_Sources/test_policy_and_connectors.py:1`
  - `tldw_Server_API/tests/External_Sources/test_token_refresh_envelope.py:1`
  - `tldw_Server_API/tests/External_Sources/test_connectors_worker_file_sync.py:1`
  - `tldw_Server_API/tests/External_Sources/test_reference_manager_contract.py:1`
  - `tldw_Server_API/tests/External_Sources/test_reference_manager_storage.py:1`
  - `tldw_Server_API/tests/External_Sources/test_reference_manager_dedupe.py:1`
  - `tldw_Server_API/tests/External_Sources/test_zotero_connector.py:1`
  - `tldw_Server_API/tests/External_Sources/test_connectors_webhooks.py:1`
  - `tldw_Server_API/tests/External_Sources/test_connectors_worker_reference_sync.py:1`
  - `tldw_Server_API/tests/MediaIngestion_NEW/integration/test_external_file_sync_integration.py:1`
  - `tldw_Server_API/tests/MediaIngestion_NEW/integration/test_external_reference_import_integration.py:1`
- Local Dev Tips:
  - Use TEST_MODE and mock provider modules in tests; set callback base URL via env for manual flows
- Pitfalls & Gotchas:
  - Quotas per role; pagination cursors; workspace and domain constraints
  - Legacy `external_items` rows without `media_id` must be replayed before cursor-only sync is trustworthy
  - Webhook callbacks should only enqueue work; never add ingestion logic to the request path
  - Manual sync queues a job and returns immediately; it does not run inline with the API request
- Follow-up candidates:
  - Additional file-hosting providers such as Dropbox, Box, and SharePoint
  - Richer admin/source observability beyond the current per-source sync status endpoint
