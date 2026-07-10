# Paperless-Inspired Source Saved Views Design

**Status:** Approved
**Backlog:** TASK-12093.2
**Parent PRD:** `Docs/Product/Paperless_Inspired_Document_Lifecycle_PRD.md`

## Purpose

Add immutable built-in source presets and user-managed, server-backed saved views to the Research Workspace source list. A saved view restores the existing source filters and sort order for one user in one workspace without introducing a generic query language.

## Goals

- Provide built-in presets for Needs review, Unreviewed, Failed ingest, Partially indexed, PDFs, Web captures, and Large files.
- Save and restore the current source filter and sort state after reload.
- Isolate saved views by authenticated user and workspace.
- Make unsupported or corrupt persisted payloads visible and recoverable without breaking the source list.
- Preserve the existing source-list filtering model and processing/review-state separation.

## Non-Goals

- Shared team views, default views, dashboard widgets, or cross-workspace views.
- A generic filter expression or query language.
- Browser-local persistence as the system of record.
- Auto-saving manual filter changes into an applied saved view.
- Broad source-search or workspace-permission refactors.

## Chosen Architecture

Use a dedicated workspace source-view resource. The workspace API owns typed CRUD routes, and `CharactersRAGDB` owns a cross-SQLite/PostgreSQL table and access methods. Built-in presets remain immutable TypeScript definitions and never create server rows.

The API receives the authenticated user ID explicitly even though the default SQLite layout already uses a per-user database. Persisting and querying `owner_user_id` provides defense in depth for shared PostgreSQL backends and makes the user/workspace isolation contract testable.

## Persisted Data Model

Add `workspace_source_saved_views` in both SQLite and PostgreSQL workspace schema initialization:

| Column | Contract |
| --- | --- |
| `id` | UUID text identifier |
| `workspace_id` | Workspace foreign key with cascade delete |
| `owner_user_id` | Authenticated user ID serialized as text |
| `name` | Trimmed display name, 1-120 characters |
| `name_key` | NFKC-normalized, case-folded uniqueness key |
| `schema_version` | Persisted state contract version; V1 is `1` |
| `state_json` | Canonical JSON for the V1 filter/sort state |
| `version` | Integer optimistic-lock version |
| `created_at` | ISO-8601 UTC text |
| `updated_at` | ISO-8601 UTC text |

Enforce uniqueness on `(owner_user_id, workspace_id, name_key)`. Do not rely on database-specific `LOWER()` behavior for case-insensitive names. Cap each user at 100 saved views per workspace and reject serialized state larger than 16 KiB.

## V1 State Contract

The server contract is narrower than `SourceListViewState`. It excludes the local `expanded` disclosure flag and contains only known filter and sort fields:

- `type_filters`
- `status_filters`
- `review_state_filters`
- `lifecycle_state_filters`
- `date_field`, `date_from`, and `date_to`
- `require_url`, `require_file_size`, `require_duration`, and `require_page_count`
- numeric file-size, duration, and page-count minimums/maximums
- `sort`

Pydantic models reject unknown fields on writes. String fields use existing source enums, dates use `YYYY-MM-DD`, and numeric ranges must be finite and non-negative. The client owns explicit camelCase/snake_case serialization helpers and always fills omitted optional fields from `DEFAULT_SOURCE_LIST_VIEW_STATE`.

`lifecycleStateFilters` is added to the client view state because Partially indexed cannot be represented by the existing coarse `ready | processing | error` status. It filters against `source.statusDetails.lifecycleState`, participates in active-filter detection, and appears in the collapsed summary so a preset never applies an invisible condition.

## Built-In Presets

Selecting a built-in preset replaces all persisted filter/sort fields while preserving only the local `expanded` flag.

| Preset | State |
| --- | --- |
| Needs review | `reviewStateFilters = ["needs_review"]` |
| Unreviewed | `reviewStateFilters = ["unset"]` |
| Failed ingest | `statusFilters = ["error"]` |
| Partially indexed | `lifecycleStateFilters = ["partially_queryable"]` |
| PDFs | `typeFilters = ["pdf"]` |
| Web captures | `typeFilters = ["website"]` |
| Large files | `fileSizeMin = 50 * 1024 * 1024` bytes |

The Large files threshold matches the existing 50 MiB Quick Ingest review convention.

## API Contract

Routes live under `/api/v1/workspaces/{workspace_id}/source-views` and use existing workspace read/write rate-limit dependencies.

- `GET /source-views`: list only the current user's rows for the workspace.
- `POST /source-views`: create a V1 view; return `409` for a duplicate `name_key` or a view-count limit.
- `PATCH /source-views/{view_id}`: explicitly replace name and/or state using the expected `version`.
- `DELETE /source-views/{view_id}`: delete only a row owned by the current user in the workspace.

Every route first confirms that the workspace is visible through the current user's workspace database. Saved views are personal preferences, so a user who can read the workspace may manage their own rows; no route mutates shared workspace content.

Responses include `id`, `workspace_id`, `name`, `schema_version`, `state`, `valid`, `invalid_reason`, `version`, `created_at`, and `updated_at`. Normal valid rows return typed V1 state. A corrupt JSON payload or unsupported stored schema version returns metadata with `valid = false`, `state = null`, and a stable `invalid_reason` instead of failing the list.

## Duplicate-Name Replacement

Create never silently overwrites. A duplicate name returns `409`. The UI refreshes the list if necessary, identifies the same normalized name, and asks the user to confirm replacement. Confirmation issues an explicit versioned `PATCH` to that view. A version conflict keeps the dialog open, refreshes the row, and asks the user to retry rather than discarding another tab's update.

## UI Design

Add a focused `SourceViewControls` row near search and advanced filters. It contains:

- A grouped menu for immutable built-in presets and server-backed user views.
- An icon button with tooltip for saving the current filter/sort state.
- Management actions for replacing or deleting a valid saved view.
- Warning, Reset, and Delete actions for an invalid saved view; invalid views cannot be applied.

Applying a view fully replaces persisted fields and preserves the `expanded` disclosure flag. Manual changes after applying a view mark the control as Modified but do not write to the server. Switching workspaces clears the applied selection and loads that workspace's personal views.

The save dialog trims names and reports validation errors inline. A duplicate uses the approved replacement confirmation. Network failures show a compact retryable error and do not disable built-in presets, manual filters, or the source list.

## Client Data Flow

1. Research Workspace identifies the active workspace.
2. A saved-view hook lists personal views for that workspace through the workspace API client.
3. Selecting a built-in preset applies a local canonical state.
4. Selecting a valid saved view deserializes and applies the returned V1 state.
5. Save serializes the current state, excluding `expanded`, and creates a server row.
6. Duplicate confirmation or invalid-row reset explicitly patches a known row version.
7. Delete removes the row and clears the active selection if necessary.

## Failure And Recovery Behavior

- List failure: retain ordinary filtering and built-in presets; show Retry for saved views.
- Invalid stored JSON: list the row as invalid, disable Apply, permit Reset or Delete.
- Unsupported schema version: same recovery path as invalid JSON.
- Duplicate name: no overwrite until explicit confirmation.
- Concurrent update: return `409`, refresh, and require retry.
- Deleted or inaccessible workspace/view: return `404` without leaking another user's row.
- Malformed write payload: return `422`; do not persist partial state.

Logs record identifiers and stable reason codes, never raw saved-state JSON.

## Security And Privacy

- Scope every database operation by `owner_user_id`, `workspace_id`, and view ID where applicable.
- Verify workspace visibility before view access.
- Use bound SQL parameters and canonical server-side name normalization.
- Bound names, payload size, and row count to prevent preference-storage abuse.
- Never return the existence or metadata of another user's view.

## Testing Strategy

### Pure Client Tests

- Every built-in preset maps to the expected canonical state.
- V1 serialization excludes `expanded` and round-trips all supported fields.
- Lifecycle filters participate in filtering, summaries, and reset.
- Unknown/malformed response state fails soft.

### Database Tests

- SQLite and PostgreSQL schema creation/migration.
- Create/list/update/delete behavior and optimistic locking.
- Case-insensitive normalized-name uniqueness.
- User and workspace isolation.
- View-count and payload-size limits.
- Corrupt JSON and unsupported versions remain listable and resettable.

### API Tests

- Workspace visibility, authenticated isolation, validation, duplicate `409`, and version conflicts.
- Invalid rows return recoverable response objects rather than a list failure.
- Rate-limit route contracts include all saved-view endpoints.

### UI Tests

- Apply each preset.
- Save and reload a view, then restore filters and sort.
- Confirm same-name replacement.
- Mark an applied view Modified after manual changes.
- Reset/delete an invalid view.
- Preserve built-in/manual filtering when saved-view requests fail.

## Rollout And Compatibility

The new table is additive and empty by default. Existing source-list behavior remains unchanged until a user applies a preset or saved view. Schema-version handling allows a future contract to coexist with V1 while keeping unsupported rows recoverable.

