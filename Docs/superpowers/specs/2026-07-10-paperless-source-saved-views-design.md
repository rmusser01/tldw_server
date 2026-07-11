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

- Shared team views, personal views while accessing another owner's shared workspace, default views, dashboard widgets, or cross-workspace views.
- A generic filter expression or query language.
- Browser-local persistence as the system of record.
- Auto-saving manual filter changes into an applied saved view.
- Broad source-search or workspace-permission refactors.

## Chosen Architecture

Use a dedicated workspace source-view resource. The workspace API owns typed CRUD routes, and `CharactersRAGDB` owns a cross-SQLite/PostgreSQL table and access methods. Built-in presets remain immutable TypeScript definitions and never create server rows.

The API receives the authenticated user ID explicitly even though the default SQLite layout already uses a per-user database. Persisting and querying `owner_user_id` provides defense in depth for shared PostgreSQL backends and makes the user/workspace isolation contract testable.

V1 is available only for an active workspace in the authenticated user's primary ChaCha store, where `workspaces.client_id` equals the authenticated user ID. Existing shared-workspace reads use a separate owner-store path; extending personal saved views into that path requires a later sharing-specific authorization design and is not inferred from mere knowledge of a workspace ID.

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

Pydantic models reject unknown fields on writes. String fields use existing source enums, dates use real `YYYY-MM-DD` calendar dates, and numeric ranges must be finite and non-negative with each minimum less than or equal to its maximum. The client owns explicit camelCase/snake_case serialization helpers and always fills omitted optional fields from `DEFAULT_SOURCE_LIST_VIEW_STATE`. Client serialization returns an explicit validation result; invalid local dates or numeric ranges block the request and identify the affected field inline rather than falling through to a generic server `422`.

`lifecycleStateFilters` is added to the client view state because Partially indexed cannot be represented by the existing coarse `ready | processing | error` status. It filters against `source.statusDetails.lifecycleState`, participates in active-filter detection, and remains visible in both collapsed and expanded Advanced states through an accessible removable summary/chip so a preset never applies an invisible condition.

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

Every route first confirms that the workspace is active and owned by the current user's primary ChaCha store (`workspaces.client_id = current user`). Knowledge of another tenant's workspace ID never grants access. Saved views are personal preferences and no route mutates shared workspace content.

Responses include `id`, `workspace_id`, `name`, `schema_version`, `state`, `valid`, `invalid_reason`, `version`, `created_at`, and `updated_at`. Normal valid rows return typed V1 state. A corrupt JSON payload or unsupported stored schema version returns metadata with `valid = false`, `state = null`, and a stable `invalid_reason` instead of failing the list.

Saved views are ordered deterministically by `updated_at DESC`, then `name_key ASC`, then `id ASC`. Valid V1 rows are canonicalized server-side: enum arrays are deduplicated and ordered by declaration order, omitted fields receive defaults, and one deterministic UTF-8 JSON encoding is used for persistence and the 16 KiB byte limit. Response invariants are strict: valid rows have non-null state and null reason; invalid rows have null state and one stable reason. Schema-version support is checked before JSON parsing, so an unsupported version takes precedence over malformed state JSON.

Conflict responses use a structured `detail` object with a stable `code`:

- `source_view_name_exists`: includes the current user's conflicting `view_id` and `version`, allowing explicit replacement without duplicating server name normalization in the client.
- `source_view_limit_reached`: includes the configured per-workspace limit.
- `source_view_version_conflict`: includes the requested view ID and current version.

The API never includes metadata for a view outside the authenticated user and workspace scope.

Delete is intentionally unconditional after user/workspace ownership checks; it does not use optimistic version matching. A second delete returns `404`. This keeps the bodyless `DELETE` contract simple while PATCH remains versioned for state/name replacement.

## Duplicate-Name Replacement

Create never silently overwrites. A duplicate name returns `409` with `code = source_view_name_exists` and the conflicting owned view's ID and version. The UI uses that server metadata and asks the user to confirm replacement; it does not attempt to reproduce Unicode normalization. Confirmation issues an explicit versioned `PATCH` to that view. A version conflict keeps the dialog open, refreshes the row, and asks the user to retry rather than discarding another tab's update.

## UI Design

Add a focused `SourceViewControls` row near search and advanced filters. It contains:

- A grouped menu for immutable built-in presets and server-backed user views.
- An icon button with tooltip and accessible name for saving the current filter/sort state.
- Management actions for replacing or deleting a valid saved view.
- Warning, Reset, and Delete actions for an invalid saved view; invalid views cannot be applied.

Applying a view fully replaces persisted fields and preserves the `expanded` disclosure flag. Manual changes after applying a view mark the control as Modified but do not write to the server. The saved-view hook is instantiated once at the Research Workspace page state boundary and its controller is passed to any desktop/drawer pane instances, preventing duplicate active/conflict state. The page renders exactly one saved-view overlay host; repeated pane controls render triggers/menus only, so shared confirmation state cannot create duplicate modal portals. The overlay host captures the controller generation and actual invoking element when opened. Any generation change synchronously closes the overlay, discards drafts/confirmation state, and clears the invoker; submit also verifies the captured generation. On ordinary close, focus returns to the original element only if it remains connected and focusable, otherwise to the visible saved-view trigger or Sources pane landmark.

The controller exposes an explicit availability state. A null workspace ID performs no request and synchronously clears loaded rows, active snapshots, conflicts, limits, errors, announcements, and mutation state while invalidating all pending completions. Save and server-view management are disabled with an accessible "Select a workspace" explanation, while built-in presets remain usable. Every workspace identity change increments a request generation; completions must match the captured generation, not merely the current ID, so an old A response cannot be accepted after A to B to A.

The control uses the existing accessible menu/dialog primitives: arrow keys move through menu items, Enter/Space applies or invokes an action, Escape closes without changes, dialogs trap focus, and closing restores focus to the invoking control. Every icon-only action has an `aria-label` in addition to its tooltip.

The save dialog trims names and reports validation errors inline. A duplicate uses the approved replacement confirmation. Reaching the saved-view limit shows non-retryable guidance with the server-provided limit and directs the user to delete an existing view. Network failures show a compact retryable error and do not disable built-in presets, manual filters, or the source list. Async errors use an alert/live region, mutations expose an accessible busy state, and successful create/replace/reset/delete operations produce concise polite status announcements.

Successful create or replacement marks the returned view active and unmodified; applying it again is unnecessary because its canonical state equals the current state. Resetting an invalid view writes `schema_version = 1` with the canonical default filter/sort state, preserves its name, increments its version, marks it active, and immediately applies that default state. Deleting the active view clears the active selection without changing current filters.

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
- Invalid stored JSON: list the row as invalid with `invalid_reason = invalid_json`, disable Apply, permit Reset or Delete.
- Valid JSON that fails the V1 state model: list it as invalid with `invalid_reason = invalid_state`, using the same recovery path.
- Unsupported schema version: list it as invalid with `invalid_reason = unsupported_schema_version`, using the same recovery path.
- Duplicate name: no overwrite until explicit confirmation.
- Concurrent update: return `409`, refresh, and require retry.
- Saved-view limit: show the limit and deletion guidance without a Retry action that cannot succeed.
- No active workspace: clear personal-view state, disable Save/manage actions with accessible guidance, and leave built-in presets enabled.
- Deleted or inaccessible workspace/view: return `404` without leaking another user's row.
- Malformed write payload: return `422`; do not persist partial state.

Logs record identifiers and stable reason codes, never raw saved-state JSON.

## Security And Privacy

- Scope every database operation by `owner_user_id`, `workspace_id`, and view ID where applicable.
- Verify an active workspace row with `workspaces.client_id = owner_user_id` before every view operation; every mutation locks that row inside the transaction.
- Use bound SQL parameters and canonical server-side name normalization.
- Bound names, payload size, and row count to prevent preference-storage abuse.
- Never return the existence or metadata of another user's view.
- Extend `build_chacha_rls_sql()` for PostgreSQL: enable and force RLS on `workspace_source_saved_views`, and create a tenant policy whose `USING` and `WITH CHECK` clauses require both `owner_user_id = current_setting('app.current_user_id', true)` and an active referenced workspace whose `client_id` equals the same setting. The PostgreSQL table-creation/migration helper applies this policy in the same initialization transaction; the general startup policy builder safely no-ops when the table does not yet exist.

Normal workspace deletion is soft deletion. Saved-view rows are retained for possible workspace recovery but become inaccessible immediately because every API/DB access and the PostgreSQL policy require `workspaces.deleted = false`. A later physical workspace deletion uses the foreign-key cascade. Create/update/delete races serialize on the workspace row.

## Testing Strategy

### Pure Client Tests

- Every built-in preset maps to the expected canonical state.
- V1 serialization excludes `expanded` and round-trips all supported fields.
- Invalid local dates, negative/non-finite values, and inverted ranges block saving with field-specific validation.
- Lifecycle filters participate in filtering, summaries, and reset.
- Lifecycle filters remain visible and keyboard-removable in collapsed and expanded Advanced states.
- Unknown/malformed response state fails soft.

### Database Tests

- SQLite and PostgreSQL schema creation/migration.
- Create/list/update/delete behavior and optimistic locking.
- Case-insensitive normalized-name uniqueness.
- User and workspace isolation.
- Active workspace ownership checks, soft-delete inaccessibility, hard-delete cascade, and mutation/delete races.
- View-count and payload-size limits.
- Corrupt JSON and unsupported versions remain listable and resettable.
- Valid JSON with invalid V1 fields remains listable and resettable.
- PostgreSQL fresh creation and V52 migration install forced RLS immediately, with owner-and-workspace-scoped `USING` and `WITH CHECK` clauses.
- Concurrent create and rename-to-duplicate races return safe duplicate metadata after rollback.

### API Tests

- Active workspace ownership, authenticated isolation, strict top-level/state validation, duplicate `409`, and version conflicts.
- Canonical full valid responses, invalid response invariants, and unsupported-version precedence.
- Invalid rows return recoverable response objects rather than a list failure.
- Rate-limit route contracts include all saved-view endpoints.

### UI Tests

- Apply each preset.
- Save and reload a view, then restore filters and sort.
- Confirm same-name replacement.
- Mark an applied view Modified after manual changes.
- Reset/delete an invalid view.
- Preserve built-in/manual filtering when saved-view requests fail.
- Render saved-view-limit guidance without a futile Retry action.
- Share one saved-view controller across simultaneous responsive pane instances, render exactly one overlay portal with invoker focus restoration, and suppress requests for a null workspace.
- Clear all controller state on non-null to null and reject stale completions across null and A to B to A generation changes.
- Close and invalidate open Save/Replace overlays on every workspace generation change, preventing stale drafts from submitting into a new workspace and exercising focus fallback when the invoker unmounts.
- Operate the view menu, save action, confirmation, reset, and delete flows by keyboard with accessible names and focus restoration.

## Rollout And Compatibility

The new table is additive and empty by default. Existing source-list behavior remains unchanged until a user applies a preset or saved view. Schema-version handling allows a future contract to coexist with V1 while keeping unsupported rows recoverable.

The Unreviewed preset is retained because Child Task 1 already shipped it with the review lifecycle. Child Task 2 adds the other required presets and includes all seven in the unified view menu; retaining the existing preset avoids a regression rather than expanding this task's scope.
