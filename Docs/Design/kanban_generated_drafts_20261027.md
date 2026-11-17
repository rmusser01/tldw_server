# Generated Kanban drafts

Task: TASK-13260.278.18.83.45. Implements the explicitly requested missing S03 Kanban generation capability. Presentation Studio is closed; Kanban precedes Document Workspace. User authorization covers the necessary implementation, diagnostics and validation.

## Behavior

Add one Generate board dialog to the existing Kanban workspace. Use the existing model picker and canonical provider-qualified selection, current sampling/token settings and captured account/server request authority. One foreground Chat request produces a board with ordered lists and cards. Validate bounded JSON with explicit allowed fields before showing a preview; invalid or truncated output cannot be saved. Stop aborts that request and leaves any prior preview and saved boards intact. A later explicit Generate is Retry. Account/server retirement aborts work and rejects late results.

Save is explicit and creates a new board. Reuse the existing authenticated Import endpoint with format `tldw_kanban_draft_v1`; do not use the legacy best-effort import loop for this save. Validate names, descriptions, list/card counts, total payload size and unknown fields in the DB abstraction before writing. The model supplies no database IDs, owners or URLs. A stable app-created client ID identifies one draft save. Store a canonical request hash in the new board metadata; same-owner matching replay returns the existing board without inserting or overwriting, while changed payload or soft-deleted identity conflicts. Other owners cannot acquire that board by reusing the identity.

Insert the board, ordered lists/cards and normal activity records in one SQLite `BEGIN IMMEDIATE` transaction; rollback every failure. Retain existing limits, per-user paths, FTS triggers and API error mapping. Successful Save opens the normal board editor. Pending Save disables dismissal/duplicate submission; a failure preserves the preview for recovery. Existing manual controls, exports and other import formats retain their contracts. No dependency or background worker is added.

## Scope and qualification

Change the Kanban DB import branch, its existing endpoint error/docs contract, shared client cancellation composition, one lazy dialog and its entry control, tests and ordinary docs. Reuse existing owner scope, model picker, transport and editor. Canonical API artifacts are generated once at final S03 integration, preserving existing public descriptions; do not hand-edit fingerprints.

Tests first: real DB success/order/replay/conflict/owner isolation/limit rejection/rollback (including injected later card/activity failure); actual API validation and ownership; strict response parsing, Stop/late result, Save failure and account/server retirement in both clients. Native acceptance uses exactly two lists and three cards, one real move, save/reload/export with independent full content/order verification, actual generation cancellation/failure/retry and four database/authentication modes. PostgreSQL modes retain the shipped per-user SQLite Kanban store. Human UX review uses a settled screenshot. Delete completed private payloads after actual teardown.

ADR required: no. No existing Kanban ADR is listed; this extends current per-user SQLite, foreground Chat and import boundaries without changing a durable global architecture rule. ADR-059 governs task edits.
