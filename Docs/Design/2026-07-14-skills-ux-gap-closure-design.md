# Skills UX Gap Closure Design

**Task:** TASK-12969
**Date:** 2026-07-14
**Status:** Implemented and verified

## Goal

Make `/skills` safe and understandable for a first-time user while preserving a fast, predictable management surface for experienced users. The implementation closes the concrete gaps confirmed in the latest `dev` source review without mixing in the unrelated MCP catalog-render change.

## Confirmed Problems

1. Pressing Enter in test arguments performs a live test rather than the safe dry render.
2. A late execution response can be displayed after the selected skill changes.
3. New-skill templates still expose raw YAML and `$ARGUMENTS` as the primary beginner workflow.
4. Drawer and text-import drafts can be dismissed without warning or recovery.
5. Search and result fields have incomplete label associations; file import nests an upload control inside a menu label.
6. Connection and capability states do not retain the page-level `Skills` heading.
7. Every filter and view option is expanded at once, increasing scanning cost.
8. Search, filters, sorting, and pagination are not represented in the URL.
9. Selection is pruned to the current response page, preventing cross-page bulk work.
10. Rows do not provide direct view, use-in-chat, copy, or duplicate workflows.
11. The desktop table has no explicit narrow-screen layout contract.
12. Save/delete conflicts tell users to reload but provide no recovery action.
13. Delete is destructive even though the registry uses a soft-delete flag; the skill files themselves are removed, so the current data model cannot truthfully offer restore.

## Product Decisions

### One page, two operating speeds

The default surface remains a searchable library. Beginners get a structured authoring mode, explicit view details, and safe dry rendering. Power users retain raw source editing, table density and column controls, server-side filtering, keyboard shortcuts, URL-backed views, cross-page selection, and bulk operations.

### Guided authoring generates source

Guided mode owns structured fields for name, description, argument hint, instructions, execution mode, visibility, model invocation, model override, and declared tools. A deterministic serializer generates valid `SKILL.md` content. Advanced mode exposes that source directly. Existing skills initialize guided fields from the structured API response and body returned by the server, avoiding a second YAML parser in the browser.

Switching from edited raw source back to guided mode requires confirmation because arbitrary YAML cannot be losslessly mapped into the supported structured fields. Saving performs immediate local validation and then relies on the existing server parser as the authoritative validator.

### Drafts are session-scoped

Create/edit and text-import drafts are stored in `sessionStorage`, keyed by workflow and skill name. A dirty close request from the close button, Cancel, Escape, or backdrop opens one confirmation with `Keep editing` and `Discard draft`. Successful save/import clears the draft. Session storage is best-effort and never blocks editing.

### Safe execution is the default

Enter in the argument field performs `dry_run=true`. Live execution is only available through the explicit `Run test` button. Each request captures skill name, arguments, request id, and an `AbortSignal`; changing or closing the skill aborts the request and invalidates late results.

### Compact controls with visible state

Search and primary commands stay visible. Filters move into one labelled popover, and sorting plus desktop density/columns move into a separate View popover. Active non-default filters render as removable chips with a clear-all action. The empty filtered state includes the same recovery action. Sorting remains independent from `Clear filters` and is directly available in the mobile row layout.

### URL is the saved-view contract

The query string stores library/trash view, search, mode, visibility, tools, model, sort, order, page, and page size. Defaults are omitted. Committed view, filter, sort, and pagination actions push history entries; debounced search/model input and canonicalization replace the current entry. This supports reload, Back/Forward navigation, bookmarking, and link sharing without creating a second named-view persistence system.

### Selection stores records, not only current keys

Selection is a map keyed by skill name. Current-page responses refresh selected records but do not discard records that are off-page or filtered out. Selection stops at the backend's 100-item atomic bulk-operation limit, disables additional unchecked rows, and announces the limit instead of submitting a request that will fail validation. Bulk export creates one ZIP containing each existing server export; partial failures are reported without discarding successful downloads.

### Mobile uses purpose-built rows

At narrow widths, each skill is an unframed bordered list row with stable metadata, selection, and 44px actions. The full table is desktop-only. Drawers and modals use viewport-constrained widths. No horizontal body overflow is permitted at 390x844.

### Trash is durable; version snapshots are not added

Deletion first moves the complete skill directory to a per-user hidden `.trash/<registry-uuid>` directory, then marks the registry row deleted with version-checked optimistic locking. If the registry transition fails, the untouched directory moves back. UUID addressing prevents a stale or corrupted path field from selecting another archive. The trash API lists deleted registry records and reports whether their files are restorable. Restore moves the archive back and reactivates the registry row. Permanent delete first stages the untouched archive so a registry failure can still restore it; after the registry purge commits, destructive removal is queued under `.trash/.cleanup` and is never treated as rollback-capable. Partial cleanup residue is retried by later service instances.

This supplies durable recovery and immediate Undo without inventing an unbounded version-history store, retention policy, quota policy, or migration. Active version snapshots are intentionally excluded: they are a separate storage product decision, not required to fix destructive delete.

Normal create/import rejects a name already present in either the Library or Trash so recovery is never discarded implicitly. An explicit overwrite may replace a deleted same-name record; the old archive is retained until the replacement registry activation succeeds, then atomically queued for retryable cleanup. Cleanup failure cannot deactivate or remove the valid replacement.

## Interaction Model

### Library row

- Select
- Open details
- Use in chat (prefills `/skill <name>` and navigates to `/chat`)
- Dry/live test dialog
- Copy invocation
- More: duplicate, edit, export, delete

The details drawer includes source-independent metadata, runtime declarations, version/timestamps, supporting-file names, and the same primary actions.

### Delete and conflict recovery

- Delete confirmations explain that the skill moves to Trash.
- Success exposes `Undo` until another delete replaces the notice or the user dismisses it.
- Delete conflicts expose `Reload latest`, refresh the record, and keep the action available.
- Save conflicts keep the local draft in place and expose `Review latest`. The comparison dialog shows the latest complete server source, including custom frontmatter, while preserving the local draft. Only an explicit `Keep draft and overwrite` decision adopts the latest version token and reenables Save.

## API Additions

- `GET /api/v1/skills/trash`
- `POST /api/v1/skills/{name}/restore`
- `DELETE /api/v1/skills/{name}/purge`

Trash responses contain only public metadata, deletion timestamp, version, and a `restorable` flag. No filesystem paths are exposed.

Normal create/get/update/import responses also expose `raw_content`, the complete `SKILL.md` source. The advanced editor uses this field so unknown frontmatter survives an API-to-editor round trip.

## Accessibility Contract

- One persistent `h1` exists in connected, loading, unsupported, and disconnected states.
- Search and all text areas have programmatic labels; visible labels are used where space allows.
- Dialog and drawer accessible names include the skill name.
- File import is one semantic menu action backed by a hidden labelled file input.
- Focus returns to the invoking row or command after drawers and dialogs close.
- Keyboard shortcuts ignore editable targets: `/` focuses search and `n` opens New Skill.
- Mobile action targets are at least 44px.
- Loading, success, conflict, and execution states are announced through status or alert regions.

## Reliability Rules

- Server error text is sanitized before display.
- Abort is not shown as a failure.
- A stale request cannot update visible result, error, or pending state.
- Filesystem and registry delete/restore operations roll back only while every source bundle remains untouched; post-commit destructive cleanup is staged, best-effort, and retryable.
- Bulk delete validates all versions before moving any directory and rolls back already-moved directories if a later move fails.
- Trash mutations and active skill updates are serialized across service instances with a per-user file lock. Cancellation waits for the complete mutation to commit or roll back before releasing that lock, then propagates to the caller.
- First-use reconciliation restores a valid canonical archive when a delete was interrupted before its registry commit. It runs during the cached service's first lock-protected worker sync, deletes a stale archive only when the active bundle is safely readable and parseable, and preserves missing, malformed, symlinked, or otherwise ambiguous bundles for recovery.

## Verification

- 158 focused Vitest tests cover serializers, query state, text and file preview safety, draft recovery, controls, selection, focus return, actions, mobile markup, conflict recovery, YAML identity validation, and API mapping.
- 278 backend Skills unit and integration tests cover Trash listing, archive delete, restore, purge, bounded plain and quoted optimistic-lock tags, rollback, interrupted commits, update/delete serialization, cross-process locking, cancellation, fail-closed reconciliation, symlink rejection, read-only cleanup, database-owned service lifetime, and import identity round trips.
- 13 deterministic Playwright scenarios cover beginner, power-user, Trash, mobile, loading, conflict, and failure workflows; 3 optional live-backend scenarios remain intentionally skipped without a configured backend.
- TypeScript typecheck, focused production ESLint, Python compilation, Ruff, JSON parsing, and git diff checks pass. Bandit reports zero findings in the touched backend scope.

## Implementation Review Outcome

The delivered surface follows the approved one-page/two-speed model and keeps MCP catalog-render work out of scope. Browser UAT confirmed the beginner, power-user, mobile, Trash, conflict, loading, and failure workflows against deterministic API fixtures.

Final read-only review surfaced additional defects, all fixed at their source:

- The global i18next ICU adapter interpolated before ICU memoization, causing repeated translation keys to retain the first skill name or count. It now converts only supplied i18next placeholders to ICU arguments before formatting and preserves literal template syntax.
- Partial recursive deletion could invalidate a rollback source and destroy both a replacement and its old archive. Replacement and purge now separate durable commit from retryable cleanup, preserve the committed valid state, and never restore a partially deleted archive.
- Skill responses omitted the complete source required for lossless custom-frontmatter editing. `raw_content` is now part of the API response contract and round-trip coverage.
- Save-conflict recovery adopted a new version token without showing remote changes. Users now review the latest source and explicitly authorize retaining their draft as an overwrite.
- Cross-page selection could exceed the backend's 100-item atomic request contract. Desktop and mobile selection now enforce and announce that boundary.
- URL synchronization replaced every history entry. Committed controls now support browser Back/Forward while debounced input remains history-neutral.
- The post-create/import `View skill` action opened Edit. It now opens the read-only details drawer with the matching eye affordance.
- Concurrent service instances could reconcile another request's in-flight purge, and cancellation during lock acquisition could strand a late-acquired file lock. Trash transitions now use a cross-process lock with cancellation-safe acquisition and release.
- An interrupted delete could leave the only valid bundle in its canonical archive while the registry remained active; startup previously classified that archive as stale. Reconciliation now restores valid interrupted deletes and preserves every malformed or ambiguous state.
- Cancellation during a filesystem or database worker could release the Trash lock before the worker finished. Complete Trash transactions are now shielded through commit or rollback, with cancellation propagated only after the lock-protected state is coherent.
- Trash listing and restore previously disagreed about malformed supporting files, and restore could reacquire its non-reentrant lock after committing. Both paths now validate the complete bundle, archive validation runs off the event loop, and restore returns without nested synchronization.
- A stale text-import preview could retain pending state and block its replacement. Preview requests are now revision-bound and abortable, so late results cannot authorize or overwrite a newer review.
- Raw source could declare a different identity from the canonical registry name, while browser and server YAML scalar rules disagreed. Create and update now enforce normalized identity, the browser mirrors PyYAML boolean semantics, and explicit import renames rewrite the source through structured YAML while preserving unknown fields and already-canonical source.
- Stale cross-page bulk versions are cleared after conflict, bulk export request pressure is bounded, destructive dialogs include the affected skill in their accessible names, and every accepted timestamp sort is visible in the selector.
- Request-scoped service construction repeated synchronous Trash scans and discarded debounce state. The dependency now attaches one service to the owning cached user database, so reuse follows the database lifecycle, while reconciliation and cleanup run once during the first lock-protected registry sync on a worker thread.
- Standard quoted `If-Match` entity tags previously failed FastAPI integer coercion with 422 responses, and unbounded digits could fail conversion. All four versioned mutation routes now accept bounded plain or quoted numeric versions and reject malformed or oversized tags with a stable 400 response.
- Recursive cleanup could stall indefinitely on Windows read-only entries or follow an in-queue symlink to a sibling directory. Cleanup now validates the original entry, adds the owner write bit only for non-symlink entries after a `PermissionError`, retries the failed operation, and preserves residue when removal remains unsafe.
- Active updates previously ran outside the Trash lock, allowing a failed update rollback to recreate a bundle after another service deleted it. Updates now complete their filesystem, registry, and rollback work under the same cancellation-safe cross-process lock as delete and restore.
- Archive lookup resolved a UUID entry before inspecting it, allowing a symlink inside Trash to redirect restore or purge to a sibling archive. Canonical archive entries are now inspected without following links and symlinks fail closed.
- File-import previews accepted every completion, so an older request could replace the user's latest file review. File previews now carry request revisions and ignore stale success and error completions.
- Opening Test from the details drawer captured a soon-to-unmount button as the focus target. Closing the test surface now returns focus to the stable View action for that skill row.

Active filter chips were also revised to expose human-readable labels and named native remove buttons with mobile-sized targets. No unbounded revision history, named-view persistence layer, or unrelated MCP dependency work was added.
