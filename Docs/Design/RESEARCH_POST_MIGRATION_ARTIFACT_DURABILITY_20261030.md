# Postmigration Research artifact durability

Task: TASK-13260.278.18.83.45. S04 remains open.

The real Deep Research bundle import was visible, then disappeared on reload. A migration tombstone suppresses the old browser snapshot, including newly generated artifacts; canonical restoration reads server artifacts only. Never remove that tombstone or merge arbitrary old snapshots.

Save generated review drafts through the existing owned Workspace artifact API. Keep the visible stable artifact ID and full content, source lineage and producer identity. This creates neither a source nor a publication. Store client-only structured payload and presentation metadata in a versioned, namespaced portion of the existing version metadata. Restore only recognized client fields, with canonical core fields authoritative.

Capture the existing authenticated request lease before asynchronous generation/import. Each write and readback must retain that account, server and workspace, and retire on navigation/account changes. Do not announce saved/imported until canonical acknowledgement. On an unknown write outcome, reconcile the exact stable ID once; do not blindly submit again. Updates retain optimistic version checks. A failed save retains the visible generated result and an actionable failure.

Use one shared persistence adapter across Deep bundle imports, normal Studio completion and ACP result saves, with their subsequent changes covered. Keep responsive browser state and existing legacy/offline cache paths; migrated canonical state must not resurrect legacy data.

ADR assessment: ADR-008 remains governing browser split-key caching. No new durable architecture rule is introduced: use the existing migrated Workspace artifact draft resource and its transaction/version contract. No new database, dependency or storage namespace.

## Validation
- Establish tombstone → import → flush → cold restore loss before repairing.
- Preserve ID, content, Deep/ACP data, lineage, review/template information and version identities.
- Exercise canonical/local coexistence, optimistic conflicts, unknown acknowledgement and failure without duplicate writes.
- Refuse retired scopes and other accounts/servers/workspaces, including same visible workspace IDs.
- Preserve legacy anti-resurrection controls and validate connected producer callers in both client configurations.
- Validate current-build native report save/reload and remaining Deep Research recovery/modes before S04 closes.

## First reviewed unit

Only Deep import, canonical reload and enrolled removal/Undo are implemented in this unit. Normal Studio/ACP producer enrollment and later edit durability remain open. Canonical IDs/optimistic versions are separate from generated resource and display identities. Ownership evidence exists only as a runtime Symbol bound by an actual scoped read or fresh owned bundle; serialized caches cannot recreate it. Archive/Undo changes only review state and the removal marker, retaining full canonical lineage and history. One absolute fifteen-second operation budget and independent account retirement prevent a held completion from reviving an old attempt. Build and native acceptance remain required.

## Ordinary Studio generation and study-output edits

Canonical enrollment is runtime metadata established by an owned server restore. It also covers server workspaces that never needed a migration tombstone. Retain workspace IDs and scope keys for the page lifetime: evicting an older enrollment can silently turn a cached workspace switch into a local-only save. This adds no persisted authority or storage namespace.

Capture click and operation retirement before asynchronous capability checks or generation. Retire on unmount, workspace changes and account/server changes; rebuild selected-source identity and titles from scoped canonical reads, and use only witnessed prior artifacts as model input. Generation transports keep the captured target and cancellation signal.

Keep the completed result and stable draft ID when saving fails. A visible save-only Retry reconciles that same draft without another generation. Retire the pending save if the card is edited, deleted or its authority changes. The final save has its own bounded acknowledgement stage; generation Cancel does not discard a completed result while it is being saved.

Worker-created media artifacts keep canonical identity/version separate from display and export IDs. New backend media generation creates a distinct version rather than sharing the previous row's display identity. Client-only enrollment and content-only study edits preserve raw canonical lineage, ownership, producer, review and redaction fields.

Flashcard and quiz editors wait for acknowledgement before publishing changes or closing. Failures/conflicts retain the edit. Pending controls are disabled, and only that editor's removal Undo actions are finalized before Save or retirement; unrelated workspace Undo remains available.

ACP enrollment and live native qualification remain open. Transient audio blob URLs are not durable resources; this draft unit preserves the script, existing metadata and durable references and makes no binary-audio restoration claim. ADR-008 remains governing; no new durable architecture rule or dependency.

## ACP history saves

Capture verified history authority before reading projects, tasks or sessions through the existing scoped transport. Keep one fifteen-second load budget and retire old results, errors and actions on account/server/workspace changes. Only the exact projects collection's structured 404 Not Found means unsupported orchestration; ordinary missing records retain their error.

Plan ACP versions from the owned canonical artifact inventory, never cached cards. Keep the existing custom ACP version identity and use it as the workspace-relative canonical ID so competing allocations conflict visibly. A bounded save publishes only after canonical acknowledgement. The same pending result keeps its ID through save-only retry and late receipt reconciliation, including history close/reopen. Changed visible run content requires a new explicit Save intent; known conflicts also require another explicit Save after a fresh inventory. Retain durable run provenance/timestamps in existing ACP data; do not serialize request authority.

Restore acknowledged artifacts through the existing ID-preserving store action. Keep the complete Header workspace/pending-entry fence inside the shared adapter. This source unit adds no database/dependency/storage namespace; ADR-008 and the existing canonical resource contract remain governing. Combined build/private native and binary-audio acceptance remain open.
