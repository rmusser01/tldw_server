# Notes Task-Backed To-Do Lists PRD Design

Date: 2026-06-05
Status: Ready for user review
Owner: Codex brainstorming session
Backlog: TASK-512

## Summary

Add first-class to-do list support to Notes so users can work through tasks while using chat, the workspace, and the floating Notes Dock.

The product model is task-backed, not merely markdown-enhanced. Checklist lines in notes remain readable plain markdown, but active checklist items reconcile to durable task records with identity, status, metadata, audit history, and MCP tool access. Users can mark items done from both the full `/notes` page and the Notes Dock. Agents can create, update, complete, and delete tasks through MCP Unified when the user's MCP policy permits it.

Markdown remains portable. The system must not insert hidden task IDs into note content. A note with tasks should still export as normal markdown containing checklist syntax such as `- [ ] Review source @due(2026-06-10) @priority(high)`.

## Product Decision

Use a first-class task model with markdown projection.

Notes continue to store free-form markdown content. Checklist lines in any note can be parsed, displayed as interactive task items, and reconciled with durable task records. Task records store stable identity and operational state, while the markdown remains the human-readable editing and export format.

This is a larger initial scope than a pure markdown checklist toggle, but it is the right model for the stated goal:

- users work through lists during chat/workspace sessions;
- the Notes Dock and `/notes` must agree;
- agents must be able to manage task state through MCP Unified;
- autonomous agent changes must be auditable and visibly surfaced;
- future task views, reminders, dashboards, and filters should not require replacing the foundation.

## Current Context

Source review found these implementation facts:

- Notes persist in ChaChaNotes with title, content, optional chat backlinks, keywords, optimistic locking, soft deletes, FTS, and sync logging.
- The current notes API already supports create, update, list, search, delete, restore, import/export, attachments, and Notes Studio paths.
- `/notes` has a markdown editor, preview/split modes, WYSIWYG support, save states, offline draft handling, backlinks, graph/manual links, and extensive staged UI tests.
- Notes templates already include markdown checklist syntax such as `- [ ]` for action items, next steps, and follow-up sections.
- The floating Notes Dock exists under `apps/packages/ui/src/components/Common/NotesDock` and is opened from chat/sidebar surfaces. It supports multiple open notes, dirty tracking, saved snapshots, note search/opening, and save/update.
- The shared markdown renderer uses GitHub-flavored markdown support in the common markdown path.
- MCP Unified already exposes a Notes module with `notes.search`, `notes.get`, `notes.create`, `notes.update`, `notes.delete`, and tag tools.
- MCP Unified already has protocol-level write-tool classification, custom validators, RBAC/tool permission checks, effective policy evaluation, runtime approval, governance preflight, idempotency support, and audit hooks.
- The existing MCP Notes module can update whole note content, but there are no checklist-specific tools. The PRD should add task-specific tools instead of encouraging agents to rewrite arbitrary note bodies for task changes.

## Problem

Users can write checklist syntax in a note today, but the product does not treat those items as tasks.

That creates several gaps:

- A user working in chat with the Notes Dock open cannot simply check items off in a reliable, task-like way.
- `/notes` and the Notes Dock do not share an explicit checklist interaction contract.
- The system has no durable task identity, completion timestamps, metadata, status history, or audit trail.
- Assistant or agent actions would need to edit free-form note markdown directly, which is too coarse for safe task management.
- There is no policy-governed MCP tool surface for agents to create, update, or complete task items.
- Autonomous agent changes would be hard to show, audit, or undo.

## Goals

1. Let any markdown checklist item in any note become an interactive task-backed item.
2. Support marking tasks done and reopening them from both `/notes` and the Notes Dock.
3. Preserve plain markdown readability, editing, and export with no hidden IDs in note content.
4. Store durable task identity, status, completion timestamp, metadata, source note linkage, version, and audit information.
5. Support lightweight metadata tokens in checklist text through a documented allowlist.
6. Expose task operations through MCP Unified with normal write-tool validation, RBAC, policy, runtime approval, idempotency, and audit behavior.
7. Allow both user-confirmed and autonomous agent task changes depending on the user's MCP policy.
8. Surface autonomous task changes visibly in the UI.
9. Avoid disrupting existing notes without checklists.

## Non-Goals

- No Kanban board in v1.
- No calendar sync in v1.
- No recurring tasks in v1.
- No notifications or reminders in v1, beyond storing metadata that can support them later. This does not exclude required in-app audit/activity notices for agent changes.
- No cross-user assignment or team task workflow in v1.
- No forced conversion of all notes into task documents.
- No hidden IDs, HTML comments, or invisible markers inserted into note markdown.
- No broad redesign of `/notes`, Chat, Workspaces, or MCP Hub management surfaces.

## Rollout Model

V1 should be delivered in explicit slices so the work does not become one oversized PR.

| Phase | Scope | Release gate |
| --- | --- | --- |
| 1. Core task foundation | Task storage, task events, parser, reconciler, lazy note reconciliation, backend/API tests. | Notes without checklist items remain unchanged; reconciliation is idempotent. |
| 2. Notes UI interaction | `/notes` preview/split checkboxes, Notes Dock task interaction, dirty-note concurrency handling, frontend tests. | Users can mark tasks done/reopen from both surfaces without clobbering unsaved content. |
| 3. MCP task tools | MCP tools for task discovery, create, update, status, delete, and reconciliation; policy/approval/audit/idempotency tests. | User-confirmed agent changes work end-to-end through MCP. |
| 4. Autonomous agent surfacing | Autonomous MCP task writes, persistent activity notices, unread/dismiss behavior, browser checks. | Autonomous changes are auditable and visible even if the affected note was not open when the change happened. |

Autonomous writes must not be enabled before Phase 3 tool policy is passing and Phase 4 notice delivery exists.

## Terminology

| Term | Meaning |
| --- | --- |
| Checklist line | A markdown list item matching task-list syntax, for example `- [ ] Review source`. |
| Task | A first-class durable record linked to a checklist line or created for projection into a note. |
| Projection | The markdown representation of a task in a note. |
| Locator | A version-bound parser result that identifies a checklist line without storing a hidden ID in markdown. |
| Reconciler | Backend service that maps note checklist lines to task records. |
| Task metadata token | An allowlisted token embedded in checklist text, such as `@due(2026-06-10)`. |
| Agent mutation | A task change made through MCP Unified by an assistant, agent, or external MCP client. |
| Autonomous mutation | An agent mutation that executes without per-change user confirmation because policy allows it. |

## Product Model

Tasks are first-class records. Notes are the primary editing and projection surface.

Any checklist line anywhere in a note can become interactive:

```markdown
## Follow-up
- [ ] Review source @due(2026-06-10) @priority(high)
- [x] Summarize findings
```

The note remains valid markdown. The backend stores task records for those checklist lines and keeps their status synchronized with the projected markdown.

For new or updated notes, checklist participation begins at successful note save/autosave. For existing notes created before task support, participation begins when the note is reconciled. Task discovery paths must not silently omit unreconciled checklist items: they must reconcile scoped candidate notes before returning complete results, or explicitly return a `reconciliation_pending`/incomplete-results signal.

The task record should hold:

- stable task ID;
- owning user/client scope;
- task text;
- status, initially `open` or `done`;
- source note ID;
- source locator/projection details;
- metadata parsed from allowlisted tokens;
- created, updated, and completed timestamps, with `completed_at` set when status becomes `done` and cleared when the task is reopened while prior completions remain in task event history;
- version for optimistic locking;
- deleted/unlinked state;
- last actor and audit references.

The note content should hold:

- visible checklist syntax;
- visible task text;
- visible metadata tokens;
- no hidden task IDs.

## Task Identity Without Hidden IDs

Task identity is durable in the task table, but markdown locators are not permanent identifiers.

Because the product forbids hidden IDs in markdown, the system cannot guarantee perfect task identity through every free-form text edit, duplicate item, or reorder. The PRD accepts that limitation in exchange for markdown portability.

Required identity rules:

- A locator is valid only with the note version it was produced from.
- Same-version checkbox toggles must include the expected note version and task version.
- Cross-version reconciliation may preserve identity only when it can make a strong match from source note, prior locator context, normalized text hash, occurrence context, and surrounding checklist block.
- If a unique match cannot be proven, the reconciler must not silently merge tasks. It should mark the old task `unlinked` or `ambiguous` and create a new task record for the new checklist line.
- Duplicate checklist text is allowed, but duplicate items cannot depend on text hash alone for identity.
- Reordered unique items may preserve identity when context still matches. Reordered duplicate items may become ambiguous.
- Renaming an item may preserve identity only if the locator/context match is still strong enough for the current note version.

This behavior must be documented for users in implementation-facing copy as "portable markdown with best-effort task continuity," not as perfect identity preservation.

## Metadata Tokens

V1 should support a small documented allowlist of extensible key/value tokens in checklist text.

Required allowlist:

| Token | Parsed metadata | Notes |
| --- | --- | --- |
| `@due(YYYY-MM-DD)` | `due_date` | Date-only, interpreted in the user's configured locale/timezone only when displayed. |
| `@priority(high)` | `priority: high` | Also support `medium` and `low`. |
| `@estimate(30m)` | `estimate` | Parse common short duration strings, but no scheduling math in v1. |

Unknown tokens remain plain text and are not stripped.

Malformed allowlisted tokens remain plain text and should not block note save. The parser may return non-blocking warnings for UI or diagnostics.

Parser normalization rules:

- Token names are case-insensitive and normalize to lowercase metadata keys.
- Leading/trailing whitespace inside parentheses is ignored for allowlisted tokens.
- Duplicate allowlisted tokens use the last valid token as the structured metadata value while preserving all original text.
- Invalid dates, unknown priorities, and unsupported estimate formats are ignored structurally and returned as warnings.
- `@due(...)` accepts strict ISO date text only.
- `@priority(...)` accepts `high`, `medium`, and `low`, case-insensitively.
- `@estimate(...)` accepts simple duration text such as `30m`, `2h`, and `1d`; richer duration parsing is out of scope for v1.

## Architecture

### Backend

Add a focused Tasks core domain beside Notes, backed by the per-user ChaChaNotes database family so it inherits local-first and self-hosted behavior.

Likely components:

- Tasks schemas in the API schema layer.
- Tasks persistence helpers under the ChaChaNotes DB management boundary.
- Task API endpoints for UI and internal callers.
- A markdown checklist parser and reconciler service.
- Notes create/update integration that triggers reconciliation when note content changes.
- MCP Unified task tools in the existing Notes module or a closely related Tasks module.
- Audit/event recording for user and agent task mutations.

### Storage

The PRD expects a schema equivalent to:

| Table | Purpose |
| --- | --- |
| `tasks` | Durable task records and current status. |
| `task_note_links` or task projection fields | Source note ID and current projection locator/hash. |
| `task_events` | Append-only task mutation/audit history, especially for agent actions. |
| `note_task_reconciliation_state` or equivalent | Per-note last reconciled version and reconciliation status for lazy backfill and discovery correctness. |

The exact table split can be decided during implementation, but the model must support:

- task identity independent of line position;
- optimistic locking;
- soft delete;
- sync/audit visibility;
- source note linkage;
- unlinked projection state when a checklist line disappears.

### Parser And Reconciler

The parser detects checklist lines and returns structured items:

- checked state;
- text without checkbox marker;
- raw line text;
- line number and character range when available;
- normalized text hash;
- metadata tokens and warnings;
- locator suitable for same-version updates.

The reconciler maps parsed items to existing tasks. It should use conservative heuristics:

1. Existing task linked to same note and same locator/hash.
2. Existing task linked to same note, unique stable normalized text hash, and compatible occurrence/block context.
3. Existing unlinked task with a strong match in the same note.
4. Otherwise create a new task.

The reconciler must be idempotent. Running it repeatedly on unchanged content should produce the same task state and links.

Hash-based matching is allowed only when the item is unique within the note or reinforced by occurrence context and surrounding checklist-block context. Ambiguous duplicate checklist lines must not be destructively merged. The reconciler should create distinct tasks or mark ambiguity for review.

### Frontend

`/notes` should support interactive task-backed checkboxes in preview/split surfaces while preserving raw markdown in the editor. Edit mode remains textarea/WYSIWYG-oriented rather than becoming a custom rich task editor in v1.

The Notes Dock should expose compact checklist interaction for the active/open note. The dock can keep its current dirty/save model, but it must also refresh task state after successful task mutations.

Both surfaces should render task metadata unobtrusively when parsed, for example due date and priority chips. Unknown tokens stay visible as plain text.

### MCP Unified

Task tools should be exposed through MCP Unified. The MCP tool layer is the authority path for assistant/agent task management, not private chat-only APIs.

Proposed tools:

| Tool | Purpose |
| --- | --- |
| `notes.tasks.list` | Discover tasks by note, status, metadata, or search query. |
| `notes.tasks.get` | Fetch a task with source note and projection details. |
| `notes.tasks.create` | Append or insert task-backed checklist items into a note under version checks. |
| `notes.tasks.update` | Edit task text and metadata. |
| `notes.tasks.set_status` | Mark one or more tasks open or done. |
| `notes.tasks.delete` | Soft-delete tasks and remove their checklist projection, or return a conflict if projection removal is unsafe. |
| `notes.tasks.reconcile_note` | Repair/reconcile note-task projection drift. |

All write tools must be MCP management/write tools with input schema validation and custom `validate_tool_arguments` coverage.

## Data Flow

### User Edits Checklist Markdown

1. User writes or edits checklist markdown in a note.
2. Note save/autosave stores note content through the existing notes save path.
3. The task reconciler parses checklist lines and metadata tokens.
4. The reconciler creates, updates, relinks, unlinks, or marks tasks ambiguous.
5. `/notes` and Notes Dock refresh note/task state.
6. Interactive checkbox surfaces reflect reconciled task status.

### User Clicks Checkbox On Clean Note

1. User clicks a task checkbox in `/notes` preview/split mode or Notes Dock.
2. UI sends task ID, desired status, expected task version, and expected note version when note projection is being rewritten.
3. Backend updates the task record.
4. Backend rewrites the corresponding markdown checkbox marker in the source note projection.
5. Backend records a task event.
6. UI updates status feedback and highlights the changed item briefly.

This immediate backend path is only valid when the active note content in that UI surface is clean for the expected note version. The `/notes` page should follow existing autosave/save-state conventions. The Notes Dock should preserve its current unsaved-change semantics for regular note editing, while task status writes that succeed on the backend should refresh the local note/task snapshot.

### User Clicks Checkbox With Dirty Note

1. User clicks a task checkbox while the active note has unsaved local edits.
2. UI changes only the local markdown checkbox marker and keeps the note dirty.
3. No durable task status, completion timestamp, projection rewrite, or task event is recorded yet.
4. If the user saves/autosaves successfully, the normal note save path reconciles the saved markdown and records the resulting task changes.
5. If the user reloads, discards, or save fails before a successful save, the local checkbox toggle is not durable and task state remains unchanged.
6. If a remote or autonomous task change arrives while the note is dirty, the UI queues a pending-change notice and must not overwrite local content.

### Dirty Note Checkbox Rules

The implementation must not perform a backend projection rewrite against stale note content.

Required behavior:

- If the active note content is clean in the current surface, checkbox clicks may call the backend task status path immediately. The backend updates the task record and rewrites the note projection under optimistic locking.
- If the active note content is dirty in `/notes` or the Notes Dock, checkbox clicks update the local markdown content only and keep the note dirty. The task record is reconciled on the next successful note save/autosave.
- If a remote or autonomous task change arrives while the user has dirty local note content, the UI must not overwrite local content. It should queue a non-destructive pending-change notice and require save, reload, or explicit merge.
- If the user saves dirty content after local checkbox toggles, reconciliation is responsible for bringing task records into alignment with the saved markdown.
- If note save fails, local checkbox changes remain unsaved note edits and must not be represented as durable task completion.

This rule applies equally to `/notes` and the Notes Dock.

### Agent Creates Or Changes Tasks

1. Agent discovers tasks using MCP tools.
2. MCP policy evaluates whether the tool is allowed, requires approval, or can run autonomously.
3. Approved tool call creates, edits, completes, or deletes tasks.
4. Backend updates task records and note markdown projection.
5. Backend records an audit event with actor and policy details.
6. UI notices the changed task/note state and shows a concise activity notice such as "Assistant marked 2 items done."

### User Or Agent Creates Task

Task creation mutates note markdown by inserting a new checklist line, so it must be version-checked like any other projection rewrite.

V1 creation behavior:

- The caller must provide a note ID and expected note version.
- The caller may append to the end of the note, append to an existing checklist block, or insert relative to a known task/locator if the current note version still matches.
- If the target note version is stale or the UI surface has dirty local content, creation must return a conflict or require the user to save/merge first.
- On success, the backend writes the checklist line, creates the task record, records an event, and returns the updated note/task versions.

### Metadata Token Update

1. User or agent edits task metadata.
2. Caller provides expected task version and expected note version when markdown projection changes.
3. Backend updates structured task metadata.
4. Backend updates visible markdown tokens in the task projection when needed under optimistic locking.
5. Unknown tokens remain untouched.

### Task Delete Flow

V1 includes explicit task soft-delete.

Deletion rules:

1. Deleting a task from task UI or MCP soft-deletes the task record.
2. If the task still has a live source projection, soft-delete and checklist-line removal must happen in the same version-checked transaction when possible.
3. If the projection cannot be safely removed because the note changed, the delete should fail with conflict details and no soft-delete should be committed, rather than leaving a live checklist line that immediately recreates a task.
4. If a user manually removes a checklist line while editing markdown, reconciliation should mark the linked task `unlinked` rather than hard-deleting task history.
5. Hard delete is out of scope for normal v1 user and agent flows.

## Permissions And Agent Behavior

MCP Unified owns permissions for agent task access.

The PRD supports both:

- **User-confirmed mode:** agent proposes changes, and the runtime requires approval before execution.
- **Autonomous mode:** agent may mutate any task/note allowed by its MCP policy and effective context.

For autonomous mutations, the policy scope may include any note allowed by the agent's MCP policy. The UI must not rely on active/open note scope as the only safety boundary.

Every agent mutation must record audit metadata:

- actor type;
- MCP client ID/session ID when available;
- tool name;
- note IDs;
- task IDs;
- old and new status/text/metadata;
- policy mode;
- approval ID or reason when applicable;
- timestamp.

Silent autonomous mutation is not acceptable. The UI must surface a notice for task changes made by an agent while the relevant notes UI is open or when the user next opens the affected note/dock session.

## Autonomous Activity Delivery

Autonomous activity notices must be persistent enough that users can discover changes after the fact.

Required behavior:

- Every agent task event is persisted in task audit/event storage.
- The backend exposes recent/unread task activity for the current user, with filters by note and task.
- The UI records dismissal/read state so the same activity is not repeatedly shown after dismissal.
- `/notes` and the Notes Dock show affected-note notices when opened.
- If an autonomous change affects a note that is not currently open, the next visit to that note or the next time the dock opens that note should show a pending activity indicator.
- Unread activity notices must be retained for at least 30 days in v1, unless the underlying note/task is permanently removed by an explicit retention policy.
- Read/dismiss state is stored per user and event, so dismissal in `/notes` and the dock converges.

This does not require a global notification center or reminders in v1.

## UX Requirements

### `/notes`

- Raw edit mode keeps markdown visible.
- Preview/split mode renders task-backed checkboxes for checklist lines.
- Clicking a checkbox on clean content updates task status and markdown projection immediately.
- Clicking a checkbox on dirty content changes local markdown only until save/autosave succeeds.
- Parsed metadata tokens can render as small chips or inline annotations.
- Save/conflict/error states reuse existing Notes patterns.
- If a task is unlinked or ambiguous, the UI shows a non-blocking reconciliation status rather than silently changing unrelated lines.

### Notes Dock

- The dock supports the chat/workspace scenario as a first-class workflow.
- Users can open a note with tasks, check items off, and keep working in chat.
- The dock should show active note checklist items in a compact interaction surface, either inline in the note body preview or as a task strip/panel attached to the active note.
- Changes made in the dock and `/notes` should converge through shared task/note state.
- The dock must keep current unsaved-change behavior for regular note edits.

### Agent Activity Notice

Autonomous agent changes should show a concise notice:

- who/what changed tasks;
- count of tasks changed;
- affected note/list;
- action to inspect the changed note;
- changed items highlighted until dismissed or until the user navigates away.

## Error Handling

### Version Conflict

If a note or task changed since the caller's expected version, reject the write with a conflict response. Include enough latest state for reload/retry.

### Missing Source Line

If a task's original checklist line disappears, keep the task record and mark its projection as `unlinked`. Do not silently recreate or append the task unless an explicit repair action is requested.

### Duplicate Or Ambiguous Lines

The reconciler must avoid destructive merges. It can create distinct task records or mark ambiguity for review.

### Malformed Metadata Token

Preserve the token as plain text. Ignore it for structured metadata and optionally return a non-blocking warning.

### Agent Partial Failure

MCP task tool responses must list succeeded, failed, and skipped task IDs for batch changes. No partial result should be represented as a full success.

### Offline Or Dock Edits

Offline/dock edits keep existing dirty/save behavior. Task projection updates become durable only after note or task writes succeed.

## API Requirements

The exact REST endpoint design can be finalized during implementation, but the UI needs equivalent capabilities:

- list tasks by note/status/query/metadata;
- fetch task details;
- create task-backed checklist items in a note;
- update task text and metadata;
- set task status;
- soft-delete tasks;
- reconcile a note;
- fetch recent task activity for visible audit notices.

Task list/discovery behavior:

- `notes.tasks.list` with a note filter must reconcile that note if its saved content version is stale before returning results.
- Broader task listing/search must identify scoped candidate notes that may contain checklist syntax, reconcile stale candidates in the returned page/scope, and include an explicit incomplete-results indicator if additional unreconciled candidates remain.
- MCP task discovery must use the same reconciliation-aware behavior, so agents do not miss checklist items solely because an old note has not been opened since migration.

Task mutation responses should include:

- task ID;
- note ID;
- task version;
- note version when projection changed;
- updated markdown projection summary;
- audit/event ID where available;
- conflict or warning details.

## MCP Tool Requirements

MCP task tools must:

- be discoverable through the existing MCP tool list;
- include JSON schemas with tight bounds;
- be classified as read/write correctly;
- implement custom argument validation;
- support idempotency keys for writes where practical;
- respect MCP RBAC, effective policy, runtime approval, path/scope evaluation where applicable, and governance preflight;
- enforce persona/note scope where active scope exists;
- return structured partial-success payloads for batch operations;
- avoid exposing raw full-note rewrites for simple task status changes;
- use reconciliation-aware list/discovery behavior so task tools do not silently miss unreconciled saved checklist lines.

## Reconciliation Requirements

Reconciliation should run:

- after note create/update when content changed;
- after task create/update/status changes when projection must be updated;
- before task list/search responses that claim completeness for stale notes in scope;
- on explicit MCP/admin repair tool invocation;
- on migration/backfill if task tables are introduced to an existing database.

Reconciliation should not:

- add hidden IDs to markdown;
- silently delete task history;
- merge ambiguous duplicate lines;
- overwrite user edits after version conflicts;
- strip unknown metadata tokens.

## Migration And Backfill

Existing notes remain valid. Adding the task model should not require changing notes without checklists.

Migration should:

1. Add task/task-event storage.
2. Leave existing note content untouched.
3. Track each note's last reconciled version/status.
4. Backfill tasks lazily when a note is opened, saved, searched/listed for tasks, or explicitly reconciled.
5. Ensure task list/discovery APIs either reconcile stale candidate notes before returning complete results or clearly report incomplete reconciliation state.
6. Provide an admin/debug repair path for rebuilding task links from notes.

The PRD does not require eager backfill of all notes during startup.

## Testing Strategy

### Parser And Reconciler

- Detect `- [ ]`, `- [x]`, and common markdown task-list variants.
- Parse due date, priority, and estimate tokens.
- Preserve unknown and malformed tokens.
- Normalize token casing and whitespace.
- Handle duplicate tokens using last-valid-token-wins semantics.
- Treat invalid dates/priorities/estimates as warnings, not save blockers.
- Avoid merging duplicate checklist text.
- Avoid hash-only identity preservation for duplicate checklist text after reorder or rename.
- Produce stable results on repeated reconciliation.
- Mark missing source lines as unlinked.

### Backend/API

- Create/update/delete task records.
- Delete task projection lines safely when explicit task delete succeeds.
- Reconcile tasks after note save.
- Reconcile stale candidate notes during task list/search and report incomplete reconciliation state when applicable.
- Rewrite note markdown when task status changes.
- Create task-backed checklist lines only under expected note-version checks.
- Enforce optimistic locking for note and task versions.
- Return structured conflicts.
- Record task events for user and agent changes.
- Preserve existing notes without checklist changes.

### MCP

- Tool schemas and validators reject invalid payloads.
- Read tools respect scope.
- Write tools are write-classified and require custom validation.
- Approval-required mode returns approval-required behavior.
- Autonomous mode succeeds only under allowed MCP policy.
- Batch status changes return succeeded/failed/skipped IDs.
- Delete tool removes task projections safely or returns conflicts.
- Task discovery tools reconcile stale candidate notes or return explicit incomplete-results state.
- Idempotency prevents duplicate task creation or duplicate repeated status changes where practical.

### Frontend

- `/notes` preview/split renders interactive task checkboxes.
- `/notes` raw edit mode preserves markdown.
- Notes Dock renders and toggles task-backed items.
- Save-state and conflict UI match existing Notes behavior.
- Agent activity notices appear for autonomous changes.
- Unread autonomous activity notices survive page reload and next-session note reopen until dismissed or retention expires.
- Checkboxes have accessible labels and states.
- Existing notes without checklist items behave unchanged.

### Browser Checks

- Open chat with Notes Dock visible.
- Create a note containing a checklist.
- Mark an item done in the dock.
- Open `/notes` and verify the same task is done.
- Reopen the item from `/notes` and verify the dock updates after refresh/sync.
- Toggle a checkbox while the dock has unsaved edits, fail or cancel save, and verify durable task state did not change.
- Simulate or trigger an MCP/agent task status change and verify the activity notice.
- Reopen the affected note in a later browser session and verify unread autonomous activity is still visible.

## Acceptance Criteria

- Any markdown checklist in a note can become an interactive task-backed item.
- Task text, status, metadata, and completion state survive note reload and dock reopen.
- No hidden IDs are added to markdown.
- `/notes` and Notes Dock can both mark items done and reopen them.
- Dirty-note checkbox toggles cannot clobber unsaved note content.
- Dirty-note checkbox toggles are not durable if the user reloads, discards, or fails to save before reconciliation.
- Existing notes with checklist syntax are discovered by task listing/search through reconciliation-aware behavior, not silently omitted.
- MCP Unified exposes task tools for discovery, creation, update, completion, deletion, and reconciliation.
- MCP write behavior is governed by existing policy, approval, RBAC, validation, idempotency, and audit mechanisms.
- Autonomous agent changes are visibly surfaced.
- Version conflicts do not silently overwrite note or task edits.
- Existing notes without checklist items behave unchanged.

## Open Implementation Decisions

These are intentionally left to the implementation plan:

- Exact table split between `tasks`, task-note projection rows, and task events.
- Whether task UI in Notes Dock is inline, a compact panel, or both.
- Exact REST endpoint names and response schemas.
- Whether to provide an optional one-time backfill command in addition to required lazy reconciliation.
- How much of task metadata is shown in list rows versus only detail/projection surfaces.
