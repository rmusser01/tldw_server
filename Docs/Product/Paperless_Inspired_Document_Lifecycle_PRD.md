# Paperless-Inspired Document Lifecycle PRD

Date: 2026-07-09
Status: Ready for implementation planning
Backlog: TASK-12093

## Summary

Improve tldw's document and source lifecycle by borrowing the smallest useful
ideas from Paperless-ngx: review state, saved source views, duplicate recovery,
clear provenance, and storage policy visibility.

This is not a Paperless clone. tldw remains a source-grounded research and media
analysis workbench. The goal is to make captured documents, web pages, PDFs,
audio, video, and imported files easier to trust after ingestion and easier to
reuse inside workspaces.

## Backlog Tasks

- TASK-12093: Umbrella PRD and tracking task.
- TASK-12093.1: Persisted source review lifecycle.
- TASK-12093.2: Saved source filter presets and views.
- TASK-12093.3: Duplicate detection and attach-existing recovery.
- TASK-12093.4: Document Workspace provenance and storage metadata panel.
- TASK-12093.5: Unified ingest entrypoints and storage policy visibility.

## Terminology And Entity Scope

- Media item: the existing per-user library record in the media database. It can
  have extracted text, storage records, search/vector readiness, trash state, and
  other global media facts.
- Workspace source: the workspace-specific association to a media item or source
  reference. V1 review state belongs here so the same media item can be reviewed
  independently in different workspaces.
- Source: the user-facing term for something available in a workspace source
  list, such as an uploaded PDF, captured web page, imported file, audio/video
  item, or existing media item attached to the workspace.
- Document: a source that can be shown in Document Workspace. Not every media
  item is a document, and Document Workspace metadata should be limited to
  source types it can represent clearly.
- Media-backed document: a Document Workspace item backed by a media item and
  optional storage record.
- Storage record: backend storage metadata for original or derived files. UI may
  show safe display facts, but raw internal storage paths remain private.
- Ingest entrypoint: a UI or API path that creates, imports, captures, or
  attaches a source.

## V1 Product Decisions

- Review state lives on workspace source associations in v1, not on global media
  items. Existing workspace sources migrate to `unset`. A later global media
  review state can be considered separately.
- Review state uses `unset`, `needs_review`, and `reviewed`. Expose it as a
  separate workspace-source field, not as a processing/status lifecycle value.
  Store `review_state_updated_at` on every transition. When a source becomes
  `reviewed`, also store nullable `reviewed_at` and `reviewed_by_user_id` when
  auth context is available; clear those reviewed-only fields when the source
  moves back to `needs_review` or `unset`. Bulk updates require the same
  workspace-source write permission as single-source updates.
- The built-in Needs review view includes only explicit `needs_review` sources.
  Existing sources migrated to `unset` appear in a separate Unreviewed view so
  migration does not flood the needs-review queue.
- Quick Ingest and extension captures only default to `needs_review` when the
  created or attached source is associated with a workspace and the entrypoint
  preset or user setting requests review.
- Saved source views are server-backed per user and per workspace in v1. They
  should restore across reloads, browsers, and devices for that user. The saved
  payload should be versioned even if it reuses the current source-list state
  shape.
- Duplicate lookup may only return confirmed duplicate actions for records the
  current user can open, attach, or restore. Inaccessible matches must be
  indistinguishable from no recoverable match and must not confirm that a match
  exists.
- If a backend uniqueness constraint blocks ingest because of an inaccessible
  duplicate, user-facing copy must stay generic and non-confirming, such as
  `Ingest could not be completed under the current duplicate policy`.
- Storage policy labels expose current backend behavior only. Disabled labels
  need an explicit reason and must not imply archive generation that does not
  exist.
- Document Workspace may show workspace membership only for the current
  workspace and other workspaces the current user can read.
- The first canonical extension path for unified ingest labels is the
  context-menu Send to tldw URL ingest path in
  `apps/packages/ui/src/entries/background.ts`. The sidepanel web-clipper flow
  can follow after this path is normalized.

## Problem

tldw already has strong ingestion, source status projection, original-file
storage, Document Workspace, Research Workspace, Quick Ingest, extension capture,
trash, quota, and Jobs foundations. The rough edge is lifecycle clarity after a
source enters the system:

- users cannot always tell which new sources still need human review;
- source filters are powerful but not packaged as repeatable work views;
- duplicate ingest outcomes are not consistently surfaced as recovery choices;
- Document Workspace metadata is useful but light on provenance and storage
  facts;
- WebUI, extension, file upload, URL paste, and drag/drop entrypoints can feel
  like adjacent paths instead of one intake model.

Paperless-ngx is useful here because it treats intake, metadata, duplicate
detection, storage facts, and review views as one document lifecycle. tldw should
adopt that discipline without adopting Paperless's office-archive assumptions.

## Goals

- Add a persisted source review lifecycle that survives after ingestion.
- Provide built-in and user-saved source views for repeated review work.
- Turn duplicate detection into an actionable recovery path.
- Expand Document Workspace metadata with provenance, storage, readiness, and
  workspace facts.
- Make ingest entrypoints share the same queue/progress/storage-policy language.
- Keep each slice independently shippable and reviewable.
- Reuse existing APIs, stores, status projections, and storage records where
  practical.

## Non-Goals

- Do not build a separate document-management database.
- Do not copy Paperless-ngx barcode, ASN, physical archive, correspondent, or
  document-type concepts into the product by default.
- Do not add public share links.
- Do not add arbitrary Jinja-style filename or storage-path templates.
- Do not build a full workflow builder in this initiative.
- Do not require a new storage backend or object-store migration.
- Do not hide permission, quota, duplicate, or readiness state behind smart
  defaults.

## Primary User Flows

### Review A New Workspace Capture

1. User captures or uploads a source into a workspace through Quick Ingest,
   extension capture, URL paste, or file upload.
2. The entrypoint shows storage policy, quota impact where known, queue state,
   and result state using shared labels.
3. If the entrypoint preset opts into review, the resulting workspace source is
   marked `needs_review`.
4. User opens the Needs review view for explicit review requests or the
   Unreviewed view for migrated `unset` sources, inspects readiness/provenance,
   and marks one or more sources `reviewed`.

### Reuse A Duplicate Source

1. User ingests a file or URL that matches an accessible existing source.
2. The duplicate result offers safe actions such as open existing, attach to the
   current workspace, restore from trash, or ingest anyway when policy allows.
3. If the only matches are inaccessible, the UI does not confirm that a match
   exists and continues with normal or generic failure behavior.

### Inspect A Document In Context

1. User opens a media-backed document in Document Workspace.
2. The Info panel shows safe provenance, storage facts, readiness, and readable
   workspace memberships.
3. Storage actions appear only when the user has permission and the original or
   derived artifact is available.

## Risks And Mitigations

- Duplicate existence leakage: only return confirmed duplicate actions for
  accessible records; use non-confirming copy for inaccessible matches and hard
  constraint failures.
- Review-state confusion: keep v1 review state on workspace source associations
  and migrate existing workspace sources to `unset`. Expose review state through
  dedicated fields and filters rather than overloading processing status.
- Saved-view drift: persist versioned per-user, per-workspace saved views and
  make stale payloads fail soft with reset.
- Storage-policy overpromising: expose only implemented backend behavior, show
  disabled reasons, and keep archive generation out of scope.
- Metadata overexposure: limit displayed workspace memberships to workspaces the
  current user can read and never expose raw storage paths.
- Entrypoint sprawl: start with the highest-traffic WebUI and extension paths
  and require shared labels before expanding to every endpoint.

## Current Repo Anchors

- Quick Ingest UI and presets:
  - `apps/packages/ui/src/components/Common/QuickIngest/`
  - `apps/packages/ui/src/components/Common/QuickIngest/presets.ts`
- Extension URL ingest and web capture:
  - `apps/packages/ui/src/entries/background.ts`
  - `apps/packages/ui/src/entries/web-clipper.content.ts`
  - `apps/packages/ui/src/entries/shared/ingest-payloads.ts`
- Research Workspace source pane and filters:
  - `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/`
  - `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/source-list-view.ts`
- Workspace source status projection:
  - `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
  - `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
- Document Workspace metadata:
  - `apps/packages/ui/src/components/DocumentWorkspace/LeftSidebar/DocumentInfoTab.tsx`
- Media original-file serving and storage records:
  - `tldw_Server_API/app/api/v1/endpoints/media/file.py`
  - `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`
- Ingestion sources:
  - `tldw_Server_API/app/api/v1/endpoints/ingestion_sources.py`
  - `tldw_Server_API/app/api/v1/schemas/ingestion_sources.py`
- Storage quota and trash:
  - `tldw_Server_API/app/api/v1/endpoints/storage_usage.py`
  - `tldw_Server_API/app/api/v1/endpoints/storage_trash.py`

## Paperless-Inspired Principles To Keep

### Lifecycle Over Upload

Ingestion is not done when the upload succeeds. It is done when the source has
reviewed metadata, known readiness, and an obvious next action.

### Repeated Work Needs Views

Users need durable views such as Needs review, Failed ingest, Partially indexed,
PDFs, Web captures, and Large files. These should be saved states over existing
filters first, not a new query builder.

### Duplicates Are Recovery Opportunities

A duplicate should help the user find, attach, restore, or intentionally reingest
content. It should not be only a warning string.

### Provenance Belongs Beside The Document

The document view should show where the source came from, what file is stored,
what content was extracted, and whether it is usable for search, vectors,
citations, and tools.

### Storage Policy Must Be Explicit

Users should understand whether tldw will only index content, keep the original
file, or use derived preview/archive artifacts when available. Quota impact
should be visible before ingest starts.

## Child Task 1: Persisted Source Review Lifecycle

Backlog: TASK-12093.1

### Scope

Add a persisted review lifecycle for ingested sources. This is separate from the
existing transient Quick Ingest row status.

### Product Behavior

- New workspace sources can be marked `needs_review` after creation or
  workspace attachment.
- Existing workspace sources start as `unset` after migration.
- Review state can be changed to `reviewed` or back to `needs_review`.
- Source rows show review state alongside processing readiness without merging
  it into queued/indexing/queryable/failed status.
- Workspace source filters include dedicated `reviewStateFilters`, separate from
  processing `statusFilters`.
- Quick Ingest and extension captures can opt into defaulting new sources to
  `needs_review` when they attach to a workspace.

### Boundaries

- Do not replace existing source lifecycle states such as queued, indexing,
  queryable, or failed.
- Do not make review required before sources become queryable.
- Do not add a complex approval workflow.

### Acceptance Criteria

- Persisted review state survives page reloads and workspace reloads.
- Review state is visible in Research Workspace source rows and details.
- User can mark one source or selected sources reviewed.
- Existing workspace sources migrate to `unset` without being treated as
  reviewed.
- Each review-state transition records `review_state_updated_at`. Review actor
  and timestamp are recorded when a source becomes `reviewed`, when auth context
  is available, and reviewed-only fields are cleared when a source leaves
  `reviewed`.
- Quick Ingest and extension workspace attachments can opt into defaulting new
  workspace sources to `needs_review`.
- Workspace source/status responses expose review state fields separately from
  processing status.
- Needs review and Unreviewed states appear in source filter presets.
- Tests cover normalization, persistence, and UI state transitions.

## Child Task 2: Saved Source Filter Presets And Views

Backlog: TASK-12093.2

### Scope

Package existing source filters into built-in presets and allow users to save
named source views.

### Product Behavior

- Built-in presets: Needs review, Unreviewed, Failed ingest, Partially indexed,
  PDFs, Web captures, Large files.
- Users can save the current source filter/sort state with a name.
- Saved views restore the existing source list filter model.
- Saved views persist server-side per user and per workspace.
- The implementation extends `SourceListViewState` with `reviewStateFilters`
  unless a narrower server contract is needed.

### Boundaries

- Do not build a generic query language.
- Do not persist saved views as browser-local-only state in v1.
- Do not add dashboard widgets in the first slice.

### Acceptance Criteria

- Built-in presets apply the expected existing filters.
- Needs review includes explicit `needs_review`; Unreviewed includes migrated or
  otherwise unset `unset` sources.
- Saving and reopening a user view restores filters and sort after reload.
- Saved views are isolated by workspace and user.
- Invalid or stale saved views fail soft and can be reset.
- Tests cover built-in presets, saved-view serialization, and reset behavior.

## Child Task 3: Duplicate Detection And Attach-Existing Recovery

Backlog: TASK-12093.3

### Scope

Turn duplicate detection outcomes into clear user choices across Quick Ingest and
workspace source add flows.

### Product Behavior

- For local files, identity is based on file checksum when available.
- For remote and web sources, identity is based on normalized URL or existing
  source hash behavior.
- Duplicate result can offer actions:
  - open existing;
  - attach existing source to current workspace;
  - restore from trash when permitted;
  - ingest anyway when policy allows.
- If lookup finds only inaccessible matches, the response uses non-confirming
  language such as `No recoverable duplicate found` and normal ingest remains
  available when policy allows.
- If a hard duplicate constraint prevents ingest, the user-facing result remains
  generic and must not identify the inaccessible duplicate as the reason.

### Boundaries

- Do not reveal inaccessible duplicate titles, paths, owners, or workspace names.
- Do not change global duplicate policy without explicit user action.
- Do not block normal ingest paths when duplicate lookup is unavailable.

### Acceptance Criteria

- Duplicate local file shows recoverable actions when user has access.
- Duplicate URL/source hash shows recoverable actions when user has access.
- Trash duplicates show restore when permitted.
- Inaccessible duplicates do not reveal metadata or confirm that a match exists.
- Tests cover permission-safe duplicate responses and UI action routing.

## Child Task 4: Document Provenance And Storage Metadata Panel

Backlog: TASK-12093.4

### Scope

Expand Document Workspace Info with source provenance and storage facts.

### Product Behavior

The Info panel can show:

- title and extracted document metadata already present today;
- original filename;
- captured URL or source reference;
- added/imported date;
- document-created date when known;
- MIME type;
- file size;
- checksum when available;
- source type;
- tags or keywords already associated with the source;
- workspace memberships limited to the current workspace and other workspaces
  the current user can read;
- readiness summary for metadata, extracted text, FTS, vector, citations, and
  tools;
- original-file availability and open/download actions when permitted.

### Boundaries

- Do not introduce Paperless-specific correspondent, document type, or storage
  path concepts.
- Do not turn the Info panel into a full edit form in this slice.
- Do not expose raw storage paths to users.

### Acceptance Criteria

- Info panel shows provenance and storage facts for media-backed documents.
- Missing facts display stable empty labels, not broken UI.
- Readiness mirrors the server source status projection.
- Workspace memberships are limited to the current workspace and other
  workspaces the current user can read.
- Unauthorized storage actions are hidden or disabled with clear state.
- Tests cover loaded, missing, and permission-limited metadata states.

## Child Task 5: Unified Ingest Entrypoints And Storage Policy Visibility

Backlog: TASK-12093.5

### Scope

Make WebUI drag/drop, extension capture, URL paste, and file upload entrypoints
use one visible ingest model: queue, progress, result, retry, and storage policy
language.

The first extension path is the context-menu Send to tldw URL ingest path. The
sidepanel web-clipper route remains a follow-up extension path unless it already
shares the same ingest queue/result model by the time this task starts.

### Product Behavior

- Entrypoints route to the same Quick Ingest queue/progress concepts where
  practical.
- Storage policy is explicit before submit:
  - index only;
  - keep original;
  - keep original plus derived preview/archive when available.
- MVP may expose only implemented policies and label unavailable options as
  future or unavailable.
- Quota impact is shown before ingest where size is known.
- Results use consistent labels for stored, indexed, skipped duplicate, failed,
  cancelled, and not submitted.

### V1 Storage Policy Labels

| Label | Backend behavior | Entry points | Default | Quota behavior | Result label | Disabled reason |
| --- | --- | --- | --- | --- | --- | --- |
| Index only | Extract and index content without intentionally retaining the original file beyond temporary processing. | URL, web capture, text, and supported file upload paths. | Preserve current preset behavior and show the selected policy explicitly. | Retained-storage quota is not charged for the original; server processing limits can still apply. | Indexed | Media type or endpoint requires retained original. |
| Keep original | Persist the original through existing original-file storage records and `keep_original_file` behavior. | File uploads and remote fetches that already support original retention. | Preserve current preset behavior and show the selected policy explicitly. | Known retained size is shown before submit and checked against existing quota rules. | Stored + indexed | Storage unavailable, quota insufficient, or endpoint does not support retained originals. |
| Keep original + derived preview/archive when available | Persist the original and expose existing derived preview/archive artifacts if the backend already produces them. This task does not create new archive generation. | Only media types and entrypoints with existing derived artifact support. | Not the default unless an existing preset already has equivalent behavior. | Quota includes original plus known retained derived artifact sizes; unknown derived sizes are labeled as calculated after ingest. | Stored + derived + indexed | No derived artifact exists for this media type or backend path. |

### Boundaries

- Do not implement new archive generation as part of this child task unless the
  backend already supports it.
- Do not rework every upload endpoint at once; start with the highest-traffic
  WebUI and extension entrypoints.
- Do not add new dependencies for file picking, drag/drop, or progress.

### Acceptance Criteria

- The main WebUI and extension capture paths use consistent ingest/result labels.
- Extension coverage includes the context-menu Send to tldw URL ingest path.
- Known file sizes show quota impact before submit.
- Unsupported storage policies are not silently selectable.
- Result summaries distinguish stored, indexed, skipped duplicate, failed,
  cancelled, and not submitted.
- Tests cover at least one WebUI and one extension entrypoint.

## Cross-Cutting Requirements

### Permissions And Privacy

- Never reveal duplicate metadata or duplicate existence across user or workspace
  boundaries.
- Storage facts shown in UI must be safe display facts, not raw internal paths.
- Displayed metadata, including workspace memberships, must respect the current
  user's read permissions.
- Actions must respect existing AuthNZ and workspace permissions.

### Error Handling

- Duplicate lookup, saved views, and metadata enrichment should fail open where
  possible.
- Error states should say what still works and what action is available.
- Long-running ingest status remains Jobs-backed where user-visible.

### Accessibility

- Review, saved-view, duplicate, and storage-policy controls must be keyboard
  accessible.
- Icon-only actions need accessible names and tooltips.
- Review state cannot rely on color alone.

### Testing

Each implementation child task should include focused tests for:

- request and response normalization where backend contracts change;
- UI state rendering and action routing;
- permission-limited or degraded states;
- persistence or serialization boundaries;
- no sensitive leakage in duplicate recovery.

### Security Validation

Any backend implementation task must run Bandit on touched Python paths. Pure
TypeScript or Markdown slices should record why Bandit is not applicable.

## Measurable Outcomes And Guardrails

- A user can filter a workspace to Needs review, mark selected sources reviewed,
  reload, and see the same review state.
- A user can filter migrated `unset` sources through the Unreviewed view without
  treating every legacy source as explicit needs-review work.
- Built-in and user-saved source views restore for the same user and workspace
  after reload.
- Duplicate recovery never confirms inaccessible matches in API responses or UI
  copy.
- Accessible duplicate results give at least one useful recovery action when an
  open, attach, or restore action is permitted.
- Document Workspace shows provenance and storage facts without exposing raw
  storage paths.
- WebUI and extension covered entrypoints use the same queue, storage-policy,
  and result labels for equivalent outcomes.

## Rollout

1. Ship persisted source review lifecycle.
2. Add saved source presets/views using the review state.
3. Add duplicate recovery actions.
4. Expand Document Workspace provenance and storage metadata.
5. Normalize ingest entrypoints and storage-policy labels.

This order keeps each slice useful alone and lets later tasks reuse earlier
state without inventing broader abstractions.

## Open Questions

- Which derived preview/archive artifacts should become selectable storage
  policies after backend support exists?
- Should a later global media library add media-level review state outside
  workspace source review?
