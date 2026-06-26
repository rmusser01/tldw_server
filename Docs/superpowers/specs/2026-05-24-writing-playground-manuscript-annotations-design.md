# Writing Playground Manuscript Annotations Design

Date: 2026-05-24
Backlog: TASK-607
Status: Approved for implementation planning; post-review hardening and margin-comment UX
implementation constraints applied

## Purpose

The Writing Playground now has a document-first revision workflow with reviewable proposed edits.
The next product layer is durable comments and annotations for saved manuscript work. V1 should let
writers attach manual notes and AI-authored critique to manuscript scenes, chapters, and projects
without turning the editor into a chat surface or silently mutating draft text.

The approved direction is backend-owned manuscript annotations:

1. Scene range comments anchor to saved manuscript scene text.
2. Chapter and project notes support higher-level critique without fake text ranges.
3. AI can create targeted selected-text comments synchronously.
4. AI can run a bounded scene review through Jobs.
5. Suggested fixes are review material, not direct document mutations; applying them should reuse
   the existing revision proposal workflow.

## Current Evidence

The design builds on the current shared WebUI and extension Writing Playground:

- Shared root: `apps/packages/ui/src/components/Option/WritingPlayground/index.tsx`.
- Revision workflow units:
  - `WritingActionBar.tsx`
  - `WritingRevisionQueue.tsx`
  - `hooks/useWritingRevisions.ts`
  - `writing-revision-types.ts`
  - `writing-revision-utils.ts`
  - `writing-revision-prompt-utils.ts`
- Existing rich/plain editor adapter: `writing-editor-adapter.ts`.
- Existing manuscript UI context:
  - `ManuscriptTreePanel.tsx`
  - `AIAgentTab.tsx`
  - `ResearchTab.tsx`
- Existing manuscript client service: `apps/packages/ui/src/services/writing-playground.ts`.
- Existing backend manuscript API:
  - `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`
  - `tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py`
  - `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py`

The backend already has durable manuscript projects, chapters, scenes, versions, citations,
analyses, search, soft delete, and optimistic version updates. Scenes expose `content_plain`,
`content_json`, `version`, `project_id`, and `chapter_id`.

Important current gap: `WritingPlayground/index.tsx` still contains a source comment for reacting to
`activeNodeId` changes to load scene content into the editor. Annotation implementation must not
pretend unsaved session draft text and saved manuscript scene text are the same thing. V1 should
require comments and AI review to target the saved scene version, or explicitly block until the
scene is saved.

Implementation planning must include saved-scene editor binding as a prerequisite before any
frontend range-comment workflow. Backend CRUD can land before this binding, but manual scene range
comments, selected-text AI critique, inline highlights, and suggested-fix handoff must not ship
until the editor can prove it is showing the saved `manuscript_scenes.content_plain` and `version`
for the active scene.

## Product Model

Annotations are manuscript review objects. They are not chat messages, not ephemeral session
payload, and not direct edit operations.

V1 supports:

- manual annotations from the writer
- AI-authored selected-text critique
- AI-authored bounded scene review comments
- range comments on saved scenes
- desktop margin comment cards for scene range comments
- general notes on chapters and projects
- open/resolved lifecycle
- one optional writer follow-up or decision note
- optional suggested fix text

V1 does not support:

- threaded replies
- collaborative multi-user comment presence
- arbitrary annotations on unsaved session drafts
- direct apply-from-comment document mutation
- full manuscript-wide review runs
- full Google Docs parity such as threaded conversations, comment assignment, or live collaborator
  presence
- backend revision-history replacement for the existing revision proposal queue

## Target UX

### Desktop Margin Comments

On desktop-sized manuscript editing surfaces, scene range comments should behave like editorial
margin notes. Inline highlights remain in the manuscript text, but the primary comment surface is a
right-side margin rail next to the editor, not a detached inspector-only list.

The margin rail should:

- render open scene range comments as compact cards aligned near their anchored text
- keep manuscript text readable by reserving layout space instead of covering text
- expand the active card and collapse non-active cards when vertical space is tight
- sync selection between text highlight, margin card, and inspector list row
- show category, source, status, anchor state, body, follow-up note, and suggested-fix actions
- keep `needs_review` anchors visible with a warning state instead of pretending the card is still
  exactly attached

Resolved comments should leave the margin by default and remain available through the inspector
filters. Chapter and project notes should not render as floating margin cards because they do not
have a text range; they belong in the inspector/list surface.

Margin rail layout contract:

- Derive card anchor positions from the current editor adapter and DOM range measurement, not from
  estimated line counts.
- Sort visible cards by derived anchor top, then `created_at`, then `id` for deterministic order.
- Apply collision avoidance inside the rail: each card's top edge must be at or below the previous
  card's bottom edge plus a small fixed gap.
- Let the active card expand in place and push following cards down; keep non-active cards compact
  when the rail is crowded.
- Keep the rail in the editor scroll context or recompute viewport-relative positions on scroll so
  cards do not drift away from highlighted text.
- If DOM range measurement is unavailable, stale, or fails for the active editor mode, hide the
  margin rail and use the inspector/drawer fallback rather than showing approximate placement.
- Do not animate layout-affecting properties for rail repositioning; use immediate placement or
  transform/opacity transitions only where they do not cause scroll jank.

### Scene Range Comments

When a saved scene is active and the editor reflects that saved scene version, the writer can:

- select text and add a manual comment
- select text and ask AI for a focused critique comment
- view subtle inline highlights for attached annotations
- click a highlight to focus the matching margin card and inspector row
- click a margin card or inspector row to focus or select the anchored text when the anchor is still
  valid
- resolve or reopen the annotation
- add or edit a single follow-up note
- copy an optional suggested fix
- convert an optional suggested fix into a revision proposal

### Chapter And Project Notes

Chapter/project annotations are general notes, not range comments. They are useful for:

- structural concerns
- high-level reader reactions
- chapter goals
- project-level continuity concerns
- task-like editorial notes

They should appear in the same annotations surface, filtered by the active manuscript context.

### Annotations Inspector

Add an `Annotations` inspector tab to the existing Writing Playground inspector. The inspector is
the management surface and responsive fallback, while the margin rail is the primary desktop surface
for active scene range comments. It should list:

1. active scene range comments
2. active chapter notes
3. active project notes

The tab should support filters for:

- status: open or resolved
- category
- source: user, AI selected text, AI scene review
- anchor status for scene comments

The default view should prioritize open annotations for the active scene.

### Responsive And Extension Behavior

The shared WebUI/extension component path should adapt to available width:

- wide desktop: editor plus reserved right margin rail, with inspector available for filters and
  all-context review
- medium width: narrower rail with collapsed cards and explicit active-card expansion
- narrow WebUI or extension options view: hide the rail and use the inspector/drawer as the comment
  surface

The extension must not attempt a cramped two-column margin layout if it makes the manuscript harder
to read. The same annotation state and actions should still be available through the fallback list.

Accessibility requirements:

- Highlighted ranges and margin cards must expose a programmatic relationship with stable ids.
- Keyboard users must be able to move from a highlighted range to its comment card, then back to the
  editor selection when the anchor is attached.
- The inspector/drawer list is the screen-reader fallback when margin positioning is hidden or not
  useful.
- Resolve, reopen, create revision, and copy suggested fix actions must be reachable without a
  pointer.

## Data Model

Add backend-owned manuscript annotations through the per-user ChaChaNotes manuscript database and
the `ManuscriptDBHelper` access layer.

Schema ownership:

- The canonical table definition belongs in the ChaChaNotes schema/migration path, not only in an
  opportunistic helper-side `CREATE TABLE IF NOT EXISTS`.
- SQLite migration coverage is required for the default per-user ChaChaNotes database.
- PostgreSQL migration/compatibility coverage is required if the current ChaChaNotes manuscript
  schema supports the same feature set for PostgreSQL in this area.
- `ManuscriptDBHelper` should expose CRUD, listing, anchor validation, duplicate suppression, and
  AI-review persistence helpers against that schema.
- Implementation planning should explicitly decide whether annotations need sync-log triggers. If
  the project wants annotations to participate in the same sync/export behavior as other manuscript
  entities, add sync logging in the schema slice rather than retrofitting it later.

Suggested table: `manuscript_annotations`.

Core fields:

- `id TEXT PRIMARY KEY`
- `project_id TEXT NOT NULL`
- `target_type TEXT NOT NULL`
- `target_id TEXT NOT NULL`
- `status TEXT NOT NULL`
- `category TEXT NOT NULL`
- `tags_json TEXT`
- `source TEXT NOT NULL`
- `body TEXT NOT NULL`
- `suggested_fix TEXT`
- `followup_note TEXT`
- `metadata_json TEXT`
- `created_at TEXT NOT NULL`
- `last_modified TEXT NOT NULL`
- `deleted INTEGER NOT NULL DEFAULT 0`
- `client_id TEXT NOT NULL`
- `version INTEGER NOT NULL DEFAULT 1`

Scene anchor fields:

- `scene_version INTEGER`
- `anchor_start INTEGER`
- `anchor_end INTEGER`
- `selected_text TEXT`
- `document_fingerprint TEXT`
- `anchor_prefix TEXT`
- `anchor_suffix TEXT`
- `anchor_status TEXT NOT NULL DEFAULT 'scene_level'`

The stored `anchor_status` column is not the V1 source of truth for list/read responses. Treat it
as initialization or future maintenance metadata; normal reads derive anchor status from the current
saved scene text as described below.

Allowed `target_type` values:

- `scene`
- `chapter`
- `project`

Allowed `status` values:

- `open`
- `resolved`

Allowed `source` values:

- `user`
- `ai_selected_text`
- `ai_scene_review`

Allowed category values:

- `style`
- `clarity`
- `pacing`
- `continuity`
- `character`
- `worldbuilding`
- `structure`
- `research`
- `other`

Optional tags should be stored but not drive the V1 UI unless the implementation needs them for
future compatibility. The fixed category is the V1 filtering contract.

V1 caps:

- `body`: 2000 characters.
- `followup_note`: 2000 characters.
- `suggested_fix`: 8000 characters.
- `tags`: up to 10 tags, 48 characters each.
- `selected_text`: 12000 characters for manual/selected-text review requests.
- `anchor_prefix` and `anchor_suffix`: 240 characters each.
- scene review `max_comments`: default 5, maximum 10.

## Anchor Semantics

Scene range annotations need conservative anchoring. They must never silently point at the wrong
text after the scene changes.

At creation time, store:

- current scene `version`
- selected `start` and `end` plain-text offsets
- `selected_text`
- a fingerprint of the saved `content_plain`
- bounded prefix and suffix windows around the selected text

Offset contract:

- Backend APIs store and validate `start` and `end` as Unicode code-point offsets, matching Python
  string indexing.
- Browser and ProseMirror selection utilities must convert DOM UTF-16 code-unit positions to
  backend code-point offsets before submitting annotation requests.
- Tests must include scenes with astral symbols before and inside selected ranges so comments do
  not drift in emoji-containing text.

On read or on explicit refresh, compute anchor state against the current saved `content_plain`:

1. If the scene version and exact range still match `selected_text`, mark `attached`.
2. If the exact range changed, try exact selected-text search.
3. If exactly one selected-text match exists, mark `reattached` and expose the new range.
4. If selected text is absent or ambiguous, try prefix/suffix local anchor matching.
5. If one unambiguous insertion/range location is found, mark `reattached`.
6. Otherwise mark `needs_review` and keep the annotation visible as a scene-level item.

Chapter/project notes always use `scene_level`-equivalent behavior because they have no text range.

Anchor reattachment should be deterministic and side-effect free on normal list/read responses.
V1 should expose derived anchor positions/status without mutating the annotation row. If a later
explicit "refresh anchors" action persists refreshed offsets, it must use expected-version conflict
checks and stay separate from normal list reads. V1 does not persist refreshed offsets.

## Backend API

Add endpoints under `/api/v1/writing/manuscripts`.

### List Annotations

`GET /projects/{project_id}/annotations`

Query filters:

- `target_type`
- `target_id`
- `status`
- `category`
- `source`
- `anchor_status`
- `limit`
- `offset`

Response should include pagination metadata consistent with existing manuscript list endpoints:
`annotations`, `total`, `limit`, `offset`, `has_more`, `next_offset`, and `pagination` using the
same `OffsetPaginationMeta` pattern as `ManuscriptProjectListResponse`.

`anchor_status` is a derived value in V1. The list service should:

1. Apply ordinary SQL-backed filters such as target, status, category, source, and deleted state.
2. Load the current saved scene text needed to derive anchor state for the remaining candidate
   annotations.
3. Compute derived `anchor_status` and derived anchor positions without mutating rows.
4. Apply any `anchor_status` filter to those derived results.
5. Compute `total`, `has_more`, `next_offset`, and page slicing after the derived filter.

If a first backend slice cannot implement accurate derived `anchor_status` filtering and totals,
it should omit the `anchor_status` query filter until that behavior is implemented rather than
returning stale database-filtered counts.

V1 should only enable the `anchor_status` query filter when the request is bounded enough to derive
accurate status without scanning broad project text. Acceptable V1 bounds are:

- `target_type=scene` with a specific `target_id`
- or a post-SQL candidate set under a documented server cap

For broader project/chapter views, the backend should either ignore/omit `anchor_status` as an
available filter or return a validation error that asks the client to narrow the target. The
inspector may still show derived anchor state for returned rows after ordinary pagination, but it
must not claim project-wide `anchor_status` totals unless the backend derived them before
pagination.

### Get Annotation

`GET /annotations/{annotation_id}`

Direct annotation access is in V1. It should:

- validate same-user ownership through the parent project
- return the same annotation response shape used by list/create/update
- derive anchor status and positions side-effect free for scene annotations
- return not found for soft-deleted annotations or annotations whose target is soft-deleted by
  default
- support margin-card focus, scene-review job result links, and suggested-fix handoff without
  forcing clients to search a paginated project list

### Create Annotation

`POST /annotations`

Creates a manual annotation. Request must include:

- `target_type`
- `target_id`
- `category`
- `body`

Scene range annotations also include:

- `scene_version`
- `start`
- `end`
- `selected_text`
- optional client-computed prefix/suffix/fingerprint, or enough information for the backend to
  compute them from current saved scene text

The backend must validate target ownership, target type, range bounds, scene version, and selected
text match before persisting a range annotation.

### Update Annotation

`PATCH /annotations/{annotation_id}`

Supports:

- `status`
- `category`
- `tags`
- `body`
- `suggested_fix`
- `followup_note`
- expected `version`

Use optimistic version checks like the existing manuscript update helpers. Existing manuscript
endpoints pass expected versions through an `expected_version` request header, so annotation update
and delete endpoints should follow that header convention instead of inventing a payload field.

### Delete Annotation

`DELETE /annotations/{annotation_id}`

Soft delete with expected version.

### Review Selected Text

`POST /scenes/{scene_id}/annotations/review-selection`

Synchronous AI critique for one selected range. Request includes:

- expected scene version
- start/end offsets
- selected text
- optional category hints
- optional instruction
- required `provider` and `model` fields matching the active Writing Playground generation
  context
- optional bounded generation options if already supported by the existing writing generation
  conventions

Flow:

1. Validate scene ownership and current saved scene version.
2. Validate selected range and selected text.
3. Call the existing provider/chat layer with a structured annotation prompt.
4. Validate one annotation result.
5. Persist it with `source = ai_selected_text`.
6. Return the created annotation.

### Review Scene

`POST /scenes/{scene_id}/annotations/review-scene`

Jobs-backed scene review. Request includes:

- expected scene version
- category filters or review focus
- maximum comment count
- optional instruction
- required `provider` and `model` fields matching the active Writing Playground generation
  context
- optional bounded generation options if already supported by the existing writing generation
  conventions

Flow:

1. Validate scene ownership and expected version.
2. Enqueue a user-visible Job.
3. Worker loads saved scene text by scene id/version.
4. Worker calls the provider/chat layer.
5. Worker validates bounded structured output.
6. Worker suppresses duplicates against existing open annotations.
7. Worker persists up to the configured max annotations with `source = ai_scene_review`.
8. Job result returns created annotation ids and diagnostics.

## Permissions And Safety

Use the existing auth and rate-limit style from `writing_manuscripts.py`.

Suggested scopes:

- `writing.manuscripts.annotations.list`
- `writing.manuscripts.annotations.create`
- `writing.manuscripts.annotations.update`
- `writing.manuscripts.annotations.delete`
- `writing.manuscripts.annotations.review`

Security and privacy constraints:

- never log manuscript text, selected text, raw model output, or annotation bodies
- validate same-user ownership for project, chapter, scene, and annotation ids
- use parameterized SQL through the existing DB helper patterns
- rate limit AI review endpoints separately from manual CRUD
- cap scene review max comments
- cap annotation body and suggested-fix lengths
- sanitize provider/model errors in HTTP responses and logs

Provider/model handling:

- The frontend should send the active Writing Playground `provider` and `model` to selected-text
  and scene-review endpoints.
- The backend should validate provider/model availability using the same provider metadata and
  "known model" conventions already used by manuscript analysis endpoints.
- If provider/model is missing or unavailable, return a validation/capability error without
  creating annotations or Jobs.
- V1 should not infer provider/model from unrelated chat defaults because that would make review
  output hard for users to reason about.

## AI Output Contract

Selected-text review expects exactly one validated annotation object.

Scene review expects a bounded list of annotation objects.

Suggested model output shape:

```json
{
  "annotations": [
    {
      "category": "clarity",
      "body": "This sentence introduces two causes at once; splitting them would make the beat easier to follow.",
      "quote": "the exact text being discussed",
      "suggested_fix": "Optional replacement suggestion"
    }
  ]
}
```

Validation rules:

- `category` must be in the fixed category set.
- `body` must be non-empty and bounded.
- `suggested_fix` is optional and bounded.
- Scene review comments must include a quote or range that can be anchored unambiguously.
- Unparseable output becomes operation/job diagnostics, not visible annotations.
- Partial streaming output must never create visible annotations.

## Suggested Fix Handoff

Annotations may include `suggested_fix`, but applying text changes belongs to the revision proposal
workflow.

The frontend should provide a "Create revision" action for annotations with a suggested fix. That
action should:

1. Resolve the annotation anchor.
2. If attached/reattached, create a revision proposal with the annotation text and suggested fix.
3. If the anchor needs review, require manual selection or copy-only behavior.
4. Let the existing revision queue own apply/reject/conflict handling.

This avoids duplicate document mutation paths.

## Frontend Architecture

Add small units instead of growing `WritingPlayground/index.tsx`.

Suggested units:

- `writing-annotation-types.ts`
- `writing-annotation-anchor-utils.ts`
- `writing-annotation-api.ts` or additions to `services/writing-playground.ts`
- `hooks/useWritingAnnotations.ts`
- `WritingAnnotationsTab.tsx`
- `WritingAnnotationList.tsx`
- `WritingAnnotationMarginRail.tsx`
- `WritingAnnotationCard.tsx`
- `WritingAnnotationHighlightLayer` or editor-specific highlight helpers
- focused tests under `WritingPlayground/__tests__`

The existing `WritingPlayground` root should orchestrate active manuscript context and pass current
scene/editor state into annotation components. It should not own anchor algorithms directly.

The rich editor path should use the existing TipTap/plain-text adapter boundary. If inline
highlighting or margin-card alignment in TipTap is not safe in the first implementation slice, the
UI can still show annotations in the inspector and mark inline positioning as unavailable rather
than showing wrong anchors.

## Save-State Rule

V1 annotations target saved manuscript records.

If the active editor has unsaved changes relative to the active scene:

- manual scene range comments should require saving first
- selected-text AI review should require saving first
- scene review should require saving first
- chapter/project notes may still be allowed because they are not text-range anchored

This rule prevents durable comments from being anchored to text that only exists in local editor
state.

## Error Handling

Selected range mismatch:

- Return a conflict response.
- Frontend asks the user to refresh/reselect after saving.

Anchor drift:

- Keep the annotation visible.
- Mark `needs_review`.
- Offer copy/manual review instead of selecting possibly wrong text.

Provider unavailable:

- Disable AI review actions.
- Manual annotations remain available.

Scene review job failure:

- Show failed job state with retry.
- Include a concise diagnostics summary without private text.

Duplicate AI comments:

- Suppress comments that match an existing open annotation on the same scene, category, and quote or
  equivalent normalized selected text.
- Record suppression count in diagnostics.

Soft-deleted targets:

- Annotation list should omit annotations for deleted targets by default.
- Direct annotation access should return not found unless an explicit admin/recovery path is later
  designed.

## Testing Strategy

Backend unit tests:

- create/list/update/soft-delete annotation helper methods
- target ownership validation
- scene range validation
- expected-version conflicts
- anchor exact attach
- exact selected-text reattach
- prefix/suffix reattach
- ambiguous anchor fallback to `needs_review`
- duplicate suppression
- soft-deleted target behavior

Backend integration tests:

- annotation CRUD endpoints
- filtered list pagination
- selected-text review endpoint with mocked provider
- selected-text conflict on stale scene version
- scene review enqueue endpoint
- Jobs worker success and failure paths

Frontend tests:

- service/client methods
- annotations tab empty/loading/error states
- margin rail renders active scene range comments on wide layouts
- narrow layouts collapse margin comments into the inspector/drawer fallback
- highlight, margin card, and inspector row focus stay in sync
- margin cards use deterministic ordering and collision avoidance when multiple comments target
  nearby text
- failed or unavailable DOM measurement hides the rail and keeps inspector/drawer actions available
- keyboard navigation moves between highlight, margin card, inspector row, and editor selection
- comment controls expose accessible names and are reachable without pointer input
- active scene annotations sorted before chapter/project notes
- status/category/source filters
- resolve/reopen
- follow-up note editing
- anchor needs-review state
- click annotation focuses attached text when possible
- suggested fix creates a revision proposal
- AI actions disabled when generation is unavailable
- unsaved scene text blocks range comments and AI review
- WebUI and extension route parity still use shared Writing Playground

Manual/browser verification:

- create a project/chapter/scene
- save scene text
- select text and add a manual comment
- verify the comment appears as a desktop margin card aligned with the highlighted text
- select text and request AI comment
- start scene review and observe job completion
- resolve/reopen an annotation
- edit scene text so an anchor drifts and verify `needs_review`
- narrow the viewport and verify margin comments collapse into the inspector/drawer fallback
- verify keyboard-only highlight/comment navigation and comment actions
- create a revision proposal from a suggested fix
- repeat core route smoke in extension options when extension build is available

Bandit:

- Run Bandit on touched backend Python paths during implementation.
- Skip only for documentation-only design/planning tasks and record the skip.

## Rollout Plan

This design should become several PR-sized implementation tasks.

### Stage 0: Saved Scene Editor Binding Prerequisite

Wire the active manuscript scene into the editor with explicit saved-version awareness, or create
an equivalent readiness contract that lets the annotation UI know whether the editor reflects the
saved scene.

Success criteria:

- selecting a scene loads saved `content_plain`, TipTap content when available, and scene `version`
- saving scene edits updates the saved scene and visible version
- the UI can detect unsaved divergence from the saved scene
- range annotation actions are disabled unless the editor reflects a saved scene version

### Stage 1: Backend Annotation CRUD

Add schemas, DB helper methods, CRUD/list endpoints, tests, and docs. No AI review or frontend
integration beyond client methods.

Success criteria:

- annotations persist as manuscript-owned records
- scene/chapter/project targets validate ownership
- version conflicts and soft delete match existing manuscript behavior

### Stage 2A: Frontend Annotation Foundation

Add client methods, annotation hook, inspector tab/list, manual create/edit/resolve/reopen, direct
annotation lookup, and saved scene gating. This stage depends on Stage 0 for scene range comments;
without Stage 0, it may only ship chapter/project notes and non-range annotation list UI.

Success criteria:

- writer can create and resolve scene range comments and chapter/project notes
- inspector list supports active scene, chapter, and project annotation management
- narrow WebUI/extension layouts expose the same actions through the inspector/drawer surface
- anchor drift is visible and non-destructive
- WebUI and extension share the same component path

### Stage 2B: Desktop Margin Rail And Focus Sync

Add the desktop margin rail, highlight/card/inspector focus synchronization, deterministic card
placement, collision avoidance, responsive collapse, and keyboard/a11y behavior.

Success criteria:

- desktop scene range comments appear in a margin rail, not only in the inspector list
- nearby comments do not overlap and remain in deterministic order while scrolling
- active-card expansion does not obscure manuscript text
- failed measurement falls back to the inspector/drawer surface
- keyboard and screen-reader users can navigate equivalent comment actions
- WebUI and extension use the same annotation state and action components

### Stage 3: Selected-Text AI Critique

Add synchronous backend review-selection endpoint, structured prompt validation, frontend action,
and mocked-provider tests.

Success criteria:

- writer can select saved scene text and ask AI for one persisted annotation
- stale scene/version conflicts are handled clearly
- provider failures do not create partial annotations

### Stage 4: Jobs-Backed Scene Review

Add review-scene enqueue and worker path, duplicate suppression, frontend progress state, and job
diagnostics.

Success criteria:

- scene review creates a bounded set of annotations
- duplicate comments are suppressed
- failure/retry states are visible without leaking manuscript text

### Stage 5: Suggested Fix To Revision Proposal

Connect annotation suggested fixes to the existing revision queue.

Success criteria:

- suggested fixes become reviewable revision proposals
- anchor drift blocks automatic proposal creation
- the revision queue remains the only automatic document mutation path

## Open Questions For Implementation Planning

- Which exact optional generation settings beyond `provider` and `model` should V1 expose?
- Should chapter/project notes be creatable from the manuscript tree context menu in the first
  frontend slice, or only from the annotations tab?

## Definition Of Done For This Design

- Design covers durable manuscript-owned annotations.
- Design covers scene range comments and chapter/project notes.
- Design covers manual and AI-authored annotation flows.
- Design defines anchoring, drift handling, and saved-scene gating.
- Design defines backend APIs, permissions, safety constraints, and Jobs usage.
- Design defines frontend components, error handling, testing, rollout stages, and non-goals.
