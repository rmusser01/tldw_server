# Browser Extension Web Clipper Design

## Summary

This design adds an Evernote-style web clipper to the browser extension.

The clipper is capture-first and save-first. Users should be able to launch it from the toolbar, context menu, or keyboard shortcut, choose a clip type, review the result, add filing metadata, and save it to `Note`, `Workspace`, or `Both`.

The user-facing destination choices are equal, but the internal storage model is intentionally asymmetric: rich clips are canonically note-backed so the system can reuse the existing notes attachment, keyword, folder, and metadata capabilities. Workspace saves are modeled as placements or linked summaries on top of that richer note-backed record.

OCR and VLM are optional enrichments, not the primary flow. When enabled, they should add a concise inline summary to the saved item and preserve the full structured analysis alongside the clip metadata.

## Goals

- Deliver an Evernote-like web clipper experience inside the extension.
- Support the first-class clip types confirmed with the user:
  - `Bookmark`
  - `Article`
  - `Simplified article`
  - `Full page`
  - `Screenshot`
  - `Selected text`
  - `Selected area screenshot`
- Keep the primary user flow short:
  - capture
  - review
  - title/comment/tags
  - choose destination
  - save
- Let the user save to `Note`, `Workspace`, or `Both`.
- Reuse existing note and workspace systems where possible instead of creating a new backend storage domain in V1.
- Support optional OCR and VLM analysis with server-first execution.
- Use progressive enhancement so baseline clipper behavior works broadly and richer capture modes appear where the browser supports them.

## Non-Goals

- Building a brand-new first-class backend `clip` entity in V1.
- Merging the clipper into Quick Ingest in V1.
- Supporting audio/video recording workflows in V1.
- Supporting fully general desktop capture beyond still-image capture.
- Guaranteeing identical feature parity across every browser for advanced screenshot modes.
- Treating OCR or VLM as mandatory parts of every save.
- Creating a clip history manager or clip inbox in V1.

## Requirements Confirmed With User

- The product reference is the Evernote web extension, not a generic capture platform.
- The primary saved object should be note-first rather than media-first.
- Phase 1 should support the core clipper set plus selection clip types.
- `Note` and `Workspace` should be equal user-facing destination choices.
- The user should be able to save to `Note`, `Workspace`, or `Both` at save time.
- OCR and VLM output should be stored both as a concise inline result and as full structured metadata.
- The design should prefer server-first OCR and VLM execution.
- The launch model should be capture-first from toolbar, context menu, and keyboard shortcut.
- Progressive enhancement is preferred over strict cross-browser parity.

## Current State

- The extension already has a background-centered architecture in [`apps/packages/ui/src/entries/background.ts`](../../../../apps/packages/ui/src/entries/background.ts).
- The extension already has screenshot primitives in [`apps/packages/ui/src/libs/get-screenshot.ts`](../../../../apps/packages/ui/src/libs/get-screenshot.ts).
- The extension already supports sidepanel vision-style chat flows and image attachments to chat.
- The backend already supports standalone notes with:
  - content
  - keywords
  - folders / keyword collections
  - optimistic locking
  - file attachments
- The backend already supports workspace notes, but they are lighter-weight than standalone notes.
- The extension already has notes and workspace UIs, so the clipper can hand users into existing surfaces after save.

## Design Constraints Discovered During Review

### Destination Symmetry Constraint

The user wants `Note`, `Workspace`, and `Both` to feel like equal save choices. The current backend, however, does not provide equal storage capabilities for those destinations.

Standalone notes already support richer attachment and organization behavior than workspace notes. V1 therefore should not pretend those systems are identical internally.

### Destination Mapping Constraint

The repo already uses the word `collection` for multiple unrelated domains, including notes keyword collections and reading collections. Neither is the right destination abstraction for rich clipped notes in V1.

To avoid routing ambiguity, V1 must lock the review-sheet destinations to:

- `Note`
- `Workspace`
- `Both`

`Collection` can be added later only after a single concrete clip-placement backend target is chosen.

### Attachment Richness Constraint

Screenshots, selected-area captures, and full-page clip payloads need durable attachment support. The note system already has attachment handling; workspace notes do not expose the same richness.

This makes canonical note-backed storage the safest V1 choice for rich clips.

### Multi-Step Save Constraint

Saving a clip is not a single atomic mutation. A successful save may involve:

- creating the canonical note
- attaching keywords and folders
- uploading one or more attachments
- adding a workspace placement
- optionally running OCR and VLM
- adding enrichment summaries without overwriting user-authored content

The design must explicitly support partial success and recovery instead of assuming one all-or-nothing request.

### Region Capture Reliability Constraint

`Selected area screenshot` is the highest-risk clip type in phase 1. Zoom level, device pixel ratio, scroll position, overlays, and iframe boundaries all create reliability problems.

V1 should scope this to visible-region selection only and define a fallback path when region capture fails.

### Content Budget Constraint

Full-page captures plus OCR text plus VLM output can create bloated saved notes. The visible note body should remain readable and user-friendly rather than becoming a raw dump.

The design needs a strict content-budget rule for what goes inline and what is stored as structured metadata or attachments.

### Extraction Fallback Constraint

`Article` and `Simplified article` clipping will fail on some sites. The design needs a deterministic fallback ladder so users understand what actually got captured.

### Privacy And Restricted Pages Constraint

Server-first OCR and VLM mean captured content may be uploaded to the configured `tldw_server`. The review sheet must disclose this clearly. The clipper should also refuse or degrade on browser-internal and otherwise restricted pages.

## Approaches Considered

### Approach 1: Note-Backed Clipper With Destination Placements

Create one normalized clip draft in the extension and one canonical rich saved record backed by the notes system. Let the user save to `Note`, `Workspace`, or `Both`, but treat workspace saves as placements or linked summaries over the richer note-backed clip.

Pros:

- Reuses the existing richer notes data model and attachment system
- Keeps clip attachment handling in one place
- Minimizes backend scope for V1
- Still supports equal user-facing destination choices
- Makes OCR/VLM enrichment storage straightforward

Cons:

- Internal storage is not perfectly symmetric with the visible save options
- Requires explicit lifecycle rules for removing placements versus deleting the canonical note

### Approach 2: Destination-Specific Storage

Treat `Note`, `Workspace`, and `Both` as fully separate persistence paths with no canonical clip record.

Pros:

- Purest interpretation of equal destination choices
- Each destination owns its own content directly

Cons:

- Duplicates attachment and metadata work
- Workspace save behavior becomes much poorer unless new backend support is added
- Raises the risk of search duplication and divergence immediately

### Approach 3: New Dedicated Clip Object

Create a new clip domain in the backend and derive notes, workspace records, and enrichments from it.

Pros:

- Cleanest long-term abstraction
- Best eventual separation of concerns

Cons:

- Much larger backend scope
- Reinvents capabilities already present in notes
- Not needed for a phase-1 Evernote-style clipper

## Recommendation

Use **Approach 1**.

Build the clipper as a note-backed rich capture system with optional workspace placements. Preserve equal user-facing save choices, but let the implementation lean on the existing richer note model for attachments, metadata, keywords, and filing behavior.

## Proposed Architecture

### Product Shape

V1 adds a dedicated web clipper flow to the extension.

The clipper should behave like a compact filing tool, not like a generic developer-facing capture inspector. The primary workflow is:

1. launch the clipper
2. choose clip type
3. review the captured result
4. edit title, comment, tags, and destination
5. save

OCR and VLM live in an `Enhance` section and are optional.

### Core Components

Recommended extension-side units:

- `ClipperLauncher`
- `ClipTypeCapabilityResolver`
- `ClipCaptureAdapterRegistry`
- `ClipDraftBuilder`
- `ClipReviewSheet`
- `ClipSaveRouter`
- `ClipEnhancementRunner`

Responsibilities:

- `ClipperLauncher`
  - opens the clipper from toolbar, context menu, or shortcut
  - hydrates active-tab metadata

- `ClipTypeCapabilityResolver`
  - computes which clip types are available for the current page and browser
  - explains disabled states

- `ClipCaptureAdapterRegistry`
  - maps clip types to capture adapters

- `ClipDraftBuilder`
  - normalizes all clip types into one shared draft model

- `ClipReviewSheet`
  - owns user-editable metadata and enhancement toggles
  - shows what will be saved

- `ClipSaveRouter`
  - coordinates note creation, attachment upload, placement creation, and final routing

- `ClipEnhancementRunner`
  - handles optional OCR and VLM execution before final save completion

### Canonical Draft Model

Every clip type should normalize into one `ClipDraft` shape.

Suggested fields:

- `clip_id`
- `clip_type`
- `source_url`
- `source_title`
- `captured_at`
- `page_domain`
- `page_text`
- `selection_text`
- `clean_article_text`
- `html_excerpt`
- `image_assets`
- `comment`
- `tags`
- `destination_mode`
- `note_destination`
- `workspace_destination`
- `enhancements`
- `capture_metadata`

Suggested `capture_metadata` fields:

- browser target
- tab id if available
- scroll position where relevant
- device pixel ratio where relevant
- region bounds for selected-area screenshot
- fallback path actually used

### Internal Save Model

The design should explicitly distinguish between **visible destination semantics** and **internal persistence semantics**.

Visible semantics:

- `Note`
- `Workspace`
- `Both`

Internal semantics:

- create a canonical note-backed rich clip record for every rich clip
- attach clip assets and analysis to that canonical note-backed record
- if the user selected workspace placement, create a linked placement or summary record

This should not be surfaced as “we secretly created a note.” It is an implementation detail required by the current backend capability shape.

### Lifecycle Rules

The design must define how deletes and edits behave:

- removing a workspace placement does not delete the canonical clip note
- deleting the canonical clip note removes or invalidates downstream placements
- editing the canonical note-backed clip is the only supported way to change clip content in V1
- workspace placements are read-only derived summaries in V1, not an independent editable source of truth
- placement updates flow one-way from canonical note to workspace placement after a successful note save
- workspace-only saves still keep the canonical note visible in the regular notes surface as the clip's source of truth, with a placement badge or backlink to the workspace
- if `Both` was chosen, the UI should make it clear whether the user is removing one placement or deleting the clip entirely

These rules must be part of the implementation plan, not left implicit.

## Clip Types

### Bookmark

Capture:

- page title
- source URL
- domain / favicon metadata when available
- optional user comment and tags

Save behavior:

- visible note body is minimal
- attachments are usually unnecessary

### Article

Capture:

- readability-style article extraction first
- fallback to main-content extraction
- fallback to broader page text if needed

Save behavior:

- body should prioritize readable article text
- source block stays visible at the top

### Simplified Article

Capture:

- stricter cleanup than `Article`
- preserve only the most article-like content

Fallback ladder:

1. simplified readability extraction
2. article extraction
3. if quality is too low, mark the clip type unavailable and offer `Article` or `Full page`

### Full Page

Capture:

- broader page text and page metadata
- optional screenshot attachment if supported

Save behavior:

- store a readable excerpt in the note body
- preserve the larger raw capture as structured metadata or attachment when needed

### Selected Text

Capture:

- exact selected text
- page title and URL
- optional surrounding context excerpt

Save behavior:

- selection becomes the primary content block
- source metadata is always preserved

### Screenshot

Capture:

- visible screenshot of the active page or supported still capture source

Save behavior:

- screenshot is an attachment
- OCR/VLM are optional enrichments

### Selected Area Screenshot

Capture:

- user-selected region inside the visible viewport only in V1

Save behavior:

- selected region image is the primary attachment
- preserve region bounds in metadata

Phase-1 scope rule:

- V1 does **not** promise arbitrary full-document or cross-scroll region capture
- if region capture fails, the clipper should offer a fallback to `Screenshot` or `Selected text`

## Clipper UX

### Entry Points

Primary launch surfaces:

- toolbar button
- context menu
- keyboard shortcut

These should all open the same compact clipper panel rather than separate workflows.

### Panel Layout

Top section:

- current page title and domain
- clip type selector
- quick preview thumbnail or excerpt

Main fields:

- `Title`
- `Comment`
- `Tags`
- `Destination`

Destination options:

- `Note`
- `Workspace`
- `Both`

Conditional destination controls:

- folder selector when `Note` is involved
- workspace selector when `Workspace` is involved

Enhancement section:

- `Run OCR`
- `Run visual analysis`
- disclosure that OCR and VLM send captured content to the configured `tldw_server`

Bottom actions:

- `Save clip`
- `Save and open`
- `Cancel`

### Review Principle

The panel is a filing review, not a technical debug sheet. Users should mostly see:

- what was captured
- what it will be called
- where it will go
- whether enhancements will run

## Save Semantics

### Save Is A Staged Workflow

The clipper save must be treated as a multi-step workflow, not a single mutation.

Recommended staged order:

1. create canonical note-backed clip record
2. apply keywords and folder placement
3. upload attachments
4. create workspace placement if requested
5. queue OCR and VLM if requested
6. persist structured enrichment results
7. update the dedicated machine-managed enrichment section only when the optimistic-lock rules below allow it

### Save Outcome States

Recommended states:

- `saved`
- `saved_with_warnings`
- `partially_saved`
- `failed`

Examples:

- note created, attachment upload failed => `saved_with_warnings`
- note created, workspace placement failed => `partially_saved`
- capture succeeded but note creation failed => `failed`

The UI should always tell the user what succeeded and what did not.

### Partial Success Policy

Prefer partial success over full rollback.

That means:

- if the canonical note was created successfully, preserve it
- if placement creation fails, surface retry actions
- if OCR/VLM fails, keep the saved clip and mark enhancement failure

### Idempotency And Retry Rules

Retries must be safe after network failures, browser restarts, and partial-success saves.

Required rules:

- `clip_id` is the idempotency key for the full clip save workflow
- canonical note creation reuses `clip_id` as the client-provided note `id`
- keyword and folder sync is idempotent by canonical `note_id` plus target membership
- attachment uploads use deterministic slot names and content-hash-aware deduplication so a retried upload does not create duplicate files
- workspace placement creation is idempotent by `(clip_id, workspace_id)`
- OCR and VLM jobs are idempotent by `(clip_id, enhancement_type, source_note_version)`
- a stage retry must update or reuse the existing saved artifact for that stage instead of creating a second note, placement, or enrichment record

If the client loses the final response, re-running `Save clip` for the same `clip_id` should converge on one canonical note, zero or one workspace placement, and zero or one active OCR result plus zero or one active VLM result for the same source note version.

### Enrichment Writeback Safety

`Save and open` means the user may begin editing the note before OCR or VLM finishes.

V1 must therefore treat post-save enrichment as append-only and machine-managed:

- OCR and VLM may write into a dedicated enrichment section only
- they must never rewrite the user comment, source citation block, or primary clipped content block
- if the canonical note version changed after save, structured enrichment metadata may still be stored, but the visible note body must not be rewritten automatically
- when a version mismatch happens, the UI should surface that enrichment completed but inline refresh requires a non-destructive follow-up action

This keeps asynchronous enrichment from clobbering user edits.

## Content Model For Saved Notes

### Visible Body Structure

The saved note body should look authored and readable.

Recommended body shape:

1. title
2. optional user comment
3. source citation block:
   - page title
   - URL
   - capture date
   - clip type
4. main clipped content
5. concise OCR summary if enabled
6. concise VLM summary if enabled

### Content Budget Rule

Do not put full raw OCR output or full VLM output inline by default.

Deterministic V1 thresholds:

- the visible main clipped content block is capped at 12,000 characters
- inline OCR summary is capped at 1,500 characters
- inline VLM summary is capped at 1,000 characters
- combined machine-generated inline enrichment is capped at 2,500 characters total
- if extracted page text exceeds 20,000 characters, preserve the full extract outside the visible body and inline only the first 12,000 characters
- full raw OCR output is always stored outside the visible body in structured metadata or a text attachment
- full VLM output is always stored outside the visible body in structured analysis metadata

Truncation policy:

- preserve the full user comment and source citation block
- truncate on a paragraph boundary when possible
- if no paragraph boundary exists near the limit, truncate at the hard character boundary
- append a short marker such as `Truncated. Full extract attached.` when spillover occurs

This keeps the saved note readable while preserving the full machine-readable result.

## OCR And VLM

### Execution Model

OCR and VLM are server-first.

The clipper should send captured material to the configured `tldw_server` when the user enables those options. Local-only OCR can remain a future fallback path, but the design should optimize for server execution first.

### Output Model

Because the user requested both inline and structured output:

- include a concise inline `Captured text` or `Visual summary` section in the visible note
- store the full OCR and VLM output alongside the clip metadata
- keep enrichment writeback within the dedicated machine-managed section described above

### Failure Behavior

- OCR failure does not block save
- VLM failure does not block save
- the saved clip should record which enhancement failed

## Browser Capability Handling

### Progressive Enhancement

Baseline clip types should work wherever the extension already has the needed access:

- `Bookmark`
- `Article`
- `Simplified article`
- `Full page`
- `Selected text`

Enhanced clip types can be enabled only where browser support is clean enough:

- `Screenshot`
- `Selected area screenshot`

### Disabled-State Policy

Unsupported clip types should remain visible but disabled with a clear explanation.

Examples:

- restricted browser page
- missing capture permission
- region capture not available on this browser

### Restricted Pages

The clipper should explicitly block or degrade on:

- browser internal pages
- extension pages
- other origins where injection or capture is restricted

## Capture Fallback Matrix

Recommended fallback rules:

- `Bookmark`
  - no fallback needed

- `Selected text`
  - if no selection exists, offer `Article` or `Bookmark`

- `Simplified article`
  1. simplified article extraction
  2. article extraction
  3. offer `Article` or `Full page`

- `Article`
  1. readability extraction
  2. main-content extraction
  3. full-page text extraction
  4. `Bookmark` if page text is unusable

- `Full page`
  1. full-page text extraction
  2. broader DOM text extraction
  3. `Bookmark` fallback

- `Selected area screenshot`
  1. region capture
  2. visible screenshot
  3. selected text if that was the original intent and selection exists

The UI should indicate the actual capture result, not just the requested clip type.

## Search And Duplication Rules

When the user saves to `Both`, the system must avoid creating indistinguishable duplicate search results.

Recommended rule:

- index the canonical note-backed clip fully
- index workspace placements as references or summaries
- search results should expose placement badges instead of duplicating full-body content where possible

## Error Handling

### Capture Failures

- explain what failed
- offer retry
- offer a nearby fallback clip type

### Save Failures

- preserve successful stages
- expose retry for failed stages
- make the final state legible

### Enhancement Failures

- do not discard the saved clip
- record the failure in metadata
- offer re-run later if the product adds that capability

## Recommended Phasing

### Phase 1

- clipper launcher
- clip type chooser
- review sheet
- core confirmed clip types
- note-backed rich save
- optional workspace placement
- OCR and VLM enhancement hooks
- progressive enhancement for screenshot modes
- visible-region-only selected-area screenshot

### Not In Phase 1

- recording workflows
- clip history manager
- fully generic desktop capture system
- new backend clip entity
- Quick Ingest unification
- fully general cross-document region capture

## Testing Strategy

### Unit Tests

- clip type availability rules
- clip draft normalization
- fallback ladder selection
- destination routing
- save outcome classification
- note body formatting under content-budget rules

### Integration Tests

- create note-backed clip from each core clip type
- save to workspace placement
- save to both with partial failure handling
- OCR and VLM merge behavior
- attachment upload and warning states
- idempotent retry after partial success
- post-save enrichment with optimistic-lock conflict

### Extension Integration Tests

- toolbar launch
- context menu launch
- shortcut launch
- permission denial and restricted-page behavior
- article extraction on representative page structures

### E2E Tests

- bookmark clip
- article clip
- simplified article fallback
- selected text clip
- screenshot clip
- selected-area screenshot on supported browser
- save to note
- save to workspace
- save to both with one destination failing
- OCR enabled save
- VLM enabled save

## Locked Implementation Notes

- V1 review-sheet destinations are exactly `Note`, `Workspace`, and `Both`
- `Note` uses the existing notes system plus folder filing
- `Workspace` uses the existing workspace note surface at `/api/v1/workspaces/{workspace_id}/notes`
- notes keyword collections and reading collections are explicitly out of scope as clip destinations in V1
- canonical note deletion invalidates or removes workspace placements
- placement removal does not delete the canonical note

## Recommendation Recap

The clipper should ship as a compact Evernote-style filing workflow with a note-backed internal storage model, optional workspace placements, and OCR/VLM as enrichments rather than the main action.

That preserves the user experience the user asked for while keeping V1 aligned with the current extension and backend capabilities.
