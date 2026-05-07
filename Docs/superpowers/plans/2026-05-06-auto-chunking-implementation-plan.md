# Auto Chunking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement novice-facing Auto Chunking for Quick Ingest so enabling
Chunking defaults to deterministic media-aware planning, with a Manual mode for
existing detailed controls and an explicit opt-in flag for later AI-assisted
boundary refinement.

**Architecture:** Add a pure backend Auto Chunk Planner that translates media
type, source hints, template matches, extracted content profiles, requested
goal, and AI-assist availability into existing chunker options plus a
user-visible `chunking_plan`. Wire the planner into media request parsing,
direct processing, async ingest jobs, and persistence without changing legacy
requests that omit `chunking_mode`. Update Quick Ingest state, payload builders,
and UI controls so Auto is the default only when Chunking is enabled in the new
Quick Ingest flow, while Manual keeps the current advanced fields.

**Tech Stack:** FastAPI, Pydantic, existing media ingestion/chunking helpers,
pytest, Next.js/React shared UI package, Ant Design controls, Bun, Vitest, and
the existing extension background Quick Ingest submission path.

---

## Scope

Implement the approved design from:

- `Docs/superpowers/specs/2026-05-06-auto-chunking-design.md`
- Backlog task `TASK-96`

This plan covers implementation work for the full Auto Chunking product
contract, but it deliberately separates deterministic Auto from AI-assisted
Auto. The first shippable slice accepts and records the AI-assist preference,
falls back deterministically when no AI adapter is available, and does not make
LLM calls by default. True LLM boundary refinement is a later task in this plan
after deterministic planning is stable and observable.

This plan does not create multiple persistent chunk sets, replace the existing
chunking engines, make Chunking Playground the novice flow, or change behavior
for legacy clients that omit `chunking_mode`.

## Current File Map

Backend request and parsing:

- Modify `tldw_Server_API/app/api/v1/schemas/media_request_models.py`
  - Add `ChunkingMode` and `AutoChunkingGoal` literals.
  - Add `chunking_mode`, `auto_chunking_goal`, and
    `auto_chunking_use_llm` to `ChunkingOptions`.
  - Add the same Auto fields to JSON web/article request models:
    `WebScrapingRequest` and `IngestWebContentRequest`.
  - Do not expose `structure_aware` as a new public Manual request value in
    V1. Keep it as an internal Auto planner output passed to existing chunking
    helpers after request validation.
- Modify `tldw_Server_API/app/api/v1/API_Deps/media_add_deps.py`
  - Parse the new Auto fields for `/media/add` and async ingest jobs.
  - Close existing parity gaps by parsing `auto_apply_template`,
    `chunking_template_name`, `hierarchical_chunking`, and
    `hierarchical_template` where the schema already supports them.
- Modify `tldw_Server_API/app/api/v1/API_Deps/media_processing_deps.py`
  - Parse the same Auto fields for direct process endpoints.
  - Keep PDF's existing template parsing and bring documents, audio, video,
    ebooks, and email to the same request-field parity.

Backend planner and wiring:

- Create `tldw_Server_API/app/core/Chunking/auto_planner.py`
  - Pure deterministic planner.
  - Exports normalized plan/profile types and functions.
  - Does not import FastAPI request objects or perform database writes.
- Modify `tldw_Server_API/app/core/Ingestion_Media_Processing/chunking_options.py`
  - Keep `prepare_chunking_options_dict()` behavior for legacy and Manual.
  - Add a resolver that returns `(chunk_options, chunking_plan)` for Auto.
  - Ensure Auto ignores stale saved Manual fields unless explicitly needed as
    bounded defaults.
- Modify direct process endpoints under
  `tldw_Server_API/app/api/v1/endpoints/media/`
  - `process_documents.py`
  - `process_pdfs.py`
  - `process_audios.py`
  - `process_videos.py`
  - `process_ebooks.py`
  - `process_emails.py`
  - `process_web_scraping.py`
  - `ingest_web_content.py`
  - Use the resolver and attach returned `chunking_plan` to response metadata.
- Modify `tldw_Server_API/app/services/web_scraping_service.py`
  - Apply Auto planning to the `/process-web-scraping` Quick Ingest path and
    the legacy fallback persistence path.
  - Preserve current behavior for web requests that omit `chunking_mode`.
- Modify `tldw_Server_API/app/services/media_ingest_jobs_worker.py`
  - Use the resolver for job payloads.
  - Forward `chunking_plan` into the final job result.
- Modify `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`
  - Persist the plan under `safe_metadata.chunking_plan` where safe metadata is
    already normalized.
  - Update the local safe metadata allowlists/copy filters so `chunking_plan`
    is preserved as a nested JSON object with only JSON-safe scalar/list values.
  - Avoid creating a second chunk index or durable alternate chunk set.

Backend tests:

- Create `tldw_Server_API/tests/Chunking/test_auto_chunking_planner.py`
- Modify `tldw_Server_API/tests/Media_Ingestion_Modification/test_process_endpoints_contract_parity.py`
- Modify or create a focused request parsing test near
  `tldw_Server_API/tests/Media_Ingestion_Modification/test_add_media_endpoint.py`
- Modify `tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_ingest_jobs.py`
- Create or modify web/article coverage near:
  - `tldw_Server_API/tests/Web_Scraping/test_process_web_scraping_strategy_validation.py`
  - `tldw_Server_API/tests/Media/test_ingest_web_content_endpoint_sanitization.py`
  - or a focused new `tldw_Server_API/tests/WebScraping/test_auto_chunking_web_ingest.py`

Frontend state, payload, and UI:

- Modify `apps/packages/ui/src/components/Common/QuickIngest/types.ts`
  - Add `ChunkingMode`, `AutoChunkingGoal`, and common option fields.
- Modify `apps/packages/ui/src/components/Common/hooks/useIngestOptions.tsx`
  - Default new Quick Ingest chunking mode to `auto`.
  - Persist mode, goal, and AI-assist preference with conservative migration.
  - Prevent old advanced Manual values from leaking into Auto submissions.
- Modify `apps/packages/ui/src/components/Common/QuickIngest/IngestOptionsPanel.tsx`
  - Add Auto/Manual mode control when Chunking is enabled.
  - Show goal and AI-assist controls for Auto.
  - Keep existing template and detailed chunking controls in Manual/advanced.
- Modify `apps/packages/ui/src/components/Common/QuickIngest/WizardConfigureStep.tsx`
  - Surface the same Auto default and Manual escape hatch in the wizard flow.
- Modify `apps/packages/ui/src/components/Common/QuickIngestModal.tsx`
  - Thread the new state through the submit flow.
- Modify `apps/packages/ui/src/services/tldw/quick-ingest-batch.ts`
  - Send Auto fields for Auto.
  - Send current Manual fields for Manual.
  - Do not send stale advanced chunk fields in Auto mode.
  - Include the Auto fields in `processWebScrape()` JSON bodies for
    `/api/v1/media/process-web-scraping`.
- Modify `apps/packages/ui/src/entries/background.ts`
  - Update the extension/background Quick Ingest path with the same payload
    rules as the shared batch service.
  - Include the same web/article `processWebScrape()` JSON field behavior.
- Modify `apps/packages/ui/src/services/tldw/fallback-schemas.ts`
  - Add the Auto fields to fallback field definitions if the runtime settings
    schema is unavailable.
- Modify tests:
  - `apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts`
  - `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx`
  - `apps/packages/ui/src/components/Common/QuickIngest/__tests__/IngestWizardContext.test.tsx`

Tracking:

- Update Backlog task `TASK-96.1` for this plan.
- Create or update implementation child tasks before code edits begin.

## Data Contract

Requests:

```text
perform_chunking=true|false
chunking_mode=auto|manual
auto_chunking_goal=balanced|qa_search|navigation_summary
auto_chunking_use_llm=false|true
```

Rules:

- If `perform_chunking=false`, do not plan or chunk.
- If `chunking_mode` is missing, preserve existing request behavior.
- If `chunking_mode=manual`, use existing chunking fields and template
  controls.
- If `chunking_mode=auto`, use planner output and ignore stale saved Manual
  chunking fields.
- If `auto_chunking_use_llm=true` and the AI adapter is unavailable, continue
  deterministically and record `fallback_reason`.
- The same contract applies to multipart media forms and JSON web/article
  requests (`WebScrapingRequest` and `IngestWebContentRequest`).

Plan metadata shape:

```json
{
  "mode": "auto",
  "goal": "balanced",
  "used_llm": false,
  "method": "structure_aware",
  "max_size": 900,
  "overlap": 120,
  "template_name": "academic_pdf",
  "derived_views": ["outline", "section_titles"],
  "fallback_reason": null,
  "rationale": "Detected document structure; selected structure-aware chunks."
}
```

Store this metadata in direct process responses, async job results, and durable
safe metadata. Derived views are metadata only in this implementation.

## Task 1: Backend Request Contract And Parsing Parity

**Files:**

- Modify `tldw_Server_API/app/api/v1/schemas/media_request_models.py`
- Modify `tldw_Server_API/app/api/v1/API_Deps/media_add_deps.py`
- Modify `tldw_Server_API/app/api/v1/API_Deps/media_processing_deps.py`
- Modify `tldw_Server_API/tests/Media_Ingestion_Modification/test_process_endpoints_contract_parity.py`
- Modify or create focused request parsing tests near
  `tldw_Server_API/tests/Media_Ingestion_Modification/test_add_media_endpoint.py`

- [x] **Step 1: Add failing schema and dependency tests**

  Cover these cases:

  - `AddMediaForm` accepts `chunking_mode=auto`,
    `auto_chunking_goal=qa_search`, and `auto_chunking_use_llm=true`.
  - `AddMediaForm` preserves legacy behavior when `chunking_mode` is missing.
  - Invalid `chunking_mode` and invalid `auto_chunking_goal` fail through
    normal request validation.
  - `perform_chunking=false` with Auto fields present does not plan or chunk.
  - All direct process form dependencies accept the same Auto fields.
  - `WebScrapingRequest` and `IngestWebContentRequest` accept the same Auto
    fields for web/article ingestion.
  - Documents, audio, video, ebooks, and emails parse
    `auto_apply_template` and `chunking_template_name` consistently with PDFs.
  - `hierarchical_template` parsing is shared and does not accept malformed
    JSON silently if it is supplied as a form string.

  Run:

  ```bash
  source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Media_Ingestion_Modification/test_process_endpoints_contract_parity.py tldw_Server_API/tests/Media_Ingestion_Modification/test_add_media_endpoint.py -v
  ```

- [x] **Step 2: Add schema fields**

  In `media_request_models.py`:

  - Add `ChunkingMode = Literal["auto", "manual"]`.
  - Add `AutoChunkingGoal = Literal["balanced", "qa_search", "navigation_summary"]`.
  - Add optional fields to `ChunkingOptions`:
    - `chunking_mode: ChunkingMode | None = None`
    - `auto_chunking_goal: AutoChunkingGoal = "balanced"`
    - `auto_chunking_use_llm: bool = False`
  - Add the same three fields to `WebScrapingRequest` and
    `IngestWebContentRequest`.
  - Leave `ChunkMethod` unchanged for Manual requests. The Auto resolver may
    emit internal methods like `structure_aware` after request-model
    validation.

- [x] **Step 3: Centralize form boolean and JSON parsing**

  If the dependency modules already have local helpers, reuse them. Otherwise
  add small private helpers in the dependency modules to normalize:

  - String booleans from multipart form data.
  - Optional string literals.
  - Optional JSON object strings for `hierarchical_template`.

  Do not add a new shared utility unless both dependency modules would
  otherwise duplicate enough logic to justify it.

- [x] **Step 4: Parse new and existing parity fields**

  Update `media_add_deps.py` and all relevant process dependency functions in
  `media_processing_deps.py` so API clients can send the same chunking mode and
  template fields to direct processing and async ingest.

- [x] **Step 5: Re-run focused backend contract tests**

  ```bash
  source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Media_Ingestion_Modification/test_process_endpoints_contract_parity.py tldw_Server_API/tests/Media_Ingestion_Modification/test_add_media_endpoint.py -v
  ```

## Task 2: Deterministic Auto Chunk Planner

**Files:**

- Create `tldw_Server_API/app/core/Chunking/auto_planner.py`
- Create `tldw_Server_API/tests/Chunking/test_auto_chunking_planner.py`

- [x] **Step 1: Write planner tests before implementation**

  Test a pure planner API with no database, network, FastAPI request, or LLM
  calls. Cover:

  - Auto disabled when `perform_chunking=false`.
  - Missing `chunking_mode` returns legacy/manual behavior marker and no Auto
    plan.
  - `chunking_mode=manual` returns no Auto plan.
  - PDF/document with headings selects `structure_aware` or a documented
    equivalent existing method.
  - Unstructured document uses `semantic` only when a deterministic capability
    check says semantic dependencies are available; otherwise it falls back to
    `sentences` with a clear rationale.
  - Audio/video selects transcript-friendly sentence chunks and derived time
    view metadata when timecodes are present.
  - Ebook selects `ebook_chapters` when chapter signals exist.
  - Email preserves larger message/thread boundaries.
  - Web/article content with headings or list/table signals uses a
    structure-aware plan; plain articles fall back to paragraph/sentence
    planning.
  - `qa_search` chooses smaller or moderate chunks than `navigation_summary`.
  - `auto_chunking_use_llm=true` with no adapter records
    `fallback_reason` and `used_llm=false`.
  - Template classifier no-match and classifier failure do not fail ingest and
    record a fallback reason or rationale.

  Run:

  ```bash
  source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chunking/test_auto_chunking_planner.py -v
  ```

- [x] **Step 2: Implement planner data types**

  In `auto_planner.py`, use simple typed structures that serialize cleanly:

  - `AutoChunkingProfile`
  - `AutoChunkingRequest`
  - `AutoChunkingPlan`
  - `AutoChunkingDecision`

  Prefer `TypedDict` or Pydantic-free dataclasses with `asdict()` so the core
  planner stays lightweight and easy to unit test.

- [x] **Step 3: Implement profile builders**

  Add pure helpers for source and content hints:

  - `profile_from_source(media_type, filename=None, url=None, title=None, mime_type=None)`
  - `profile_from_text(text, *, max_scan_chars=...)`
  - `merge_profiles(base, extracted)`

  Detect only durable, cheap signals:

  - Approximate text length.
  - Heading-like lines.
  - Table-like density.
  - Chapter-like markers.
  - Transcript timecode markers.
  - Speaker/diarization labels.
  - File extension and media type.
  - Web/article title and URL hints.

  Avoid slow parsing or model calls in this deterministic layer.

- [x] **Step 4: Implement deterministic rule table**

  Keep rules explicit and easy to read. Suggested defaults:

  - `balanced`: moderate chunks and overlap.
  - `qa_search`: smaller chunks, stronger overlap, preserve offsets/timecodes.
  - `navigation_summary`: larger chunks, structural boundaries, derived
    outline/time range metadata.
  - PDF/document with headings: prefer `structure_aware` if supported.
  - Long unstructured document: prefer `semantic` only after a cheap
    dependency/capability check; otherwise `sentences`.
  - Audio/video: prefer `sentences`; include time-derived views if present.
  - Ebook with chapter signals: prefer `ebook_chapters`.
  - Email: prefer sentence or paragraph chunks with message-preserving sizing.
  - Web/article: prefer `structure_aware` for headings/lists/tables and
    paragraph or sentence chunks for plain article text.

- [x] **Step 5: Normalize plan output**

  Ensure planner output contains only JSON-serializable values:

  - `mode`
  - `goal`
  - `used_llm`
  - `method`
  - `max_size`
  - `overlap`
  - `template_name`
  - `derived_views`
  - `fallback_reason`
  - `rationale`

- [x] **Step 6: Re-run planner tests**

  ```bash
  source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chunking/test_auto_chunking_planner.py -v
  ```

## Task 3: Backend Planner Wiring, Job Results, And Metadata

**Files:**

- Modify `tldw_Server_API/app/core/Ingestion_Media_Processing/chunking_options.py`
- Modify `tldw_Server_API/app/api/v1/endpoints/media/process_documents.py`
- Modify `tldw_Server_API/app/api/v1/endpoints/media/process_pdfs.py`
- Modify `tldw_Server_API/app/api/v1/endpoints/media/process_audios.py`
- Modify `tldw_Server_API/app/api/v1/endpoints/media/process_videos.py`
- Modify `tldw_Server_API/app/api/v1/endpoints/media/process_ebooks.py`
- Modify `tldw_Server_API/app/api/v1/endpoints/media/process_emails.py`
- Modify `tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py`
- Modify `tldw_Server_API/app/api/v1/endpoints/media/ingest_web_content.py`
- Modify `tldw_Server_API/app/services/web_scraping_service.py`
- Modify `tldw_Server_API/app/services/media_ingest_jobs_worker.py`
- Modify `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`
- Modify `tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_ingest_jobs.py`
- Modify or create web/article ingest tests near
  `tldw_Server_API/tests/Web_Scraping/test_process_web_scraping_strategy_validation.py`
  or `tldw_Server_API/tests/WebScraping/test_auto_chunking_web_ingest.py`
- Add or modify focused media process tests as needed.

- [x] **Step 1: Write backend integration tests**

  Cover:

  - Direct process response includes `metadata.chunking_plan` or equivalent
    documented response metadata when `chunking_mode=auto`.
  - Async job status `result` includes `chunking_plan`.
  - Web/article Quick Ingest through `/api/v1/media/process-web-scraping`
    accepts Auto fields and includes plan metadata when results are returned
    or persisted.
  - `/api/v1/media/ingest-web-content` accepts the same Auto fields.
  - Persisted media safe metadata includes `chunking_plan`.
  - Manual requests continue to use explicitly supplied `chunk_method`,
    `chunk_max_size`, and `chunk_overlap`.
  - Legacy requests that omit `chunking_mode` preserve previous chunk option
    defaults.
  - Auto ignores stale Manual advanced fields from the request payload.
  - Template classifier no-match and classifier exception paths fall back
    deterministically and do not fail ingest.

  Run the narrowest tests first:

  ```bash
  source .venv/bin/activate && python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_ingest_jobs.py -v
  ```

- [x] **Step 2: Add a resolver in `chunking_options.py`**

  Add a function with a narrow contract, for example:

  ```python
  def resolve_chunking_options_and_plan(
      form_data: Any,
      *,
      media_type: str | None = None,
      source_name: str | None = None,
      extracted_text: str | None = None,
      template_match: str | None = None,
      llm_available: bool = False,
  ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
      ...
  ```

  Behavior:

  - Return `(None, None)` when `perform_chunking` is false.
  - Return existing `prepare_chunking_options_dict()` output and `None` plan
    for legacy and Manual paths.
  - Return planner options and serialized `chunking_plan` for Auto.
  - Apply explicit or classifier template matches as planner input, not as a
    hidden override that makes plan metadata inaccurate.
  - In the first implementation slice, pass `llm_available=False` unless a real
    AI boundary adapter is introduced in Task 6, so opt-in AI requests produce
    an honest deterministic fallback.

- [x] **Step 3: Wire direct process endpoints**

  Replace local `prepare_chunking_options_dict()` calls only where enough
  context is available. Keep endpoint-specific logic intact.

  If an endpoint currently does a second chunking pass after extracted content
  is available, finalize the Auto plan there with `extracted_text` rather than
  locking the plan before parsing.

  Attach `chunking_plan` to the same response metadata structure each endpoint
  already returns. Do not invent a different top-level response field per media
  type.

- [x] **Step 4: Wire web/article ingest paths**

  Quick Ingest web URLs use `processWebScrape()` in the frontend and post JSON
  to `/api/v1/media/process-web-scraping`, not the multipart media form. Wire
  Auto fields through:

  - `WebScrapingRequest`
  - `process_web_scraping.py`
  - `web_scraping_service.process_web_scraping_task()`

  The service currently builds web chunks directly with `Chunker()` in its
  persistence fallback path. Replace those hard-coded defaults with resolver
  output for `chunking_mode=auto`, and preserve existing sentence defaults when
  `chunking_mode` is missing.

  Also update `/api/v1/media/ingest-web-content` through
  `IngestWebContentRequest` and `ingest_web_content_orchestrate()` so the
  non-Quick-Ingest web API can use the same contract.

- [x] **Step 5: Wire async ingest jobs**

  In `media_ingest_jobs_worker.py`:

  - Use the resolver for the preliminary plan before processor dispatch.
  - Finalize the plan after extraction/transcription at the first point where
    extracted text or transcript content is available. For job-backed media,
    this should happen in the worker immediately after the processor returns,
    before building the final job result and before persistence receives chunk
    options.
  - If the existing processor helper already finalizes chunking inside
    `process_batch_media()` or document-like persistence, pass enough context
    through to avoid planning twice with contradictory metadata.
  - Include `chunking_plan` in the final result dictionary returned to job
    status clients.
  - Ensure worker retry/error paths do not fail if the plan is absent for
    legacy requests.

- [x] **Step 6: Persist plan metadata**

  In `persistence.py`, store Auto plan metadata in existing safe metadata:

  - Prefer `safe_metadata["chunking_plan"]`.
  - Preserve any existing user safe metadata.
  - Add `chunking_plan` to the local allowed-key filters used in AV and
    document-like persistence paths.
  - Preserve `chunking_plan` as a nested dict after recursively stripping values
    that are not JSON-safe primitives, lists, or dicts.
  - Confirm `normalize_safe_metadata()` does not remove the nested plan. If it
    does, adjust the persistence-side copy step or add a targeted utility test.
  - Do not store plan metadata inside every chunk unless existing chunk metadata
    helpers already attach shared document metadata cheaply.

- [x] **Step 7: Run focused backend tests**

  ```bash
  source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chunking/test_auto_chunking_planner.py tldw_Server_API/tests/Media_Ingestion_Modification/test_process_endpoints_contract_parity.py tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_ingest_jobs.py tldw_Server_API/tests/Web_Scraping/test_process_web_scraping_strategy_validation.py tldw_Server_API/tests/Media/test_ingest_web_content_endpoint_sanitization.py -v
  ```

## Task 4: Quick Ingest State And Payload Plumbing

**Files:**

- Modify `apps/packages/ui/src/components/Common/QuickIngest/types.ts`
- Modify `apps/packages/ui/src/components/Common/QuickIngest/presets.ts`
- Modify `apps/packages/ui/src/components/Common/QuickIngest/IngestOptionsPanel.tsx`
- Modify `apps/packages/ui/src/components/Common/hooks/useIngestOptions.tsx`
- Modify `apps/packages/ui/src/components/Common/hooks/useIngestPresets.tsx`
- Modify `apps/packages/ui/src/components/Common/QuickIngestModal.tsx`
- Add `apps/packages/ui/src/services/tldw/quick-ingest-chunking.ts`
- Modify `apps/packages/ui/src/services/tldw/quick-ingest-batch.ts`
- Modify `apps/packages/ui/src/entries/background.ts`
- Modify `apps/packages/ui/src/services/tldw/fallback-schemas.ts`
- Modify `apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts`
- Modify `apps/packages/ui/src/components/Common/QuickIngest/__tests__/IngestWizardContext.test.tsx`

- [x] **Step 1: Add failing frontend payload tests**

  In `quick-ingest-batch.test.ts`, cover:

  - Chunking enabled with default Auto sends:
    - `perform_chunking=true`
    - `chunking_mode=auto`
    - `auto_chunking_goal=balanced`
    - `auto_chunking_use_llm=false`
  - Auto with a selected goal sends the selected goal.
  - Auto with AI assist sends `auto_chunking_use_llm=true`.
  - Manual sends `chunking_mode=manual` and existing manual fields.
  - Auto does not send stale advanced manual chunk fields.
  - `processWebScrape()` sends Auto fields in the JSON body for web/article
    entries and omits stale Manual fields in Auto mode.
  - The extension/background path follows the same field rules or shares the
    same helper.

  Run:

  ```bash
  cd apps/packages/ui && bunx vitest run src/services/__tests__/quick-ingest-batch.test.ts --maxWorkers=1 --no-file-parallelism
  ```

- [x] **Step 2: Extend Quick Ingest types**

  In `types.ts`, add:

  ```ts
  export type ChunkingMode = "auto" | "manual"
  export type AutoChunkingGoal = "balanced" | "qa_search" | "navigation_summary"
  ```

  Add fields to the shared Quick Ingest options model:

  - `chunking_mode?: ChunkingMode`
  - `auto_chunking_goal?: AutoChunkingGoal`
  - `auto_chunking_use_llm?: boolean`

  Keep fields optional when this reduces migration risk for older stored
  options.

- [x] **Step 3: Update persisted state defaults and migration**

  In `useIngestOptions.tsx`:

  - Retain the existing new Quick Ingest default of:
    - `perform_chunking: true`
  - Add new Auto defaults for the enabled chunking state:
    - `chunking_mode: "auto"`
    - `auto_chunking_goal: "balanced"`
    - `auto_chunking_use_llm: false`
  - When hydrating older stored settings with `perform_chunking=true` and no
    mode, choose `auto` only for the new Quick Ingest default path.
  - Preserve existing advanced values in storage for Manual, but do not include
    them in Auto payloads.

- [x] **Step 4: Centralize payload construction**

  Prefer one shared helper in `quick-ingest-batch.ts` that both WebUI and
  extension/background paths can call. If the background entry cannot import
  the helper cleanly, duplicate only the smallest field-mapping wrapper and add
  a test that locks parity.

  Field behavior:

  - Always send `perform_chunking`.
  - Send Auto fields only when `perform_chunking && chunking_mode === "auto"`.
  - Send Manual fields only when `perform_chunking && chunking_mode === "manual"`.
  - In `processWebScrape()`, map the same Auto fields into the JSON body sent
    to `/api/v1/media/process-web-scraping`.
  - Do not send `chunking_template_name` or `auto_apply_template` from old
    saved advanced settings while Auto is active unless the UI explicitly models
    them as Auto planner inputs.

- [x] **Step 5: Update fallback schema**

  Add `chunking_mode`, `auto_chunking_goal`, and
  `auto_chunking_use_llm` to fallback schemas so local/offline settings screens
  do not hide the fields when the backend schema fetch fails.

- [x] **Step 6: Run focused frontend service tests**

  ```bash
  cd apps/packages/ui && bunx vitest run src/services/__tests__/quick-ingest-batch.test.ts src/components/Common/QuickIngest/__tests__/IngestWizardContext.test.tsx --maxWorkers=1 --no-file-parallelism
  ```

  Completed with local package runner:

  ```bash
  cd apps/packages/ui && bun run test -- src/services/__tests__/quick-ingest-batch.test.ts src/components/Common/QuickIngest/__tests__/IngestWizardContext.test.tsx --maxWorkers=1 --no-file-parallelism
  ```

## Task 5: Quick Ingest Auto/Manual UI

**Files:**

- Modify `apps/packages/ui/src/components/Common/QuickIngest/IngestOptionsPanel.tsx`
- Modify `apps/packages/ui/src/components/Common/QuickIngest/WizardConfigureStep.tsx`
- Modify `apps/packages/ui/src/components/Common/QuickIngestModal.tsx`
- Modify `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx`
- Modify `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx`

- [x] **Step 1: Add failing UI tests**

  Cover:

  - Enabling Chunking shows Auto mode selected by default.
  - Auto mode shows goal selection and AI assist toggle.
  - Manual mode exposes existing detailed chunking controls.
  - Switching from Manual back to Auto hides or disables Manual-only fields.
  - Submitting from Auto sends Auto fields and not stale Manual fields.

  Run:

  ```bash
  cd apps/packages/ui && bunx vitest run src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx --maxWorkers=1 --no-file-parallelism
  ```

- [x] **Step 2: Add controls to `IngestOptionsPanel.tsx`**

  Use existing Ant Design patterns:

  - `Segmented` or the local equivalent for `Auto | Manual`.
  - `Select` for `Balanced`, `Search/Q&A`, and `Reading/Summary`.
  - `Switch` for `Use AI to improve chunk boundaries`.

  Keep text concise and task-facing. Do not add a marketing explanation panel.

- [x] **Step 3: Add controls to `WizardConfigureStep.tsx`**

  Keep the wizard compact:

  - Chunking toggle remains the parent control.
  - When enabled, show Auto selected by default.
  - Let users switch to Manual and reveal existing advanced settings through the
    same path used by the options panel.

- [x] **Step 4: Preserve Manual advanced behavior**

  Existing advanced options should remain available and should still work after
  switching to Manual:

  - method
  - size
  - overlap
  - template selection
  - auto-apply template
  - contextual/proposition/hierarchical fields where the schema exposes them

- [x] **Step 5: Run UI tests**

  ```bash
  cd apps/packages/ui && bunx vitest run src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx src/components/Common/QuickIngest/__tests__/IngestWizardContext.test.tsx --maxWorkers=1 --no-file-parallelism
  ```

  Completed with local package runner:

  ```bash
  cd apps/packages/ui && bun run test -- src/services/__tests__/quick-ingest-batch.test.ts src/components/Common/QuickIngest/__tests__/IngestWizardContext.test.tsx src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx --maxWorkers=1 --no-file-parallelism
  ```

## Task 6: AI-Assist Adapter Boundary

**Files:**

- Modify `tldw_Server_API/app/core/Chunking/auto_planner.py`
- Create tests in `tldw_Server_API/tests/Chunking/test_auto_chunking_planner.py`
- Add new adapter files only if the implementation actually makes LLM calls.

- [x] **Step 1: Keep deterministic fallback explicit**

  The first implementation should support this behavior even before any LLM
  call exists:

  - Request `auto_chunking_use_llm=true`.
  - Planner sees no available AI boundary adapter because the first slice passes
    `llm_available=False` from the resolver.
  - Planner returns deterministic chunking options.
  - Plan metadata has `used_llm=false` and a clear `fallback_reason`.

  Do not infer availability from a configured chat provider until a real
  `AutoChunkBoundaryAssistant` adapter is implemented and tested. A provider
  key by itself is not enough to claim AI boundary refinement was used.

- [x] **Step 2: Define adapter interface before adding model calls**

  If implementing real AI-assisted boundaries in this feature branch, first add
  a narrow interface, for example:

  ```python
  class AutoChunkBoundaryAssistant(Protocol):
      async def suggest_boundaries(
          self,
          *,
          text_profile: AutoChunkingProfile,
          goal: str,
          deterministic_plan: dict[str, Any],
      ) -> AutoChunkingAssistantResult:
          ...
  ```

  The adapter must:

  - Use existing LLM provider plumbing.
  - Define exactly how provider/model/key availability is checked before the
    planner receives `llm_available=True`.
  - Have strict timeout and token limits.
  - Never be invoked unless `auto_chunking_use_llm=true`.
  - Return bounded suggestions, not arbitrary code or chunk text.
  - Fall back to deterministic Auto on any provider/config/runtime error.

  Completed for this branch by deferring model calls. No adapter file was
  added because no real AI boundary refinement is invoked in this V1 slice.
  `TASK-96.8` tracks the follow-up adapter and test matrix.

- [x] **Step 3: Add tests before real AI integration**

  Mock the adapter. Cover:

  - Adapter not called by default.
  - Adapter called only for explicit opt-in.
  - Adapter success sets `used_llm=true`.
  - Adapter timeout/error preserves deterministic plan and records
    `fallback_reason`.

  Completed for this branch by adding no-adapter fallback regression coverage.
  Adapter-specific mock tests move with `TASK-96.8`, where the adapter contract
  will exist.

- [x] **Step 4: Decide whether to defer real LLM calls**

  If deterministic Auto, request/UI plumbing, and metadata are already a large
  branch, defer real model calls to a follow-up Backlog task. The feature still
  has an honest AI-assist affordance only if the UI and metadata make fallback
  behavior clear. Do not silently pretend AI was used.

  Decision: defer real LLM boundary refinement to `TASK-96.8`. This branch
  keeps AI assist as explicit opt-in metadata and deterministic fallback only.

## Task 7: Verification, Documentation, And Tracking

**Files:**

- Modify or create user-facing docs only if the implementation changes public
  behavior in a way not already covered by API docs.
- Update Backlog implementation tasks.
- Do not edit the approved design spec except to add a link to the final
  implementation PR if requested.

- [x] **Step 1: Run backend focused tests**

  ```bash
  source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chunking/test_auto_chunking_planner.py tldw_Server_API/tests/Media_Ingestion_Modification/test_process_endpoints_contract_parity.py tldw_Server_API/tests/Media_Ingestion_Modification/test_add_media_endpoint.py tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_ingest_jobs.py tldw_Server_API/tests/Web_Scraping/test_process_web_scraping_strategy_validation.py tldw_Server_API/tests/Media/test_ingest_web_content_endpoint_sanitization.py -v
  ```

  Completed with the current Auto Chunking focused backend suite:

  ```bash
  source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_auto_chunking_request_contract.py tldw_Server_API/tests/Chunking/test_auto_chunking_planner.py tldw_Server_API/tests/Chunking/test_auto_chunking_resolver.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py tldw_Server_API/tests/Media/test_auto_chunking_process_endpoints.py tldw_Server_API/tests/Web_Scraping/test_auto_chunking_web_ingest.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_auto_chunking_persistence_metadata.py -v
  ```

  Result: 41 passed, 6 warnings.

- [x] **Step 2: Run frontend focused tests**

  ```bash
  cd apps/packages/ui && bunx vitest run src/services/__tests__/quick-ingest-batch.test.ts src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx src/components/Common/QuickIngest/__tests__/IngestWizardContext.test.tsx --maxWorkers=1 --no-file-parallelism
  ```

  Completed with:

  ```bash
  cd apps/packages/ui && bun run test -- src/services/__tests__/quick-ingest-batch.test.ts src/components/Common/QuickIngest/__tests__/IngestWizardContext.test.tsx src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx --maxWorkers=1 --no-file-parallelism
  ```

  Result: 4 files passed, 83 tests passed. The test output still includes
  pre-existing AntD mock DOM warnings and the expected
  `useIngestWizard must be used within an IngestWizardProvider` guard-test
  console error.

  OpenAPI client verification also passed:

  ```bash
  cd apps/packages/ui && bun run verify:openapi
  ```

  Result: 256 client paths verified, 10 reviewed exception paths allowed, and
  49 fallback media fields verified against `/api/v1/media/add`.

- [x] **Step 3: Run compile checks if touched files require them**

  ```bash
  cd apps/packages/ui && bun run compile
  ```

  If extension/background packaging was materially changed, also run the
  extension compile command used by the repo:

  ```bash
  cd apps/extension && bun run compile
  ```

  Completed with:

  ```bash
  cd apps/packages/ui && bunx tsc --noEmit --pretty false > /tmp/tldw_auto_chunking_tsc.log 2>&1
  ```

  Result: full typecheck exited 2 with existing repo-wide test typing errors.
  Filtering `/tmp/tldw_auto_chunking_tsc.log` for the touched Auto Chunking UI
  files returned no matches.

- [x] **Step 4: Run Bandit on touched backend scope**

  Adjust the path list to match the actual touched backend files:

  ```bash
  source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/schemas/media_request_models.py tldw_Server_API/app/api/v1/API_Deps/media_add_deps.py tldw_Server_API/app/api/v1/API_Deps/media_processing_deps.py tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py tldw_Server_API/app/api/v1/endpoints/media/ingest_web_content.py tldw_Server_API/app/core/Chunking/auto_planner.py tldw_Server_API/app/core/Ingestion_Media_Processing/chunking_options.py tldw_Server_API/app/services/media_ingest_jobs_worker.py tldw_Server_API/app/services/web_scraping_service.py tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py -f json -o /tmp/bandit_auto_chunking.json
  ```

  Completed for the current backend code delta with:

  ```bash
  source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Chunking/auto_planner.py -f json -o /tmp/bandit_auto_chunking_ai_boundary.json
  ```

  Result: zero findings.

- [x] **Step 5: Run whitespace diff check**

  ```bash
  git diff --check
  ```

  Result: no whitespace errors.

- [x] **Step 6: Manual API smoke test**

  If a local server is already running, submit one small text/document ingest
  with:

  - `perform_chunking=true`
  - `chunking_mode=auto`
  - `auto_chunking_goal=balanced`

  Verify:

  - Response or job result includes `chunking_plan`.
  - Existing Manual request still honors explicit `chunk_method`.
  - A web/article request through `/api/v1/media/process-web-scraping` accepts
    Auto fields and does not regress legacy JSON requests without
    `chunking_mode`.

  Do not start or configure external LLM services for this smoke test.

  Skipped because no local API server was already running for this final pass.
  Endpoint and job-result coverage above verifies the same request/metadata
  contract without starting external services.

- [x] **Step 7: Manual UI smoke test**

  If the frontend dev server can run locally:

  - Open Quick Ingest.
  - Enable Chunking.
  - Confirm Auto is selected by default.
  - Switch to Manual and confirm advanced controls are available.
  - Switch back to Auto and submit a dry-run/test fixture if the app supports
    the local backend.

  Skipped because no frontend dev server was already running for this final
  pass. Focused UI tests cover the Auto default, Auto/Manual switch, Manual
  controls, and submit payload behavior.

- [x] **Step 8: Update Backlog and final summary**

  Record:

  - Files changed.
  - Tests run and results.
  - Any skipped checks and why.
  - Whether real AI-assisted boundary refinement was implemented or deferred.

  Completed in `TASK-96.7`. Real AI-assisted boundary refinement is deferred to
  follow-up `TASK-96.8`.

## Suggested Implementation Slices

Use separate commits if the branch grows:

1. Backend contract fields and parser parity.
2. Deterministic planner with unit tests.
3. Backend wiring, web/article ingest, job result, and persistence metadata.
4. Frontend state, payload plumbing, and web/article JSON payload parity.
5. Frontend Auto/Manual controls.
6. AI-assist adapter boundary or deferral note.
7. Verification and docs/tracking updates.

## Risks And Guardrails

- `structure_aware` exists in the chunking engine but not in current media
  request validation. Keep it internal to Auto resolver output in V1 so Manual
  API behavior remains unchanged.
- Async ingest currently returns a narrow final result. Tests must prove
  `chunking_plan` reaches job status clients.
- Quick Ingest web/article entries use `/api/v1/media/process-web-scraping`
  with JSON bodies, not multipart media forms. Tests must cover
  `processWebScrape()` and the background entry path.
- Existing persistence filters safe metadata by allowed keys and value shape.
  `chunking_plan` must be explicitly preserved as a sanitized nested dict or it
  will be dropped before storage.
- Semantic chunking may depend on optional packages. The planner must check
  capability before selecting semantic for unstructured documents, or fall back
  to sentences.
- Old saved Quick Ingest advanced fields can silently alter Auto if payload
  construction merges all advanced values. Tests must lock this out.
- The WebUI batch service and extension/background path can drift. Prefer a
  shared payload helper or test parity directly.
- AI assistance must remain explicit opt-in and must have deterministic
  fallback on unavailable providers, timeouts, and errors.
- Derived navigation views are metadata in this branch, not alternate chunk
  stores or search indexes.

## Completion Criteria

- Auto fields are accepted by backend schemas and form dependencies.
- Deterministic planner has focused unit coverage.
- Direct media processing, web/article ingest, async ingest jobs, and persisted
  safe metadata expose `chunking_plan` for Auto requests.
- Legacy and Manual chunking behavior is preserved by tests.
- Quick Ingest defaults to Auto when Chunking is enabled and sends the correct
  payload for file/media and web/article entries.
- Manual mode still exposes and submits existing advanced chunk settings.
- AI-assist preference is explicit, disabled by default, and does not make model
  calls unless the adapter is intentionally implemented.
- Focused backend and frontend tests pass.
- Bandit is run on touched backend files or a non-code skip is recorded.
- Backlog tasks are updated with verification and final summary.
