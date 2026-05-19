# Auto Chunking for Quick Ingest Design

Task: TASK-96
Status: Approved design, documentation-only spec
Date: 2026-05-06

## Goal

Add a novice-facing Auto Chunking mode for Quick Ingest. The feature should let less technical users turn on chunking and get useful defaults for search, Q&A, navigation, and summaries without choosing chunk methods, sizes, overlaps, templates, or LLM settings.

The product contract is:

- Turning `Chunking` off disables chunking.
- Turning `Chunking` on in the new Quick Ingest UI defaults to `Auto`.
- `Auto` uses deterministic media-aware planning unless the user explicitly enables AI assistance.
- `Manual` exposes the existing detailed chunking controls.
- Auto supports `Balanced`, `Search/Q&A`, and `Reading/Summary` goals.

## Existing System Inventory

The repo already has substantial chunking functionality:

- `tldw_Server_API/app/core/Chunking/` supports words, sentences, paragraphs, tokens, semantic, JSON, XML, ebook chapters, propositions, rolling summarize, structure-aware, code, hierarchical chunking, templates, classifier auto-apply, and template learning.
- `tldw_Server_API/app/api/v1/endpoints/chunking.py` exposes standalone text/file chunking and advertises runtime capabilities.
- `tldw_Server_API/app/api/v1/endpoints/chunking_templates.py` supports template CRUD, apply, validate, match, and learn.
- `tldw_Server_API/app/api/v1/schemas/media_request_models.py` already defines `auto_apply_template` and `chunking_template_name`.
- `apps/packages/ui/src/components/Option/ChunkingPlayground/` is a power-user test and template-authoring surface.
- `apps/packages/ui/src/components/Common/QuickIngest/` already has simple chunking toggles plus a hidden-ish template selector and auto-detect switch.
- `apps/packages/ui/src/services/tldw/quick-ingest-batch.ts` sends `chunking_template_name` and `auto_apply_template` when selected.
- `apps/packages/ui/src/entries/background.ts` has a parallel Quick Ingest submission path that also needs any new Auto fields.

The main gap is not missing chunking algorithms. The gap is a first-class, explainable Auto mode that turns existing primitives into a safe default for nontechnical users.

## Existing Integration Gaps

The implementation plan should account for these current inconsistencies:

- General media-add/job parsing currently accepts basic chunking fields but appears not to parse `auto_apply_template`, `chunking_template_name`, `hierarchical_chunking`, or `hierarchical_template` through `get_add_media_form`.
- PDF processing parses `auto_apply_template` and `chunking_template_name`, but documents, audio, video, ebooks, and emails appear to expose only basic chunking fields through their process-form dependencies.
- `apply_chunking_template_if_any()` applies explicit or auto-selected templates only when the template contains hierarchical configuration. Template selection is therefore a partial planning input, not a complete Auto Chunking solution.
- Async ingest job finalization currently returns a narrow result payload. Auto Chunking plan metadata must be explicitly forwarded there or persisted somewhere shared; adding it only to intermediate processing results will not make it visible to job status clients.
- The existing Agentic Chunking design is query-time RAG evidence assembly. It is useful precedent but is not ingest-time Auto Chunking.

## User-Facing Model

Quick Ingest should show:

```text
Chunking: off | on

When on:
  Mode: Auto | Manual

Auto:
  Goal: Balanced | Search/Q&A | Reading/Summary
  Use AI to improve chunk boundaries: off by default

Manual:
  Existing advanced settings:
    method, size, overlap, adaptive, multi-level, template,
    contextual chunking, proposition options, and related controls
```

Template auto-detect should move behind Auto as a planner input. Explicit template selection should remain available in Manual or advanced settings.

Auto and Manual should have explicit precedence. In Auto mode, manual chunking controls from old saved advanced settings should not silently affect planning. Switching to Manual re-enables the detailed fields. The only LLM-related Auto control is `auto_chunking_use_llm`.

## API Contract

Add explicit fields while preserving old clients:

```text
perform_chunking=true
chunking_mode=auto|manual
auto_chunking_goal=balanced|qa_search|navigation_summary
auto_chunking_use_llm=false|true
```

Compatibility rules:

- If `chunking_mode` is missing, preserve today's behavior.
- The new Quick Ingest UI should explicitly send `chunking_mode=auto` when a user enables Chunking and has not switched to Manual.
- `auto_chunking_use_llm` defaults to false.
- If `auto_chunking_use_llm=true` but provider/model/key support is unavailable, the ingest should fall back to deterministic Auto and record the fallback reason.
- Manual mode keeps existing request fields and should not reinterpret advanced user settings.

Auto mode should not require the client to send an existing `chunk_method`. If the planner chooses an internal method that is not currently accepted by media request validation, such as `structure_aware`, the implementation must either apply that method after request validation or extend the media chunk-method allowlist before exposing it as a request value.

## Planner Component

Introduce an Auto Chunk Planner that returns normalized existing chunker options plus user-visible metadata.

Proposed input:

- media type
- filename, URL, title, MIME type, and extension
- content length and extracted text profile when available
- headings, chapters, table-like blocks, transcript timecodes, diarization, caption availability, OCR signal, and language when available
- existing template classifier matches
- user goal
- explicit AI-assist preference
- provider/model availability for AI assist

Proposed output:

```json
{
  "mode": "auto",
  "goal": "balanced",
  "used_llm": false,
  "method": "structure_aware",
  "max_size": 900,
  "overlap": 120,
  "template_name": "academic_pdf",
  "derived_views": ["outline", "section_titles", "summary_anchors"],
  "fallback_reason": null,
  "rationale": "Detected headings and long sections; selected structure-aware chunking for retrieval plus readable section anchors."
}
```

The planner should return chunking options in the same shape expected by existing chunking helpers so processors can reuse current chunking strategies.

Derived navigation views in V1 are metadata outputs, not separate persistent chunk sets or secondary indexes. Examples include outline entries, section labels, time ranges, and summary anchors attached to the chunking plan or response metadata.

## Planner Behavior by Goal

`qa_search`:

- Optimize for retrieval quality, citations, and chunk boundaries that do not bury the answer.
- Prefer smaller or moderate chunks with appropriate overlap.
- Preserve source offsets, section metadata, and timecodes.

`navigation_summary`:

- Optimize for human browsing, chapter-like sections, readable titles, summaries, and time spans.
- Prefer structural boundaries and larger coherent segments.
- Produce derived navigation metadata where possible.

`balanced`:

- Default novice goal.
- Use a primary retrieval-friendly chunk set plus derived navigation views when possible.
- Avoid dual persistent chunk sets in V1 unless the storage/indexing model already supports this cleanly.

## Planner Behavior by Media Type

PDF and document:

- Prefer `structure_aware` or hierarchical sentence chunking when headings/sections are detected.
- Use semantic chunking as a fallback for unstructured long prose.
- Use OCR and table signals to preserve boundaries and adjust sizing.
- With AI assist enabled, use the LLM to refine section labels or boundaries, not to rewrite chunks.

Audio and video:

- Use transcript length, sentence boundaries, timecodes, diarization, and caption availability.
- Default to sentence chunks with timecode metadata.
- For `navigation_summary`, preserve larger segments around topic shifts.
- With AI assist enabled, optionally infer topic breaks from transcript windows.

Ebook:

- Prefer `ebook_chapters`.
- If chapters are weak or missing, fall back to structure-aware, paragraph, or sentence chunks.
- For `navigation_summary`, produce chapter-style navigation metadata.

Email:

- Preserve thread and message boundaries.
- Prefer larger sentence or paragraph chunks.
- If attachments are ingested, plan per attachment type instead of forcing email defaults onto every child item.

Web or article content:

- Prefer structure-aware headings, lists, and tables.
- Fall back to paragraphs or sentences depending on length.
- Template matching can help, but should be only one planner signal.

## AI Assistance

AI assistance is explicit opt-in through `auto_chunking_use_llm=true`.

AI assistance may:

- suggest topic breaks for long transcripts
- refine section labels
- identify weak headings or chapter-like boundaries
- choose among deterministic candidate plans

AI assistance must not:

- rewrite source chunks as the canonical stored text
- silently turn on because a provider is configured
- fail the whole ingest when unavailable unless a future advanced policy explicitly requests strict failure

If AI assistance is unavailable, the result metadata should state that deterministic Auto was used and why.

## Persistence and Result Metadata

Persist or return the final plan where users and jobs can inspect it. At minimum, job/result metadata should include:

- common key: `chunking_plan`
- `chunking_mode`
- `auto_chunking_goal`
- chosen method, size, overlap, and template
- `used_llm`
- fallback reason, if any
- estimated or final chunk count when available
- short rationale

This is important for trust. The user should be able to understand why Auto made a choice without reading logs.

Use the same `chunking_plan` object across direct process responses, async job result payloads, and durable media metadata where available. For persisted media items, prefer storing it under `safe_metadata.chunking_plan` unless implementation review finds a more appropriate existing metadata contract.

## Data Flow

1. Quick Ingest sends `perform_chunking=true`, `chunking_mode=auto`, selected goal, and AI-assist preference.
2. Backend form dependencies preserve these fields for media-add, async ingest jobs, and process endpoints.
3. Manual mode follows the current chunking option path.
4. Auto mode builds a preliminary plan from request metadata.
5. After extraction/transcription, Auto finalizes the plan using text/profile signals.
6. Existing chunking helpers receive normalized options.
7. Job/result metadata records the final plan and fallback details.

## Error Handling

- Invalid `chunking_mode` or `auto_chunking_goal` should produce normal validation errors.
- If chunking is disabled, Auto fields should be ignored.
- If AI assist is requested but unavailable, fall back to deterministic Auto and record the reason.
- If template matching fails, continue without template and record the reason.
- If Auto planning fails unexpectedly, use the existing safe default chunking behavior and record a planner fallback. Do not fail ingestion for planner-only errors unless the underlying chunker fails.

## Rollout

Suggested implementation sequence:

1. Add schemas/form parsing for Auto fields and fix current template-field parity across media-add/jobs/process endpoints.
2. Add deterministic planner with unit tests and no LLM dependency.
3. Wire planner into media ingestion and process flows while preserving manual behavior.
4. Persist/return plan metadata in job/result payloads.
5. Update Quick Ingest UI so when Chunking is enabled, Mode defaults to Auto, the goal selector and AI-assist toggle are visible, and detailed controls move behind Manual.
6. Update both Quick Ingest submission paths: the WebUI batch service and the extension/background entry.
7. Migrate or hydrate persisted Quick Ingest options so existing saved `perform_chunking=true` settings default to Auto without leaking old manual advanced values.
8. Add explicit AI-assist support after deterministic planning is stable.

## Testing

Backend tests:

- Form dependencies preserve `chunking_mode`, `auto_chunking_goal`, `auto_chunking_use_llm`, `auto_apply_template`, and `chunking_template_name` for media-add/jobs and process endpoints.
- Missing `chunking_mode` preserves old behavior.
- Auto planner returns expected deterministic plans for representative media profiles.
- AI-assist unavailable falls back without failing ingestion.
- Template classifier failure or no match falls back without failing ingestion.
- Manual mode preserves explicit method, size, overlap, template, and contextual settings.

Frontend tests:

- Enabling Chunking in Quick Ingest selects Auto by default.
- Auto mode shows goal and AI-assist controls only.
- Manual mode reveals existing advanced settings.
- Quick Ingest request payload includes the new Auto fields.
- Manual payloads continue to send existing advanced fields.

Integration tests:

- Async media ingest jobs retain Auto fields in job payload options.
- Stored job/result metadata includes the final plan/rationale.
- Async job status exposes `chunking_plan`, not only intermediate processor metadata.
- PDF and at least one non-PDF media type exercise the same Auto contract.

## Non-Goals for V1

- Multiple persistent chunk sets per item.
- LLM-only chunking.
- A new chunking engine replacing the current strategies.
- Breaking old clients that only send `perform_chunking=true`.
- Making the Chunking Playground the novice workflow.

## Open Implementation Risks

- Some processing paths choose chunk options before extracted text is available. The implementation should introduce a clear finalization point after extraction/transcription so Auto can use text/profile signals.
- Result metadata shape may differ across immediate process endpoints, media-add, and async jobs. The implementation should choose a common metadata key and document it.
- Template JSON shapes should be checked before relying on form-created templates for Auto planning, because media ingestion currently only consumes hierarchical config from templates.
