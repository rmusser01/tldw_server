# Slides And Infographics Work Products PRD

Status: Proposed
Owner: Product / WebUI / API
Created: 2026-07-10
Tracking: TASK-12102

## Summary

tldw_server already has a Slides API, Presentation Studio routes, persisted
`studio_data`, image metadata, visual styles, Reveal.js/Markdown/JSON/PDF
exports, and render jobs. This PRD extends that foundation into a first-class
study and research work-product surface for editable slide decks and
presentation-quality infographics generated from media, notes, chats, and RAG
queries.

The core product bet is conservative: do not build a PowerPoint clone. Instead,
make the existing slide model better at producing, editing, and exporting
structured knowledge artifacts that users can present, revise, cite, and share.

## Problem

Users can ingest source material and ask questions, but turning that knowledge
into presentable work products still requires manual transfer into external
tools. The current Slides module is a strong base, but it needs clearer
work-product semantics, better templates, richer infographic layouts, source
traceability, and WebUI/extension workflows that make deck creation feel like a
natural continuation of research and study.

## Goals

- Let users generate editable slide decks from media, notes, chats, and RAG
  results.
- Add infographic-oriented slide templates for timelines, process flows,
  comparisons, hierarchies, concept summaries, quote cards, and statistic cards.
- Preserve citations and source provenance from generation through editing and
  export.
- Keep Presentation Studio as the editable source of truth, with rendered media
  and exported files as derivatives.
- Support both WebUI and browser extension entry points where feasible.
- Provide task boundaries that can be implemented in reviewable phases.

## Non-Goals

- Full PowerPoint or Keynote feature parity.
- Collaborative real-time multi-user slide editing.
- Arbitrary JavaScript execution, preview, or rendering inside tldw.
- Unbounded image generation or media rendering.
- Replacing existing notes, media, RAG, or quiz workflows.

## Existing Foundation

- API: `Docs/API/Slides.md`
- Backend endpoint: `tldw_Server_API/app/api/v1/endpoints/slides.py`
- Schemas: `tldw_Server_API/app/api/v1/schemas/slides_schemas.py`
- Core: `tldw_Server_API/app/core/Slides/`
- WebUI: `apps/packages/ui/src/components/Option/PresentationStudio/`
- Routes: `/presentation-studio`, `/presentation-studio/new`,
  `/presentation-studio/:projectId`
- Existing capabilities: create, list, update, patch, search, generate from
  prompt/chat/media/notes/RAG, export Reveal.js/Markdown/JSON/PDF, render MP4
  or WebM jobs, store visual style selection, store `studio_data`, attach slide
  images.

## User Stories

- As a student, I can generate a lecture slide deck from a video transcript and
  revise it before presenting.
- As a researcher, I can create a concise cited presentation from selected notes
  and papers.
- As a journalist, I can turn a collection of findings into an infographic deck
  with source-backed claims.
- As a browser extension user, I can capture a page or selection and start a
  deck draft without leaving the browser.
- As a self-hosted user, I can export a deck without sending my sources to a
  third-party presentation service.

## Product Requirements

### PR-1: Work-Product Modes

Presentation creation must expose clear modes:

- Standard slide deck
- Teaching deck
- Research briefing
- Infographic deck
- Executive summary
- Source comparison

Each mode should map to a template policy, default slide count, tone, source
coverage expectation, and allowed visual layouts.

### PR-2: Infographic Layout Library

Add an initial layout library with structured metadata for:

- Timeline
- Process flow
- Compare / contrast
- Pros / cons
- Hierarchy
- Concept map slide
- Statistic cards
- Claim with evidence
- Quote with context
- Key takeaways

The MVP may represent these as slide `layout` plus `metadata.visual_blocks`
rather than adding a separate infographic database.

### PR-3: Source Provenance

Generated decks must persist source context sufficient to render citations in
the editor and exports:

- Source type and ID
- Human-readable source label
- Snippet or quote, bounded in length
- Media timestamp or document chunk when available
- Retrieval score or generation confidence when available

Editors must be able to inspect source citations per slide and per visual
block. Exports must include citation output in an unobtrusive, readable format.

### PR-4: Editable Studio Workflow

Presentation Studio should support:

- Selecting a generation mode and source bundle.
- Regenerating one slide without replacing the whole deck.
- Adding, duplicating, deleting, and reordering slides.
- Editing text, speaker notes, image metadata, and structured visual blocks.
- Previewing export/render readiness.
- Showing stale source or missing asset warnings.

### PR-5: Export Contract

Exports should preserve the best possible representation by format:

- JSON: full structured presentation payload.
- Markdown: Marp-compatible, including citation blocks.
- Reveal.js ZIP: presentable HTML with local assets.
- PDF: static shareable deck.
- Future optional: PPTX export if dependency and layout fidelity are acceptable.

PPTX is explicitly a later phase because it creates a second layout/rendering
contract.

### PR-6: Extension Handoff

The extension should not host a full editor at first. It should support:

- Start deck from page, selection, captured image, or current source.
- Send the seed to the server or WebUI route.
- Open the WebUI Presentation Studio for full editing.
- Show job/result status for generation where useful.

## UX Requirements

- The first screen for this feature must be the usable Presentation Studio
  workspace or creation flow, not a marketing page.
- Controls must distinguish generation mode, source selection, style selection,
  and export/render actions.
- Dense editing surfaces should use compact panels, predictable tabs, and
  stable dimensions.
- Empty states must point users to source ingestion or note selection.
- Long deck generation must show progress and cancellation where backend jobs
  are used.

## API And Data Model Direction

Prefer extending existing Slides contracts:

- Add `work_product_mode` or equivalent field in `studio_data` first.
- Add `source_bundle` and per-slide citation metadata using existing slide
  `metadata`.
- Add `visual_blocks` under slide metadata for infographic elements.
- Add template identifiers for infographic presets.
- Avoid a new database until existing Slides persistence is insufficient.

Potential later schema fields:

- `presentation_kind`
- `generation_profile`
- `source_bundle_json`
- `export_profile`
- `slide.metadata.citations`
- `slide.metadata.visual_blocks`

## Backend Requirements

- Add generation profiles for each work-product mode.
- Validate generated slide JSON against an allowlisted schema.
- Enforce source and output size limits.
- Reuse existing Jobs only when generation/rendering is long-running.
- Record metrics by source type, mode, and export format.
- Preserve existing optimistic-locking behavior with `If-Match`.

## WebUI And Extension Requirements

- WebUI owns full creation, editing, preview, and export.
- Extension owns capture/start handoff, not full editing in MVP.
- Shared UI services should live in the shared package where practical.
- Capability flags should hide unavailable render/export features.
- Existing Presentation Studio route tests should be expanded rather than
  duplicated with a parallel route.

## Security And Privacy

- Never log source text, generated deck content, API keys, or image payloads.
- Sanitize custom CSS and HTML-bearing export content.
- Bound citation quotes and generated metadata sizes.
- Keep data local/self-hosted; external LLM calls must follow existing provider
  configuration and user intent.
- Preserve per-user DB boundaries.

### Narrow Standalone HTML Exception

Standalone JavaScript may be generated, stored, edited, versioned, and
downloaded only as bounded opaque text. This exception does not permit tldw to
execute, preview, render, navigate to, or load resources from that text. The
storage validator enforces a fixed document contract; it does not sanitize the
document into safety. The browser extension remains source-free and hands
standalone projects to the WebUI using source-free metadata.

## Success Metrics

- Users can generate and save a cited deck from at least one media item.
- Users can generate and save an infographic-style deck from notes or RAG.
- Users can edit at least title/content/notes/citations after generation.
- PDF/Markdown/JSON exports preserve readable citations.
- Extension can start a deck from a captured page or selection.
- Existing slide tests and route guards remain passing.

## Rollout Plan

### Phase 1: Product Contract And Infographic Templates

Define generation modes, infographic layout metadata, citation metadata, and
validation rules. Add tests for serialization, validation, and export-safe
rendering.

### Phase 2: Generation And Studio Editing

Wire generation profiles into existing slide generation paths. Add editor
controls for source bundle, mode, visual blocks, and citations.

### Phase 3: Export And Render Polish

Improve Markdown, Reveal.js, PDF, and JSON exports for infographic layouts and
citations. Add readiness diagnostics for missing assets and unsupported blocks.

### Phase 4: Extension Handoff And Documentation

Add extension start flows and update user/developer documentation. Validate
extension route parity and degraded states.

### Phase 5: Optional PPTX Investigation

Prototype PPTX export separately. Ship only if layout fidelity and dependency
cost are acceptable.

## Backlog Task Set

- Parent: Implement Slides and Infographics work products.
- Phase 1: Define slides work-product modes and infographic schema.
- Phase 2: Add generation profiles and Presentation Studio editing controls.
- Phase 3: Improve cited infographic exports and render readiness.
- Phase 4: Add extension deck-start handoff and documentation.
- Phase 5: Investigate PPTX export feasibility.

## Open Questions

- Should `work_product_mode` become a top-level database field or remain in
  `studio_data` for the first release?
- Which exports are release-blocking for the first phase: PDF, Markdown, JSON,
  Reveal.js, or all current formats?
- Should image generation be integrated, or should MVP use only captured/uploaded
  images and layout primitives?
- Should generated citations be mandatory for all source-backed decks?
