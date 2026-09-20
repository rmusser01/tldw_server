# UAT325: Character evidence readability

Backlog: TASK13260.263. The existing QA preview is plain text with query highlighting. The Character retriever manufactures Markdown headings and empty sections. Keep that plain-text UI contract; remove generated formatting noise before retrieval evidence is shown or sent to generation. Preserve authored content and metadata.

## Stage 1: Reproduce
**Goal**: Verify noisy Character evidence from both adapter and legacy retrieval.
**Success Criteria**: Real SQLite/PostgreSQL and legacy tests expose generated Markdown/empty sections and preserve source records.
**Tests**: Actual database fixtures; sparse and fully populated Character cards, authored text and metadata controls.
**Status**: Complete

## Stage 2: Repair and review
**Goal**: Share one small plain-text Character formatter between existing retrieval paths.
**Success Criteria**: Name plus nonblank labeled fields remain readable; no new renderer or storage changes.
**Tests**: Focused and adjacent Character/Chat evidence tests, Bandit, lint and independent review.
**Status**: Complete

## Stage 3: Native acceptance
**Goal**: Inspect source cards/previews and navigation on committed code.
**Success Criteria**: No generated Markdown or irrelevant empty sections, with supported authorized source actions intact.
**Tests**: Targeted SQLite/PostgreSQL acceptance alongside UAT324, source audit and owned cleanup.
**Status**: Not Started

Six corrected causal baseline failures precede final58affected passes/0skips including actual PostgreSQL. Related stale fixture task13260.264/UAT326 verified without production query changes. Bandit/Ruff clear; reviewer clear. Native acceptance remains pending.
