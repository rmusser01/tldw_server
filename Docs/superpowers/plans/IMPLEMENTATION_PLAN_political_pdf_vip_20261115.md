# Political PDF VIP workflow implementation plan

> Execute using subagent-driven development with focused regression tests and independent review. Preserve this ledger through compaction; do not redispatch completed units.

**Goal:** Complete the approved PDF ingestion → chapter understanding → cited QA → reviewed/exported research flow.
**Architecture:** Extend current ingestion, Media navigation, RAG, Research output Jobs and artifact/export contracts. No parallel analysis subsystem.
**Tech stack:** Python/FastAPI/SQLite metadata/Jobs; shared React/TypeScript/Next.js; pytest/Vitest/Playwright.
**Spec:** [Approved design](../../Design/POLITICAL_PDF_VIP_FIXES_2026_11_15.md).
**Tracking:** TASK-15505, children .1–.4. Worktree `/Users/macbook-dev/.codex/worktrees/political-pdf-vip-fixes/tldw_server2`; baseline `a1ec6cc5d1`.

## Global constraints

- Real same PDF and DeepSeek acceptance; no hardcoded book structure or fake successful generation.
- All source mutations tracked before edits; only backlog-py edits task records. Existing user checkout untouched.
- Preserve ownership, upload validation, quota enforcement, explicit settings, source scope and WebUI/extension parity.
- Physical pages one-based; exact stored-text spans inclusive/exclusive; unknown printed pages remain unknown.
- Test first for behavior changes, then minimal implementation, scoped tests/lint/Bandit, independent review and local commit.
- ADR assessment: no new ADR planned; preserve ADR-003/007/065 as specified in design.

## Stage 1: Original and location integrity — TASK-15505.1

**Status:** In Progress
**Goal:** Retain original and preserve exact page/section spans.
**Files:** `persistence.py`, `PDF/PDF_Processing_Lib.py`, `media_ingest_jobs_worker.py`, Media navigation endpoint/core; corresponding Media/MediaIngestion tests.
**Interface:** Safe metadata `page_count,page_map`; chunk metadata `page_number,page_spans`; exact section resolver/content contract from design A.
**Success criteria:** File survives staging cleanup/reopen; known page and parent section slices equal stored content; unavailable maps fail honestly.
**Tests:** `test_persistence_original_storage.py`, `test_original_file_replacement.py`, `test_media_ingest_jobs_worker.py`, `test_pdf_text_normalization.py`, `test_media_chunk_fts_metadata.py`, `test_media_navigation_content.py` plus property-based partition tests where useful.

- [ ] Add failing cleanup-retention and exact-range regressions, including Warning results and quota refusal.
- [ ] Share existing retention safeguards; generate/allowlist page map; enrich chunk metadata.
- [ ] Replace proportional/fullsource navigation fallback with exact shared core resolution.
- [ ] Run focused tests, formatting/lint/Bandit; review and commit.

## Stage 2: Scoped questions and bounded retrieval — TASK-15505.2

**Status:** In Progress
**Goal:** First scoped question retrieves relevant persisted chunks and reports truthful outcomes within bounded work.
**Files:** RAG `unified_pipeline.py`, `streaming_executor.py`, `database_retrievers.py`; KnowledgeQA provider/state/view; `ragMode.ts`, chat-action params, ChatPane; request builder.
**Interface:** Explicit clarification overrides preserved; original retrieval question separate from prompt message; request-wide timeout forwarded to backend. Page-map retrieval consumption coordinated with Stage1.
**Success criteria:** Scoped/unscoped question policies differ appropriately; explicit phrase semantics retained; prepared Media avoids implicit full re-chunk; expiry stops further variants.
**Tests:** Existing clarification pipeline/streaming, retrieval/FTS edge cases, KnowledgeQA authority/a11y, ragMode/ChatPane/query length; actual Education/late-book passage evaluation during Stage5.

- [ ] Reproduce exact scoped question, clarification SSE outcome and persisted-chunk natural-language miss in failing tests.
- [ ] Repair defaults/state and focused term fallback; preserve unprepared-media and explicit advanced paths.
- [ ] Carry original question separately; forward overall deadline and implement bounded cancellation.
- [ ] Run focused tests, lint/Bandit, review and commit.

## Stage 3: Complete chapter outputs — TASK-15505.3

**Status:** Not Started
**Goal:** Resumable chapter summaries/comparisons cover selected exact ranges and expose exclusions.
**Files:** `core/Research_Workspace/output_jobs.py`, output schemas/endpoint, output worker; Studio artifact generation/derived state/settings/rendering; existing Media navigator; output-job tests.
**Interfaces:** Consume Stage1 core exact resolver; selected section references and canonical `workspaceClient.data.sectionAnalysis` from design C; existing Jobs submission/status/cancel and artifact persistence.
**Success criteria:** Every input character in selected sections is batched once; retries reuse successful compatible batches; canonical outputs retain citations/coverage; two sections from one PDF can be compared.
**Tests:** Partition coverage, child headings, stale version and unauthorized scope, failure/retry/cancel/restart, canonical save/read, output token defaults, comparison gates, export-visible partial labels.

- [ ] Add failing contract/partition/authorization/checkpoint tests before worker changes.
- [ ] Extend existing Jobs types/handler with bounded batches, exact evidence and resumable checkpoints.
- [ ] Connect Studio section selection, progress/retry/cancel, model defaults and Markdown coverage.
- [ ] Run tests, lint/type checks/Bandit, review and commit.

## Stage 4: Citation review, export and UX — TASK-15505.4

**Status:** Not Started
**Goal:** Open the right original page and preserve research across save/export/import.
**Files:** Knowledge source action, Research citation handler, DocumentWorkspace URL handling, `workspace-rag-sources.ts`, workspace bundle/list slice, source annotation utility/SourcesPane, tests and locale strings as applicable.
**Interfaces:** Validated `page_number/page_spans` from Stage1; existing authenticated PDF viewer; optional `sourceAnnotations` on workspace bundle; canonical Notes provenance fields.
**Success criteria:** Click→correct PDF page; page/citation evidence survives Notes; annotations/coverage survive imported workspace with remapped identity; clear readiness/recovery copy.
**Tests:** Page-link validation, same-document navigation, malformed metadata, old/new bundle import, unrelated-source rejection, source-ID remapping, refresh and individual artifact export.

- [ ] Add failing page navigation/projection/bundle roundtrip tests.
- [ ] Connect existing viewer and preserve provenance fields; include validated source annotations in snapshots.
- [ ] Improve task-facing labels/recovery based on ledger findings without redesigning unrelated surfaces.
- [ ] Run scoped verification and independent review; commit.

## Stage 5: Real acceptance and NN/g retest — TASK-15505

**Status:** Not Started
**Goal:** Prove the complete workflow with the actual PDF and DeepSeek; retain reproducible evidence.
**Success criteria:** Fresh real ingestion, 30 reviewed chapter summaries, complete book synthesis, cited early/middle/late answers and both opposing positions, honest current-enactment unknown, citation-page verification, save/reopen, text export and ZIP import. Fix encountered blockers and rerun affected paths.
**Tests:** Approved design verification contract; desktop/phone and keyboard spot checks; scoped full regression, lint/type checks/Bandit; independent code and heuristic review.

- [ ] Establish isolated services and fresh DB; configure existing authorized DeepSeek credential only in temporary environment.
- [ ] Verify PDF hash and manually checked chapter/page gold references; record actual results and timings separately for retrieval/generation.
- [ ] Perform browser flow including successful artifacts and export/import; record NN/g heuristic/severity/evidence/retest for every observed issue.
- [ ] Fix required remaining failures, verify targeted paths, review branch, stop temporary services and scrub credentials.
- [ ] Finish task records, final evidence/report and commits. Remove only this completed implementation plan per repository policy; preserve durable design/retest report.

## Execution ledger

- 2026-11-15: User approved report priority sequence and real end-to-end rerun. Read-only specialists traced original retention, exact navigation, scoped RAG and existing output Jobs. Managed worktree created from review commit; original checkout preserved.
- Design clarification: current navigation's proportional page estimates cannot substantiate chapter coverage; exact span resolution is part of Stage1. Existing output Jobs support extending summaries without a new queue owner.
- No implementation unit completed yet. NN/g severity ledger will live in the durable retest report.
