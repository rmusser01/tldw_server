# Political PDF research: approved remediation design

Task: TASK-15505 and children .1–.4. Baseline: a1ec6cc5d1. Approval: user requested implementation of the prioritized [UAT findings](../Reviews/LARGE_POLITICAL_PDF_UAT_2026_11_15.md) and an end-to-end rerun with the same PDF and DeepSeek on 2026-11-15.

## Outcome and scope

A nontechnical researcher uploads a large political document, understands its chapters, asks questions with inspectable evidence, compares opposing positions, and exports research that preserves provenance and annotations. Fix the existing Media → Knowledge/Research → Document viewer → Notes/export flow. Preserve WebUI/extension parity, owner isolation and explicit expert settings. No new dashboard, provider abstraction, vector service or political-document-specific parser.

Use the existing components and storage owners. Merely increasing prompt limits would still omit content or exceed context; a separate document-analysis engine would duplicate existing navigation, Jobs and artifacts. The selected approach extends the established mechanisms with exact provenance, bounded generation and visible coverage.

## A. Retained originals and exact locations (TASK-15505.1)

Extract existing original retention into one shared orchestration helper. Call it before Jobs staging cleanup and from the synchronous ingest path. Preserve quota/path validation, checksums, registration rollback and replacement cleanup. Successful persisted results with warnings remain eligible. Propagate `original_file_stored`; failed retention is a visible partial success, never a false guarantee.

Native PDF extraction uses per-page Markdown, normalizes each page, joins deterministically and computes offsets against the final stored string. Optional safe metadata:

```json
{"page_count":2,"page_map_content_hash":"<SHA-256 of exact stored UTF-8 text>","page_map":[{"page_number":1,"start_char":0,"end_char":120,"printed_page_label":"i"},{"page_number":2,"start_char":122,"end_char":450}]}
```

Physical pages are one-based; character ranges are inclusive/exclusive Python string positions. Preserve blank pages. Printed labels require evidence from the PDF, never a guessed constant offset. OCR/alternative parser replacement must rebuild provenance or explicitly omit it. Existing imports without a map require re-extraction for reliable locations.

Intersect persisted chunk ranges with the map. Keep compatibility `page_number` plus complete `page_spans`; retrieval must not prepend titles or strip mapped source text before using offsets. Reuse arbitrary chunk metadata storage; avoid schema migrations where existing metadata suffices.

Move exact navigation resolution to core for use by both the API and output worker. Section content returns `navigation_version`, `resolved_start_char`, `resolved_end_char`, `page_spans`, `coverage_status` (`exact` or `unavailable`). A heading includes child sections until the next same/higher-level heading. Physical bookmark pages and printed TOC pages are distinct; unresolved ranges must not silently become the full source. Validate stale navigation versions before generation.

## B. Scoped, focused and bounded questions (TASK-15505.2)

Validated selected Media/note IDs disambiguate the default clarification decision. Explicit true/false settings still win. Preserve clarification as a distinct streaming/nonstream/persisted outcome; announce that search did not run.

Natural-language FTS questions use a bounded persisted-chunk term fallback rather than exact-phrase-only matching. Preserve deliberate quoted/Boolean/prefix syntax. Do not implicitly regenerate all chunks for Media that already has stored chunks. Preserve supported fallback for unprepared Media and explicit late-chunk requests, with bounded work and cooperative cancellation.

Use one request deadline across retrieval/expansion, pass remaining budgets to stages and stop optional work after expiry. Forward server timeout settings instead of relying only on streaming idle timeout. Blocking work must yield/offload safely; thread offloading alone is not proof of cancellable computation.

Carry the original retrieval question separately from full-source prompt context. Keep context bounded and visibly explain when focused evidence replaces an oversized full-text request. Do not discard the actual question or silently present an opening prefix as a complete document.

## C. Complete section summaries and comparisons (TASK-15505.3)

Extend existing Research Workspace output Jobs for `summary` and `compare_sources`; do not create a browser-owned queue. Typed section selection references `{source_id,node_id,navigation_version}`. Resolve ownership and exact ranges server-side. The existing outline proposes sections; the user can review/select chapter starts. Do not assume 95 derived nodes equal 30 chapters or hardcode this book.

Read complete selected ranges and partition into bounded batches without gaps. Summarize batches, synthesize each section, then synthesize completed sections. Preserve opposing authors/positions and supporting page evidence. Missing or failed material stays visible and cannot count as covered. Persist per-section/batch checkpoints in canonical artifact `version_metadata.workspaceClient.data.sectionAnalysis`. Job retry/resume skips completed compatible work; stale source/navigation invalidates reuse. Cancellation and reopening retain the job/artifact identity.

Reuse Media navigation controls in Studio, with named section selection, page ranges, completed/failed/unread counts, retry/cancel and source links. Both comparison gates accept two sections from one source. Use a 4,096-token unset output default compatible with the tested reasoning model, preserving explicit settings. Remove separate low implicit caps that prevent a final answer. Source text, model output and evidence remain separate untrusted inputs.

Render coverage, citations and limitations into the artifact's Markdown as well as structured metadata so individual text exports remain honest. Existing output jobs, workspace ownership, canonical artifacts and codecs remain authoritative.

## D. Evidence review, export and UX (TASK-15505.4)

Route citation clicks to the existing authenticated Document Workspace at a validated physical page, preserving selected source and same-document navigation. Clearly label PDF versus printed pages. Keep unsupported/unavailable evidence qualified rather than showing a positive citation claim. Preserve page/citation fields through Research → Notes projection and restoration.

Keep source annotations browser-local as today, but include a validated optional `sourceAnnotations` payload in workspace bundles and restore it under the imported workspace identity. Reject malformed or unrelated source references. Old bundles remain importable. Explain the local/live versus exported snapshot contract in the UI. Verify individual summary/brief export as well as workspace ZIP roundtrip.

Track UX issues using NN/g's ten usability heuristics and 0–4 severity, considering frequency, impact and persistence. The ledger records scenario, heuristic, severity rationale, observed evidence, fix owner, tests and live retest disposition. This is a heuristic evaluation and agent walkthrough, not a participant study or NN/g certification. Consult [NN/g heuristics](https://www.nngroup.com/articles/ten-usability-heuristics/) and [severity guidance](https://www.nngroup.com/articles/how-to-rate-the-severity-of-usability-problems/). Preserve the incumbent visual system; focus on state, familiar language, recovery, recognition and evidence.

## Verification contract

Regression tests cover real retention after cleanup; page-map normalization/Unicode/blanks/cross-page chunks; exact parent section bounds; scoped/unscoped and explicit clarification controls; persisted natural-language retrieval and explicit search syntax; deadline/cancellation; original question preservation; complete range partitioning; resume without duplicated successful calls; owner/stale-version rejection; page links and Notes; export/import annotations and coverage. Run scoped linters, type checks and Bandit before completion.

Rerun the actual 922-page source in fresh disposable state with DeepSeek Flash. Verify original retrieval, all 30 reviewed chapter ranges, chapter summaries and a whole-document synthesis, scoped Education/FBI/Fed/late-FTC questions, opposing Ex-Im/trade positions, enactment uncertainty, page-opening actions, artifact save/reopen, text export and workspace export/import. Verify gold evidence directly from the PDF. Measure retrieval and generation separately with no overlapping heavy probes. Failures remain open until corrected or reported concretely; no mock-provider result substitutes for this acceptance run.

## ADR assessment

ADR required: no new ownership decision planned. This design follows [ADR-003](../ADR/003-jobs-vs-scheduler-default.md), [ADR-007](../ADR/007-research-workspace-canonical-first-slice-shell.md) and [ADR-065](../ADR/065-independent-notes-knowledge-provenance.md). It adds backward-compatible provenance and output payloads under their existing owners. Reassess if implementation needs a new persistence owner, public lifecycle or queue architecture.
