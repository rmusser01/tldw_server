# Large political PDF: ingestion, research and review UAT

Date: 2026-11-15. Task: TASK-15504. Code baseline: `3451c811d0`.

**Verdict: not ready for this VIP to use independently.** The actual 922-page PDF imports successfully, and useful source selection, outline, annotation and export components already exist. However, a whole-document summary receives only opening material, a natural first question is incorrectly stopped before search, and evidence cannot reliably be checked against the original PDF's pages. These are workflow and evidence-delivery failures; the two successful DeepSeek diagnostics were reasonably honest about incomplete inputs.

This is an agent-operated cognitive walkthrough with real ingestion and limited real generation, not a participant study or a completed answer-quality benchmark. No product code or dependencies were changed. Findings below distinguish browser reproduction, controlled API diagnostics, code inspection and unfinished checks.

## Document and runtime

Source: [Mandate for Leadership, full PDF](https://static.heritage.org/project2025/2025_MandateForLeadership_FULL.pdf). Download: 4,721,064 bytes; SHA-256 `7f2e97507370bbadb109034ecda0085c5a2e5f3837cfb843da272c3a41523c4c`. Physical pages: 922; native text on 921. The contents organize the book into five sections and 30 numbered chapters. Preserve opposing contributions rather than presenting the book as having one position on every topic.

The local Next.js app and full FastAPI service ran against disposable databases and a separate media Jobs worker. Upload used the real UI, with Analysis off and automatic balanced chunking on. Vector embeddings were not requested. DeepSeek was subsequently enabled with the user's authorization; the supported model used was `deepseek-flash`. The supplied credential was confined to a temporary environment file, never added to repository evidence.

The existing mixed Bun/pnpm installation initially loaded two React instances, crashing Research with `useReducer` null at `useVirtualizer`. A temporary runtime-only React alias allowed testing to continue. This is a local dependency-installation finding, not a demonstrated product hook defect. Subsequent browser findings are from that runtime. No production build or clean-install certification is claimed.

Extraction produced 344,792 words and 2,620,160 normalized characters. Beginning, middle and late-book samples were present; this is not a page-by-page completeness proof. Media displayed 95 derived outline items, which are not equivalent to the book's 30 numbered chapters. Source metadata lacked title/author/bookmarks, so import fell back to the filename and “Unknown.”

## Actual walkthrough

| Step | Observed outcome | Assessment |
|---|---|---|
| Media → Add media → upload PDF → configure → review → process | Real upload completed in 112 seconds server time, about 1:56 displayed. Text and automatic chunks were saved. | Extraction/storage passed this sample. |
| Inspect imported Media | Full extracted text, 95 derived outline items, chunking completed, vector not indexed. Result said “Saved” and “Knowledge readiness unconfirmed.” | Useful diagnostics, but “what can I do now?” remains unclear. |
| Ask this item → Knowledge | Selected scope correctly contained Media 1; external web fallback was off. | Source handoff passed. |
| Ask “What does this document propose for the Department of Education? Cite the relevant pages and distinguish proposals from enacted policy.” | A deterministic clarification asked what document/context was intended. UI announced “Search complete. 0 sources found.” | Failed before retrieval; not evidence of a missing index. |
| Research → My Media → add/select the PDF | Scope was explicit. Selection survived refresh. | Passed same-browser continuation. |
| Studio → Summary, DeepSeek Flash | Actual request contained only the first 6,000 source characters and an 800-token output cap. Provider returned an output-limit error. At 4,096 tokens, browser retries encountered a media timeout and credential-store 503; no completed browser artifact. | Failed end to end. Three attempts then stopped. |
| Research chat → same scoped question | Retrieval request used chunk search settings. After over 100 seconds there was no answer, the question returned to the composer, and no provider completion request was dispatched. | Failed; precise cause not isolated under overlapping diagnostic load. |
| Preview and annotate source | Preview exposed the first 3,000 characters and first three chunks, with character offsets. Annotation survived refresh and was explicitly labeled browser-local. | Local annotation passed; page-level evidence review did not. |
| Export workspace | ZIP download contained the workspace and selected source. The saved source annotation was absent. | Export creation passed; annotation portability did not. |
| Phone layout, 390 × 844 | Sources/Chat/Studio tabs present; no horizontal page overflow. Status strips were compressed and the composer was crowded by setup and technical controls. | Narrow responsive spot check only. |

Screenshots: [ingestion result](assets/2026-11-15-large-political-pdf-uat/ingest-complete.png), [clarification shown as no results](assets/2026-11-15-large-political-pdf-uat/knowledge-clarification.png), [Studio failure](assets/2026-11-15-large-political-pdf-uat/studio-summary-token-failure.png), [source preview](assets/2026-11-15-large-political-pdf-uat/source-preview-annotation.png), [phone](assets/2026-11-15-large-political-pdf-uat/research-mobile.png). Compact machine-readable observations: [evidence.json](assets/2026-11-15-large-political-pdf-uat/evidence.json).

## Prioritized findings

P1 means fix before a VIP self-service pilot; P2 means substantial usability/recovery improvement. “Code-confirmed” establishes a path or limitation, not that every resulting failure was reproduced live.

### F1 — P1: whole-book summaries silently use front matter

**Browser + code + controlled generation.** Studio constructs selected-source context from at most 6,000 characters per source and 18,000 total. For this PDF, that ends inside the table of contents, around chapter 27. Summary, brief/report and comparison workflows share this collection path. Truncation metadata is computed but is not carried into the artifact's visible coverage contract.

The same captured prompt, replayed through the API with a 4,096-token cap, succeeded in 7.10 seconds. DeepSeek correctly described the input as front matter/table of contents, not the full book. Raising the cap fixes neither source coverage nor chapter summaries.

Code: [useArtifactGeneration.tsx](../../apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactGeneration.tsx), limits at line 83, clipping/truncation at 1024–1044, summary at 1471 and returned artifact data at 1511.

**Acceptance:** show exactly which chapters/pages were analyzed; represent missing/failed sections explicitly. Summarize each real chapter, then synthesize across completed chapter outputs. Preserve opposing authors and cite the passages supporting each major point. Reuse the existing Media outline, with reviewed chapter boundaries, instead of adding another disconnected outline.

### F2 — P1: the retained original is missing after a successful job

**Browser + API + code.** Quick Ingest requests `keep_original_file=true`, but Media reports `has_original_file:false`; `GET /api/v1/media/1/file` returns 404, “No original file available for this media item.” The Jobs worker calls document processing directly and cleans staging, bypassing original-file retention performed by the other persistence orchestration path.

Code: [quick-ingest-batch.ts](../../apps/packages/ui/src/services/tldw/quick-ingest-batch.ts), lines 465/1165; [media_ingest_jobs_worker.py](../../tldw_Server_API/app/services/media_ingest_jobs_worker.py), lines 660/758; [persistence.py](../../tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py), line 3282.

**Acceptance:** completion must include a retrievable retained original when requested, verified after staging cleanup and restart. Make failure to retain it visible rather than reporting the research source fully ready.

### F3 — P1: page provenance is not available for verification

**Browser + API + code.** Default extraction produces flat Markdown. Retrieved excerpts expose offsets/section paths, but no physical or printed page metadata. Preview opens the beginning of the source, not the passage cited in an answer; a “Citation ready” label does not establish page-verifiable evidence. Missing original retention compounds this, but fixing retention alone does not restore page mappings.

Code: [PDF_Processing_Lib.py](../../tldw_Server_API/app/core/Ingestion_Media_Processing/PDF/PDF_Processing_Lib.py), line 437; [persistence.py](../../tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py), line 5826; [SourcesPane](../../apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/index.tsx), preview at 1092/2813; [sourceOpenAction.ts](../../apps/packages/ui/src/components/Option/KnowledgeQA/sourceOpenAction.ts), line 6.

**Acceptance:** retain extraction-to-PDF page spans through chunks, retrieval, answer citations, notes and exports. One click opens the supporting passage in the original. Display PDF page and printed page separately when known; do not assume a constant offset for every document. Qualification must survive saving/sharing when a page or supporting span is unavailable.

### F4 — P1: selected-document questions trigger a misleading clarification failure

**Browser + executed production heuristic + code.** “This document” is treated as an ambiguous pronoun despite `include_media_ids:[1]`. The assessor sees query/history, not selected scope. Research chat already handles explicit scope; Knowledge QA does not apply the same policy. Its stream handler then loses the distinct clarification outcome and renders a completed search with no results.

Code: [KnowledgeQAProvider.tsx](../../apps/packages/ui/src/components/Option/KnowledgeQA/KnowledgeQAProvider.tsx), lines 2578/2713; [clarification_gate.py](../../tldw_Server_API/app/core/RAG/rag_service/clarification_gate.py), lines 19/44; [unified_pipeline.py](../../tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py), lines 3184/3198. Existing scoped policy: [ragMode.ts](../../apps/packages/ui/src/hooks/chat-modes/ragMode.ts), line 440. Related work: TASK-478.5, TASK-13260.4, TASK-15482.

**Acceptance:** the exact first question above searches the selected PDF. The same genuinely unscoped question may request clarification. Preserve explicit overrides, and render “Needs clarification; search not run” distinctly. Do not globally disable clarification or undo the existing single-generation contract.

### F5 — P1: natural-language retrieval takes an expensive fallback and misses important evidence

**Controlled API + code; browser failure is consistent but not fully attributed.** Explicit FTS/chunk search of a related Education/funding question returned ten excerpts in 84.98 seconds. The diagnostic asked what the chapter proposes for the federal department and who should control elementary/secondary funding, with printed/PDF citations. It was not identical to the shorter browser question. Results were marked `late_chunk`, reporting 12,915 regenerated chunks. Relevant evidence was partial; four excerpts concerned other topics, and an expected chapter-opening passage was absent. No page metadata was returned.

The FTS builder quotes a multiword question as one exact phrase. A zero-match chunk query falls back to fetching whole media and chunking it in memory, even when late chunking is not preferred. Additional query variants ran sequentially; expansion retrieval accounted for 63.38 seconds. A 60-second request setting did not impose a 60-second global deadline.

Code: [database_retrievers.py](../../tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py), query construction at 2720, fallback at 1296, scoring at 1470/1530; [unified_pipeline.py](../../tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py), per-call timeout at 2959 and expansion at 4438.

Default API probes that omitted `fts_level` returned a full 2.62-million-character document. These **do not reproduce Research's default chunk-level request**. Neither result proves the persisted chunk index is broken. Timings include possible overlap with browser tests, so they are observations, not a clean performance benchmark.

**Acceptance:** ordinary questions retrieve focused, relevant persisted excerpts without reprocessing the entire book; a chapter-scoped query reaches that chapter; cancellation and overall deadlines bound work. Measure retrieval separately from generation with a prewarmed, isolated run. Compare actual chunk IDs and a single-term control before claiming an indexing defect.

### F6 — P1: “Full source text” can remove the question from the retrieval input

**Code-confirmed, not executed with this giant prompt.** Chat's full-source option prefixes the document to the user question. The API client truncates a long retrieval query to its first 20,000 characters. With this book, the actual question falls outside that prefix. The final model context also risks exceeding its input limit.

Code: [ChatPane](../../apps/packages/ui/src/components/Option/ResearchWorkspace/ChatPane/index.tsx), lines 2311–2332; [TldwApiClient.ts](../../apps/packages/ui/src/services/tldw/TldwApiClient.ts), query normalization at 1925.

**Acceptance:** the retrieval query remains the user's actual question. Document context is bounded independently. Offer a clear long-document path without asking a nontechnical reader to understand model context windows.

### F7 — P2: recovery requires technical decisions and obscures the stage that failed

**Browser + API + code.** Studio's 800-token default failed for this reasoning model; increasing it exposed separate media/credential failures. Credential refresh errors later recovered with the same key, without bypassing checks. Logs do not establish database contention versus event-loop delay. Research chat did not reach model dispatch during its failed attempt. “Processing,” “Text searchable,” “Knowledge readiness unconfirmed,” and “Citation ready” describe different capabilities but do not give the user one clear next step. Native text was described as “OCR + Extract + Chunk.”

**Acceptance:** show upload, extraction, search preparation, generation and saving as distinct stages, with measured progress and an actionable retry for the failed stage. A model-compatible default must leave room for reasoning plus a final response. Do not invite a credential change when the failure is temporary store unavailability. Preserve the question and completed work through retry.

### F8 — P2: research outputs do not form one understandable review-and-sharing flow

**Browser + code.** Media already offers an outline; Research does not connect it to chapter-by-chapter summary coverage. “Compare sources” requires multiple sources, so comparing opposing positions inside this one book lacks a direct entry point. Source annotations survive local refresh but were absent from the workspace ZIP. Research's note-source projection also omits page/citation fields supported by the Notes provenance contract; that omission is code-confirmed, not a successful answer-save reproduction.

Annotations are explicitly browser-local and stored separately from the workspace snapshot. Their omission is an export-completeness limitation without an export-specific exclusion notice, not a failed save or proof that canonical Notes were lost. Workspace export contains state/references; it is distinct from an answer/brief export or a PDF backup. See [workspace.ts](../../apps/packages/ui/src/store/workspace.ts), line 4382, and [workspace-bundle.ts](../../apps/packages/ui/src/store/workspace-bundle.ts), line 192.

Code: [StudioPane](../../apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/index.tsx), comparison gate at 2287; [workspace-rag-sources.ts](../../apps/packages/ui/src/utils/workspace-rag-sources.ts), line 98; [knowledge-note-provenance.ts](../../apps/packages/ui/src/utils/knowledge-note-provenance.ts), line 15.

**Acceptance:** let a reader select chapters/sections as the scope for explain, compare and collect-evidence actions. A saved brief retains claim, quotation/excerpt, author/section, citation, interpretation and limitations. Clarify which annotations are local and which are portable; verify export/import rather than treating download success as research preservation.

Reuse [useMediaNavigation.ts](../../apps/packages/ui/src/hooks/useMediaNavigation.ts): `useMediaSectionContent` already retrieves an outline node's title, content and target range. Carry the node identity and navigation version into each result. The existing section navigator and reading-position handling provide the starting point; a new outline parser is not a prerequisite.

## What worked, and what model quality testing established

Source handoff and explicit selection are useful safeguards. Automatic structure-aware chunking and the existing Media outline provide a foundation. Annotations and selected sources survived refresh. The export action produced a valid ZIP. Phone layout had no horizontal page overflow in the single checked state.

Two substantive DeepSeek API generations completed after retrieval load settled:

- Cached Education excerpts, 4,096-token cap: 5.65 seconds. The answer attributed its supported proposal and explicitly declined missing funding details/page citations. **Faithfulness to supplied excerpts was acceptable; completeness and page verification failed because evidence was missing.**
- Exact captured Studio summary prompt, only cap raised to 4,096: 7.10 seconds. The answer recognized front matter/contents and truncation. **This is not a book-summary pass or browser UAT pass.**

The model was not the only bottleneck. A faithful model can still give the VIP an incomplete answer when the application selects insufficient text. Conversely, model honesty in two examples is not evidence that it will consistently preserve political nuance.

## Recommended VIP flow and implementation sequence

Use the existing Research Workspace as the destination, with the Media outline and current Knowledge/Notes export capabilities connected to it:

1. **Add the PDF.** Show its recognizable title, page count, retained-original status and preparation progress. Offer automatic settings; advanced provider/chunk/retrieval settings stay secondary.
2. **Choose an outcome:** understand the book, explain a chapter, ask with evidence, compare positions within the book, or build a sourced brief.
3. **Review coverage and evidence.** Show completed/partial/unread chapters. Separate what the document says from the assistant's interpretation and from facts requiring current external sources. Open each supporting passage directly.
4. **Save and share.** Keep evidence, qualifications and selected notes with the output; reopen or import the result and verify them.

Suggested reviewable work units, in order:

| Unit | Scope | Exit gate |
|---|---|---|
| A. Trustworthy first answer | F2–F4: retained original, page mappings, selected-source clarification and truthful UI states | Exact first question → supporting passage → original page; metadata survives note/export. |
| B. Long-document coverage | F1/F5/F6: focused retrieval, bounded work, chapter-based summaries, full-source query separation | All 30 chapters accounted for; late-book evidence and both sides of an internal disagreement reachable. |
| C. Guided review and reuse | F7/F8: readiness/retry, outcome actions, chapter comparisons and portable annotations | Nontechnical reader completes upload → explanation → verification → saved brief without configuring RAG. |
| D. Acceptance with the VIP | Fixed question set, isolated timings, then observed customer session | Correct answers and citations, complete coverage where claimed, graceful unknowns and reproducible saved output. |

These are proposed scopes, not implemented fixes or completion promises. Avoid a new dashboard or separate analysis engine before connecting and repairing the existing surfaces.

ADR check: **no new ADR required for this review**; no durable architecture rule changes. Future design should preserve [ADR-007, canonical Research Workspace](../ADR/007-research-workspace-canonical-first-slice-shell.md), [ADR-003, Jobs ownership](../ADR/003-jobs-vs-scheduler-default.md), and [ADR-065, independent Notes provenance](../ADR/065-independent-notes-knowledge-provenance.md). Reassess during implementation design.

## Repeatable acceptance scenarios

Use this exact PDF and a second unrelated source to check scope exclusion. Gold evidence should be reviewed directly in the PDF before scoring; do not let the same generation judge establish its own correctness.

| Scenario | Required behavior | Current evidence |
|---|---|---|
| Education question above | Relevant chapter evidence, complete requested major points, verifiable citations, proposals distinguished from enacted policy | UI clarification failed; controlled retrieval/generation partial. |
| FBI oversight | Find the relevant passage, name the author/section and open its PDF page | Gold source identified; answer generation not completed. |
| Opposing Export–Import Bank contributions | Explain both positions, attribute authors separately, cite both; do not flatten disagreement | Default retrieval probe only; generated answer unverified. |
| Opposing trade contributions | Same two-sided evidence requirement within one PDF | Gold source identified; generated answer unverified. |
| Federal Reserve chapter | Chapter-specific explanation with major points and caveats | Gold source identified; generated answer unverified. |
| Late-book FTC passage | Retrieve near-end evidence, proving retrieval is not limited to opening text | Default retrieval probe only; generated answer unverified. |
| Which proposals are enacted as of the test date? | Explain that this document alone cannot establish current enactment; external corroboration must be separately sourced | Retrieval probe only; negative-answer generation unverified. |
| Chapter summaries and shareable brief | Five sections/30 chapters accounted for, coverage visible, evidence and annotations survive export/reopen | Studio coverage failed; ZIP/refresh partially passed. |

Also test no matches, provider failure, interrupted preparation, retry without duplicate ingestion, cancellation, reopening after restart and export/import on another browser. These failure and portability checks are not completed by the present pass.

## Verification and limits

Existing focused tests passed during this review: 29 PDF normalization/preflight/upload tests, 24 media-worker tests, and 69 frontend query-length/provenance tests. These 122 passes validate narrow existing contracts; they do not override the reproduced UAT failures. No new implementation tests were added because no product behavior changed. Bandit is not applicable to the Markdown/JSON/image-only deliverables.

Equivalent backend reproduction commands, from the repository root (original presentation flags were not retained):

```sh
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Media/test_pdf_text_normalization.py tldw_Server_API/tests/Media/test_document_upload_processing.py --basetemp=/tmp/project2025-uat-tests -o cache_dir=/tmp/project2025-uat-pytest-cache
python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py --basetemp=/tmp/project2025-worker-tests -o cache_dir=/tmp/project2025-uat-pytest-cache
```

Frontend command, from `apps/packages/ui`:

```sh
node node_modules/vitest/vitest.mjs run src/services/__tests__/tldw-api-client.rag-query-length.test.ts src/utils/__tests__/workspace-rag-sources.test.ts src/utils/__tests__/knowledge-note-provenance.test.ts --maxWorkers=1
```

An independent mechanical UI scan reported zero findings for a target directory containing 147 files; its output did not report a parsed-file count. It does not establish accessibility or usability. The qualitative review instead found unclear system states, technical vocabulary, weak recovery and excessive controls around the main task. No screen-reader audit, clean-install test, semantic indexing run, OCR/scanned-PDF test, full chapter summary, full question benchmark, cross-device restore or successful browser answer-to-brief export is claimed. Ingest processing exceeding the five-minute client polling timeout and direct chat-attachment limits remain code risks, not reproduced failures in this walkthrough.

Evidence intentionally excludes the original PDF, raw prompts, full document-containing API responses, credentials and runtime databases. Temporary runtime files are diagnostic working material, not a portable acceptance environment.

Cleanup: the disposable frontend, backend and media worker were stopped; their credential-bearing temporary configuration was scrubbed. The source/answer findings above remain open product work, although this review task is complete.
