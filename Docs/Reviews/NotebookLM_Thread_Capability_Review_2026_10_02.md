# NotebookLM thread capability review

tldw_server already implements most of the concrete needs raised in the linked NotebookLM discussion. The strongest follow-ups improve evidence precision, scholarly exports, structured data analysis, and consistency across research and audio workflows.

## Scope and evidence

- Review date: October 2, 2026.
- Reviewed remote dev revision: `8140e493f2d0a79e2039084930151eba6565df82`.
- Discussion: [Is notebooklm slowly dying](https://www.reddit.com/r/notebooklm/comments/1wn3or8/is_notebooklm_slowly_dying/). Treat its anecdotes as user feedback, not a benchmark or proof of product decline.
- Tracking: [TASK-13421](../../backlog/tasks/task-13421%20-%20Save-NotebookLM-thread-capability-review-and-follow-up-proposals.md).

The review inspected Git objects at the pinned revision, implementation paths, existing tests, and Backlog records. The local dev checkout was a July snapshot and was not used as current capability evidence. No live provider, browser, or test-suite execution was performed for this review. Code presence, tests present, recorded historical acceptance, and fresh runtime certification are distinct forms of evidence.

This document saves observations and proposed discussion topics. The five follow-ups below are not approved designs, implementation plans, or completed features.

## Current support and remaining opportunities

| User need | Existing support | Remaining opportunity |
| --- | --- | --- |
| Research grounded in selected sources | Explicit Media, Note, and collection scoping; strict extractive answering; citation validity and missing-evidence states; optional claim checks. [Scope][scope] | Preserve the selected corpus through verification and repair, and verify claims against their cited passages. |
| Inspectable citations | Source previews, chunk information, citation links, and answer trust states. [Trust][trust] [Preview][preview] | Carry exact passage locators into citation navigation, and distinguish resolvable citations from verified claims. |
| Scholarly bibliography | MLA, APA, Chicago, Harvard, IEEE, and BibTeX formatting; reference extraction and enrichment; workspace citation export. [Styles][styles] [References API][reference_api] | Correct publication metadata, export the full bibliography, and improve reference-type and format interoperability. |
| Reference managers | Zotero collection browsing, metadata normalization, attachment handling, import and sync backend support. [Zotero][zotero] | A Zotero connection/import/sync UI was not found in the searched shared UI scope. Expose the existing backend in the research workflow. |
| Permanent instructions | Saved normal-chat and RAG prompts, assistant prompt snapshots, and persona mechanisms. [Saved prompts][prompts] | Define how workspace instructions apply consistently to chat and generated artifacts. |
| Audio voices and personalities | Workspace voice selection, preview, backend/speed/fallback settings; Watchlists cast, persona, tone, and custom editorial instructions. [Audio][audio] [Preview][voices] [Cast][cast] | Extend the existing cast/editorial controls to Audio Overview and finish Podcast host/guest voice wiring. |
| Personal statistics and uploaded tables | Markdown/CSV/TSV/JSON table processing, structured OCR output, date filtering, and numeric guardrails. [Tables][tables] [OCR][ocr] [Numbers][numbers] | Bind values to a source, event date, entity, metric, row/cell, and unit; preserve calculation and OCR provenance. |
| Quality stability | Selectable providers/models, answer-quality comparison recipes, baseline and regression infrastructure. [Recipe][eval] | Use exact factual regression cases for dates, values, missing data, citations, and reload behavior. |

## Concrete findings

### Citation validity and verification scope

The Knowledge QA classifier explicitly checks that citations resolve to returned sources with inspectable evidence; it does not judge whether those sources support each claim. Claim verification is a separate capability, so a cited answer should not be described as a guarantee of factual correctness. [Trust contract][trust]

Optional post-generation verification receives base documents and database paths, but its call and method signature do not carry selected Media/Note IDs or collection scope. Claim retrieval searches the Media database again without an allowlist. This is a code-level scope-propagation gap; the review did not reproduce an out-of-scope answer. [Pipeline call][verify_call] [Verifier][verify]

Workspace citation clicks focus a source by Media/source identity without passing a page, timestamp, chunk index, or character span. Preview infrastructure already supports centered chunk windows. Local annotation records store typed quote text and notes without verifying the quotation or retaining an immutable passage anchor. [Navigation][navigation] [Preview][preview] [Annotations][annotations]

### Numbers and event dates

Numeric fidelity unions numbers from all retrieved documents. A value can pass because it occurs somewhere in that union, even if the answer attaches it to the wrong workout, date, or column. Media temporal filtering uses ingestion dates; that does not resolve an event date inside a source. [Numeric check][numbers] [Date filtering][dates]

Tables and OCR are useful existing foundations, but their presence does not establish a complete screenshot-to-verified-statistic workflow. Original image/crop evidence, row/cell identity, units, event dates, and OCR uncertainty need to survive retrieval and citation handling. [Tables][tables] [OCR types][ocr]

### Bibliography fidelity

Workspace BibTeX export creates an `@misc` entry for every source, derives `year` from `source.addedAt`, and includes title, Media/type note, URL, and access date. Publication authors, date, DOI, journal, and reference type are not carried into that export. [Export builder][bibtex]

The document References tab exports its currently loaded `references` array rather than fetching the entire paginated bibliography. Other existing formatting paths also have incomplete metadata mapping. No general RIS import/export, citeproc engine, or CSL style catalog was identified in the searched implementation and dependency scopes. Zotero already normalizes richer publication metadata, providing a reuse path. [Reference export][references] [Styles][styles] [Zotero metadata][zotero]

### Instruction and audio workflow coverage

Audio Overview is an implemented script-to-TTS workflow with selected voice/model/backend, speed, output format, and fallback handling. Its script prompt fixes a 2-3 minute, single-narrator format without speaker labels; the request schema has no tone, duration, cast, or custom-instruction fields. [Generation][audio] [Prompt][audio_prompt] [Request contract][artifact_contract]

Audio Studio Podcast saves `hostSpeaker` and `guestSpeaker` in section settings, but its generation action passes only `options.workflow`. The speech provider constructs one request with one voice. Those labels therefore do not currently implement alternating voices in that path. Watchlists already propagates cast, tone, and editorial instructions. [Podcast form][podcast] [Queue payload][podcast_queue] [Speech provider][speech] [Watchlists][cast]

### Verification differs across generated products

Literature Matrix and other literature workproducts are implemented and retain source coverage information. Coverage and structural JSON validation do not themselves prove each generated quotation or claim. Deep Research normalizes citations to source IDs, losing additional locator fields. Conversely, audio overview, data table, and mindmap draft generation already invoke Claims verification and reject non-grounded results. Verification should be extended consistently, not described as wholly absent. [Literature generation][literature] [Deep Research citations][deep_research] [Artifact checks][artifact_verification]

### Regression evidence and its limits

The answer-quality recipe supports comparison work, but its default grounding rubric uses token overlap. Wrong dates, swapped values, or negation can require stronger factual oracles. Existing tests and quality tooling should be extended with such cases. [Recipe][eval] [Default rubric][eval_rubric]

UAT390 records bounded native single-user acceptance on SQLite and restricted PostgreSQL, including exact source identity, cited QA, persistence/reload, provider dispatch, and absent-price refusal. The record also separates that pass from later integrated controlled-provider qualification. Studio generation qualification has a separate open record. Neither an open task nor a historical pass certifies every current deployment. [Recorded UAT390 evidence][uat] [Studio qualification][studio_uat]

## Proposed brainstorming sequence

### 1 Evidence integrity

Goal: users can restrict an answer to a selected corpus, inspect support for individual claims, and understand when evidence is insufficient. Reuse source scoping, Claims verification, trust states, and source preview machinery.

Topics to decide: strict behavior by default or by opt-in; how supported, unsupported, conflicting, and unverifiable claims appear; whether to refuse an entire answer or omit unsupported claims; which workflows receive the first complete slice. Scope propagation through optional verification is a concrete candidate repair.

Candidate acceptance cases: an unselected distractor never enters verification/repair; a valid citation to irrelevant text does not count as claim support; missing facts produce explicit abstention; citation navigation opens the actual passage; saved evidence remains identifiable after reload.

### 2 Scholarly export correctness

Goal: researchers can reuse references without fixing invented publication years or incomplete bibliography exports. Reuse reference extraction/enrichment, Zotero normalization, existing format UI, and BibTeX exports.

Topics to decide: authoritative publication metadata and missing-field behavior; priority formats and reference types; full-bibliography versus selected-reference exports; Zotero UI scope; reconciliation of metadata from multiple import sources.

Candidate acceptance cases: ingestion/access dates never become publication dates; book/article/web references retain their types; authors and DOI survive import/export; exports include all requested pages; missing metadata stays explicitly missing.

### 3 Structured data analysis

Goal: answer questions about uploaded personal data using the correct event and measurement, with reproducible arithmetic. Reuse table parsing, OCR structures, Media-owned data tables, and existing ingestion abstractions.

Topics to decide: initial CSV/JSON/table/screenshot scope; event-date and unit representation; deterministic calculation versus LLM narration; correction review; whether dedicated activity-file importers are a later extension.

Candidate acceptance cases: swapped rows, wrong-date distractors, duplicate numeric values, unit conversions, missing metrics, OCR ambiguity, and calculations with cited input rows.

### 4 Consistent instructions and audio casts

Goal: users can reuse instructions and host/voice choices across relevant research outputs. Reuse saved prompts, Watchlists cast/editorial controls, existing audio settings, and Audio API preset ownership.

Topics to decide: instruction precedence and per-output exceptions; named host presets combining persona and voice; narration versus dialogue; propagation into script generation; visible model/backend/fallback provenance. Podcast wiring is a concrete existing-path repair.

Candidate acceptance cases: saved instructions reach the actual generation request; host and guest use the selected distinct voices; settings survive reopen; previews use the selected backend; fallback use remains visible.

### 5 Release regression cases

Goal: detect factual and workflow regressions before release and when models or prompts change. Reuse evaluation recipes, frozen retrieval baselines, quality checks, and existing exact-source UAT.

Topics to decide: corpus ownership/versioning; exact deterministic oracles versus model judges; database/provider coverage; latency/cost limits; hard gates versus warnings; separation of product failures from provider and harness failures.

Candidate acceptance cases: wrong dates and swapped statistics fail even with high token overlap; fabricated excerpts fail; absent facts cannot be supplied after a disclaimer; selected-source and citation identities survive streaming and reload; repeated runs retain model/prompt/corpus versions.

## Existing work to consult

- TASK-2279.4 and TASK-2279.11 record completed citation-validity, abstention, and live Knowledge QA regression work.
- TASK-13260.278.3 tracks source-grounding qualification and contains the bounded native acceptance above.
- TASK-12020.24 tracks configured-provider Studio generation qualification.
- TASK-12020.8 records workspace share/export/import affordances.
- TASK-12450, TASK-12454, TASK-12459, TASK-573, and TASK-12623 record literature workproducts, Deep Research integration, and related verification work.
- TASK-12731, TASK-12732, TASK-12734, and TASK-12739 record Audio Studio foundation, integration, UI, and roadmap work; a completed roadmap task is not proof every roadmap feature shipped.

Search and read these records again before creating implementation tasks. This review does not create duplicate feature commitments or finalize those tasks.

## ADR assessment

ADR required: no for this documentation-only review. It records observations and unapproved proposals without changing APIs, persistence, security, ownership, or architecture.

Future designs must assess the [ADR index][adr], including ADR-007 for the canonical Research Workspace shell, ADR-023 for Media-owned Data Tables and Jobs, ADR-047 for per-user Audio preset ownership, ADR-053 for RAG result fusion, and applicable security/provider decisions. A new durable rule needs its own ADR assessment; these proposals do not amend existing accepted decisions.

## Review validation

Documentation verification checked all 33 immutable source paths and line anchors against the reviewed Git revision, the relative task link, the five proposed discussion areas, and the scope qualifications. All checks passed. Staged diff scope and whitespace are checked before commit.

Application tests, builds, audible output verification, and live-provider certification were not run. Bandit is inapplicable to this Markdown-only change; no Python or application code is touched.

[scope]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py#L298
[trust]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/tldw_Server_API/app/core/RAG/rag_service/trust_contracts.py#L203
[verify]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/tldw_Server_API/app/core/RAG/rag_service/post_generation_verifier.py#L369
[verify_call]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py#L8208
[numbers]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/tldw_Server_API/app/core/RAG/rag_service/guardrails.py#L243
[dates]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py#L1368
[tables]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/tldw_Server_API/app/core/RAG/rag_service/table_serialization.py#L61
[ocr]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/types.py#L9
[styles]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/apps/packages/ui/src/hooks/document-workspace/useCitation.ts#L8
[bibtex]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/apps/packages/ui/src/components/Option/ResearchWorkspace/workspace-header.utils.ts#L252
[references]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/apps/packages/ui/src/components/DocumentWorkspace/LeftSidebar/ReferencesTab.tsx#L486
[reference_api]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/tldw_Server_API/app/api/v1/endpoints/media/document_references.py#L1206
[zotero]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/tldw_Server_API/app/core/External_Sources/zotero.py#L421
[navigation]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/apps/packages/ui/src/components/Option/ResearchWorkspace/ChatPane/index.tsx#L2803
[preview]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/tldw_Server_API/app/core/Workspaces/source_preview.py#L118
[annotations]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/index.tsx#L1204
[prompts]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/apps/packages/ui/src/services/tldw-server.ts#L593
[audio]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactGeneration.tsx#L2019
[voices]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useAudioTtsSettings.tsx#L382
[audio_prompt]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/tldw_Server_API/app/core/Research_Workspace/artifact_generation.py#L52
[artifact_contract]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/tldw_Server_API/app/api/v1/schemas/research_workspace_artifacts.py#L11
[cast]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py#L318
[podcast]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/apps/packages/ui/src/components/Option/AudioStudio/PodcastWorkflow.tsx#L31
[podcast_queue]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/apps/packages/ui/src/components/Option/AudioStudio/useAudioStudioGenerationActions.ts#L66
[speech]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/tldw_Server_API/app/core/Audio_Studio/providers/speech.py#L50
[literature]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactGeneration.tsx#L2197
[deep_research]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/tldw_Server_API/app/core/Research/synthesizer.py#L362
[artifact_verification]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/tldw_Server_API/app/core/Research_Workspace/artifact_generation.py#L211
[eval]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/tldw_Server_API/app/core/Evaluations/recipes/rag_answer_quality.py#L42
[eval_rubric]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/tldw_Server_API/app/core/Evaluations/recipes/rag_answer_quality_execution.py#L689
[uat]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/backlog/tasks/task-13260.278.3%20-%20Verify-exact-source-identity-and-grounding-in-ingestion-to-Chat-acceptance-UAT390.md#L59
[studio_uat]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/backlog/tasks/task-12020.24%20-%20Certify-Research-Workspace-Studio-generation-with-a-configured-provider.md#L29
[adr]: https://github.com/rmusser01/tldw_server/blob/8140e493f2d0a79e2039084930151eba6565df82/Docs/ADR/README.md#L1
