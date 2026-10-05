# Knowledge UX remediation design

Approved requirements: the requester reviewed the NN/g audit and instructed us to address all K01–K15 issues and potential enhancements. Defaults: **Ask added items** after ingestion; **Research Workspace with sources** for continuation. This document makes that approved design executable without another approval gate.

Base: latest `origin/dev` 75ab224081bf140ef52017c1a9b0a04f6878d488, plus audit commit c202e2b69a. Backlog: TASK-13453 and TASK-13453.1–.4. Audit: `Docs/Reviews/KNOWLEDGE_NNG_UX_REVIEW_2026_10_04.md`.

## Intended behavior

1. Source scope is part of the user's question. Presets change retrieval/generation tuning, preserving `sources`, `include_media_ids`, `include_note_ids`, `collection_id`, `keyword_filter`, chosen generation provider/model, and Web choice. Successful ingest continues with exactly its successful added IDs. Reviewed items offer scoped Ask and Research actions. Invalid/empty transfer scope must never silently become the whole library.
2. Exact selection uses existing server title/search endpoints and pagination rather than loading only the first 200 records. Selection survives search/page changes. Counts distinguish loaded results from known totals. Unknown totals are not invented.
3. Research continuation retains original source type/ID, excerpts, links, citation mapping, question, answer qualifications, and scope. Media sources attach directly. ResearchWorkspace's existing media-backed model remains canonical: non-media evidence attaches as a labeled excerpt snapshot through existing ingestion, preserving its original reference in the snapshot and source metadata. The UI states that these are retrieved excerpts rather than a live full-note/web mirror. Partial import failure retains the payload and offers retry of the unfinished subset; successful attachments are not duplicated. The imported draft visibly retains unsupported/uncited status.
4. Saving Knowledge output retains the returned note ID and provides Open saved note. Notes distinguishes original Knowledge QA provenance from later manual edits. Analysis leads with Summary, Key claims, and Custom; advanced prompts remain available. Generation opens the saved version for review, and saving prompt defaults is named accurately.
5. Reuse existing controls for research recipes: Compare these papers, Extract claims with evidence, Summarize this interview, and Save a sourced brief set up a visible question without silently running it or changing scope. Named source summaries and explicit destination labels follow every handoff.
6. Scope, source-preview, and mobile evidence dialogs manage focus entry, containment, Escape, and return. Use established AntD/shared dialog components rather than a new modal framework. Exact selection remains visible within a short viewport with one bounded result scroll, labeled search, and reachable completion. Suggestion lists expose active selection accessibly.
7. Add/Ask appears before optional orientation. The ready state uses searchable personal-content readiness, distinguishes available services from items, and offers Add your first source when appropriate. Scope remains visible on mobile. Evidence, Send, and New Topic share reserved layout space rather than colliding. Global Cmd/Ctrl+K belongs to the command palette; Knowledge help must describe actual shortcuts.
8. Post-ingest results expose Stored/Searchable/Analysis/Needs attention as known, with actual remedy links. Wire eligible item/all-failure retries to the existing queue/session, retaining options and required files and not reprocessing successful items. No-results actions separately change included sources or retrieval depth and link to relevant media/note owners.
9. Keep shared UI and vocabulary for WebUI and extension. Validate capture-to-ingest-to-scoped-question continuity with disposable data. Preserve specialized Media, Document Workspace and Research Studio routes.

## Constraints

- No new package dependencies, modal framework, workspace abstraction, or public API family.
- Work only in the attached isolated checkout; do not edit the original dirty checkout.
- Preserve current auth/account/request-scope fencing. Transfers must not publish results after owner/server changes or expose credential material in URLs/storage/logs.
- Reuse existing ingestion, source status, Notes, workspace persistence and queue operations. Do not fabricate media IDs for notes/web results.
- Store all imported evidence/provenance via existing source/snapshot metadata and the imported note; snapshots must remain inspectable after save/reopen and visibly state excerpt limitations.
- Regression tests precede behavior changes. Test actual state/results rather than source-text mirrors or mock call counts. No new property-test dependency; use bounded table-driven invariants for ID normalization and scope retention.
- English strings use existing i18n defaultValue conventions; synchronize existing locale tooling where generated locale output is required.
- Do not push, merge or publish as part of this request.

## ADR check

ADR required: **no**. ADR-007 (`Docs/ADR/007-research-workspace-canonical-first-slice-shell.md`) retains canonical ResearchWorkspace ownership and specialized routes. ADR-008 (`Docs/ADR/008-workspace-split-key-persistence-and-indexeddb-offload.md`) governs persistence and source lineage. ADR-053 (`Docs/ADR/053-rag-cross-source-fusion.md`) governs retrieval. Repairs use those existing boundaries and a backward-compatible handoff; no new persistence owner or public API is adopted. Reassess if a required behavior cannot fit those existing contracts.

## Coverage matrix

| Unit | Audit / enhancements | Acceptance |
|---|---|---|
| 1 | K01–K04; named source-set handoffs | Every preset retains deliberate scope/model; ingest and multi-review pass exact IDs; older sources are findable; invalid handoffs fail visibly. |
| 2 | K05–K06, K15; recipes and output review loop | Research sources/excerpts/trust survive; snapshots identify originals; partial imports retry; exact saved note opens; outcome analysis and recipes are reviewable. |
| 3 | K07–K08, K10, K13–K14; minor accessibility observations | Keyboard stays in dialogs; picker is unclipped/named; first action and exact scope are visible at 390px; actions do not overlap; shortcuts/suggestions are accurate. |
| 4 | K09, K11–K12; readiness summary and extension vocabulary | Empty content receives first-add guidance; counts are accurate; failed subset retries; recovery changes its advertised dimension; both surfaces show the same scope/readiness/destination terms. |

## Verification

Use existing Vitest suites with regressions for scope/preset changes, pagination/search, simultaneous/expired/invalid transfers, imported note/web evidence and partial failure, export-to-Notes origin, analysis outcomes, modal focus, retries and empty readiness. Include the real source/pre-fill/store methods wherever practical; transport stubs are allowed at API boundaries.

Run shared UI type checks, applicable formatter/linter checks, Next/WebUI and extension builds, and relevant existing tests. Run Bandit on touched Python if any; for frontend-only changes record the non-Python scope explicitly. Browser verification uses isolated server data and a deterministic local inference provider for interface behavior, not answer-quality claims. Check one and mixed-batch ingestion, exact distractor exclusion, saved/reopened outputs, note-backed research, keyboard/short desktop/390px, and native extension capture/handoff. Stop only owned processes and clean only owned generated artifacts afterward.
