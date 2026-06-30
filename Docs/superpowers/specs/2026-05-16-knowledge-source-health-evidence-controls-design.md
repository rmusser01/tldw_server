# Knowledge Source Health And Evidence Controls Design

Date: 2026-05-16
Status: Review-hardened draft for user review
Owner: Codex brainstorming session
Backlog: TASK-297.7

## Summary

Improve `/knowledge` as a QA-only surface by making source readiness visible before a query and making answer evidence easier to trust and reuse after a query. This slice is intentionally conservative: it adds read-only source health, clearer evidence controls, answer-level trust summaries, and recovery copy. It does not add server-backed saved evidence, shared evidence sets, new source CRUD, or a new knowledge management hub.

The design keeps `/knowledge` focused on asking grounded questions over existing knowledge surfaces. Creation, import, editing, and full management stay in owner surfaces such as Quick Ingest, Media, Notes, Chats, Characters, Task Boards, Prompts, World Books, Dictionaries, and Workspaces.

## Current Evidence

The current active route is the shared Knowledge QA UI. `apps/tldw-frontend/pages/knowledge.tsx` delegates to `@/routes/option-knowledge`, and the shared implementation lives under `apps/packages/ui/src/components/Option/KnowledgeQA`.

Relevant current surfaces:

- `KnowledgeContextBar` controls presets, source categories, specific media/note selection, local profiles, web fallback, model/provider selection, and specific-source filters.
- `SourceList` and `SourceCard` already support source filtering, source viewing, copy text, copy citation, citation jumping, feedback, and workspace navigation for supported source types.
- `EvidenceRail` already exposes `Sources` and `Details` tabs and counts retrieved sources/citations.
- `AnswerPanel` already renders inline citation buttons, copy-answer, confidence/recovery cues, and Continue in editor handoff.
- `SearchDetailsPanel` already exposes retrieval details such as reranking, average relevance, web fallback status, candidate counts, coverage, latency, and verification metrics.
- `NoResultsRecovery` already surfaces post-query source diagnostics when `metadata.source_status` is present.
- Existing post-query source diagnostics use `KnowledgeSourceStatus` with statuses such as `searched`, `empty`, and `unavailable`. The new source health model must not replace or rename this search-response metadata.

The gap is not the absence of evidence primitives. The gap is that readiness and trust are mostly reactive or scattered. Users still need a compact, pre-query answer to: what am I searching, is it indexed, what is unavailable, why was this answer grounded this way, and what can I do with the evidence now?

## Goals

- Show source health before the user runs a query.
- Preserve `/knowledge` as QA-only and avoid new CRUD/import management responsibilities.
- Make health metadata safe, read-only, and non-leaky.
- Make evidence reuse obvious through existing low-risk actions: copy citation, copy excerpt/text, open source, continue in workspace, and save to note only if an existing service contract is available.
- Explain answer trust in plain language: searched sources, results, citations, web fallback use, model/provider, and degraded source states.
- Improve recovery from empty, stale, unavailable, or low-evidence states.
- Keep WebUI and extension behavior aligned through shared components and parity tests.

## Non-Goals

- Do not add server-backed saved evidence or shared evidence sets.
- Do not add new team/profile sharing APIs.
- Do not make `/knowledge` the canonical place to create, import, edit, delete, or organize source records.
- Do not auto-enable web fallback.
- Do not expose generated, test, or workspace artifacts in normal source selection without explicit workspace scope.
- Do not redesign unrelated pages or perform repo-wide frontend cleanup.
- Do not store source excerpts, answers, query history, or citations in a new persistence model for this slice.

## Product Decisions

- `/knowledge` remains a QA workspace, not a knowledge CRUD hub.
- Source health is read-only. It may include counts and status, but not content excerpts or arbitrary source metadata.
- Evidence actions reuse existing capabilities. If a save-to-note API is already available, `/knowledge` may call it as a handoff action; otherwise the action should be hidden or disabled with clear copy.
- Web fallback remains user-toggleable and follows the configured server default provider.
- Health indicators must be useful in compact WebUI, mobile, and extension contexts.
- Handoff labels must name the destination. Use `Open Quick Ingest` or `Open source page`, not ambiguous `Add sources` copy that implies inline creation inside `/knowledge`.
- Implementation must map existing registry/source ids to this display taxonomy without renaming existing API values unless those ids are already part of the shared RAG contract.
- Keep pre-query source health and post-query source diagnostics as separate concepts. Pre-query health answers "is this source ready to search?" Post-query `metadata.source_status` answers "what happened during this search?"

## Approach

Use an inline health and evidence polish approach rather than a dedicated new drawer or answer-only report.

Why:

- It improves the two highest-value moments: source selection before search and evidence review after search.
- It fits existing component boundaries: `KnowledgeContextBar`, `SourceList`, `SourceCard`, `EvidenceRail`, `AnswerPanel`, and `SearchDetailsPanel`.
- It avoids a large new panel that users must discover.
- It keeps implementation PR-sized while leaving clean extension points for later server-backed evidence persistence.

## Source Health Contract

Add a safe source health model exposed through a read-only RAG source-health contract. Prefer a focused endpoint such as `GET /api/v1/rag/source-health` rather than overloading liveness/readiness endpoints like `/api/v1/rag/health`. If implementation finds an existing source-status endpoint by implementation time, it may reuse that endpoint only if the response remains read-only, stable for WebUI/extension clients, and clearly separate from per-search `metadata.source_status`.

Recommended shape:

```ts
type KnowledgeSourceHealth = {
  source_id: RagSource
  label: string
  available: boolean
  searchable: boolean
  item_count: number | null
  indexed_count: number | null
  last_updated: string | null
  last_indexed: string | null
  index_status: "ready" | "indexing" | "stale" | "empty" | "unavailable" | "error" | "unknown"
  embedding_status: "ready" | "indexing" | "missing" | "unavailable" | "not_applicable" | "error" | "unknown"
  disabled_reason: string | null
  workspace_scoped: boolean
  hidden_by_default: boolean
  privacy_note: string | null
}
```

Rules:

- Use canonical source ids: `media_db`, `notes`, `chats`, `characters`, `kanban`, `prompts`, `world_books`, `dictionaries`.
- Keep compatibility with existing frontend/backend aliases. If implementation code still receives aliases such as task boards, character cards, or worldbooks, normalize them at the existing source-contract boundary rather than introducing a second id system.
- Return all canonical sources even when a source is empty or unavailable, so the UI can explain absence.
- Counts may be `null` when a source cannot produce a cheap count safely.
- `indexed_count` means searchable/indexed items for that source, not necessarily total records.
- `last_updated` and `last_indexed` must be coarse source-level timestamps, not per-record private metadata.
- `privacy_note` is short copy for trust-sensitive sources, such as web fallback or workspace-scoped data.
- `embedding_status` reflects vector readiness only where embeddings are part of the searchable path. Use `not_applicable` for sources or configurations that are intentionally FTS-only or non-vector.
- Generated/test/workspace artifacts stay hidden by default unless explicit workspace scope is present.
- The contract must pass through normal auth, ownership, workspace, and visibility checks. Imported IDs or saved profiles are not access grants.
- The contract must not instantiate search retrievers or source databases just to compute health. Source availability should come from existing files or metadata that can be checked without creating directories, schema, indexes, vector stores, records, or request-scoped database handles.

V1 minimum:

- Required for every source: `source_id`, `label`, `available`, `searchable`, `index_status`, `embedding_status`, and `disabled_reason`.
- Optional in V1: `item_count`, `indexed_count`, `last_updated`, `last_indexed`, `workspace_scoped`, `hidden_by_default`, and `privacy_note`.
- Counts and timestamps may be `null` until each source can compute them cheaply and safely.
- `stale` should only be emitted when changed-since-index or source-specific freshness metadata already exists. Otherwise use `unknown` instead of guessing.

Fallback behavior:

- If health cannot be loaded, `/knowledge` remains usable and displays `Source health unavailable` with a retry action.
- Health load failure must not block query submission when source selection is otherwise valid.
- Partial health data should render per-source unknown states instead of hiding the source.

## Source Picker UI

Add compact health summaries to the existing source category and specific-source flows.

Placement:

- In `KnowledgeContextBar`, add a small health summary near the source selector: for example `6 ready, 1 stale, 1 unavailable`.
- In source category selection, each source row shows one compact status chip plus a count when available.
- In the specific media/note picker, keep the existing status filters but align labels with the new source health taxonomy.
- On mobile and extension-sized layouts, show a single summary line and put detailed status in the source picker popover.

Status labels:

- `Ready`: searchable now.
- `Indexing`: recently added or currently processing.
- `Stale`: content changed since last index or health is older than the freshness threshold.
- `Empty`: source exists but has no searchable items.
- `Unavailable`: source cannot be searched in the current server/account state.
- `Workspace only`: hidden globally because it belongs to an explicit workspace scope.
- `Unknown`: health could not be determined.

Suggested copy:

- Summary: `Sources ready: 6 of 8`
- Stale: `Stale index. Results may miss recent changes.`
- Empty: `No searchable items yet. Open Quick Ingest or the source owner page to add content.`
- Unavailable: `This source is not available in the current server configuration.`
- Workspace only: `Hidden until a workspace scope is selected.`
- Health failure: `Source health could not be loaded. You can still search selected sources.`

## Evidence Actions

Consolidate and clarify existing source/evidence actions rather than creating a new persistence model.

Implementation should audit current `SourceCard` actions before adding controls. Prefer relabeling, grouping, or exposing existing actions more clearly over adding duplicate copy/open buttons.

Actions:

- `Copy citation`: copy a formatted citation for the source or citation index.
- `Copy excerpt`: copy the retrieved excerpt/chunk text. If current UI uses `Copy text`, prefer `Copy excerpt` in evidence contexts.
- `Open source`: open the source viewer or owner surface when resolvable.
- `Continue in workspace`: reuse the existing workspace handoff for source-supported contexts.
- `Save to note`: optional only if an existing note handoff already preserves source backlinks without creating a new evidence persistence model. If that is not already true, defer it.

Interaction model:

- Keep primary evidence actions inside `SourceCard`; do not move them into a detached global toolbar.
- In `EvidenceRail`, add a compact action hint or per-card action row so users see evidence can be copied/opened without expanding every card.
- Do not add `Pin evidence` in this slice unless it is purely in-memory for the current page session and clearly labeled as temporary. Prefer omitting pinning to avoid implying persistence.

Accessibility:

- Every icon action must have an accessible name and visible text in compact menus or tooltips.
- Copy actions must announce success/failure through existing message/toast patterns.
- Keyboard users must reach action controls in citation/source order without losing their place.
- Focus returns to the invoking button after source viewer or menu close.

## Answer Trust Summary

Add a compact answer-level trust summary above or near the existing answer controls. This summary should reuse existing `searchDetails` where possible and add missing fields from the source health/search metadata only when needed.

Recommended fields:

- Source categories searched.
- Retrieved sources count.
- Citation count.
- Web fallback status: disabled, enabled not used, or used with provider/engine if available.
- Model/provider used for generation.
- Source health caveat count: stale, unavailable, unknown, empty.
- Verification/trust descriptor when available: strong, partial, weak.

Suggested copy:

- `Searched Documents & Media, Notes, and Chats. 12 sources returned, 5 cited.`
- `Web fallback disabled.`
- `Web fallback used via the configured server default.`
- `2 selected sources were stale or unavailable.`
- `Trust: Partial. Some claims are weakly supported.`

The detailed diagnostics remain in `SearchDetailsPanel`. The summary should be a compact trust strip, not a replacement for the details tab.

## Recovery States

Update recovery copy and actions to use source health when available.

No results:

- Show which selected sources were empty, stale, unavailable, or unknown.
- Offer `Broaden source scope`, `Enable web fallback`, `Show nearest matches`, `Open Quick Ingest`, or `Open source page` only where those actions already exist.
- `Open Quick Ingest` and `Open source page` are navigation handoffs only. `/knowledge` must not add inline import/create/edit controls.
- `Show nearest matches` is allowed only when nearest-match data already exists in current RAG/search response metadata or current UI behavior. Otherwise defer it.

Low-confidence answer:

- Keep `LowQualityRecoveryBanner`, but include source-health context when relevant.
- Suggested copy: `This answer has limited evidence. Try expanding sources, checking source status, or enabling web fallback.`

Health unavailable:

- Do not block search.
- Suggested copy: `Source health is unavailable. Search can continue, but readiness details may be incomplete.`

Recently ingested:

- Keep the current indexing hint. If source health says `Indexing`, use that instead of relying only on local quick-ingest state.

## Data Flow

1. `/knowledge` loads canonical source metadata from the frontend registry as it does today.
2. The UI requests read-only source health after server readiness succeeds.
3. The source health response is normalized into shared UI types distinct from existing `KnowledgeSourceStatus`.
4. `KnowledgeContextBar` uses health to show category status and specific source filter labels.
5. Search requests continue to use existing RAG settings and selected source ids.
6. Search responses continue to populate `answer`, `results`, `citations`, and `searchDetails`.
7. Answer trust summary combines selected source settings, source health caveats, and search response details, including post-query `metadata.source_status` when present.
8. Evidence actions operate on existing source card/result metadata and existing handoff services only.

## Error Handling

- Health endpoint unavailable: show non-blocking unknown state and retry.
- Source count expensive or unsupported: render count as unavailable, not zero.
- Source unavailable: keep visible if canonical, but explain why it cannot be searched.
- Copy failure: show `Unable to copy citation` or `Unable to copy excerpt`.
- Open-source failure: show `Unable to open this source from Knowledge QA`.
- Save-to-note unavailable: omit the action rather than presenting a broken control.
- Save-to-note failure: show the backend error when safe, otherwise `Unable to save this evidence to notes`.
- Quick Ingest or owner-page navigation failure: show `Unable to open the source owner page from Knowledge QA`.

## Privacy And Trust

- Do not expose source content, arbitrary metadata, provider keys, user identifiers, database paths, or workspace-private item titles in source health.
- Health timestamps are coarse enough to explain freshness without revealing hidden item details.
- Web fallback copy must state that web fallback uses the configured server default and is off unless enabled.
- Export/import profile behavior remains browser-local and unchanged by this slice.
- Existing auth and visibility checks remain authoritative for all source use and handoffs.

## Testing Strategy

Backend:

- Contract test for source health response shape and canonical source ids.
- Contract test that source health does not change the existing search-response `metadata.source_status` semantics.
- Test empty, unavailable, stale, and workspace-scoped source states.
- Test that hidden/generated/workspace artifacts do not affect global counts unless workspace scope is active.
- Test that health response excludes arbitrary safe metadata and content excerpts.

Frontend unit/Vitest:

- `KnowledgeContextBar` renders health summary and per-source status chips.
- Specific-source picker keeps existing filters and labels align with health taxonomy.
- Health load failure does not disable search.
- Pre-query source health and post-query `sourceStatus` diagnostics render without type or label collisions.
- `AnswerPanel` renders trust summary from selected sources, citations, web fallback, model/provider, and health caveats.
- `SourceCard` evidence actions expose accessible names, success/failure feedback, and do not offer unavailable handoffs.
- `EvidenceRail` keeps source/details tabs reachable and action hints visible in desktop/mobile contexts.
- `NoResultsRecovery` uses health diagnostics for empty/stale/unavailable/unknown states.

Browser/Playwright:

- WebUI `/knowledge` desktop: health visible before search, answer trust summary visible after search.
- Mobile `/knowledge`: source health details reachable without overlapping the composer or evidence rail.
- Extension options `#/knowledge`: shared health/evidence UI renders with the same labels.
- No-results flow: selected stale/unavailable source diagnostics appear and recovery actions are reachable.

Verification:

- Focused backend pytest for touched RAG/source health code.
- Focused KnowledgeQA Vitest tests for touched components.
- Extension parity/static route tests if shared component imports change.
- `git diff --check`.
- Bandit for touched backend code if a backend contract is implemented.

## Staged Implementation Plan

### Stage 1: Source Health Contract

Goal: expose safe read-only source health for all canonical `/knowledge` sources.

Success criteria:

- All canonical source ids return a health entry.
- Endpoint placement prefers `GET /api/v1/rag/source-health` or an equivalent focused read-only source-health endpoint, not a liveness endpoint.
- Empty/unavailable/stale/workspace-scoped states are representable.
- Response excludes content and arbitrary metadata.
- Health failure is non-fatal for existing search.
- V1 requires only availability, searchability, index status, embedding status, and disabled reason; counts/timestamps may be null.
- Existing post-query `metadata.source_status` remains backward compatible.

### Stage 2: Source Picker Health UI

Goal: make source readiness visible before users ask a question.

Success criteria:

- `KnowledgeContextBar` shows an aggregate health summary.
- Source category rows show status and counts where available.
- Specific-source picker filters remain intact and taxonomy is consistent.
- Mobile and extension layouts remain usable.

### Stage 3: Evidence Action Clarification

Goal: make existing evidence reuse controls obvious and reliable.

Success criteria:

- Source cards expose clear copy/open/workspace actions where supported.
- Save-to-note appears only when an existing note handoff already preserves backlinks without adding new persistence.
- Unsupported actions are omitted or disabled with clear copy.
- Copy success/failure is announced.
- Keyboard and screen-reader access are covered.

### Stage 4: Answer Trust Summary

Goal: explain what powered the answer without replacing the detailed diagnostics panel.

Success criteria:

- Answer summary shows searched sources, result count, citation count, web fallback state, model/provider, and health caveats.
- Trust descriptor reuses existing verification/faithfulness details where available.
- Summary remains compact and does not crowd the answer body.

### Stage 5: Recovery Copy And Diagnostics

Goal: make empty, stale, unavailable, and low-evidence states actionable.

Success criteria:

- No-results and low-confidence recovery include source health when available.
- Recovery actions are concrete and scoped: broaden sources, enable web fallback, show nearest matches only if existing nearest-match data is present, open Quick Ingest, open the source owner page, or continue in workspace.
- Quick Ingest and owner-page actions are route handoffs only, never inline source creation inside `/knowledge`.
- Health-unavailable state is visible but non-blocking.

### Stage 6: Verification And PR Packaging

Goal: ship a focused, reviewable PR without unrelated frontend cleanup.

Success criteria:

- Backend, frontend, and extension-relevant tests pass for touched scope.
- Bandit is run for touched backend code.
- Known unrelated verifier blockers are documented, not hidden.
- Backlog task and PR notes record the QA-only boundary and excluded persistence features.

## Open Questions

- If implementation finds an existing source-status endpoint by implementation time, does it satisfy the read-only, pre-query, WebUI/extension-stable contract well enough to reuse instead of adding `GET /api/v1/rag/source-health`?
- Which source types can cheaply provide `indexed_count` today, and which should initially return `null`?
- Does the existing `/api/v1/chat/knowledge/save` contract preserve source backlinks for Knowledge QA citation types? If not, defer `Save to note` from this slice.
- What freshness threshold should mark a source as `Stale`: fixed time window, changed-since-index metadata, or source-specific logic?

## Deferred Follow-Ups

- Server-backed saved evidence sets.
- Shared source/evidence profiles.
- Cross-device saved views.
- Advanced evidence pinboards or research packets.
- Source reindex controls from `/knowledge`.
- Full source CRUD/import management from `/knowledge`.
