# Explicit Research web capture and refresh implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development to execute the six tasks below sequentially, with a fresh implementer, a task review and a final whole-branch review.

**Goal:** Complete TASK-13530.1: explicitly preview, accept and refresh public article snapshots in the shared Research UI while retaining original evidence.

**Architecture:** Use the existing individual extraction endpoint with a credential-free opt-in profile, then existing WebClipper persistence after acceptance. Each accepted refresh gets a fresh clip UUID, Note, Media identity and Workspace source. Resolve and pin the actual Media version after readback; existing RAG remains current-head retrieval.

**Tech Stack:** FastAPI/Pydantic, existing scraper policy/egress hooks, MediaDatabase and ChaChaNotes, shared React/TypeScript, Zustand, Ant Design, Web Crypto, pytest/Vitest and Playwright over CDP.

**Spec:** `Docs/Design/2026-10-07-knowledge-web-capture-refresh.md`, approved by requester on 2026-10-07.

**Baseline:** `origin/dev` at `2c5f19d0328360d45bce99b8bae75ce252cc312f`, isolated branch `codex/knowledge-capture-refresh-20261007`.

**ADR check:** ADR required: yes; `Docs/ADR/066-explicit-web-capture-and-refresh-snapshots.md` governs the approved durable fetch/accept/security/version behavior and remains Proposed until implementation evidence is recorded. Governing ADR007/018/026/031/034/036/042/065 remain unchanged.

## Global Constraints

- Explicit public HTTP(S) credential-free readable-text capture only; no automatic fetch on import, selection, preview, reopen, or Ask.
- No new endpoint, crawler, persistence subsystem, dependency, Sync domain, authenticated-browser bridge, or background refresh.
- Strip all site-specific extra headers and all caller/route/browser cookies before policy, probes, acquisition or fallback; preserve robots, public-address SSRF, redirects, transport admission and quotas. Fail closed if enforcement is unavailable.
- `credential_free: bool = False` is additive; profile requires one URL, `individual`, no URL userinfo/cookies, analysis/translation/LLM extraction/chunking disabled; reject conflicting options.
- Accepted trimmed UTF-8 text is never silently shortened; WebClipper body limit is 1,000,000 characters, capture metadata JSON limit is 65,536 characters.
- Reserved `capture_metadata.web_capture_v1` has `mode: server_article`, `requested_url`, UTC `captured_at`, exact `content_sha256`, and nullable previous `refresh_of` clip UUID. Server validates and recomputes digest from promoted `full_extract`.
- Freeze owner/origin/workspace/UUID/body before save. Same acceptance retries same identity/body; refresh is a new identity. Canceling preview saves nothing. Partial saves remain recoverable, never compensated by deletion.
- Success requires canonical workspace source plus exact active Media version readback matching clip, descriptor and text digest. Note revision is never Media version. No historical pin falls back to latest or mixes current chunks.
- Original retrieved references, answers and provenance remain intact. Strict Notes provenance v1 is unchanged; capture references use its existing source fields only.
- Current RAG uses Media IDs/current chunks: recheck captured heads before Ask, surface changed heads and exclude them until recaptured. No server-atomic historical-version RAG guarantee.
- Both clients use the shared UI. Render extracted text as text, use existing accessible dialogs, explain the additional capture Note and exact local snapshot version.
- All browser operations use CDP as explicitly requested. Never fake native context-menu dispatch or claim CDP checks as spoken/device/human validation.
- Use existing TASK-13530.1 before file edits; task records only via backlog-py. Preserve primary checkout/dependencies, unrelated worktrees, recovery objects and service9099.
- Activate primary `.venv-uat-fixall-py312-20261007` before Python commands. Run meaningful focused tests, scoped formatting/lint, touched-scope Bandit and manual pre-commit hooks before commits; never bypass hooks.

---

## Stage 1: Credential-free preview
**Goal:** Governed, nonpersisting and unambiguous public extraction.
**Success Criteria:** Conflicting options rejected; credentials absent on every path; denied/empty/oversized extraction cannot appear successful.
**Tests:** Scraper orchestration/model/browser/outbound-policy and Media endpoint tests.
**Status:** Complete

### Task 1: Credential-free extraction profile

**Files:**
- Modify `tldw_Server_API/app/api/v1/schemas/media_request_models.py` (`IngestWebContentRequest`).
- Modify `tldw_Server_API/app/api/v1/endpoints/media/ingest_web_content.py` and `app/services/web_scraping_service.py`.
- Modify `app/core/Web_Scraping/orchestration/article.py` and `article_models.py`; existing policy/preflight/browser/Security seams only where strict public enforcement needs them.
- Test existing `tests/Media/test_ingest_web_content_endpoint_sanitization.py`, `tests/Web_Scraping/test_phase4_article_orchestration.py`, `test_phase4_article_models.py`, `test_phase4_article_browser.py` and relevant outbound-policy tests.

**Interfaces:**
- Existing `scrape_article(url, custom_cookies=None, *, allow_llm_extraction=True)` adds keyword-only `credential_free: bool = False`.
- Existing `IngestWebContentRequest` adds `credential_free: bool = False`; HTTP route unchanged.
- Public preview success preserves `status/results` envelope; one successful result contains nonempty trimmed `content`, `title`, `url`, `extraction_successful: true`, UTC `ingested_at`. Failure preserves a bounded stable `error` and `extraction_successful: false`, with envelope `status` not success. Ordinary callers remain compatible.

- [x] Read all scraper callers and guard/preflight/browser paths, endpoint RBAC/media-create patterns and owning tests. Run unchanged scoped tests and record baseline qualification.
- [x] Add failing tests for profile validation, configured arbitrary headers/cookies removal, canonical negotiation headers, probes/redirect/private-target/browser fallback denial, no analysis/monitoring, and bounded failure/size results.

```python
request = IngestWebContentRequest(
    urls=["https://example.org/article"], credential_free=True,
    perform_analysis=False, perform_chunking=False,
)
# New profile must propagate to the scraper and preserve its safe denial result.
```

- [x] Run focused tests RED; record command and failure proving the new behavior is absent.
- [x] Implement the profile using existing immutable ArticlePlan and plan_modifier before admission; regenerate canonical browser headers and clear cookies/browser custom_cookies. Thread strict request-scoped public guards into all concrete targets; reuse central egress, never a parallel network stack. Keep normal routes and injected test seams compatible.
- [x] Add shared Media-create RBAC/rate-limit/expected-owner dependencies; make unused token header optional/deprecated. Skip preview monitoring but preserve usage/governance. Validate profile options, disable analysis/extraction and bound empty/oversized/failure results; UTC timestamps.
- [x] Run focused then owning suites GREEN; format/lint/Bandit touched code, manual pre-commit changed files, self-review and commit with TASK-13530.1. Write report with RED/GREEN commands, counts, qualifications and exact changed files.

## Stage 2: Canonical accepted snapshot contracts
**Goal:** Save verified capture descriptors and preview exact historical versions.
**Success Criteria:** Mismatched capture text rejected; old active version remains previewable without current chunks; unavailable pin cannot resolve latest.
**Tests:** WebClipper service/API/Sync/tenancy, Workspace preview/core/API and Media versions.
**Status:** Complete

### Task 2: Descriptor validation and pinned source preview

**Files:**
- Modify `app/core/WebClipper/schemas.py`, `service.py`, `app/core/Workspaces/source_preview.py`, `app/api/v1/schemas/workspace_schemas.py`, `app/api/v1/endpoints/workspaces.py` (under `tldw_Server_API`).
- Test existing Notes_NEW WebClipper unit/integration tests, ChaChaNotesDB WebClipper/official PostgreSQL tenancy tests, Workspaces preview/core/API tests and Media version reads.
- Owning compatibility expansion: existing `app/core/DB_Management/ChaChaNotes_DB.py` native boolean binding and specific primary-key conflict handling for canonical source promotion/selection/retry, proven by official PostgreSQL RED. Verify SQLite/PostgreSQL insert, duplicate first-source retention, single/batch selection, optimistic versions and foreign ownership. Preserve foreign-key/deletion semantics; normalize backend failures in existing partial-save handling where needed. No schema change or new abstraction.

**Interfaces:**
- Existing WebClipper request shape unchanged. Reserved descriptor validation only when `capture_metadata.web_capture_v1` present; `mode`, bounded public URL, UTC timestamp, lowercase SHA256 and nullable UUID refresh parent only. Recompute hash from the exact trimmed `full_extract` used for promotion, reject disagreement before persistence.
- Existing `build_workspace_source_preview(..., focus_chunk_index=None)` adds `version_number: int | None = None`.
- Existing GET `/workspaces/{workspace_id}/sources/{source_id}/preview` adds optional positive query `version_number`, response optional `document_version_number`. Pin resolves exact active version only after current membership/ownership checks. Missing/deleted pins unavailable/404, no latest fallback. Pinned snippets exclude current chunks.

```python
preview = build_workspace_source_preview(
    workspace_id="owned", source=source, source_status=status, media_db=db,
    max_chars=3000, chunk_limit=3, version_number=1,
)
assert preview["document_version_number"] == 1
assert all(item["kind"] != "chunk" for item in preview["snippets"])
```

- [x] Read WebClipper promotion/receipts and existing exact version DB abstractions/callers; identify exact saved metadata path and version response shape.
- [x] Add and run RED tests for malformed/credential URLs, timestamp/hash/descriptor bounds, full text beyond Note budget, mismatch-before-save, exact retry without duplicates, refreshed identity preserving old sources, partial promotion, removed workspace and foreign owner.
- [x] Add and run RED preview tests for newer head with old pin, deleted/missing pin, suppression of latest chunks and unauthorized membership.
- [x] Implement reserved validation via existing Pydantic models/validators and service digest guard. Reuse Media DB version abstraction; preserve unpinned behavior and shared Workspace context projection compatibility.
- [x] Run owning suites GREEN; use official PostgreSQL fixture availability (never homemade DB setup); scoped lint/Bandit/hooks/self-review and commit. Report contracts and evidence for Task3.

## Stage 3: Shared scoped capture orchestration
**Goal:** Small shared client and acceptance helpers retain immutable identity and resolve real version pins.
**Success Criteria:** Same pending body retries; exact version/digest readback gates confirmation; requests stay owner-bound and abortable.
**Tests:** Shared client and narrow capture helper tests.
**Status:** Complete

### Task 3: Scoped clients and capture acceptance helpers

**Files:**
- Modify shared `services/tldw/TldwMedia.ts`, `services/tldw/domains/media.ts`, `domains/web-clipper.ts`, `domains/workspace-api.ts`, `services/web-clipper/types.ts`, `types/workspace.ts` under `apps/packages/ui/src`.
- Create `utils/research-web-capture.ts` and `utils/__tests__/research-web-capture.test.ts`.
- Extend existing scoped client tests. Regenerate checked OpenAPI/types with repository scripts after Tasks1/2 API changes; no hand-edited generated types.

**Interfaces (new functions defined by this task):**
- `tldwMedia.extractPublicArticle(url: string, options?: ScopedRequestOptions): Promise<PublicArticleExtractionResponse>` calls `/media/ingest-web-content` with correct one-URL individual credential-free disabled-analysis/chunking body; never change ordinary `processUrl` callers silently.
- `getWebClipStatus(clipId, options?: ScopedRequestOptions)`, `getWorkspaceSources(workspaceId, options?: ScopedRequestOptions)`, `getWorkspaceSourcePreview(workspaceId, sourceId, params?, options?: ScopedRequestOptions)` extend existing methods; preview params adds `version_number`.
- Existing Media domain adds typed scoped `listMediaDocumentVersions(mediaId, options?)` and `getMediaDocumentVersion(mediaId, versionNumber, options?)` using exact current version API shapes; shared handwritten DTOs follow the existing domain pattern and are verified against Task2's report and regenerated canonical schema, without importing ignored WebUI-only generated files.
- New `WebArticleCapturePin` in `types/workspace.ts`: `clipId`, `requestedUrl`, `capturedAt`, `contentSha256`, `refreshOf: string | null`, `mediaId`, `versionNumber`, `versionUuid`; optional `WorkspaceSource.webCapture` retains it.
- `prepareWebCaptureAcceptance(input: {url: string; title: string; text: string; capturedAt: string; workspaceId: string; refreshOf?: string | null}): Promise<WebClipperSaveRequest>` produces frozen trimmed body with fresh UUID, validated URL/time/hash, destination workspace, needs_review and enhancements off. No owner credential stored in body.
- `confirmWebCaptureAcceptance(body: WebClipperSaveRequest, options: ScopedRequestOptions, assertCurrent: () => void): Promise<{source: WorkspaceSourceApiResponse; pin: WebArticleCapturePin}>` verifies exact source ID `web-clipper:<UUID>`, URL/workspace/media, active version descriptor/hash/full text and version UUID. Throws on partial/unconfirmed state; never guesses version1 or uses Note revision.
- `assertWebCaptureHeadCurrent(source: WorkspaceSource, workspaceId: string, options: ScopedRequestOptions): Promise<void>` checks membership and latest owned active version matches stored pin before scoped Ask. No historical-version RAG promise.

```typescript
const body = await prepareWebCaptureAcceptance({
  url, title, text, capturedAt, workspaceId, refreshOf: prior?.clipId ?? null
})
await tldwClient.saveWebClip(body, options)
const confirmed = await confirmWebCaptureAcceptance(body, options, assertCurrent)
// Persist confirmed.pin and the immutable body in the existing owner-bound recovery path.
```

- [x] Read actual background proxy scope and Media version API; use exact schema types and standard Web Crypto. Do not add an abstraction layer or retry framework.
- [x] Add RED behavioral tests for no extraction until called, abort/scope forwarding, trimmed/full/oversized extraction, immutable acceptance/UUID, descriptor/hash match, Note-vs-Media versions, partial/lost-response exact retry, wrong workspace/URL/owner/media, deleted pin and changed head.
- [x] Implement and run GREEN; regenerate OpenAPI checked artifacts using repo scripts and verify drift; shared scoped lint/typechecks/hooks/self-review and commit. Report exact exported signatures/types for Task4.

## Stage 4: Shared Research workflow and evidence retention
**Goal:** Users can preview/save/refresh/retry captures and Ask only on a current owned snapshot.
**Success Criteria:** Explicit UI actions, honest extraction/version labels, no accidental save, retained original evidence and safe retirement/recovery.
**Tests:** SourcesPane and capture workflow, prefill/provenance/import/export/restore tests.
**Status:** In Progress

### Task 4: Integrate capture and refresh in Research

**Files:**
- Modify shared `components/Option/ResearchWorkspace/SourcesPane/index.tsx`, `ResearchWorkspace/index.tsx`, `ResearchWorkspace/ChatPane/index.tsx` (the actual scoped Ask submit/full-source path) and owning tests.
- Create focused `SourcesPane/WebArticleCaptureModal.tsx` and `utils/use-research-web-capture.ts` only to keep the large existing components manageable.
- Modify existing Workspace store/checkpoints, `workspace-server-restore.ts`, research prefill/import/export/provenance utilities only as required for capture pins and immutable pending body recovery. Preserve strict Notes v1.
- Update English shared locale source strings and regenerate existing locale artifacts normally.

**Interfaces:**
- Consumes Task3 clients/helpers and optional `WorkspaceSource.webCapture`.
- Existing source selection/store APIs remain authoritative. Capture UI retirement observes owner/origin/workspace/source membership and aborts pending reads; pending accepted body stays in the existing owner-bound Workspace storage, retried only under original scope.
- Modal uses existing Ant Design Modal, explicit `Capture article`, `Save capture`, `Refresh capture`, `Retry capture`, `Cancel` and a text expand action. No remote fetch in effects on mount/import/selection/reopen/Ask.
- Saved labels are `Extracted article snapshot`, capture time and `Source snapshot: Media version N`; prior web results `Retrieved excerpt`. Refresh creates a new source; unchanged digest shows `Text unchanged`. Changed current head shows `Snapshot changed outside refresh` and cannot participate in Ask.

```typescript
await assertWebCaptureHeadCurrent(source, workspaceId, options)
assertCurrent()
// Continue the existing scoped Ask only after this owned-head check succeeds.
```

- [ ] Read all shared Ask entry points and exact owner/workspace/checkpoint/restore/export paths; preserve each existing reference and manual selection.
- [ ] Repair the pre-existing SourcesPane.stage2 test fixture using the complete current source-list-view defaults: baseline has 33 passed and one TypeError from omitted lifecycleStateFilters at source-list-view.ts:157. Preserve assertions and production filter contract.
- [ ] Write RED tests for explicit capture network boundary, preview cancel, full accepted text/extra Note disclosure, save-confirm sequencing, changed/unchanged refresh identity preservation, frozen partial retry, rapid duplicates, manual selection during readback, source deletion, origin/account/workspace switch, component retirement, reopen/export pin retention, stale-head Ask exclusion and original evidence coexistence.
- [ ] Implement focused accessible modal/hook and shared SourcesPane actions with Task3 helpers. Freeze pending acceptance before mutation and retain recoverable readback failures. Confirm before adding/selecting; do not overwrite intervening manual selection. Persist under original owner before retired response is discarded.
- [ ] Wire exact version preview and source provenance using existing Notes v1 fields when a later sourced Note explicitly references capture; keep original references alongside and do not restore removed provenance implicitly.
- [ ] Run focused then owning suites GREEN, both client typechecks, scoped lint/locales/hooks/self-review and commit. Report every Ask call site covered and any qualified historical-RAG limitations.

### Task 5: Clarify Notes editing-state wording and qualify panel analysis

**Associated task:** TASK-13512 (already In Progress).

**Files:** Shared `components/Notes/hooks/useNotesEditorState.tsx`, English Notes locale strings and owning AI-assist/backlink/source-history tests. Existing extension chat integration tests and report only unless a reproducible root cause requires a minimal shared fix.

**Interfaces:** No new API, capture metadata field, Notes provenance wire change or inferred capture tag. The editor's `editProvenance` describes editing mode/last AI assist; an unknown origin cannot be called “Typed manually.” Authoritative Knowledge history/chat backlink labels remain governed by their existing contracts.

```typescript
// No authoritative source history: report actual editor state, not inferred authorship.
t('option:notesSearch.editingManual', { defaultValue: 'Editing: Manual' })
// An actual recorded AI-assist event can name its action/time without changing source origin.
t('option:notesSearch.latestAssistPrefix', { defaultValue: 'Latest AI assist' })
```

- [ ] Record this refinement in TASK13512 with backlog-py before code edits. Read actual editor state/history/backlink branches and owning tests.
- [ ] Add/run RED tests that a reopened captured/ordinary unknown-origin Note does not claim manual authorship, while recorded assist and authoritative Knowledge/chat source history retain correct independent meaning. Use existing current fixtures and assertions.
- [ ] Implement the minimal truthful wording above rather than adding an origin lookup/store based on editable tags. Run Notes AI-assist, backlink and source-history suites GREEN; scoped lint/types/hooks/self-review, commit with TASK13512 and report exact evidence.
- [ ] In the integrated CDP run, investigate the previously qualified direct-panel Stream completion failed using actual request status/cause and current built artifact. If reproducible, trace all callers, write RED regression and implement a minimal shared root fix only within existing chat contracts, then verify/review. If native launch evidence cannot be obtained with CDP, document the qualification; never fake onClicked or claim a renderer handoff proves native launch.

## Stage 5: Integrated verification and accurate tracking
**Goal:** Reviewable feature evidence and honest remaining followups.
**Success Criteria:** Owning test/type/build/security gates pass; CDP confirms both clients; tasks and ADR accurately describe implemented behavior and outstanding human/device checks.
**Tests:** Integrated backend/sharedUI tests, builds, OpenAPI drift, lint, Bandit, pre-commit, CDP workflows.
**Status:** Not Started

### Task 6: Verify and reconcile workstream

**Files:**
- Create `Docs/Reviews/KNOWLEDGE_CAPTURE_REFRESH_2026_10_07.md` and sanitized owning artifacts only.
- Update approved design status, ADR066/index/Published copy consistently only after implementation evidence; no accepted rationale rewritten.
- Update this plan stages and related Backlog records through backlog-py: TASK13530.1, TASK13530, TASK13530.2, TASK13514.1, TASK13512, and verify TASK13514 remains Done on dev.

**Interfaces:** All Tasks1–4 shared API and UI contracts; no new product behavior belongs in this verification task. Route defects to the owning implementer with scoped tests/review.

- [ ] Run once against final reviewed tree: owning pytest/Vitest suites, both client types/builds, checked OpenAPI drift, lint/format, Bandit touched scope and manual pre-commit hooks. Record commands/counts/skips and actual dependency setup qualifications.
- [ ] Launch isolated backend/WebUI/test Chromium with dedicated ports/profile and synthetic owner/data only. Drive via `chromium.connectOverCDP`; capture actual preview→cancel→save→readback→pinned preview→scoped Ask; refresh changed and unchanged content; retain original excerpt/old capture; retry partial promotion/retirement, reopen/export. Verify shared extension UI in built Chrome extension too. Sanitize receipts; never commit keys/raw profiles.
- [ ] Reconcile prior PR3211 hosted evidence: seven required checks, optional Watchlists success and PG tenancy/schema evidence; classify queued/infra-failed broader CI separately. Native menu/spoken/Safari/iOS/real participant checks remain unverified until actual evidence exists.
- [ ] Update Backlog completion only for verified criteria; public capture parent closes only when acceptance criteria satisfied. ADR066 can become Accepted after approved implemented policy verified; update canonical and Published entries together. Final summary includes current-version RAG boundary.
- [ ] Self-review, scoped hooks and commit docs/tracking. Package branch for independent whole-branch review, fix findings via owning agent and re-review. Create PR against dev within existing authorization, attach it, present final reviewed implementation for a new requester-written Change summary before merge.
