# Explicit Research web capture and refresh implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development to execute the six tasks below sequentially, with a fresh implementer, a task review and a final whole-branch review.

**Goal:** Complete TASK-13530.1: explicitly preview, accept and refresh public article snapshots in the shared Research UI while retaining original evidence.

**Architecture:** Use the existing individual extraction endpoint with a credential-free opt-in profile, then existing WebClipper persistence after acceptance. Each accepted refresh gets a fresh clip UUID, Note, Media identity and Workspace source. Resolve and pin the actual Media version after readback; existing RAG remains current-head retrieval.

**Tech Stack:** FastAPI/Pydantic, existing scraper policy/egress hooks, MediaDatabase and ChaChaNotes, shared React/TypeScript, Zustand, Ant Design, Web Crypto, pytest/Vitest and Playwright over CDP.

**Spec:** `Docs/Design/2026-10-07-knowledge-web-capture-refresh.md`, approved by requester on 2026-10-07.

**Baseline:** `origin/dev` at `2c5f19d0328360d45bce99b8bae75ce252cc312f`, isolated branch `codex/knowledge-capture-refresh-20261007`.

**ADR check:** ADR required: yes; `Docs/ADR/066-explicit-web-capture-and-refresh-snapshots.md` governs the approved durable fetch/accept/security/version behavior and is Accepted after approved implementation and local verification. Governing ADR007/018/026/031/034/036/042/065 remain unchanged.

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

- [x] Task3 fix1/fix2: admit only required capture and canonical readback path/method pairs through existing shared scoped transport inventory; real direct/worker regressions and canonical raw-control rejection independently approved. Final271focused/619owning pass, both types/hooks and zero new lint. Actual rebuilt Capture/save/refresh chain remains Task6 after scoped maintenance.

## Stage 4: Shared Research workflow and evidence retention
**Goal:** Users can preview/save/refresh/retry captures and Ask only on a current owned snapshot.
**Success Criteria:** Explicit UI actions, honest extraction/version labels, no accidental save, retained original evidence and safe retirement/recovery.
**Tests:** SourcesPane and capture workflow, prefill/provenance/import/export/restore tests.
**Status:** Complete

### Task 4: Integrate capture and refresh in Research

**Files:**
- Modify shared `components/Option/ResearchWorkspace/SourcesPane/index.tsx`, `ResearchWorkspace/index.tsx`, `ResearchWorkspace/ChatPane/index.tsx` (the actual scoped Ask submit/full-source path) and owning tests.
- Follow the real dispatch boundary through existing `hooks/chat/useChatActions.ts` and `hooks/handlers/messageHandlers.ts` plus meaningful owning action tests: propagate capture retirement through asynchronous preparation and check immediately before actual request. Preserve ordinary callers; reuse existing abort/scope primitives, no new framework.
- Create focused `SourcesPane/WebArticleCaptureModal.tsx` and `utils/use-research-web-capture.ts` only to keep the large existing components manageable.
- Modify existing Workspace store/checkpoints, `workspace-server-restore.ts`, research prefill/import/export/provenance utilities only as required for capture pins and immutable pending body recovery. Preserve strict Notes v1.
- Update English shared locale source strings and regenerate existing locale artifacts normally.

**Interfaces:**
- Consumes Task3 clients/helpers and optional `WorkspaceSource.webCapture`.
- Explicit cited-message Save-to-Notes may use optional local WorkspaceNote.pendingKnowledgeProvenance and existing strict-v1 knowledgeNoteWriteFields replacement, preserving canonical optimistic head, original references and deleted/unsupported guards. No wire/schema/origin inference.
- Existing source selection/store APIs remain authoritative. Capture UI retirement observes owner/origin/workspace/source membership and aborts pending reads; pending accepted body uses existing research-workspace-prefill safe-storage, public owner keys and serialized checkpoints in an adjacent capture record under the same owner key family, retried only under original scope. No fabricated knowledge_qa_thread is created for capture-only recovery; ordinary handoffs cannot replace capture records. Global Zustand snapshots retain display pins, never accepted-body recovery authority.
- Modal uses existing Ant Design Modal, explicit `Capture article`, `Save capture`, `Refresh capture`, `Retry capture`, `Cancel` and a text expand action. No remote fetch in effects on mount/import/selection/reopen/Ask.
- Saved labels are `Extracted article snapshot`, capture time and `Source snapshot: Media version N`; prior web results `Retrieved excerpt`. Refresh creates a new source; unchanged digest shows `Text unchanged`. Changed current head shows `Snapshot changed outside refresh` and cannot participate in Ask.

```typescript
await assertWebCaptureHeadCurrent(source, workspaceId, options)
assertCurrent()
// Continue the existing scoped Ask only after this owned-head check succeeds.
```

- [x] Read all shared Ask entry points and exact owner/workspace/checkpoint/restore/export paths; preserve each existing reference and manual selection.
- [x] Repair the pre-existing SourcesPane.stage2 test fixture using the complete current source-list-view defaults: baseline has 33 passed and one TypeError from omitted lifecycleStateFilters at source-list-view.ts:157. Preserve assertions and production filter contract.
- [x] Write RED tests for explicit capture network boundary, preview cancel, full accepted text/extra Note disclosure, save-confirm sequencing, changed/unchanged refresh identity preservation, frozen partial retry, rapid duplicates, manual selection during readback, source deletion, origin/account/workspace switch, component retirement, reopen/export pin retention, stale-head Ask exclusion and original evidence coexistence.
- [x] Implement focused accessible modal/hook and shared SourcesPane actions with Task3 helpers. Freeze pending acceptance before mutation and retain recoverable readback failures. Confirm before adding/selecting; do not overwrite intervening manual selection. Persist under original owner before retired response is discarded.
- [x] Wire exact version preview and source provenance using existing Notes v1 fields when a later sourced Note explicitly references capture; keep original references alongside and do not restore removed provenance implicitly.
- [x] Run focused and owning suites, both client typechecks, scoped lint/locales/hooks/self-review and commit. Record baseline/timing failures without a broad-green claim; report every actual Ask call site covered, unsupported edited-send behavior, and qualified historical-RAG limitations. Focused fix1 tests116/116; owning broad1182/1193 with qualified failures; independent six-finding re-review passed.

- [x] Task4 fix2: repair live single-user public-owner/request-scope identity mismatch without changing recovery namespace or verified authority; cover real identity builders and owner/credential retirement, then independent scoped review and real CDP capture verification. Task6 found same-owner Capture account changed before HTTP at6210798ef5.

- [x] Task4 fix3: publish proven captured-head refusal before synchronous real-store deselection, retain the refusal through generic readiness and verified matching-pin canonical rehydrate from available display state; transient errors remain retryable and tombstoned state is not revived. Final88 affected tests/both types/scoped checks and independent three-finding re-review pass at9234163cf3; Task6 affected CDP proof is recorded in the canonical closeout, with matching migrated-browser restore explicitly unverified.

### Task 5: Clarify Notes editing-state wording and qualify panel analysis

**Associated task:** TASK-13512 (already In Progress).

**Files:** Shared `components/Notes/hooks/useNotesEditorState.tsx`, English Notes locale strings and owning AI-assist/backlink/source-history tests. Existing extension chat integration tests, sidepanel-chat handoff catch and English feedback for intentional selected-history rejection; model adapter only if an observed reproducible root cause requires a minimal shared fix.

**Interfaces:** No new API, capture metadata field, Notes provenance wire change or inferred capture tag. The editor's `editProvenance` describes editing mode/last AI assist; an unknown origin cannot be called “Typed manually.” Authoritative Knowledge history/chat backlink labels remain governed by their existing contracts.

```typescript
// No authoritative source history: report actual editor state, not inferred authorship.
t('option:notesSearch.editingManual', { defaultValue: 'Editing: Manual' })
// An actual recorded AI-assist event can name its action/time without changing source origin.
t('option:notesSearch.latestAssistPrefix', { defaultValue: 'Latest AI assist' })
```

- [x] Record this refinement in TASK13512 with backlog-py before code edits. Read actual editor state/history/backlink branches and owning tests.
- [x] Add/run RED tests that a reopened captured/ordinary unknown-origin Note does not claim manual authorship, while recorded assist and authoritative Knowledge/chat source history retain correct independent meaning. Use existing current fixtures and assertions.
- [x] Implement the minimal truthful wording above rather than adding an origin lookup/store based on editable tags. Run Notes AI-assist, backlink and source-history suites GREEN; scoped lint/types/hooks/self-review, commit with TASK13512 and report exact evidence.
- [x] In the integrated CDP run, investigate the previously qualified direct-panel Stream completion failed using actual request status/cause and current built artifact. If reproducible, trace all callers, write RED regression and implement a minimal shared root fix only within existing chat contracts, then verify/review. If native launch evidence cannot be obtained with CDP, document the qualification; never fake onClicked or claim a renderer handoff proves native launch.

- [x] Preserve ADR049 selected-history admission; for Clipper handoff rejection at the existing catch, show translated actionable New Chat guidance while retaining current tab and pending handoff. Add behavioral RED/GREEN for pending recovery, owner/retirement, and no duplicate dispatch. Diagnose the actual legacy analysis adapter separately from successful ordinary chat SSE; do not infer a Clipper pass from that different pipeline.

- [x] Repair the confirmed empty-prompt root in existing application.getPrompt for WEB_CLIPPER_ANALYZE_MESSAGE_TYPE using existing DEFAULT_CUSTOM_PROMPT; retain unknown/custom behavior and transport validation. Bind the existing transient pending Analyze record to producer verified serverChatMirrorOwnerKey, normalize unowned requests fail closed, and fence current owner/origin/view before dispatch and post-await retirement/notification. Preserve saved clips and conditional replacement identity; add actual producer/normalizer/same-owner, cross-account/origin and retired same-ID replacement RED/GREEN, then verify rebuilt real transport/SSE. No new wire API, persistence subsystem or history authority.

Task5 verification: qualified native-resolution407/407; default383passed24failed+32errors remains qualified. Both types/Chrome build/manual hooks pass; zero new lint diagnostics (inherited1error71warnings). Actual built Clipper save200→singlecompletions200/SSEsuccess/DONE verified and pending retired; independent spec+quality review approved49701beaec. Native/spoken/device/participant evidence remains open.

### Task 7: Burn down inherited Research assertion failures

**Associated task:** TASK-13531 (separate reviewable maintenance unit).

**Files:** Owning workspace storage, saved-normal Chat, ChatPane.stage1 and SourceViewControls test files; existing shared production roots only if actual defects are proved.

**Interfaces:** Existing selected-source persistence/quota feedback, exact selected provider identity, workspace chat ownership/hydration and accessible saved-view keyboard/dialog lifecycle. No new feature, store, dependency, persistence authority or durable policy.

- [x] Diagnose the ten exact failures reproduced on canonical dev2c5f19d (233 cases,223 passed10 failed) and feature621. Distinguish obsolete fixtures/expectations from actual defects through current contracts and real behavior; retain exact baseline receipts.
- [x] Use behavioral RED/GREEN and the existing minimal primitives; preserve meaningful keyboard, focus, busy state, server confirmation, owner/retirement and stale-submit assertions. Correct obsolete structural expectations only with explicit contract evidence; never skip, weaken security checks, blanket mock UI or enlarge timeouts without a diagnosed cause.
- [x] Run focused and four owning suites, current client typechecks, scoped lint/format/hooks and applicable security/browser checks. Record qualified/default environment outcomes; independent task review before Task6 final reconciliation. No inaccurate TASK12116 assignment or broad-green claim.
- [x] Record final evidence and status via backlog-py; Task6 includes this scoped unit in final report and current reviewed validation.

Task7 fix2: shared saved-view dialog focus now restores from existing native close-completion callback with pending token/generation retirement. Final51 owning/default6 focused/both types/scopedlint/hooks pass; independent scoped review approved3f610227de. Original successful acknowledgments and outside/new-dialog/removed-invoker guards retained. Historical checkpoint: rebuilt both-client normal CSS acceptance was pending. Current Resume7 acceptance now passes; see the canonical closeout.

## Stage 5: Integrated verification and accurate tracking
**Goal:** Reviewable feature evidence and honest remaining followups.
**Success Criteria:** Owning test/type/build/security gates pass; CDP confirms both clients; tasks and ADR accurately describe implemented behavior and outstanding human/device checks.
**Tests:** Integrated backend/sharedUI tests, builds, OpenAPI drift, lint, Bandit, pre-commit, CDP workflows.
**Status:** In Progress

### TASK-13530.1: PR3213 media shard inventory completion
**Goal:** Keep the new current-media deletion regression in every existing media-core-api shard and its exact inventory contract.
**Success Criteria:** All five workflow matrices and the exact-once contract agree on the regression path.
**Tests:** `test_full_suite_splits_slow_chat_and_retrieval_shards`; scoped YAML/Python checks, Bandit and hooks.
**Status:** Complete locally; publication and updated-head CI remain controller gates.

The original hosted Ubuntu Python 3.12 shard failed because `test_capture_version_deletion_current_material.py` was absent from its inventory. Adding the path after `test_cache_index.py` in all five existing shard path sets and the exact contract made the targeted five-matrix test pass. See the canonical review and sanitized artifact for RED/GREEN receipts and hashes. No current-head hosted CI success is inferred; no ADR is required because the existing CI contract governs.

Evidence lineage qualification retained from the independent frozen fixture review: `/private/tmp/knowledge-pr3213-scoped-format.log` originally hashed to `1be9f70e9aa41e84f5ff2994cf1d8dce3fa9ddf39578becd6232780906b9c4e0` (107 bytes; comparison `False`); the same path later held a 191-byte explanatory note hashing to `3a8c723f90fb789aca7e6701ae9b71a9aae35c90bc1c280c9d7a98b5ee50eba6`. No fresh formatter pass is claimed. The independent fixture review is `/private/tmp/knowledge-pr3213-ci-fixtures-review.md`, SHA-256 `6d3ecbb0b9d4cc0fe721f91150ab45ba89bff45fcbeb8688df52e1f85fc6375f`; its separate implementation report is `/private/tmp/knowledge-pr3213-ci-repair-report.md`, current SHA-256 `38281f93608a6ec0534ab9f04288671dbe2d2a83982a12a0e6b4e1793b04b72f` (prior versions `b72ebdd14655d68a13482fb2587e2945af78e544a71028067c814a4abe5747c9` and `ff8f1df9d282b04109972f86d81a163b44c3bc34c5c2936a7b931311051fe797`).

Receipt lineage: current inventory test, lint, format, Bandit, hook and diff-check outputs are preserved under unique `/private/tmp/knowledge-pr3213-shard-inventory-*` paths with byte counts and hashes in the canonical JSON. Earlier generic Ruff/Black/Bandit/hook/diff-check paths now hold inventory output. The original hook SHA `3901a38bf379d0a01891fe4ccca12714bc91d1f047e338c098fa206a946464c3` was verified by the controller before its path was overwritten; preserve it as historical, not as the current file hash. The old diff-check fingerprint is unavailable here. No tests were rerun for this evidence correction; scoped documentation/task hooks were run and passed to validate the record changes.

### Task 6: Verify and reconcile workstream

**Files:**
- Create `Docs/Reviews/KNOWLEDGE_CAPTURE_REFRESH_2026_10_07.md` and sanitized owning artifacts only.
- Update approved design status, ADR066/index/Published copy consistently only after implementation evidence; no accepted rationale rewritten.
- Update this plan stages and related Backlog records through backlog-py: TASK13530.1, TASK13530, TASK13530.2, TASK13514.1, TASK13512, and verify TASK13514 remains Done on dev.

**Interfaces:** All Tasks1–4 shared API and UI contracts; no new product behavior belongs in this verification task. Route defects to the owning implementer with scoped tests/review.

- [x] Run once against final reviewed tree: owning pytest/Vitest suites, both client types/builds, checked OpenAPI drift, lint/format, Bandit touched scope and manual pre-commit hooks. Record commands/counts/skips and actual dependency setup qualifications.
- [x] Launch isolated backend/WebUI/test Chromium with dedicated ports/profile and synthetic owner/data only. Drive via `chromium.connectOverCDP`; capture actual preview→cancel→save→readback→pinned preview→scoped Ask; refresh changed and unchanged content; retain original excerpt/old capture; retry partial promotion/retirement, reopen/export. Verify shared extension UI in built Chrome extension too. Sanitize receipts; never commit keys/raw profiles.
- [x] Reconcile prior PR3211 hosted evidence: seven required checks, optional Watchlists success and PG tenancy/schema evidence; classify queued/infra-failed broader CI separately. Native menu/spoken/Safari/iOS/real participant checks remain unverified until actual evidence exists.
- [x] Update Backlog completion only for verified criteria; public capture parent closes only when acceptance criteria satisfied. ADR066 can become Accepted after approved implemented policy verified; update canonical and Published entries together. Final summary includes current-version RAG boundary.
- [x] Self-review, scoped hooks and normal docs/tracking commit; exact exits and changes recorded in Task6 report.
- [ ] Controller: independent Task6 and whole-branch review, owning repairs/re-review, PR against dev and attachment, new requester-written Change summary, seven updated-head required CI checks, merge and owned cleanup.

Task7 fix3 f12bb3e743: verified account-scoped saved-view reads/writes reuse the existing snapshot lease, account watcher and generation/retry lifecycle; typed scope loss retires, ordinary transient resolution can explicitly retry, rejected-request teardown cannot focus after unmount. Independent scoped review approves these three findings with383owning/default4/finalworker-root38/both types0 and inherited style qualifications. Historical checkpoint: rapid WebUI pointer Cancel was open at fix3. Independently reviewed fix4 stabilizes validation geometry; current Resume7 built WebUI and extension pass trusted pointer/native CSS closure at desktop, narrow and measured wrapped feedback layouts, rapid current Cancel/Close/Escape, account retirement and eight real acknowledgments. TASK13531 can close; Stage5 controller review/publication gates remain.

Task6 local verification at63a3abd7ff0e3ece0e82bb72f5eb7fdbb6532fec is DONE_WITH_CONCERNS; [canonical evidence](Docs/Reviews/KNOWLEDGE_CAPTURE_REFRESH_2026_10_07.md) distinguishes historical failed runs, exact unchanged scopes, real persistence/FTS/citation from the deterministic extraction-response fixture and exhausted three429 public probes. Stage4 complete; Stage5 remains In Progress until controller gates.


### TASK-13531: PR3213 head-specific frontend CI repair
**Goal:** Repair only the quota-warning assertion and corrected same-workspace saved-view assertion from head45925ab.
**Success Criteria:** Meaningful RED→GREEN evidence with unchanged quota event, native completion, generation/retirement/focus authority and current built acceptance.
**Tests:** CI deterministic sequencer/options, owning/default cases plus necessary storage/native-dialog neighbors; types, scoped lint/format, manual hooks and normal commit.
**Status:** Complete

- [x] Diagnose both actual head assertions against dev2c5f19d and CI logs.
- [x] Reproduce meaningful RED; implement minimal repair without timeouts/assertion waivers.
- [x] Verify owning scope and preserve all historical/broader/default/native qualifiers.
- [x] Update canonical evidence/task through backlog-py and pass scoped hooks; prepare the normal commit. Independent review remains controller-owned.

ADR required: no; restoring existing approved test contracts introduces no durable rule. ADR066/007/036/042 remain governing. No runtime, dependency, public probe, browser, push, PR, merge or cleanup actions.

Exact source: native Storage instance spy misses real write; Tooltip/dialog test-id collision mislabels real Replace dialog. First two native-close hypotheses failed and were reverted; third narrow real Tooltip ID fixture passes. Owning99/native59 pass; failed local neighbor receipts and prior broader/browser qualifiers remain explicit. No product roots changed. Independent review/current hosted CI remain controller-owned.
