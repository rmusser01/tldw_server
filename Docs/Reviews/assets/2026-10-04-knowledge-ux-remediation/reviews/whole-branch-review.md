# Whole-branch Knowledge UX review

Reviewed base `61589721bc9e05c2be619a076cf5d575adea3d56` through head `5d24666688b13792884f027dfe8c3cb058ab8bb4` in the isolated Knowledge checkout.

**Specification verdict: Issues found. Technical quality verdict: Needs fixes. Ready to merge: No.**

Open findings: **0 Critical, 3 Important, 1 Minor.** The three Important findings affect ordinary Research continuation and preservation of the authenticated request boundary. Scoped approvals and passing checks do not cover these combinations. This review does not approve a merge or satisfy the human-written PR Change summary gate.

## Scope and evidence

Reviewed the supplied whole-branch U10 package in bounded passes, the full original K01–K15/five-enhancement audit, the current amended design and implementation plan, chronological controller rulings, implementer reports and scoped review/fix evidence. Followed changed callers into the existing canonical Notes, request-scope, workspace restore/reconcile and media-storage contracts where needed. No broad diff was regenerated, passing checks were not rerun, no subagents were dispatched, and no product or Git state was changed. This report is the only review deliverable written.

The controller's final-verification record at this head reports 120 suites / 1654 tests passed, official WebUI and extension type checks passed, and the WebUI development build passed. The extension build and remaining live/native gates were still controller-owned when the record was read. The controller subsequently reported a failed stricter Research completion checkpoint; finding I1 independently explains that result from the source. No private runtime descriptor or raw browser failure log was read or reproduced here.

## Strengths

- Scope is treated as user intent: preset tuning preserves corpus/model settings; explicit arrivals wait for defaults hydration; invalid or empty transfers cannot become whole-library requests. Server search/pagination and retained selected titles address the original first-200 limitation without a new source store.
- The persistence work follows the actual API. Canonical note content carries bounded, validated provenance because NoteResponse drops nested metadata. Normal Notes and Quick Notes hide recognized markers while retaining human text, trust and original references. UUID and legacy numeric identity, lost acknowledgments, later edits and version headers receive substantive regression coverage.
- Research import uses the existing media-backed workspace and ingestion for labeled excerpt snapshots. Retry checkpoints, exact selection, folder clearing, current-owner checks, tombstones and matching-server-source restore are deliberately integrated with the established persistence owners.
- Ingestion retry uses actual retained Results/queue/session state and durable collection identity. Tests cover reattachment, pre-ack remount, start rejection, transport failure, subsequent retry lineage and retained successes, rather than reconstructing hypothetical successful arrays.
- Accessibility changes reuse AntD dialogs/drawer, add focused Tab-boundary handling, restore focus, expose active suggestions and reserve a composer slot for Evidence. The stateful shortcut regressions demonstrate that the old handler actually destroys rendered state; the mobile duplicate Evidence header was removed.
- No new package dependency, API family, modal framework or workspace owner is needed. Keeping specialized Media, Document Workspace and Research Studio destinations is consistent with the amended design and ADR boundaries.

## Issues

### Critical

None identified.

### Important

**I1 — Accept the normal empty keyword filter before persisting Research provenance.**

Primary location: `apps/packages/ui/src/utils/knowledge-note-provenance.ts:93–94`. Related: the fallback at `:223–228`, `apps/packages/ui/src/components/Option/KnowledgeQA/KnowledgeQAProvider.tsx:1893`, and canonical persistence in `apps/packages/ui/src/utils/use-research-workspace-prefill.ts:404–423`.

The scope validator rejects `keyword_filter: ""`: it is neither a nonempty shortString nor a string list. Empty string is the normal no-filter value, including the explicit Knowledge arrival state. AnswerPanel carries that scope into imported evidence, so otherwise valid media and note-snapshot evidence makes the entire new Research marker invalid. `retainKnowledgeNoteProvenance` then silently uses an older marker already in the body, or returns unmarked content. Canonical writes can succeed before readback detects that the current import ID is missing; retry cannot complete the same payload. With an existing marker, the saved body can carry stale evidence/import identity.

This violates K05 and the amended requirement that the current draft/evidence survive canonical save and restore. The controller's stricter live checkpoint observed PUT/GET success and increasing versions while the old Review import marker survived and the new import never became completed/draftRetained. Source inspection independently confirms the cause. Existing importer fixtures use a reduced scope, so they do not exercise the normal empty-filter payload.

Remedy: accept or consistently normalize the legitimate empty no-filter representation while retaining bounded validation. At the import persistence boundary, validate required new provenance before the network write; do not silently substitute old provenance when a newly required Research checkpoint is invalid. Add a mounted Provider/AnswerPanel-shaped payload regression with default empty keyword filter and null collection, covering fresh content and existing old provenance. Assert the current import ID, typed source evidence, completed checkpoint, canonical readback/restore and no duplicate snapshot upload.

**I2 — Preserve canonical IDs for stored web media in Review → Research.**

Primary location: `apps/packages/ui/src/utils/research-workspace-prefill.ts:133–145`. Caller: `apps/packages/ui/src/components/Review/MediaKnowledgeActions.tsx:73–78`. Failure boundary: `apps/packages/ui/src/utils/use-research-workspace-prefill.ts:196–200`.

`resolveMediaId` rejects every source type matching note/web/url before examining explicit `metadata.media_id`. Review supplies that canonical ID together with the media item's content subtype. Real stored articles have type `web_document`: the existing database producer passes `media_type="web_document"` to `db.add_media_with_keywords` in `tldw_Server_API/app/services/enhanced_web_scraping_service.py:1007–1012`. Thus an already stored, valid reviewed web article loses its media ID. Review sends no retrieved excerpt, so import then follows the non-media snapshot path and fails with “No retrieved excerpt to import.” Repeating Retry does not repair it.

This leaves K04/K05 and the shared source-set enhancement incomplete for a supported media category. The Review regression currently covers PDF/audio; the external-web regression correctly rejects arbitrary result IDs but misses stored web media.

Remedy: distinguish canonical media identity from its content subtype, either in the shared builder or by providing an explicit canonical-origin discriminator from Review. Preserve genuine canonical media IDs while continuing to reject an external web result's arbitrary numeric result ID. Test the actual Review caller with `web_document` through the real builder/importer: the same canonical item must attach directly with no snapshot upload. Retain the external-web numeric-ID snapshot test.

**I3 — Merge the expected-user fence into canonical importer PUT headers.**

Primary location: `apps/packages/ui/src/utils/use-research-workspace-prefill.ts:418–423`.

For updates, the object spread replaces all `request.headers` with `{ "expected-version": ... }`. The existing `requestScopeFields` contract adds `X-TLDW-Expected-User-ID` for non-null principals (`apps/packages/ui/src/services/tldw/domains/service-prompts.ts:318–336`), so the importer drops the server principal assertion on PUT while its GET/POST/readback retain it. The transport does not recreate that header. Its client configuration check deliberately excludes cookie-session/hosted principal matching (`apps/packages/ui/src/services/background-proxy.ts:493–510`); the server assertion is therefore material, not redundant. `require_expected_user` returns without checking identity when the header is absent (`tldw_Server_API/app/api/v1/API_Deps/auth_deps.py:1611–1625`).

This regresses the binding requirement to preserve auth/account/request-scope fencing. Existing abort/current-owner checks remain useful, but they do not replace the server check when cookie authentication changes before a write. This is a verified missing precondition, not a claim that a cross-account write was reproduced. The nearby Quick Notes save repair correctly merges scoped headers; the importer test lease uses a null user and misses this sibling path.

Remedy: merge `request.headers` when adding `expected-version`. Add a canonical GET → PUT → readback importer test with a non-null principal that requires both headers. Include a changed-authenticated-principal response and assert rejection before mutation/completion, while preserving current draft state.

### Minor

**M1 — Use the required translation fallback convention for new Review continuation controls.**

Location: `apps/packages/ui/src/components/Review/MediaKnowledgeActions.tsx:110–122` and `:94`.

The new shared Ask/Research labels and failure message are literal English strings, with no translation hook. This bypasses the binding design's existing i18n/defaultValue convention and leaves those controls English in a translated Review interface. Use the existing namespace and translation fallback pattern for these new labels/message (including the fallback source title where applicable), and verify the keys can resolve with English defaults retained. This is a narrow localization fix; it does not justify unrelated locale or formatting churn.

## Complete audit and enhancement assessment

“Supported” means the implementation and supplied evidence support the repair within the documented validation limits; it does not mean participant or screen-reader validation occurred.

| Audit item | Assessment |
|---|---|
| K01 — Presets preserve exact scope | Supported: scoped categories/IDs/filter/collection, Web choice and model survive tuning; explicit arrival hydration is gated. |
| K02 — Ingestion Ask retains newly added set | Supported: successful canonical IDs drive Ask added items; invalid arrivals fail closed; Results retains success/failure distinctions. |
| K03 — Large-library selection | Supported: server search and 50-item paging, retained selection/title cache, explicit result counts; controller found an older item beyond 200. |
| K04 — Review Ask/Research | Ask supported. Research incomplete for stored web media: I2. New controls also need M1. |
| K05 — Research grounding, trust and reopen | Substantial implementation, but not accepted: ordinary default scope breaks canonical completion (I1), stored web media fails (I2), and importer PUT loses a principal fence (I3). |
| K06 — Exact saved note and original origin | Supported: returned note link, validated content provenance/backlink, hidden editor marker, origin retained across title/body edits and reopen. |
| K07 — Scope/preview focus | Supported by established primitives, focus guard, behavioral tests and controller keyboard runs; no screen-reader certification claimed. |
| K08 — Unclipped/named exact picker | Supported: bounded dialog list, named search, reachable controls and paging; short/narrow viewport evidence recorded. |
| K09 — Empty-library readiness | Supported: actual personal totals, separate service/storage/searchability, honest unknown indexed counts and first-add guidance. |
| K10 — Primary task and mobile scope | Supported: Add/Ask precedes collapsed guidance; compact exact scope remains visible. Physical onscreen-keyboard/zoom coverage remains limited. |
| K11 — Actual failed-subset retries | Supported: retained queue/options/files, reattachment validation, item/all eligible failures, durable retry lineage, no reprocessing successes. |
| K12 — Recovery labels and dimensions | Supported: source inclusion opens the selector; retrieval depth changes top_k; Media/Notes links name their owners. Actual empty-result and controlled HTTP-error paths recorded. |
| K13 — Mobile Evidence/action overlap | Supported: Evidence has a reserved composer position; single named drawer heading/Close; controller narrow/short geometry and focus checks. |
| K14 — Global shortcut ownership | Supported: local destructive Cmd/Ctrl+K behavior removed; accurate help; real rendered-state retention tests plus controller palette observation. |
| K15 — Outcome-first analysis | Supported: Summary/Key claims/Custom, Advanced prompts, accurate defaults action, saved version remains visible; late cancelled generation is not published. |

| Enhancement theme | Assessment |
|---|---|
| One canonical source-set handoff | Shared helpers and recognizable selections implemented; end-to-end acceptance blocked by I1/I2. |
| Item readiness summary | Implemented with known Stored/Analysis/warnings and honest unknown search readiness; no invented indexed/vector readiness. |
| Outcome recipes | Four editable evidence-aware prompts implemented; tests/controller evidence show no automatic query or scope change. |
| Visible output review loop | Notes export/open/revise and Analysis version review supported; Research durable branch blocked by I1/I3. |
| Extension/full-page continuity | Shared vocabulary, canonical captured-note scope and full-options destination implemented; final native capture/Ask/build evidence remains controller-owned. M1 is a shared localization gap. |

Original minor follow-ups are addressed in code: explicit Evidence semantics/focus with one header, active suggestion identification, one collapsed guide surface, selected source titles rather than only counts, and separation of original Knowledge provenance from subsequent editing. These are not counted as outstanding findings.

## Prior rulings and retained observations

The chronological fixes are consistent with the amended requirements: gated pre-hydration arrivals; exact direct/folder scope; persistent readiness intent without overriding deliberate selection; no completed handoff replay; Notes edit-race protection; canonical content provenance and metadata-stripping API behavior; hidden marker previews; exact legacy note acknowledgment rather than a blanket numeric-ID exemption; real expected-version headers; evidence enrichment of reused source rows; canonical restore only for server-present sources; and Quick Notes version-read/write/acknowledgment ownership. The earlier retry reconstruction findings were fixed for both recovered terminal results and pre-submit failures, using existing retry metadata. Stateful shortcut proof and removal of duplicated mobile Evidence controls satisfy the earlier Task 3 findings. I1–I3 are cross-path gaps remaining after those scoped repairs, not reasons to discard their evidence.

- Node experimental-webstorage and disclosed React/CSS/i18n fixture warnings are test/environment observations, with no demonstrated production defect in this review. They are not additional blocking findings and do not support a warning-free claim.
- The large wizard test's existing formatter-idempotence limitation is distinct from parsing/test failure. Preserve unrelated formatting as instructed; do not broaden this repair into a file rewrite.
- Broad shared UI remains at 352 diagnostic blocks against the same-dependency 353 baseline, with zero introduced blocks reported. Official WebUI/extension type checks pass. This supports no introduced diagnostic regression, not a globally clean shared-UI claim.
- Frontend directory Bandit scans reported zero Python lines; this is correct non-applicability and supplies no TypeScript security proof. I3 comes from manual contract review.
- Final docs, Backlog status/summary cleanup, hooks and owned runtime/generated-artifact cleanup remain controller responsibilities. Historical task completion summaries must not substitute for successful final Research/native gates.

## Recommendations and technical assessment

Apply one focused fix wave for I1–I3 and M1, with regressions at the actual caller/transport boundary rather than helper-only fixtures. Revalidate the current import's durable completion ID, then actual canonical reload/reconcile, retained original note/web references, trust/scope, exact active source set and unchanged snapshot upload count. Finish the controller's stored-web Review→Research, external-web snapshot, native capture→Notes UUID→full-options Ask and extension build checks, updating the verification record honestly after any code change.

The overall design is appropriate and the regression effort is meaningful. Nevertheless, ordinary no-filter Research continuation is broken, a canonical media category fails its handoff, and a write drops an existing principal precondition. **Specification approval and technical merge readiness are withheld until these findings are corrected and the remaining controller gates are recorded.** Deterministic model/controlled web evidence does not establish live-provider answer quality or external search relevance; no recruited-user, screen-reader, physical keyboard/mobile, Safari/iOS or latency claim is made.
