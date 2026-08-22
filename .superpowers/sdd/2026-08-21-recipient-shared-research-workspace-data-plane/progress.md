# SDD ledger - plan: Docs/superpowers/plans/2026-08-21-recipient-shared-research-workspace-data-plane.md

## Execution preflight

- 2026-08-21: Rebased eight feature commits onto `origin/dev` at `2e0815c1e4` without conflicts. New HEAD: `0e006b5388`.
- 2026-08-21: Confirmed this is an isolated linked worktree on `codex/research-workspace-power-user-uat`.
- 2026-08-21: Confirmed ChaChaNotes current schema remains V60; Task 2 may use V61.
- 2026-08-21: Two unrelated untracked watchlist templates remain excluded from every task and commit.

## Preflight consistency scan

| Tasks/interfaces | Producer -> consumer | Finding |
| --- | --- | --- |
| 1 -> 5/7 | Task 1 removes unsafe recipient routes; Tasks 5 and 7 install typed bounded replacements | Intentional fail-closed interval; no alias or redirect allowed. |
| 1 -> 8/9 | Task 1 creates route parser/gate and temporary shared shell; Tasks 8/9 connect and render it | Interfaces and file lifecycle agree. |
| 1 -> 10 | Task 1 route-absence assertions become Task 10 request-ledger/security invariants | Tests and final contract agree. |
| 2 -> 3 | Task 2 creates V61 tables/RLS; Task 3 wires the modular store in `ChaChaNotes_DB.py` | Shared file and schema interfaces agree; Task 3 must preserve Task 2 policy wiring. |
| 2 -> 10 | Task 2 migration/policy tests feed the integrated matrix | Final command includes the policy contract and both backends. |
| 3 -> 5 | Recipient store supplies bounded history without creating a thread on read | API/history contract agrees. |
| 3 -> 7 | Store claims, source freezing, leases, replay, and atomic completion back chat orchestration | Lifecycle and failure semantics agree. |
| 4 -> 5 | Access service and read helpers feed bootstrap/source/preview/history endpoints | Authorization-before-owner-DB rule is consistent. |
| 4 -> 6 | Access context supplies authoritative owner/share/scope data to retrieval | Scope and ownership boundaries agree. |
| 4 -> 10 | Authoritative membership repository is exercised by security/UAT matrices | Stale token/request claims remain non-authoritative. |
| 5 -> 7 | Task 5 creates recipient schemas/routes; Task 7 extends chat and generation default | Task 5 is deliberately fail closed until Task 7 wires provider readiness. |
| 5/7 -> 8 | Approved API schemas feed typed frontend client/controller | Task 8 may implement from the spec but cannot be considered integrated before Tasks 5/7. |
| 5 -> 10 | Typed bounded envelopes feed OpenAPI and cross-user tests | Contract and generation steps agree. |
| 6 -> 7 | Retrieval creates verified evidence; generation consumes only budgeted evidence | Dropped/trimmed evidence cannot become citations. |
| 6 -> 10 | Locked RAG policy and provenance postconditions feed sentinel security tests | Security checks agree. |
| 7 -> 9 | Generation default, model target, errors, and citations feed shared chat UI | Generic model catalog remains discovery-only. |
| 7 -> 10 | BYOK scope, context budget, and no-fallback behavior feed integrated tests | Final matrix covers exact share scope and local-only token counting. |
| 8 -> 9 | Typed client, reducer, and controller feed dedicated panes | File lifecycle and state ownership agree. |
| 8 -> 10 | Abort/stale-response/request-ID behavior feeds E2E request ledger | Tests and implementation agree. |
| 9 -> 10/11 | Shared UI feeds Playwright and live CDP acceptance | No banner stack, local controls, or computer-control fallback. |
| 10 -> 11 | Integrated contracts/docs establish the live runner acceptance matrix | Task 11 remains real-process truth; Task 10 stubs only deterministic UI behavior. |

| Task | Internal consistency | Finding |
| --- | --- | --- |
| 1 | Files, route tests, removal assertions, and fail-closed behavior | Consistent. |
| 2 | V61 migration, constraints, PostgreSQL forced RLS, and parity tests | Consistent after post-rebase V60 confirmation. |
| 3 | DB-derived text tenant key, fenced receipts, cleanup, and two-backend tests | Consistent. |
| 4 | Authoritative membership, access ordering, Jobs helper, and preview helper | Consistent. |
| 5 | Typed envelopes, route-scoped errors, bounds, and neutral failures | Consistent; provider readiness remains fail closed until Task 7. |
| 6 | Frozen source scope, locked RAG policy, and runtime provenance checks | Consistent. |
| 7 | Target resolution, exact-scope BYOK, local context budget, generation, and endpoint | Consistent. |
| 8 | Typed client, reducer, abort behavior, idempotent retry, and route reset | Consistent. |
| 9 | Compact Sources/Chat UI, evidence, accessibility, responsive layout, and copy | Consistent. |
| 10 | Cross-layer tests, docs, OpenAPI, Bandit, and browser request ledger | Consistent. |
| 11 | Real backend/WebUI/CDP, ingestion wait, personas, race probe, evidence, closeout | Consistent. |

No preflight ruling was required; the apparent Task 5/7 and Task 8 dependency overlaps are explicit fail-closed staging, not contradictory contracts.

## Baseline verification

- Frontend route/layout baseline: 2 files passed, 17 tests passed.
- Focused sharing, migration, and RLS baseline: 91 passed, 1 failed. The pre-existing failure is `test_sqlite_migration_v38_to_v39_reopens_legacy_database`, which reaches the current V59 migration and fails closed on `Notes attachment v59 registry collision requires explicit repair`. No Task 1 changes existed when this was recorded.
- PostgreSQL AuthNZ sharing integration baseline: 5 tests skipped through the repository fixture because PostgreSQL remained unavailable after its Docker-start attempt. This is an explicit environment limitation, not a passing PostgreSQL result.
- Task implementations must preserve the 91 passing backend tests and may not attribute the V39 baseline failure or PostgreSQL skip to their own changes.

## Task 1 - Fail-closed recipient route gate

- Base: `0e006b5388b48c40a59e6326b68df9934fe75089`
- Implementer: `01a02663-595c-79e2-bfe2-189b54709736` (`Cicero`), `gpt-5.6-terra`, high reasoning.
- Brief: `task-1-brief.md`
- Commit: `1c5ad69f58 fix(workspaces): fail closed for recipient shares`.
- Implementer verification: frontend 23/23; backend 53/53; Ruff, Bandit, and `git diff --check` clean. Full UI typecheck remains non-green on reported pre-existing errors.
- Reviewer: `01a0267e-3fe5-7423-a021-084c1bbe4eae` (`Locke`), `gpt-5.6-terra`, high reasoning.
- Review package: `review-task-1-0e006b5..1c5ad69.diff`.
- Reviewer verdict: spec compliant; approved; no Critical, Important, or Minor findings.
- Controller-resolved review cautions: `rg` found no remaining active UI consumer of the deleted context/banner; `git status --short --branch` confirmed only the two unrelated watchlist templates remain untracked and unstaged.
- Status: complete.

## Task 8 - Typed frontend client and abortable shared controller

- Base: `8c199fce37f788439074d08b78f27059830874d0`.
- Brief: `task-8-brief.md`.
- Implementation: added strict recipient response types and bounded runtime parsing, authenticated canonical read/chat calls with abort propagation, normalized typed shared errors, a fail-closed reducer, and a hook controller with per-operation abort ownership and monotonically fenced share generations.
- Submission lifecycle: request UUIDs are allocated once, source IDs are normalized before an immutable payload is frozen, only ambiguous transport failures retain an exact-payload retry receipt, edits invalidate failed receipts, source conflicts refresh only while current, and stale responses cannot mutate state or trigger refresh side effects.
- Route shell: replaced Task 1's static surface with stable loading, neutral not-found, unavailable, and loaded placeholders driven only by the shared controller; local `/research-workspace` behavior and lazy isolation remain unchanged.
- TDD RED: the exact brief command produced two failed suites because the client and reducer/controller modules were absent; the unchanged route suite passed all 14 existing tests. Self-review RED added three expected failures for the canonical preview shape, normalized source IDs, and stale aborted conflict side effects.
- GREEN: exact Task 8 command `44 passed`; focused existing authenticated-fetch/domain-client/route-state regressions `42 passed`; `git diff --check` passed.
- Static gates: repository ESLint 9 passed every touched TS/TSX file, with only the shared Next.js config's expected missing-pages notice for the UI library package. Package-wide TypeScript exhausted Node's 4 GB heap before diagnostics; the focused Task 8 config reached the known package baseline (jest-dom matcher types, `chrome`/`browser`, and `import.meta.env`) with no Task 8 production diagnostics. Bandit is not applicable because no Python production code changed.
- Scope guards: canonical `/api/v1/sharing/shared-with-me/{share_id}` paths only; no alias, redirect, local workspace store import, owner mutation hook, banner/trust UI, or visual redesign. The unrelated watchlist templates remain untouched and unstaged.
- Report: `task-8-implementer-report.md`.
- Residual boundary: Task 9 still owns the complete recipient visual workspace, and Task 11 owns live backend/provider/browser UAT. Parent task remains In Progress.
- Status: implementation complete; pending controller review.
- Fix Round 1 reviewed head: `231136f213ff8c4da4187d0755747557257cb58c`.
- Fix Round 1: modeled initial selection as explicit `all` scope and explicit subsets as persistent cross-page `include` IDs; partial source pages reconcile only returned readiness while a complete unfiltered page may authoritatively remove absent IDs. The authoritative unfiltered source summary blocks an over-500 all-scope request until an include subset is selected, with select-all/clear controller transitions exposed for Task 9.
- Fix Round 1 lifecycle: every operation now checks current controller identity as well as share generation; chat successes additionally require the response request ID to match the submitted UUID. Rate deadlines gate submit/retry and drive a bounded remaining-millisecond countdown. Pending submissions retain the exact raw draft and revision, and only a matching unchanged revision clears after stored success. Untyped received HTTP errors discard the UUID; only ambiguous fetch `TypeError` failures retain exact-payload replay.
- Fix Round 1 TDD RED: focused controller target `11 failed, 15 passed`. GREEN: focused controller `27 passed`; exact Task 8 matrix `54 passed`; existing auth-fetch/domain-client/route-state regressions `42 passed`.
- Fix Round 1 gates: ESLint passed all three touched TS/test files with only the UI library's shared missing-pages notice; focused production typecheck reached 14 existing imported `import.meta.env` diagnostics and no changed-file diagnostic; Bandit not applicable to frontend-only production; `git diff --check` passed. The two unrelated watchlist templates remained untouched and unstaged. Parent remains In Progress.
- Initial reviewer: `01a0286f-5f29-7e13-a5f0-2805253f74b6` (`Bohr`), `gpt-5.6-sol`, high reasoning; review package `review-8c199fce37..231136f213.diff`.
- Initial review findings: cross-page selection loss/incomplete initial scope; same-share stale operations and missing chat response correlation; unenforced rate-limit countdown; in-flight exact draft edits cleared by success; and untyped HTTP failures misclassified as ambiguous transport errors.
- Fix Round 1 commit: `6e42b02dd4 fix(workspaces): harden shared recipient controller`.
- Scoped re-reviewer: `01a02888-d40a-7df1-a5a2-ec4f8d79da8c` (`Jason`), `gpt-5.6-sol`, high reasoning; review package `review-231136f213..6e42b02dd4.diff`.
- Task 8: fix round 1/5 (5 addressed, 0 open; commits `231136f..6e42b02`).
- Final re-review verdict: all findings addressed; no new Critical, Important, or Minor breakage.
- Task 8: complete (commits `8c199fc..6e42b02`, review clean).

## Task 9 - Dedicated recipient Sources and Chat surface

- Base: `5f2f1221ef`.
- Brief: `task-9-brief.md`.
- Implementation: replaced the shared placeholder with the approved compact recipient header, stable desktop Sources/Chat panes, mobile semantic tabs, responsive preview drawer/sheet, server-backed source controls, grounded chat history/composer/citations, exact server generation-default model seeding, direct in-pane recovery states, and owner revocation/provider copy.
- Task 8 semantics preserved: `all` means every queryable source across pages; `include` is a persistent explicit 1-500 ID subset; over-500 all scope and zero scope cannot submit; immutable retry/rate-limit state stays controller-owned.
- Safety and scope: server `allowed_actions` is the sole action authority; preview/model/citation content uses escaped text or sanitized Markdown; no local workspace store/API, mutation controls, banners, aliases, redirects, or `/research` coupling was added.
- TDD RED: exact three-file UI command failed 3 files/12 tests against the placeholder. GREEN: the same three files passed 18 tests. Final changed-scope matrix passed 8 files/80 tests; locale mirror passed 2 tests; route boundary passed 17 tests; ESLint and `git diff --check` passed.
- Complete focused suite attempt: 816 passed/20 failed. Seven route-fixture failures were corrected and reran 17/17; the remaining 13 reproduce standalone in three untouched baseline source-view files (7 SourceViewControls, 5 Stage 12 saved-view state, 1 SourcesPane fixture). Package TypeScript exhausted 4 GB; focused roots emitted only existing test/environment/global diagnostics and no changed production diagnostic. Bandit is not applicable to frontend-only production.
- English mirror: added the `sharedWorkspace` namespace to both locale formats and corrected 17 pre-existing missing extension aliases exposed by the full mirror test.
- Report: `task-9-implementer-report.md`.
- Residual boundary: Task 11 owns live backend/provider/browser UAT. The two unrelated watchlist templates remain untouched and unstaged.
- Status: implementation complete; pending controller review.
- Initial reviewer: `01a028b6-17f2-74e2-b954-343b01a50c8e` (`Lovelace`), `gpt-5.6-sol`, high reasoning; review package `review-5f2f1221ef..9846518465.diff`.
- Initial review blockers: generic Markdown crosses into artifact/Zustand/executable-preview paths; source inspection is not fully governed by `allowed_actions`; UI error mappings use noncanonical backend codes; older-history loading loses scroll position and has no error recovery; sequential previews retain stale evidence while loading; and responsive claims lack browser-level geometry evidence.
- Task 9: minor (deferred): remove duplicate native-button Enter/Space handlers and verify with `userEvent.keyboard`.
- Task 9: minor (deferred): keep stable hidden mobile tabpanels or remove invalid inactive `aria-controls` relationships.
- Task 9: minor (deferred): distinguish filtered no-results copy from a genuinely empty shared workspace.
- Fix Round 1 commit: `07b17d50ef fix(workspaces): harden shared recipient UX`.
- Fix Round 1 re-reviewer: `01a028ed-7ea6-7410-8902-010176ebebc0` (`Mendel`), `gpt-5.6-sol`, high reasoning; review package `review-9846518465..07b17d50ef.diff`.
- Task 9: fix round 1/5 (5 addressed, 1 open: canonical bootstrap parser and header still conflate server action authority with readiness/tier; commits `9846518..07b17d5`).
- Task 9: prior minors addressed in Fix Round 1: native-button keyboard handling and stable hidden mobile tabpanels.
- Task 9: minor (deferred): correct CDP checker/report wording about Playwright-launched Chrome plus CDP session and post-load text versus accessible-name evidence.
- Task 9: minor (deferred): enter the deterministic desktop draft before waiting for submit readiness rather than timing out into the setup path.
- Fix Round 2 commit: `c5a254b434 fix(workspaces): preserve shared action authority`.
- Fix Round 2 re-reviewer: `01a02902-5a5c-7460-ac8d-14c29ab22994` (`Ptolemy`), `gpt-5.6-terra`, high reasoning; review package `review-07b17d50ef..c5a254b434.diff`.
- Task 9: fix round 2/5 (1 addressed, 0 open; commits `07b17d5..c5a254b`).
- Final re-review verdict: canonical bootstrap actions remain verbatim, header capability/reason derives only from server actions, contradictory tier/action client and UI regressions pass, and no new breakage was found.
- Task 9: complete (commits `5f2f122..c5a254b`, review clean; 3 deferred minors ledgered for final review).
- Fix Round 1/5: replaced generic Markdown with a structurally isolated ReactMarkdown+GFM renderer using inert code, no raw HTML/images, and safe links; enforced `inspect_sources` and `ask_grounded_questions` as the only source/chat authorities without rewriting server actions for model readiness; aligned canonical errors; preserved older-history anchors with visible retry; and added preview-start clearing/loading/target state with stale-response fencing.
- Fix Round 1 TDD: blocking RED 8 failed/36 passed; semantic RED 4 failed/19 passed; inspect-controller RED 1 failed; final action-rewrite RED 1 failed. GREEN: exact Task 9 UI 23 passed; Task 8 controller plus safe renderer 32 passed; final combined changed surface 55 passed; changed-scope regressions 90 passed; locale mirror 2 passed.
- Fix Round 1 browser evidence: raw CDP only at 390x844 and 1440x900 passed with zero root/pane horizontal overflow, no out-of-bounds controls, a full-height 390x844 mobile loading sheet, stable adjacent desktop tracks, dynamic preview/submit names, and stable 44x44 submit geometry. The mandated directory suite produced 832 passed/14 failed: 13 documented untouched baseline failures plus one load-only Studio timeout that passed standalone in 530 ms.
- Task 9 Fix Round 2/5: preserved the strictly parsed server `ask_grounded_questions` decision when `generation_default` is unavailable, while retaining generation readiness as an independent fail-closed submit prerequisite. The header now renders `access_level` only as the literal policy-ceiling tier and exact ceiling tooltip; all capability and denial-reason copy comes from `allowed_actions`.
- Task 9 Fix Round 2 TDD: focused client/header UI RED produced 4 failed/25 passed; GREEN passed 29. Final amended client/header/UI plus locale mirror passed 31; exact Task 9 UI passed 24; affected Task 8 controller/route regressions passed 55. Focused ESLint passed; package TypeScript retained unrelated repository baseline diagnostics with no changed Task 9 diagnostic; static authority guards and `git diff --check` passed.
- Fix Round 1 gates: focused ESLint and checker syntax passed; full TypeScript retained repository baseline diagnostics with none in changed shared-workspace files; static isolation/error-code guards and `git diff --check` passed. Bandit is not applicable. Full report appended to `task-9-implementer-report.md`; unrelated watchlist templates remain untouched and unstaged; Task 11 retains live-backend/provider UAT.

## Task 7 - Provider resolution, grounded generation, and safe chat API

- Base: `fa5ce0b131ceb70422c1c36bf39fe4f423e1aa38`.
- Brief: `task-7-brief.md`.
- Implementation: added direct canonical provider/model resolution without fallback routing; exact-share-scope BYOK and separately derived trusted-base-URL authority; locally counted bounded grounded generation over an immutable budgeted `VerifiedSharedEvidence` subset; strict JSON response/citation validation; and the canonical authorization-first, claim-first, fenced shared-chat endpoint with replay, rate limiting, frozen-source revalidation, atomic completion, sanitized errors, and bounded audit metadata.
- Self-review: added exact frozen-snapshot hash enforcement and provider-timeout-derived 5-30 minute receipt leases.
- TDD RED: absent provider resolution produced `4 failed, 1 passed, 8 errors`; generation and endpoint initially failed collection on their absent contracts; focused frozen-receipt and lease tests each produced one failing test before their fixes.
- GREEN: exact Task 7 suite `78 passed`; recipient/retrieval/security regressions `171 passed`; ordinary chat provider/default regressions `200 passed, 1 skipped`.
- Gates: Ruff passed the touched scope (`chat.py`'s three whole-file I001 findings are unchanged from the base and were ignored only for that legacy file); Bandit reported zero findings/errors; `git diff --check` passed.
- PostgreSQL state: live PostgreSQL was not started or available. No schema, migration, RLS, or store SQL changed.
- Report: `task-7-implementer-report.md`.
- Residual boundary: Task 8 and controller-owned UAT still consume and validate the typed API. Parent task remains In Progress.
- Status: implementation complete; pending controller review.
- Fix Round 1 reviewed head: `0307fc49daac405dfcd2b63b300c929e66daf102`.
- Fix Round 1: split recipient claim persistence from owner retrieval resources so replay/active/conflict dispositions never open owner data; mapped claimant owner context entry/exit failures through fenced typed recovery; re-resolved frozen targets under current adapter/override policy and required an exact canonical pair; derived leases from a narrow local provider identity that handles qualified models, aliases, and defaults without adapter/credential work; made retryable/conflicted transitions mandatory with read-only durable completed/newer-active winner classification; and required a loadable enabled adapter for canonical target readiness.
- Fix Round 1 TDD RED: provider/readiness/lease target `4 failed, 2 passed`; replay owner-resource target `1 failed`; receipt-state/owner/frozen-target target `11 failed, 1 passed`.
- Fix Round 1 GREEN: provider/readiness/lease target `6 passed`; replay/claim ordering target `3 passed`; receipt-state target `12 passed`; combined provider/endpoint/store target `75 passed`; final exact Task 7 suite `97 passed`; recipient/read/retrieval/security regressions `171 passed`; ordinary chat defaults/resolution `200 passed, 1 skipped`.
- Fix Round 1 gates: Ruff passed all changed production/tests; Bandit reported zero findings/errors across the three changed production files; `git diff --check` passed.
- Fix Round 1 environment: live PostgreSQL and real provider UAT were intentionally not run. No schema, migration, RLS, frontend, route alias, or noncanonical endpoint changed. The two unrelated watchlist templates remained untouched and unstaged.
- Fix Round 1 report: appended to `task-7-implementer-report.md`. Parent task remains In Progress; status is implementation complete and pending controller review.
- Fix Round 2 reviewed head: `9901d7a06373bd77e2a4213ddb3894f0df6f2e60`.
- Fix Round 2: unified lease identity and full target selection behind one local candidate normalizer, so an unqualified model's unique pricing-catalog match now selects the same canonical provider for execution and receipt lease timing. Ambiguous/no-match requests preserve the default, explicit providers remain authoritative, and lease identity does not load adapters, resolve credentials, or consult credential-backed default-model storage.
- Fix Round 2 TDD RED: focused provider/lease target `2 failed, 6 passed`; only the unique catalog-provider identity and lease assertions failed.
- Fix Round 2 GREEN: focused provider/lease/target target `17 passed`; exact Task 7 suite `102 passed`; ordinary chat defaults/resolution `200 passed, 1 skipped`.
- Fix Round 2 gates: Ruff passed the changed production/tests; Bandit reported zero findings/errors for `chat_target_resolution.py`; `git diff --check` passed.
- Fix Round 2 environment: live PostgreSQL and live provider UAT were intentionally not run. No orchestration, receipt transition, adapter-readiness, frontend, route, schema, migration, RLS, or unrelated watchlist path changed. Parent remains In Progress; status is implementation complete and pending controller review.
- Initial reviewer: `01a0281f-7d8f-72e3-9208-087a3e2df1b1` (`Copernicus`), `gpt-5.6-sol`, xhigh reasoning; review package `review-task-7-fa5ce0b..0307fc4.diff`.
- Fix Round 1 re-reviewer: `01a0283f-be1d-7f53-bf76-df39250aa82f` (`Ramanujan`), `gpt-5.6-sol`, high reasoning; review package `review-task-7-fix1-0307fc4..9901d7a.diff`.
- Final re-reviewer: `01a02853-9f72-7c52-bace-659b683ed023` (`Herschel`), `gpt-5.6-terra`, high reasoning; review package `review-task-7-fix2-9901d7a..410b860.diff`.
- Final re-review verdict: all findings addressed; no new Critical or Important breakage. Live PostgreSQL and real provider/BYOK execution remain explicit Task 11 acceptance gaps.
- Status: complete.

## Task 5 - Typed bounded recipient read APIs

- Base: `52e95b5bb4bae2efefea1acb5b5147e759cfd776`.
- Implementer: `01a0270a-769c-72f3-a7cf-0dadde73d69e` (`Russell`), `gpt-5.6-sol`, high reasoning.
- Brief: `task-5-brief.md`.
- Staging guard: any interim chat route may validate/fail closed only; the removed unconstrained chat contract must not return.
- Commit: `6be3d619b8 feat(sharing): add bounded recipient workspace reads`.
- Reviewer findings accepted for Fix Round 1: raw URL search oracle, non-aggregate preview text bound, malformed canonical cursor misclassification, and OpenAPI/runtime response mismatch.
- Fix Round 1: source search now uses only projected safe fields; preview uses one focus-first aggregate text budget; canonical cursor `InputError` maps to exact typed 422 while operational failures remain 503; every recipient operation declares the strict error wrapper and interim chat advertises only its fail-closed 503 path while retaining its request schema.
- Fix verification: exact Task 5 matrix 102 passed; focused auth/introspection/OpenAPI/old-route target 11 passed; all-new-fix target 7 passed; Ruff clean; Bandit zero findings across 2,022 touched production LOC; `git diff --check` clean.
- PostgreSQL state: untouched. Residual staging condition: Task 7 still owns canonical safe chat generation.
- Fix Round 2 findings: broad API handling of all store `InputError` and representation-count-based preview truncation semantics.
- Fix Round 2: exported a canonical cursor-input-specific subtype and normalized every decoder rejection to it; the route now maps only that subtype to 422 while operational/corrupt stored-data errors remain 503. Preview truncation now follows the canonical source-content flag plus primary-preview shortening, not omitted supplemental chunks.
- Fix Round 2 verification: focused cursor/preview target 8 passed; full canonical SQLite store 33 passed; Ruff clean; Bandit zero findings across 3,112 production LOC with one existing fixed-SQL `nosec B608` skip; `git diff --check` clean. Serial aggregate attempt 1 and xdist/loadfile aggregate attempt 2 exceeded established cleanup-latency profiles; no assertion failed, no third aggregate was run, and no Task 5 pytest worker remained. Fix Round 1 accepted matrix evidence remains 102 passed plus 11 focused auth/introspection/OpenAPI/old-route passes.
- Final re-reviewer: `01a02773-878e-7051-b938-84ca5b4bb887` (`Leibniz`), `gpt-5.6-sol`, high reasoning.
- Final fix review package: `review-task-5-fix2-d800b952da..e09af9a232.diff`.
- Final re-review verdict: approved with no actionable findings. Cursor decoding is isolated to the canonical cursor subtype, operational stored-data failures remain 503, and aggregate preview allocation preserves source-content truncation semantics.
- Residual verification gap: aggregate tests were not rerun by the reviewer; the implementer recorded `104 passed` from the over-profile xdist diagnostic plus focused/store coverage.
- Status: complete.

## Task 6 - Frozen source scope and fail-closed retrieval provenance

- Base: `e09af9a232a3a606cf2117f684bff73e9c8d0d60`.
- Implementer: `01a02779-731d-7d02-bdb6-3000973551d5` (`Wegener`), `gpt-5.6-sol`, high reasoning.
- Brief: `task-6-brief.md`.
- Security gate: owner media is retrieved only through the frozen canonical source snapshot and a fully pinned retrieval-only RAG policy; any missing or out-of-scope provenance fails the whole result before generation.
- Implementation: added immutable canonical source snapshots, exact frozen-scope revalidation, duplicate-media canonical mapping, a signature-sentinel guarded retrieval-only RAG policy, complete-result provenance validation, and bounded deterministic evidence.
- TDD RED: focused Task 6 collection failed because `shared_workspace_chat_service` did not exist. GREEN: 35 focused tests passed.
- Regression verification: bounded Task 4/5/6 sharing/access matrix passed 199 tests; Ruff passed; Bandit reported zero findings across 917 production LOC; `git diff --check` passed.
- PostgreSQL state: untouched and not started. Task 6 adds no schema, policy, migration, fixture, or PostgreSQL query.
- Report: `task-6-implementer-report.md`.
- Residual boundary: Task 7 still owns API orchestration, receipt integration, generation, and citation serialization.
- Fix Round 1 reviewed head: `53665b9a979608964a2066d2a63f1ab064fc3c00`.
- Fix Round 1: preserved authoritative top-level RAG provenance, tightened actual `UnifiedRAGResponse` postconditions and metadata allowlisting, pinned active retrieval controls plus an AST hidden-kwargs sentinel, sourced canonical titles from current workspace rows, enforced exact snapshot identities and strict chunk/locator/conflict validation, sanitized malformed `all` rows, and raised the immutable evidence budget to 4,000 characters per item/48,000 aggregate.
- Fix Round 1 TDD RED: serializer `3 failed, 1 passed`; response/policy `28 failed, 3 passed`; snapshot/title `9 failed`; identity/capacity `15 failed`; timeout policy `1 failed`; contentless locator self-review `1 failed`.
- Fix Round 1 GREEN: serializer and focused Task 6 `94 passed`; final Task 4/5/6 regression matrix `258 passed`; Ruff passed; Bandit reported zero findings across 9,003 touched production LOC; `git diff --check` passed.
- Fix Round 1 PostgreSQL state: untouched and not started; no schema, migration, policy, fixture, or PostgreSQL query changed.
- Fix Round 2 reviewed head: `f741f881e486540b43df4e5efd57f6ca718a42ca`.
- Fix Round 2: added a default-on unified-pipeline retrieval-diagnostics gate and pinned it off for shared retrieval, supplied the owner user namespace on the real retrieval path, rejected malformed authoritative media identities as storage-shape failures, hardened the hidden-kwargs sentinel against every unapproved outer `kwargs` load, and restored ordinary serializer metadata precedence for every non-source field while retaining authoritative top-level provenance.
- Fix Round 2 TDD RED: real-pipeline diagnostics `1 failed`; owner namespace integration `1 failed, 2 passed`; malformed all-mode identities `14 failed, 6 passed`; kwargs analyzer negative fixtures `8 failed, 1 passed`; serializer/shared actual-shape regressions `3 failed`.
- Fix Round 2 GREEN: serializer plus focused Task 6 `126 passed`; focused Task 4 access `9 passed`; focused Task 5 recipient/security `40 passed`. The 290-item xdist aggregate reached `pytest_sessionfinish` without an assertion failure but exceeded the established cleanup profile in `xdist`/`execnet` node teardown and was interrupted once with exit 130; it was not repeated. Ruff passed; Bandit reported zero findings across 9,006 touched production LOC; `git diff --check` passed.
- Fix Round 2 PostgreSQL state: untouched and not started; no schema, migration, RLS policy, fixture, or PostgreSQL query changed.
- Fix Round 3 reviewed head: `91b278aef3102ebcbc87e670d147e00f26057734`.
- Fix Round 3: made the test-only hidden-kwargs AST sentinel inspect nested callable decorators, defaults, argument/return annotations, and optional runtime type parameters in the enclosing scope before skipping only a body with independently bound `kwargs`; preserved approved literal header reads and added bypass/allowed fixtures.
- Fix Round 3 TDD RED: required header fixtures `6 failed, 10 passed`; expanded rejection plus approved-literal header selection `9 failed`. GREEN: focused sentinel `20 passed`; full Task 6 plus serializer `136 passed`; Ruff passed the touched test; `git diff --check` passed. Bandit was not rerun because production code was unchanged.
- Fix Round 3 PostgreSQL state: untouched and not started; no production code, schema, migration, RLS policy, fixture, or query changed.
- Final re-reviewer: `01a027e6-48a2-7680-8d38-f0b1b58bda9c` (`Socrates`), `gpt-5.6-terra`, high reasoning.
- Final review package: `review-task-6-fix3-91b278a..e4045fb.diff`.
- Final re-review verdict: all findings addressed; no new Critical or Important breakage.
- Status: complete.

## Task 4 - Authoritative access and reusable read helpers

- Base: `ad1fbe49e79424fc2bbcd2353b6c201d7ae37c54`.
- Implementer: `01a026ea-8a7c-7601-a0b3-153f6042c40d` (`Turing`), `gpt-5.6-sol`, high reasoning.
- Brief: `task-4-brief.md`.
- Reconciliation: existing `test_authnz_sharing_postgres.py` must be extended rather than recreated.
- Commit: `52e95b5bb4 feat(sharing): authorize canonical shared workspace reads`.
- Implementer verification: final combined 81 passed/6 standard PostgreSQL skips; Ruff clean for Task 4 scope; Bandit zero findings; diff check clean.
- Reviewer: `01a02706-66f5-7891-a026-907f5e46fe33` (`Carver`), `gpt-5.6-sol`, high reasoning.
- Review package: `review-task-4-ad1fbe4..52e95b5.diff`.
- Reviewer verdict: spec compliant and approved; no Critical, Important, or Minor findings.
- Residual risk: live PostgreSQL execution standard-skipped; internal scope serialization remains a Task 5 gate.
- Status: complete.

## Task 3 - Fenced recipient chat store

- Base: `6378eb63e4c5ffef7856ebd212ee60cfeed882c2`.
- Implementer: `01a026ac-32d1-7403-8a5b-5595069ae815` (`Aquinas`), `gpt-5.6-sol`, high reasoning.
- Brief: `task-3-brief.md`.
- Commit: `7ba1ed7caf feat(sharing): persist fenced recipient chat turns`.
- Implementer verification: SQLite store 27 passed; combined store/RLS 46 passed/2 standard skips; regressions 35 passed; Task-scope Ruff clean; Bandit zero findings; diff checks clean.
- Residual environment limitation: live PostgreSQL store tests standard-skipped.
- Reviewer: `01a026ce-0f45-73e0-bb0c-a18c43a55061` (`Kuhn`), `gpt-5.6-sol`, high reasoning.
- Review package: `review-task-3-6378eb6..7ba1ed7.diff`.
- Reviewer verdict: cohesive design, but four Important findings: SQLite conflict-retention timestamp encoding, reclaim-race winner misclassification, non-discriminating PostgreSQL RLS store test, and role-cleanup ordering after pool closure.
- Controller ruling: all four findings are confirmed against the implementation and accepted. No schema change is required.
- Fix round 1 commit: `8d56a2f4c0 fix(sharing): harden recipient chat store races`.
- Fix verification: SQLite 32 passed; combined 51 passed/2 standard skips; RLS contracts 19 passed; Ruff clean; Bandit zero findings; diff check clean.
- Scoped re-reviewer: `01a026df-981f-71c3-bf67-b152a5b2fd2a` (`Descartes`), `gpt-5.6-terra`, high reasoning.
- Fix review package: `review-task-3-fix1-7ba1ed7..8d56a2f.diff`.
- Task 3 fix round 1/5: 3 addressed, 1 open; commits `7ba1ed7..8d56a2f`.
- Open finding: cleanup still lacks executable handling for pre-fix/native SQLite `CURRENT_TIMESTAMP` rows; canonicalizing only new transitions does not protect existing conflicted receipts.
- Fix round 2 commit: `ad1fbe49e7 fix(sharing): support legacy chat receipt timestamps`.
- Fix verification: focused legacy timestamp RED then GREEN; SQLite 33 passed; combined 52 passed/2 standard skips; Ruff clean; Bandit zero findings; diff checks clean.
- Scoped re-reviewer: `01a026e8-01df-76b1-aa4c-484b3c7e6805` (`Arendt`), `gpt-5.6-terra`, high reasoning.
- Fix review package: `review-task-3-fix2-8d56a2f..ad1fbe4.diff`.
- Task 3 fix round 2/5: 1 addressed, 0 open; commits `8d56a2f..ad1fbe4`.
- Scoped re-review verdict: all findings addressed; no new Critical/Important breakage.
- Residual risk: live PostgreSQL remains unavailable locally; typed PostgreSQL paths and integration collection are preserved with standard fixture skips.
- Status: complete.

## Task 2 - V61 recipient chat schema and forced RLS

- Base: `1c5ad69f58e34db13f486ccf175fb8a9bdfdcf2b`.
- Implementer: `01a02680-fc23-7f33-bd6b-147922fa9e6a` (`Hume`), `gpt-5.6-sol`, high reasoning.
- Brief: `task-2-brief.md`.
- Commit: `8cd3f1d900 feat(sharing): add recipient chat receipt schema`.
- Implementer verification: GREEN 41 passed/2 standard skips; related regressions 69 passed/1 standard skip; Ruff task scope clean; Bandit zero findings; `git diff --check` clean.
- Residual environment limitation: live PostgreSQL fixture unavailable; deterministic PostgreSQL DDL/RLS contracts passed.
- V39 treatment: historical V38->V39 step is tested directly; unrelated V59 repair behavior is unchanged.
- Reviewer: `01a0269b-e840-7761-beb0-bf3e1097752b` (`Faraday`), `gpt-5.6-sol`, high reasoning.
- Review package: `review-task-2-1c5ad69..8cd3f1d.diff`.
- Reviewer verdict: production schema/RLS coherent, but task needs fixes for executable PostgreSQL RLS isolation and constraint coverage.
- Controller ruling: both Important findings are accepted. Task 3's store-level isolation does not replace a restricted-role database-policy test, and Task 2's brief requires PostgreSQL constraint parity rather than source-string checks alone.
- Fix round 1 commit: `6378eb63e4 test(sharing): exercise recipient chat postgres contracts`.
- Fix verification: deterministic PostgreSQL contracts 3 passed; live integration 3 standard fixture skips; full Task 2 matrix 41 passed/4 standard skips; Ruff and diff checks clean.
- Scoped re-reviewer: `01a026a9-9c2c-7862-a253-e3501c5a609b` (`Newton`), `gpt-5.6-terra`, high reasoning.
- Fix review package: `review-task-2-fix1-8cd3f1d..6378eb6.diff`.
- Task 2 fix round 1/5: 2 addressed, 0 open; commits `8cd3f1d..6378eb6`.
- Scoped re-review verdict: all findings addressed; no new Critical/Important breakage.
- Residual risk: live PostgreSQL execution remains unavailable locally, but the new restricted-role and constraint tests standard-skip without custom suppression.
- Status: complete.
