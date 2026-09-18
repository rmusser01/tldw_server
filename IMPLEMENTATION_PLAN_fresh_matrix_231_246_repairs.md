# Fresh matrix findings231–246 implementation plan

> **For agentic workers:** Use the existing coordinated implementation/review workflow. Do not start new full UAT before the repair gate. Root owns shared tracker, Backlog, runtimes, native browser and commits.

**Goal:** Resolve every new finding from the frozen four-configuration matrix, including follow-up findings discovered during repair (now258), and verify its original workflow before another full UAT.
**Architecture:** Make bounded corrections in the existing schema, authorization, proxy, UI state and scheduling paths. Keep independent work in disjoint files and review each unit before integration.
**Tech stack:** FastAPI/Python, SQLite/PostgreSQL, Next16.1.4, React/TypeScript, pytest/Vitest, native Playwright CLI.
**Design:** [Repair decisions](Docs/Design/2026-09-17-fresh-matrix-repairs-231-246.md).
**Task:** TASK13260 and children173–200.

## Global constraints

- Branch `codex/fresh-install-uat-fixes`; retain original ancestry disclosure. Never edit archived frozen source trees.
- Do not weaken RLS, quota failures, foreign-access denial, model readiness or ownership guards.
- Existing official PostgreSQL fixtures are mandatory; zero PostgreSQL skips.
- Activate `.venv` before Python/pytest/Bandit; no secrets in logs/evidence.
- One author owns overlapping paths; root stages/commits reviewed allowlists only.
- User already authorized all issue repairs. Routine reversible work proceeds without another permission gate.

## Stage1: Preserve the completed matrix

**Goal:** Commit the final PostgreSQL multi-user evidence and reconcile all48 outcomes.
**Success criteria:** Independent hashes/credential scan/row references pass; all new findings have tasks; bounded failures remain explicit.
**Tests:**186payload hashes, source parity,15frozen harness hashes, all native row references, current credential/JWT scan and scoped diff check.
**Status:** Complete

- [x] Finish native PostgreSQL multi-user journeys, including actual expiry, reciprocal access and real outage/Retry.
- [x] Stop only owned apps/browser, preserving database fixture holders and data.
- [x] Retain packet and correct review-found row7 filename reference.
- [x] Complete independent evidence review and bind auxiliary checksums.
- [x] Include evidence, tracker, Backlog and this plan/design in the retention checkpoint commit.

## Stage2: Repair backend and proxy blockers

**Goal:** Restore fresh PostgreSQL upload admission/content persistence, catalogue access and long-running generation.
**Success criteria:**245,238,239,233 and234 have causal regressions, minimal reviewed fixes and adjacent controls.
**Tests:** Normal fresh/repeated AuthNZ quota bootstrap; actual restricted worker→executor→Media INSERT/foreign denial/scope cleanup; world-book catalogue read; warning uniqueness; actual installed Next response after30seconds plus abort/failure controls.
**Status:** In Progress

- [x]245: extend canonical bootstrap and real quota/guard controls; retain partial-index upserts, FK/check constraints, fractional usage and fail-closed behavior. Independent52/0skip; native admission pending.
- [x]238/233: reproduce actual worker persistence with trusted owner in required PostgreSQL; carry scope through executor and reset it; remove warning self-extension at its producer. Independent48/0skip within combined165; native pending.
- [x]239: reproduce catalogue failure through actual PostgreSQL connection wrapper and use supported read lifecycle; independent32/0skip, native acceptance pending.
- [x]234: installed rewrite deadline repaired and independently verified; original five-fact/five-card generation/save/distinct Study/reload accepted in both PostgreSQL modes at44.469/35.542seconds, with paired native audits and retention review.
- [x]247: prevent own Media sequence rewind under RLS-filtered tenant initialization, retaining high-water IDs and explicit-ID repair without weakening authority; separate from147's foreign-table allowlist. Independent117/0skip within combined165, including lock lifetime/caller rollback; native pending.
- [x] Review and commit each independently testable backend/proxy unit; native acceptance remains pending until Stage4.

## Stage3: Repair UI, study and stream findings

**Goal:** Correct231,232,235–237,240–244 and determine246's actual failure boundary.
**Success criteria:** Each original behavior has a causal regression and minimal reviewed correction;246 has a supported disposition, not an assumed timeout cause.
**Tests:** Actual component/hook/QueryClient lifetimes; provider alias conflict controls; saved ordinary-mode transition; canonical re-rate previews; real SQLite/PG Hard/Again analytics; one/multiple localization; transport first-byte/idle/abort boundaries.
**Status:** In Progress

- [x] Study235/242: independent100/0skip plus both PostgreSQL modes native authoritative re-rate14day preview/persistence, practice-only no write, separate one-card Due singular completion and reload; paired native audits and retention clear.
- [x] Analytics240: independent296pass/0skip across SQLite/PostgreSQL, Bandit0/no added lint; both PostgreSQL modes native Good/Hard/reload retain lapses0 and100%retention/0%lapse analytics, independently accepted.
- [x] Media237/241/244: reviewed193+37 independent checks, including two review-discovered deletion callback races; exact90compiler baseline and no new lint diagnostics. Native acceptance pending.
- [x] Chat/model231/232/236/243: independently reviewed327focused/103adjacent pass;2existing speech fixture failures reproduced on baseline; exact90compiler baseline, no new lint diagnostics. Native acceptance pending.
- [x]249: repair only the stale speech timeout fixture using current configuration storage; retain original timeout and sanitizer assertions, with independent verification.
- [x]250: restore per-test console-spy isolation in the background proxy fixture; preserve all assertions and verify the seven baseline failures plus combined adjacency.
- [x]248: use the existing captured Character request scope/lifetime for transport and every persistence boundary; actual-auth causal regression, same-owner/recovery controls, independent139focused/338adjacent review clear; native account-boundary verification pending.
- [x]246 implementation: correlate native late gzip delivery and prove installed Next compression buffers the first role frame; share no-transform headers across all three Character SSE branches. Independent35backend/8socket checks pass with zero skips. Exact TestBot native frame delivery and canonical reload on repaired source remain Stage4 requirements.
- [ ] Review and commit disjoint units, recording tests and touched-scope static checks.

## Stage4: Combined verification and original native acceptance

**Goal:** Verify all repairs together before another full matrix.
**Success criteria:** Relevant tests pass, mandatory PostgreSQL0skips, no added compiler/security findings, and each finding has native original-scenario evidence or an explicit verified external disposition.
**Tests:** Combined affected frontend/backend suites; recorded90diagnostic compiler comparison; scoped lint/Bandit; exact real model workflows on declared repair profiles.
**Status:** In Progress

- [x] Run combined tests:616required PostgreSQL/SQLite,1055frontend plus37consumer; zero skips. Compiler90identical baseline diagnostics. Fixture249/250 failures repaired without removing assertions.
- [ ] Review the final touched diff independently.
- [ ] Repeat exact quota→ingest/source/owner paths, Biology5generation/draft/save/five-card Study, model selection/TestBot, Media delayed handoff/refresh/disconnect and all remaining issue-specific native checks.
- [ ] Hash and retain evidence; update per-issue ledger with revision, tests, native acceptance and remaining limits.
- [ ] Only release the next matrix when no identified issue remains unresolved or awaiting acceptance.

- [ ]251: repair actual PostgreSQL MCP permission-profile nullable filtering; retain scope controls and native Save packs acceptance.
- [x]252: repair the baseline-confirmed protected-pool DDL test fixture while preserving the missing governance-table assertion and production guards; independent139combined tests, zero skips and no new static findings.
- [x]253 implementation: repair actual PostgreSQL MediaFiles bindings in six repository methods; independent93tests/zero skips and retention review clear. Original persisted source/native full-content handoff acceptance remains pending.
- [x]196 harness task: independently review a bounded source-upgrade launcher preserving original profiles, data, fixture holders and historical receipts, then record explicit new-runtime provenance for targeted acceptance. This is not a new UAT finding or a fresh-install matrix cell.
- [x]254: actual first upgrade frontend exposes incompatible build-directory prefix despite124 synthetic passes. Preserve failed attempt, correct only helper prefix, prove real Next config acceptance (causal RED→125combined passes), independently review and retry with new copy-run identity. Product config guard and original profile/fixture records remain unchanged.

## Stage5: Repeat full fresh UAT

**Goal:** Re-run the same48named outcomes on one reviewed frozen revision with new config/data/browser state.
**Success criteria:** Every row has truthful evidence; every new issue is tracked; no blanket acceptance with product failures.
**Tests:** Both modes × SQLite/PostgreSQL, authoritative12journeys, real provider, canonical persistence, isolation, natural expiry and outage recovery.
**Status:** Not Started

- [ ] Freeze/recheck source and dev ancestry, create isolated new profiles using existing reviewed harnesses.
- [ ] Run cells serially with a running tracker and retain all failures/limits.
- [ ] Review/commit the results and report actual acceptance status.

## Execution ledger

- 2026-09-17 17:51UTC: frozen matrix execution finished;16newfindings231–246 remain open. Runtime ports18603/18683 refused; PG holders29823/96865 remain.
- Retention review corrected one row7 filename typo; initial manifest081a0d27... superseded by1d4d60585e9cb0a48544ce212448559f51a4592f7b1bd93ca04f1be6b4f2c593. No outcome changed.

- 22:06UTC checkpoint:243/248 and254 independently native-accepted with retained74payload packet and retention review. Account transitions, cancellation/canonical history and corrected real startup/data preservation verified.246 and253/241 native positives collected; independent acceptance pending. MCP251 uses a new targeted fresh PG single profile without resetting completed original setup. Full matrix remains held.

- 22:33UTC:232 ordinary native400 still renders generic guidance. Existing task174 reopened implementation; cause-chain mapping passes20focused checks after causal red2/18controls. Independent review and native updated-source retry required.241/246/251/253 native reviews clear, retention review pending. Alice/Bob reciprocal media isolation captured; both upload analyses warn truthfully on truncation. Full matrix remains held.

- 22:40UTC:241/246/251/253 accepted after independent native/retention review; tasksDone.232 cause-chain fix independently203passes and90unchangedcompiler, nativeupdatedsourcepending. New255/task197 actualPGworldbookcreation500 requires tests-first supported transaction/readback repair and independent review. Fullmatrixheld.

## Task 255: PostgreSQL world-book creation and endpoint readback

**Backlog:** TASK13260.197. **Status:** In Progress. **Baseline:**6f6983b0620aae1f0892c6b0d3ae3bebfc105e02.

**Requirements:** Repair the observed normal Create World Book500 on actual PostgreSQL using existing supported DB lifecycle interfaces. Native Alice POST22:37:24.025UTC on a7d3155 returns Failed to create world book; backend15:37:24PDT confirms BackendConnectionWrapper context-manager error. The creation method uses an unsupported connection context; endpoint subsequently calls get_world_book and get_entries. Investigate and cover that complete endpoint path, avoiding unrelated CRUD rewrites.

**Owned scope:** tldw_Server_API/app/core/Character_Chat/world_book_manager.py and narrowly necessary tests under tldw_Server_API/tests/DB_Management/. Existing adjacent world-book tests and WorldBookService are references. Parent owns all Git, Backlog, tracker, native browser/runtime, frozen archives and model services; author must not edit these or commit. No subagents from implementer.

**Steps:**
1. Read actual manager, service, endpoint and existing portable transaction/read helpers; reproduce before implementation with official PostgreSQL fixtures plus SQLite controls.
2. Use supported write transaction and read-only lifecycle boundaries. Preserve outer caller rollback/commit ownership, standalone durability, duplicate-name conflict mapping, owner isolation, flags/defaults, soft-deletion filters and entry-count readback. No raw SQL outside existing DB abstraction, no schema/RLS/role relaxation.
3. Make the smallest repair for the actual create/readback path; run focused and relevant adjacent SQLite/PostgreSQL tests with zero skips. Record causal RED and final GREEN with commands, evidence files and exit codes.
4. Run .venv Ruff/compile/Bandit on touched Python scope, compare any existing findings rather than silently excluding new ones; self-review, write source hashes and full report. Parent dispatches independent spec/code review and original native acceptance before closure.

**Mandatory test runner:** activate .venv first, then use `TLDW_UAT_EVIDENCE_LABEL=worldbook255-<unique> node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs <test paths> -q --tb=short`. This runner uses official fixtures with explicit required PostgreSQL provisioning. Do not set no-Docker or create substitute databases. Never print/cat private credentials/configs/raw runtime logs. Retain sanitized outputs under .tmp/uat-repairs-231-246/worldbook255/. Actual browser failure is .tmp/uat-repairs-231-246/native-targeted/pg-multi/worldbook239-created.txt; safe cause extraction is worldbook255-error-cause-v2.json in that directory.

**Success criteria:** Real create+endpoint readback succeeds with persisted unique ID/metadata/entries on both DB backends; caller transactions and other-owner controls remain valid; no added static/security findings; independent review and later native Create/catalog/Character-editor readback accepted. Existing catalog239 and initialization112/Character-read153 repairs remain intact. No native runtime/source archive changes by implementer.

- 22:50UTC:232 native model_not_available guidance nowcorrect on6f6983b062. New256: repeated new identicalquestion/clientID failsRetry409againstolder answeredquestion; original5canonicalrows unchanged.256 repair required before232recovery acceptance/fullmatrix.255 implementation remains independent; no concurrent second implementer.

## Task 256: distinguish a new repeated question from an answered retry replay

**Backlog:** TASK13260.198. **Status:** In Progress. **Baseline:**6f6983b0620aae1f0892c6b0d3ae3bebfc105e02.

**Requirements:** A new ordinary saved-chat user turn may have the exact text/images of an earlier answered turn. If the new request fails model validation before persistence, explicit Retry with its same new client identity must create that new question and answer normally. Preserve the existing completed-tail replay rejection, pending/error-tail content and attachment mismatches, conservative legacy callers without valid correlation IDs, conversation/owner isolation and canonical row identity. Do not use timestamps or text equality alone as identity. This repair does not introduce a global historical idempotency contract.

**Observed scenario:** cb561345-2d7f-43eb-93e2-ef60d5d3e07f contains the earlier answered question with client pa_e8ad-169b-120-f818. A new identical question has client pa_556d-145c-831-c040, receives real invalid-model400 at22:47:04.251 before persistence, then real Retry409 at22:47:46.363. Earlier5canonical rows remain after reload. Native evidence under .tmp/uat-repairs-231-246/native-targeted/pg-single/model232-upgraded-*.txt. Read-only diagnosis at .tmp/uat-repairs-231-246/retry256-diagnosis/REVIEW.md when available. Existing Chat/chat_service.py retry matching inspects an answered content-matching tail before the saved client-ID match used for pending/error tails.

**Owned scope:** tldw_Server_API/app/core/Chat/chat_service.py and narrowly necessary existing/new ordinary Chat retry tests. Parent alone owns Git, Backlog/docs, native browser/runtime/model, profile config and frozen archives. No implementation subagents, no commits or runtime actions by author. WorldBook255 changes are unrelated and must be left untouched.

**Steps:**
1. Read the relevant retry service, endpoint ordering and existing actual SQLite/PostgreSQL retry tests; add causal red regression for a new distinct client ID following an answered identical question, including canonical persisted new user and assistant rows.
2. Make the smallest identity-aware repair. Test same answered-tail ID replay, no/invalid ID legacy behavior, unresolved-tail mismatch, changed content/attachments, existing acknowledged-tail reuse, owner/conversation isolation and transaction behavior as applicable. Preserve request shape and existing frontend recovery metadata. Confirm the newly accepted turn cannot itself be replayed after success.
3. Use official SQLite/PostgreSQL fixtures via mandatory runner, with zero skips; run focused and relevant adjacent suites. Preserve causal RED and final GREEN logs, commands, exit codes and source hashes.
4. Activate .venv and run scoped Ruff/compile/Bandit; distinguish unchanged baseline findings and pytest B101 from newly introduced production issues. Self-review and write full report. Parent requires independent review then native exact repeated-question failure/Retry/reload on reviewed source before closure.

**Mandatory runner:** `source .venv/bin/activate && TLDW_UAT_EVIDENCE_LABEL=retry256-<unique> node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs <test paths> -q --tb=short`. No fixture substitution, disabled PostgreSQL, weakening guards, broad refactors, timeout/provider tuning or mocked successful native responses. Keep test artifacts in .tmp/uat-repairs-231-246/retry256/. Never output private profiles, credentials or raw runtime logs. Maximum3 failed implementation attempts per issue before documented reassessment.

**Success criteria:** causal regression passes on both actual backends; fresh new identical retry persists its own rows; real answered replays and ownership/content/image violations remain rejected; no new production lint/security findings; independent review clear and parent native canonical reload positive.

-23:17UTC:255 implementation committed15c1bd5134 after36actualSQLite/PostgreSQL passes and independent round-one review. Retained packetworldbook255-reviewed manifest9833ed2b817b980e0f484c27397d33431704a06c4d6f329a7a3594b16aa4d1b0,46payload/checksum comparisons; original audit-overwrite correction explicit. Runtime remains6f6983b062; no native255 claim yet.
- New257/task199: source-only QA factual and broaderRowan searches return emptycontexts despite ready/readable originalPostgreSQLsource. Independent diagnosis active. New258/task200: fresh unconfiguredWebUI sends protected ingestion-capability request401; separately tracked from corrected231guidance.256 mandatoryPG initially blocked by sandbox reachability; realDockerfixture up, rerun with authorized local-network access. Do not skip PostgreSQL.

- 23:34 UTC: UAT231/233/237/244/245/247 accepted and tasks closed after independent native and retention review. Total 258 findings: 250 verified, 5 awaiting native acceptance, 3 active repairs. Retained manifest c4f50b39282f2d6f71d8ac82763357608f7f6d9dffd475b3186fe0433fb95cbf. Running metadata hash changes are historical observations, not payload corruption. Full matrix remains held.

## Task 257: restore owned PostgreSQL source retrieval in Knowledge QA

**Backlog:** TASK13260.199. **Status:** In Progress. **Baseline:** 9bb3450cb8 (reviewed 256 source is present but disjoint and owned by root until commit).

**Requirement:** Original ready Rowan source remains readable by ordinary Media full-text search, yet actual standard hybrid/chunk Knowledge QA returns empty contexts for both a facts question and Rowan. Establish the first failing actual PostgreSQL stage before editing production. Preserve owner isolation, legitimate no-match behavior, external-service failure semantics, and existing SQLite behavior. No blind threshold changes, reingestion, admin bypass, all-rows fallback, timeout/provider tuning or mocked successful native response.

**Owned scope:** Narrow RAG retrieval/pipeline/stream implementation and focused RAG tests; use existing DB abstractions. If root cause requires a shared DB implementation change, report the precise proven boundary before expanding scope. Parent owns Git, Backlog/docs, frozen sources, browser/runtime/model/profiles. No commits, subagents or native operations by implementer.

**Plan:**
1. Read .tmp/uat-repairs-231-246/retrieval257-diagnosis/REVIEW.md and existing actual worker content-scope, dual-backend retrieval, and stream parity tests. Use the official PostgreSQL fixture with a restricted non-superuser/non-BYPASSRLS role. Reproduce through actual persistence and actual retrieval/stream adapter, with external embedding/reranker/generation controlled. Preserve normal same-owner Media FTS positive control.
2. Trace safe counts at raw PostgreSQL FTS, normalization, merge/filter/rerank and emitted contexts; verify actual owner scope at the failing stage. Record causal RED and identify exact fault. Repair only that fault with existing supported interfaces. Do not swallow errors into successful empty output.
3. Cover same-owner Rowan and facts question, hybrid with no vector results and media fallback when no chunks exist, no-match negative, other-owner exclusion, sequential pooled owner1→owner2→owner1 and cleanup/cancellation where affected. Keep actual DB/retriever boundaries real; do not mock the defective stage. Compare stream/non-stream behavior.
4. Run official focused/adjacent SQLite/PostgreSQL checks with zero skips; scoped Ruff/compile/Bandit after activating .venv. Classify unchanged warnings and pytest B101 separately. Write report with commands, exit codes, source hashes, limitations and evidence. Parent independently reviews and repeats original native QA before closure.

**Runner:** source .venv/bin/activate && TLDW_UAT_EVIDENCE_LABEL=retrieval257-UNIQUE node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs TESTS -q --tb=short. Use require_escalated for fixture network; sandbox refusal is not PostgreSQL unavailability. Never skip/substitute/disable PostgreSQL. Never print private profiles/config/credentials or raw runtime logs. Evidence .tmp/uat-repairs-231-246/retrieval257/. Maximum three failed implementation attempts before documented reassessment.

**Success:** causal defect repaired with real restricted PostgreSQL evidence and unchanged isolation, independent review clear, original source-only native QA yields cited owned evidence. Full matrix stays held.

- 23:43 UTC: new UAT259/TASK13260.201 records native DELETE404 despite same-owner GET200 for Media1. Read-only diagnosis assigned; no second implementation author. UAT238 analysis/reanalysis and real failed-generation preservation captured, but QA257 and Trash259 keep the dependent workflow open. Retry256 committed2787043410; explicit new copy preparation is running before native acceptance.

## Task 258: defer protected ingestion-capability probing until authentication is configured

**Backlog:** TASK13260.200. **Status:** In Progress. **Baseline:**2787043410fc918b2c280d90f753d8fd02b6b35b.

Fresh unconfigured WebUI Media bootstrap sends GET /api/v1/ingestion-sources/capabilities and receives401. The generic credential gate and corrected231 guidance already work. Source server-capabilities.ts fetchCapabilitiesFromServer performs public OpenAPI/docs-info then unconditionally probes the protected endpoint when generic support exists. Existing cache is authority-scoped by buildChatSurfaceScopeKeyFromConfig; preserve this.

**Owned scope:** apps/packages/ui/src/services/tldw/server-capabilities.ts and services/__tests__/server-capabilities.test.ts; narrow shared authentication helper only if existing supported helpers cannot express readiness. Root owns Git/Backlog/docs/runtime/browser/model and frozen sources. No subagents or commits. RAG257 is awaiting actual native-row diagnosis with no active implementation author.

**Design:** Defer only the optional protected entitlement probe while required credentials are absent or invalidated. Keep public capability discovery, generic source availability and unknown entitlement state. Once configured, perform the existing protected probe; keep legitimate connected failures and cache/account boundaries. Reuse existing single-user/manual key, multi-user token, runtime override, hosted and cookie-session semantics; do not invent an API-key-only gate that breaks supported authenticated transports. Avoid introducing a general readiness framework or coupling capability discovery to a circular store dependency.

**Steps:** (1) Read existing TldwApiClient auth/config helpers, connection readiness and capability tests. Write causal request-dispatch regression before production changes. (2) Minimal readiness guard with tests for fresh/unconfigured/disconnected/placeholder credentials, configured manual key/token, supported cookie/hosted/runtime transports, ordinary connected endpoint failure and later config/authority changes without stale entitlement reuse. Keep relevant existing tests meaningful; update fixture credentials only where those tests claim authenticated probing. (3) Focused and adjacent Vitest, scoped lint, compiler diagnostics compared against known90 baseline; run Bandit per repo policy and report TS parse limitation truthfully, not as security assurance. Retain RED/GREEN, commands, exit codes and hashes. (4) Independent review then root repeats new clean-context no401 and authenticated entitlement UI positive.

Never print credentials/private files/raw logs; no disabling tests or weakening auth. Tests should not call native model/services. Evidence .tmp/uat-repairs-231-246/capability258/. Report .superpowers/sdd/IMPLEMENTATION_PLAN_fresh_matrix_231_246_repairs/task-258-report.md. Maximum three failed repair attempts then reassess. Full matrix remains gated.

- 23:57 UTC: original Retry232/256 now returns real200 and ORBIT-742 with new canonical pair; normal reload returns7rows. Independent native review active. WorldBook255 create201/catalogue200 and normalreload pass; Character editor239 requests include_disabled=true and returns the populated book200. New260/task202 records timezone-less timestamps rendered in7hours. Root continues236 normalChatpicker→Character variant.

- 2026-09-18 00:01 UTC: new261/TASK13260.203 records TestBot exact-response deviation after healthy model selection. Read-only instruction-precedence diagnosis precedes any product change; no provider tuning or answer hardcoding. New260/TASK13260.202 remains a separate WorldBook timestamp boundary repair. Native232/256 passes initial14 plus supplemental14 independent checks; retention is pending. Native239/255 is under independent review.
- Task258 fix round1: use actual API-client cookie eligibility, including origin and auth mode, instead of trusting authSource alone. Preserve valid cookie/hosted/manual/token/runtime transports and cache boundaries. Two actual-client negative controls reproduce the review gap; the valid cookie control passes. Sole author remains258 until its review correction is complete.

## Tasks 257/259: restore authenticated content scope before permission-first database work

**Backlog:** TASK13260.199 and TASK13260.201. **Status:** Prepared; implementation waits for258 review correction. **Baseline:**2787043410.

**Proven boundary:** `.tmp/uat-repairs-231-246/trash259-diagnosis/REVIEW.md` proves that the configured single-user key branch of `core/AuthNZ/auth_principal_resolver.py` caches an authenticated principal/user without activating content scope. The cached `get_request_user` return in `core/AuthNZ/User_DB_Handling.py` likewise leaves scope unset. User-first authentication activates owner1; principal-first leaves PostgreSQL app.user_id empty and hides the same original owned row. Both Media DELETE and RAG stream declare permission-first dependencies. Initial257 direct-context probes bypass this defect and are controls only, not causal regression coverage.

**Design:** Activate the existing content authorization context from the already authenticated identity at the canonical resolver/cached-user boundary. Reuse existing claim/user normalization and admin semantics; preserve org/team memberships, active selectors, session-role behavior and request isolation. Prefer the smallest shared correction over endpoint-specific scope patches or changing SQL/RLS. No schema, role, permission, retrieval threshold, provider, timeout or native data changes. Do not add a broad auth redesign. If evidence points to a different necessary boundary, report it before expanding production scope.

**Owned source:** `tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py`, `User_DB_Handling.py`, and a narrowly necessary existing auth helper if justified; focused tests under AuthNZ/AuthNZ_Unit and the existing untracked RAG probe file. Root alone owns Git, Backlog, docs, runtime/model/browser, frozen archives, profiles, and native data. No subagents or commits by author. One implementation author at a time.

**Steps:**
1. Reproduce causal RED through actual FastAPI dependency resolution using official restricted PostgreSQL fixtures (NOSUPERUSER/NOBYPASSRLS) and SQLite controls. Exercise actual Media DELETE/restore and RAG stream database/retrieval path, without overriding authentication or pre-seeding request content scope. Test fixture data can be created in explicit temporary setup scope that is reset before request. Controlled external generation/embeddings are allowed; mocking the broken auth or owned lookup is not.
2. Make the narrow authenticated-scope repair. Cover principal-first/user-first, cached authenticated user/principal with absent and prior unrelated scope, configured key/bearer compatibility and actual cookie paths, explicit invalid-header precedence, claim-derived membership/admin state, ordinary multi-user JWT/API-key isolation, and stream/task propagation/cleanup where affected. Use existing tests/fixtures rather than inventing a new framework. Preserve other-owner denial and legitimate no-match behavior. Verify same-owner retrieval, deletion and restoration against actual database behavior.
3. Run focused plus relevant adjacent actual SQLite/PostgreSQL tests with zero skips using the mandatory official runner and escalated fixture-network access. Run scoped .venv Ruff/compile/Bandit, compare pre-existing findings and distinguish test B101. Retain causal RED, final GREEN, commands, exit codes, source hashes and limits. Maximum3 failed attempts per issue, then reassess and record.
4. Independent spec/code review follows before commit. Root explicitly upgrades immutable native copies and repeats original Rowan QA and Delete/Trash/Restore, preserving all original failures and data. Neither finding closes merely because direct helper tests pass.

**Runner:** `source .venv/bin/activate && TLDW_UAT_EVIDENCE_LABEL=authscope257259-UNIQUE node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs TESTS -q --tb=short`. Use require_escalated for local fixture network. Do not skip PostgreSQL, substitute a DB or roll a new cluster. Never print private credentials/configs/raw runtime logs or provider reasoning. Store safe results in `.tmp/uat-repairs-231-246/authscope257259/` and full report in this plan's SDD workspace.

**Success:** Actual permission-first endpoint regressions fail before and pass after the narrow repair; owned source retrieval and Trash lifecycle work with restricted PostgreSQL and SQLite controls; unrelated identities remain denied; no new static/security findings; independent review and original native acceptance both clear.

## Task 260: preserve World Book timestamp chronology

**Backlog:** TASK13260.202. **Status:** Prepared, no active author. Native PostgreSQL WorldBook1 response emits naive `2026-09-17T23:51:02.389993`; America/Los_Angeles list/detail renders in7hours. Actual frontend formatter reproduction confirms a25200000ms shift; explicitZ and equivalent-07:00 values agree and show a few seconds ago. `.tmp/uat-repairs-231-246/worldbook260-diagnosis/frontend-projection.json` is a source-hashed read-only control, not new API/database acceptance.

**Requirement/design boundary:** establish the WorldBook timestamp storage/serialization contract before correcting naive timestamps. Prefer a small correct boundary normalization using existing helpers, preserving explicit offsets, invalid/null behavior, numeric/Date frontend inputs and stored chronology. WorldBookService tables currently use TIMESTAMP DEFAULT CURRENT_TIMESTAMP; PostgreSQL session timezone must not be silently assumed for arbitrary legacy values. Avoid a global date parser or unrelated API rewrite. Current evidence establishes browser-local interpretation of a timestamp produced in UTC by this fixture; it does not alone establish every database deployment's timezone.

**Owned candidate scope:** `api/v1/schemas/world_book_schemas.py`, narrow WorldBook manager/response serialization if necessary, or `apps/packages/ui/src/components/Option/WorldBooks/worldBookListUtils.ts` with focused tests, chosen after causal investigation. Preserve existing255 transaction/readback and239 catalogue behavior. Root owns all Git/Backlog/docs/native/runtime/model/profile/archive operations. One implementation author; this waits for257/259 author completion/pause.

**Steps:** use official actual PostgreSQL and SQLite create/read/list fixtures to retain native-equivalent response and a non-UTC formatter/browser control; write causal failing tests; repair only proven timestamp contract; test explicit offsets, relevant legacy inputs, null/invalid and inherited WorldBook response types. Run focused/adjacent tests, Python Ruff/compile/Bandit for touched Python and frontend lint/compiler comparison when touched. Independent review then native creation/read/reload in original local timezone. Do not change system/browser timezone to conceal the bug. Task closes only after visible recent-time and canonical time evidence agree.

- 2026-09-18: UAT232/239/255/256 accepted and tasksDone after independent native and20-check retention review. Current261total/254verified/2awaiting236238/5active257258259260261. Original provider-bearing captures/private records are explicit hash-only local evidence; safe reports and check facts retained. Full matrix remains held.

## Task 262: let PostgreSQL virtual-key INSERTs pass the existing SQL guard

**Backlog:** TASK13260.204. **Status:** Prepared. The adjacent `test_api_key_scopes_org_and_team_membership` fails before authentication in virtual-key creation, both current source and278 baseline auth modules. Runtime repo log identifies `Profile-visible AuthNZ users write rejected`; source-query classification rejects both JSONB and text INSERT variants before database I/O. Installed sqlglot tokenization treats adjacent positional bind tokens as a dollar-quoted delimiter. A read-only transformation inserting spaces after commas makes both actual source queries classify as unprotected writes; no source changed during this probe.

**Owned scope:** `tldw_Server_API/app/core/AuthNZ/repos/api_keys_repo.py` and narrowly necessary maintained AuthNZ repository/integration regression tests. Use a query-formatting correction (explicit whitespace separators) if the causal tests confirm it, preserving SQL parameters/values, column order, JSON casts, returned ID, transactions, mandatory audit, scope and budgets. Do not change/relax the profile users write guard, dependency version, schema or security controls. Do not broaden into authscope257/259 code. One author;258 tests-only author is complete.

**Steps:** retain existing actual PG causal RED and add focused maintained behavior coverage for the two creation paths where feasible; prove guard classification plus actual stored-key readback. Apply smallest source query formatting change. Run the original failed adjacent integration test, relevant managed SQL guard tests and virtual-key SQLite/PG controls through mandatory official runner with no skips. Run scoped Ruff/compile/Bandit and source diff; document pre-existing findings. Independent review then rerun the affected broad auth suite to close its adjacent failure. This finding arose in real integration verification; no native browser-failure claim or extra native key-management workflow is implied.

Root owns Git/Backlog/docs/runtime/nativebrowser/model/profiles/archives. Author must not modify these or spawn subagents. Evidence `.tmp/uat-repairs-231-246/virtualkey262/`; report `task-262-report.md` in this plan's SDD workspace. Activate .venv before Python. Runner `TLDW_UAT_EVIDENCE_LABEL=virtualkey262-UNIQUE node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs TESTS -q --tb=short` requires escalated local-fixture network. Never print private credentials/configs/raw runtime logs/provider reasoning. Maximum3 failed attempts, then documented reassessment.

## Task263: synchronize the Character retry branch with its owned route

**Backlog:** TASK13260.205. **Status:** Prepared. Existing Character regeneration deliberately branches partial assistant content. Native Retry created and saved branch10bf0caa while the URL stayedfc286396; normal reload restored the failed parent. Preserve intentional branching and canonical saved data; fix route identity at the existing accepted branch boundary, with account/cancellation/ordinary Chat controls. Root has preserved both canonical readbacks and all original failures. Author begins only after262 author finishes. Causal maintained test, minimal correction, scoped checks, independent review, explicit frozen upgrade and original Retry/reload acceptance are required. No new provider settings, timeout changes or response hardcoding.

-00:53UTC:258 production+maintained test review CLEAR (25independent/120adjacent);31payload reviewed packet retained with one credential-pattern candidate kept hash-only.257/259 reviewed72actualDB tests0skips; baseline attribution separates262. Full matrix remains held.

## Task264: use canonical Sources collection routes

**Backlog:** TASK13260.206. **Status:** In Progress. Authenticated WebUI source list redirects cross-origin and loses Authorization. Preserve auth/scope and canonical backend contract by correcting only client list/create collection paths with maintained causal coverage, independent review and native originalAliceSourcesreload. Brief task-264-brief.md. 262cleanup and263source are frozen;264 soleauthor begins next.

- 01:35UTC:262verified/Done committed4a5d61b383;263reviewed committedc3d5cce7aa;264reviewed committed2ff90d14ae. Original257citedQA and259Delete/Trash/Restore/canonicalreload now observed passing with full source+versionsunchanged; independent native review pending.260 sole implementation author active under contract-BRIEF.md; no global timezone/schema change. Fullmatrixheld.

- UAT258 native+retention accepted, TASK13260.200Done.256verified/6awaiting236238257259263264/2active260261. Fullmatrixheld.

## Task265: portable PostgreSQL World Book update lifecycle

TASK13260.207. OfficialPG260suite reveals unchanged rawconnectioncontext inupdate_world_book (4passed1failed). Same soleauthor owns narrow supportedtransactionrepair plus conflict/callerrollback/SQLitecontrols. Preserve260pre265diff; independentreview andnativeedit/reload required. No globalwrapper/schema/authchanges.

## Task266: portable PostgreSQL World Book attachment lifecycle

TASK13260.208. OfficialPGsuite6passed1failed exposes rawconnectioncontext inattach_to_character after265corrected. Sameauthor owns boundedtransactionfix and validation/idempotency/rollback controls. Retain failedrun and260attachedtimestampcoverage. Independentreview/nativeattachment required.

## Task267: authenticated Sources backend catalogue failure

TASK13260.209. Corrected264canonicalauthroute reaches actual500 on frozen2ff90d14ae. Root retainedoriginalAliceUI/requestevidence; read-onlydiagnosis assigned. No productioneditwhileWorldBookauthoractive. Require actualPG+SQLitecausalcontrols, preserveentitlements/ownership andoriginalAlicecatalogue200acceptance.

- UAT238/257/259native38+retention16checks CLEAR; tasks180199201Done.259verified/3awaiting236263264/5active260261265266267. Fullmatrixheld.265nativeSave500 retained.

-236remainsopen: rootrejectspicker-onlyreviewasAC3closurebecausecapturedreselection. Missingfreshsetup/noreselectioncriterionnowbeingtestedonexistingpreservedMCP251freshprofilea7. Firstchatready200, UIkeyentrydone, originalmodelselectionunchanged.

-263native26+retention16checksaccepted; TASK13260.205Done.260verified/2awaiting236264/5active260261265266267.260265266implemented/frozenunderindependentreview;267soleauthoractive.

## Task210: full World Book lifecycle compatibility

**Status:** Complete. User-approved concurrent ownership is disjoint from Sources267. Official fixture RED identifies268 PostgreSQL entry connection misuse and269 SQLite entry writes escaping caller rollback. Add independent seeded operation cases for remaining supported mutations; preserve all causal failures and narrow fixes to proven boundaries. Independent review/native acceptance required. Existing260/265/266 fixes committed0d7f2a23c9 and remain the frozen native candidate.

## Task211: legacy World Book regression harness

**Status:** Complete. TASK13260.211/UAT270 records ten failures reproduced on both0d7 and210. An implementer owns only the legacy test file, with production210 and its new lifecycle test frozen. Require faithful database doubles, unchanged behavior assertions, independent review and adjacent real-backend controls.

- UAT236 fresh-profile AC3 and independent retention are verified; TASK13260.178 Done. Current270 findings:261verified,4reviewed/native-pending,5active. Root independent210 checks pass21 actual SQLite/PG lifecycle and timestamp cases plus10permissions/negative cases. Native260265266 runs on immutable0d7 source with unchanged original profile/init/holder; no inference.

- Task210 implementation review is clear; actual database/permission checks pass and native lifecycle acceptance remains pending. Task267 implementation review likewise passes actual SQLite/PG and worker/API/cleanup checks; original Alice catalogue acceptance remains pending. Task211 is Complete:39legacytests pass with independent review and test-only scope. Current270 findings:262verified,7reviewed/native-pending,1active261.

## Task212: preserve Character editor optimistic version

**Status:** Complete. Native271 PUT422 lacks required expected_version beforeWorldBook attachment. TASK13260.212 scopes loaded-version hydration/request propagation and causal frontend regressions; preserveconflicts/ownership. OriginalAliceCharacter4 nativeedit/attachment/reload required after reviewedcommit. Sources267 andWorldBook210 are committed97036058f8/1b9ea8bc27; legacy211committedcb9cf4a4a4. Hold nexttargetedupgrade toinclude271 ifreviewready.

## Task213: truthful reciprocal WorldBook attachment view

**Status:** Complete. TASK13260.213/UAT272 preserves actualsavedCharacter6/book1 versusreciprocalfalsezero. WorldBooks Manager/detailpanel/tests ownership isdisjoint from271Characterhook/tests. Reproducecache/hydrationcause, preservelazyfetch/owner/errorsemantics, independentreview andoriginalnativebook1/Character6readback required.

- Native260/265/266accepted after30checks61inputs andexactretentionvalidation; tasks202207208Done. Native271/272remainseparatefrontendrepairs. UAT273optionalvisualauthoringcapabilitygate isTASK13260.214, waiting271freezebeforeimplementation. Model261criterionclarification ispending whilefrontendworkcontinues.

## Task214: gate optional visual metadata authoring

**Status:** Complete. TASK13260.214/UAT273 covers automatic unsupported packs requests despite a false capability. The author owns Common/VisualIdentity/VisualIdentityPackPanel and focused tests after freezing UAT271 for independent review. Require a capability-false causal test, supported and legacy controls, useful availability guidance, and original PostgreSQL Metadata expansion acceptance.

- UAT271 is under independent review. The original author observed the causal failure but did not retain its log; the reviewer is reproducing that failure through an isolated baseline-hook overlay without changing production files. UAT272 review requested truthful handling of relationship-read failures, a concrete retry action, and direct coverage of both client implementations. Previous passing receipts remain retained, not final acceptance.
- UAT261 diagnosis is preserved in `output/playwright/fresh-matrix-repairs-2026-09-17/model261-diagnosis-pending`. The original extra-introduction stream has terminal stop/DONE; missing observer bodies leave the other terminal results unknown. No application cause has been established. The finding and the model-output criterion question remain open; no provider settings or runtime behavior changed.
- UAT271 source review is clear: independent causal overlay and five focused current-source passes, 18-check audit, no new lint diagnostic. TASK13260.212 criteria 1–2 are complete; original native Character4 Save/attachment/reload remains pending. Current gate: 273 findings, 265 verified, five reviewed/native-pending, three active.

- UAT273 source review is clear after the test-only type correction. Four maintained UI and three sequencing controls pass; original Metadata expansion remains pending. UAT272 now also covers disabled links hidden by the default API filter; 30 actual SQLite/PostgreSQL read-contract checks pass, and the Manager/client correction is active. Current273 findings:265verified,6reviewed/native-pending,2active.

- UAT272 review is clear after the disabled-link follow-up:63frontend and30actual SQLite/PostgreSQL checks pass, no new lint diagnostic. All known source repairs are reviewed; seven findings await targeted native acceptance and model261 remains open. Prepare one combined committed upgrade of original PostgreSQL multi-user data for Sources, Character, WorldBook, and optional Metadata acceptance before any full fresh matrix.

## Task215: preserve canonical World Book entry identifiers

**Status:** Complete. TASK13260.215/UAT274 records native `PUT .../entries/undefined` →422 after successful book3/entry1 creation on committed270682. The canonical entry carries `id`; the editor reads `entry_id`. Trace the boundary, preserve a causal failing test using the actual wire shape, repair the narrow mapping, and cover adjacent edit/delete/selection behavior. Root owns bookkeeping and immutable runtime; the entry author owns bounded source/tests, and the native reviewer continues independent attachment checks on its exclusive browser. Preserve original books1/2 and Characters4/6. Review, commit and repeat entry lifecycle UI acceptance before closing268/269 or274. Current gate274:265verified,7native-pending,2active.

## Task216: refresh parent World Book counts after entry mutations

**Status:** Complete. TASK13260.216/UAT275 records stale parent0entries after successful create while the entry panel shows1; normal reload corrects it. After274 freezes, the same entry-file author owns a causal query-refresh regression and minimal invalidation correction. Require create/delete native summaries to update without reload and agree with canonical reload. Separate task and evidence preserve this UX defect; no concurrent edits to the shared entry manager. Gate275:265verified,7native-pending,3active.

- Tasks206/209/212/213/214 are Complete after original Alice native38checks76inputs and root98retention comparisons. Accepted packet `native-worldbook-sources264267271272273-accepted`, manifestf8d42ec77678ea579c5e67c88deb3d504d52b5a0fdb74b74f007130b0944a5a7. Gate275:270verified,2native-pending268269,3active261274275. Source remains270682 in the running original PGmulti instance; browser returned to root with preserved book3/blueentry1 and detachedCharacter7. No full matrix or inference. Continue274 then275, independently review/commit, upgrade and complete entry/book lifecycle acceptance.

- Task215/UAT274 source review is Complete; native acceptance remains pending. Final guarded identifier mapping passes independent13checks44inputs, author17focused and independent12focused tests; reviewed packet2e28dc41a94261ad60353cd18aa469a6a3ea9faa6726eff2ab0a4c21d8ee838f retained with281rootcomparisons. Task216/UAT275 is In Progress with a separate real-QueryClient summary regression before source edits. Gate275:270verified,3native-pending268269274,2active261275. Model261 isolated diagnostic capture independently replays1test; no live inference or disposition change.

- Task216/UAT275 source review is clear: independent13checks30inputs, three real-QueryClient controls pass and the exact baseline overlay reproduces two failures with its negative control passing. Author adjacent suite17passed/one unchanged retired-drawer skip; partial bulk-add is source-reviewed only. Reviewed packet manifest21b55c880f4e46b6438ec688fc89d2e957e2f5334236f1ca28e5d6b92e2e3be7 retained with74rootcomparisons. Current275 findings:270verified,4reviewed/native-pending268269274275,1active261. Original PostgreSQL combined entry/count acceptance is next; no full fresh matrix or model criterion change.

## Task217: contain catalogue table beside World Book detail

**Status:** In Progress. TASK13260.217/UAT276 retains native pointer interception and screenshot/geometry at1200×953: table extends outside its35% list column underneath detail. Keyboard menu is a workaround only. Implementer diagnoses minimal existing layout overflow correction and meaningful layout/pointer regression; independent review and original committed PostgreSQL pointer/reload acceptance required. Gate276:270verified,4native-pending268269274275,2active261276.

## Task218: truthful World Book deletion wording

**Status:** In Progress. TASK13260.218/UAT277 retains native permanent-removal text versus default soft-delete response. Root owns a two-string single/bulk dialog correction describing library removal; no endpoint/timer/undo change. Existing selection/keyboard3controls pass; independent review and committed native single/bulk Cancel checks remain required.

- Tasks210/215/216 Complete after native33checks62inputs and root85retention comparisons; packet43a483292f336032a9fda08ecf89efe51b57965e1ccf34ae05b6020c9882ae35. Current277 findings:274verified,3active261276277. Root owns returned PGmulti browser; runtimeac713a4/API84240/Next84458 unchanged.276 containment and277 copy source review pending;261 capture correction is under root review, no live calls yet.
