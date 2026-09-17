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
