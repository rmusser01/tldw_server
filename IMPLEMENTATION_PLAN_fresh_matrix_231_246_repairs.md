# Fresh matrix findings231–246 implementation plan

> **For agentic workers:** Use the existing coordinated implementation/review workflow. Do not start new full UAT before the repair gate. Root owns shared tracker, Backlog, runtimes, native browser and commits.

**Goal:** Resolve every new finding from the frozen four-configuration matrix, including follow-up findings discovered during repair (now254), and verify its original workflow before another full UAT.
**Architecture:** Make bounded corrections in the existing schema, authorization, proxy, UI state and scheduling paths. Keep independent work in disjoint files and review each unit before integration.
**Tech stack:** FastAPI/Python, SQLite/PostgreSQL, Next16.1.4, React/TypeScript, pytest/Vitest, native Playwright CLI.
**Design:** [Repair decisions](Docs/Design/2026-09-17-fresh-matrix-repairs-231-246.md).
**Task:** TASK13260 and children173–196.

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
