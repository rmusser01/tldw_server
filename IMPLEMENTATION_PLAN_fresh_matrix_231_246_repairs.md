# Fresh matrix findings231–246 implementation plan

> **For agentic workers:** Use the existing coordinated implementation/review workflow. Do not start new full UAT before the repair gate. Root owns shared tracker, Backlog, runtimes, native browser and commits.

**Goal:** Resolve every new finding from the frozen four-configuration matrix and verify its original workflow before another full UAT.
**Architecture:** Make bounded corrections in the existing schema, authorization, proxy, UI state and scheduling paths. Keep independent work in disjoint files and review each unit before integration.
**Tech stack:** FastAPI/Python, SQLite/PostgreSQL, Next16.1.4, React/TypeScript, pytest/Vitest, native Playwright CLI.
**Design:** [Repair decisions](Docs/Design/2026-09-17-fresh-matrix-repairs-231-246.md).
**Task:** TASK13260 and children173–188.

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
**Status:** Not Started

- [ ]245: extend canonical bootstrap and real quota/guard controls; retain partial-index upserts, FK/check constraints, fractional usage and fail-closed behavior.
- [ ]238/233: reproduce actual worker persistence with trusted owner in required PostgreSQL; carry scope through executor and reset it; deduplicate terminal warnings at their producer.
- [ ]239: reproduce catalogue failure through actual PostgreSQL connection wrapper and use supported read lifecycle.
- [ ]234: prove the installed rewrite deadline with a controlled upstream, set the finite budget, repeat >30s success/client-abort/upstream-failure controls.
- [ ] Review and commit each independently testable unit; leave native acceptance pending until Stage4.

## Stage3: Repair UI, study and stream findings

**Goal:** Correct231,232,235–237,240–244 and determine246's actual failure boundary.
**Success criteria:** Each original behavior has a causal regression and minimal reviewed correction;246 has a supported disposition, not an assumed timeout cause.
**Tests:** Actual component/hook/QueryClient lifetimes; provider alias conflict controls; saved ordinary-mode transition; canonical re-rate previews; real SQLite/PG Hard/Again analytics; one/multiple localization; transport first-byte/idle/abort boundaries.
**Status:** Not Started

- [ ] Study235/242: actual Due and Cram re-rate14day preview, practice-only no write, repeated rating and singular/multiple completion.
- [ ] Analytics240: actual Hard versus true lapse outcomes on both DBs, explicit legacy fallback and owner/date filters.
- [ ] Media237/241/244: missing-auth initial render, late success/rejection after disconnect, delayed selection, actual wizard completion, preserved filters and old-owner rejection.
- [ ] Chat/model231/232/236/243: correct surface copy, actionable sanitized400, qualified alias readiness/recovery and fresh ordinary-mode transition without private-state regressions.
- [ ]246: correlate first-byte delivery using the existing actual stream path and bounded provider control; test a cause before changing budgets/forwarding. Repeat original TestBot with exact visible response and canonical reload on accepted source.
- [ ] Review and commit disjoint units, recording tests and touched-scope static checks.

## Stage4: Combined verification and original native acceptance

**Goal:** Verify all repairs together before another full matrix.
**Success criteria:** Relevant tests pass, mandatory PostgreSQL0skips, no added compiler/security findings, and each finding has native original-scenario evidence or an explicit verified external disposition.
**Tests:** Combined affected frontend/backend suites; recorded90diagnostic compiler comparison; scoped lint/Bandit; exact real model workflows on declared repair profiles.
**Status:** Not Started

- [ ] Run combined tests once after integration; investigate failures rather than disable cases.
- [ ] Review the final touched diff independently.
- [ ] Repeat exact quota→ingest/source/owner paths, Biology5generation/draft/save/five-card Study, model selection/TestBot, Media delayed handoff/refresh/disconnect and all remaining issue-specific native checks.
- [ ] Hash and retain evidence; update per-issue ledger with revision, tests, native acceptance and remaining limits.
- [ ] Only release the next matrix when no identified issue remains unresolved or awaiting acceptance.

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
