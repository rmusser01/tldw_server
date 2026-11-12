# C02: Preserve saved evaluation provenance in list and detail

Associated task: TASK-13260.278.18.83.45. Root recorded and read back the actual saved-evaluation defect before edits. The existing repository already retains the saved name, case/run IDs, configuration and metrics. The list discards those fields; the detail UI reads only one legacy model configuration. This unit reads existing records and never evaluates or rewrites them.

## ADR assessment

No new ADR is required. [ADR-055](../ADR/055-prompt-studio-db-single-implementation.md) governs the existing backend-neutral repository, and [ADR-020](../ADR/020-db-management-per-user-paths-and-content-backend.md) governs per-user databases and explicit shared PostgreSQL. Both were read for the preceding immutable-history unit and still govern this read-only projection. Preserve producers, schema, storage, scoring and immutable snapshots. Reuse the current GET-detail public configuration whitelist for both list and detail, retaining every recorded configuration and legacy aliases without exposing arbitrary execution configuration or secrets. The source-confirmed list omission of the existing project-access and optional expected-user guards is corrected only at that read boundary, as root approved.

## Stage 1: Establish saved projection and display contracts
**Goal**: Real API regressions and mounted normal UI regressions for the actual lost fields.
**Success Criteria**: Saved name/description, IDs, all public configs, zero metrics and missing metadata have precise oracles; real owner and captured-principal refusals precede listing.
**Tests**: Official Alice/Bob fixture, actual DB create/update abstractions, list/detail GETs; real AntD/QueryClient and actual service at external apiSend seam. No provider requests or auth overrides.
**Status**: Complete

## Stage 2: Reuse canonical projection and show honest provenance
**Goal**: Minimal changes to the existing evaluation endpoint, table, detail panel and shared DTO types.
**Success Criteria**: List/detail share the same public projection; canonical configs all render, supported legacy aliases remain; absent case count is unavailable while an actual empty list is zero. Finite zero metrics remain visible. Existing creation, budgets, scorer and storage bytes remain unchanged.
**Tests**: New causal tests plus genuinely affected pagination/error-mapping/design-system contracts; no replay of passed history cases.
**Status**: Complete

## Stage 3: Qualify, review and freeze
**Goal**: Accurate configured test results, scoped static baseline/current checks and independent root review.
**Success Criteria**: Natural exits and explicit bounded attempt ledger; no new security/lint findings; pre-unit bytes remain only in memory and exact hashes are reported directly. Root owns global types/build/runtime and actual saved-result acceptance.
**Tests**: Activated supported Python 3.12 pytest with 600s outer/300s case; Node 20.19.5 normal UI/Web configs; compile, Ruff, Bandit and configured ESLint/syntax.
**Status**: In Progress

## Attempt ledger

- Four existing production paths are captured exactly in memory from the frozen history candidate. No generated baseline or proof files.
- Backend causal attempt 1 naturally exited 1 in 61.050s: seven failed and three real controls passed. Six cases substantiate dropped list provenance or missing owner/expected-user guards. The missing-metadata case instead revealed fixture setup: the actual create writer defaults absent case IDs to an empty list; seed historical NULL through the supported update writer before testing absence. No producer change or red replay.
- UI causal attempt 1 naturally exited 1 in 6.124s: five failed and three genuine controls passed. Canonical model lists/aliases, missing values and legacy metrics are not displayed; null f1 also causes a real `.toFixed` TypeError. Native QueryClient/AntD and actual service run at the external apiSend seam; no provider/scorer calls.
- Backend green attempt 2 selected only the seven prior failures: six passed; the missing-case fixture failed because the canonical schema requires `test_case_ids` JSON NOT NULL. The supported create writer records absent input as an actual empty list, and update-to-SQL-NULL cannot represent it. No production defect, schema change, or raw SQL is inferred from this setup failure; root is assessing the truthful fixture representation.
- UI green attempt 2 selected only the five prior failures: four passed; the remaining legacy-metrics assertion missed AntD Statistic's split integer/decimal markup despite a displayed 25.0%. The final failed-ID-only fixture uses the actual Accuracy Statistic text content and keeps the exact 25.0% value oracle. Prior three controls and four new green bodies are retained without replay.
- UI attempt 3 stopped with only the same selected legacy-metrics case: the synchronous Accuracy lookup preceded the detail GET response. Opening a Drawer/title is not an oracle for completed data. No source defect or fourth execution is inferred; the exact failed prerequisite is reassessed against existing mounted detail tests that await returned content.
- Root approved the backend NOT NULL representation: keep the persisted empty case list and assert it, retaining None assertions for genuinely optional configs/metrics. The mounted missing-count UI control remains separate. Existing pagination and two manager-error fixtures now provide an ordinary matching owner project to the real access guard; all original paging/error assertions remain.
- Scoped Bandit found no production finding before/current. Two synthetic metric literals in the new test triggered B105's lexical heuristic; precise test-only annotations identify metrics and leave the AST/assertions unchanged. Configured ESLint reports 33 inherited production warnings, no errors and no new-test warnings.
- Root recorded a separate fixture reassessment before the one remaining UI prerequisite edit: await actual Accuracy from the detail response, then assert unchanged Statistic 25.0% text. Existing mounted TestRunHistory line115 and OptimizationProgressPanel.i18n line107 wait for returned content before checking dependent details. Execute only the failed UI ID, then the first affected Web scope; retain every previously passing case.

## Frozen qualification scope

- Backend: ten distinct new real-auth API cases have clean passes across retained three controls, six selected green bodies, and the sole corrected missing-metadata case. Three affected existing offset/database-error/unexpected-error contracts passed with that final case in one natural exit 0, 14.343s (pytest 9.60s). No combined ten-case green execution is claimed, and no history/provider/scorer suite was replayed.
- UI: eight distinct new mounted cases have clean passes across retained three controls, four selected green bodies, and the approved sole async-fixture recovery (natural exit 0, 1.781s; seven unselected). The affected existing detail-status case separately passed (natural exit 0, 0.714s; unrelated wizard case unselected). No combined nine-case UI execution is claimed.
- Web: first configured affected scope passed nine cases in one natural exit 0, 3.061s: all eight new cases plus existing detail status. The unrelated wizard case was explicitly unselected.
- Python compile: four touched Python files pass. Ruff baseline/current is zero for existing endpoint and two legacy test files; new API test is zero after import-only formatting. Bandit production baseline/current and new test are zero; the two legacy test files retain exactly 176 and 34 inherited B101 assertion findings, with zero new findings by rule/source-line comparison.
- Configured ESLint: existing production warning counts are unchanged (table five, detail four, service twenty-four); zero errors or new-test warnings. TypeScript syntax parses all four touched TS/TSX files. Root owns full configured types/build and actual saved-evaluation acceptance; this source is not loaded or live accepted by these tests.
- Self-review covers every new API/mounted body and all changed projection/display functions, actual repository/manager/schema/auth guards, aliases and omitted secret fields. Producer/scorer/storage/migration and unrelated creation/budget code are preserved. Independent root review remains pending.
