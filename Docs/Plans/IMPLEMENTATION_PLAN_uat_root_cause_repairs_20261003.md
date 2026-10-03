# Resumed UAT root-cause repairs

Task: TASK13260.281. Human authorized one repair PR for all recorded bugs before continuing UAT.

## Stage 1: Trace and reproduce
**Goal:** Identify the current source cause for each of the fifteen bug rows.
**Success Criteria:** Causal source regressions fail before correction; the small output-color repair uses the original observed failure and source inspection. Upstream corrections are distinguished.
**Tests:** Existing actual service/component/SQLite test suites; no replay of failed live fixtures.
**Status:** Complete

## Stage 2: Correct and review
**Goal:** Apply narrow corrections in the current-dev repair worktree.
**Success Criteria:** Affected tests, formatting/lint/type checks, applicable Bandit, and independent source review; no auth/ownership or budget weakening.
**Tests:** Domain regressions plus existing adjacent suites.
**Status:** Complete

- TASK13260.280: Kanban list envelopes, item hydration and canonical mutation fields; ported correction passes14 tests on current dev.
- TASK13260.281.1: Character backup image serialization/atomic failure, plaintext preservation, Prompt FTS failure boundary, supported Character generation metadata.
- TASK13260.281.2: Notes selection/save identity and export tags, Media permalink hydration, unsaved connection edit guidance.
- TASK13260.281.3: Chat semantic scope validation, Research workspace identity and retrieval-only preflight.
- TASK13260.281.4: Character single-envelope exports, synthesis readiness and themed repository output.

## Stage 3: Publish and reconcile
**Goal:** Keep one draft PR current with verified repairs and remaining blockers.
**Success Criteria:** No recorded bug silently omitted; corrected runtime/live UAT, native app lifetime and installer/consumer gates remain separate until actually verified.
**Tests:** Combined affected checks and independent actual-source review before readiness.
**Status:** In Progress

PR3084 remains the separate published app-lifetime correction. First-import/native and installer53/55/consumer56 causes are not accepted by these frontend/backend source checks. No repeated negative GC/native controls, speculative repairs or changes to CI budgets/cache/dependencies.

Browser recovery is pending from the prior human question. UAT stays on hold under the new instruction until the repair work is addressed. No new browser/API/profile/model actions are authorized by publication. Keep concise bug/result notes; no evidence bundles.

Final source review is clear. Affected checks pass342 distinct backend and241 frontend cases; Python Bandit0findings/0errors. Full frontend types remain blocked by inherited unrelated dependency/React diagnostics, with0touched-path diagnostics. Stage3 awaits final push and PR readback; live UAT/native/installer gates remain open.
