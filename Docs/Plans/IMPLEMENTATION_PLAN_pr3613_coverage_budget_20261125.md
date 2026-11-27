# PR3613 coverage budget implementation plan

> **For agentic workers:** Use the executing-plans workflow in this existing isolated worktree; request independent actual and final-staged reviews. Steps use checkbox syntax.

**Goal:** Give the unchanged global and AuthNZ coverage executions independent budgets, with both required for the existing protected result.

**Architecture:** Add global-coverage and authnz-coverage jobs after the existing admission and changes jobs. Move each existing coverage step verbatim with its existing checkout and dependency setup. coverage-required keeps its context and refusal/no-op behavior, waits for both children, and fails on any child result other than success when coverage is required.

**Tech Stack:** GitHub Actions YAML, Bash, pytest and existing PyYAML/expression-test helpers.

**Spec:** Requester-approved in-chat design on 2026-11-25; TASK-15637 records provenance and approval. Separate runners isolate coverage databases and fixture runtime files. Independent AuthNZ runtime remains unverified.

## Global constraints

- Keep both execution jobs at 30 minutes; retain AuthNZ per-test timeout 120 seconds.
- Keep complete command/env/coverage target/selections/floors verbatim: global 12, AuthNZ 35.
- Preserve triggers, permissions, concurrency, checkout head, license-first admission and change detection.
- Preserve the protected coverage-required name and queue-tick dependency; unexpected child skips, cancellations and failures must fail the protected gate.
- No application, fixture, dependency, suppression, baseline or accepted ADR rationale change.
- ADR required: no. Existing required-gate result semantics remain; ADR-059/084 govern tracking and ADR-063 describes the existing queue.

## Stage 1: Pin behavior
**Goal**: Write failing contracts for independent budgets and complete result enforcement.
**Success Criteria**: Existing serial workflow fails the new tests for missing execution jobs and result enforcement; command hashes pin the original steps.
**Tests**: New coverage execution/result cases in test_merge_queue_gate_workflows.py plus existing required/license workflow contracts.
**Status**: Complete

- [x] Add exact step-digest preservation, independent-job and dependency assertions.
- [x] Execute the actual protected-result shell against success/failure/cancelled/skipped/empty outcomes; only two successes pass.
- [x] Evaluate execution-job admission across existing event/license/change/cancellation truth-table rows and relevant/irrelevant paths.
- [x] Run the new tests under activated APPENV and retain the intended RED.

## Stage 2: Split executions
**Goal**: Move the unchanged executions to isolated jobs and make the existing protected result require both.
**Success Criteria**: New behavioral tests pass; existing refusal/no-op semantics and all original workload bytes remain.
**Tests**: Same RED tests GREEN, existing required workflow contracts and complete license/queue tests.
**Status**: Complete

- [x] Change .github/workflows/coverage-required.yml only for job placement, dependencies and the fail-closed rollup.
- [x] Update existing workflow-surface/step-location contracts narrowly in test_required_workflow_contracts.py and test_license_first_workflow_contracts.py.
- [x] Verify global/AuthNZ step digests still equal 14996dc3ac2432f01223fc7d1df48a0841f228263e5086d53b260edd6dac5178 and f9e3f60c9777835ee1ff663d370509a2c3b9ca3f2083908e8bafd91bbdae49f1.

## Stage 3: Verify and review
**Goal**: Verify affected CI behavior and independently review the final candidate.
**Success Criteria**: Affected tests, syntax, actionlint, scoped baseline/current Ruff/Bandit and actual independent review pass with no new runtime/security diagnostic.
**Tests**: All affected CI contract modules, actual shell result controls, existing cached actionlint and task/ID contracts.
**Status**: Complete

- [x] Run affected checks once, correct causal failures and record inherited skips without promotion.
- [x] Compare static diagnostics on touched Python against exact committed blobs; no baseline/suppression writes.
- [x] Request actual and final-staged independent review of the exact patch.

## Stage 4: Publish and land
**Goal**: Publish normally to PR3613 and verify actual protected landing.
**Success Criteria**: Necessary final integration preserves owned work/stashes, configured cached checks pass, normal commit and exact lease push succeed, and fresh required/admitted actual executions qualify before protected merge.
**Tests**: Source/index/reviewed-patch/body/remote/criteria guards; one complete census per foreground interval; actual job/step bindings; independent merge-parent/current-dev ancestry verification.
**Status**: In Progress

- [ ] Perform only necessary final dev integration, normal cached commit checks and exact lease publication to the same branch; reset old-head qualification.
- [ ] Verify both hosted coverage jobs actually finish within their own budgets and enforce their floors; no local/partial/old-head substitution.
- [ ] Verify normal exact-head merge and update only honest landing completion. PR3616 merge remains unauthorized; skipped/native paths remain unverified.


## Validation record

- Hosted RED: the shared 30-minute job cap cancelled AuthNZ after global coverage succeeded; no AuthNZ finish summary or complete coverage result. Log consumed once and closed, never refetch.
- New independent-budget/result tests RED on the original missing global job, then 41 PASS. A separate cancelled-context regression RED on the inherited !cancelled() parent guard. Coverage-only always() plus an explicit failure-only cancellation refusal now prevents a skipped protected result; both execution jobs retain cancellation guards.
- Original global/AuthNZ coverage-step dictionaries remain exact, including env, full selections, targets, floors 12/35 and per-test 120-second limit. Admission, await, changes, queue-tick and protected name remain.
- Independent actual review CLEAR, including the narrowly pinned failure-only status-function exception. Baseline/current Ruff is 0/0 on three touched Python modules. Bandit adds pytest assertion diagnostics and two low trusted-workflow-shell test warnings, independently reviewed as intentional test execution (fixed absolute /bin/bash, fixed result values, no shell=True or untrusted interpolation). No suppression or baseline write.
- Existing cached actionlint source build (devel, Go1.26.2) passes the touched workflow; hosted pinned actionlint and independent AuthNZ runtime remain unverified.

- Final authored-scope verification: 429 affected CI cases pass (the 45 new cases are subsets), three Python 3.12 syntax checks pass, Ruff baseline/current 0/0. Bandit adds 22 pytest-assertion findings and two reviewed low test-shell warnings; other codes unchanged. The actual test-quality ratchet reports no enforceable baseline or current flags on the three touched modules.
- Necessary final integration onto dev `941c7f9` completed normally. All four published commits remain range equal, their 93,723-byte patch is exact, all 57 incoming paths match pinned dev, seven working-file hashes were restored, and the committed tree matches the independent clean preview. Three protected stashes and backup remain individually exact. Incoming review is CLEAR.
- Bounded incoming verification: SQLite 46 passed, Auth unit 94 passed, Chat/executor/rag_split 323 passed and one skipped; frontend identity/Notes 200 passed in nine modules, WebUI login/runtime/session 147 passed in three modules. First WebUI attempt failed during installed-package ESM setup before any test; correcting private tool resolution produced the same 147 passes without source or assertion changes. Incoming 31 Python files parse; their exact-dev static scan records inherited Bandit 1,023 findings (zero high/seven medium) and Ruff 65 findings. No authored suppression or baseline edit; the upstream Ruff change applies only to the Chat facade. Actual combined canonical OpenAPI fingerprint and path-union checks pass without exports.
- Cookie validation limits: the 69-case aggregate and two ten-case batches reached the unchanged local 600-second outer cap without final summaries; no partial passes are counted. Following the repository three-attempt rule, broad cookie retries stopped. Three successful ten-case groups plus an already-running five-case experiment validate 35 unique cookie cases, including the only newly added incoming install-isolation regression. Exactly 20 remaining legacy cookie cases stay unverified; the full cookie module and original aggregate are not qualified. The separate three login cases pass. Eleven PostgreSQL cap cases skip through official fixtures with the approved port closed and Docker autostart disabled; these remain unverified. Independent review accepts this bounded local selection; no fixture/assertion/timeout changes, native signals or instrumentation were used.
- Hosted qualification resets on publication. These local results do not establish independent hosted AuthNZ completion, full native health, skipped-path coverage or whole-product acceptance.
