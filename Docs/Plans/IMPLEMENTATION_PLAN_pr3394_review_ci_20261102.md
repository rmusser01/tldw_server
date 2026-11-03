# PR3394 review and current-head CI repairs — 2026-11-02

Task: TASK-13260.278.18.83.45. User explicitly requests latest-dev rebase, all issue/comment repairs, requesting-code-review, and correction of every substantiated finding. Existing publication authority continues. Fresh GitHub metadata now reports PR3394 ready for review (isDraft=false), auto-merge unset; preserve that observed state.

## Scope and preservation

Start at published dc4ea071d9c0e651f683c0f75540362c70ef8461 on codex/uat-latest-dev-20261023. Fresh remote dev remains b2b7d5d2b04256458fe6f33e02a36019e2c46795, already an ancestor; no unnecessary history rewrite is needed. Preserve all160 UAT commits plus the reviewed integration repair, existing human-written Change summary/authored PR body, normal hooks, exact publication lease, original running API/frontend/Chat/data, other WIP and untracked model/cache files. All43 selected live acceptances retain their recorded frozen/composed scope. Broader native/installer/release gates stay separate; no unchanged workflow replay or queue/merge action follows automatically.

Read current-head CI/reviews once with complete pagination; verify exact run/head/event/attempt/workflow/job bindings before one in-memory read of each new completed failed-job log. Retain diagnostic content in memory, not proof/log files. No assertion/fixture/security/CI bypass, speculative retry or tolerance increase. Reuse supported existing dependencies/tools and official PostgreSQL fixtures with isolated/restricted roles. Keep600s run/build,300s case,120s startup,180s generation and15s graceful cleanup caps where relevant. Investigate actual causes one at a time; do not rerun already-passing controls without a new affected change.

ADR assessment: no new durable architecture rule is expected for review/CI repairs. Preserve governing ADR025 gateway/override policy, ADR059 backlog-py, proposed ADR069 API contract, canonical ADR083 quota and owned ADR089 publication records and current authority/migration decisions. Reassess only if a proven correction changes a durable rule.

## Stage1: Complete intake and establish actual causes
**Goal**: Fully paginated comments/checks and exact new failure bindings; classify independent causes.
**Success Criteria**: Current source/dev/head verified, all actionable feedback read, logs consumed once in memory, duplicate gate failures separated from causal jobs.
**Tests**: Metadata/path/ownership guards; source and actual completed-log diagnosis.
**Status**: Complete

## Stage2: Repair and validate each cause
**Goal**: Correct all verified CI and review defects with minimal shared changes.
**Success Criteria**: Real causal evidence, affected normal checks, preserved original assertions and boundary guards; scoped baseline/current security and lint; no new findings left silently open.
**Tests**: Only genuinely affected failed targets and connected contracts, both configured clients where shared source changes.
**Status**: Complete

## Stage3: Independent source review
**Goal**: Apply requesting-code-review to the PR and corrected candidate, not only the prior rebase delta.
**Success Criteria**: One independent read-only reviewer; no inherited chat history, no child reviewers, no edits/tests/runtime. Review changed production hunks and affected callers in passes, report coverage limits and substantiated findings. Correct and review all remaining findings before publication.
**Tests**: Actual/staged source and result-scope review; no duplicate execution by reviewer.
**Status**: Complete

## Stage4: Publish and verify
**Goal**: Normal reviewed commit(s) and exact-lease update of existing PR3394.
**Success Criteria**: Relevant checks/hooks pass, source/dev/remote guards hold, authored body/human summary preserved exactly, remote branch/PR head read back, completed owned plan removed when all stages finish. Do not claim green hosted/native/fullrelease acceptance from local checks.
**Tests**: Committed patch identity, exact lease and final remote readback; fresh current-head hosted checks after publication.
**Status**: In Progress

## Intake checkpoint

One full published-head census:352 checks across4pages,7statuses across1page;148completed/20in-progress/184queued,24completed failures. Formal reviews0/inline comments0/issue comments2, each fully paginated; GraphQL review threads0/onecompletepage. CodeRabbit skips because647 reviewed files exceed150; Qodo is billing-blocked. These are reviewer availability notices, not code findings and not authority to change billing/config/split scope. Root reads their full comment content and actual CI causes. Independent review will cover the current PR diff (668 net paths/58,924 additions/5,144 deletions), with coverage stated honestly.

## Verified correction checkpoint

Five Important review findings have causal regressions and reviewed source corrections: retain all hash-only migration content; bind Data Tables wizard and both saves to account authority; enforce Quiz's final provider/model policy; drain actual provider workers before cancellation/retry; refuse flashcard creates after unknown bulk acknowledgements. The lost-LLM-context finding was withdrawn after independently reading the existing context-carrying default executor. Migration eligibility UI now reflects effective No even against an older server.

Evidence: migration API15pass; complete Quiz/DataTables worker modules50pass; eight connected storage/account/request/generation modules157pass in each UI/WebUI configuration. Studio/migration connected modules83pass/4 already-recorded CI seam failures; the new lost-ACK cases pass in that run. Fixture/partial-run errors were not counted as causal proof. Original runtime, data, browser, models, frontend builds and remote head remain untouched.

CI shard guard is green with11missing test files assigned to existing domain shards; no ignore/baseline expansion. The required-workflow module's output was lost during context truncation and is not claimed as qualified; its final affected run remains required. Scope predicate source comparison proves all28 current conditional nodes and enclosing functions AST-identical to dev, with12relocations; the official coordinate regeneration adds no offense. Quality ratchet/body semantics, remaining UI/Research/e2e causes, Turbopack budgets, published docs/strict links and ADR identity collisions still need correction/validation.

ADR check: no new durable rule for these safety corrections; existing ownership, gateway policy and worker-lifetime rules govern. Migration receipt hashes never constituted content recovery. Any ADR numbering repair will preserve accepted rationale and change identity/reference metadata only.


## Final local qualification checkpoint

Independent current-source review is CLEAR in the reviewed correction scope, with honest broader PR/native/release limits. All identified Important findings and subsequent Minor classifier/Important same-task test findings are corrected and reviewed. No baseline or bundle budget increased. Quality+shard42pass; Research E2E3pass; official isolated/restricted PG16pass; focused authority/provider18pass; both ChatPane38pass/client; prior157 storage/account cases/client and Web title18pass plus normal runner/controller45pass remain qualified with their scopes. Docs/strict build and predicate tests passed within124pass/one shard failure, then the updated shard contract passed. Python24compile and baseline/current static scopes have no new findings after narrow pytest annotations/import cleanup; full TypeScript0.

Actual production Turbopack build89.082s/155pages/shared541.9KB against604/heaviest596.0KB against900 passed with tracked source unchanged. Advanced-profile preflight failed before compiler startup because the operator omitted required NEXT_PUBLIC_API_URL; supply its normal loopback setting for that distinct failed-CI profile. Original runtime/browser/models/data are untouched. Dev remains exactb2 and remote PR head exactdc4; PR is now ready for review, not draft. Remaining: advanced profile, normal hooks, final staged source review and exact-lease publication/current-head hosted outcomes.


## Packaging and publication checkpoint

Both actual Turbopack profiles pass unchanged budgets: production89.082s/shared541.9KB/604/heaviest596.0KB/900; advanced68.921s/shared538.9KB/604/heaviest592.9KB/900.155pages/profile, token sync0, tracked source unchanged during each build. The advanced preflight-only operator failure was corrected with its required normal absolute browser API URL, without relaxing configuration validation. No original runtime transition occurred.

Final TypeScript0,24Python compile, all touched production/test lint and baseline/current Bandit have no new findings. Inherited diagnostics remain scoped; formatter changes preserve whole-module ASTs and the test-window type change emits identical JavaScript. Source review CLEAR/no remaining Critical/Important/Minor within the stated correction coverage. Current new-head hosted CI and broader native/installer/release acceptance remain separate. Stage3 awaits final staged review; Stage4 awaits normal hooks, exact-lease publication/body preservation and fresh hosted outcomes. Retain this owned plan while those stages remain open.


## Final staged source review

Independent staged review CLEAR/no Critical/Important/Minor: binary patch6c96c9564647027b7d55083fcb0c600eb51a53d1742a459c2ec99dea101ce329,72files1586insertions3164deletions; tracked unstaged diff empty. Four ADR renames and identical canonical/published blobs verified; no ordinary test/gate removal; quality baseline and bundle budgets unchanged. Normal configured pre-commit hooks passed on the unit. Only this authorized status/prose closeout follows that review; source patch identity is retained. Stage4 remains In Progress until exact-lease publication and fresh hosted outcomes. No all43freshlive/native/release/merge certification follows from these local results.
