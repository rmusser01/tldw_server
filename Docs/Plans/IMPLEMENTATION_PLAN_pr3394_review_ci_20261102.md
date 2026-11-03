# PR3394 review and current-head CI repairs — 2026-11-02

Task: TASK-13260.278.18.83.45. User explicitly requests latest-dev rebase, all issue/comment repairs, requesting-code-review, and correction of every substantiated finding. Existing publication authority continues. Fresh GitHub metadata now reports PR3394 ready for review (isDraft=false), auto-merge unset; preserve that observed state.

## Scope and preservation

Current published candidate is62fa86ff23225bab7123ef023aeda373cf23a490 on latest checked dev576feef820e6521f021e6f57c82c24df2429b901. The bounded Linux-shard and required-gate follow-up below is local and unpublished. The162-commit rebase and89-path repair were published normally with exact lease; older4eb/d11 publication descriptions below are historical. Preserve pre-rebase branches/stash, human-authored summary/body, normal hooks, original running API/frontend/Chat/data, other WIP and untracked caches. All43 selected live acceptances retain their recorded frozen/composed scope; broader native/installer/release acceptance is separate.

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
**Status**: In Progress

## Stage3: Independent source review
**Goal**: Apply requesting-code-review to the PR and corrected candidate, not only the prior rebase delta.
**Success Criteria**: One independent read-only reviewer; no inherited chat history, no child reviewers, no edits/tests/runtime. Review changed production hunks and affected callers in passes, report coverage limits and substantiated findings. Correct and review all remaining findings before publication.
**Tests**: Actual/staged source and result-scope review; no duplicate execution by reviewer.
**Status**: In Progress

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


## 2026-11-03 current-head continuation

Published4eb507dff9 remains the actual source. New fully paginated census355check rows/4pages: latestchecknames199success16failed13skipped1neutral59queued6running; requiredbackend/frontend are red. Prior pending-only state is historical. Fifteen substantive failed-job logs were read once in memory after exacthead/run/pull_request/attempt1/workflow binding, with no proof/log files. No repeated census or unchanged passing control follows automatically.

First verified cause: getDataTable/regenerateDataTable are owned by the real Media domain methods and absent from the base prototype; transitional overlap metadata still lists both. Remove only those two stale entries, retain exact inventory/ownership assertions, validate both configured clients plus connected real-client generation guards. No runtime method or schema changes.

Backend diagnosis/repairs remain open, including size/import/lifetime ratchets and PostgreSQL migration/bootstrap fixtures. Preserve all source/security/account/storage guards and existing ceilings; do not adopt a raised baseline to silence defects. This continuation does not accept the new hosted head or requalify all43 live/native/release variants. Use existing supported environments and official isolated PostgreSQL fixtures, without browser/runtime/model actions.


## Rebased candidate qualification — 2026-11-03

2026-11-03 rebased repair checkpoint: current committed candidate d11bc383337d10e891c2ef4fb68e59ed39e564c0 on576feef remains unpublished, with pending reviewed fixes. Credential import/service-slot seam cohort72passed/4warnings/33.80pytest38.054outer natural0; independent source review CLEAR. Core-owned PDF/classification contracts and shared PostgreSQL catalog split preserve moved bodies;11 function ASTs unchanged and six async AuthNZ validators/old named exports retained. Connected259-case run completed natural1/151.63outer: selected document/moderation/catalog/answer/template functionality passed, remaining architecture gates failed; its long printed failure report was truncated, so no exhaustive whole-report/no-bug claim. Supported fresh AST inventory recovered all24 current function-size growths, class869>863, two layering edges and SearX scope metadata; no baseline increased. Normal tighten-only layering helper changed no baseline bytes. Both readiness origins were independently traced to managed/admin configuration; exact-origin SearX and local-runtime scope records now documented. Keyword defect causally reproduced:8/8 expected assertions failed on unchanged keyword production,61.16outer, official isolated/restricted PostgreSQL on retained15492/no Docker autostart. Minimal shared read-only validate_only guard triggers existing owner repair and refuses serving opens without DDL; role-aware Collections fixture invalidation corrected. Expanded full keyword/serving/collection validation41passed33warnings/99.68pytest107.927outer natural0, independent review CLEAR. Scoped static/security, remaining gate repairs and final reviews/hooks/publication are still OPEN. No original API/frontend/browser/model/data transition or fresh all43/native/release/queue/merge acceptance.

The normal fresh AST counter exposes24 additional size growths against latest dev; fix through cohesive shared extraction, not raised ceilings or formatter golf. Reviewed directions: shared first-upload filename and evidence-authority rules; fresh initial Hub registration group; fixed RAG depth presets; successful Workflow prefix recovery; Chatbook result construction; persistence-owned migration/receipt helpers and shared metadata allowance. Preserve every credential/owner/lease/cancellation/transaction/partial-progress boundary. Stage2/3/4 remain In Progress. The original published4eb/remote and human/authored body are preserved; there is no new published head yet.


## 2026-11-03 continued qualification

2026-11-03 continued PR3394 repair: rebase d11bc383 onto dev576feef remains unpublished. Qualified cohorts retain scoped evidence: DB ownership/migration73pass, keyword/serving41pass, MCP lazy composition63pass, filename/authority18pass, Workflow48pass plus corrected3pass; later source-cut cohort123pass/2fail, fixture/gate follow-up12pass. All architecture gates now pass composed from106controls plus corrected function-size check; only one stale ceiling deleted, no limit raised. New causal voice-admission and quoted-DSN regressions3fail; minimal fixes authored, live-mode/credential/migration validation running. No original runtime/browser/model/data/queue/merge actions or fresh all43/native/release acceptance. Final security/source/staged/hook/publication gates remain open.


## 2026-11-03 final local checks

2026-11-03 local qualification: latest source has no remaining local failure in covered scope. New correlated voice-admission denial and percent-decoded fixture credentials have causal regressions; real Research dispatch/lease and transition-anchored5s retry deadline retain guards. Final cohorts98pass (PG historical/catalog/repair/ownership),43pass (Auth/voice/Research/retry),10pass (Prompt Studio setup timing),6pass (connected refusal/size gates); previous qualified cohorts remain scoped. First timing reporter lost its result and is unqualified; corrected10case run147.72s does not prove the prior55m Linux abort root cause. Canonical fingerprint/path check0;85Python compile; Ruff120baseline/118current/no new; Bandit2125baseline/2114current inherited findings/0errors/no new after4narrow AST-identical pytest annotations across3files. Normal89-file precommit natural0/6.568s; wizard-only Ruff/Black hooks skipped, not whole-scope formatter acceptance. Final staged review/publication/current-head hosted qualification remain open. Original runtime/browser/models/data and all broader acceptance gates preserved.


## 2026-11-03 combined staged review

2026-11-03 final independent combined staged review CLEAR/no Critical/Important/Minor across89paths, full-index patch e943242924205c22bb866d7694abaaa7d00bc1e12ee0ab3ae66005c6e241e63a; tracked unstaged diff empty. Full transcription-auth module8passed310warnings42.66pytest51.641outer, including hosted-only target without claiming its old environment-specific cause. Source-review Stage3 complete; only this status/evidence note follows, source/gate patch identity must remain exact. Stage2/4 retain fresh published-head hosted qualification and the unexplained historical Linux abort. No queue/merge/runtime/model/browser action.


## 2026-11-03 Linux timeout investigation

2026-11-03 Linux-timeout continuation authorized: exactpublished62fa/live576/sourceclean; onefullcurrentcensus55checks/no newfailures, currentCI41687392266/attempt1/pull_request stillqueued with no assignedrunner. Read-only CI-plugin diagnostic3casespass; reset-only setup0.0845s versus freshcookie32.2357s undercProfile. Unprofiled original-call diagnostic3casespass: cookie setups16.1465/13.5094s, appreload2.9868/3.2616s, OpenAPIwarmup7.6166/7.2973s, bootstrap2.9765/1.9510s; wrappers calloriginals and allbindings restored. Exactcollection658cases/56files. Bounded design: four existing-style auth matrix partitions (b-o193, PostgreSQL/profile125, Prompt/provider/RBAC153, s-z187) in all5platform matrices, exactall-file coverage once, fourexpensivefamilies separate; no test/guard/plugin/appwarmup skip, no55m/300s increase. Add causal coverage/separation regression first, thenminimalmatrix change, affectedCI/security/reviews/hooks/exactlease. This addresses confirmed cumulative setup cost; priorLinuxabort root remains unproven until hosted newhead succeeds. No oldlogs refetched, localwhole658execution, native controls, installs/models/runtimes/browser/queue/merge action.


## Linux partition and required-gate correction

Causal partition regression fails on the old one-shard matrix and passes after the four-way partition. Full affected workflow contracts61pass; supported shard coverage887shards/5442files/no new uncovered paths. Parsed workflow equality proves only the five literal partition arrays changed; Windows3.13 inherits the sixth consumer. All658 collected cases remain assigned exactly once, with193/125/153/187 cases; original commands/plugins/markers/env/startup/test55m/300s guards stay intact. Read-only independent review CLEAR. No whole658 local execution or prior-log reread.

Remaining required boundary/tenant/quality cohort193cases:192pass/one quality failure. The explicitly unit-marked post-core coordination test was in a mixed integration file; move the complete test to existing bootstrap unit module, preserving its five collaborator probes, exact order, sequence-rewind guard, parent fixtures and same db-management-m shard. The first moved-module check8pass exposed inherited collector-only quality metadata at its new path. Replace handwritten collectors with stdlib Mock exact ordered calls; all14 original expected call names/arguments independently matched by AST, all5 patches and sequence-rewind guard retained. Other module ASTs unchanged; reviewer direction CLEAR and actual staged review remains. Quality baseline/detector untouched. Final changed-unit plus only failed quality gate2pass/6warnings59.94pytest63.927outer; the7 unchanged destination controls remain qualified from the8pass run. All193 required gates are now composed passing, not a repeated whole-cohort run. Scope baseline regeneration changes only3767to3769, conditional If AST identical toHEADparent; all28 entries retained and13tenant/security tests pass. Ruff changed3testfiles clean; Bandit386+307 baseline/current signatures unchanged/no new findings/zero errors; no production Python changed in this follow-up. Hosted62fa mainCI41687392266 stillqueued/no assigned runner at last targeted read. Historical Linux abort cause remains an evidence-limited aggregate-setup diagnosis until actual partition jobs complete.


Final local candidate is seven paths only: workflow partitions, three test modules, one line-only security coordinate and owned task/plan. No production code, test removal, new test file, dependency or runtime change. Normal hooks and final staged source review precede commit; latest dev/remote/human-summary/exact-lease guards precede publication. Fresh hosted outcomes on the resulting head remain required, so Stage2/4 remain open and no merge claim follows.
