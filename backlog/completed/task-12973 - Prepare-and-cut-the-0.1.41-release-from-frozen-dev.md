---
id: TASK-12973
title: Prepare and cut the 0.1.41 release from frozen dev
status: Done
labels:
- release
- documentation
- operations
priority: High
references:
- origin/dev@4c2ad2070ed63992dac8a97a6c4cf3c7d75f6de8
- https://github.com/rmusser01/tldw_server/pull/2744
- Docs/Development/Release_Process.md
- Docs/Release_Checklist.md
- https://github.com/rmusser01/tldw_server/pull/2745
- https://github.com/rmusser01/tldw_server/pull/2748
documentation:
- Docs/superpowers/specs/2026-07-15-release-0.1.41-design.md
- Docs/superpowers/plans/2026-07-15-release-0.1.41-implementation-plan.md
modified_files:
- CHANGELOG.md
- README.md
- pyproject.toml
- tldw_Server_API/app/main.py
- Docs/mkdocs.yml
- Docs/RELEASE_NOTES.md
- Docs/Published/RELEASE_NOTES.md
- Docs/superpowers/specs/2026-07-15-release-0.1.41-design.md
- Docs/superpowers/plans/2026-07-15-release-0.1.41-implementation-plan.md
- backlog/tasks/task-12973 - Prepare-and-cut-the-0.1.41-release-from-frozen-dev.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare release 0.1.41 from frozen origin/dev commit 4c2ad2070ed63992dac8a97a6c4cf3c7d75f6de8 (through PR #2744), excluding all open PRs. Update authoritative version metadata, CHANGELOG.md, README.md, and release-note entry points; replace the inherited MkDocs warning baseline with deterministic canonical-to-Published generation and zero-warning strict validation in both the checked-in tree and CI-equivalent refresh pipeline; verify the expanded release diff and required gates; merge the reviewed release branch into main; and sync main back into dev. Per the requester's 2026-07-16 instruction, do not create an annotated v0.1.41 tag or GitHub Release. Preserve the user's dirty primary checkout by working only in isolated worktrees. Governing plans: Docs/superpowers/plans/2026-07-15-release-0.1.41-implementation-plan.md and Docs/superpowers/plans/2026-07-16-release-0.1.41-zero-warning-docs-implementation-plan.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Release scope is frozen at origin/dev 4c2ad2070ed63992dac8a97a6c4cf3c7d75f6de8 and open PRs are excluded
- [x] #2 Authoritative project, FastAPI, MkDocs, README, changelog, and release-note version surfaces consistently report 0.1.41
- [x] #3 CHANGELOG.md contains a curated 0.1.41 rollup for merged PRs included after 0.1.40 through PR #2744
- [x] #4 README.md current status and What's New accurately summarize 0.1.41
- [x] #5 Focused release metadata/docs tests, diff checks, and Bandit policy are satisfied
- [x] #6 Release PR into main is merged only after required checks are green
- [x] #7 No annotated v0.1.41 tag or GitHub Release is created, per the requester's 2026-07-16 instruction
- [x] #8 Released main history is synchronized back into dev; tag-triggered artifact workflows are intentionally not applicable
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-15-release-0.1.41-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Preflight (2026-07-15): release source is frozen at 4c2ad2070ed63992dac8a97a6c4cf3c7d75f6de8 through PR #2744. Refreshed origin refs without merging or rebasing: HEAD=7c2c7d07e5396fdaf9f4e0dd4d9c9076e8f22e8d, origin/dev=4c2ad2070ed63992dac8a97a6c4cf3c7d75f6de8, origin/main=7273cca4926abaa242a682f558fe1d3173f230e7. The frozen SHA is an ancestor of HEAD (git merge-base --is-ancestor exit 0) and the exact merge-base is 4c2ad2070ed63992dac8a97a6c4cf3c7d75f6de8. Post-freeze first-parent history contains only reviewed release design/gate/plan commits 6e1fc05b637933d6fb279ae2222c0e374c42d43a, 8841419f6965a488415fcf00da82bdf75ebd82cf, and 7c2c7d07e5396fdaf9f4e0dd4d9c9076e8f22e8d; the post-freeze first-parent merge query is empty. The frozen inventory from 7273cca4926abaa242a682f558fe1d3173f230e7 through the frozen SHA is exactly 37 first-parent merge commits and ends at PR #2744. v0.1.41 was absent at check time: origin tag query was empty; GitHub Release lookup returned "release not found" (exit 1); PyPI returned HTTP 404 (status-only request 404, HTTP/1.1 fail-mode exit 22; the environment's default HTTP fail-mode command also surfaced the 404 with curl exit 56). Open PRs and all post-freeze PRs/commits are excluded.
Scope clarification: Open PRs and all post-freeze `dev` PRs/commits are excluded; the reviewed 0.1.41 release design, gate, plan, and preflight evidence commits are intentionally present on the isolated release branch.
Release CI capacity focus rule (user-authorized, 2026-07-15): During Task 4 only, after the 0.1.41 release PR exists and immediately before its CI wait, resolve the release PR number and exact headRefOid with `gh pr view`; enumerate all repository GitHub Actions runs in `queued` or `in_progress` status with pagination; preserve every run whose `head_sha` equals that release head SHA or whose `pull_requests` includes the release PR number; and cancel every other active run. Print and record all candidate decisions plus cancelled run IDs, names, and URLs, then re-enumerate and prove only release-PR runs remain active. Repeat the complete sweep immediately before merge because unrelated runs may start during the wait. Never cancel a release-PR run; never touch completed runs. The policy ends when the release PR merges and does not authorize cancelling the post-merge main snapshot, tag, Docker release, PyPI, or any other publication workflow. The executable fail-closed sweep is documented in `Docs/superpowers/plans/2026-07-15-release-0.1.41-implementation-plan.md`. No Actions cancellation was executed while recording this rule.
Task 2.5 CI-focus quality follow-up (2026-07-15): The documented sweep now activates `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate` before invoking Python and treats the initially resolved release PR number/headRefOid as immutable. It re-resolves and asserts that exact expected PR remains OPEN on `main` before each cancellation, after every cancellation batch, and immediately before emitting `PROOF`; close, merge, or head movement therefore fails closed instead of producing a stale proof. No GitHub Actions cancellation was executed during this documentation fix.
2026-07-16 requester-approved scope expansion: eliminate the entire MkDocs warning baseline before cutting v0.1.41. Acceptance is zero warnings for both the checked-in `Docs/Published` snapshot and the CI-equivalent `Helper_Scripts/refresh_docs_published.sh` followed by `mkdocs build --strict -f Docs/mkdocs.yml`. Warning suppression and indiscriminate public-boundary expansion are rejected. Approved and independently reviewed design: `Docs/superpowers/specs/2026-07-16-release-0.1.41-zero-warning-docs-design.md`. Supplemental implementation plan: `Docs/superpowers/plans/2026-07-16-release-0.1.41-zero-warning-docs-implementation-plan.md`; it runs deterministic publication, canonical link repair, strict CI restoration, and expanded verification before the existing release PR/merge/tag/publish/sync tasks. The main plan at `Docs/superpowers/plans/2026-07-15-release-0.1.41-implementation-plan.md` was reconciled to remove earlier seven-file-only, no-generated-output, direct-Docs/Published, and checked-in-only strict-build assumptions. Continue using the approved fail-closed release-PR CI-capacity sweep and preserve the human-authored Change summary gate.
Expanded local verification checkpoint (2026-07-16, before release PR):
- Remote preflight refreshed without merging/rebasing. origin/dev remains the frozen release source 4c2ad2070ed63992dac8a97a6c4cf3c7d75f6de8, origin/main remains 7273cca4926abaa242a682f558fe1d3173f230e7, the frozen SHA is the exact merge-base of HEAD, and there are zero post-freeze merge commits. Remote tag v0.1.41 is absent, GitHub Release lookup returns "release not found", and PyPI returns HTTP 404.
- Authoritative version checks and FastAPI AST parsing both print 0.1.41. Changelog extraction produced 2,561 characters with Added/Changed/Fixed/Removed and PR #2744; duplicate-bullet normalization reports none. Historical 0.1.40 references are confined to prior-release sections. The 0.1.41 changelog and maintainer/public release-note dates were updated to the intended 2026-07-16 main merge date; Docs/Site and Docs/Published release notes remain identical after canonical refresh.
- Expanded focused release/docs suite: 101 passed, 2 runtime warnings. Full Docs plus public/private boundary suite: 195 passed, 4 runtime warnings. Standalone public/private, speech-link, onboarding-command, and endpoint-drift scanners all pass.
- Canonical refresh ran twice with no generated diff. Refresh followed by strict MkDocs exits 0 with zero MkDocs WARNING records. A clean detached checkout at committed snapshot 262d590597, without refresh, also exited 0 with zero MkDocs warnings and preserved SHA-256 hashes for Docs/Site/index.md and Docs/Site/RELEASE_NOTES.md. The final post-evidence commit will receive the same detached proof before push.
- Security: /tmp/bandit_release_0_1_41_runtime.json has errors=0/results=0 with all rules for runtime/helper scope. /tmp/bandit_release_0_1_41_tests.json has errors=0/results=0 with only B101 skipped for pytest assertions. git diff --check passes.
- CI readiness repair: commit 18ebbae16dadff446028722881f682bf35f481f4 restores explicit shard assignments for the three frozen tests omitted from CI manifests and updates the exact contract for two historically registered Workspace auth/DB tests. Independent spec and quality reviews approved. Full CI workflow contract: 40 passed. Five affected tests under shard-like settings: 96 passed. Shard coverage reports new_uncovered=0; YAML loads; duplicated touched shards are identical and singly cover each normal test path.
- Known non-new tooling baseline: whole-file Black checks still request broad formatting in legacy Helper_Scripts/release.py, test_release_helper.py, and test_required_workflow_contracts.py; parent/frozen versions produce the same baseline. Ruff on the release-owned changed Python surfaces is clean except the same three pre-existing Helper_Scripts/release.py findings already present at the frozen SHA. No mass unrelated reformat was included.
- Plan/status evidence is recorded in Docs/superpowers/plans/2026-07-15-release-0.1.41-implementation-plan.md and Docs/superpowers/plans/2026-07-16-release-0.1.41-zero-warning-docs-implementation-plan.md. Release PR, human-authored Change summary, required remote checks, merge, tag, publication, and main-to-dev sync remain pending.
Exact-head local gate (2026-07-16): committed release-preparation HEAD 8c4020f19585d86f534b33b0fd32343cce2539e8 passed the full Docs/public-private suite (195 passed, 4 runtime warnings), CI required-workflow contracts (40 passed, 2 runtime warnings), all four standalone documentation boundary/hygiene scanners, and canonical refresh idempotence. Strict MkDocs after refresh exited 0 with zero MkDocs WARNING records. Final Bandit reports /tmp/bandit_release_0_1_41_runtime_final.json and /tmp/bandit_release_0_1_41_tests_final.json each contain errors=0/results=0; B101 is skipped only in test scope. Shard coverage reports shards=757, test_files=4078, new_uncovered=0. git diff --check is clean; HEAD has exact frozen merge-base 4c2ad2070ed63992dac8a97a6c4cf3c7d75f6de8 and zero post-freeze merges. A clean detached checkout at 8c4020f195, without refresh, built strict with exit 0/zero warnings, remained git-clean, and preserved Docs/Site hashes (index 2be3a376890009c977bcb9320222806ed864e21bab036abd9f1ecd220266b2db; RELEASE_NOTES 3bff550c70ad52b6f0924dde69423d40be67b83450fe00fa7b1bf43ed0b1dfbe). Temporary worktrees and build outputs were removed. Local verification stages are complete; final whole-branch review is next, followed by push/PR and remote gates.
PR/base conflict recovery (2026-07-16): PR #2745 was opened against main from reviewed head 2e3f85e3da but GitHub correctly reported CONFLICTING because dev and main diverged before the 0.1.40 release merge (merge-base 440478b6cbaf9cb66e759ef6e8cebb859c81b44e).
Per the approved optional recovery plan, origin/main 7273cca4926abaa242a682f558fe1d3173f230e7 was merged without rebase in merge commit 60bc99db79cfc63cc8b08ceb8b9ce28371ffefa8; parents are exactly 2e3f85e3daca09a8ba52d0b82019810e90952363 and 7273cca4926abaa242a682f558fe1d3173f230e7.
The reconstructed merge had 22 conflicts. git cherry and file comparisons proved all eight main 0.1.40 commits touching those conflicts were patch-equivalent to commits already on frozen dev, whose versions also contain later reviewed functionality; the three truly main-unique MCP publication commits/files are preserved byte-for-byte.
Release resolutions preserve 0.1.41, 2026-07-16, historical 0.1.40 sections, strict zero-warning docs, case-safe Docs/_site, and deterministic generated parity.
Combined merge resolution differs from first parent only by a six-line Vitest-hoisted test mock for the real transitive tldwModels.subscribeInvalidation dependency; it returns an unsubscribe callback and fixes a pre-existing release-head collection failure without changing production code.
Independent merge spec and quality reviews approved. Verification at merge: strict MkDocs zero warning-level diagnostics; Docs/boundary 195 passed; CI contracts 40 passed; shard coverage new_uncovered=0; focused release 101 passed; backend conflict suites 73 passed; conflicted frontend suites 44 passed; broader Playground/model-invalidation frontend tests 22 passed; boundary/hygiene/parity/version/diff/Bandit application checks passed.
Worktree clean. The corrected merge ancestry has not yet been pushed at note time; PR human-authored Change summary and six remote required gates remain pending.
2026-07-16 pre-wait CI capacity evidence for PR #2745 at head 0a7e56b5ee66179f63edfde68ff225a56f80dc0a:
- Repo sweep audit: /tmp/tldw-0.1.41-ci-focus-pre-wait.log. Normal cancellation wedged on 11 unrelated tldw_server runs; guarded force-cancel succeeded; rerun emitted PROOF with active_unrelated_runs=[] and 29 release runs.
- Account sweep audit: /tmp/tldw-0.1.41-ci-focus-account-pre-wait.log. Today's Actions compute repositories were tldw_server, tldw_chatbook, and puzzle-attack. Normal cancellation of unrelated tldw_chatbook runs wedged; guarded force-cancel cleared them. ACCOUNT_PROOF emitted active_unrelated_runs=[] while preserving PR #2745 runs.

2026-07-16 backend-required failure diagnosis and approved in-scope fix:
- Failed run 29531588334 / job 87736407394 on exact release head. OpenAPI drift gate reported checked-in sha256 8dace04f... (1985 paths, 2892 schemas) versus current sha256 22324cf1... (1985 paths, 2890 schemas).
- CI installed FastAPI 0.136.3, Pydantic 2.13.4, Pydantic Core 2.46.4, Pydantic Settings 2.14.2, and Starlette 1.3.1; the older local release venv used Pydantic 2.11.7 and Starlette 1.2.1.
- Reproduced CI hash exactly in an isolated /tmp dependency environment. Contract comparison found no path changes. Pydantic unified identical ChatWorkflowTemplateStep input/output components and their draft wrappers; five common schemas only retargeted $ref values.
- Updated apps/tldw-frontend/lib/api/openapi.fingerprint.json to schema_count=2890 and sha256=22324cf1f3e80b7bdf7eec1807e85de4ff9eab1602173fcc8a92e880ca1d621c.
- Reproduced CI-version drift check passed. openapi-typescript 7.13.0 generated both old/new views successfully; the ignored generated diff only reflected unified component names/references.
2026-07-16 release merge and post-merge evidence:
- Release PR https://github.com/rmusser01/tldw_server/pull/2745 merged normally into main at 7a23be3202e360f2d8e7cfe208e13ba406cf0507. Frozen source 4c2ad2070ed63992dac8a97a6c4cf3c7d75f6de8 is an ancestor of that merge.
- Human-authored Change summary was inserted verbatim: "0.1.41 Brings in a lot of fixes to issues, including media ingestion and authentication." and "Used a frozen release branch due to the velocity of development and needing to pause without stopping."
- Exact immutable PR head c88285c532c8e2f4c512bf96cfc8926c7d5c4e56 passed all six release gates: frontend-required 29533600067, e2e-required 29533600100, backend-required 29533600187, security-required 29533600113, container-build-check 29533600035, and coverage-required 29533600083.
- Mandatory pre-merge repository proof /tmp/tldw-0.1.41-ci-focus-pre-merge.log emitted active_unrelated_runs=[] while preserving release run 29533600213. Account proof /tmp/tldw-0.1.41-ci-focus-account-pre-merge.log emitted active_unrelated_runs=[] for tldw_server, tldw_chatbook, and puzzle-attack. GitHub queued container runs that ignored normal cancellation were force-cancelled only after exact PR/head classification. The user-authorized cancellation policy ended immediately when PR #2745 merged; no post-merge run was cancelled.
- Clean detached post-merge verification at 7a23be3202 confirmed pyproject, FastAPI AST, README, and MkDocs versions are 0.1.41; release-note extraction length is 2561 with no duplicate bullets; historical 0.1.40 references remain only in prior-release sections; checked-in and refreshed strict MkDocs builds exited zero with no WARNING records; two refreshes were deterministic and left the worktree clean.
- Requester instruction on 2026-07-16: "Do not create a new tagged release." Tag and GitHub Release publication are therefore intentionally skipped. Confirmed no local v0.1.41 tag, no remote matching tag ref, and no GitHub Release for v0.1.41. Do not create or publish either later under this task.
- Sync branch codex/sync-main-0.1.41-to-dev was created from released main merge 7a23be3202. TASK-12973 remains In Progress until the sync PR merges and closeout evidence is committed.
Main-to-dev sync PR opened: https://github.com/rmusser01/tldw_server/pull/2748. Base=dev, head=codex/sync-main-0.1.41-to-dev. Its human-authored Change summary reuses the requester's exact two sentences from release PR #2745. No tag or GitHub Release will be created.
Closeout (2026-07-16): main-to-dev sync PR https://github.com/rmusser01/tldw_server/pull/2748 merged normally at 7d72382d9c671e3c8d11cc2f4ae1de8d30c78977. After fetching origin/dev and origin/main, git merge-base --is-ancestor origin/main origin/dev exited zero (origin/main=7a23be3202e360f2d8e7cfe208e13ba406cf0507, origin/dev=7d72382d9c671e3c8d11cc2f4ae1de8d30c78977). Tag-triggered publication checks are intentionally not applicable because the requester explicitly prohibited creating a tagged release. No local or remote v0.1.41 tag and no GitHub Release exist.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
0.1.41 was prepared from frozen dev commit 4c2ad2070ed63992dac8a97a6c4cf3c7d75f6de8 through PR #2744, with all relevant version surfaces, CHANGELOG.md, README.md, and release notes updated. The release also replaced the inherited 101-warning documentation baseline with deterministic canonical publication and strict zero-warning MkDocs validation. Release PR #2745 merged into main only after all six exact-head gates passed, and sync PR #2748 carried the released main history back into dev. The frozen release branch was used because development velocity required pausing a stable reviewed snapshot without stopping ongoing development. Per the requester's final instruction, no v0.1.41 tag or GitHub Release was created.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
