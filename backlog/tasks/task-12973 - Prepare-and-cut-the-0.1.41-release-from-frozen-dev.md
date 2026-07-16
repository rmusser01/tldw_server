---
id: TASK-12973
title: Prepare and cut the 0.1.41 release from frozen dev
status: In Progress
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
Prepare release 0.1.41 from frozen origin/dev commit 4c2ad2070ed63992dac8a97a6c4cf3c7d75f6de8 (through PR #2744), excluding all open PRs. Update authoritative version metadata, CHANGELOG.md, README.md, and release-note entry points; replace the inherited MkDocs warning baseline with deterministic canonical-to-Published generation and zero-warning strict validation in both the checked-in tree and CI-equivalent refresh pipeline; verify the expanded release diff and required gates; merge the reviewed release branch into main; tag the final main merge commit as v0.1.41; publish the GitHub Release; verify publication workflows; and sync main back into dev. Preserve the user's dirty primary checkout by working only in the isolated release worktree. Governing plans: Docs/superpowers/plans/2026-07-15-release-0.1.41-implementation-plan.md and Docs/superpowers/plans/2026-07-16-release-0.1.41-zero-warning-docs-implementation-plan.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Release scope is frozen at origin/dev 4c2ad2070ed63992dac8a97a6c4cf3c7d75f6de8 and open PRs are excluded
- [x] #2 Authoritative project, FastAPI, MkDocs, README, changelog, and release-note version surfaces consistently report 0.1.41
- [x] #3 CHANGELOG.md contains a curated 0.1.41 rollup for merged PRs included after 0.1.40 through PR #2744
- [x] #4 README.md current status and What's New accurately summarize 0.1.41
- [x] #5 Focused release metadata/docs tests, diff checks, and Bandit policy are satisfied
- [ ] #6 Release PR into main is merged only after required checks are green
- [ ] #7 Annotated v0.1.41 tag and GitHub Release point at the final main merge commit
- [ ] #8 Release artifact workflows are verified and main is synchronized back into dev
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
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
