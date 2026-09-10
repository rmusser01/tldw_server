# PR #2761 release refresh implementation plan

**Tracking:** TASK-13013.3; historical release record TASK-12988 on this branch.
**Goal:** Refresh the existing draft 0.1.42 candidate with current dev and produce reviewable release metadata and validation evidence before publication.
**Architecture:** Preserve the existing PR branch and merge the frozen dev source without rewriting history. Tag only the eventual reviewed main merge commit; do not run `make release-patch` after the pre-bump because it would select 0.1.43.
**Spec:** `Docs/superpowers/specs/2026-07-26-release-0.1.42-reviewed-metadata-design.md`, superseded for the source/date values below.

## Current release status

**Release is not ready to merge or publish.** This document is the active execution plan for [PR #2761](https://github.com/rmusser01/tldw_server/pull/2761), including unfinished work. The July design is historical; its source/date values are superseded here.

| Item | Recorded state |
| --- | --- |
| Integrated release code | `b3287b5437c122a12edf0dcafb155578b978a2ae` |
| Pushed candidate including generated-doc synchronization | `e5ad549c211ba96a1873e356b20f4883358b4476` |
| PR branch / target | `codex/release-main-0.1.42` → `main` |
| PR state at inspection | Draft, `UNSTABLE`; no merge or publication performed |
| Remote CI snapshot for e5ad549c21 | 17 successful checks, 49 pending; one failed **CodeQL** check ([run](https://github.com/rmusser01/tldw_server/runs/102918248263)) |
| Local focused checks | 241 passed; one docs-build test unpassed due host multiprocessing failure |
| Local docs alternative | Strict serial build passed; standard build still needs successful evidence |
| Latest observed GitHub publication | v0.1.38; no remote v0.1.39–v0.1.42 tags |
| Primary checkout | `dev`, unchanged; its local `a27ecb12f0` tracking commit is outside this release freeze |

CI counts are an observation, not a permanent state. Documentation follow-ups advance the PR head; always retrieve its current SHA and require results for that SHA before merge. The next work is **Stage 4.1: inspect the failed CodeQL result and complete current-head CI**, then resolve the remaining rows in Stage 4. No gate is waived by this plan.

## Stage 1: Recover and freeze
**Goal:** Restore the existing release branch and identify the current inputs.
**Success Criteria:** Original PR ancestry is retained; unrelated local work is excluded.
**Tests:** Live PR/release lookup, git ancestry and merge preview.
**Status:** Complete

- Original PR head: `9ca1823c99c5108f781764ee740346e1901b70d8`.
- Frozen dev: `50c1f689575b1bc21ed3e78cdb193b03fe968cdd` (through PR #2939).
- Frozen main: `d9c245ac14c40df855d1ab6cd19b3c137b16b47b`.
- The primary dev checkout has one local tracking commit, `a27ecb12f0`, excluded from this freeze.
- Current dev contains 2,331 commits absent from the old release candidate.
- Merge preview identifies one conflict in `tldw_Server_API/tests/CI/test_frontend_required_workflow.py`; retain the current dev assertion for the workflow's actual full-history checkout.

## Stage 2: Refresh reviewed metadata
**Goal:** Make the release notes, license record, manifest, and tests describe the frozen source.
**Success Criteria:** Version surfaces agree on 0.1.42; protected trees match frozen dev; manifest covers every tracked protected file.
**Tests:** Release helpers, licensing policy, release documentation contracts, docs regeneration idempotence.
**Status:** Complete

- Requester approved `today/now` on 2026-09-10 for the refreshed dates.
- Release date: `2026-09-10`; Countdown start: `2028-09-10T12:00:00Z`, preserving the original two-year interval and the verbatim template's fixed noon UTC activation.
- Refresh `LICENSES/releases/0.1.42/`, `CHANGELOG.md`, `Docs/RELEASE_NOTES.md`, `Docs/Site/RELEASE_NOTES.md`, `README.md`, and the licensing regression test; regenerate `Docs/Published` from canonical source.
- Keep the historical July design and tracking notes as history; these source/date values replace them for this candidate.
- Initial updated license-record test fails against the July source SHA, as expected, before regenerating the record.

## Stage 3: Verify and update draft PR
**Goal:** Commit the refreshed candidate and update PR #2761 with accurate verification and blockers.
**Success Criteria:** Focused checks pass or exact failures are documented; remote PR points to the verified candidate.
**Tests:** Focused pytest release/docs/CI suite, strict MkDocs, Actionlint, scoped Bandit, diff checks, protected-tree equality, final ancestry checks.
**Status:** Complete (candidate preparation and push; release gates remain in Stage 4)

- [x] Commit frozen-dev integration and release refresh as `b3287b5437` (TASK-13013.3).
- [x] Commit the regenerated Buddy/persona guide and evidence as `e5ad549c21`.
- [x] Push the existing PR branch without rewriting history.
- [x] Update PR #2761 title/body with current scope, evidence, and blockers; preserve draft status and human-summary placeholder.
- [x] Verify clean release worktree, remote/local candidate equality, frozen-input ancestry, and unchanged primary dev checkout.

Run tests with the project virtual environment activated and execute from this worktree. Keep the human Change summary placeholder intact. Local focused tests do not replace the required remote backend, security, coverage, frontend, E2E, container, and trusted license gates on the final candidate.

## Stage 4: Close release blockers
**Goal:** Obtain explicit evidence for every prerequisite before marking the candidate merge-ready.
**Success Criteria:** Every unchecked item below is completed or has a requester-approved scope decision recorded in its owning task; no failed or stale required check is accepted as green.
**Tests:** Current-head GitHub checks, standard strict docs build, dependency-task verification, publication inventory, migration/restore rehearsal, and human review.
**Status:** In Progress

### 4.1 Current-head CI and local docs failure — agent

- [ ] Inspect the failed [CodeQL check](https://github.com/rmusser01/tldw_server/runs/102918248263), retain its failure details, and determine whether it is a code, configuration, permission, or infrastructure failure. The snapshot alone does not establish the cause.
- [ ] Read all current-head check results and fix actionable regressions. Do not cancel required checks to manufacture readiness. After changes, push and revalidate the new head.
- [ ] Record success and run URLs for `backend-required`, `security-required`, `coverage-required`, `frontend-required`, `e2e-required`, `container-build-check`, and `frontend-license-policy/trusted/main`; capture remaining failing review/security checks as well.
- [ ] Re-run `test_strict_local_build_preserves_canonical_site_sources` and the unchanged strict MkDocs command on a host or CI runner that can allocate multiprocessing semaphores. Attach successful evidence for the candidate. The serial build is useful evidence but does not close this failed test.

Read-only status commands:

```bash
gh pr view 2761 --repo rmusser01/tldw_server --json headRefOid,isDraft,mergeStateStatus,statusCheckRollup
gh pr checks 2761 --repo rmusser01/tldw_server
```

Local reproduction, from the release worktree:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Docs/test_release_docs_contract.py::test_strict_local_build_preserves_canonical_site_sources -q --override-ini addopts=''
python -m mkdocs build --strict -f Docs/mkdocs.yml --site-dir /tmp/pr2761-docs-site
```

**Exit evidence:** Candidate SHA, check name, conclusion, and run URL for each gate; standard docs-build success. Main's ruleset enforces only the trusted license context, whereas the six core gates are enforced on dev. That ruleset minimum does not replace the release-readiness requirements.

### 4.2 Open readiness dependencies — agent, with requester ownership of scope decisions

Inspect existing child work and merged evidence before starting duplicate implementation. Current task status alone does not prove either completion or absence of work.

| Owning task | Remaining work and required completion evidence |
| --- | --- |
| [TASK-13013.7](../../../backlog/tasks/task-13013.7%20-%20Close-dependency-and-software-supply-chain-release-gaps.md) | Prove the supported frontend security baseline; Bun dependency-update and SBOM coverage; reproducible Python production resolution; immutable base images/artifact provenance; vulnerability scans and explicit exceptions. Record exact versions, digests, reports, and tested source SHA. |
| [TASK-13013.8](../../../backlog/tasks/task-13013.8%20-%20Prove-reusable-tenant-isolation-and-data-lifecycle-primitives.md) | Run cross-user and cross-organization negative tests for selected API/job/media/note/RAG/storage paths. Verify export, deletion, durable cleanup and partial-failure recovery, including logs/jobs/caches/backups. Record the tested profile and results. |
| [TASK-13013.9](../../../backlog/tasks/task-13013.9%20-%20Create-a-reusable-release-capacity-and-soak-test-harness.md) | Supply and run the reproducible capacity/soak profile with datasets, duration, pass thresholds and artifact output. Measure authentication/workflow load, queue depth, database pools, storage and overload recovery against an exact artifact. Record the supported operating envelope. |
| [TASK-12116](../../../backlog/tasks/task-12116%20-%20Re-enable-frontend-type-safety-and-lint-gates.md) | Reconcile TypeScript merge gating/strictness, React hooks enforcement, persisted Zustand version/migration contracts, and shared dependency majors. Link actual CI and migration-test evidence; a frontend build alone is insufficient. |

- [ ] Close each dependency with its required evidence, or obtain an explicit requester decision specifying what is outside this release's supported scope, why, and what risk remains. Record decisions in both the owning task and this plan. None has been granted in this session.

### 4.3 Distribution lineage, migration and rollback — agent

- [ ] Reconcile repository 0.1.39–0.1.41 metadata with GitHub, PyPI and container publication inventories. Record each existing version, source commit, artifact digest, and publication status. Do not invent missing releases or recreate tags merely to match documentation.
- [ ] Confirm 0.1.42 is unused across the intended publication targets before publication. If any artifact already exists, establish its provenance and follow recovery instead of overwriting it.
- [ ] Select a verified deployed rollback version and its immutable image/package identity. `v0.1.38` being GitHub's latest release does not prove it is the deployed rollback target.
- [ ] Assess authentication, conversation, notes-sync, presentation, personal-context and webhook schema changes from that target to the candidate; record configuration changes and incompatible downgrade paths.
- [ ] Rehearse backup, upgrade/health verification, and restore using representative data. Cover databases, uploaded content and configuration. Record backup checksums, restore commands and results. Do not assume an older binary can read migrated databases.

Use [Production Reference Deployment](../../Deployment/Production_Reference_Deployment.md), [Admin Webhooks Migration Runbook](../../Admin_Webhooks_Migration_Runbook.md), and [Standalone HTML Presentations](../../Deployment/Standalone_HTML_Presentations.md). Keep standalone HTML generation disabled until its documented schema-v2 backup and rollout prerequisites are met.

**Exit evidence:** Distribution inventory, approved version, exact rollback artifact, configuration/schema compatibility assessment, and successful backup/restore rehearsal linked here and from TASK-13013.3.

### 4.4 Human review — requester

- [ ] Write the PR's `Change summary` in the requester's own words, explaining what changed and why the implementation choices were made, per [the repository policy](../AI_GENERATED_PR_CHANGE_SUMMARY_POLICY_2026_04_17.md). The current placeholder does not satisfy this gate.
- [ ] Review the completed [0.1.42 legal record](../../../LICENSES/releases/0.1.42/release.json), [Countdown grant](../../../LICENSES/releases/0.1.42/PolyForm-Countdown-1.0.0.txt), source revision and manifest. The `today/now` response authorized refreshing dates; it did not record final legal-file review or waive other release gates.
- [ ] If release day changes before publication, obtain revised release/Countdown dates and refresh every dated surface, manifest verification and review as required by the original design.

**Exit evidence:** Requester-owned PR summary and an explicit review record covering the final legal record and source snapshot. Agent-generated copy cannot substitute for either.

## Stage 5: Merge, publish, verify and sync
**Goal:** Publish exactly the reviewed candidate and prove source/artifact/rollback lineage.
**Success Criteria:** Reviewed main merge SHA, immutable v0.1.42 tag, GitHub release, server-only package/container provenance, and main-to-dev synchronization agree.
**Tests:** Pre-merge gate recheck, tag/source checks, publication workflow results, artifact attestations, and ancestry checks.
**Status:** Not Started — blocked on Stage 4

These are future execution steps, not authorization to merge or publish while Stage 4 is incomplete.

- [ ] Re-read the PR head and require all Stage 4 evidence to cover it. Obtain the final release decision once the reviewable result is complete.
- [ ] Merge PR #2761 using the allowed merge-commit method; do not use an administrative bypass or reuse historical bypass claims.
- [ ] Read PR #2761's `mergeCommit.oid`, fetch main, and require `origin/main` to equal that exact SHA. Stop if main moved; do not guess a tag target.
- [ ] Recheck whether v0.1.42 or any corresponding artifact exists; recover an existing publication rather than creating a duplicate.
- [ ] Create the annotated `v0.1.42` tag on the verified merge SHA; verify the tag resolves to that SHA before pushing it.
- [ ] Extract release notes from the **tagged** `CHANGELOG.md` using `Helper_Scripts.release.extract_release_notes_for_version`, then create the GitHub Release with `--verify-tag` and those notes.
- [ ] Verify the automatic `publish-pypi.yml` run caused by the main version change; do not dispatch a duplicate publish. Record wheel/sdist identity and source correspondence.
- [ ] Verify `publish-docker.yml` succeeds for `app`, `worker` and `audio-worker`. Record versioned GHCR image digests and provenance attestations. Verify protected frontend source/build output is absent from server artifacts; no protected frontend binary is to be published.
- [ ] Sync released main into dev through a reviewed PR. Verify `origin/main` is an ancestor of `origin/dev` after merge.
- [ ] Record the final merge/tag/artifact identities, publication links, verification, rollback target, sync PR, and remaining accepted caveats; close TASK-13013.3 only when its acceptance criteria are satisfied.

**Do not run `make release-patch` for this pre-bumped candidate:** it would choose 0.1.43. If tag push succeeds but release creation fails, create only the missing GitHub Release. If a publication workflow fails, recover that workflow for the immutable release; never move the tag to fix an artifact.

## Verification evidence (2026-09-10)

- Focused release/helper/docs/licensing/CI suite: **241 passed, 1 deselected, 4 warnings**. The deselected test, `test_strict_local_build_preserves_canonical_site_sources`, was run both sandboxed and unsandboxed and failed with host multiprocessing `SemLock` / `OSError: [Errno 28] No space left on device`. It remains an explicit unpassed gate, not a skipped release requirement.
- Strict MkDocs API build passed with only the git-date plugin switched to serial processing in memory; repository configuration was unchanged. Two historical git timestamp warnings remain. Output: `/tmp/pr2761-docs-site`.
- `bash Helper_Scripts/refresh_docs_published.sh` is idempotent across all 453 generated files.
- Actionlint **1.7.12** passed across all workflows; Ruff passed on touched Python tests.
- Bandit on touched Python tests: `errors=[]`, `results=[]`, with standard test assertions excluded using B101. Artifact: `/tmp/bandit_pr2761_refresh.json`.
- Updated release-record test was red on the old source SHA before regeneration; final licensing tests pass.
- Existing routing contract caught 18 missing Watchlists E2E paths in `.github/license-first-paths.json`; the manifest now exactly copies the existing workflow filters.
- Independent reviewer confirmed all 7,099 protected manifest entries, legal-file digests, frozen-source tree equality, dates, route parity, and published/source release-note parity; no actionable findings within this refresh scope.
- Working refresh diff whitespace check passed. Full accumulated merge diff contains inherited whitespace warnings in the June Claims plan, pinned MCP protocol schema fixtures, and recurring-question models; these upstream files were preserved.
- Remote tag inventory contains v0.1.37 and v0.1.38 but no v0.1.39, v0.1.40, v0.1.41, or v0.1.42. Main rulesets require the trusted main license context and merge commits; the six documented core gates are enforced on dev, and remain required readiness evidence for this candidate.
- Raw logs are in `/tmp/pr2761-final-tests.log`, `/tmp/pr2761-docs-unsandboxed.log`, `/tmp/pr2761-mkdocs-serial.log`, and `/tmp/pr2761-actionlint.log`.
