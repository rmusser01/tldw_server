# PR #2761 release refresh implementation plan

**Tracking:** TASK-13013.3; historical release record TASK-12988 on this branch.
**Goal:** Refresh the existing draft 0.1.42 candidate with current dev and produce reviewable release metadata and validation evidence before publication.
**Architecture:** Preserve the existing PR branch and merge the frozen dev source without rewriting history. Tag only the eventual reviewed main merge commit; do not run `make release-patch` after the pre-bump because it would select 0.1.43.
**Spec:** `Docs/superpowers/specs/2026-07-26-release-0.1.42-reviewed-metadata-design.md`, superseded for the source/date values below.

## Current release status

### Current-dev continuation (2026-09-11 UTC)

The requester confirmed promotion of current remote `dev` followed by a release.
Reuse this candidate and TASK-13013.3. Freeze the additional dev input at
`6c4bdcbc48f4fe4bab7019d59ad8cf962ab240da` (PRs #2940 and #2941), retaining
all existing release fixes and the approved legal dates.

1. Integrate the frozen dev input without rewriting history; verify the Buddy
   portability and lifecycle tests. Status: Complete.
2. Investigate current-head frontend failures, refresh source/manifest and
   release notes, and run focused release/security checks. Status: In Progress.
3. Push the complete source/metadata batch to PR #2761, obtain fresh CI, and
   retain draft status until all recorded gates and human review are satisfied.
   Status: In Progress; follow-up security batch verified, fresh CI pending.

At the initial `43165c8c82` snapshot, 72 checks pass, none remain pending, and Characters Harness,
frontend unit shard 5, frontend-required and CodeQL fail. Characters Harness
times out in the import-completion test; shard 5 fails to find the post-copy
pack-refresh error in VisualPackEditor. CodeQL reports 7 critical and 414 high
changed-code alerts. These findings are unresolved, not waived.

The clean merge includes Buddy attribution validation/portability and the new
route lifecycle regression. Current verification: 44 backend portability tests
pass with one skip; the lifecycle test passes; all 65 VisualPackEditor tests
pass; the isolated Characters import-completion test passes. Full Characters
harness verification is running. Ruff is clean and Bandit reports zero findings
on the integrated Python scope. The release/docs/CI helper selection passes 79
tests; the strict docs test fails in the installed Git-revision plugin while
creating a multiprocessing semaphore (`SemLock`, errno 28), both inside and
outside the sandbox. This is the existing host limitation; the standard test and
build remain unchanged and must pass remotely.

The CodeQL review resolved 420 individually evidenced false positives or synthetic-test findings, including the requester-approved 416 remaining global dispositions. Actual UAT persistence, snapshot/Research session alias, Whisper and checkpoint probe repairs are committed with regression tests. Main checkpoint ownership findings remain undismissed. See [current remediation status](2026-09-11-pr2761-codeql-remediation-plan.md) and [disposition ledger](../../Evidence/PR2761-codeql-dispositions.md); hosted analyzer closure is still pending.

The requester reports no existing deployment. An installed-version rollback
baseline is therefore not applicable to this deployment; this does not waive
fresh-install, candidate recovery, migration-compatibility documentation,
security, lifecycle, supply-chain, capacity or human-review requirements.

**Release is not ready to merge or publish.** The candidate same-image backup/restore smoke now passes; the published rollback baseline is broken. See [recovery evidence](../../Evidence/PR2761-candidate-recovery.md). This document is the active execution plan for [PR #2761](https://github.com/rmusser01/tldw_server/pull/2761), including unfinished work. The July design is historical; its source/date values are superseded here.

| Item | Recorded state |
| --- | --- |
| Integrated release code and verified blocker fixes | `3f9866a860033b70b3434319fadfdd37b12819a2`; CodeQL rescan follow-up repairs included |
| Latest observed CI head | `f7c8af3af397a0eeec355a1cea333996d2e8287c`; 73 checks pass, one frontend aggregate runs, 38 skip and CodeQL fails. Verified follow-up source requires a new current-head scan. |
| Protected source snapshot | `3f9866a860033b70b3434319fadfdd37b12819a2`, 7,117 files; manifest `e38f39788bdbf6ee691a27cc91357e0d92a6b21a47355d745409938ea6e66f76` |
| PR branch / target | `codex/release-main-0.1.42` → `main` |
| PR state at inspection | Draft; no merge or publication performed |
| Remote CI snapshot | On `6150040801`, container aggregate/all five images, E2E, backend/security/license and all eight frontend shards pass; frontend aggregate and coverage still run. Completed CodeQL inventory: 447 open instances, all on `6150040801` (7 critical, 437 high, 3 medium). Its failed changed-code aggregate reports 424 alerts; that is a different measure from all open PR instances. |
| Latest metadata checks | 37 passed, one local docs-build test deselected due host multiprocessing failure; unchanged standard test/build passes remotely. Other scoped verification is recorded below. |
| Standard docs evidence | Unchanged standard build and docs suite pass in CI [run 34491682436](https://github.com/rmusser01/tldw_server/actions/runs/34491682436/job/102919740634). |
| Latest observed GitHub publication | v0.1.38; no remote v0.1.39–v0.1.42 tags |
| Primary checkout | `dev`, unchanged; its local `a27ecb12f0` tracking commit is outside this release freeze |

CI counts are an observation, not a permanent state. Documentation follow-ups advance the PR head; always retrieve its current SHA and require results for that SHA before merge. The next work is **Stage 4.1: complete fresh-head CI/security review**, alongside the isolated backup/upgrade/restore rehearsal and remaining dependency work in Stage 4. No gate is waived by this plan.

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
**Success Criteria:** Version surfaces agree on 0.1.42; protected trees match the final verified source commit; manifest covers every tracked protected file.
**Tests:** Release helpers, licensing policy, release documentation contracts, docs regeneration idempotence.
**Status:** Complete — protected source/manifest refreshed after verified fixes

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

### Active closure batch following `decdf9db77`

Requester explicitly authorized addressing all issues and blockers. Work is
tracked in the existing owning tasks; no release scope exception was granted.

Continuation after `6150040801` has verified three independently reviewed fixes:

- [Notification hooks](../../Evidence/PR2761-notification-hooks.md): scope-keyed
  snapshots recover automatically without displaying another account's count;
  stale reads/watch events cannot overwrite the current subscription. The Antd
  adapter supports frozen APIs without mutating other consumers' methods.
  WebUI selection passes 25 tests; the overlapping shared-package selection
  passes 23 tests. Both installed-Plasmo and WebUI focused typechecks pass,
  with all four remaining compiler rules enabled on the touched hook scope.
- [Web-clipper storage](../../Evidence/PR2761-web-clipper-storage-strictness.md):
  six optional-method narrowing diagnostics are fixed in the extracted runtime
  adapter, now included in required strict checking. All 13 handoff tests pass;
  callback/Promise settlement, receiver and browser fallback behavior are retained.
- [CodeQL test fixtures](../../Evidence/PR2761-critical-CodeQL-assessment.md):
  remove three no-op category replacements and replace an incomplete script-tag
  regex with the existing HTML parser. Four parser regressions failed before
  the repair; 58 fixture tests now pass under installed Vitest 4.0.18. The later
  `0929b44a5a` JavaScript scan confirms alerts 2356 and 2631–2633 are no longer
  open. Its mixed-language PR inventory is 443 while preceding Python instances
  remain, so this is not a completed current-head total. The aggregate still
  fails; no production security boundary or alert state was changed.

Across these selections, 96 focused frontend tests pass. The full nonincremental
WebUI typecheck also passes. Release docs/workflow tests pass 26 cases with one
unchanged host-limited docs test deselected. No Python implementation changed;
the required shared hook gate scans 5,160 files with zero failures, while 1,379
unrelated ESLint errors remain outside that three-rule gate. No global four-rule
rescan or full shared-lint success is claimed. For the TypeScript changes,
Bandit cannot parse TypeScript and its parser errors are explicitly recorded in
the evidence. The metadata companion still requires its own Python validation.
Independent reviews found no actionable issues in the three bounded scopes.
Whole-WebUI strictness, the remaining global compiler rules and the security
inventory remain open; these scoped fixes do not close TASK-12116.

Current container evidence is [run 34526384567](https://github.com/rmusser01/tldw_server/actions/runs/34526384567).
The separate supply-chain PR #2869 remains at `78c3f92228` with nine failed
checks; its unverified concurrent work is not integrated.

The next bounded batch after pushed `0929b44a5a` is verified:

- [RAG ref ownership](../../Evidence/PR2761-rag-input-ref.md) moves the unused
  hook-exported input ref into its sole component consumer. All 128 scoped
  refs diagnostics disappear; focus/query/filter behavior is unchanged, with
  11 RAG tests, the full WebUI typecheck and independent review passing.
- A complete 5,161-file four-rule rescan now reports **265 findings across
  138 files**: refs 105, set-state-in-effect 90, immutability 21 and
  preserve-manual-memoization 49. This replaces the earlier 403 count as the
  current inventory. [Every remaining finding and affected-source hashes](../../Evidence/PR2761-remaining-hook-findings.json)
  are recorded, including the additional ACP permission-clock effect finding
  from the earlier purity correction. No global rule is enabled or suppressed
  by this inventory; each remaining cluster still requires investigation.
- [DSR preview coverage](../../Evidence/PR2761-dsr-preview-coverage.md) now
  queries only selected categories. Existing/unknown embedding storage and
  failed collection counts reject coverage instead of producing zero or
  partial totals. Confirmed absent optional storage still returns zero.
  Six service and seven API regressions failed before the fix; 50 DSR tests
  plus four existing endpoint sanitizer tests pass after it. Preview/intake
  return the fixed error and failed coverage stores no intake record.
  Independent review, Ruff/changed-range formatting and Bandit pass.

The DSR API fixture now uses the canonical account bootstrap/seeding helper
instead of a raw insert rejected by the profile-write guard. An intermediate
validation run was invalidated by host disk exhaustion; about 695 MiB of known
completed synthetic test directories were reclaimed, and the final isolated
run passed. Private backup artifacts and unrelated caches/builds remain intact.
Actual erasure/attachment restrictions and whole-account lifecycle certification
are outside this preview correction. Fresh source metadata and CI are required
before release. The previous frontend aggregate was superseded by the branch
update and was never recorded as passing.

- [Frontend hardening](../../Evidence/PR2761-frontend-hardening.md): Flashcard
  template suite 8 passes; five shared dependency majors aligned with 38
  persistence/Markdown/Dexie tests and both client typechecks passing. A required
  strict TypeScript project covers four shared URL/API-key/error utilities;
  use-memo is re-enabled after repairing Timeline dependency expressions.
  Timeout selection is now included in the strict project (22 focused tests),
  and the Skills runner harness passes 47 tests and a scoped strict-null compile.
  [Hooks follow-up](../../Evidence/PR2761-hooks-enforcement.md) repairs and enforces
  purity/static-components in the shared UI scope. Whole-WebUI strictness and
  the remaining four disabled compiler-era rules stay open. The subsequent
  [chat hook repairs](../../Evidence/PR2761-chat-hook-state.md) resolve nine refs
  findings from the 403-finding baseline: delayed saves preserve newer voice turns,
  assistant IDs update on reset/completion, and empty token caches retain zero.
  34 tests and focused typecheck/lint pass; no global rescan is claimed.
- [CodeQL fixes](../../Evidence/PR2761-critical-CodeQL-assessment.md): three
  reproduced raw-exception disclosures repaired in ACP and embedding requeue
  responses; credential reads separated from the TTS preference tuple.
  31 backend and 42 frontend tests pass. The completed `d5ba8b5be7` scan confirms
  the three targeted exception alerts are no longer open on the PR;
  path-injection and Actions findings still need individual investigation.
  Follow-up audio validation now distinguishes existing local directories from
  remote model IDs and checks input symlinks before canonicalization; the
  audio suite now passes 65 tests. This is a concrete boundary fix, not blanket
  alert closure. Manual CI change detection now honors the requested base commit;
  39 detector/admission/workflow tests pass and Actionlint passes.
- [Worker packaging](../../Evidence/PR2761-worker-packaging.md): both workers
  include canonical config/local packages/license inputs; the container gate
  builds all five images and checks backend imports in isolated containers.
  40 contracts pass and both workflow files pass Actionlint. Both worker images
  and all three other images now pass fresh CI builds on `d5ba8b5be7`.
  Four ChromaDB advisories have no published
  patched version; no speculative upgrade or exception was added.
- [Lifecycle recovery](../../Evidence/PR2761-dsr-erasure.md): DSR honors SQLite
  foreign keys and excludes raw error text; 26 tests pass. Real Sync object GC
  verifies deletion, restart after failure before/after unlink and isolation of
  equal-content data belonging to another owner; 68 tests pass. Coordinated
  whole-account attachment/history/backup erasure is not yet certified.
  Five additional embedding/preview failure-log privacy regressions were
  reproduced and fixed; the DSR suite now passes 31 tests. The combined DSR/audio
  selection passes 96 tests, with no new Bandit findings in changed Python.
- The published 0.1.38 startup failure, unknown deployed rollback baseline,
  measured production capacity envelope, supply-chain vulnerability disposition
  and final human review remain explicit prerequisites. Passing candidate
  same-image restoration does not replace them.
- Capacity evidence was being prepared from clean `910c526513` Dockerfile
  source in an isolated, resource-capped Docker environment. The supported
  profile uses multi-user PostgreSQL AuthNZ and SQLite content, with one
  disposable actor and real pool/queue/storage measurements. Single-user mode
  cannot provide a production PostgreSQL pool because it selects SQLite.
  Build steps and runtime package imports passed, but final image export was
  stopped when concurrent Docker builds reduced host free space to about 12 GiB.
  No startup/preflight/traffic phase ran, so there is no capacity result.
  [Stopped-build evidence](../../Evidence/PR2761-capacity-diagnostic.md) records
  exclusive fixture/context cleanup and approximately 8.56 GB of exact-ID cache
  reclamation. Host space subsequently fell to about 7 GiB while other builds
  continued; physical recovery is not confirmed. Cleanup is limited to task-owned
  artifacts; shared caches and unrelated builds must not be pruned.
  The run needs a suitable runner with
  adequate disk capacity. This diagnostic cannot certify the final artifact or
  an unspecified deployed production profile.

### 4.1 Current-head CI and local docs failure — agent

- [x] Inspect the failed [CodeQL check](https://github.com/rmusser01/tldw_server/runs/102918248263), retain its failure details, and determine whether it is a code, configuration, permission, or infrastructure failure. The snapshot alone does not establish the cause.
- [ ] Read all current-head check results and fix actionable regressions. Do not cancel required checks to manufacture readiness. After changes, push and revalidate the new head.
- [ ] Record success and run URLs for `backend-required`, `security-required`, `coverage-required`, `frontend-required`, `e2e-required`, `container-build-check`, and `frontend-license-policy/trusted/main`; capture remaining failing review/security checks as well.
- [x] Re-run `test_strict_local_build_preserves_canonical_site_sources` and the unchanged strict MkDocs command on a host or CI runner that can allocate multiprocessing semaphores. Attach successful evidence for the candidate. The serial build is useful evidence but does not close this failed test.

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
| [TASK-12116](../../../backlog/tasks/task-12116%20-%20Re-enable-frontend-type-safety-and-lint-gates-harden-persisted-stores.md) | Reconcile TypeScript merge gating/strictness, React hooks enforcement, persisted Zustand version/migration contracts, and shared dependency majors. Link actual CI and migration-test evidence; a frontend build alone is insufficient. |

- [ ] Close each dependency with its required evidence, or obtain an explicit requester decision specifying what is outside this release's supported scope, why, and what risk remains. Record decisions in both the owning task and this plan. None has been granted in this session.

### 4.3 Distribution lineage, migration and rollback — agent

- [ ] Reconcile repository 0.1.39–0.1.41 metadata with GitHub, PyPI and container publication inventories. Record each existing version, source commit, artifact digest, and publication status. Do not invent missing releases or recreate tags merely to match documentation.
- [ ] Confirm 0.1.42 is unused across the intended publication targets before publication. If any artifact already exists, establish its provenance and follow recovery instead of overwriting it.
- [ ] Select a verified deployed rollback version and its immutable image/package identity. `v0.1.38` being GitHub's latest release does not prove it is the deployed rollback target.
- [ ] Assess authentication, conversation, notes-sync, presentation, personal-context and webhook schema changes from that target to the candidate; record configuration changes and incompatible downgrade paths.
- [ ] Rehearse backup, upgrade/health verification, and restore using representative data. Cover databases, uploaded content and configuration. Record backup checksums, restore commands and results. Do not assume an older binary can read migrated databases.
- [x] Complete a separately scoped candidate same-image SQLite/Redis backup/restore smoke: authenticated account/note/conversation/attachment checks, config-byte equality, Redis RDB→AOF recovery, and all 12 databases passing integrity before and after. [Evidence](../../Evidence/PR2761-candidate-recovery.md). The published 0.1.38 startup failure still blocks the cross-version item above.

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

- Focused release/helper/docs/licensing/CI suite: **241 passed, 1 deselected, 4 warnings**. The deselected test, `test_strict_local_build_preserves_canonical_site_sources`, was run both sandboxed and unsandboxed and failed with host multiprocessing `SemLock` / `OSError: [Errno 28] No space left on device`. The host limitation remains; the unchanged standard test/build subsequently passed on candidate 7d7a2e708d in CI run 34491682436, closing this docs blocker.
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

## Stage 4 execution update — 2026-09-10

All results below precede the next fix commit. Required remote checks must be repeated on the final pushed head.

### CI fixes and security triage

- Standard docs gate closed: [onboarding-docs-gate](https://github.com/rmusser01/tldw_server/actions/runs/34491682436/job/102919740634) runs the unchanged docs tests and strict MkDocs successfully.
- At `7d7a2e708d`, [backend](https://github.com/rmusser01/tldw_server/actions/runs/34491682522/job/102920038746), [security](https://github.com/rmusser01/tldw_server/actions/runs/34491682130/job/102922117075), [container](https://github.com/rmusser01/tldw_server/actions/runs/34491682218/job/102922254677), [E2E](https://github.com/rmusser01/tldw_server/actions/runs/34491682309/job/102921012296) and trusted main licensing pass. Coverage is pending.
- Frontend shard 6: repaired stale character-routing fixtures (assistant metadata/setters and optional request-scope argument); image-event and adjacent character suites plus locale parity now **23 passed**. No production routing bypass was added.
- Frontend shard 2: restored three missing role-play failure messages in the extension English locale mirror.
- Frontend shard 1: repaired Notes connection/principal fixtures and made Companion loading-state timing deterministic. **21 tests pass**; added a negative test proving note reads wait for verified identity. Existing backlink/card assertions remain. Shard 5 scheduled-reminder validation now awaits the asynchronous timezone error before asserting no create request; **50 tests pass** in the complete file. All known frontend unit regressions have local fixes.
- CodeQL Actions analysis reports 238 findings: 232 poisonable steps, five direct cache writes, one inherited untrusted checkout. Cache queries combine workflow-wide dispatch/schedule triggers and PR-head checkout expressions without evaluating event guards; no reachable workflow-run cache-write exploit was demonstrated. Ten checkout jobs lack direct admission dependencies, so deleting raw-event fallbacks would select the wrong SHA. Alerts remain unresolved; no suppressions or dismissals applied.
- JavaScript scan identified a real Speech Playground persistence problem: object spread admitted unexpected API-key/credential properties. Explicit persisted-field projection replaces that spread; regression reproduced red then passed, **78 tests** across five suites. Three avatar regressions confirm existing normalization rejects javascript/SVG-data URLs and accepts HTTPS; no duplicate sanitizer was added.
- TASK-12116: valid frozen Bun install exposed a bounded WebUI TypeScript baseline. Typed mock contracts and structured-presentation narrowing are repaired: the full nonincremental WebUI typecheck passes, alongside **105 Skills script tests**, **57 presentation tests**, and collection of **39 Playwright security cases**. An explicit nonincremental typecheck now gates frontend-required; **7 workflow contract tests** and Actionlint pass. All nine named persisted stores already declare version 1 plus identity migration; five existing store suites pass **17 tests**, closing TASK-12116 criterion 4 for the current schema. Strictness, additional hook rules and dependency-major alignment remain open.

### Readiness dependencies

- TASK-13013.7 remains owned by active PR #2869 and its separate supply-chain worktree. That worktree contains additional unpushed native-applicability work; its published head still fails scans. Do not integrate it blindly. Its latest evidence records 93 scoped exceptions expiring September 17 and retained vulnerability replay rows; final publication needs current scans/exception review.
- TASK-13013.8: initial cross-user API/cache/permission selection **38 passed**. Additional storage/lifecycle selection initially **37 passed, 4 failed, 17 errors**; investigations found missing allowed-root setup and missing AuthNZ account fixtures. These failures are repaired: **76 tests pass** across Sharing, Chatbooks export/import and MediaFiles. Beyond fixture setup, a real repository contract violation returned RowAdapter objects instead of dictionaries, silently omitting export artifacts; dictionary conversion plus a serialization regression fixes it. Production Bandit has zero findings; test baseline adds only assertions. Account deactivation is not erasure; partial primitives do not close full lifecycle certification. Active Reading cleanup PR #2903 remains separate and unfinished.
- TASK-13013.9: implemented [capacity harness](../../Development/Release_Capacity_Soak.md), with [dedicated plan](2026-09-10-release-capacity-soak-harness.md), **30 passing behavioral tests**, zero production Bandit findings and zero test findings with B101 assertions excluded. Independent review caught a final-sample recovery false pass; regression and fix included. This proves runner behavior, not a release operating envelope. A real artifact, fresh collector measurements and representative workload run remain required. Existing full-suite Helper_Scripts directory selection includes the new tests.

### Distribution and recovery findings

- Public PyPI lists only **0.1.32**; versions 0.1.38–0.1.42 return 404. GitHub release labels and repository metadata do not imply PyPI publication.
- GHCR app `0.1.38`/`latest`: `sha256:70fb5ef2ce0e7bd11d0c359d16ccacae5493bf3c3064a5f4eefd86d99a06f933`, source `ffb6f106e9a96fc6214191135c4afc7d916c5b29`, [publication](https://github.com/rmusser01/tldw_server/actions/runs/28827294234). Worker and audio-worker builds failed in that run.
- GHCR app `main`/`sha-7a23be3`: `sha256:16ea0dbc11f5493f730451007f8b25933cfb79c966f37d5cf658bfd4f95dba4c`, source `7a23be3202e360f2d8e7cfe208e13ba406cf0507`, [publication](https://github.com/rmusser01/tldw_server/actions/runs/29559405432). Current main image publication was cancelled.
- App 0.1.39–0.1.42 tags are absent; worker/audio-worker inventory returns 403 and authenticated package access lacks read:packages. Their version-collision status is **unverified**, not absent. Recheck before publication.
- OCI/SLSA metadata matches these source/workflow identities; this is metadata inspection, not cryptographic attestation verification. No actual user's deployed rollback version has been inferred.
- Recovery selection repaired the CLI stderr test to use a real subprocess because Loguru retains its import-time stream. **86 tests pass** across SQLite/WAL recovery, app archives, Redis-file restore, deployment ordering and webhook recovery. Ruff/Black pass; Bandit baseline and final findings are identical. This is not a live full-stack PostgreSQL/Redis deployment rehearsal.
- Remaining Stage 4.3 work: complete private worker inventory; assess schema/config changes against selected immutable rollback artifact; rehearse full backup/upgrade/restore in a disposable deployment with checksummed databases/uploads/configuration; record authenticated acceptance checks. Publication and human review remain later stages.

Raw local evidence: `/tmp/pr2761-checks-latest.json`, `/tmp/pr2761-chat-locale-final.log`, `/tmp/pr2761-ui-security-final.log`, `/tmp/pr2761-repository-recovery-final-results.xml`, `/tmp/pr2761-ghcr-summary.json`, `/tmp/pr2761-pypi-lineage.json`. Important findings and external run identities are retained above so resumption does not depend on temporary files.

The final protected source identity must advance from frozen dev to the verified blocker-fix source commit. Regenerate all tracked protected-file hashes and the legal record, update the licensing regression source pin, and verify tree equality before pushing the final batch. Source and metadata commits are prepared together; never publish the intermediate record.

### Final local blocker-fix verification

- Slash-command parsing: CodeQL alert #2599 reproduces quadratic rejection of a 32 KiB malformed argument (about 5.5 seconds). Requiring the argument's first non-whitespace character removes ambiguous separator backtracking; fixed raw match is about 0.3 ms. **65 router/injection/endpoint/replace-mode tests pass**; 111,111 short inputs preserve previous captures. Production Bandit reports zero findings. The regex is inherited, and this release adds a call site; no claim that every alert is a new vulnerability.
- CodeQL completed all three language jobs on `7d7a2e708d`; Python reports 566 results and 173 open alerts, including 142 alert IDs already open on main. Further reviewed path/hash reports did not demonstrate an exploit. Findings are not dismissed; final-source rescan and remaining security review stay open.
- Harness independent review initially found one test violating the normal HTTP-mocking guard. Replaced that constructor monkeypatch with a real ephemeral loopback collector. **30 tests now pass under the normal repository configuration**, with four existing warnings; production/test Bandit reports are clean (test assertions excluded). No guard or test configuration was disabled.
- Complete ScheduledTasks suite: **50 passed**. Notes/Companion: **21 passed**. Chat/image/locale: **23 passed**. Full WebUI nonincremental typecheck: **passed**. Standard docs contracts after release-note edits: **18 passed**, host-limited build test separately covered by remote standard docs success.
- Real bugs repaired in this batch: speech credential-field persistence, registered media artifacts omitted from export, and slash-command regex backtracking. Other frontend changes repair types, missing locale copy, incomplete fixtures and asynchronous test assertions.

Protected source freeze refreshed to **`0cec0bb409ddbd2de0089e1909b4b6b718823de3`** after the verified blocker fixes. The manifest covers **7099 tracked files**, SHA-256 **`540fd61c20a6ffd6356d25238d49efc9568ba41a6f3b56d44173e4868c4edc64`**. Release date and Countdown/legal bytes are unchanged; source ancestry retains both original frozen inputs. The following metadata commit and CI must validate this final source snapshot.

Final metadata verification: **36 passed, 1 host-limited docs test deselected, 4 existing warnings**; that unchanged docs build already passed on the CI runner and must pass again on the final head. Scoped licensing-test Bandit has zero findings, protected trees equal `0cec0bb409`, and frozen dev/main remain ancestors. The primary checkout remains clean at `a27ecb12f0f6371314555ee4929dfd4d3372b7ea`.

[Migration compatibility assessment](../../Evidence/PR2761-0.1.38-migration-assessment.md) now records the exact published app reference and candidate source: AuthNZ 89→98, ChaChaNotes 51→66, forward-only Slides, canonical Notes/Sync blobs, Personal Context keys, and conditional webhook rollback. All 14 linked local sources resolve. This closes the bounded source comparison; it does not claim a chosen deployed baseline or a live restore rehearsal. Next: push the source/metadata batch, inspect fresh CI, and work any new actionable failures. Retain draft status for unresolved CodeQL, readiness dependencies, full deployment evidence and final human review.

## Fresh CI and security follow-up after push `7c79e085df`

- Source/metadata batch is pushed as `7c79e085dfcc19b1ce40cf0783681100a815ced9`; PR remains draft. Fresh standard [docs](https://github.com/rmusser01/tldw_server/actions/runs/34495654425/job/102933315596), [security](https://github.com/rmusser01/tldw_server/actions/runs/34495654307/job/102933544259), [container](https://github.com/rmusser01/tldw_server/actions/runs/34495654502/job/102935239974), and trusted main license gates pass. Backend/coverage/E2E/frontend continue.
- Fresh frontend shard 3 exposed a Research Runs sources/outline-review readiness race. A deliberately delayed SSE checkpoint reproduced it; the test now waits for the actual editable field and retains validation/patch assertions. The complete file passes **19 tests**. Passing prior-head or local selections do not waive this new failure.
- [Critical CodeQL assessment](../../Evidence/PR2761-critical-CodeQL-assessment.md) enumerates all seven critical IDs, baseline comparisons, source guards and unresolved limits. **28 HTTP**, **21 XPath**, and **9 URL-guard** tests pass under normal fixtures. No additional bypass was demonstrated; alerts remain open and this evidence is not a suppression or final security disposition.
- TASK-13013.7 remains actively edited in its separate worktree (local `00bda5b77b`, additional uncommitted applicability work). Published PR #2869 is still `78c3f92228c6411ee4637b9d2df9aa3b50aacdc8`, failing five image scans, source scans and related gates. Integration awaits coherent verified dependency work.
- Live rehearsal feasibility: the Production Reference CLI enforces PostgreSQL/Caddy and hardcodes production volume names, so a project-name override alone would not isolate SQLite testing. A separate temporary single-user app/Redis harness is being prepared with unique volumes, internal networking, no published ports, generated test credentials and no production mounts. A pinned amd64 probe runs on this arm64 Docker host. Baseline image is not cached and candidate build is required; acquisition/build/startup/upgrade/restore remain in progress. This verifies only a synthetic SQLite/Redis profile and does not waive other supported profiles or establish a capacity envelope.

- Live PostgreSQL supplement: **2 passed, 0 skipped, 0 failed**, using existing repository `pg_database_config` fixtures on PostgreSQL 18. Verified real two-owner attachment CRUD/tombstone/restore, FORCE RLS under a non-superuser/non-bypass role, denied cross-owner writes, index plans and task operations clearing session dataset scope. `TLDW_TEST_NO_DOCKER=1` prevented replacement of the already-running fixture server; `TLDW_TEST_POSTGRES_REQUIRED=1` made missing prerequisites fail. Normal fixture cleanup ran. Relevant runtime/test files equal `0cec0bb409`. This extends TASK-13013.8 evidence; it does not close full data lifecycle certification.
- Rehearsal acquisition is running in `/tmp/pr2761-sqlite-rehearsal-Y4OqTa`: exact published baseline pull, pinned Redis, and candidate build from clean archive `0cec0bb409`. The unmodified Dockerfile resolves floating ML dependencies, including large Torch/CUDA downloads. The resulting image and dependency identities must be retained; this is not reproducible-build or final supply-chain certification. No production mounts/volumes are used.

### Research follow-up source and metadata verification

- Protected source now **`d1af177bf3f55d05231ebec79ee6e7f2528a55e3`**, retaining the verified runtime fixes in `0cec0bb409`. Manifest: **7,099 files**, SHA-256 **`8a2ad512bb17328c436e3a77cc346f1d98dac91e1973a0a76dc985de7a973bdc`**. Legal/date bytes remain unchanged. Runtime rehearsal continues against `0cec0bb409`; subsequent source changes are frontend test and evidence only.
- Metadata validation: **36 passed, 1 host-limited docs test deselected, 4 existing warnings**; scoped Bandit and Ruff clean; protected-tree equality and diff whitespace checks pass. The standard docs CI test/build passed again on `7c79e085df`.
- On `7c79e085df`, all seven other frontend shards and E2E passed; only shard 3's Research readiness race failed. Its replacement retains assertions and has **19 passing tests**. Coverage and Python CodeQL analysis were still running at this snapshot. Push the complete source/metadata pair and require fresh-head results.

### Lifecycle supplement

- **35 passed, 4 warnings** on six existing suites: SQLite Jobs idempotency and owner attribution; RAG semantic-cache tenant scoping; storage cleanup including unregister failure and sanitized error logging; file-export garbage collection; mocked embeddings-erasure orchestration. Logs: `/tmp/pr2761-lifecycle-supplement.log`, `/tmp/pr2761-lifecycle-supplement.xml`. Tests use their normal fixtures; Chroma/category erasure is mocked, so this does not certify whole-account deletion or backup erasure. Pytest also warned while cleaning an unrelated older temporary directory. TASK-13013.8 remains In Progress.

## Container startup and final-head follow-up

- At `30338ef7de`, [backend](https://github.com/rmusser01/tldw_server/actions/runs/34497893928/job/102941664409), [security](https://github.com/rmusser01/tldw_server/actions/runs/34497893730/job/102941363352), [coverage](https://github.com/rmusser01/tldw_server/actions/runs/34497893374/job/102943511571), [E2E](https://github.com/rmusser01/tldw_server/actions/runs/34497893572/job/102941132694), [container build](https://github.com/rmusser01/tldw_server/actions/runs/34497894239/job/102944274722), [standard docs](https://github.com/rmusser01/tldw_server/actions/runs/34497893739/job/102940917378), and trusted main license pass. Only frontend shard 5 fails alongside CodeQL. The aggregate never reaches its typecheck while a shard fails; TASK-12116 criterion 1 is still open.
- The new shard-5 failure is the whitespace-only reminder validation case. It and two adjacent one-time negative cases waited for a no-call assertion that could pass before asynchronous validation. They now await positive error messages before asserting no create request. The entire ScheduledTasks file passes **50 tests** (103.36 seconds); no application validation changed. One initial local invocation used the wrong workspace runner and failed before running tests; the recorded passing run uses the installed WebUI Vitest with its normal config.
- Actual image startup exposed omissions that a successful image build missed. Published `0.1.38` cannot import `mcp_unified`; candidate runtime `0cec0bb409` cannot import `tldw_profile_core`. Candidate app packaging now copies the already-declared local profile package and imports both local packages during the runtime image build. The explicit server-only Docker COPY allowlist test is updated for that one source directory. **31 Docker/reference/packaging tests pass**; touched Python Ruff/Bandit are clean (B101 assertions excluded). Full image rebuild/startup is in progress. Worker Docker changes remain with active supply-chain work.
- The published baseline failure blocks a genuine baseline-seeded cross-version rehearsal. Preserve that failure; do not silently patch an immutable historical image and call it the published artifact. If candidate startup succeeds, perform a separately labeled candidate backup/restore smoke, retaining the cross-version gate.
- Actual owner-scoped Jobs receipts are verified separately from low-level create deduplication: **90 SQLite/PostgreSQL tests passed, zero skips**, including owner isolation, rollback, corrupt correlation, archive behavior and concurrency. The low-level scenario's owner claim was inaccurate; the documented boundary is `(domain, queue, job_type, idempotency_key)`. A temporary assertion demanding an incompatible fifth owner dimension failed on both backends and was removed after contract review. Only its docstring was corrected. This was not evidence of a new public cross-tenant exploit. Logs: `/tmp/pr2761-jobs-receipts.log`, `/tmp/pr2761-jobs-receipts.xml`.
- Release Backlog integrity rerun passes: **13 release nodes, 4 historical records**.
- TASK-12116 remaining strictness baseline is now measured: separate flag runs produce **948 noImplicitAny diagnostics in 252 files** and **664 strictNullChecks diagnostics in 177 files**. A 328-file hook sample adds **25 diagnostics in 14 files** under the seven disabled compiler rules. Five shared runtime dependency-major mismatches remain. Enabling all these gates is substantial uncompleted work, not a passing current baseline.

Package/reminder source commit: **`50dcf5453b1ba10b0eabb505ff32b831f47f8185`**. Protected manifest regenerated across **7,099 files**, SHA-256 **`2e495751c4c3ad02842881167da31ab15a4e0f665f0f872c5906590dc213b099`**; legal/date bytes unchanged. Local image uses clean `0cec0bb409` plus the identical Dockerfile fix; later changes are evidence, metadata and tests. This diagnostic artifact is not the final published artifact or full supply-chain certification.

## Completed candidate recovery smoke

The [recovery report](../../Evidence/PR2761-candidate-recovery.md), [machine-readable evidence](../../Evidence/PR2761-candidate-recovery.json), and [286-entry dependency inventory](../../Evidence/PR2761-candidate-dependencies.txt) are retained in the repository. Tested image `sha256:e01578be89000870b86c09164ad724160654f410a755a44fb84500857977b630` starts successfully. It restored the exact authenticated synthetic account, note, conversation and 34-byte attachment into a second volume set. All **12 SQLite databases** passed integrity checks before and after; runtime config bytes were identical and the Redis marker survived RDB restoration, AOF reconstruction and restart.

The temporary controller initially omitted volume-root ownership and was corrected; an early readiness request also preceded completed startup. All final acceptance checks passed after those corrections. Rehearsal containers/network are removed; private backups, own test volumes and images are retained. No production data/configuration, unrelated fixture server or active dependency worktree was modified.

This local diagnostic image uses clean `0cec0bb409` plus the exact Dockerfile repair. Its application runtime source equals `50dcf5453b`; embedded tests/docs/metadata are older and this is not a published artifact. The published 0.1.38 image failed importing `mcp_unified`, so a working deployed rollback baseline and seeded cross-version upgrade/restore remain open. PostgreSQL production-reference/Caddy, multi-worker/provider load, encrypted personal-context data, full lifecycle and capacity certification are not covered by this smoke.

PR #2869 remains unchanged at `78c3f92228c6411ee4637b9d2df9aa3b50aacdc8`, with conflicts and nine failed image/source/security checks. Its active separate work is not integrated. At `44c6b45c85`, backend, security, app-container build and five frontend shards pass; other frontend shards, coverage and E2E were still running at this snapshot. Final current-head gate results are maintained in the PR description; this evidence update does not waive fresh checks.

## Verified source freeze for the current closure batch

Source **`910c526513a03b305b2976fec6f1dee73bacc1e9`** contains the verified fixes above. The regenerated
manifest covers **7,100 protected files**, SHA-256
**`96b17d0aa64f09247145ac8beb88ee75bfa60cf5f84e7cb43b9698678677d12a`**. Legal/date bytes are unchanged.
Independent review found no frontend regressions. Independent integration
verification passed **42 tests**; docs/workflow contracts passed **26 tests**,
with only the previously documented host-limited strict docs build excluded.
The final full nonincremental WebUI typecheck passed after all frontend edits.
The pre-refresh license regression failed as expected against the stale manifest;
source and metadata are pushed together only after the refreshed record passes.
