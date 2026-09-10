# PR #2761 release refresh implementation plan

**Tracking:** TASK-13013.3; historical release record TASK-12988 on this branch.
**Goal:** Refresh the existing draft 0.1.42 candidate with current dev and produce reviewable release metadata and validation evidence before publication.
**Architecture:** Preserve the existing PR branch and merge the frozen dev source without rewriting history. Tag only the eventual reviewed main merge commit; do not run `make release-patch` after the pre-bump because it would select 0.1.43.
**Spec:** `Docs/superpowers/specs/2026-07-26-release-0.1.42-reviewed-metadata-design.md`, superseded for the source/date values below.

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
**Status:** In Progress

Run tests with the project virtual environment activated and execute from this worktree. Keep the human Change summary placeholder intact. Local focused tests do not replace the required remote backend, security, coverage, frontend, E2E, container, and trusted license gates on the final candidate.

## Stage 4: Publication gate and handoff
**Goal:** Record the remaining release prerequisites and the exact publication path.
**Success Criteria:** No merge, tag, or publication is represented as complete without corresponding evidence.
**Tests:** Recheck PR head, gates, human review, tags and releases before publication; verify package/container provenance after publication.
**Status:** Not Started

- The repository requires a requester-authored Change summary explaining what changed and why; the PR currently contains only a placeholder.
- The completed frontend legal record requires requester review before merge.
- TASK-13013.7 (supply chain), TASK-13013.8 (tenant/data lifecycle), TASK-13013.9 (capacity/soak), and TASK-12116 (frontend safety) remain open dependencies; resolve them or record explicit release-scope decisions before claiming core release readiness.
- GitHub currently lists v0.1.38 as the latest published release; a remote query for v0.1.4-prefixed tags returned no entries. Historical 0.1.39–0.1.41 metadata is not proof of publication. Reconcile this distribution gap and select a verified deployed rollback target before release.
- Back up persistent data before upgrading: the accumulated train includes AuthNZ, conversation, notes sync, presentation, personal-context, and webhook schema changes. Do not assume an old binary can read migrated databases. Use `Docs/Deployment/Production_Reference_Deployment.md` and `Docs/Admin_Webhooks_Migration_Runbook.md` for operational preparation.
- After approved merge, fetch main and require it to equal PR #2761's merge SHA. Tag that exact commit as v0.1.42 and create the GitHub Release from its curated changelog with `--verify-tag`. Verify automatic PyPI and Docker publications, then sync released main back to dev.

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
