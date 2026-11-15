---
id: TASK-15494
title: Prepare and publish v0.1.47 from frozen dev
status: Done
labels:
- release
- documentation
references:
- Docs/superpowers/plans/2026-11-11-release-0.1.47-plan.md
- https://github.com/rmusser01/tldw_server/pull/3454
- Docs/Evidence/PR3454-codeql-frontend.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Promote reviewed dev through PR #3443 to main, preserving main-only updates and publishing coherent server release metadata and changelog after review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Release candidate preserves the frozen main and dev histories and passes release metadata checks.
- [x] #2 Changelog summarizes notable shipped changes and upgrade requirements.
- [x] #3 PR #3454 preserves the requester’s Change summary and approved release and Countdown dates, and CI passes before merge.
- [x] #4 The reviewed main merge is tagged and published, server artifacts are verified, and main is synchronized back to dev.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. This promotion follows the existing release and licensing process without changing policy or architecture. Candidate cutoff: dev 04905e02eafc34f3b8d82d30e013a81c660028d0 and main 25dbf5e7b26b973ae78ac302350589e720eb7980.
PR #3454 retains the requester’s original Change summary. The requester approved the 2026-11-12 release date and the 2028-11-12T12:00:00Z Countdown start, approved the CodeQL and pre-commit fixes, and waived additional human rationale for this release.
The reviewed starting candidate passed 333 focused tests and built the server wheel and sdist with GPL-3.0-only metadata and no protected frontend files. The CI repair separates the backend classifier checkouts by event and corrects whitespace in six files. The repair passed 336 focused tests and scoped pre-commit. Bandit found only the test assertion in new Python lines. The protected source manifest now includes the final newline of the extension document. Final GitHub CI, publication and dev reconciliation are pending. Full release UAT has not been run.
GitHub pre-commit passed on b8ceb43acf. CodeQL still flagged the conditional checkout, so the follow-up removes the inactive workflow-run path from backend-required. Pull requests still wait for the trusted license check, and manual diagnostics keep using their selected commit. All 357 focused workflow and ratchet tests passed with no new Bandit findings. GitHub CI must pass again on the final commit.
The requester approved the focused frontend CodeQL follow-up. Reviewed all 37 blocking frontend annotations against the exact scan. The candidate validates Audio Studio ticket destinations, constrains the workspace diagnostic link to the runtime config route, and replaces a dynamic test regex with literal matching. 588 focused Node 20 tests passed. The evidence records the separate WebUI credential persistence path and the reviewed test and cache traces. No alerts were dismissed. This is a bounded repair to existing contracts, with no new ADR required.
GitHub CI run 43266259032 exposed a mismatch in the required workflows’ concurrency expressions after the backend checkout repair. Restored the shared group expression and removed the special case in the license workflow contract test. Backend triggers, checkout, and the trusted license check are unchanged. Both workflow contract suites passed locally (116 tests). Fresh GitHub CI is required before merge.
CI run 43459486430, job 132126056378 reproduced a separate checkout-boundary test failure on dccdda9895. The test now excludes only the shared concurrency group from its event-source inspection. The workflow YAML, direct trigger set, no-admission checks, and checkout-source guard remain covered. Bandit findings matched the baseline.
Local verification for the checkout-boundary contract: CI tests and shared concurrency contracts passed (1100 passed, 6 skipped), and actionlint and scoped pre-commit passed. A local negative check confirmed the test rejects workflow_run references elsewhere in the workflow.
This release task was TASK-15491 on main and in v0.1.47. Dev had already assigned TASK-15491 to the evaluation judge-call work, so the reconciliation preserves both records under distinct IDs.
Confirmed full CI 43530216167, CodeQL, pre-commit, and the trusted frontend license check passed on accc1cd14c. PR #3454 merged to c786f6fdd8 with its original Change summary. The annotated v0.1.47 tag points to that commit, and its published GitHub release body matches the merged changelog section. The release-triggered Docker run is 43602937843. The PyPI push run did not appear, so the existing PyPI workflow was run manually on the verified main commit as 43603207649. Artifact and dev PR verification are pending.
The dev reconciliation keeps dev's post-cutoff changes and retains both branch histories. Locally, 309 workflow contract tests and 15 license-policy tests passed, along with actionlint, scoped pre-commit, and Backlog ID checks. All 8,300 protected source hashes match the published tag. Bandit found only low-severity findings in the synchronized Python files and reported no errors.
GitHub Actions runs 43602937843 (Docker) and 43603207649 (production PyPI) succeeded on c786f6fdd819a506df6bc33389e2435c99bfe2c7. The public tldw-server 0.1.47 wheel and source distribution matched their PyPI SHA-256 digests and contained backend files only. The three GHCR 0.1.47 indexes were reachable at tldw_server, tldw_server-worker, and tldw_server-audio-worker. All 8,300 protected frontend source hashes were verified at the release tag.
All eight required checks, including frontend-license-policy/trusted/dev, passed on cf2ea01aa65319e25ebe2902dc379f8f2649c3b6. PR #3466 merged to dev as e5f1a394f9f47d693a81afee3d2a1c4e932fa9dc. Dev contains release main c786f6fdd8 and its own prior tip d6bba5fca1, including the three native setup backlog updates. Both TASK-15491 (evaluation) and TASK-15494 (release) are present on dev. The tagged record remains TASK-15491. Full release UAT was outside the verification recorded here.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Published v0.1.47 from the reviewed main merge, verified the public PyPI and GHCR server artifacts, and reconciled the release to dev through PR #3466 while retaining the newer dev history. The release date is 2026-11-12 and the frontend license Countdown starts 2028-11-12 at 12:00 UTC.
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
