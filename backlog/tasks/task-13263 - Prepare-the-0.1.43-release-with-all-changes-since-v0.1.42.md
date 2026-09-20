---
id: TASK-13263
title: Prepare the 0.1.43 release with all changes since v0.1.42
status: In Progress
assignee: []
created_date: '2026-09-20 19:56'
updated_date: '2026-09-20 22:07'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare a reviewed release candidate based on v0.1.42 and frozen dev d72b1d2850ea947b6d12cac19f6b95867b68a580. Preserve 0.1.42 release fixes, reconcile released main into dev, inventory every new commit and merged PR, update release metadata and protected source records, and open a draft release PR. Track outstanding 0.1.42 publication verification in TASK-13013.3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The candidate includes v0.1.42 and frozen dev as ancestors, with reviewed conflict resolutions.
- [ ] #2 Changelog and release notes cover all post-0.1.42 changes, with an exhaustive commit inventory.
- [ ] #3 Version metadata, documentation and protected-source records are consistent and verified.
- [ ] #4 A draft PR and release plan record checks, publication state and remaining human decisions.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Release plan: Docs/superpowers/plans/2026-09-20-release-0.1.43-plan.md. Frozen delta: 616 commits and 19 first-parent merges after v0.1.42. Five conflicts resolved retaining transport security and dev persistence. Independent static review found no concrete merge regression. Notes regression exposed obsolete authority mock, removed; 26 Notes tests pass and 107 other merge regressions pass. Historical license manifest check updated to pin immutable published bytes.

2026-09-20 PR2971 recovery synchronization: merge approved main recovery cd2dbc792b8888555abac5c5c9eafa7a43d9b0e4 into existing sync head4b995abe6eceee9ece76a62fdc9d3dd7545c3e7c in isolated /private/tmp worktree. Preserve both histories and all application/package/legal/version inputs. Only three AuthNZ fixture registrations change, alongside inherited recovery task/plan records. Baseline plugin isolation reproduces1failed/5passed; merged regression, combined AuthNZ/Admin_Webhooks collection, scoped lint/Bandit and exact tree-delta checks pending. This does not merge2971 or waive its human Change summary gate.

PR2971 recovery validation found one dev-only direct AuthNZ.conftest plugin registration in Ingestion_Sources/test_service_postgres.py:7 after the three main recovery bridges. Equivalent one-line authnz_full_fixtures correction authorized and matches release candidate2972 existing repair. Existing plugin-isolation regression provided red evidence before this fix. No production or test semantics change.

PR2971 recovery sync verified: plugin-isolation6passed; combined AuthNZ/Admin_Webhooks/Ingestion_Sources collection2559tests without errors; SQLite webhook delivery behavior15passed/3PostgreSQLcases deselected; Ruff/diff checks pass. Scoped Bandit retains exactly2preexisting synthetic-fixture B105 findings and0new findings/errors; Ingestion Sources scope clean. Exact comparison against prior sync head proves four one-line fixture-bridge substitutions are the only executable-source changes, with application/package/version/legal inputs unchanged. Main recovery merges cleanly retaining task history. Ready to commit/push2971 update; required exact-head CI and requester-written Change summary remain pending, no2971merge or waiver authorized.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
