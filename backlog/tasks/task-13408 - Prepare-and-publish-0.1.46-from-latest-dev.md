---
id: TASK-13408
title: Prepare and publish 0.1.46 from latest dev
status: In Progress
created_date: 2026-10-02 00:22
labels:
- release
priority: high
updated_date: 2026-10-02 00:55
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Release remote dev 3304f6cf0c2cab8836c6a73308d7210534a759d9 through PR #3073 as 0.1.46 using existing metadata, protected-source records, CI and publication workflows. No new collectors or broader certification gates; preserve unrelated local work.
<!-- SECTION:DESCRIPTION:END -->
## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Version, changelog and protected-source records agree on the frozen source
- [ ] #2 Existing required checks pass on the release candidate
- [ ] #3 Merge main, publish v0.1.46 using existing workflows and synchronize main into dev
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Freeze latest dev and prepare release metadata/source record. 2. Run existing release/licensing tests and PR CI; fix actual release failures. 3. Publish reviewed main merge and synchronize dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
CLI task creation failed with Maximum call stack size exceeded; using official Python MCP with explicit isolated project path.
Prepared0.1.46 from5f3ed81e88ec44750a5baa7838f7c69672e32694. Protected source: `5f3ed81e88ec44750a5baa7838f7c69672e32694`. Protected manifest SHA-256: `2f10da6c5c91e30356d15568d427e3e8dc2750123ebd48d1e5c2580cf958ab47`.7451files; proposed date2026-10-01/Countdown2028-10-01T12:00:00Z. Existing helper baseline46tests pass. Added only version/changelog/license records, compact merged-change inventory and release plan; existing publication workflows unchanged. No v0.1.46 tag/release exists. Python minimum changed by included dev commits to3.12, documented as upgrade requirement.
80 existing release/helper/Makefile/docs/licensing tests passed including strict MkDocs and TLDW_VERIFY_RELEASE_SOURCE=1 checkout equality. Touched main.py version change compiles; Bandit0findings/errors. Ruff9baseline findings are unchanged; no unrelated formatting repair. Metadata tests use existingPython3.11venv; runtimePython3.12 validation uses existingCI. Preparing main releasePR; no application features/workflow changes/new certification gates.
Opened release PR https://github.com/rmusser01/tldw_server/pull/3074; attached to release chat. Latest dev moved to 3304f6cf0c2cab8836c6a73308d7210534a759d9 via backlog-only PR #3073. Incorporate that merge and update frozen source/inventory before final CI. Protected source content remains unchanged.
Merged latest dev PR #3073 (backlog-only), refreshed protected source revision to 3304f6cf0c2cab8836c6a73308d7210534a759d9 and inventory to 980 commits. Protected manifest unchanged. All 80 existing release/docs/licensing/helper contracts passed again with source verification enabled.
Requester instructed: 'update the changelog and use that'. Use expanded 0.1.46 changelog for PR Change summary and GitHub release notes; requester explicitly chose this instead of a separate human-written summary. Do not falsely label the text human-authored. CI currently reports three CodeQL JavaScript alerts (2695-2697); investigate under the existing release checks stage.
Expanded 0.1.46 CHANGELOG with merged Persona Buddy/startup, VN recovery, authenticated Email uploads, encrypted agent-message storage slice, Python 3.12/FastAPI upgrade, Resource Governance, Chat/PostgreSQL/auth/workflow and CI/platform repairs. Reused it for PR Change summary and prepared GitHub release notes as requester directed. Existing changelog/docs/helper tests: 62 passed, four inherited warnings. No Python changes; prior Bandit zero findings remains applicable. Independently traced CodeQL 2695-2697: no code change justified. All production handoff scopes explicitly remove apiKey (useHomeMilestoneScope); SHA256 journey expressions assert ownership fingerprints, not password verifiers. Recorded false-positive dismissals with rationale. Generic FNV scope migration is excluded because it is unrelated and would invalidate persisted sessions. Legal-date review remains unresolved.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
