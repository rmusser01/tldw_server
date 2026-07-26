---
id: TASK-12988
title: Prepare and cut the 0.1.42 release from current dev
status: In Progress
assignee: []
created_date: '2026-07-26 13:05'
updated_date: '2026-07-26 15:19'
labels:
  - release
  - operations
  - ci
dependencies: []
references:
  - Docs/Development/Release_Process.md
  - Docs/Release_Checklist.md
  - TASK-12986
  - 'https://github.com/rmusser01/tldw_server/pull/2761'
documentation:
  - Docs/superpowers/specs/2026-07-26-release-0.1.42-reviewed-metadata-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare a normal patch release by integrating current origin/main into current origin/dev in an isolated worktree, adding the complete 0.1.42 changelog, public release notes, and visible version metadata to the reviewed release PR, merging it to main only after repository merge gates are satisfied, tagging the reviewed main merge commit as v0.1.42, publishing the GitHub Release, syncing main back to dev, and then unblocking TASK-12986 license-first CI activation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The release branch contains current origin/dev and current origin/main with conflicts resolved without dropping reviewed changes.
- [x] #2 Focused release, CI workflow, license gate, security, and diff verification pass or exact blockers are recorded.
- [ ] #3 The release PR targets main and includes the required requester-authored Change summary before merge.
- [ ] #4 main is synchronized back into dev before TASK-12986 license-first activation continues.
- [ ] #5 The reviewed main merge commit is tagged v0.1.42, the GitHub Release is published from the curated changelog entry, and publication is verified.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve frozen dev/main ancestry. 2. Add and verify the approved 0.1.42 protected-source release record (source 0f3983788c413e0d17ffe7eabe8cff4a9f6ae723; release 2026-07-26; Countdown 2028-07-26T12:00:00Z). 3. Add changelog, release notes, and all version surfaces. 4. Push PR #2761 and require exact-head trusted license success, requester legal-record review, and requester-authored Change summary. 5. Merge, tag the reviewed main commit, publish server-only artifacts, sync main to dev, and resume TASK-12986.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Branch integration completed at merge commit 2e63d09aa0. Frozen inputs were origin/dev=0f3983788c413e0d17ffe7eabe8cff4a9f6ae723 and origin/main=d9c245ac14c40df855d1ab6cd19b3c137b16b47b. The merge preview and real merge produced five conflicts, all in the license bootstrap/actionlint surfaces. Each was resolved to the newer origin/dev version because dev contains the completed live rollout plus subsequent TASK-12986 hardening; the resolved files are byte-for-byte identical to origin/dev. The merge commit therefore changes ancestry only and retains all current dev content.

Focused local release verification on merge head 2e63d09aa0 passed: 211 pytest tests across the release helper, required workflow contracts, license-first workflow/admission contracts, trusted frontend license workflow/classifier, and release workflow contracts (2 pre-existing warnings). Pinned Actionlint 1.7.12 reported no findings across all workflows. Bandit scanned 363 LOC in Helper_Scripts/ci/check_frontend_license_gate.py and Helper_Scripts/ci/license_first_admission.py with errors=[] and results=[]. Both origin/dev and origin/main are ancestors of the merge head; git diff --check is clean. No production runtime file was changed by conflict resolution, so broader release security coverage is delegated to the security-required PR gate.

Release PR #2761 opened as a draft against main from exact head a9818a21f8b5bb100d4899e60b86728ca48e5590. Its Change summary section intentionally contains only a requester placeholder; the repository policy requires Robert to replace it in his own words before the PR can be marked ready or merged.

Requester selected the reviewed-metadata release approach: PR #2761 will carry the complete 0.1.42 changelog, release notes, and visible version surfaces. Because the existing patch helper would calculate 0.1.43 after that pre-bump, publication will tag the reviewed main merge commit as v0.1.42 and create the GitHub Release from the curated changelog entry.

Requester approved the release-specific legal dates on 2026-07-26. The protected trees on the release branch are identical to frozen dev source 0f3983788c413e0d17ffe7eabe8cff4a9f6ae723. Design and implementation plan now include LICENSES/releases/0.1.42, source-only protected publication, and a human legal-record review gate.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
