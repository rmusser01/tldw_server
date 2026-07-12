---
id: TASK-12948
title: Replay PR 2702 delta onto latest dev
status: In Progress
labels:
- release
- rebase
- review
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a fresh dev-targeted branch from current origin/dev, replay only the PR #2702-specific commits, resolve conflicts in favor of newer dev behavior where appropriate, validate the resulting net delta, and open a PR targeting dev. Do not modify or open a PR against main.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Branch is based on the latest origin/dev and contains no main-targeted correction
- [ ] #2 Only PR #2702-specific changes still missing from current dev are included
- [ ] #3 Conflicts are resolved against newer dev behavior with minimal changes
- [ ] #4 Focused backend/frontend/workflow validation passes
- [ ] #5 A pull request is opened targeting dev
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created codex/pr-2702-dev-rebase directly from origin/dev at d28c16bfa3 and replayed the eight PR #2702-specific commits in order. Git reported no textual conflicts. Semantic overlap with newer dev was limited to pyproject.toml; the newer dev dependency changes remain intact and the branch changes only the package version to 0.1.40. Net delta: 35 files, branch ancestry 0 behind / 8 commits ahead of origin/dev. Verification: focused frontend Vitest 53/53 passed; focused backend pytest 69/69 passed; frontend extension compile passed; changed extension Playwright test discovered; workflow YAML and stable concurrency assertion passed; Bandit reported zero findings; git diff --check passed; release metadata is consistently 0.1.40. Full frontend typecheck reports a pre-existing origin/dev error in untouched QuickIngestWizardModal.tsx (Modal styles overflowY typing); no unrelated fix is included.
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
