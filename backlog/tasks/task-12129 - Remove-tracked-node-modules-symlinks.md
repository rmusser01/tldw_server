---
id: TASK-12129
title: Remove tracked node_modules symlinks
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean up tracked node_modules symlink entries so dependency installs are not represented in git. Scope is limited to node_modules entries only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] No tracked `node_modules` paths remain in git.
- [x] Existing ignore rules cover `node_modules` paths so they are not re-added.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Removed tracked symlink entries under `apps/packages/ui/node_modules` and `apps/packages/voice-assistant-sdk/node_modules` with `git rm -r`.

No `.gitignore` change was needed because `.gitignore` already contains `/apps/**/node_modules/`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the tracked node_modules symlink entries only. Verification: git ls-files -s -- ':(glob)**/node_modules/**' returned no output. Bandit skipped because this was non-code dependency artifact cleanup.

PR: https://github.com/rmusser01/tldw_server/pull/2624
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
