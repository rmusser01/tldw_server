---
id: TASK-12162
title: Prepare 0.1.36 corrective release
status: Done
labels:
- release
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Prepare a 0.1.36 corrective release from current dev to main after the accidental 0.1.35 release merge. Scope: bump version metadata, add changelog/README release summary, validate release contracts, commit, push, and open a main-bound PR.
Prepared 0.1.36 corrective release metadata: bumped pyproject.toml, FastAPI app metadata, README release line, and Docs/mkdocs.yml from 0.1.35 to 0.1.36; added CHANGELOG.md 0.1.36 corrective entry covering PR #2653 and dev/main sync repair; added README 0.1.36 corrective release summary and marked 0.1.35 as superseded. Validation: git diff --check passed; release docs + PyPI workflow contract tests passed 17/17; py_compile tldw_Server_API/app/main.py passed; Bandit on tldw_Server_API/app/main.py wrote /tmp/bandit_release_0_1_36.json with zero results.
Opened PR #2655 against main: https://github.com/rmusser01/tldw_server/pull/2655. Branch codex/release-0.1.36-corrective contains the 0.1.36 corrective release metadata and validation evidence.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Prepared and opened PR #2655 for the 0.1.36 corrective release. The branch bumps canonical release metadata to 0.1.36, adds CHANGELOG/README corrective release notes, and records validation: git diff --check, release docs/PyPI workflow contracts 17/17, py_compile main.py, and Bandit on main.py with zero results.
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
