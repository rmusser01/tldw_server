---
id: TASK-13958
title: Align standalone HTML parser smoke with tinycss2 1.5.1
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #2977 bumps tinycss2 to 1.5.1 but the existing parser smoke asserts 1.4.0. Update only that expected version, keeping real HTML/CSS validation assertions. PR: https://github.com/rmusser01/tldw_server/pull/2977
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The dependency smoke and standalone HTML validator suite pass with tinycss2 1.5.1.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
At refreshed PR head025d6c40, reproduced existing test failure: installed tinycss2 1.5.1 did not equal the stale expected1.4.0. Changed that single constant to match pyproject; HTML fixture, title and indexable_text assertions remain intact. No production code changed.

Verification: 262 standalone HTML validator and dependency smoke tests passed with tinycss2 1.5.1, html5lib1.1 and bleach6.4.0 in a disposable venv. Independent review also ran25 Slides export checks with unchanged production code, including CSS preservation, escaped/nested URL rejection and fail-closed validation. Targeted pipcheck clean. Ruff and py_compile passed. Scoped Bandit:0 findings,0 errors,14 LOC; B101 excluded for intentional pytest assertions. git diff --check passed. Independent final review approved exact one-line correction.

Full application suites were not run. Hosted required trusted/main check remains pending. This is a mechanical expected-version alignment for a Dependabot pin; no additional AI-authored runtime or design changes. CLI writes used newline-preserving temporary-file mode to avoid Windows CRCRLF translation, retaining standard CLI validation/atomic writes. Evidence: TEMP/tldw-pr-batch2-20261030 and TEMP/tldw-pr-python-batch2-20261030. PR: https://github.com/rmusser01/tldw_server/pull/2977
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Aligned the existing tinycss2 smoke assertion with the Dependabot 1.5.1 pin. Parser/sanitizer behavior and export checks pass, with no runtime change. Ready for the required hosted gate and merge.
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
