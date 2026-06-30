---
id: TASK-404
title: Address PR 1753 chat cockpit review comments
status: Done
labels:
- chat
- cockpit
- webui
- pr-review
priority: HIGH
references:
- https://github.com/rmusser01/tldw_server/pull/1753
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix actionable Gemini, CodeRabbit, and Qodo review comments on PR 1753 for the main WebUI /chat cockpit branch. Keep scope limited to the review findings and main chat cockpit-related files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Model selector keyboard accessibility review finding is addressed with coverage where applicable.
- [x] #2 PromptSelect Escape handling review finding is addressed without breaking focus recovery.
- [x] #3 Composition preview renders composition/status rows and resolves design-system labels at module scope.
- [x] #4 Model availability handles provider-qualified selections when only base model IDs are available, including unknown provider keys, with regression tests.
- [x] #5 Real-server cockpit E2E selection normalizes configured provider model IDs before deriving keys/assertions.
- [x] #6 Duplicate Backlog task IDs and completed-task unchecked checklists in PR 1753 are reconciled.
- [x] #7 Focused tests and static checks are run and PR review threads are resolved or commented with technical status.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added regression coverage for provider-qualified/base model availability, unknown provider prefixes, composition/status rows, and configured model ID normalization.
- Moved PromptSelect's global Escape listener to bubble phase, added dropdown-local Escape fallback handling for AntD-swallowed keydown paths, and preserved focus restoration.
- Kept `PlaygroundForm` production code unchanged for the model selector because the reviewed element is already a native `button`; added real-server keyboard proof that Enter opens the selector.
- Renumbered the visual/copy polish Backlog record to `TASK-405`, reconciled completed-task checklists, and updated the related plan references.
- Verification: focused Vitest suite passed 29 tests; real-server Playwright `/chat` cockpit spec passed 10/10 against the running backend with no mocked routes; targeted ESLint exited 0 with warnings only; `git diff --check` passed. Bandit skipped because touched code is frontend TypeScript/tests/docs/Backlog only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the actionable PR 1753 review comments for the main `/chat` cockpit slice: PromptSelect Escape handling is child-first with a non-capture global fallback, composition preview now renders composition/status rows, provider-qualified model availability handles base-only catalogs and unknown provider keys, real-server E2E normalizes configured provider model IDs, keyboard selector proof is covered, and duplicate Backlog metadata is reconciled.
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
