---
id: TASK-337
title: Address PR 1686 documentation review comments
status: In Progress
assignee: []
created_date: '2026-05-14 05:46'
updated_date: '2026-05-14 05:48'
labels:
  - acp
  - docs
  - review-fix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1686'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the still-actionable Gemini review threads on PR #1686 for the ACP admin reporting closeout docs. Keep the change scoped to documentation navigability and command hygiene.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Dependency issue references in the ACP production readiness prose are linked consistently.
- [x] #2 The Agent Registry UI verification command in the admin reporting readiness row uses a subshell so the working directory change is local to that command.
- [x] #3 Docs-only verification and review-thread closeout are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Addressed both Gemini review threads: linked ACP maturity dependency issue references and changed the Agent Registry UI verification command to a subshell. Verification: git diff --check passed; targeted rg guard confirmed the old unlinked dependency and non-subshell command patterns are absent. Bandit skipped because this slice only changes Markdown documentation.
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
