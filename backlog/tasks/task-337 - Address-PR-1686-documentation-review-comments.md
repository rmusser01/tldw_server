---
id: TASK-337
title: Address PR 1686 documentation review comments
status: In Progress
assignee: []
created_date: '2026-05-14 05:46'
updated_date: '2026-05-14 05:51'
labels:
  - acp
  - docs
  - review-fix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1686'
  - 'https://github.com/rmusser01/tldw_server/pull/1686#discussion_r3239357872'
  - 'https://github.com/rmusser01/tldw_server/pull/1686#discussion_r3239357916'
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
- [x] #4 PRD compatibility field paths use fully qualified compatibility.* JSON paths.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Addressed both Gemini review threads: linked ACP maturity dependency issue references and changed the Agent Registry UI verification command to a subshell. Verification: git diff --check passed; targeted rg guard confirmed the old unlinked dependency and non-subshell command patterns are absent. Bandit skipped because this slice only changes Markdown documentation.

Resolved Gemini review threads after pushing c8c6f836d: linked dependency issue references thread https://github.com/rmusser01/tldw_server/pull/1686#discussion_r3239357872 and subshell verification-command thread https://github.com/rmusser01/tldw_server/pull/1686#discussion_r3239357916.

Qodo posted a new actionable review thread after the Gemini fixes: PRD compatibility fields need fully qualified compatibility.* paths.

Addressed Qodo compatibility-field thread by changing the PRD contract row to compatibility.documented_unverified_agents, compatibility.live_certification_required, and compatibility.docs_url. Verification: git diff --check passed; targeted rg confirmed the unqualified PRD field paths are absent and the qualified paths are present.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed all actionable PR #1686 review comments by linking ACP dependency issue references in the readiness prose, wrapping the Agent Registry UI verification command in a subshell, and qualifying PRD compatibility field paths under compatibility.*. Verification: git diff --check and targeted rg guards passed. Skips: Bandit and backend/frontend tests were skipped because the changes are Markdown-only. Gemini threads were replied to and resolved; the Qodo compatibility thread is ready to resolve after this commit is pushed.
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
