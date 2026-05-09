---
id: TASK-170.1
title: Refresh character-chat UX re-audit after P1 fixes
status: Done
assignee: []
created_date: '2026-05-09 18:38'
updated_date: '2026-05-09 18:50'
labels:
  - character-chat
  - ux-audit
  - frontend
  - puppeteer
dependencies:
  - TASK-173
documentation:
  - Docs/Reviews/CHARACTER_CHAT_WEBUI_UX_REAUDIT_2026_05_09.md
  - Docs/Reviews/assets/2026-05-09-character-chat-reaudit/puppeteer-states.json
  - >-
    Docs/Reviews/assets/2026-05-09-character-chat-p1-smoke/puppeteer-p1-smoke.json
  - >-
    Docs/superpowers/plans/2026-05-09-character-chat-post-implementation-reaudit-plan.md
  - >-
    Docs/superpowers/plans/2026-05-09-character-row-chat-implicit-model-fallback-plan.md
parent_task_id: TASK-170
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow up the completed post-implementation character-chat UX re-audit after the P1 route/model-readiness fixes landed. Use Puppeteer/Chrome-driver evidence, not Computer Use, to rerun first-time and returning-user character-chat workflows and update the re-audit report so its findings reflect the current fixed build. Keep the default corrupted user DB untouched; use an isolated temporary backend profile.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 First-time direct /characters and explicit character-chat onboarding intent are re-tested with Puppeteer/Chrome evidence after the P1 fix.
- [x] #2 Returning-user search/edit/row Chat as workflow is re-tested and the selected-character/no-model behavior is documented from browser evidence.
- [x] #3 The re-audit report and artifact directory are updated or appended with the post-P1 findings, distinguishing resolved P1s from remaining blockers such as missing LLM provider state.
- [x] #4 Verification commands are recorded, including JSON parse/shape checks and whitespace checks for updated docs/artifacts.
- [x] #5 Bandit is run for touched Python code or explicitly documented as not applicable if the touched scope remains frontend/docs/test artifacts.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Post-P1 Puppeteer refresh initially showed direct `/characters` and explicit character-chat onboarding were fixed, but row `Chat as...` still navigated to Companion Home because full-chat readiness inherited the quick-chat first-model fallback.

Created child task TASK-170.1.1 to fix the row-action gap, then reran the Puppeteer/Chrome walkthrough. Final evidence now shows the row action stays on `/characters` with the selected-character model blocker.

Updated `Docs/Reviews/CHARACTER_CHAT_WEBUI_UX_REAUDIT_2026_05_09.md` and refreshed the screenshot/JSON artifacts under `Docs/Reviews/assets/2026-05-09-character-chat-reaudit/`.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refreshed the character-chat re-audit after the P1 route fixes and corrected the remaining row-action live gap through child task TASK-170.1.1. Final Puppeteer evidence shows direct `/characters` works, explicit character-chat onboarding intent is preserved, character creation succeeds, and row `Chat as...` stays in Characters with the selected-character model blocker. Remaining documented issues are search count semantics, no-provider message-generation coverage, connected setup copy priority, and console/request noise. Bandit was skipped because no Python files changed.
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
