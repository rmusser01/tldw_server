---
id: TASK-167
title: Implement route-aware character-chat onboarding
status: Done
assignee: []
created_date: '2026-05-09 16:43'
updated_date: '2026-05-09 16:53'
labels:
  - character-chat
  - frontend
  - ux-audit
  - onboarding
dependencies:
  - TASK-159
  - TASK-161
  - TASK-166
documentation:
  - >-
    Docs/superpowers/plans/2026-05-09-character-chat-route-aware-onboarding-plan.md
  - Docs/superpowers/specs/2026-05-09-character-chat-ux-work-packages-design.md
  - Docs/Reviews/CHARACTER_CHAT_WEBUI_UX_AUDIT_2026_05_09.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the route-aware onboarding work package so first-run guidance respects character-chat intent from /characters and character-chat entry points while preserving the existing ingestion-first default for users with no stated intent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Existing first-run onboarding state sources are inventoried and the ingestion-first path remains the default with no character-chat intent.
- [x] #2 A failing test first captures character-chat route intent being lost or ignored during first-run onboarding.
- [x] #3 Users arriving from character-chat intent see character-chat aligned actions: create character, import character, choose model, and start character chat.
- [x] #4 Onboarding completion or skip behavior preserves or intentionally clears character-chat route intent without trapping returning users.
- [x] #5 Model-readiness blockers remain local and use the shared readiness contract rather than duplicating model logic.
- [x] #6 Focused unit/component tests and full UI typecheck are run and recorded.
- [x] #7 Bandit is skipped only if final touched scope remains frontend-only TypeScript/tests/docs/backlog.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Route-aware character-chat onboarding implementation plan completed in Docs/superpowers/plans/2026-05-09-character-chat-route-aware-onboarding-plan.md.

Implemented:
- Preserve first-run /characters route intent through WorkspaceConnectionGate via intent=character-chat and sanitized returnTo.
- Add shared onboarding route-intent utilities for safe internal return paths and character create/import action routes.
- Render a character-chat first-run lane on home onboarding when intent is present.
- Pass character-chat intent into OnboardingWizard / OnboardingConnectForm so the post-connect success screen prioritizes create/import/model/chat actions.
- Keep default ingestion-first onboarding unchanged when no character-chat intent is present.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Focused RED tests confirmed the prior behavior: /characters first-run setup navigated to plain /, OptionIndex lacked a character-chat onboarding lane/return navigation, and OnboardingConnectForm success screen lacked character-chat actions.

Focused GREEN verification: bunx vitest run src/components/Common/__tests__/WorkspaceConnectionGate.test.tsx src/routes/__tests__/core-route-identity.test.tsx src/components/Option/Onboarding/__tests__/OnboardingConnectForm.success-screen.guard.test.tsx --testTimeout=20000 passed, 18 tests.

Full UI typecheck: ../../tldw-frontend/node_modules/.bin/tsc --noEmit -p tsconfig.json --pretty false passed. The earlier bunx tsc attempt used transient TS 6 and failed on the repo baseUrl deprecation gate, so the pinned project compiler was used for verification.

git diff --check passed. Bandit skipped because touched scope is frontend TypeScript/tests plus plan/backlog documentation.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented route-aware character-chat onboarding. First-run users who start from /characters now enter home onboarding with intent=character-chat and a sanitized returnTo, see character-chat actions in both the first-run shell and post-connect success screen, and return to their interrupted character route after finishing. Default users without character-chat intent continue to see the existing ingestion-first onboarding. Verification passed for focused component tests, full UI typecheck with the project-pinned TypeScript compiler, and git diff hygiene. Bandit was skipped because the touched implementation scope is frontend TypeScript/tests plus plan/backlog documentation.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Implementation plan updated with executed outcomes and blockers
- [x] #8 Typecheck command and result recorded
<!-- DOD:END -->
