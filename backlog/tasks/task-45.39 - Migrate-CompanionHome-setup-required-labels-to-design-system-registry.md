---
id: TASK-45.39
title: Migrate CompanionHome setup-required labels to design-system registry
status: Done
assignee: []
created_date: '2026-05-13 14:55'
updated_date: '2026-05-13 18:20'
labels:
  - design-system
  - webui
  - extension
dependencies: []
references:
  - apps/packages/ui/src/components/Option/CompanionHome/CompanionHomePage.tsx
  - >-
    apps/packages/ui/src/components/Option/CompanionHome/__tests__/CompanionHomePage.test.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
documentation:
  - Docs/Design/tldw_web_design_system_contract.md
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the shared product-state design-system migration by routing CompanionHome card setup-required state labels through the canonical design-system state registry. Keep scope limited to CompanionHome setup-required card labels, focused test coverage, and removal of the migrated canonical-state-label baseline entries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CompanionHome setup-required card labels resolve through the design-system setup_required state with safe fallback copy.
- [x] #2 Focused CompanionHome coverage proves the visible setup-required card state can come from the design-system registry while preserving existing setup-band and card descriptions.
- [x] #3 The product-state guard baseline no longer contains CompanionHome setup-required canonical-state-label exceptions.
- [x] #4 Focused tests, product-state guard tests, the design-system product-state verifier, and diff checks pass or unrelated repo-wide failures are documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED: bunx vitest run src/components/Option/CompanionHome/__tests__/CompanionHomePage.test.tsx --reporter=dot first reached the intended failing assertion after dependencies were installed: the setup card state did not render the distinct design-system registry label and still used local hardcoded copy.

Implementation: CompanionHomePage now resolves setup_required through getDesignSystemState with DESIGN_SYSTEM_STATES fallback and uses that label for the setup-required card states. The focused test partially mocks the design-system registry to prove visible card labels come from the registry without asserting implementation call details. Removed the three CompanionHome setup-required canonical-state-label baseline entries.

Verification: focused CompanionHome test passed 8/8; product-state guard test passed 52/52; bun run verify:design-system-state passed with 500 allowed legacy exceptions, antd-product-state-import 481, canonical-state-label 19. The verifier also required refreshing current-dev AgentTasks AntD baseline IDs after PR #1634 moved those existing legacy findings; no AgentTasks code changed in this slice. git diff --check passed. bunx tsc --noEmit --pretty false --project tsconfig.json still exits 2 on existing unrelated UI TypeScript baseline errors, and touched-file filtering for CompanionHome, AgentTasks baseline, design-system-product-state-baseline, and TASK-45.39 returned no matches. Bandit skipped because this slice only touches UI TypeScript/JSON/Backlog metadata.

PR #1637 review fix: SETUP_REQUIRED_LABEL now uses optional chaining for both getDesignSystemState("setup_required") and DESIGN_SYSTEM_STATES.setup_required, with a final runtime fallback. The fallback avoids reintroducing the guarded canonical state-label literal in app source, so the product-state verifier remains the enforcement point.

PR #1637 review verification refresh: CompanionHome focused test passed 8/8; product-state guard test passed 52/52; bun run verify:design-system-state passed with the existing 500 allowed legacy exceptions; git diff --check passed; touched-file TypeScript filtering returned no diagnostics. GitHub review sweep showed Qodo no issues and CodeRabbit skipped auto-review on this non-default base. One Gemini thread remained to resolve after pushing the review-fix commit. Optional Full Suite jobs were cancelled in broad Admin/Audio matrix steps while required gates were passing.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated CompanionHome setup-required card labels to the design-system state registry and hardened the PR review fallback. The card state labels now resolve setup_required through getDesignSystemState, fall back through DESIGN_SYSTEM_STATES.setup_required with optional chaining, and have a final runtime fallback that does not reintroduce the guarded canonical state-label literal. The baseline also refreshes AgentTasks legacy AntD finding IDs from current dev so the product-state verifier remains passing after PR #1634. Focused tests, guard tests, the design-system verifier, and diff checks passed; full UI TypeScript remains blocked by unrelated existing baseline diagnostics with no touched-file matches. Bandit is not applicable for this UI-only TypeScript/JSON slice.
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
