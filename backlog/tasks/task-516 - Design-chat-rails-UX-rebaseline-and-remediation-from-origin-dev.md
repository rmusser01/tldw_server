---
id: TASK-516
title: Design chat rails UX rebaseline and remediation from origin/dev
status: Done
labels:
- chat
- ux
- frontend
- spec
priority: high
documentation:
- Docs/superpowers/specs/2026-05-27-chat-rails-ux-rebaseline-design.md
- Docs/superpowers/plans/2026-05-27-chat-rails-ux-rebaseline-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-05-27-chat-rails-ux-rebaseline-design.md
- Docs/superpowers/plans/2026-05-27-chat-rails-ux-rebaseline-implementation-plan.md
- Docs/Reviews/CHAT_RAILS_UX_REBASELINE_2026_05_27.md
- Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/README.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design the clean origin/dev rebaseline workflow for verifying the /chat cockpit rails, re-running the /chat UX evaluation on the correct rail-enabled page, and planning remediation for remaining chat and extension handoff issues.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design spec exists and records the clean origin/dev branch/worktree workflow.
- [x] #2 Spec records why the no-siderails finding was branch-provenance related.
- [x] #3 User decisions are recorded: rail restoration/regression tests first, extension full-screen handoff targets /chat.
- [x] #4 Implementation plan exists and passed review.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Approved design moved to implementation planning. Plan prioritizes rail restoration/regression tests, real-server rail verification with screenshots/audit artifacts, sidepanel full-screen handoff to /chat, refreshed rail-enabled UX evaluation, and final verification/handoff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan review approved after fixing the SidepanelHeaderSimple Vitest hoisting snippet. This is a non-code planning task; Bandit is not applicable.

Task 1 provenance (2026-05-27):

```text
$ pwd
/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-rails-ux-rebaseline

$ git branch --show-current
codex/chat-rails-ux-rebaseline

$ git rev-parse --short HEAD
69a80b4b5

$ git rev-parse --short origin/dev
efe42fe0c

$ git merge-base --is-ancestor origin/dev HEAD
# no stdout
exit status: 0

$ git ls-files apps/packages/ui/src/components/Option/Playground | rg 'Cockpit|Rail|Inspector|CharacterControl'
apps/packages/ui/src/components/Option/Playground/CharacterControlRail.tsx
apps/packages/ui/src/components/Option/Playground/PlaygroundCockpitShell.tsx
apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx
apps/packages/ui/src/components/Option/Playground/PlaygroundRailSection.tsx
apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx
apps/packages/ui/src/components/Option/Playground/__tests__/CharacterControlRail.test.tsx
apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx
apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx
```

Audit artifact path: `Docs/Reviews/CHAT_RAILS_UX_REBASELINE_2026_05_27.md`; evidence directory README: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/README.md`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the chat rails UX rebaseline design. The spec records the branch-provenance cause for the missing-siderails observation, defines the clean origin/dev branch/worktree workflow, requires exact git provenance evidence, requires screenshot-backed rail-enabled /chat verification, and gates implementation fixes on a refreshed audit artifact.
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
