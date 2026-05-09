---
id: TASK-185
title: Write conversation context workflow design spec
status: Done
assignee:
  - Codex
created_date: '2026-05-09 19:41'
updated_date: '2026-05-09 19:42'
labels:
  - docs
  - ux
  - character-chat
  - worldbooks
  - chat-dictionaries
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the approved brainstorming design spec for a conversation-first context workflow that supports first-time and power users. The design must correct the prior character-exclusive framing: worldbooks and chat dictionaries are reusable conversation context assets that can attach to blank chats, character chats, workspace chats, and other non-character-focused conversations. The spec should be grounded in the May 9 character-card/worldbook/dictionary UX audit and should remain a design artifact only, not an implementation plan.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec is written under Docs/superpowers/specs with a dated conversation-context workflow filename.
- [x] #2 Spec states the corrected conversation-first product model and explicitly says character cards are optional context assets, not owners of worldbooks or dictionaries.
- [x] #3 Spec covers first-time and power-user workflows for context selection, verification, chat use, assignment clarity, workspace controls, and diagnostics.
- [x] #4 Spec captures reliability states for configured, active, matched, skipped, and blocked context and lists validation scenarios.
- [x] #5 Spec references the relevant May 9 UX audit evidence and remains scoped to design, with implementation deferred to a later plan.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Draft a design-only spec in Docs/superpowers/specs/2026-05-09-conversation-context-workflow-design.md using the user-approved brainstorming sections. 2. Ground the spec in Docs/Reviews/CHARACTER_CARD_WORLDBOOK_DICTIONARY_UX_AUDIT_2026_05_09.md, preserving the corrected model: conversations own runtime context; characters, worldbooks, dictionaries, workspaces, and providers are context inputs. 3. Include first-time and power-user workflows, concrete UI/process work packages, reliability states, and validation scenarios. 4. Self-review the spec for scope drift and terminology consistency because subagent dispatch is unavailable unless explicitly requested. 5. Verify the diff and commit the design artifact with the Backlog task update.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification: reviewed the full markdown spec, checked terminology against the approved conversation-first model, confirmed key evidence and diagnostics coverage with rg, and ran git diff --check successfully. Bandit skipped because the touched scope is documentation and Backlog metadata only. Spec-review subagent was not dispatched because this session only permits subagents when the user explicitly asks for delegated agent work; performed local self-review instead.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created Docs/superpowers/specs/2026-05-09-conversation-context-workflow-design.md as the approved design artifact for a conversation-first runtime context workflow. The spec explicitly frames character cards as optional context assets, preserves worldbooks and chat dictionaries as reusable assets for blank, character, workspace, and other conversations, and defines first-time and power-user workflows, work packages, reliability states, error handling expectations, and validation scenarios. Verification was documentation-focused: full spec review, terminology/evidence spot checks, and git diff --check. Bandit is not applicable to this docs-only change.
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
