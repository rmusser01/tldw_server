---
id: TASK-188
title: Implement client-managed conversation context workflow
status: In Progress
assignee:
  - Codex
created_date: '2026-05-09 20:05'
updated_date: '2026-05-09 20:34'
labels:
  - implementation
  - frontend
  - backend
  - character-chat
  - worldbooks
  - chat-dictionaries
  - conversation-context
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-09-conversation-context-workflow-implementation-plan.md
  - Docs/superpowers/specs/2026-05-09-conversation-context-workflow-design.md
  - Docs/Reviews/CHARACTER_CARD_WORLDBOOK_DICTIONARY_UX_AUDIT_2026_05_09.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved client-managed Conversation Context implementation plan. The client composes effective context from server primitives; the server only provides composable processing, settings, prompt pieces, and validation primitives.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Server primitive tests cover explicit worldbook processing without character context, ordered dictionary processing, conversationContext settings preservation, and invalid asset ID domain errors
- [x] #2 Client context composer assembles preview and send payload from server primitives without a monolithic backend context-preview endpoint
- [ ] #3 Composer popover replaces or evolves the existing chat-composer character picker and keeps character selection as one slot
- [ ] #4 Worldbooks and dictionaries remain conversation-scoped and usable without a selected character
- [ ] #5 Targeted pytest, Vitest, Playwright/browser checks, Bandit touched-scope scan, and git diff checks are recorded or concrete skips documented
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Execute Task 1 from Docs/superpowers/plans/2026-05-09-conversation-context-workflow-implementation-plan.md: server primitive audit and hardening. 2. Follow TDD: write failing tests first, then minimal primitive changes. 3. Continue through client composer, hook/send integration, composer popover, asset selection, browser validation, and docs closeout in separate commits.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 1 backend primitive hardening complete. Added integration coverage for explicit worldbook processing without a character, ordered dictionary_ids processing, conversationContext settings preservation, and invalid worldbook/dictionary ID domain errors. Implemented dictionary_ids on /api/v1/chat/dictionaries/process, explicit ID validation, ordered processing, and worldbook missing-ID validation. Verification: pytest tldw_Server_API/tests/Character_Chat_NEW/integration/test_conversation_context_primitives.py -v (4 passed); pytest tldw_Server_API/tests/Chat/unit/test_chat_dictionary_endpoints.py -q (54 passed); bandit touched backend paths -> 0 findings; git diff --check -> clean.

Task 2 client composer core complete. Added typed conversation-context models, settings normalization for nested conversationContext plus legacy chat_dictionary_ids fallback/mirror, and a pure composer that calls dictionary processing before worldbook processing and returns shared previewSections/providerMessages from one composition object. Verification: bunx vitest run ../packages/ui/src/services/conversation-context/__tests__/conversationContextComposer.test.ts ../packages/ui/src/services/conversation-context/__tests__/conversationContextSettings.test.ts --config vitest.config.ts (8 passed); bunx tsc --noEmit -p ../packages/ui/tsconfig.json --pretty false (passed); git diff --check (clean).

Task 3 hook/send integration slice complete. Added useConversationContextComposition to compose one shared preview/send object from client-managed settings, debounce background preview composition, compose immediately at send time, and preserve the authored user message while sending transformed model-only text/history overrides. Wired the Sidepanel chat form to read conversationContext settings and pass requestOverrides into the normal/persona/image-backed normal chat send paths; local chat settings now preserve conversationContext and legacy chat_dictionary_ids keys. Verification: bunx vitest run ../packages/ui/src/hooks/chat/__tests__/useConversationContextComposition.test.tsx ../packages/ui/src/services/conversation-context/__tests__/conversationContextComposer.test.ts ../packages/ui/src/services/conversation-context/__tests__/conversationContextSettings.test.ts --config vitest.config.ts (12 passed); bunx tsc --noEmit -p ../packages/ui/tsconfig.json --pretty false (passed); git diff --check (clean). Known remaining scope: visible composer popover/asset selectors and character complete-v2 context pass-through remain for subsequent tasks; this slice intentionally keeps server composition primitive-only.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
