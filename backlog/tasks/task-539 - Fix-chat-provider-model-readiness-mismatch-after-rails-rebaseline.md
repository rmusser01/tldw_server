---
id: TASK-539
title: Fix /chat provider-model readiness mismatch after rails rebaseline
status: Done
labels:
- chat
- ux
- webui
- regression
priority: High
modified_files:
- apps/packages/ui/src/utils/chat-model-availability.ts
- apps/packages/ui/src/utils/__tests__/chat-model-availability.test.ts
- apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundChat.server-load-state.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and fix the /chat readiness mismatch left after the rails UX rebaseline where global provider setup state can report no provider configured while the runtime/model rails still present a selected chat model as ready. Keep scope limited to /chat WebUI provider/model readiness, banners, send blocking, and focused regressions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 /chat uses a consistent source of truth for no-provider banners, runtime readiness, and send blocking.
- [x] #2 A usable local/provider-qualified chat model does not get blocked solely because the aggregate provider status reports any_configured=false.
- [x] #3 The empty/unusable provider state still clearly blocks first send and shows setup guidance.
- [x] #4 Focused regression tests cover the mismatch and the empty-provider case.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root cause: mergeChatProviderStatusIntoModels treated aggregate any_configured:false as authoritative when provider status had no explicit configured provider rows, and downgraded model descriptors that already carried is_configured/provider_is_configured:true. That made /chat show the no-provider setup banner for callable local models.

Implementation: preserve descriptor-level configured:true unless an explicit provider status row overrides it. Explicit provider false still blocks, empty/unusable catalogs still show setup guidance.

Verification:
- RED: bun run test src/utils/__tests__/chat-model-availability.test.ts src/components/Option/Playground/__tests__/PlaygroundChat.server-load-state.test.tsx failed on the new stale aggregate provider status regressions.
- GREEN: same command passed 58/58 after the fix.
- Rails guard: bun run test src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx passed 19/19.

Bandit: not run; touched files are TypeScript UI/test files only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the /chat provider/model readiness mismatch by preserving explicit callable model descriptors when provider aggregate status is stale. Added regressions for the stale any_configured:false + usable local model case at the shared readiness utility and PlaygroundChat banner level. Verified focused readiness tests and the cockpit rails controls tests pass. Bandit skipped because this is TypeScript UI/test-only work.
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
