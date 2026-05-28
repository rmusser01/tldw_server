---
id: TASK-540
title: Improve /chat provider no-response failure copy
status: Done
labels:
- chat
- ux
- webui
- regression
priority: Medium
modified_files:
- apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the remaining /chat UX follow-up for richer response failure copy when a provider returns an empty or no-response result. Scope is limited to /chat WebUI runtime/composer failure messaging and focused regressions; do not broaden into sidebar/history architecture or extension draft transfer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Empty or no-response assistant results in /chat surface provider/actionable recovery copy instead of a generic silent/ambiguous state.
- [x] #2 The runtime rail still exposes regenerate/retry affordance context for empty responses.
- [x] #3 Existing provider error and visible assistant response behavior is preserved.
- [x] #4 Focused regression tests cover the new copy path and the unchanged non-empty path.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root cause: the runtime rail empty-assistant-response status used generic copy ('No response text returned') even when /chat knew the active provider/model route. That made provider no-response cases less actionable than other model/provider readiness states.

Implementation: PlaygroundRuntimeInspector now derives empty-response summary from the existing provider/model route label when available, e.g. 'openai:gpt-4.1-mini returned no response text.', and keeps a generic fallback when no route is known. The detail copy now recommends regenerating or choosing a different model if the provider keeps returning empty output. Regenerate wiring is unchanged.

Verification:
- RED: bun run test src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx failed on the new provider-route-specific empty-response expectations.
- GREEN: same command passed 55/55 after the fix.
- git diff --check passed.

Known baseline noise: Playground.cockpit-shell.test.tsx logs provider-status mock warnings because the test's mocked tldwClient does not implement getProvidersStatus; this warning existed in the focused run and did not fail the suite.

Bandit: not run; touched files are TypeScript UI/test files only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Improved /chat empty/no-response recovery copy in the runtime rail by naming the active provider/model route when available and preserving regenerate recovery. Added focused regressions in PlaygroundRuntimeInspector and Playground cockpit shell coverage. Verified red-green focused tests and git diff --check. Bandit skipped because this is TypeScript UI/test-only work.
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
