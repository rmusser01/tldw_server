---
id: TASK-540
title: Improve /chat provider no-response failure copy
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-28 19:53'
labels:
  - chat
  - ux
  - webui
  - regression
dependencies: []
priority: medium
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

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root cause: the runtime rail empty-assistant-response status used generic copy ('No response text returned') even when /chat knew the active provider/model route. That made provider no-response cases less actionable than other model/provider readiness states.

Implementation: PlaygroundRuntimeInspector now derives empty-response summary from the existing provider/model route label when available, e.g. 'openai:gpt-4.1-mini returned no response text.', and keeps a generic fallback when no route is known. The detail copy now recommends regenerating or choosing a different model if the provider keeps returning empty output. Regenerate wiring is unchanged.

Verification:
- RED: bun run test src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx failed on the new provider-route-specific empty-response expectations.
- GREEN: same command passed 55/55 after the fix.
- git diff --check passed.

Known baseline noise: Playground.cockpit-shell.test.tsx logs provider-status mock warnings because the test's mocked tldwClient does not implement getProvidersStatus; this warning existed in the focused run and did not fail the suite.

Bandit: not run; touched files are TypeScript UI/test files only.
Review fix pass (PR #2095): review feedback was valid. The PR version used current providerRouteLabel/selectedModel for empty-response copy while the empty-response state was triggered by latestAssistantMessage. That could misattribute the failure after model switching. Playground now derives emptyAssistantResponseRouteLabel from latest assistant message metadata (modelId/modelName/name when provider-qualified), passes it separately to PlaygroundRuntimeInspector, and keeps the normal current-route display unchanged. PlaygroundRuntimeInspector now uses i18next interpolation with defaultValue '{{route}} returned no response text.' for localization-compatible copy.

Review-fix verification:
- RED: bun run test src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx failed when an empty OpenAI assistant message remained after the current model switched to Anthropic; banner incorrectly named the Anthropic route.
- GREEN: same command passed 55/55 after the fix.
- git diff --check passed.

PR check review: Full Suite failures inspected from run 26583264296 are backend/Python failures outside this TypeScript UI PR surface. Ubuntu 3.11 reported failing modules Audio and Audit, including Audit test_audit_db_deps.py::test_schedule_service_stop_clears_flag_on_failure; the job log also showed unrelated PostgreSQL COALESCE(boolean, integer) AuthNZ query errors. Frontend lint/build/playground checks had passed on the PR run.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Improved /chat empty/no-response recovery copy in the runtime rail by naming the active provider/model route when available and preserving regenerate recovery. Added focused regressions in PlaygroundRuntimeInspector and Playground cockpit shell coverage. Verified red-green focused tests and git diff --check. Bandit skipped because this is TypeScript UI/test-only work.

Review-fix pass for PR #2095 addressed all unresolved inline comments: empty-response route labels now come from the assistant message that triggered the banner instead of the current selection, and the route-bearing copy now uses i18next interpolation. Re-ran focused Playground tests (55/55) and git diff --check. Inspected failing Full Suite checks and found backend/Python failures outside this TypeScript UI PR surface.
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
