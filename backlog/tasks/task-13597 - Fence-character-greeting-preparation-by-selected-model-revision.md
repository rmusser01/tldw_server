---
id: TASK-13597
title: Fence character greeting preparation by selected model revision
status: Done
labels:
- chat
- ux
- bug
dependencies:
- TASK-13593
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A same-value model selection, or A-to-B-to-A selection, during the existing Scene read or draft-retention await can keep a character greeting current because the shared greeting guard captures selectedModel value but omits its existing selectedModelRevision. Qualify causal controls against the real option setter and reuse the existing shared guard so Full and Compact callers reject retired model intent without creating or submitting. Preserve original draft, Scene, account and greeting seed for unchanged positive paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Real setter same-value/ABA intent controls fail before the repair and refuse submit after it at actual existing Scene/retention await boundaries; unchanged held preparation still submits exact original Scene/scope/seed and retains draft.
- [x] #2 Smallest shared repair uses existing selectedModelRevision authority and reaches Full and Compact through the existing hook without new counters, schemas, fallback or protocol.
- [x] #3 Connected relevant tests, both client types, scoped static/security assessment and two independent source reviews pass; evidence and current runtime limitations are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required:no; existing ADR049/050 and accepted chat NN/G/H2 owner/intent plans govern. No new durable architecture. Scope: shared useCharacterGreetingStart inputKeys and its existing fixture, current caller validation, uniquely named three-stage plan and review/task records. Root sole runner, no imported-source edits during runners/compiler, no new browser/profile, no public H2/physical/extension acceptance.
Causal RED4/10pass at real same-value/ABA model setters across Scene/retention awaits; existing-field shared repair gives fixture14/14. Normal WebUI six-suite781/781,50.00s; normal WebUI tsconfig and extension application tsconfig.compile bothexit0. Matched source/test existing lint baseline0/final0/warnings0/introduced0. Both authorized source reviewers CLEAR. Bandit N/A TS/doc-only; unchanged Python scope retains previous qualified zero findings. Wrong UI-package broad-fixture environment and aggregate-test type-target exploratory diagnostics excluded; no source/test suppression. Review Docs/Reviews/CHAT_GREETING_MODEL_INTENT_13597_2026_10_12.md. TASK13596 separately qualifies fresh preview/user journeys; public H2/extension/AT/PG/all-ten remainopen.
All seven exact-owned initial repository hooks exit0 without bypass. Current normal development-profile Next production webpack/token/budget build including the one-field repair also exits0; actual current-tab refresh/postfix walkthrough stays TASK13596. Three stages complete; retire only this unit own plan privately. Final six-file metadata hooks and normal local incremental commit follow.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The shared greeting guard now fences existing selectedModelRevision, rejecting retired same-value/ABA model intent at preparation waits while preserving original draft and unchanged Scene/account/seed behavior. Four causal RED controls; final14 fixture and781 connected WebUI tests, both normal application type projects, zero lint findings, two bounded source reviews and exact-owned normal hooks qualify this repair. Current build checks pass; actual preview refresh remains TASK13596. No new authority/schema/fallback/dependency or public/native/physical acceptance. Normal local incremental commit follows.
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
