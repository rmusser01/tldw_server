---
id: TASK-13885
title: Cost Anthropic fast mode (2x) and US-only inference (1.1x) price multipliers
status: To Do
labels:
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-13858 prices Anthropic calls at list prices. Anthropic bills two request options above them (platform.claude.com/docs/en/about-claude/pricing, fetched 2026-10-20): fast mode (speed: fast) on Claude Opus 5.5 ($8/$40 per MTok, 2x), Opus 5 and Opus 4.8 ($10/$50, 2x), and US-only inference (inference_geo: us) on Claude 4.6 and later models, a 1.1x multiplier on input, output, cache writes and cache reads. They stack with each other and with the prompt-caching multipliers. The Messages API native path forwards client fields unchanged (AnthropicMessagesRequest is extra=allow) and Anthropic reports both in the usage block (usage.speed, usage.inference_geo; messages.py _MESSAGES_NATIVE_USAGE_FIELDS), but compute_costs uses list prices, so those calls are undercharged and cost limits undercount fast-mode traffic by half. Found by the #3286 clean review (M5).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A recorded Anthropic call whose usage reports speed fast is costed at the fast-mode price of its model, with a test
- [ ] #2 A recorded Anthropic call whose usage reports inference_geo us is costed at 1.1x on every token category, stacking with fast mode and the cache rates, with a test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
