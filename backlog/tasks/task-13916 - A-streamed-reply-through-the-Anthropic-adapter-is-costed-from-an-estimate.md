---
id: TASK-13916
title: A streamed reply through the Anthropic adapter is costed from an estimate
status: To Do
labels:
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the TASK-13878/13879/13880/13885 PR (plan Docs/superpowers/plans/2026-10-23-pricing.md). Confirmed in that plan's review (M3): AnthropicAdapter.stream (anthropic_adapter.py, the stream method) never handles message_start, yields message_delta only as a finish_reason chunk with the usage nested under provider_response, and _StreamUsage.feed (usage_accounting.py) reads only the top-level usage. So a streamed /chat/completions reply (and any other adapter caller) is costed from a prompt estimate: cache reads and writes, 1-hour writes, fast mode and US-only inference are all missed. Usage_Quotas.md documents it. The Messages API path reads the real usage already.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A streamed reply through the Anthropic adapter is recorded with the provider's real usage (input, cache, output tokens, speed, inference_geo), with a test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
From TASK-13936 (2026-10-26): fixing the adapter's stream usage does not change /chat/completions, whose streamed replies chat_service records from an estimate; that is TASK-13951.
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
