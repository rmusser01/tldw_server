---
id: TASK-13878
title: Cost Anthropic 1-hour prompt-cache writes at 2x, not the 5-minute 1.25x rate
status: To Do
labels:
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-13858 gave the Anthropic models one cache_write rate, the 5-minute write price (1.25x input). Anthropic bills a 1-hour cache write at 2x input (platform.claude.com/docs/en/about-claude/pricing) and reports the split in usage.cache_creation.ephemeral_5m_input_tokens and ephemeral_1h_input_tokens. The server asks for 1-hour writes when a cache intent sets ttl 1h or ttl_seconds >= 3600 (LLM_Calls/cache_intents.py), and Messages API clients can ask directly, so those writes are costed 0.75x input too low and cost quotas undercount them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The usage normalizer reads the 1-hour share of cache writes, and compute_costs prices it at a cache_write_1h rate, with a test
- [ ] #2 model_pricing.json gives the Anthropic models a cache_write_1h rate (2x input), and the Anthropic adapter keeps the cache_creation breakdown
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
