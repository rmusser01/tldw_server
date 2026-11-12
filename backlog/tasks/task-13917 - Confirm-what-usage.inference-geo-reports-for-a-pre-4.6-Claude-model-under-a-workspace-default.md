---
id: TASK-13917
title: Confirm what usage.inference_geo reports for a pre-4.6 Claude model under a
  workspace default
status: Done
labels:
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the TASK-13878/13879/13880/13885 PR (plan Docs/superpowers/plans/2026-10-23-pricing.md). compute_costs applies Anthropic's 1.1x US-only multiplier whenever usage.inference_geo is 'us', with no model gate. The pricing page scopes the multiplier to Claude 4.6 and later and says earlier models reject the parameter; the data-residency page says a workspace default_inference_geo applies when the parameter is omitted. What a pre-4.6 model reports in such a workspace is undocumented. If it reports 'us', that call is over-billed 1.1x; the alternative is a 4.6-and-later model gate in compute_costs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The behaviour is confirmed from Anthropic documentation or support and recorded, and compute_costs gates the multiplier by model if a pre-4.6 call can report 'us', with a test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
platform.claude.com/docs/en/manage-claude/data-residency.md, fetched 2026-10-23 (WebFetch). Model availability: "The `inference_geo` parameter is supported on Claude 4.6 and later models. Requests with `inference_geo` on Claude Opus 4.5, Claude Sonnet 4.5 (deprecated), Claude Haiku 4.5, or earlier models return a 400 error." Pricing: "Claude 4.6 and later models: US-only inference (`inference_geo: \"us\"`) is priced at 1.1x the standard rate across all token pricing categories (input tokens, output tokens, cache writes, and cache reads)." and "Older models: Don't support `inference_geo` (see Model availability). Standard pricing applies. Requests that include the parameter return a 400 error." Migration from legacy opt-outs: "your workspace has been automatically configured with `allowed_inference_geos: [\"us\"]` and `default_inference_geo: \"us\"`." and "Legacy models are unaffected by this migration." Response: "The response `usage` object includes an `inference_geo` field indicating where inference ran".
Resolved in PR #3306 (review fix round 1, M3). The page names the exempt models, so compute_costs now gates the 1.1x: a usage block reporting inference_geo 'us' for Claude Opus 4.5, Sonnet 4.5, Haiku 4.5 or an earlier model (claude-<opus|sonnet|haiku>-4[-0..5][-date], claude-0..3-*, claude-instant-*; dotted names mapped first) is costed at list price; Claude 4.6 and later, and any unrecognised (newer) family, pay 1.1x. Whether a legacy-migrated workspace's pre-4.6 call reports 'us' no longer matters: either way it is costed at standard, as Anthropic bills it.
Tests: test_us_only_inference_is_not_charged_on_models_before_claude_4_6 (10 IDs, RED 10 failed at 1.1x, GREEN) and test_us_only_inference_is_charged_on_claude_4_6_and_later (7 IDs, guard). Docs: Usage_Quotas.md says US-only applies to Claude 4.6 and later only. Bandit -ll on usage_tracker.py: no issues.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The data-residency page says inference_geo is supported on Claude 4.6 and later and older models bill standard. compute_costs now applies the 1.1x US-only multiplier only to Claude 4.6 and later, so a legacy-migrated workspace's pre-4.6 call is never over-billed.
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
