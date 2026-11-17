---
id: TASK-13915
title: 'OpenAI pricing catalog gaps: new models, 30-minute and 12-hour cache writes,
  regional uplift'
status: To Do
labels:
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the TASK-13878/13879/13880/13885 PR (plan Docs/superpowers/plans/2026-10-23-pricing.md). developers.openai.com/api/docs/pricing (2026-10-23) lists what the catalog lacks: (a) the gpt-5.6-* and gpt-6* models and their prices; (b) 30-minute and 12-hour cache-write columns for those models; (c) a 10% uplift for regional processing (data residency) endpoints on models released on or after 2026-03-05, the OpenAI counterpart of Anthropic's 1.1x US-only multiplier. All 18 dated snapshots that had an alias with a cached rate were confirmed on their model pages, so no snapshot gap is known.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The models the pricing page lists that the catalog lacks are priced, including their cache-write rates, with a test
- [ ] #2 The regional-processing uplift is costed when a usage record shows it, or the gap is documented in Usage_Quotas.md
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
From TASK-13936 (2026-10-26): when the gpt-6* and gpt-5.6* models are catalogued, give them service_tiers too (their Batch, Flex and Fast rows include 30-minute and 12-hour write columns), and price the Ultrafast tier the page lists for gpt-6-astra and gpt-6.1-sol. Key their Fast rates by the service_tier the response reports: OpenAI's Fast-mode guide (developers.openai.com/api/docs/guides/priority-processing) says 'For GPT-5.6 and earlier models, the response returns priority whether the request specifies priority or fast', so gpt-5.6* stay keyed priority; the guide (2026-10-26) does not say what gpt-6* report, likely fast, so check it then and key those rates by the value it gives.
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
