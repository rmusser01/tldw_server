---
id: TASK-15423
title: 'Gemini price changes of 2026-12-01 and 2027-01-01: re-fetch the page and bump
  the catalog'
status: To Do
labels:
- quotas
- backend
- high
dependencies:
- TASK-13952
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
model_pricing.json holds Google's rates in force on 2026-11-03 (aistudio.google.com/docs/pricing). Two dates. 2026-12-01: gemini-4-argon doubles (standard $2.00 / $10.00 / $0.10 becomes $4.00 / $20.00 / $0.20 per million tokens, the flex, priority and batch tables likewise), so it is under-charged by half until the catalog follows; and all requests to gemini-2.5-pro route to gemini-3.8-flash and all requests to gemini-2.5-flash route to gemini-3.5-flash-lite, billed at those models' rates (the response's modelVersion says what ran): gemini-2.5-pro calls become over-charged (3.8 Flash costs less), and gemini-2.5-flash calls cost the same on standard text ($0.30 / $2.50 / $0.03 on both; its audio is then billed at the text rate). 2027-01-01: gemini-3.8-flash and gemini-3.6-flash double ($0.75 / $3.75 / $0.075 becomes $1.50 / $7.50 / $0.15; tiers likewise), which also makes gemini-2.5-pro's catalogued input and cache rates ($1.25, $0.125) lower than what Google bills for it ($1.50, $0.15): under-charged on input and cache, over-charged on output. On each date: re-fetch the page, update GEMINI_PRICES, GEMINI_TIER_PRICES (and the long tables) in tests/Usage/test_pricing_service_tiers.py, re-run the Gemini script from Docs/superpowers/plans/2026-11-03-gemini-pricing-gaps.md (Task 3) and update the _catalog_note and Usage_Quotas.md. Decide whether gemini-2.5-pro and gemini-2.5-flash should be priced as the model they route to, or kept at their own rates. Not catalogued: TASK-13952 added the four text models; the page's specialised models (Live, TTS, Image, Transcribe, Embedding, Omni Flash, Robotics ER) are not, so a call to one costs the unknown-model rate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On 2026-12-01: gemini-4-argon's standard, flex and priority rates are the page's new ones, with the tests in test_pricing_service_tiers.py updated; the 2.5 Pro/Flash routing is priced one way or the other, with the decision in Usage_Quotas.md
- [ ] #2 On 2027-01-01: gemini-3.8-flash's and gemini-3.6-flash's rates are the page's new ones, with the tests updated, and gemini-2.5-pro's input and cache are no longer under-charged
- [ ] #3 Each bump cites the page and its fetch date in the _catalog_note
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
