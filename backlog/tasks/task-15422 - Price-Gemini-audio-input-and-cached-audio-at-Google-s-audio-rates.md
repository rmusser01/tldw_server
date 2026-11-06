---
id: TASK-15422
title: Price Gemini audio input and cached audio at Google's audio rates
status: To Do
labels:
- quotas
- backend
dependencies:
- TASK-13952
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Google charges audio input and cached audio more than text on gemini-3.1-flash-lite ($0.50 and $0.05 per million tokens against $0.25 and $0.025), gemini-3-flash-preview ($1.00 / $0.10 against $0.50 / $0.05), gemini-2.5-flash ($1.00 / $0.10 against $0.30 / $0.03) and gemini-2.5-flash-lite ($0.30 / $0.03 against $0.10 / $0.01), on the standard, flex and priority tiers (aistudio.google.com/docs/pricing, fetched 2026-11-03). The usage record counts audio tokens inside promptTokenCount at the text rate, so a reply with audio input is under-billed by up to 3.3x on its input. The modality split is in usageMetadata.promptTokensDetails and cacheTokensDetails (ModalityTokenCount: modality TEXT/IMAGE/VIDEO/AUDIO/DOCUMENT, tokenCount). Fix: GoogleAdapter._usage_block keeps both lists (teach _sanitize_raw_usage_metadata that a list of counter mappings is not prompt text, or fold them to flat counts); normalize_llm_usage's Gemini branch sums the AUDIO entries into two new NormalizedLLMUsage fields (audio_input_tokens, cache_read_audio_tokens); the catalog accepts prompt_audio and cache_read_audio rates (in _parse_rates and _entry_rate_details, on a tier too); compute_costs prices the uncached audio share at prompt_audio and the cached audio share at cache_read_audio, falling back to the text rates; the rows above (and the four 2.5 -preview aliases) get the rates via the script in Docs/superpowers/plans/2026-11-03-gemini-pricing-gaps.md. Recorded as not priced in Docs/Operations/Usage_Quotas.md (Gemini bullet) until then.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Gemini reply whose promptTokensDetails and cacheTokensDetails carry AUDIO entries is costed at the model's audio input and cached-audio rates on the tier it reports, with a test per rate
- [ ] #2 A reply with no modality details or with TEXT only is costed as today, with a test
- [ ] #3 Docs/Operations/Usage_Quotas.md and the _catalog_note say audio is priced
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
