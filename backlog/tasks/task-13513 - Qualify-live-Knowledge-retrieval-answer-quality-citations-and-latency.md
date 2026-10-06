---
id: TASK-13513
title: Qualify live Knowledge retrieval answer quality citations and latency
status: Done
labels:
- rag
- validation
- knowledge-followup
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Use a bounded curated corpus and available real inference/search providers to inspect retrieval traces separately from answer faithfulness, relevance, citation semantics and latency. Include factual, comparative and unanswerable cases; retain credential-free evidence and avoid claiming synthetic tests represent participant validation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Read end-to-end traces and map factual expectations to exact source passages
- [x] #2 Run bounded available real-provider cases and distinguish retrieval from generation failures
- [x] #3 Report citations, answer relevance and latency with the exact tested configuration and limitations
- [x] #4 Unmatched answer and quoted text cannot fabricate positive hard-citation coverage
- [x] #5 Ordinary multiword Knowledge Notes queries retrieve relevant canonical notes without weakening literal library search, owner boundaries or deletion filters.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Related TASK-13453. Provider preference and availability requested asynchronously. ADR required:no; existing RAG/evaluation APIs and provider configuration remain authoritative.
No provider preference received yet. Available local llama-server serves Gemma4-26B-A4B at127.0.0.1:9099; use bounded local generation without commercial charges. The established isolated runtime helper is reused; controlled embeddings remain separate from real answer generation and must be labeled.
Five real local-model answers match the curated factual/comparative/unanswerable expectations. Trace inspection found guardrails._find_offsets fabricates offset-zero spans on every non-empty unmatched document, falsely inflating hard and quoted citation coverage. Repair the shared offset mapper using exact source text only, with negative and partial-match regressions; keep structural citation availability distinct from semantic support. No new architecture/ADR required.
Live browser and API reproduction: an ordinary Notes-only query Vega next review date returns no documents; explicit note ID returns the canonical note. SQLite search_notes quotes the whole question as a phrase. Add a bounded term-match mode for RAG with existing literal library behavior, owner filters, soft-delete filters and result limits preserved. Also qualify ClaimsEngine hard-citation spans rather than crediting refuted claims or unknown/out-of-bounds spans.
Qualified five bounded real local Gemma4-26B-A4B factual/comparative/unanswerable cases, separately inspecting retrieval, answer support and exact citation spans. Repaired fabricated hard/quote offsets and unvalidated ClaimsEngine spans; ordinary Notes term retrieval now succeeds while library literal phrase behavior remains. Final deterministic API regressions: 91 passed. Production Bandit: zero findings. Evidence/configuration/latency limits and full frontend failure comparisons are retained in Docs/Reviews/KNOWLEDGE_FOLLOWUP_RESULTS_2026_10_06.md and its live-evidence.json artifact.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed bounded live-model qualification and the two demonstrated RAG defects. Five answers match curated expectations; correct paraphrases are not mislabeled as exact-span support. This does not qualify general model quality or live-vector retrieval.
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
