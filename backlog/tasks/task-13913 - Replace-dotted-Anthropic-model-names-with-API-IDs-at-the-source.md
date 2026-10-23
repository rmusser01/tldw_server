---
id: TASK-13913
title: Replace dotted Anthropic model names with API IDs at the source
status: To Do
labels:
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the TASK-13878/13879/13880/13885 PR (plan Docs/superpowers/plans/2026-10-23-pricing.md). That PR keeps saved dotted names working: canonical_anthropic_model_id (core/LLM_Calls/anthropic_model_ids.py) maps claude-sonnet-4.5 to claude-sonnet-4-5 in the price lookup, the availability gate, provider inference, the metadata and router-rank lookups, the model allowlists and every request sent to Anthropic. So this is cleanup, not a fix. Dotted names still come from: config.py:3861 (default anthropic_model = claude-sonnet-4.5), Evaluations/cli/api_utils.py:158, Ingestion_Media_Processing/Audio/Audio_Streaming_Insights.py:336, RAG/rag_service/quick_wins.py:309-311 (CostTracker.PRICING, its own exact lookup), the claude-3-5-sonnet-latest catalog entry (alias of a model retired 2025-10-28; still a picker entry), and docs and tests: Docs/API-related/Providers_API_Documentation.md (mirrored under Docs/Published, so run bash Helper_Scripts/refresh_docs_published.sh), Docs/Code_Documentation/Chat_Developer_Guide.md, tests Chat_NEW/conftest.py, LLM_Adapters/unit/test_anthropic_config_and_payload.py, LLM_Calls/test_llm_providers.py, Media_Ingestion_Modification/test_media_contextual_integration.py, schemas/test_chat_validators.py, test_contextual_properties.py. claude-sonnet-4-5-20250929 retires 2026-11-30 and Opus 4.1 and Sonnet 4 are retired, so the defaults likely need a newer model: a product choice for this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No defaults table, config default or catalog entry in tldw_Server_API/app names a dotted Anthropic model, with a test that guards it
- [ ] #2 The chosen replacement default model is recorded in the notes with the reason
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
From the PR #3306 clean review (M1): a dotted [API] anthropic_model in config.txt (and the config.py:3861 fallback default claude-sonnet-4.5) now gives two picker entries for one model, claude-sonnet-4.5 from config and claude-sonnet-4-5 from the catalog, because llm_providers.py:2010-2012 dedupes config models against catalog models by exact string. Before #3306 the catalog key was dotted too, so they merged. The shipped config.txt uses a dashed ID, so only custom or older configs see it. Fix here: dedupe Anthropic names by canonical_anthropic_model_id, or replace the dotted defaults at the source.
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
