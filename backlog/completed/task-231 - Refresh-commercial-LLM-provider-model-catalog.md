---
id: TASK-231
title: Refresh commercial LLM provider model catalog
status: Done
assignee: []
created_date: '2026-05-10 15:49'
updated_date: '2026-05-10 16:17'
labels:
  - providers
  - models
  - catalog
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update the commercial-provider model availability catalog so the backend and WebUI expose current documented model IDs for configured commercial LLM providers. Treat tldw_Server_API/Config_Files/model_pricing.json as the primary model enumeration source for commercial providers, remove clearly unavailable/stale entries, keep placeholders only for entries intentionally hidden from selectors, and preserve existing config-driven/local-provider behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Commercial provider model entries are refreshed using current official provider documentation or official model-list endpoints where available.
- [x] #2 Stale or unavailable commercial model IDs are removed from selectable provider blocks unless a clear compatibility alias is added outside provider enumeration.
- [x] #3 Pricing catalog JSON remains valid and continues excluding placeholder entries from list_provider_models().
- [x] #4 Focused tests or existing pricing/provider catalog tests verify representative refreshed entries and placeholder exclusion behavior.
- [x] #5 Docs or inline catalog notes are updated only where needed to explain availability source or limitations.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation completed on 2026-05-10. Refreshed tldw_Server_API/Config_Files/model_pricing.json using official provider documentation and official model-list endpoint data where available: OpenAI, Anthropic, Google Gemini, Groq, Mistral, DeepSeek, xAI, Cohere, Qwen/Alibaba, Moonshot, Z.AI, MiniMax, and OpenRouter. Added placeholder entries to hide stale defaults and non-chat SKUs from list_provider_models(), including legacy Groq/Mistral/Anthropic aliases, Google Gemini 3 Pro Preview, OpenAI image/audio/realtime/search models, and Moonshot cache-hit pseudo-models.

Updated pricing_catalog.py so override entries can carry estimated=true and exact lookups preserve that estimated flag. This lets available model IDs remain selectable while approximate/source-dependent USD rates are reported as estimates.

Verification: node JSON parse for tldw_Server_API/Config_Files/model_pricing.json passed; python sanity script confirmed representative provider lists and hidden placeholders; source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Usage/test_pricing_catalog.py tldw_Server_API/tests/Usage/test_pricing_catalog_path.py tldw_Server_API/tests/Usage/test_pricing_catalog_overrides.py -q passed with 10 tests; source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Usage/pricing_catalog.py -f json -o /tmp/bandit_task231.json passed with 0 findings; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refreshed the commercial LLM provider model catalog with current documented model IDs and current OpenRouter model-list entries, hid stale/default/non-chat placeholder entries from provider enumeration, and added estimated-rate metadata support so approximate catalog rates are surfaced correctly. Focused Usage tests now cover refreshed provider entries, placeholder exclusion, and estimated metadata preservation.
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
