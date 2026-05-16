---
id: TASK-408
title: Implement llama.cpp profile capability resolver
status: Done
labels:
- llamacpp
- backend
- tests
priority: high
documentation:
- Docs/superpowers/plans/2026-05-16-llamacpp-model-family-mmproj-profile-wiring-plan.md
references:
- https://github.com/rmusser01/tldw_server/pull/1777
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 1 from the llama.cpp model-family/mmproj profile wiring plan: add a public local asset resolver and a profile capability/launch resolver that validates base GGUF and optional mmproj assets, derives mode capabilities/modalities, and normalizes launch args without changing supervisor or WebUI wiring yet.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 resolve_asset_id supports expected-kind filtering, missing IDs, stale paths, and optional pre-scanned assets without changing resolve_model_id compatibility.
- [x] #2 resolve_profile_launch resolves base model paths, requires mmproj for vision mode, injects resolved mmproj args, rejects conflicting projector args, and derives mode capability/modalities.
- [x] #3 Profile capability metadata returns bounded public metadata without raw path exposure.
- [x] #4 Focused llama.cpp inventory/profile capability tests pass.
- [x] #5 Bandit and git diff checks are recorded for touched code.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
RED: asset resolver tests failed with AttributeError because resolve_asset_id did not exist. RED: profile capability tests failed with ModuleNotFoundError because llamacpp_profile_capabilities did not exist. Additional RED: folder expected-kind resolver test failed until folder assets were accepted explicitly.

GREEN: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_capabilities.py tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py -v -> 36 passed, 5 warnings.

Validation: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py tldw_Server_API/app/core/Local_LLM/llamacpp_profile_capabilities.py -f json -o /tmp/bandit_llamacpp_profile_capability_resolver.json -> 0 findings. git diff --check -> exit 0.

Known deferred work: supervisor startup wiring, /api/v1/llm/models/metadata integration, and WebUI display remain later tasks from the approved plan.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Task 1 backend resolver slice for llama.cpp managed profiles: public asset ID resolution in inventory service plus profile launch/capability metadata helpers for GGUF, mmproj, vision, embedding, rerank, and server_generic modes. Supervisor wiring, llm metadata endpoint integration, and WebUI display remain intentionally deferred to later plan tasks.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation/task notes updated when relevant
- [x] #4 Bandit run for touched code when applicable or documented skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
