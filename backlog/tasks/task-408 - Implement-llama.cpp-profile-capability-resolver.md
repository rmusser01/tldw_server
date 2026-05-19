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

PR #1777 review RED: python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py::test_resolve_asset_id_fails_closed_without_allowed_paths tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_capabilities.py::test_manual_profile_path_fails_closed_without_allowed_paths -v -> failed because current code did not raise ServerError for empty allowlists.

Final GREEN: python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_capabilities.py tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py -v -> 38 passed, 5 warnings.

Warnings investigated with python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_capabilities.py tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py -o addopts='' -q -rw -> 36 passed, 5 warnings before review-fix tests were added. Warning details accepted as existing baseline outside this llama.cpp slice: PytestConfigWarning at .venv/lib/python3.11/site-packages/_pytest/config/__init__.py:1474, Unknown config option: import_mode; PytestConfigWarning at .venv/lib/python3.11/site-packages/_pytest/config/__init__.py:1474, Unknown config option: plugins; DeprecationWarning at .venv/lib/python3.11/site-packages/passlib/utils/__init__.py:854, crypt is deprecated and slated for removal in Python 3.13; UserWarning at .venv/lib/python3.11/site-packages/pydantic/_internal/_fields.py:198, Field name schema in ResponseFormatJsonSchemaSpec shadows an attribute in parent BaseModel; UserWarning at .venv/lib/python3.11/site-packages/pydantic/_internal/_fields.py:198, Field name schema in JSONValidateConfig shadows an attribute in parent BaseAdapterConfig.

Validation: python -m bandit -r tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py tldw_Server_API/app/core/Local_LLM/llamacpp_profile_capabilities.py -f json -> 0 findings. git diff --check -> exit 0.

Known deferred work: supervisor startup wiring, /api/v1/llm/models/metadata integration, and WebUI display remain later tasks from the approved plan.

Review disposition: fixed the Qodo allowlist finding by making asset/profile path validation fail closed when no allowed bases are configured. Fixed Qodo line-length comments in the touched regions. Fixed CodeRabbit Backlog command portability and warning-detail comments. Gemini redundant-validation/private-helper comments were addressed where valid by returning validated resolved asset paths directly and moving manual path validation into a public inventory helper; the remaining asset-kind recheck in resolve_asset_id is intentionally retained because callers can pass pre-scanned assets and the resolver should validate the actual path as well as the supplied asset metadata.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Task 1 backend resolver slice for llama.cpp managed profiles and addressed PR #1777 review feedback. Asset and manual profile path validation now fails closed when no allowed bases are configured, manual profile-path validation reuses a public inventory helper, resolved asset IDs return canonical paths, and the Backlog task records portable commands plus warning details. Supervisor wiring, llm metadata endpoint integration, and WebUI display remain intentionally deferred to later plan tasks.
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
