---
id: TASK-550
title: Implement Explainer Chatbook export and import
status: Done
labels:
- backend
- chatbooks
- explainer
- implementation
priority: High
references:
- TASK-546
- TASK-547
- Docs/superpowers/specs/2026-06-09-explainer-workspace-design.md
- Docs/superpowers/plans/2026-06-09-explainer-workspace-implementation-plan.md
modified_files:
- tldw_Server_API/app/core/Explainer/chatbook_adapter.py
- tldw_Server_API/app/core/Chatbooks/chatbook_models.py
- tldw_Server_API/app/core/Chatbooks/chatbook_service.py
- tldw_Server_API/app/core/Chatbooks/chatbook_validators.py
- tldw_Server_API/app/api/v1/endpoints/chatbooks.py
- tldw_Server_API/app/api/v1/endpoints/explainer.py
- tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py
- tldw_Server_API/app/api/v1/schemas/explainer.py
- tldw_Server_API/app/core/Explainer/repository.py
- Docs/Schemas/chatbooks_manifest_v1.json
- tldw_Server_API/tests/Explainer/test_explainer_chatbook_export.py
- tldw_Server_API/tests/Chatbooks/test_explainer_session_content_type.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implementation notes:

- RED: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Explainer/test_explainer_chatbook_export.py tldw_Server_API/tests/Chatbooks/test_explainer_session_content_type.py -v` failed during collection with `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.Explainer.chatbook_adapter'`.
- GREEN: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Explainer/test_explainer_chatbook_export.py tldw_Server_API/tests/Chatbooks/test_explainer_session_content_type.py -v` passed: 10 passed.
- Regression: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Explainer/test_explainer_jobs.py tldw_Server_API/tests/Explainer/test_explainer_endpoints.py -v` passed: 31 passed.
- Chatbooks selector: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chatbooks -k explainer -v` passed: 4 passed, 174 deselected.
- Manifest contract: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_contract.py -v` passed: 1 passed.
- Bandit: `source .venv/bin/activate && python -m bandit -r <touched backend app files> -f json -o /tmp/bandit_task550.json` exited 0 with 0 results.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement Task 3 from Docs/superpowers/plans/2026-06-09-explainer-workspace-implementation-plan.md: first-class Chatbook explainer_session export/import plus generated_document subtype import fallback. Follow TDD and keep Explainer serialization in core/Explainer/chatbook_adapter.py.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Local review fixed two issues after subagent handoff: `asyncMode=false` on the Explainer export endpoint now returns a job-backed sync download URL, and imported root nodes now preserve exported `kind`/`intent` through `ExplainerRepository.update_node`.
- Fresh verification after review fixes: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Explainer/test_explainer_chatbook_export.py tldw_Server_API/tests/Chatbooks/test_explainer_session_content_type.py -v` passed 11/11.
- Regression verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Explainer/test_explainer_jobs.py tldw_Server_API/tests/Explainer/test_explainer_endpoints.py -v` passed 31/31; `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chatbooks -k explainer -v` passed 4/4 selected; `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_contract.py -v` passed 1/1.
- Hygiene/security: `git diff --cached --check` passed; `git diff --check -- <Task 3 touched local files>` passed. Full `git diff --check` still reports pre-existing unrelated trailing whitespace in `Docs/Design/Agents.md`. `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Explainer tldw_Server_API/app/core/Chatbooks/chatbook_models.py tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py tldw_Server_API/app/core/Chatbooks/chatbook_service.py tldw_Server_API/app/api/v1/endpoints/explainer.py tldw_Server_API/app/api/v1/schemas/explainer.py tldw_Server_API/app/api/v1/endpoints/chatbooks.py tldw_Server_API/app/core/Chatbooks/chatbook_validators.py -f json -o /tmp/bandit_task550_local.json` exited 0 with zero results.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented and locally reviewed first-class Explainer Chatbook export/import support. Added `explainer_session` content type, manifest statistics/schema support, a single-item Explainer session serializer with structured JSON plus rendered markdown, Chatbook collection/import restoration, generated_document subtype fallback restoration, and the ownership-checked `POST /api/v1/explainer/sessions/{session_id}/export-chatbook` endpoint. Export scrubs sensitive generation metadata keys while preserving provider/model/prompt-version metadata; import restores sessions into the importing user's Explainer DB with new session/node/citation IDs, unresolved source-reference metadata, restored node kind/intent, and sync or async Chatbook export behavior according to request configuration.
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
