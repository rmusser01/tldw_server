---
id: TASK-12140
title: Harden audit Chat RAG resource authorization
status: Done
created_date: 2026-07-04 01:03
labels:
- audit-remediation
- chat
- rag
- authnz
- security
priority: Medium
documentation:
- Docs/superpowers/reviews/2026-06-27-repo-audit/final-report.md
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/chat-rag-llm.md
- Docs/superpowers/reviews/2026-06-27-repo-audit/remediation-backlog-draft.md
modified_files:
- Docs/superpowers/plans/2026-07-04-audit-chat-resource-auth-remediation-plan.md
- tldw_Server_API/Config_Files/privilege_catalog.yaml
- tldw_Server_API/app/api/v1/endpoints/rag_unified.py
- tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py
- tldw_Server_API/app/api/v1/endpoints/chat_documents.py
- tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py
- tldw_Server_API/tests/AuthNZ/unit/test_scoped_token_route_auth_chain.py
- tldw_Server_API/tests/AuthNZ/test_auth_dependency_contract.py
updated_date: 2026-07-04 01:17
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remediate comprehensive repository audit finding AUDIT-2026-06-27-CHAT-001. Scope is externally reachable Chat/RAG/LLM/embedding resource-spending routes that drift from shared token-scope endpoint and max-call enforcement. CHAT-002 logging hygiene was handled separately; this task focuses on endpoint IDs, token-scope guards, and focused runtime reproduction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Affected resource-spending routes have stable endpoint IDs through TokenScopeGuard or an equivalent shared enforcement dependency.
- [x] #2 Alternate RAG routes reject scoped JWT/API-key requests whose endpoint/method/path constraints do not allow the operation and count permitted calls where applicable.
- [x] #3 Character completion and chat document generation routes use consistent token-scope enforcement without breaking existing owner/auth dependencies.
- [x] #4 Embedding create/batch route behavior around endpoint allowlists and max-call counters is covered or explicitly documented if middleware already provides equivalent enforcement.
- [x] #5 Focused AuthNZ quota/resource-auth tests pass and Bandit reports no new production findings on touched backend code.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created before repository edits per Backlog.md workflow. Branch starts from origin/dev 638aad2077 in .worktrees/audit-chat-resource-auth-2026-07-04. Initial audit evidence points to rag_unified.py, character_chat_sessions.py, chat_documents.py, embeddings_v5_production_enhanced.py, and AuthNZ quota tests.
Implemented focused remediation for AUDIT-2026-06-27-CHAT-001. Added route-level TokenScopeGuard metadata to RAG alternate spending routes (/ablate, /batch, /batch/resume/{checkpoint_id}, /simple, /search/stream, /advanced) using logical endpoint_id rag.search/count_as call. Added TokenScopeGuard to character completion routes (/complete, /completions, /complete-v2) and chat document generation/bulk routes using endpoint_id chat.completions/count_as call. Added TokenScopeGuard to embeddings create/batch using the documented endpoint_id embeddings/count_as call and registered embeddings in privilege_catalog.yaml so startup privilege metadata validation recognizes the legacy virtual-key endpoint ID. Added route graph regression coverage in test_scoped_token_route_auth_chain.py plus an embeddings alias assertion in test_auth_dependency_contract.py. Broader non-spending/info/admin route-scope policy surfaced by explorer agents remains outside this focused spending-route slice and should be handled as a separate audit task if desired.

Verification completed: initial red run of `python -m pytest -q tldw_Server_API/tests/AuthNZ/unit/test_scoped_token_route_auth_chain.py -k resource_spending` failed for the targeted missing TokenScopeGuard routes; after implementation it passed with 13 selected tests. `python -m pytest -q tldw_Server_API/tests/AuthNZ/unit/test_scoped_token_route_auth_chain.py tldw_Server_API/tests/AuthNZ/test_auth_dependency_contract.py` passed 83 tests. `python -m pytest -q tldw_Server_API/tests/AuthNZ/unit/test_scoped_token_enforcement.py tldw_Server_API/tests/AuthNZ/unit/test_virtual_keys_limits_unit.py` passed 13 tests. `python -m pytest -q tldw_Server_API/tests/AuthNZ_SQLite/test_quota_enforcement_http_sqlite.py tldw_Server_API/tests/AuthNZ_SQLite/test_llm_budget_402_sqlite.py -k 'quota_enforced or embeddings'` passed 3 selected tests after cataloging embeddings. `python -m pytest -q tldw_Server_API/tests/Privileges/test_privilege_role_normalization.py tldw_Server_API/tests/Privileges/test_privilege_service_sqlite.py tldw_Server_API/tests/Privileges/test_privilege_endpoints.py` passed 27 tests. `git diff --check` passed. Bandit on touched endpoint Python files wrote /tmp/bandit_audit_chat_resource_auth_2026_07_04.json and reported 0 findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened the audit-identified Chat/RAG/embedding resource-spending routes by attaching shared TokenScopeGuard enforcement and stable logical endpoint IDs. Registered the documented embeddings endpoint ID in the privilege catalog so startup route-scope validation and virtual-key allowlists agree. Added route graph regression coverage and verified focused AuthNZ, SQLite quota, privilege-map, diff, and Bandit checks.
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
