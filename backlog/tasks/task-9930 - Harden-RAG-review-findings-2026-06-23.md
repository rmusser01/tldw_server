---
id: TASK-9930
title: Harden RAG review findings 2026-06-23
status: Done
assignee: []
created_date: 2026-06-23 18:53
updated_date: 2026-06-24 04:33
labels:
- rag
- security
- review
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and address the validated RAG module review findings under tldw_Server_API/app/core/RAG. Scope is current module code, not git diffs. Findings verified and fixed: PGVector metadata order_by SQL injection, raw PII audit metadata, hard-coded anonymous security filtering / role hierarchy, mutable RAG cache document payloads, and unescaped PGVector wildcard collection matching.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PGVector metadata order_by validates/binds metadata keys and rejects unsafe keys.
- [x] #2 RAG security audit metadata never stores raw query text or raw PII match text.
- [x] #3 Security filtering uses the resolved request user where available and role sensitivity access is cumulative.
- [x] #4 RAG cache document payloads are cloned on store and retrieval so later mutations do not poison cached results.
- [x] #5 PGVector multi-collection wildcard matching escapes literal metacharacters and supports intended wildcards only.
- [x] #6 Focused regression tests and Bandit touched-scope verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Initial task file TASK-2420 was created before edits, but an unrelated task-2420 file is now present in the task directory. This task records the completed RAG review-fix work without modifying the unrelated task.

Plan: IMPLEMENTATION_PLAN_rag_review_fixes.md. Red verification: new unified helper pytest initially failed to import missing helpers before production changes. Focused production verification after fixes: python -m pytest tldw_Server_API/tests/VectorStores/unit/test_pgvector_adapter_helpers.py tldw_Server_API/tests/RAG_NEW/unit/test_security_filters_sanitizers.py tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_security_cache_helpers.py -q passed with 14 passed, 38 warnings. Bandit: python -m bandit -r tldw_Server_API/app/core/RAG/rag_service/vector_stores/pgvector_adapter.py tldw_Server_API/app/core/RAG/rag_service/security_filters.py tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py -f json -o /tmp/bandit_rag_review_fixes.json exited 0 with results=0 errors=0. git diff --check passed.

Moved to isolated worktree .worktrees/rag-review-fixes-9930 on branch codex/rag-review-fixes-9930 from local dev. Worktree verification: source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest tldw_Server_API/tests/VectorStores/unit/test_pgvector_adapter_helpers.py tldw_Server_API/tests/RAG_NEW/unit/test_security_filters_sanitizers.py tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_security_cache_helpers.py -q passed with 14 passed, 41 warnings. Worktree Bandit: python -m bandit -r touched RAG source files -f json -o /tmp/bandit_rag_review_fixes_worktree.json exited 0 with results=0 errors=0. Worktree git diff --check passed.

PR branch rebuilt directly on origin/dev after dropping local-only dev history. Final PR-range verification should compare origin/dev..HEAD and contain only the RAG fix/task files.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verified and fixed the validated RAG review findings, then rebased PR #2472 on latest dev and addressed the current Qodo review comments. PGVector metadata order_by validates and binds metadata keys through the centralized InvalidMetadataOrderKeyError, and multi_search now normalizes collection patterns into the sanitized collection-name space while preserving only the intended star wildcard. Security query audit metadata stores query hash/length, PII counts, masked query, and PII type/offset metadata without raw query or match text. Access roles are cumulative, and unified pipeline security filtering resolves the request user before falling back to feedback user or anonymous. Cache hit/store boundaries now use type-aware clones that isolate mutable document metadata without duplicating embedding buffers. Focused regression coverage and touched-scope verification were updated for the PR feedback fixes.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Reopened to rebase draft PR #2472 on latest dev and evaluate/address current PR checks and review comments.
PR #2472 feedback pass: rebased on latest origin/dev, validated Qodo review comments, moved InvalidMetadataOrderKeyError to core/exceptions.py, normalized PGVector collection glob patterns in sanitized collection-name space, added test type hints/docstrings and parameterized role access coverage, and replaced cache document deepcopy with type-aware cloning that preserves embedding references while isolating mutable metadata. Verification: focused pytest passed with 19 passed, 50 warnings; compileall passed for touched source files; Bandit touched-source JSON results=0 errors=0; git diff --check passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
