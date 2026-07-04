---
id: TASK-12135
title: Remediate RAG raw-query logging audit finding
status: Done
created_date: 2026-07-03 23:57
labels:
- audit
- remediation
- rag
- logging
- security
- wave-2
priority: medium
references:
- AUDIT-2026-06-27-CHAT-002
documentation:
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/chat-rag-llm.md
modified_files:
- Docs/superpowers/plans/2026-07-03-rag-query-log-sanitization-remediation.md
- backlog/tasks/task-12135 - Remediate-RAG-raw-query-logging-audit-finding.md
- tldw_Server_API/app/api/v1/endpoints/rag_unified.py
- tldw_Server_API/tests/RAG/test_rag_query_logging_sanitization.py
updated_date: 2026-07-04 00:04
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track remediation for AUDIT-2026-06-27-CHAT-002: RAG search endpoints should not log raw user queries at info level because queries can contain private content.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan is written before production code changes.
- [x] #2 RAG/search endpoint logs avoid raw query text at info level and preserve useful non-sensitive diagnostics.
- [x] #3 Focused tests or log-capture checks prove raw query content is not emitted at info level.
- [x] #4 Bandit touched-scope verification is recorded.
- [x] #5 Residual logging/privacy tradeoffs are documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-07-03 planning: added implementation plan at Docs/superpowers/plans/2026-07-03-rag-query-log-sanitization-remediation.md. Scope is limited to AUDIT-2026-06-27-CHAT-002: replace raw query text in unified and advanced RAG info logs with query_hash and query length, plus focused log-capture tests.
2026-07-03 implementation: replaced unified and advanced RAG info logs that included raw query text with query_hash and len metadata; refactored simple search to use the same helper while preserving existing non-sensitive output. Added focused Loguru INFO-capture tests for unified and advanced endpoints. Red check failed on raw query text as expected; green check passed after implementation.

Verification recorded: PYTHONDONTWRITEBYTECODE=1 /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider tldw_Server_API/tests/RAG/test_rag_query_logging_sanitization.py -q --tb=short --disable-warnings -> 2 passed, 13 warnings. Bandit touched-scope scan: PYTHONDONTWRITEBYTECODE=1 /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit tldw_Server_API/app/api/v1/endpoints/rag_unified.py -f json -o /tmp/bandit_rag_query_logging_12135.json -> 0 results, 0 errors. git diff --check -> clean.

Residual tradeoff: this remediation targets the audited info-level request logs. The original query still intentionally flows to RAG execution and topic monitoring. Broader exception/diagnostic logging policy is unchanged and should be evaluated separately if production diagnostic backtraces are configured to expose local state.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Remediated AUDIT-2026-06-27-CHAT-002 by removing raw query text from unified and advanced RAG info logs and logging only a short deterministic query hash plus query length. Added focused regression coverage proving sentinel query content is absent from INFO logs while preserving useful non-sensitive diagnostics. Verification passed with the focused pytest suite, touched-scope Bandit scan, and diff whitespace check.
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
