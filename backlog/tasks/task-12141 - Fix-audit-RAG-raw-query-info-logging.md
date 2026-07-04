---
id: TASK-12141
title: Fix audit RAG raw query info logging
status: Done
created_date: 2026-07-04 03:25
labels:
- audit
- remediation
- rag
- logging
- security
priority: medium
references:
- AUDIT-2026-06-27-CHAT-002
documentation:
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/chat-rag-llm.md
- Docs/superpowers/reviews/2026-06-27-repo-audit/remediation-backlog-draft.md
modified_files:
- tldw_Server_API/app/api/v1/endpoints/rag_unified.py
- tldw_Server_API/tests/RAG/test_rag_query_logging.py
updated_date: 2026-07-04 03:29
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remediate AUDIT-2026-06-27-CHAT-002 by replacing info-level RAG logs that include raw user queries with hash/length metadata and tests preventing raw query leakage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Unified RAG search info logs do not include raw query text.
- [x] #2 Advanced RAG search info logs do not include raw query text.
- [x] #3 RAG request logs include non-security query hash and query length metadata for correlation.
- [x] #4 Focused tests assert sensitive query content is absent from rendered info logs.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `_log_rag_search_request()` to log a non-security MD5 query hash and query length instead of raw query text.
- Replaced raw info-level query logging in unified and advanced RAG search paths; simple search now reuses the same helper.
- Added focused unit coverage asserting sensitive query fragments and private paths are absent from rendered info logs while hash/length metadata remains present.
- Verification: `python -m pytest tldw_Server_API/tests/RAG/test_rag_query_logging.py -q` passed with 2 tests; Bandit over `tldw_Server_API/app/api/v1/endpoints/rag_unified.py` reported 0 findings; `git diff --check` passed; targeted source scan found no old unified/advanced raw-query info log patterns.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed AUDIT-2026-06-27-CHAT-002 by routing RAG request info logs through a shared hash/length helper and removing raw query text from unified and advanced search logs. Focused tests now prevent sensitive query fragments from appearing in rendered info logs.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused RAG query logging tests pass.
- [x] #2 Bandit runs clean over touched production code.
- [x] #3 git diff --check passes.
- [x] #4 AUDIT-2026-06-27-CHAT-002 closure evidence is recorded in task notes.
<!-- DOD:END -->
