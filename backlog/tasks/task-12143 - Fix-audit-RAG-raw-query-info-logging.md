---
id: TASK-12143
title: Fix audit RAG raw query info logging
status: Done
created_date: 2026-07-04 18:20
labels:
- audit
- remediation
- rag
- logging
- security
priority: medium
references:
- AUDIT-2026-06-27-CHAT-002
- https://github.com/rmusser01/tldw_server/pull/2614
documentation:
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/chat-rag-llm.md
- Docs/superpowers/reviews/2026-06-27-repo-audit/remediation-backlog-draft.md
modified_files:
- tldw_Server_API/app/api/v1/endpoints/rag_unified.py
- tldw_Server_API/tests/RAG/test_rag_query_logging.py
updated_date: 2026-07-05 00:25
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
- Tracking hygiene: moved this RAG audit record from duplicate `TASK-12141` to `TASK-12143` so it does not collide with another active audit PR.
- Review follow-up: kept Loguru `{}` placeholder logging because this file imports Loguru and that formatting style is supported; added defensive query coercion for non-string helper input with regression coverage.
- Current-dev refresh: rebased `codex/audit-rag-query-logging-2026-07-04` onto `origin/dev` `09d9ec901e1d4548f7924f1c6bcefa963fadd9bd`; merge-base matches `origin/dev`.
- Current-dev validation: `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/RAG/test_rag_query_logging.py -q` passed with 3 tests; `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/rag_unified.py -f json -o /tmp/bandit_rag_query_logging_origin_dev_09d9ec.json` reported 0 findings over 1900 LOC; `git diff --check HEAD~1..HEAD` passed.
2026-07-04 latest-dev refresh: rebased and validated PR #2614 on origin/dev 6b727b221e55646eba663a03571e38302f7fafc2. Tested head beb6e22200b3. Verification: python -m pytest tldw_Server_API/tests/RAG/test_rag_query_logging.py -q => 3 passed, 15 warnings; bandit -r tldw_Server_API/app/api/v1/endpoints/rag_unified.py => 0 findings over 1900 LOC; git diff --check HEAD~1..HEAD => clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened RAG query logging to avoid sensitive-query exposure while preserving request tracing. Final refresh validated against origin/dev 6b727b221e55646eba663a03571e38302f7fafc2 with focused tests passing, Bandit clean on touched production scope, and whitespace check clean.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused RAG query logging tests pass.
- [x] #2 Bandit runs clean over touched production code.
- [x] #3 git diff --check passes.
- [x] #4 AUDIT-2026-06-27-CHAT-002 closure evidence is recorded in task notes.
<!-- DOD:END -->
