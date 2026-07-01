---
id: TASK-12076
title: Address main CodeQL alerts in PR against dev
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-01 02:01'
labels:
  - security
  - codeql
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/security/code-scanning'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate open CodeQL code-scanning alerts reported on refs/heads/main, determine which findings apply to origin/dev, fix the applicable issues on a new branch targeting dev, add focused regression tests, run touched-scope verification including Bandit, and open a PR against dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-30-main-codeql-alerts-dev-pr-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Alert classes addressed from refs/heads/main against origin/dev:
- Frontend clear-text secret storage: durable config/history storage no longer persists API keys or bearer/access/refresh tokens; test-only E2E seed values are scoped with CodeQL suppressions.
- Backend path injection: storage/download/import/export/temp paths now use existing safe_join/root validation or narrow CodeQL annotations after validated roots.
- Stack-trace exposure: audio voices, skills import preview, RAG streaming events, and OmniVoice sidecar responses now return public-safe diagnostics.
- SQL injection: Jobs event filters use a static allowlist of SQL fragments.
- Weak sensitive hashing: metrics labels and companion activity log refs use keyed HMAC-SHA256; legacy SHA1 lookup remains compatibility-only and annotated.
- ReDoS: selected email/checklist/metadata/media navigation/data-URI/email redaction regexes were replaced with linear parsers/scanners.
- Sensitive logging/storage: monitoring notification payloads and WebSearch debug output redact secret-like keys and values.
- Bind-all-interface: local LLM port probing maps wildcard runtime hosts to loopback before binding.

Verification recorded:
- Frontend Vitest: 3 files / 32 tests passed for useConfig, request history, and extension runtime bootstrap.
- Backend focused batch: 156 passed for Jobs, Metrics, Monitoring, Notifications, Notes/Tasks, Personalization, Media navigation, AuthNZ email, Storage, VN assets, and media ingestion safe paths.
- Backend broader batch: 543 passed before contract fixes; failing contracts were fixed and the affected files reran with 150 passed.
- Python py_compile passed for changed Python files.
- git diff --check passed.
- Bandit touched Python scope written to /tmp/bandit_main_codeql.json; raw scan has 0 high/medium findings, remaining low findings are baseline B101 asserts plus existing B311/B404/B603 in touched legacy files. Actionable profile excluding those baseline IDs written to /tmp/bandit_main_codeql_filtered.json with 0 results.

Follow-up requested: rebase PR #2564 on latest dev and address unresolved PR review threads/comments from Gemini, Cubic, Qodo, and CI checks.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR opened against dev: https://github.com/rmusser01/tldw_server/pull/2564

Final result: branch codex/main-codeql-alerts-dev addresses the open CodeQL alert classes observed on refs/heads/main against origin/dev, including frontend durable secret storage, backend path containment, stack-trace exposure, SQL identifier validation, sensitive hashing, ReDoS, sensitive logging/storage, and bind-all-interface probes. Verification is recorded in implementation notes; actionable Bandit profile is clean with 0 results.
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
