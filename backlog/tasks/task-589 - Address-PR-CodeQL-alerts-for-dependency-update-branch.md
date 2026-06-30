---
id: TASK-589
title: Address PR CodeQL alerts for dependency update branch
status: Done
labels:
- security
- codeql
modified_files:
- mcp_unified/gateway/fastapi.py
- tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py
- apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/literature-workproducts.ts
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the four CodeQL alerts requested for PR #2207: stack-trace exposure findings in mcp_unified/gateway/fastapi.py and incomplete sanitization in the literature workproducts UI. Record verification and PR notes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] MCP gateway HTTP error responses do not expose raw management exception text for profile, external registry, or external runtime errors.
- [x] Literature matrix markdown cell formatting escapes table separators and existing escape characters without partial sanitization.
- [x] Focused backend and frontend tests cover the CodeQL remediation paths.
- [x] Touched-scope Bandit and formatting checks are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented safe public payload builders in mcp_unified/gateway/fastapi.py for profile management, external registry, and external runtime errors. The responses preserve status mapping, reason_code, and resource identifiers but no longer return str(exc) to clients. Updated gateway FastAPI package tests to assert raw exception text is redacted.

Updated literature matrix markdown cell escaping to iterate over characters and escape both backslashes and pipes while normalizing CR/LF to spaces. Added a focused regression in StudioPane.literature-workproducts.test.tsx.

Verification:
- source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py tldw_Server_API/tests/Security/test_dependency_security_floor.py -q -> 153 passed, 7 warnings
- ./node_modules/.bin/vitest run src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx from apps/packages/ui -> 33 passed
- source .venv/bin/activate && python -m bandit -r mcp_unified/gateway/fastapi.py -f json -o /tmp/bandit_task589_gateway.json -> 0 findings
- source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -s B101,B404,B603,B105 -f json -o /tmp/bandit_task589_gateway_tests.json -> 0 findings after skipping test-only assert/subprocess/hardcoded-fixture checks
- git diff --check -> clean

Initial frontend verification attempts failed before tests because Vitest was run from the repo root and then package workspace links were incomplete. Ran bun install --frozen-lockfile in apps/ to repair local node_modules links; no lockfile changes were reported by git status.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the four requested CodeQL alerts for PR #2207 by redacting raw MCP gateway management exception text from HTTP responses and by making literature matrix markdown table-cell escaping handle both existing backslashes and pipe separators. Added focused backend and frontend regression coverage and recorded the local verification results.
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
