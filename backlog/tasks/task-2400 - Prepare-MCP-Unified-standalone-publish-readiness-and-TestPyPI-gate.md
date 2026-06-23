---
id: TASK-2400
title: Prepare MCP Unified standalone publish readiness and TestPyPI gate
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-23 07:05'
labels:
  - mcp
  - packaging
  - release
  - uat
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden the standalone MCP Unified package for publish-readiness after the internal RC pipeline merge. Add metadata and release-gate checks needed before TestPyPI/public publishing, while keeping publication guarded and opt-in only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Standalone package metadata includes publish-ready project URLs, authors/maintainers, keywords, classifiers, license-file handling, and README rendering coverage without changing the internal experimental status.
- [x] #2 Release tooling provides explicit, guarded TestPyPI/publish dry-run support and cannot publish from normal PR CI by accident.
- [x] #3 Tests cover the package metadata contract, workflow/manual-dispatch guardrails, and publish command construction/redaction without requiring live credentials.
- [x] #4 Docs explain internal RC, TestPyPI dry run, and real publish prerequisites/credentials clearly.
- [x] #5 Focused package-boundary/RC validation, Ruff, Bandit, compileall, and diff checks are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented publish-ready metadata, package-local LICENSE handling, guarded publish-plan support in Helper_Scripts/mcp_unified_rc.py, Makefile dry-run target, manual-only publish workflow, package docs, and focused regression coverage. Validation: focused pytest suite reported 73 passed; Ruff reported all checks passed; compileall completed; Bandit wrote /tmp/bandit_mcp_unified_publish_readiness.json with exit 0; make mcp-unified-rc reported RC status ok; make mcp-unified-publish-dry-run reported RC status ok for build and publish-plan.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
MCP Unified standalone package is now publish-ready for dry-run/TestPyPI planning while still internal/not-published. Live upload remains opt-in through manual workflow confirmation, environment-scoped tokens, and MCP_UNIFIED_ALLOW_PUBLISH=1.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched Python scope or documented skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
