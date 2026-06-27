---
id: TASK-12053
title: Clean up MCP Unified GatewayConfigSnapshot schema warning
status: Done
assignee: []
created_date: '2026-06-27 17:39'
updated_date: '2026-06-27 17:44'
labels:
  - mcp
  - packaging
  - cleanup
dependencies: []
references:
  - >-
    backlog/tasks/task-2402 -
    Fix-MCP-Unified-TestPyPI-wheel-missing-policy-grants-package.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the known TASK-2402 follow-up: installed MCP Unified CLI commands emit a non-fatal Pydantic warning because GatewayConfigSnapshot defines a schema field that shadows a BaseModel attribute. Keep snapshot compatibility while making CLI output warning-clean.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Regression coverage fails when importing or running the gateway snapshot model emits the Pydantic shadow warning.
- [x] #2 GatewayConfigSnapshot no longer triggers the schema-field shadow warning while preserving serialized snapshot compatibility.
- [x] #3 Focused MCP Unified package/gateway tests, Bandit touched-scope scan, and diff checks pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented warning cleanup with TDD. Red regression: test_snapshot_model_import_is_warning_clean_and_preserves_schema_key failed under -W error::UserWarning because GatewayConfigSnapshot defined a Pydantic field named schema, shadowing BaseModel.schema. Fix: renamed the internal field to snapshot_schema and used validation/serialization aliases plus serialize_by_alias so public snapshots still read/write the schema JSON key. Verification so far: focused snapshot+CLI pytest passed 39 tests; package CLI package-info passed under -W error::UserWarning; py_compile passed for touched module/test; Ruff F/I/UP passed; git diff --check passed; Bandit on production snapshots.py reported zero findings; Bandit on the test file reports only existing low-severity pytest assert baseline after marking the intentional subprocess import/call.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Cleaned up the MCP Unified standalone GatewayConfigSnapshot Pydantic warning from TASK-2402. The internal model field is now snapshot_schema with validation/serialization aliases for the public schema JSON key, so snapshot import/export compatibility is preserved while imports and CLI commands are warning-clean. Verification: red regression failed under -W error::UserWarning before the fix; focused snapshot+CLI pytest passed 39 tests; package-info CLI passed under -W error::UserWarning; py_compile passed for touched files; Ruff F/I/UP passed; git diff --check passed; production Bandit on snapshots.py reported errors=[] and results=[]. Test-file Bandit reports only existing low-severity pytest assert baseline after intentional subprocess use was nosec-marked.
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
