---
id: TASK-2331
title: Harden MCP filesystem path policy and command-runtime defaults
status: In Progress
labels:
- mcp
- filesystem
- security
- policy
priority: High
references:
- 'User-approved approach: policy hardening first'
- runtime routing second; diff parser expansion only where current gaps block real
  use.
documentation:
- Docs/superpowers/specs/2026-06-09-mcp-filesystem-policy-hardening-design.md
modified_files:
- Docs/superpowers/specs/2026-06-09-mcp-filesystem-policy-hardening-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement the next MCP filesystem slice: verify path/action constraints for fs.read, fs.write, fs.edit, and fs.patch first, then update command-runtime defaults to prefer structured primitives over legacy fs.read_text/fs.write_text where safe.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design approved for A then B: first add path/action enforcement regression coverage for filesystem primitives, then update the virtual CLI runtime to prefer structured primitives where semantics are unambiguous. Local spec review added the adapter seam: TldwPathScopeEnforcer must accept and forward module-derived path_scope_candidates so fs.patch works through the real runtime path. Second review tightened command-runtime details: cat should prefer fs.read with a legacy fs.read_text fallback when safe; structured write-create may be added; write-replace must carry expected_sha256/read_receipt or be deferred; RunCommandModule write classification must include structured fs.write-backed commands; path-scope tests should extend the existing MCP protocol path-scope test file.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
