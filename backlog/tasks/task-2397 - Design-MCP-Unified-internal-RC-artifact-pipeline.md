---
id: TASK-2397
title: Design MCP Unified internal RC artifact pipeline
status: Done
labels:
- mcp
- packaging
- uat
- release
documentation:
- Docs/superpowers/specs/2026-06-22-mcp-unified-internal-rc-artifact-pipeline-design.md
modified_files:
- Docs/superpowers/specs/2026-06-22-mcp-unified-internal-rc-artifact-pipeline-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write a design spec for a private/internal MCP Unified release-candidate artifact pipeline that builds the nested standalone package, runs artifact and UAT gates from built wheels, produces release evidence, and avoids PyPI/TestPyPI publishing until later gates pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design spec documents the private/internal MCP Unified RC artifact contract.
- [x] #2 Spec separates nested `mcp_unified` package build/UAT from root `tldw-server` packaging.
- [x] #3 Spec defines installed-wheel UAT phases for artifact validation, clean install, extras, CLI, and smoke harness transports.
- [x] #4 Spec defines evidence report schema, failure categories, redaction rules, and publishing guardrails.
- [x] #5 Spec explicitly defers TestPyPI/PyPI publishing and `id-token` permissions.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Spec work started in isolated worktree `codex/mcp-unified-internal-rc-spec`. Current-state review found that the standalone package descriptor lives at `mcp_unified/pyproject.toml`, package metadata still marks the package `internal-experimental`/`not-published`, and the existing root `pypi-check` path builds the root `tldw-server` package rather than the nested standalone MCP package.

Self-review completed for the design spec. Verification run: unresolved-marker scan was clean. This is documentation-only work; Bandit is not applicable to this design-only change.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the MCP Unified internal RC artifact pipeline design spec. The design covers nested package artifact creation, installed-wheel UAT, extras checks, CLI and smoke harness coverage, cross-platform considerations, security/supply-chain checks, evidence reporting, and root-vs-standalone publishing guardrails. Implementation is intentionally deferred until the spec is reviewed and approved.
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
