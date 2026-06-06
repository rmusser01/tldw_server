---
id: TASK-2263
title: Design MCP tool-use evaluation reporting surface
status: Done
labels:
- mcp
- design
- observability
- evals
- gateway
references:
- backlog/tasks/task-2256 - Apply-MCP-tool-observability-and-evaluation-contract-across-all-tools.md
- Docs/superpowers/plans/2026-06-04-mcp-tool-observability-contract-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-06-06-mcp-tool-use-eval-reporting-design.md
- backlog/tasks/task-2263 - Design-MCP-tool-use-evaluation-reporting-surface.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the design spec for metadata-only MCP tool-use event capture, storage, export, and aggregate reporting across standalone gateway and in-process MCPProtocol paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Metadata-only default capture is specified.
- [x] Protocol and gateway capture paths are covered.
- [x] Aggregate report dimensions and export/cleanup surfaces are specified.
- [x] Privacy, storage, dependency-compatibility, and double-counting guardrails are documented.
- [x] Follow-up boundaries for payload capture, `fs.patch`, and governed Git aliases are explicit.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/specs/2026-06-06-mcp-tool-use-eval-reporting-design.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created `Docs/superpowers/specs/2026-06-06-mcp-tool-use-eval-reporting-design.md`.

Integrated the ten design review findings: backward-compatible recorder dependency, double-counting guard, partial denial events before metadata resolution, requested/effective tool names for `profile.tools.call`, compliant storage through existing package patterns, privacy-safe identifiers, idempotency replay semantics, precise recorder failure/backpressure policy, UTC epoch ordering, and cautious tool-call outcome terminology.

Second local review added three hardening improvements: no eager optional storage imports, bounded recorder write timeout, and capture scope for method-level `tools/call` rate-limit failures.

Verification: `git diff --check` passed. ASCII scan over the spec and task file found no non-ASCII characters. Bandit skipped because this task only adds design documentation and Backlog metadata.

Spec review note: subagent review was not run because the available multi-agent tool requires explicit user authorization to spawn agents. A local spec review was performed instead and the findings were integrated.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Wrote the MCP tool-use evaluation reporting design spec for metadata-only event capture and aggregate reports across gateway and in-process protocol paths. The spec incorporates the ten design review findings plus a second local review pass covering optional dependency compatibility, double-counting guards, partial denial events, requested/effective tool names, compliant storage, privacy-safe identifiers, idempotency replay semantics, bounded recorder writes, UTC ordering, cautious reporting terminology, lazy storage imports, recorder write timeout, and method-level rate-limit capture.
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
