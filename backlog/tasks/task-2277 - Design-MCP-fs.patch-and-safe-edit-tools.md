---
id: TASK-2277
title: Design MCP fs.patch and safe edit tools
status: In Progress
labels:
- mcp
- filesystem
- security
- design
references:
- Docs/superpowers/specs/2026-06-06-mcp-tool-use-eval-reporting-design.md
- Docs/superpowers/specs/2026-06-04-mcp-filesystem-helper-tools-design.md
- Docs/superpowers/specs/2026-03-28-mcp-virtual-cli-run-command-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the design/spec for the next MCP slice after tool-use reporting: native fs.patch and safer workspace-bounded write-edit tools. The design must preserve profile/policy enforcement, path-scope controls, conflict detection, cross-platform behavior, audit/eval metadata, and avoid raw shell or unsafe arbitrary filesystem mutation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec defines fs.patch as the preferred unified-diff edit primitive with parser scope, conflict detection, limits, atomicity caveats, and no shell/subprocess delegation.
- [x] #2 Spec defines fs.write as the paired whole-file create/replace primitive with explicit modes, required replacement hashes, UTF-8 bounds, and write-action path grants.
- [x] #3 Spec defines action-aware path_grants for read/edit/write and includes Profile A/B/C permission examples.
- [x] #4 Spec covers protocol/module derived path-candidate preflight for diff-embedded paths and fail-closed behavior when candidates cannot be enforced.
- [x] #5 Spec covers observability/evaluation redaction so raw diffs and file contents are not persisted.
- [x] #6 Spec includes rollout and testing strategy for parser, filesystem module, path-scope, protocol, and profile coverage.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design review pass incorporated fs.write as the paired primitive, explicit action semantics, segment-aware path grant matching, cross-platform diff path rejection, fail-closed derived path enforcement for fs.patch, preset migration guidance, and filesystem-wide eval redaction.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the MCP fs.patch/fs.write safe edit design spec at Docs/superpowers/specs/2026-06-07-mcp-fs-patch-write-safe-edit-tools-design.md. The design covers unified-diff parsing, fs.write create/replace semantics, action-aware path grants, derived path preflight, cross-platform path constraints, observability redaction, rollout slices, and focused tests. Validation: git diff --cached --check passed; placeholder scan found no outstanding marker text. Bandit was skipped because this task changed only documentation and Backlog task metadata.
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
