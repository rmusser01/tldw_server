---
id: TASK-2306
title: Design MCP profile policy decision model
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-07 15:55
labels:
- mcp
- profiles
- policy
- security
- design
dependencies: []
references:
- Docs/superpowers/specs/2026-06-07-mcp-fs-patch-write-safe-edit-tools-design.md
- Docs/superpowers/specs/2026-06-07-mcp-tool-call-hooks-design.md
- Docs/superpowers/specs/2026-06-03-mcp-default-profile-tooling-presets-design.md
- Docs/superpowers/specs/2026-06-01-mcp-stdio-process-policy-design.md
- Docs/superpowers/specs/2026-06-07-mcp-profile-policy-decision-model-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a follow-up MCP/profile design spec that incorporates senior developer feedback on explicit deny/ask/allow outcomes, permission modes, path-rule semantics, shell alias parsing and wrapper handling, hooks-as-enforcement, sandbox semantics, MCP server/tool wildcard policy, and effective-permission explanations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and refined a follow-up MCP/profile design spec for the shared deny/ask/allow decision model, permission modes, catalog visibility, path pattern semantics, external MCP wildcards, session override tightening, hooks-as-enforcement, shell/run alias hardening, sandbox semantics, and effective-permission explanations.

Review pass incorporated on 2026-06-07:
- Clarified that visible askable tools must not be represented only by `canExecute=false`; direct ask tools should use `canExecute=true`, `requiresApproval=true`, and `decision.outcome="ask"` so existing adapters do not hide them.
- Reordered the evaluation pipeline so sandbox/process assertions can downgrade decisions before final approval routing.
- Added denied-call hook semantics: blocking hooks do not loosen already-denied calls; managed observer hooks may receive redacted deny events only for audit/metrics.
- Clarified that authored gitignore-style path policy compiles to flat matcher IR, not filesystem-expanded paths.
- Added unresolved/out-of-workspace symlink denial and redacted symlink explain requirements.
- Added external MCP server/tool canonicalization and collision rules.
- Replaced command-string examples with argv-token command rules and added command matcher grammar.
- Resolved the session override design conflict by keeping only tightening overrides and narrowing the open question to storage/audit location.
- Added canonical risk classes and updated implementation slices with the new obligations.

Validation recorded on 2026-06-07:
- `git diff --check` -> clean.
- `rg -n "TODO|TBD|FIXME|\\?\\?|PLACEHOLDER" Docs/superpowers/specs/2026-06-07-mcp-profile-policy-decision-model-design.md` -> no unresolved markers.
- Bandit skipped because this remains a docs-only design branch with no Python code changes.

Spec review note: the brainstorming workflow normally calls for a spec-review subagent, but the available subagent tool contract says to spawn agents only when the user explicitly asks for delegation. Local review passes were performed instead.

Known skips/blockers: no implementation in this branch; remaining open questions are documented in the spec for the next planning pass.
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
