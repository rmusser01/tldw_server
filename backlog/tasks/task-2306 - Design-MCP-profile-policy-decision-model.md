---
id: TASK-2306
title: Design MCP profile policy decision model
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-07 15:55'
labels:
  - mcp
  - profiles
  - policy
  - security
  - design
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-06-07-mcp-fs-patch-write-safe-edit-tools-design.md
  - Docs/superpowers/specs/2026-06-07-mcp-tool-call-hooks-design.md
  - >-
    Docs/superpowers/specs/2026-06-03-mcp-default-profile-tooling-presets-design.md
  - Docs/superpowers/specs/2026-06-01-mcp-stdio-process-policy-design.md
  - >-
    Docs/superpowers/specs/2026-06-07-mcp-profile-policy-decision-model-design.md
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
Created a follow-up MCP/profile design spec for the shared deny/ask/allow decision model, permission modes, catalog visibility, path pattern semantics, external MCP wildcards, session override tightening, hooks-as-enforcement, shell/run alias hardening, sandbox semantics, and effective-permission explanations.

Validation recorded on 2026-06-07:
- `git diff --check` -> clean.
- `rg -n "TODO|TBD|FIXME|\\?\\?|PLACEHOLDER" Docs/superpowers/specs/2026-06-07-mcp-profile-policy-decision-model-design.md` -> no unresolved placeholders; matches are intentional policy/design terms only.
- Bandit skipped because this is a docs-only design branch with no Python code changes.

Spec review note: the brainstorming workflow normally calls for a spec-review subagent, but the available subagent tool contract says to spawn agents only when the user explicitly asks for delegation. A local review pass was performed instead and folded in clarifications for subject-specific rules, legacy `Bash(...)` pattern migration, session overrides that can only tighten policy, and ask-tool visibility defaults.

Known skips/blockers: no implementation in this branch; open questions are documented in the spec for the next planning pass.
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
