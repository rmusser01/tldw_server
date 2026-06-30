---
id: TASK-2275
title: Tighten Workspace Persona Defaults PRD broad contract
status: Done
labels:
- persona
- workspaces
- prd
- docs
priority: Medium
references:
- https://github.com/rmusser01/tldw_server/issues/1911
- https://github.com/rmusser01/tldw_server/issues/1902
- Docs/Product/Workspace_Persona_Defaults_PRD.md
- TASK-468
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Patch the existing Workspace Persona Defaults PRD after brainstorming so it defines the broader Workspace assistant defaults contract: assistant_defaults naming, V1 Persona-only validation, separated stored/effective response shapes, permission-aware degraded states, read_write confirmation, Chat Workspace as first implementation target, and later adoption criteria for other Workspace surfaces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PRD defines Workspace Assistant Defaults as the broad contract while keeping Chat Workspace as the first implementation target.
- [x] #2 PRD uses a structured assistant_defaults model with Persona-only V1 validation and no Persona snapshots.
- [x] #3 PRD distinguishes stored defaults from permission-filtered effective defaults and covers shared Workspace degraded states.
- [x] #4 PRD records read_write confirmation, existing-session immutability, and adoption criteria for later Workspace surfaces.
- [x] #5 Docs-only verification and Bandit applicability are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Patched `Docs/Product/Workspace_Persona_Defaults_PRD.md` to define Workspace Assistant Defaults as a broad Workspace-scoped contract with Chat Workspace as the first implementation target.
- Added structured `assistant_defaults` semantics with Persona-only V1 validation, no Persona snapshots, stored-versus-effective response shapes, permission-aware degraded states, `read_write` confirmation, and later-surface adoption gates.
- Self-review tightened ambiguous precedence language so the contract now falls back from explicit surface choice to Workspace default, then user/global assistant default, then server fallback.
- Verification: stale-field `rg` check returned no `persona_defaults` or old voice field hits; positive-contract `rg` check found expected `assistant_defaults`, `effective_assistant_default`, `read_write`, and surface references; `git diff --check` passed.
- Bandit: skipped because this change touches only Markdown PRD/task documentation and no Python code.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Revised the Workspace Persona Defaults PRD into a broader Workspace Assistant Defaults PRD for #1911. The contract now keeps V1 Persona-backed and reference-only, separates stored defaults from permission-filtered effective defaults, captures `read_write` confirmation and degraded shared-Workspace states, and defines adoption gates for later Workspace surfaces without making them V1 blockers.
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
