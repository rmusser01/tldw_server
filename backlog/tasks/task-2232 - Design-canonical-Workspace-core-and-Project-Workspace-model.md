---
id: TASK-2232
title: Design canonical Workspace core and Project Workspace model
status: Done
assignee: []
created_date: 2026-06-03 07:34
labels: []
dependencies: []
priority: high
documentation:
- Docs/superpowers/specs/2026-06-03-canonical-workspace-core-project-model-design.md
modified_files:
- Docs/superpowers/specs/2026-06-03-canonical-workspace-core-project-model-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the design spec for a canonical Workspace core that supports research-only workspaces upgrading into Project Workspaces with optional primary roots, sandbox-managed roots, MCP/tool bindings, ACP/agent harness runtime context, files, git state, deployment/preview trajectory, and future team governance. Keep this as a design/spec slice; do not implement runtime code in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec defines Workspace as the canonical identity and treats Research Workspace and Project Workspace as profiles/capability states.
- [x] #2 Spec supports research-only creation followed by upgrade through one primary project root.
- [x] #3 Spec makes host-local and sandbox-managed roots first-class root backends from the first Project Workspace slice.
- [x] #4 Spec treats Git as optional root capability state rather than a creation prerequisite.
- [x] #5 Spec aligns MCP Shared Workspaces, MCP Workspace Sets, ACP sessions, Sandbox sessions/runs, and future harness adapters around a shared workspace runtime context.
- [x] #6 Spec defines file metadata tracking by default and explicit file-content indexing policy.
- [x] #7 Spec leaves room for team governance, comments, and private/team preview or deploy instances without making them first-slice requirements.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated the canonical Workspace core design to address the review findings before implementation planning. The spec now explicitly persists workspace_profile while computing operational capability states; separates profile intent from project_root_state; assigns primary root ownership to Workspace Core with new Workspace DB/table persistence as the preferred target; defines Sandbox-volume lifecycle through a Workspace-bound Sandbox API wrapper; routes root-wide file scans and indexing progress through Jobs; clarifies MCP Shared Workspaces as trusted-root compatibility terminology; treats workspace_group_id as runtime/policy lineage only; adds fail-closed Workspace context resolver semantics; narrows preview defaults to owner-private/workspace-member access; and records external product references with URLs and access date. Verification: git diff --check and targeted rg consistency checks. Bandit skipped because this is a docs/backlog-only design task.
<!-- SECTION:FINAL_SUMMARY:END -->
