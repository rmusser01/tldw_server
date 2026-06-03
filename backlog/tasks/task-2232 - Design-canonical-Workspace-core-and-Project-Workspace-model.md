---
id: TASK-2232
title: Design canonical Workspace core and Project Workspace model
status: In Progress
assignee: []
created_date: '2026-06-03 07:34'
labels: []
dependencies: []
priority: high
documentation:
  - Docs/superpowers/specs/2026-06-03-canonical-workspace-core-project-model-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the design spec for a canonical Workspace core that supports research-only workspaces upgrading into Project Workspaces with optional primary roots, sandbox-managed roots, MCP/tool bindings, ACP/agent harness runtime context, files, git state, deployment/preview trajectory, and future team governance. Keep this as a design/spec slice; do not implement runtime code in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Spec defines Workspace as the canonical identity and treats Research Workspace and Project Workspace as profiles/capability states.
- [ ] #2 Spec supports research-only creation followed by upgrade through one primary project root.
- [ ] #3 Spec makes host-local and sandbox-managed roots first-class root backends from the first Project Workspace slice.
- [ ] #4 Spec treats Git as optional root capability state rather than a creation prerequisite.
- [ ] #5 Spec aligns MCP Shared Workspaces, MCP Workspace Sets, ACP sessions, Sandbox sessions/runs, and future harness adapters around a shared workspace runtime context.
- [ ] #6 Spec defines file metadata tracking by default and explicit file-content indexing policy.
- [ ] #7 Spec leaves room for team governance, comments, and private/team preview or deploy instances without making them first-slice requirements.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
