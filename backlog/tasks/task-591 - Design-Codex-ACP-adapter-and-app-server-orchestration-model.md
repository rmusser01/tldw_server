---
id: TASK-591
title: Design Codex ACP adapter and app-server orchestration model
status: In Progress
labels:
- ACP
- Codex
- agents
- design
priority: High
documentation:
- Docs/superpowers/specs/2026-06-01-acp-codex-orchestration-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write and review the design spec for Codex support in the ACP/agent orchestration model. Scope includes external ACP adapter support using zed-industries/codex-acp as the first Codex path, future Codex app-server backend support, reusable runner-adapter fallback concepts, registry/status semantics, security and worktree/session model implications, and a staged implementation roadmap. This is a design/spec task only; implementation planning follows after spec approval.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A design spec is written under Docs/superpowers/specs and cites verified local context plus prior-art sources.
- [ ] #2 The spec clearly separates external ACP adapters, runner adapters, and Codex app-server support.
- [ ] #3 The spec includes risks, non-goals, acceptance criteria for the first implementation slice, and a staged roadmap.
- [ ] #4 The spec review loop is run and issues are addressed before requesting user approval for implementation planning.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
['Spec re-review completed after adapter pin/install clarifications. Status: Approved. Blocking issues: none. Advisory planning notes: separate profile/runtime work from live certification evidence so metadata/runner changes can land before certification; include tests for legacy adapter_acp input and native ACP profiles without acp_command to preserve fallback behavior. Awaiting user approval before implementation planning.']
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
