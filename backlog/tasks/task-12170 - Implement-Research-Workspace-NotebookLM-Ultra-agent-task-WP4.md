---
id: TASK-12170
title: Implement Research Workspace NotebookLM Ultra agent-task WP4
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-05 19:53'
labels:
  - research-workspace
  - notebooklm
  - wp4
  - agent-tasks
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the WP4 slice from the approved Research Workspace NotebookLM Pro/Ultra design: make Ultra-style agentic actions discoverable as governed tldw Workspace Agent Tasks from Research Workspace chat and Studio, with visible observable activity and save-back affordances where existing ACP/sandbox/storage contracts support them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A WP4 implementation plan is written and linked from this task before product code changes.
- [x] #2 Research Workspace exposes a chat and Studio entrypoint for starting a workspace task using selected sources and user instructions, routed through existing governed agent-task/ACP capabilities.
- [x] #3 The UI surfaces observable task activity/provenance affordances without exposing hidden chain-of-thought or bypassing capability/sandbox checks.
- [x] #4 Generated files/reports can be saved back as workspace artifacts or notes where the existing storage model supports it.
- [x] #5 Focused frontend tests and applicable verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-07-05-research-workspace-notebooklm-agent-tasks-wp4-plan.md.

Commits:
- 3d0fe2c66d feat: preserve research workspace agent task context
- b64e2e2221 feat: start research workspace tasks from chat
- b78cebe120 feat: start research workspace tasks from studio
- 84913a7db7 feat: save ACP run results to workspace artifacts
- eb2c5f56a2 fix: address research workspace WP4 review issues

Implementation notes:
- Chat and Studio now expose Start workspace task entrypoints that prefill the existing ACP handoff with workspace, selected-source, and user-instruction context.
- The handoff preserves context metadata under research_workspace_task_context and retains existing ACP capability/sandbox/approval flow.
- ACP run history shows observable counts for artifacts/files, diagnostics/warnings, audit/approvals, and events/tool activity without hidden chain-of-thought.
- Completed ACP run results can be saved into Studio outputs as traceable report artifacts with producer metadata and version chains.
- Code review by subagent Laplace found three Important issues and one Minor issue; all were addressed in eb2c5f56a2 with regressions for duplicate saves, unsent chat drafts, and non-usable selected Studio sources.

Verification:
- bunx vitest run ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/ChatPane.stage1.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage1.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage2.responsive.test.tsx => 177 passed.
- bun run typecheck => passed.
- git diff --check => passed.
- Bandit: not run because touched scope is frontend TypeScript/TSX only and this worktree has no .venv; no backend Python touched.

PR #2664 review pass after rebase:
- Rebased codex/research-workspace-notebooklm-wp4 onto latest origin/dev.
- Addressed Gemini/Qodo/CodeRabbit review comments: nullable truncation guard, bounded source IDs with full selectedSourceCount, draft-vs-latest-message labeling, safe draft restore, localized ACP activity labels, generatedArtifacts fallback, translated saved artifact title, task producerId provenance, version suffix parsing for artifacts without numeric version, and invalid completedAt fallback.
- Verification after review fixes: focused Research Workspace vitest suite => 180 passed; bun run typecheck => passed; git diff --check => passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
WP4 is implemented for Research Workspace agent tasks: Chat and Studio can start governed ACP workspace tasks with selected-source/user-instruction context, ACP history exposes observable task activity, and completed ACP run results can be saved back as versioned traceable Studio artifacts. Focused frontend tests and typecheck pass; Bandit is documented as not applicable for this frontend-only slice.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed or explicitly deferred with rationale.
- [x] #2 Tests or verification recorded.
- [x] #3 No new security findings in touched backend Python scope, or Bandit skip documented if frontend-only.
- [x] #4 Plan and task records updated.
- [x] #5 Known skips or blockers documented.
<!-- DOD:END -->
