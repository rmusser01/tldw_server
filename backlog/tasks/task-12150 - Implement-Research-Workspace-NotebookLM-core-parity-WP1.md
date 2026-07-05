---
id: TASK-12150
title: Implement Research Workspace NotebookLM-core parity WP1
status: Done
assignee: []
created_date: '2026-07-05 00:46'
updated_date: '2026-07-05 01:55'
labels:
  - research-workspace
  - notebooklm
  - frontend
dependencies: []
references:
  - TASK-12149
documentation:
  - >-
    Docs/superpowers/plans/2026-07-04-research-workspace-notebooklm-core-parity-wp1-plan.md
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add Source modal explains supported imports and Google-specific skips using existing ingestion capabilities.
- [x] #2 ChatPane exposes response style and length presets without new prompt-template plumbing or global settings.
- [x] #3 Studio output grouping foregrounds Notebook basics while keeping advanced outputs available.
- [x] #4 Extension save-and-open routes clips with workspace placement to Research Workspace.
- [x] #5 Focused Research Workspace and clipper tests pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented from Docs/superpowers/plans/2026-07-04-research-workspace-notebooklm-core-parity-wp1-plan.md in worktree .worktrees/research-workspace-notebooklm-wp1 on branch codex/research-workspace-notebooklm-wp1.

Commits:
- ee7c3e252f chore: track research workspace notebooklm wp1
- 51c1d4d77d test: avoid inline snapshot in source ingestion utils
- fd6fbfb964 feat: clarify research workspace source intake
- 4eebce9957 feat: add research workspace chat response presets
- 0c50621999 feat: clarify research workspace output groups
- 9606edaf9a fix: open clipped workspace sources in research workspace

Review gates: each implementation task passed spec and code-quality review. Final whole-branch review approved with only a minor optional test-hardening note for secondary Studio group labels.

Verification passed: bunx vitest run ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage2.intake.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage1.ingestion.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/source-ingestion-utils.test.ts ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/ChatPane.stage3.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx ../packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx => 7 files passed, 131 tests passed.

Verification caveat: bun run typecheck failed on pre-existing unrelated errors outside touched WP1 files: AudioStudio/TimelineEditor, ScheduledTasks editor/control-plane, Skills Manager, mcp-hub, voice-cloning, and e2e fixtures. No WP1 touched file was listed.

Bandit: skipped for final touched scope because changes are frontend TS/TSX/test and Backlog markdown only; no Python files were touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented WP1 NotebookLM-core parity for Research Workspace. Add Source now states supported imports and Google-specific exclusions. Chat has local response style/length presets using per-turn message shaping only. Studio foregrounds Notebook basics outputs while keeping advanced outputs behind More outputs. The extension clipper now opens Research Workspace whenever a save response includes workspace placement. Focused verification passed; repo-wide frontend typecheck remains blocked by unrelated existing errors outside this scope.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Acceptance criteria completed
- [x] #8 Tests or verification recorded
- [x] #9 Documentation updated when relevant
- [x] #10 Bandit run for touched code when applicable or document frontend-only skip
- [x] #11 Final summary added
<!-- DOD:END -->
