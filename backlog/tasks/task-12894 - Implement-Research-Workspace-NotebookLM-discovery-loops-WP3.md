---
id: TASK-12894
title: Implement Research Workspace NotebookLM discovery loops WP3
status: Done
assignee: []
created_date: ''
updated_date: 2026-07-05 15:33
labels:
- research-workspace
- notebooklm
- wp3
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved WP3 slice from the Research Workspace NotebookLM Pro/Ultra review spec. Scope: make server web search results visibly reviewable/importable/tracked as workspace sources, surface Deep Research return/import provenance and recoverable failures inside the workspace, capability-gate discovery by existing source/web readiness, and tighten extension handoff actions for save/open/chat/agent context without adding Google Drive integration or a full Research Workspace sidepanel clone.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 WP3 implementation plan is saved under Docs/superpowers/plans with exact file paths, tasks, tests, and verification commands.
- [x] #2 Server web search discovery shows selected/imported/failed result state clearly enough for source-to-chat recovery.
- [x] #3 Deep Research return/import state exposes run provenance, imported report metadata, skipped or failed source/import details, and retry/recovery copy where applicable.
- [x] #4 Extension clipper handoffs clearly support capture to workspace, open workspace, ask chat about captured page, and agent-task context using existing routing/storage where available.
- [x] #5 Scope excludes Google Drive search, Google account integration, new ingestion providers, and a full Research Workspace sidepanel clone.
- [x] #6 Focused frontend/backend verification, git diff --check, and Bandit touched-scope results are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started WP3 in isolated worktree `.worktrees/research-workspace-notebooklm-wp3` on branch `codex/research-workspace-notebooklm-wp3` from `origin/dev` at merge commit 242297a2b8. Initial code scan found existing web search import, Deep Research return/import, and clipper save/open/analyze flows; plan should connect and clarify those paths rather than adding new providers or a sidepanel clone.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

WP3 implementation plan created: `Docs/superpowers/plans/2026-07-05-research-workspace-notebooklm-discovery-loops-wp3-plan.md`. Plan review and agent-task handoff exploration dispatched to subagents before implementation.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented WP3 discovery-loop connections for Research Workspace.

Changes:
- Search Server imports now thread Research Workspace capability state into the Add Source modal, block unavailable source-browse paths, keep the modal open after imports, and show row-level imported/failed states with retryable failed rows.
- Deep Research bundle returns now persist and render bounded selected/imported sources, source inventory, skipped sources, failed sources, unsupported claims, contradictions, unresolved questions, source artifact/run provenance, and selected-source import summary copy. Existing source artifact coverage and lineage are sanitized to the MAX_IMPORT_LIST_ITEMS cap before persistence, including dropping unknown oversized lineage fields.
- Web clipper save actions now label chat handoff as Ask chat, add compact destination-aware action hints, and add Start agent task using shared extension storage (`chrome.storage.session`, with fallbacks) plus the existing Research Workspace WorkspaceAgentTaskHandoffModal. Note-only agent-task clicks are blocked before save.
- WorkspaceHeader treats web-clip agent-task prefill as one-shot modal state so later manual task creation does not reuse stale clipped-page context.
- Added/updated focused Vitest coverage for source import state, Deep Research provenance bounds, Research Workspace route handoff/prefill, web clipper save/open/chat/agent flows, header one-shot prefill behavior, and chrome storage callback-error fallback/clear/tombstone behavior.

Review fixes:
- Subagent spec review P1 fixed by bounding copied Deep Research source coverage/lineage before persistence.
- Subagent code-quality P1/P2/P2 fixed by moving agent-task handoff off cross-tab `sessionStorage`, whitelisting lineage fields, and clearing stale web-clip prefill for manual task creation.
- Subagent code-quality storage callback error findings fixed by checking `chrome.runtime.lastError` for get/set/remove callbacks, falling back when set/get fail, tombstoning stale extension handoffs when remove fails, and allowing later fallback handoffs after a tombstone.

Verification:
- PASS `bunx vitest run ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage2.intake.test.tsx` (22 tests)
- PASS `bunx vitest run ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/deep-research-bundle-import.test.ts` (7 tests)
- PASS `bunx vitest run ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage2.responsive.test.tsx` (23 tests)
- PASS `bunx vitest run ../packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx` (37 tests)
- PASS `bunx vitest run ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx` (64 tests)
- PASS `bunx vitest run ../packages/ui/src/services/web-clipper/__tests__/agent-task-handoff.test.ts` (3 tests)
- PASS `git diff --check`
- INFO `NODE_OPTIONS=--max-old-space-size=8192 bun run --cwd ../packages/ui tsc --noEmit` reports existing unrelated UI package type errors in ChatGreetingPicker, MCPHub, background session store, setup onboarding, TldwChat abort, and character-export SSRF tests; no reported diagnostics were in touched WP3 files.
- SKIP Bandit: touched scope is frontend TypeScript/tests, Docs plan, and Backlog task only; no Python/backend touched paths.
<!-- SECTION:FINAL_SUMMARY:END -->
<!-- SECTION:FINAL_SUMMARY:END -->

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
