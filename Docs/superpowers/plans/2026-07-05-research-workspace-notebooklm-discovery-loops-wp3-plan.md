## WP3: Connect Discovery Loops

**Backlog**: TASK-12894
**Branch**: `codex/research-workspace-notebooklm-wp3`
**Worktree**: `.worktrees/research-workspace-notebooklm-wp3`

### Context

WP3 follows the approved Research Workspace / NotebookLM Pro+Ultra parity design:

- Server web search results should become reviewable workspace sources with visible import state.
- Deep Research returns should show provenance and import outcome details inside Research Workspace.
- Browser extension clipper actions should clearly hand off captured pages into workspace, chat, and agent-task workflows where existing storage/routing supports it.
- Explicit non-scope: Google Drive search, Google account integration, new ingestion providers, and a full Research Workspace sidepanel clone.

Existing implementation already has the core paths:

- `AddSourceModal` has a Search Server tab that calls `tldwClient.webSearch()` and imports selected results with `tldwClient.addMedia()`.
- `ResearchWorkspace` already reads Deep Research return query params and imports bundles via `buildDeepResearchBundleArtifactPayload()`.
- `WorkspaceAgentTaskHandoffModal` already creates agent-orchestration tasks from Research Workspace canonical workspace context.
- `WebClipperPanel` already saves clips to note/workspace destinations, opens workspace/notes, and queues chat handoff state for Analyze.

The WP3 implementation should strengthen these existing paths instead of adding parallel flows.

## Stage 1: Server Web Search Import State

**Goal**: Make Search Server results behave like discovery candidates that visibly become workspace sources.

**Files**:

- `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/AddSourceModal.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/index.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage2.intake.test.tsx`

**Implementation**:

- Thread existing `researchWorkspaceCapabilities` from `ResearchWorkspace` to `SourcesPane`, then to `AddSourceModal`, and only pass the existing `source_browse` capability into the Search Server tab.
- Use `getCapabilityCopy()` for a small inline warning/block notice in Search Server. Do not add a new backend capability id.
- Treat `source_browse` as the available workspace source-readiness signal. The current backend capability model does not expose a separate web-search-provider readiness bit; provider/search failures remain surfaced through the Search Server error path. If a separate web-search capability is found while implementing, use it and add a matching test; otherwise record the narrower signal in TASK-12894.
- When selected results are imported, preserve row-level state after the import attempt:
  - imported rows show a concise success status, media id if available, and copy that they are queued as workspace sources.
  - failed rows show a concise failure status and reason so the user can retry or import fewer results.
  - partially successful batches keep successful rows added and leave failed rows selectable/retryable.
- Keep the current ingestion fields and `onAddSources` path. Do not add providers, new routes, or side effects outside the workspace source add flow.

**Tests**:

- Add a test where two search results are selected, one `addMedia` succeeds and one fails; assert the successful source is added and both row states are visible.
- Add a test for `source_browse` capability mode `block`; assert the Search action is disabled and the capability copy is shown.

**Success Criteria**:

- User can search, select results, import them, and see which results became workspace sources.
- Import failures are visible without losing the original search result context.
- Search Server is capability-gated by the existing workspace capability model.

**Status**: Complete

## Stage 2: Deep Research Return Provenance

**Goal**: Make Deep Research return/import state explain what was imported, what evidence came with it, and what needs review.

**Files**:

- `apps/packages/ui/src/components/Option/ResearchWorkspace/deep-research-bundle-import.ts`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/deep-research-bundle-import.test.ts`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage2.responsive.test.tsx`

**Implementation**:

- Extend imported report content using fields already present in the Deep Research bundle:
  - source inventory count and source titles/ids, bounded by existing import list limits.
  - unresolved questions.
  - unsupported claims and contradictions counts/details when present.
  - skipped/failed source details if the returned bundle includes such lists.
- Store these bounded details under `artifact.data.deepResearch` alongside existing provenance, not in a new artifact type.
- Explicitly store/display selected or imported source entries with source ids, titles, and import/review status. Use source artifact coverage when present; otherwise derive from bundle `source_inventory`.
- Bound persisted and rendered source/provenance/skipped/failed detail lists with the existing `MAX_IMPORT_LIST_ITEMS` cap of 50 items.
- Enhance the return handoff banner with concise provenance/outcome copy:
  - run id.
  - source artifact.
  - imported status.
  - selected/imported source count.
  - source inventory/import-review summary when import succeeds.
  - failed import reason and retryable button state when import fails.
- Keep the existing import button retry behavior for failed imports.

**Tests**:

- Extend the bundle import unit test to assert imported content includes selected/imported source ids/titles/statuses, source inventory, unresolved questions, unsupported claim or contradiction details, and bounded skipped/failed source lists.
- Extend the workspace responsive test to assert the handoff shows imported selected-source/provenance summary after a successful import and retains failure copy on failed import.

**Success Criteria**:

- User can tell which Deep Research run and source artifact produced an imported report.
- User can tell which selected/imported sources belong to the returned report.
- Imported reports carry enough source/failure/review metadata to recover or audit the discovery loop.
- Failed imports show a clear reason and can be retried without navigating away.

**Status**: Complete

## Stage 3: Extension Clipper Handoff Affordances

**Goal**: Make the browser extension actions plainly map to workspace, chat, and agent-task discovery loops using existing save/handoff mechanisms.

**Files**:

- `apps/packages/ui/src/components/Sidepanel/Clipper/WebClipperPanel.tsx`
- `apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx`
- Existing Research Workspace route/header/handoff files needed to open `WorkspaceAgentTaskHandoffModal` from extension context.
- New tiny web-clipper pending agent-task handoff helper only if direct query params would make the route too large or leak page text into URLs.

**Implementation**:

- Update action labels/copy so the current behaviors are explicit:
  - save to workspace destination remains the capture-to-workspace path.
  - Analyze/chat action is labeled as asking chat about the captured page.
  - open action says workspace when workspace placement is selected.
- Add a small action hint that changes with destination/action readiness; keep it compact for the sidepanel.
- Wire a fourth action, `Start agent task`, for saved workspace clips by reusing the existing Research Workspace agent-task creation surface:
  - save the clip to a workspace first, requiring a successful `workspace_placement`.
  - pass captured page context to Research Workspace as a pending agent-task handoff, using compact session storage rather than long URLs if page text is included.
  - open `options.html#/research-workspace` with enough route state to focus the workspace and trigger `WorkspaceAgentTaskHandoffModal`.
  - prefill task title/description with the captured page title, URL, note id, workspace placement, and bounded extract preview.
  - do not create an agent-orchestration task directly in the sidepanel; keep root-path and ACP readiness checks in `WorkspaceAgentTaskHandoffModal`.

**Tests**:

- Update existing save/open/analyze tests for the new labels.
- Add an assertion that workspace destination shows workspace-specific action copy.
- Add an assertion that `Start agent task` saves a pending handoff and opens Research Workspace with the existing agent-task modal route.
- Add an assertion that `Start agent task` requires a workspace destination/placement, stores bounded captured-page pending context, and opens Research Workspace with the handoff trigger.
- Add or extend a Research Workspace/header test that consumes the pending clipper agent-task context and opens/prefills `WorkspaceAgentTaskHandoffModal`.

**Success Criteria**:

- User can clearly capture the page to a workspace, open the workspace, and ask chat about the captured page.
- Agent-task behavior is connected to the existing Research Workspace agent-task modal; no direct sidepanel orchestration path or fake action is shipped.
- Sidepanel actions remain usable in narrow layouts.

**Status**: Complete

## Stage 4: Integration And Verification

**Goal**: Prove the WP3 slices work together without broad churn.

**Verification**:

- `bunx vitest run apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage2.intake.test.tsx`
- `bunx vitest run apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/deep-research-bundle-import.test.ts`
- `bunx vitest run apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage2.responsive.test.tsx`
- `bunx vitest run apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx`
- `bunx vitest run apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx`
- `bunx vitest run apps/packages/ui/src/services/web-clipper/__tests__/agent-task-handoff.test.ts`
- `git diff --check`
- Bandit: skipped if the final touched scope is frontend/docs/backlog only; otherwise run Bandit on touched backend paths from the project virtual environment.

**Success Criteria**:

- Focused frontend tests pass.
- Whitespace check passes.
- Backlog TASK-12894 records implementation summary and verification.
- Commit contains the Backlog task, plan, implementation, and tests.

**Status**: Complete
