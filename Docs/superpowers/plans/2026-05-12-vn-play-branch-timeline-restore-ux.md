# VN Play Branch Timeline Restore UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a player-facing VN Play branch timeline and restore UX that consumes the backend-owned branch navigation API for Story/CYOA sessions.

**Architecture:** Keep branch semantics server-authoritative. The frontend adds typed client helpers for `/branch-navigation` and `/branches/{branch_id}/restore`, renders the server-shaped read model in a focused play component, and refreshes session state from backend responses after restore. The component must not reconstruct branches from raw events or infer restore targets beyond the backend-provided `restore` capability payload.

**Tech Stack:** Next.js/React, TypeScript, Vitest, Testing Library, existing `@web/lib/api/vnPlay` API helper style, existing VN Play runtime idempotency and recoverable-error helpers.

---

### Task 1: Typed Branch Navigation Client Contract

**Files:**
- Modify: `apps/tldw-frontend/types/vn-play.ts`
- Modify: `apps/tldw-frontend/lib/api/vnPlay.ts`
- Test: `apps/tldw-frontend/__tests__/vn-play/vnPlayApi.test.ts`

- [x] **Step 1: Add failing API client tests**

Add coverage that calls:

```ts
await getVNPlayBranchNavigation(1);
await restoreVNPlayBranch(1, 12, {
  client_scene_version: 6,
  idempotency_key: 'restore-branch-12',
  target: 'choice_point',
});
```

Expected mocked calls:

```ts
expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-play/sessions/1/branch-navigation');
expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn/vn-play/sessions/1/branches/12/restore', {
  client_scene_version: 6,
  idempotency_key: 'restore-branch-12',
  target: 'choice_point',
});
```

- [x] **Step 2: Run red test**

Run from `apps/tldw-frontend`:

```bash
bun run test:run __tests__/vn-play/vnPlayApi.test.ts -t 'branch navigation'
```

Expected: FAIL because the helper/types do not exist yet.

- [x] **Step 3: Add TypeScript types**

Mirror `VNPlayBranchNavigationResponse`, `VNPlayBranchNavigationNode`, `VNPlayBranchPathStep`, `VNPlayBranchRestoreRequest`, `VNPlayBranchRestoreResponse`, warnings, event ranges, generated-choice refs, and restore capability from `tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py`.

Keep literal targets:

```ts
export type VNPlayBranchRestoreTarget = 'branch_latest' | 'choice_point';
```

- [x] **Step 4: Add API helpers**

Add:

```ts
export function getVNPlayBranchNavigation(sessionId: number): Promise<VNPlayBranchNavigationResponse> {
  return apiClient.get(`${VN_PLAY_BASE}/sessions/${sessionId}/branch-navigation`);
}

export function restoreVNPlayBranch(
  sessionId: number,
  branchId: number,
  request: VNPlayBranchRestoreRequest
): Promise<VNPlayBranchRestoreResponse> {
  return apiClient.post(`${VN_PLAY_BASE}/sessions/${sessionId}/branches/${branchId}/restore`, request);
}
```

- [x] **Step 5: Run green test**

```bash
bun run test:run __tests__/vn-play/vnPlayApi.test.ts -t 'branch navigation'
```

Expected: PASS.

### Task 2: Branch Timeline Presentation Component

**Files:**
- Create: `apps/tldw-frontend/components/vn-play/BranchTimelinePanel.tsx`
- Test: `apps/tldw-frontend/__tests__/vn-play/BranchTimelinePanel.test.tsx`

- [x] **Step 1: Write failing render tests**

Cover:
- no Story session or no branches shows a concise no-branches empty state.
- active path renders as ordered choice breadcrumbs from `active_path`.
- branches render labels, choice text, active/on-path state, depth, event range summary, warnings, and restore buttons only when `restore.supported`.
- `branch_latest` and `choice_point` buttons use backend-provided `restore.target_names`; do not invent unavailable targets.

- [x] **Step 2: Run red component tests**

```bash
bun run test:run __tests__/vn-play/BranchTimelinePanel.test.tsx
```

Expected: FAIL because the component does not exist.

- [x] **Step 3: Implement the component**

Props:

```ts
interface BranchTimelinePanelProps {
  navigation: VNPlayBranchNavigationResponse | null;
  isLoading?: boolean;
  restoringBranchId?: number | null;
  restoreTarget?: VNPlayBranchRestoreTarget | null;
  onRestoreBranch?: (branchId: number, target: VNPlayBranchRestoreTarget) => void | Promise<void>;
}
```

Rendering rules:
- Show only player-safe branch data from `navigation`.
- Label `branch_latest` as `Resume branch`.
- Label `choice_point` as `Return to choice`.
- Mark `is_active` as `Active`.
- Mark `is_on_active_path` as `On path`.
- Render warnings with `message ?? code`.
- Keep layout dense and inside one component panel; do not nest cards inside cards.

- [x] **Step 4: Run green component tests**

```bash
bun run test:run __tests__/vn-play/BranchTimelinePanel.test.tsx
```

Expected: PASS.

### Task 3: Workspace Wiring And Restore Flow

**Files:**
- Modify: `apps/tldw-frontend/components/vn-play/VNPlayWorkspace.tsx`
- Test: `apps/tldw-frontend/__tests__/vn-play/VNPlayWorkspace.test.tsx`

- [x] **Step 1: Add failing workspace tests**

Add tests that:
- load `getVNPlayBranchNavigation(sessionId)` with the selected Story session.
- render player-facing branch timeline content outside the runtime inspector.
- call `restoreVNPlayBranch(sessionId, branchId, { client_scene_version, idempotency_key, target })`.
- update selected session, events, checkpoints, branches, and branch navigation from the restore response.
- surface `stale_scene_version`, `turn_in_progress`, and `restore_action_in_progress` as recoverable play-state messages.

- [x] **Step 2: Run red workspace tests**

```bash
bun run test:run __tests__/vn-play/VNPlayWorkspace.test.tsx -t 'branch'
```

Expected: FAIL for missing API mock/import/component wiring.

- [x] **Step 3: Load branch navigation with session collections**

Replace branch-only loading with:
- `getVNPlayBranchNavigation(selectedSession.id)` for branch UX.
- `listVNPlayBranches(selectedSession.id)` may remain for inspector compatibility if needed.

Failure behavior:
- If branch navigation fails, keep session playable and show an unobtrusive branch-panel error/empty state.

- [x] **Step 4: Implement branch restore handler**

Use:

```ts
idempotency_key: createVNPlayIdempotencyKey('restore-branch')
client_scene_version: sceneVersion
target
```

On success:
- set `selectedSession` from `response.session`.
- merge the session in the session list.
- update scene from `response.current_scene`.
- update branch navigation from `response.branch_navigation`.
- refresh `events`, `checkpoints`, and compatibility `branches`.

On recoverable conflicts:
- use `getVNPlayErrorInfo` and `isRecoverableVNPlayConflict` patterns already used by turn handling.
- reload selected session where appropriate.

- [x] **Step 5: Run green workspace tests**

```bash
bun run test:run __tests__/vn-play/VNPlayWorkspace.test.tsx -t 'branch'
```

Expected: PASS.

### Task 4: Docs, Task Notes, And Verification

**Files:**
- Modify: `Docs/API-related/VN_PLAY_API.md`
- Modify: `backlog/tasks/task-284 - Add-VN-Play-branch-timeline-and-restore-UX.md`

- [x] **Step 1: Update frontend workspace docs**

In the Frontend Workspace section, document that `/vn-play` now uses backend branch navigation for Story branch timeline and restore controls.

- [x] **Step 2: Run focused frontend tests**

```bash
bun run test:run __tests__/vn-play/vnPlayApi.test.ts __tests__/vn-play/BranchTimelinePanel.test.tsx __tests__/vn-play/VNPlayWorkspace.test.tsx __tests__/vn-play/SceneStage.test.tsx __tests__/vn-play/vnPlayRuntime.test.ts
```

Expected: PASS.

- [x] **Step 3: Run lint and diff hygiene**

```bash
bun run lint -- components/vn-play/BranchTimelinePanel.tsx components/vn-play/VNPlayWorkspace.tsx __tests__/vn-play/BranchTimelinePanel.test.tsx __tests__/vn-play/VNPlayWorkspace.test.tsx __tests__/vn-play/vnPlayApi.test.ts
git diff --check
```

Expected: lint exits 0 with only existing repo-wide warnings if present; diff check exits 0.

- [x] **Step 4: Record Bandit status**

Bandit is not applicable if this slice only touches TypeScript/React, Markdown, and Backlog metadata. If backend Python changes become necessary, run Bandit on touched backend paths before finalizing.

- [x] **Step 5: Update Backlog task**

Record verification results, known skips, and final summary in `TASK-284`.

- [x] **Step 6: Commit**

```bash
git add \
  apps/tldw-frontend/types/vn-play.ts \
  apps/tldw-frontend/lib/api/vnPlay.ts \
  apps/tldw-frontend/components/vn-play/BranchTimelinePanel.tsx \
  apps/tldw-frontend/components/vn-play/VNPlayWorkspace.tsx \
  apps/tldw-frontend/__tests__/vn-play/BranchTimelinePanel.test.tsx \
  apps/tldw-frontend/__tests__/vn-play/VNPlayWorkspace.test.tsx \
  apps/tldw-frontend/__tests__/vn-play/vnPlayApi.test.ts \
  Docs/API-related/VN_PLAY_API.md \
  "backlog/tasks/task-284 - Add-VN-Play-branch-timeline-and-restore-UX.md" \
  Docs/superpowers/plans/2026-05-12-vn-play-branch-timeline-restore-ux.md
git commit -m "Add VN Play branch timeline UX"
```
