# VN Play Runtime Playback Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Polish the main VN Play WebUI runtime surface so backend-resolved scene visuals, generated choices, warnings, and recovery states are playable without requiring the generation inspector.

**Architecture:** Keep the API server authoritative. The frontend should render only data already present in `VNPlaySceneState`, `VNPlayChoice`, and existing turn responses; it should not derive generation state, resolve asset packs, or inspect debug payloads. Changes stay scoped to the existing VN Play components and focused tests.

**Tech Stack:** Next.js/React, TypeScript, Vitest, Testing Library, existing `@web/components/ui` primitives.

---

### Task 1: Scene Visual Metadata And Fallbacks

**Files:**
- Modify: `apps/tldw-frontend/components/vn-play/SceneStage.tsx`
- Test: `apps/tldw-frontend/__tests__/vn-play/SceneStage.test.tsx`

- [x] **Step 1: Write failing tests**

Add tests that render:
- background/depth/sprite images from backend `content_url` payloads
- location/mood/time/weather metadata
- missing or rejected visual warnings as user-safe messages
- a no-visual fallback when no backend asset URL is present

- [x] **Step 2: Run tests to verify failure**

Run:
```bash
bun run test:run __tests__/vn-play/SceneStage.test.tsx -t "renders backend scene metadata"
```

Expected: fail before `SceneStage` renders the new metadata/fallback copy.

- [x] **Step 3: Implement minimal scene rendering**

Update `SceneStage` to:
- read only `sceneState.background`, `sceneState.depth`, `sceneState.active_sprites`, `sceneState.active_sprite_items`
- show labels/metadata already provided by the backend
- render clear fallback copy for missing scene visuals
- format warnings with known safe fields such as `reason`, `code`, `message`, `slot_key`, `asset_type`

- [x] **Step 4: Verify tests pass**

Run:
```bash
bun run test:run __tests__/vn-play/SceneStage.test.tsx
```

### Task 2: Generated Choice Presentation

**Files:**
- Modify: `apps/tldw-frontend/components/vn-play/ChoicePanel.tsx`
- Test: `apps/tldw-frontend/__tests__/vn-play/SceneStage.test.tsx`

- [x] **Step 1: Write failing tests**

Add tests for choices with backend metadata such as `source: generated`, `generation_point_key`, or `status`, and for the no-choice state.

- [x] **Step 2: Run tests to verify failure**

Run:
```bash
bun run test:run __tests__/vn-play/SceneStage.test.tsx -t "labels generated choices"
```

Expected: fail until generated choice metadata is rendered.

- [x] **Step 3: Implement choice display**

Update `ChoicePanel` to render generated-choice badges/copy from `choice.metadata` without changing `submitVNPlayTurn` behavior. Preserve loading and idempotency behavior through existing helpers.

- [x] **Step 4: Verify tests pass**

Run:
```bash
bun run test:run __tests__/vn-play/SceneStage.test.tsx
```

### Task 3: Workspace Play Surface Separation

**Files:**
- Modify: `apps/tldw-frontend/components/vn-play/VNPlayWorkspace.tsx`
- Test: `apps/tldw-frontend/__tests__/vn-play/VNPlayWorkspace.test.tsx`

- [x] **Step 1: Write failing tests**

Add focused assertions that the main play surface shows the scene and choices without needing the generation inspector, while keeping the inspector available through a link.

- [x] **Step 2: Run tests to verify failure**

Run:
```bash
bun run test:run __tests__/vn-play/VNPlayWorkspace.test.tsx -t "keeps generation inspector separate"
```

- [x] **Step 3: Adjust workspace copy/layout**

Make the play surface primary. Keep `GenerationInspector` as an audit/debug panel and link, with no new frontend-owned generation logic.

- [x] **Step 4: Verify tests pass**

Run:
```bash
bun run test:run __tests__/vn-play/VNPlayWorkspace.test.tsx
```

### Task 4: Final Verification And PR

**Files:**
- Modify: `backlog/tasks/task-283 - Polish-VN-Play-runtime-playback-surface.md`

- [x] **Step 1: Run focused verification**

Run:
```bash
bun run test:run __tests__/vn-play/SceneStage.test.tsx __tests__/vn-play/VNPlayWorkspace.test.tsx __tests__/vn-play/vnPlayRuntime.test.ts
bun run lint -- components/vn-play/SceneStage.tsx components/vn-play/ChoicePanel.tsx components/vn-play/VNPlayWorkspace.tsx __tests__/vn-play/SceneStage.test.tsx __tests__/vn-play/VNPlayWorkspace.test.tsx
git diff --check
```

- [x] **Step 2: Record known skips**

Record TypeScript baseline status and Bandit skip rationale if only frontend TypeScript files changed.

- [ ] **Step 3: Commit and open PR**

Commit the focused slice and open a PR against `dev` referencing #1587.
