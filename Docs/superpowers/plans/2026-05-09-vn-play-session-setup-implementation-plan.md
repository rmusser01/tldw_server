# VN Play Session Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make VN Play session creation usable from the WebUI by replacing raw character and asset-pack ID entry with named selectors, compatibility/readiness guidance, and a manual fallback when selector data cannot load.

**Architecture:** Keep the change local to the existing VN Play frontend surface. Add a small character API wrapper beside the existing VN asset API wrapper, load character and VN asset-pack options inside `NewSessionDialog`, derive compatibility/readiness warnings client-side from existing pack fields and readiness endpoint responses, and preserve the existing `VNPlaySessionCreate` payload.

**Tech Stack:** Next.js/React, Vitest + Testing Library, Playwright smoke test, existing `apiClient`, existing VN asset-pack API, existing characters endpoint.

---

## Stage 1: Selector Data Contracts
**Goal:** Add the minimal WebUI character client/types needed by VN Play setup.
**Success Criteria:** VN Play setup can request character summaries through a small API wrapper without importing the larger shared package client.
**Tests:** Add focused API-wrapper coverage if an equivalent API test pattern exists; otherwise cover through mocked dialog behavior in Stage 2.
**Status:** Complete

**Files:**
- Create: `apps/tldw-frontend/types/characters.ts`
- Create: `apps/tldw-frontend/lib/api/characters.ts`
- Modify: `apps/tldw-frontend/__tests__/vn-play/VNPlayWorkspace.test.tsx`

- [ ] **Step 1: Write failing tests that mock character loading**

Expected behavior:
- Dialog calls `listCharacters()` when opened.
- Character options expose `id`, `name`, `description`, `tags`, and `image_present` enough for display metadata.

- [ ] **Step 2: Run focused Vitest and verify RED**

Run:
```bash
cd apps/tldw-frontend
bunx vitest run __tests__/vn-play/VNPlayWorkspace.test.tsx
```
Expected: failure because `@web/lib/api/characters` does not exist or is not called.

- [ ] **Step 3: Add minimal character types and wrapper**

Implementation notes:
- Use `apiClient.get('/characters/', { params: { limit: 1000, offset: 0 } })`.
- Accept both array and `{ items: [...] }` shaped responses to tolerate existing endpoint variants.
- Do not add character authoring or editing behavior in this slice.

- [ ] **Step 4: Run focused Vitest and verify GREEN for wrapper-driven behavior**

Run:
```bash
cd apps/tldw-frontend
bunx vitest run __tests__/vn-play/VNPlayWorkspace.test.tsx
```

## Stage 2: New Session Selectors and Payload
**Goal:** Replace default raw-ID entry with character and compatible VN asset-pack selectors.
**Success Criteria:** A user can open the dialog, select named records, and create the existing session payload using selected IDs.
**Tests:** `VNPlayWorkspace.test.tsx` covers happy-path selector loading and create payload.
**Status:** Complete

**Files:**
- Modify: `apps/tldw-frontend/components/vn-play/NewSessionDialog.tsx`
- Modify: `apps/tldw-frontend/__tests__/vn-play/VNPlayWorkspace.test.tsx`

- [ ] **Step 1: Write failing selector happy-path test**

Expected behavior:
- Character selector shows a named character.
- VN asset-pack selector shows compatible packs for the selected character.
- Submit sends `primary_character_id` and `vn_asset_pack_id` from selected records, not raw typed IDs.

- [ ] **Step 2: Verify RED**

Run:
```bash
cd apps/tldw-frontend
bunx vitest run __tests__/vn-play/VNPlayWorkspace.test.tsx
```

- [ ] **Step 3: Implement selector loading and derived selection state**

Implementation notes:
- Load characters and packs when `open` becomes true.
- Load readiness for listed packs with `getVNAssetReadiness(pack.id)`.
- Auto-select the first character and then the first compatible ready pack where possible.
- Preserve `linked_chat_id`, `content_rating`, `mode`, `title`, and the existing payload schema.

- [ ] **Step 4: Verify GREEN**

Run:
```bash
cd apps/tldw-frontend
bunx vitest run __tests__/vn-play/VNPlayWorkspace.test.tsx
```

## Stage 3: Readiness, Compatibility, and Empty-State Guidance
**Goal:** Make unready, incompatible, draft, missing-byte, trust-level, and content-rating conditions visible before submit.
**Success Criteria:** Incompatible or unready packs are hard to submit accidentally, and empty/error states tell the user what to fix next.
**Tests:** `VNPlayWorkspace.test.tsx` covers incompatible pack warning, unready/readiness errors, empty selectors, and manual fallback when loading fails.
**Status:** Complete

**Files:**
- Modify: `apps/tldw-frontend/components/vn-play/NewSessionDialog.tsx`
- Modify: `apps/tldw-frontend/__tests__/vn-play/VNPlayWorkspace.test.tsx`

- [ ] **Step 1: Write failing warning and empty-state tests**

Expected behavior:
- Packs for other characters show an incompatibility warning and cannot be submitted through selector mode.
- Readiness warnings/errors are shown, including missing runtime assets.
- Draft/unapproved status is called out before submit.
- Content-rating mismatch is visible when pack rating differs from the session rating.
- No characters or no packs show direct guidance to create/import characters or prepare/review packs.

- [ ] **Step 2: Write failing manual-fallback test**

Expected behavior:
- If character or asset-pack selector loading fails, raw numeric ID inputs are available as a secondary fallback and still create the same payload.

- [ ] **Step 3: Verify RED**

Run:
```bash
cd apps/tldw-frontend
bunx vitest run __tests__/vn-play/VNPlayWorkspace.test.tsx
```

- [ ] **Step 4: Implement warning, submit gating, empty states, and fallback mode**

Implementation notes:
- Selector mode should require a selected character, selected pack, character compatibility, and readiness.
- Manual fallback should be shown automatically after selector load failure and should keep positive-integer validation.
- Avoid adding new backend APIs in this slice.

- [ ] **Step 5: Verify GREEN**

Run:
```bash
cd apps/tldw-frontend
bunx vitest run __tests__/vn-play/VNPlayWorkspace.test.tsx
```

## Stage 4: Smoke Coverage and Closeout
**Goal:** Keep the browser smoke route aligned with selector-based setup and finish repository tracking.
**Success Criteria:** Smoke test mocks characters, packs, and readiness, then creates a Story session through selectors.
**Tests:** Vitest targeted suite, eslint touched frontend files, Playwright smoke if local browser setup is available.
**Status:** Complete

**Files:**
- Modify: `apps/tldw-frontend/e2e/smoke/vn-play.spec.ts`
- Modify: `backlog/tasks/task-157 - Make-VN-Play-session-setup-usable.md`
- Modify: `Docs/superpowers/plans/2026-05-09-vn-play-session-setup-implementation-plan.md`

- [ ] **Step 1: Update smoke test mocks and selectors**

Expected behavior:
- `GET /api/v1/characters/` returns at least one character.
- `GET /api/v1/vn-assets/packs` returns at least one compatible pack.
- `GET /api/v1/vn-assets/packs/:id/readiness` returns ready.
- Test creates the session without filling raw IDs.

- [ ] **Step 2: Run focused verification**

Run:
```bash
cd apps/tldw-frontend
bunx vitest run __tests__/vn-play/VNPlayWorkspace.test.tsx __tests__/vn-play/vnPlayApi.test.ts __tests__/vn-play/SceneStage.test.tsx __tests__/vn-play/vnPlayRuntime.test.ts
bunx eslint components/vn-play/NewSessionDialog.tsx __tests__/vn-play/VNPlayWorkspace.test.tsx e2e/smoke/vn-play.spec.ts lib/api/characters.ts types/characters.ts
```

- [ ] **Step 3: Run Playwright smoke when environment permits**

Run:
```bash
cd apps/tldw-frontend
TLDW_WEB_URL=http://localhost:18081 TLDW_WEB_CMD='bun run dev -- -p 18081' bunx playwright test e2e/smoke/vn-play.spec.ts --reporter=line
```

- [ ] **Step 4: Update Backlog task and plan status**

Expected:
- Acceptance criteria checked if met.
- Verification commands recorded.
- Bandit documented as not applicable for frontend-only touched code.

- [ ] **Step 5: Commit, push, and open PR for issue #1407**

Run:
```bash
git status --short
git add apps/tldw-frontend/components/vn-play/NewSessionDialog.tsx apps/tldw-frontend/__tests__/vn-play/VNPlayWorkspace.test.tsx apps/tldw-frontend/e2e/smoke/vn-play.spec.ts apps/tldw-frontend/lib/api/characters.ts apps/tldw-frontend/types/characters.ts Docs/superpowers/plans/2026-05-09-vn-play-session-setup-implementation-plan.md "backlog/tasks/task-157 - Make-VN-Play-session-setup-usable.md"
git commit -m "feat: make vn play session setup selectable"
git push -u origin codex/vn-play-session-setup-1407
```

---

## Review Note

The writing-plans skill normally asks for a plan-document-reviewer subagent. Current session policy only allows subagents when the user explicitly asks for subagent work, so this plan proceeds without dispatching a reviewer unless requested.
