# Knowledge QA Stage 4 Scoped Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make category selection, exact document/note selection, saved profiles, request payloads, and rendered result sources auditable across WebUI and extension.

**Architecture:** Keep scope state in the existing Knowledge QA provider/context controls and assert it round-trips into `buildRagSearchRequest()`. Add result-source validation so excluded sources cannot appear unless web fallback or explicit response metadata explains the broadened scope.

**Tech Stack:** TypeScript, React, Vitest, Playwright.

**Backlog Task:** TASK-2279.6

---

## Boundaries

- Do not redesign the source picker.
- Do not add source-owner CRUD flows to `/knowledge`.
- Do not add flashcard behavior to `/knowledge`.

## Files

- Modify: `apps/packages/ui/src/services/rag/unified-rag.ts`
- Create: `apps/packages/ui/src/services/rag/__tests__/unified-rag.test.ts`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/KnowledgeQAProvider.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/context/KnowledgeContextBar.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/context/CompactToolbar.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/layout/KnowledgeQALayout.tsx`
- Create: `apps/packages/ui/src/components/Option/KnowledgeQA/scopeValidation.ts`
- Create: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/scopeValidation.test.ts`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.profiles.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.scalable-source-picker.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeQALayout.behavior.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.streaming.test.tsx`
- Modify: `apps/tldw-frontend/e2e/ux-audit/knowledge-empty-recovery.spec.ts`
- Modify: `apps/extension/tests/e2e/knowledge-empty-recovery.spec.ts`

## Task 1: Assert Scope Request Payload

- [ ] **Step 1: Write failing request-builder test**

Create `apps/packages/ui/src/services/rag/__tests__/unified-rag.test.ts` if it
does not already exist, then assert:

```ts
const request = buildRagSearchRequest({
  ...DEFAULT_RAG_SETTINGS,
  query: "Only selected docs",
  sources: ["media_db", "notes"],
  include_media_ids: [42],
  include_note_ids: ["note-a"],
  enable_web_fallback: false,
})

expect(request.include_media_ids).toEqual([42])
expect(request.include_note_ids).toEqual(["note-a"])
expect(request.sources).toEqual(["media_db", "notes"])
expect(request.enable_web_fallback).toBe(false)
```

Run:

```bash
cd apps/packages/ui
bunx vitest run src/services/rag/__tests__/unified-rag.test.ts
```

Expected: fail if test file or exact payload support is missing.

- [ ] **Step 2: Update request builder**

Modify `unified-rag.ts` so exact media and note filters are preserved when building RAG requests.

## Task 2: Validate Result Scope

- [ ] **Step 1: Write failing scope validation tests**

Create `scopeValidation.test.ts`:

```ts
import { validateKnowledgeResultScope } from "../scopeValidation"

it("flags excluded local sources", () => {
  const result = validateKnowledgeResultScope({
    selectedMediaIds: [42],
    selectedNoteIds: [],
    webFallbackEnabled: false,
    results: [{ metadata: { source_type: "media_db", source_id: "99" } }],
  })

  expect(result.violations).toEqual([{ sourceId: "99", reason: "excluded_source" }])
})
```

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/KnowledgeQA/__tests__/scopeValidation.test.ts
```

Expected: fail because helper does not exist.

- [ ] **Step 2: Implement `scopeValidation.ts`**

Keep it pure and small. It should allow:

- selected local media/note ids
- web fallback sources only when web fallback is enabled and origin is `web_fallback`
- explicit metadata reason such as `scope_broadened_by_workspace`

## Task 3: Wire Profiles And Compact Mode

- [ ] **Step 1: Update profile tests**

Add assertions to `KnowledgeContextBar.profiles.test.tsx` that saving and restoring a profile preserves source categories, media ids, note ids, preset, web fallback, and provider/model fields.

- [ ] **Step 2: Update compact parity tests**

Update `KnowledgeQALayout.behavior.test.tsx` so compact source controls expose exact counts and can restore saved profiles.

- [ ] **Step 3: Implement minimal state wiring**

Modify `KnowledgeQAProvider.tsx`, `KnowledgeContextBar.tsx`, and `CompactToolbar.tsx` to preserve exact ids in profile save/restore and compact actions.

## Task 4: Verify

- [ ] **Step 1: Run focused Vitest**

```bash
cd apps/packages/ui
bunx vitest run src/services/rag/__tests__/unified-rag.test.ts src/components/Option/KnowledgeQA/__tests__/scopeValidation.test.ts src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.profiles.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeQALayout.behavior.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.streaming.test.tsx
```

- [ ] **Step 2: Run WebUI and extension route-state checks**

```bash
cd apps/tldw-frontend
bunx playwright test e2e/ux-audit/knowledge-empty-recovery.spec.ts --project=chromium --reporter=line

cd apps/extension
bunx playwright test tests/e2e/knowledge-empty-recovery.spec.ts --project=chromium-extension --reporter=line
```

Record any WXT blocker under TASK-2279.5 and TASK-2279.6.

- [ ] **Step 3: Commit**

```bash
git add apps/packages/ui/src/services/rag apps/packages/ui/src/components/Option/KnowledgeQA apps/tldw-frontend/e2e/ux-audit/knowledge-empty-recovery.spec.ts apps/extension/tests/e2e/knowledge-empty-recovery.spec.ts "backlog/tasks/task-2279.6 - Verify-Knowledge-QA-scoped-search-and-saved-profile-round-trip.md"
git commit -m "feat: verify knowledge qa scoped search"
```
