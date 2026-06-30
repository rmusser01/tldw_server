# Knowledge QA Stage 1A Trust Taxonomy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a shared Knowledge QA trust taxonomy and safe response handling so uncited, partial, failed, unknown, no-result, and unsynced states cannot render as grounded success.

**Architecture:** Put trust normalization in a small shared helper consumed by `KnowledgeQAProvider`, answer UI, history, and export-entry surfaces. This stage should classify and render states, but it should not enforce backend citation validity yet. Backend enforcement is owned by TASK-2279.4 after evidence payloads are materialized.

**Tech Stack:** TypeScript, React, Vitest, shared Knowledge QA UI.

**Backlog Task:** TASK-2279.2

**Status:** Complete on 2026-06-07. Verification passed for the focused Stage 1A test set, provider slice, full Knowledge QA test folder, scope guard, diff hygiene, and Bandit touched-scope scan.

---

## Boundaries

- Do not change backend RAG behavior in this stage.
- Do not add flashcard behavior to `/knowledge`.
- Do not infer grounded success from older or partial payloads.
- Keep labels neutral and operational: cited, degraded, unknown, no result, failed, unsynced.

## Files

- Create: `apps/packages/ui/src/components/Option/KnowledgeQA/trustState.ts`
- Create: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/trustState.test.ts`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/types.ts`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/trustSummary.ts`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/KnowledgeQAProvider.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/AnswerPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/panels/LowQualityRecoveryBanner.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/HistorySidebar.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/empty/InlineRecentSessions.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/AnswerPanel.states.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/trustSummary.test.ts`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.history.test.tsx`

## Task 1: Define Shared Trust Types

- [x] **Step 1: Write failing trust-state tests**

Create `trustState.test.ts`:

```ts
import { describe, expect, it } from "vitest"
import { normalizeKnowledgeAnswerTrust } from "../trustState"

describe("normalizeKnowledgeAnswerTrust", () => {
  it("fails closed for older payloads without trust metadata", () => {
    expect(normalizeKnowledgeAnswerTrust({ answer: "Answer", results: [], citations: [] }).state)
      .toBe("unknown_trust")
  })

  it("marks answer text without valid citations as degraded", () => {
    expect(normalizeKnowledgeAnswerTrust({
      answer: "Answer without citations",
      results: [{ id: "source-1", content: "Evidence" }],
      citations: [],
      hasRequiredMetadata: true,
    }).state).toBe("uncited_degraded_answer")
  })

  it("preserves unsynced local result over cited answer", () => {
    expect(normalizeKnowledgeAnswerTrust({
      answer: "Answer [1]",
      results: [{ id: "source-1", excerpt: "Evidence" }],
      citations: [{ index: 1, documentId: "source-1" }],
      hasRequiredMetadata: true,
      syncFailed: true,
    }).state).toBe("unsynced_local_result")
  })
})
```

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/KnowledgeQA/__tests__/trustState.test.ts
```

Expected: fail because `trustState.ts` does not exist.

- [x] **Step 2: Implement minimal trust helper**

First add the public trust state type to `types.ts`:

```ts
export type KnowledgeAnswerTrustState =
  | "cited_answer"
  | "uncited_degraded_answer"
  | "no_answer_insufficient_evidence"
  | "no_results"
  | "failed_search"
  | "unsynced_local_result"
  | "unknown_trust"
```

Then create `trustState.ts`:

```ts
import type { CitationRef, KnowledgeAnswerTrustState, RagResult } from "./types"

export type KnowledgeTrustInput = {
  answer: string | null
  results: RagResult[]
  citations: CitationRef[]
  hasRequiredMetadata?: boolean
  transportFailed?: boolean
  syncFailed?: boolean
  weakEvidence?: boolean
}

export function normalizeKnowledgeAnswerTrust(input: KnowledgeTrustInput) {
  if (input.transportFailed) return { state: "failed_search" as const }
  if (input.syncFailed) return { state: "unsynced_local_result" as const }
  if (!input.hasRequiredMetadata) return { state: "unknown_trust" as const }
  if (input.results.length === 0) return { state: "no_results" as const }
  if (input.weakEvidence && !input.answer) return { state: "no_answer_insufficient_evidence" as const }
  if (input.answer && input.citations.length === 0) return { state: "uncited_degraded_answer" as const }
  if (input.answer && input.citations.length > 0) return { state: "cited_answer" as const }
  return { state: "unknown_trust" as const }
}
```

- [x] **Step 3: Add trust fields to shared types**

Modify `types.ts`:

```ts
export type SearchHistoryItem = {
  // existing fields
  trustState?: KnowledgeAnswerTrustState
}

export type KnowledgeQAState = {
  // existing fields
  answerTrustState: KnowledgeAnswerTrustState
}
```

Run the focused test again. Expected: pass after imports are corrected.

## Task 2: Wire Provider State

- [x] **Step 1: Write failing provider tests**

Update `KnowledgeQAProvider.history.test.tsx` to assert:

- failed search history items are `failed_search`
- local-only thread after sync timeout is `unsynced_local_result`
- payload with answer but no citations is `uncited_degraded_answer`

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.history.test.tsx
```

Expected: fail because provider does not persist trust state.

- [x] **Step 2: Normalize in `KnowledgeQAProvider.tsx`**

Call `normalizeKnowledgeAnswerTrust()` when setting results, partial results, errors, local-only thread state, and search history entries.

- [x] **Step 3: Preserve backwards compatibility**

Older payloads without trust metadata must become `unknown_trust`, not `cited_answer`.

## Task 3: Render Trust States

- [x] **Step 1: Write failing UI tests**

Update `AnswerPanel.states.test.tsx` and `trustSummary.test.ts` to assert visible labels for:

- `cited_answer`
- `uncited_degraded_answer`
- `unknown_trust`
- `no_answer_insufficient_evidence`
- `failed_search`
- `unsynced_local_result`

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/KnowledgeQA/__tests__/AnswerPanel.states.test.tsx src/components/Option/KnowledgeQA/__tests__/trustSummary.test.ts
```

Expected: fail until UI labels are wired.

- [x] **Step 2: Update answer and low-quality surfaces**

Modify `AnswerPanel.tsx`, `LowQualityRecoveryBanner.tsx`, and `trustSummary.ts` to style uncited, unknown, failed, and unsynced states as degraded or blocked. Do not use success styling for unsupported answers.

- [x] **Step 3: Update history previews**

Modify `HistorySidebar.tsx` and `InlineRecentSessions.tsx` to show compact trust labels without increasing layout height unexpectedly.

## Task 4: Verify

- [x] **Step 1: Run focused tests**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/KnowledgeQA/__tests__/trustState.test.ts src/components/Option/KnowledgeQA/__tests__/trustSummary.test.ts src/components/Option/KnowledgeQA/__tests__/AnswerPanel.states.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.history.test.tsx
```

Expected: all tests pass.

- [x] **Step 2: Run scope guard**

```bash
rg -n "flashcard|deck|spaced repetition|study set" apps/packages/ui/src/components/Option/KnowledgeQA
```

Expected: no matches in touched Knowledge QA runtime files.

- [x] **Step 3: Run diff hygiene**

```bash
git diff --check -- apps/packages/ui/src/components/Option/KnowledgeQA
```

Expected: exit 0.

- [x] **Step 4: Update Backlog and commit**

```bash
git add apps/packages/ui/src/components/Option/KnowledgeQA "backlog/tasks/task-2279.2 - Define-Knowledge-QA-trust-taxonomy-and-safe-response-handling.md"
git commit -m "feat: add knowledge qa trust taxonomy"
```
