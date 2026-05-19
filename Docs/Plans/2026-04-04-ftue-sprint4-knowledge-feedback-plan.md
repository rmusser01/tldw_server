# FTUE Sprint 4: Knowledge Feedback & Category Naming

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve Knowledge QA for first-time users by renaming confusing source categories and adding indexing-awareness to the no-results state.

**Architecture:** Two isolated changes to existing components: (1) rename source category labels in BasicSettings, (2) enhance NoResultsRecovery with a "content may still be indexing" hint that checks the QuickIngest store for recently ingested items. No new API endpoints, no backend changes.

**Note:** XC-003 (sequenced tutorial) is deferred — it requires multi-page tutorial infrastructure that overlaps with v1 audit Improvement 11. Should be merged with that existing work in a dedicated sprint.

**Tech Stack:** TypeScript/React, existing QuickIngest Zustand store

---

### Task 1: Rename source category labels (KNW-010)

**Files:**
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/SettingsPanel/BasicSettings.tsx:57-84`

**Step 1: Read the current file**

Read `BasicSettings.tsx` and find the source selector (around lines 57-84). Current labels:

```typescript
{ value: "media_db", label: "Documents & Media" },
{ value: "notes", label: "Notes" },
{ value: "characters", label: "Character Cards" },
{ value: "chats", label: "Chat History" },
{ value: "kanban", label: "Kanban" },
```

**Step 2: Rename to plain language**

Change the labels to:

```typescript
{ value: "media_db", label: "Your Documents" },
{ value: "notes", label: "Your Notes" },
{ value: "characters", label: "Characters & Profiles" },
{ value: "chats", label: "Conversations" },
{ value: "kanban", label: "Boards" },
```

**Step 3: Run tests**

```bash
cd apps/packages/ui && npx vitest run src/components/Option/KnowledgeQA/
```

**Step 4: Commit**

```
fix(knowledge): rename source categories to plain language

"Documents & Media" → "Your Documents"
"Character Cards" → "Characters & Profiles"
"Chat History" → "Conversations"
"Kanban" → "Boards"
Notes stays as "Your Notes" (KNW-010).
```

---

### Task 2: Add indexing hint to no-results recovery (KNW-002)

**Files:**
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/panels/NoResultsRecovery.tsx`

**Step 1: Read the current component**

Read `NoResultsRecovery.tsx` completely. Note:
- Props it receives
- The 3 bullet points of suggestions
- The 3 recovery buttons
- Whether it has access to any context about recent ingests

**Step 2: Add indexing hint**

Import the QuickIngest store:
```typescript
import { useQuickIngestStore } from "@/store/quick-ingest"
```

Inside the component, check if there are recently ingested items:
```typescript
const recentlyIngestedDocIds = useQuickIngestStore(s => s.recentlyIngestedDocIds)
const hasRecentIngests = recentlyIngestedDocIds.length > 0
```

Add a conditional hint before the existing suggestions. When `hasRecentIngests` is true, show:

```tsx
{hasRecentIngests && (
  <div className="mb-3 rounded-md border border-amber-500/30 bg-amber-500/5 px-3 py-2">
    <p className="text-xs text-amber-700 dark:text-amber-400">
      You recently ingested documents. If they don't appear in results yet,
      they may still be indexing. Try searching again in a moment.
    </p>
  </div>
)}
```

**Step 3: Run tests**

```bash
cd apps/packages/ui && npx vitest run src/components/Option/KnowledgeQA/
```

**Step 4: Commit**

```
feat(knowledge): add indexing hint when recently-ingested content not found

When a search returns no results and the user has recently ingested
documents (tracked via QuickIngest store), a hint appears explaining
content may still be indexing (KNW-002).
```

---

### Task 3: Regression and verification

**Step 1: Run all relevant tests**

```bash
cd apps/packages/ui && npx vitest run src/components/Option/KnowledgeQA/ src/components/Common/QuickIngest/__tests__/
```

**Step 2: Manual verification**

1. Open Knowledge QA Settings → verify source categories show new names
2. Ingest a document via QuickIngest → search immediately in Knowledge → verify indexing hint appears in no-results state
3. Search for content that exists → verify no hint appears (normal results)

---

## Summary

| File | Change | Issue |
|------|--------|-------|
| `BasicSettings.tsx` | Rename 5 source category labels | KNW-010 (P1) |
| `NoResultsRecovery.tsx` | Add indexing hint for recently-ingested content | KNW-002 (P1) |

## Deferred

| Issue | Reason |
|-------|--------|
| XC-003 (sequenced tutorial) | Requires multi-page tutorial infrastructure; overlaps v1 audit Improvement 11. Prerequisites field exists in TutorialDefinition but is unused. Recommend merging with existing v1 work in a dedicated sprint. |
