# TASK13260.216 / UAT275 — World Book parent summary invalidation

## Problem

The parent World Books query uses `['tldw:listWorldBooks']` and displays each book's `entry_count`. Entry mutations invalidate only their `['tldw:listWorldBookEntries', worldBookId]` query, leaving the parent count stale until a reload.

## Bounded design

1. Add a standalone real-QueryClient test which renders a parent summary query and `WorldBookEntryManager` together.
2. Retain RED: successful add changes the backing service state but the visible parent count remains zero without the parent query invalidation.
3. After UAT274 is committed, add the parent-list invalidation alongside existing successful entry-list invalidations. Preserve error behavior by calling it only from existing `onSuccess` handlers.
4. Verify add and delete change the visible count 0→1→0, and run focused World Book regressions.

No client transport, API, runtime, or global query behavior changes are in scope.
