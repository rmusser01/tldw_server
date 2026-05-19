# E5 Storage Quota Warnings + Mission Control Home — Design

## Context

After completing the full FTUX audit (PRs #993, #997, #998, #1000), two deferred items remain:
- **E5:** Storage quota handling is invisible until failure
- **Mission Control Home:** No progressive guidance after onboarding

Both build on the persona infrastructure shipped in PR #1000.

## Feature 1: Storage Quota Warnings

### Problem
5MB localStorage budget with shard/split/IndexedDB offload. Users get no warning before data loss. Recovery only triggers after write failure.

### Architecture
1. **`useStorageQuota()` hook** — event-driven measurement of all `tldw*` keys, thresholds at 70/85/95%
2. **`StorageQuotaBanner`** — warning/critical/exceeded banners with actionable guidance
3. **`checkStorageBeforeWrite()`** — advisory pre-write guard for cross-feature use
4. Extract `estimateWorkspacePersistedPayloadBytes` and `resolveWorkspacePayloadBudgetBytes` into shared `storage-budget.ts`

### Phases
1. Core hook + shared utils
2. Banner component + mount
3. Refactor WorkspacePlayground to use hook
4. Cross-feature guard integration

## Feature 2: Mission Control Home

### Problem
After onboarding, 47 nav items visible. No guided path from "connected" to "productive."

### Architecture
1. **Milestone store** — tracks first_connection, first_ingest, first_chat, etc. with localStorage persistence
2. **Mission card registry** — ~15-20 curated cards filtered by persona + milestones + capabilities
3. **`useMissionCards()` hook** — resolves cards into gettingStarted, whatsNext, discovery
4. **GettingStartedSection + WhatsNextCard** — integrated into CompanionHomePage
5. **Feature discovery badges** — session-scoped "new" dots on nav items

### Constraint
Progressive unlock does NOT gate features — only surfaces/highlights them.

### Phases
1. Milestone store + bootstrap from existing usage
2. Mission card registry + resolution hook
3. Getting Started + What's Next UI
4. Feature discovery badges
