# Character Row Chat Implicit Model Fallback Fix Plan

**Goal:** Prevent the Characters row `Chat as...` action from treating the first fetched model catalog entry as an explicit full-chat model selection.

## Stage 1: Reproduce The Live Gap

**Goal:** Capture the post-P1 re-audit state where the row action still leaves character-chat context.

**Success Criteria:** Puppeteer evidence distinguishes a real product gap from stale report text or an audit-script artifact.

**Tests:** `node /private/tmp/character-chat-reaudit.mjs`

**Status:** Complete

Notes:

- Initial post-P1 refresh showed direct `/characters` and explicit character-chat onboarding were fixed.
- The row action still navigated to `/` before the code fix because `activeQuickChatModel` fell back to the first model catalog entry even when `selectedModel` was unset.
- The audit script was tightened to close the edit drawer before row-chat interaction so the final evidence reflects the actual row action.

## Stage 2: Add Regression Coverage

**Goal:** Pin the implicit catalog fallback case in the Characters manager tests.

**Success Criteria:** A focused test fails before the fix when `selectedModel` is null but `getModelsForFieldGeneration` returns a model.

**Tests:** `bunx vitest run src/components/Option/Characters/__tests__/Manager.first-use.test.tsx -t "implicit row chat selection" --testTimeout=30000`

**Status:** Complete

Notes:

- The new regression verifies that row `Chat as...` keeps the selected character local and shows `Choose a chat model before chatting as ...` instead of navigating.

## Stage 3: Scope Full-Chat Readiness To Explicit Selection

**Goal:** Keep the quick-chat popup fallback separate from full chat row navigation.

**Success Criteria:** `useCharacterCrud` receives `selectedChatModel` for full-chat readiness while `useCharacterQuickChat` can keep using the popup fallback model.

**Tests:** Focused regression plus existing row no-model and stale-model tests.

**Status:** Complete

Notes:

- The fix changes the `activeChatModel` passed to `useCharacterCrud` from `activeQuickChatModel` to `selectedChatModel`.

## Stage 4: Refresh Browser Evidence And Report

**Goal:** Re-run the Puppeteer walkthrough and update the re-audit report to reflect the fixed build.

**Success Criteria:** `09-returning-user-row-chat-action.png` stays on `/characters` and shows the selected-character model blocker.

**Tests:** Puppeteer re-audit, `jq empty`, pinned UI typecheck, and `git diff --check`.

**Status:** Complete

Notes:

- Final evidence uses `Reaudit Character 20260509184619`.
- Remaining blockers are search count semantics, no-provider message generation coverage, connected setup copy priority, and console/request noise.
