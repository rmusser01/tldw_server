# First-Value Starter Questions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement PR3 of the solo onboarding V2 roadmap: after first-source readiness is confirmed, show safe starter questions and route the selected question through the existing grounded media chat handoff.

**Architecture:** Extend the existing onboarding UAT harness first, then add small frontend-only product wiring. Keep source readiness authoritative through the current first-source quick-ingest session summary plus post-onboarding media readiness; do not infer readiness from unrelated ingest runs. Reuse `tldw:discuss-media` and `MediaChatHandoffPayload.content` for the selected starter question instead of creating a new chat handoff path.

**Tech Stack:** React, TypeScript, Vitest, Playwright UAT harness, Backlog.md.

---

## Source Documents

- Roadmap/backlog context: `TASK-514`, `TASK-592`
- Current first-source prompt: `apps/packages/ui/src/components/Option/Onboarding/FirstSourceMilestonePrompt.tsx`
- Current first-source route gate: `apps/packages/ui/src/routes/option-index.tsx`
- Existing media chat handoff: `apps/packages/ui/src/services/tldw/media-chat-handoff.ts`
- Current UAT first-source spec: `apps/tldw-frontend/e2e/onboarding-uat/first-source.spec.ts`
- Current UAT helpers: `apps/tldw-frontend/e2e/onboarding-uat/helpers.ts`

## Scope

This PR adds first-value starter questions only. It must not:

- Generate questions with an LLM.
- Show starter questions before the first-source run has a media id and media readiness is ready.
- Show starter questions for unrelated quick-ingest runs.
- Add a second onboarding state system.
- Change first-chat completion semantics.
- Claim grounded chat readiness if backend/media readiness cannot be confirmed.

Starter questions are fixed templates:

- `Summarize this source.`
- `List the key claims.`
- `What should I remember?`

## File Map

- Modify `apps/tldw-frontend/e2e/onboarding-uat/helpers.ts`
  - Add helper(s) for completing the first-source paste flow and clicking a starter question.
- Modify `apps/tldw-frontend/e2e/onboarding-uat/first-source.spec.ts`
  - Extend the first-source scenario to assert starter questions only after source readiness, then click one and record the chat handoff detail.
- Modify `apps/packages/ui/src/components/Option/Onboarding/FirstSourceMilestonePrompt.tsx`
  - Add starter question rendering for ready state.
- Modify `apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx`
  - Cover hidden/not-ready states and ready starter selection.
- Modify `apps/packages/ui/src/routes/option-index.tsx`
  - Define starter templates, pass them only when ready, and dispatch the selected template through `tldw:discuss-media` as `content`.
- Modify `apps/packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx`
  - Cover no starters before readiness, starters after persisted ready first-source session, and dispatch payload content.
- Modify `backlog/tasks/task-592 - Implement-first-value-starter-questions-after-onboarding.md`
  - Track plan, touched files, verification, and final summary.

## Commit Plan

Ship as three small commits stacked after the current diagnostics/recovery branch:

1. `feat(onboarding): offer first-source starter questions`
2. `test(onboarding): extend UAT for first-source starter questions`
3. `docs(onboarding): close first-value starter question plan`

If the UAT extension needs product selectors to exist before a meaningful red run, add the failing Vitest coverage first, then add the Playwright assertions in the same first commit after the product behavior is green. Keep the final UAT run as verification evidence.

---

## Task 0: Backlog And Baseline

**Files:**
- Modify: `backlog/tasks/task-592 - Implement-first-value-starter-questions-after-onboarding.md`

- [x] **Step 1: Confirm worktree and branch**

Run:

```bash
git branch --show-current
git status --short
```

Expected: branch is `codex/onboarding-diagnostics-recovery-clean` or an intentional PR3 branch/worktree. Any unrelated dirty files are noted and left untouched.

- [x] **Step 2: Confirm focused baseline**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx ../packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx --reporter=dot
```

Expected: current tests pass before adding starter-question expectations.

- [x] **Step 3: Record baseline in Backlog**

Update `TASK-592` implementation notes with branch, baseline command, and result.

---

## Task 1: Add RED Coverage For Starter Questions

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx`
- Modify: `apps/packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx`

- [x] **Step 1: Add prompt-level failing tests**

Add tests asserting:

- Ready state renders a `Starter questions` group when `starterQuestions` and `onAskStarterQuestion` are provided.
- Clicking `Summarize this source.` calls `onAskStarterQuestion("Summarize this source.")`.
- Idle, processing, and error states do not render starter question buttons.

- [x] **Step 2: Add route-level failing tests**

Extend `option-index.unified-setup.test.tsx`:

- Existing ready first-source session should show the three templates.
- Clicking `Summarize this source.` dispatches one `tldw:discuss-media` event with:

```ts
{
  mediaId: "persisted-42",
  title: "Saved PDF",
  mode: "rag_media",
  content: "Summarize this source."
}
```

- A completed first-run state without first-source media readiness still does not show any starter question buttons.
- An unrelated quick-ingest success still does not show starter question buttons.

- [x] **Step 3: Run RED**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx ../packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx --reporter=dot
```

Expected: FAIL because `FirstSourceMilestonePrompt` has no `starterQuestions`/`onAskStarterQuestion` props and `OptionIndex` still dispatches only the generic ask action.

---

## Task 2: Implement Starter Question UI And Handoff

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Onboarding/FirstSourceMilestonePrompt.tsx`
- Modify: `apps/packages/ui/src/routes/option-index.tsx`

- [x] **Step 1: Add prompt props**

Add:

```ts
starterQuestions?: string[]
onAskStarterQuestion?: (question: string) => void
```

Keep `onAskAboutSource` optional for backward compatibility, but prefer starter questions when supplied.

- [x] **Step 2: Render fixed-size starter buttons only in ready state**

In ready state, if `starterQuestions.length > 0` and `onAskStarterQuestion` exists:

- Render a compact `Starter questions` label.
- Render up to three buttons.
- Use stable button dimensions and wrapping so long labels do not resize the shell badly.
- Do not render the idle picker, Add Source button, or Retry button in ready state.

- [x] **Step 3: Wire OptionIndex starter templates**

Add a local constant:

```ts
const FIRST_SOURCE_STARTER_QUESTIONS = [
  "Summarize this source.",
  "List the key claims.",
  "What should I remember?"
] as const
```

Change `discussFirstSource` to accept `question?: string` and include `content: question` only when selected.

Pass `starterQuestions={firstSourceAskReady ? [...FIRST_SOURCE_STARTER_QUESTIONS] : []}` and `onAskStarterQuestion` only when `firstSourceMediaId && firstSourceAskReady`.

- [x] **Step 4: Run GREEN**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx ../packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx --reporter=dot
```

Expected: PASS.

- [x] **Step 5: Commit product tests and implementation**

Run:

```bash
git add apps/packages/ui/src/components/Option/Onboarding/FirstSourceMilestonePrompt.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx apps/packages/ui/src/routes/option-index.tsx apps/packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx "backlog/tasks/task-592 - Implement-first-value-starter-questions-after-onboarding.md"
git diff --cached --check
git commit -m "feat(onboarding): offer first-source starter questions"
```

---

## Task 3: Extend UAT Harness For First-Value Assertion

**Files:**
- Modify: `apps/tldw-frontend/e2e/onboarding-uat/helpers.ts`
- Modify: `apps/tldw-frontend/e2e/onboarding-uat/first-source.spec.ts`
- Modify: `apps/tldw-frontend/e2e/onboarding-uat/scenarios.ts` only if a separate scenario id is cleaner than extending `first-source-after-chat`.

- [x] **Step 1: Add helper for paste source processing**

Add a helper that:

- Assumes the spec has selected Paste in the milestone and opened Quick Ingest.
- Fills the `Pasted text input` with the structured note fixture text.
- Clicks `Use defaults & process`.
- Waits for Quick Ingest completion/results.
- Returns the first-source session summary from `__tldw_useQuickIngestSessionStore.getState().session`.

Use real UI controls and backend APIs. Do not route-mock ingest/provider behavior.

- [x] **Step 2: Add helper for starter handoff capture**

Add a helper that installs a page-side listener for `tldw:discuss-media`, clicks a starter question button, and returns the captured detail.

Expected detail includes `mediaId`, `title`, `mode: "rag_media"`, and `content` equal to the selected starter question.

- [x] **Step 3: Extend first-source UAT spec**

After first-source processing reaches success and the milestone returns to ready state:

- Assert all three starter questions are visible.
- Click `Summarize this source.`
- Assert captured handoff `content` is `Summarize this source.`
- Capture a screenshot and JSON step with the selected starter and handoff detail.

- [x] **Step 4: Run UAT RED/GREEN check**

Run from `apps/tldw-frontend`:

```bash
bun run e2e:onboarding:uat -- --scenario first-source-after-chat --viewport desktop --mock-config hosted-success.json
```

Expected after implementation: PASS with screenshots, JSON summary, backend/frontend/mock logs, and no critical diagnostics.

- [x] **Step 5: Commit UAT extension**

Run:

```bash
git add apps/tldw-frontend/e2e/onboarding-uat/helpers.ts apps/tldw-frontend/e2e/onboarding-uat/first-source.spec.ts apps/tldw-frontend/e2e/onboarding-uat/scenarios.ts "backlog/tasks/task-592 - Implement-first-value-starter-questions-after-onboarding.md"
git diff --cached --check
git commit -m "test(onboarding): extend UAT for first-source starter questions"
```

---

## Task 4: Final Verification And Closeout

**Files:**
- Modify: `backlog/tasks/task-592 - Implement-first-value-starter-questions-after-onboarding.md`
- Optional modify: `Docs/superpowers/plans/2026-06-02-first-value-starter-questions-implementation-plan.md` only for implementation discoveries.

- [x] **Step 1: Run focused frontend suite**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx ../packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx ../packages/ui/src/utils/__tests__/quick-ingest-open.test.ts ../packages/ui/src/hooks/__tests__/usePostOnboardingMediaReadiness.test.tsx --reporter=dot
```

Expected: PASS.

- [x] **Step 2: Run focused UAT scenario**

Run from `apps/tldw-frontend`:

```bash
bun run e2e:onboarding:uat -- --scenario first-source-after-chat --viewport desktop --mock-config hosted-success.json
```

Expected: PASS. Record the preserved `summary.json` path in Backlog.

- [x] **Step 3: Security gate**

If only frontend TypeScript/E2E files changed, record Bandit skip as non-Python/frontend-only. If backend Python changed, run Bandit on touched backend production files.

- [x] **Step 4: Whitespace and status checks**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended PR3 files are modified.

- [x] **Step 5: Finalize Backlog**

Update `TASK-592`:

- Implementation notes with commands and results.
- Final summary.
- Definition of Done checked.

- [x] **Step 6: Commit closeout**

Run:

```bash
git add Docs/superpowers/plans/2026-06-02-first-value-starter-questions-implementation-plan.md "backlog/tasks/task-592 - Implement-first-value-starter-questions-after-onboarding.md"
git diff --cached --check
git commit -m "docs(onboarding): close first-value starter question plan"
```

## Risks And Review Notes

## Verification Evidence

- Baseline focused suite before RED: `bunx vitest run ../packages/ui/src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx ../packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx --reporter=dot` passed, 19 tests.
- RED focused suite after adding starter expectations failed as expected because starter-question UI/handoff were missing.
- GREEN focused suite after implementation passed, 20 tests.
- Final focused suite: `bunx vitest run ../packages/ui/src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx ../packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx ../packages/ui/src/utils/__tests__/quick-ingest-open.test.ts ../packages/ui/src/hooks/__tests__/usePostOnboardingMediaReadiness.test.tsx --reporter=dot` passed, 4 files and 28 tests.
- Final UAT: `bun run e2e:onboarding:uat -- --scenario first-source-after-chat --viewport desktop --mock-config hosted-success.json` passed. Summary: `apps/tldw-frontend/test-results/onboarding-uat/2026-06-02T21-07-43-211Z-c6kmj4/summary.json`.
- UAT ready-state screenshot verified the Quick Ingest modal is closed and the three starter buttons are visible. Handoff JSON recorded `mediaId: "1"`, `mode: "rag_media"`, and `content: "Summarize this source."`.
- Bandit: skipped because this PR3 slice touched frontend TypeScript, Playwright UAT, Backlog, and plan files only; no backend Python files changed.
- Whitespace: `git diff --check` passed.

- If Quick Ingest processing in UAT cannot reliably produce a media id from paste text, stop after three attempts and document the blocker. Do not fake readiness with Playwright route mocks.
- If `tldw:discuss-media` cannot carry starter text cleanly through `content`, add a narrow typed field to `MediaChatHandoffPayload` and update `buildDiscussMediaHint` tests before changing route code.
- If the current first-source prompt leaves the user on the Companion Home while Quick Ingest runs, the UAT helper should wait on the quick-ingest session store plus the prompt ready state instead of brittle modal text alone.
