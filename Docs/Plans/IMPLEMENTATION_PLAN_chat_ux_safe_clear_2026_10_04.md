# Chat UX safe clear implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Release the active history owner when an accepted shared new-chat/clear action starts a fresh view, including actions outside the mounted provider, while rejecting late callbacks and preserving other compact logical views.

**Architecture:** Reuse `HistorySelectionController.reset()` and the existing owner-qualified `CHAT_ROUTE_REPLACEMENT_EVENT`. The clear hook handles its own ancestor controller; the mounted Playground handles qualified clear events from controls outside that provider. Character replacement and branch events keep their current semantics.

**Tech Stack:** React 18, Zustand, existing selected-history controller, Vitest 4, React Testing Library; no new dependencies.

**Spec:** [Approved chat UX design](../Design/2026-10-04-chat-conversation-ux-remediation-design.md), conversation transitions and pre-implementation review checkpoint.

**Task:** TASK-13454, first delivery within program Stage 1. The task remains In Progress after this delivery: save-status/recovery, explicit modes, full setup preflight, draft preservation and greeting-only startup are separate remaining acceptance criteria. Programs TASK-13455–TASK-13458 retain the complete approved scope.

**Baseline:** `origin/dev` at `75ab224081bf140ef52017c1a9b0a04f6878d488`, refreshed 2026-10-04. Start HEAD `a010319566`. Overlapping PRs #3147, #3152, #3174 and #3178 are open; do not import them.

## Global Constraints

- Preserve owner/account/workspace fences and explicit legacy-history review.
- Scope reset to the active view; compact tabs retain their independently owned snapshots.
- Cancel or a failed preflight preserves source owner, draft, and settings.
- Temporary content is not written to history as a side effect of switching modes.
- Do not redirect an in-flight turn's completion to a newly selected view.
- Use existing dependencies, locale source files and mirror tooling, shared components, and current test harnesses.
- Use canonical backlog-py task edits. Do not change the dirty primary checkout.
- This delivery does not change copy, storage authority, public APIs, or the character/greeting workflow.

**ADR check:** ADR required: no. [ADR-049](../ADR/049-chat-history-selection-ownership.md) already governs release of selected-history ownership. [ADR-050](../ADR/050-native-chat-fork-storage-lifecycle.md) remains unchanged; branch replacement must not be treated as clear. [ADR-059](../ADR/059-backlog-py-task-editor-cutover.md) governs task edits.

---

## Stage 1: Reproduce both shared clear paths

**Goal:** Demonstrate a retained real controller owner after accepted clear, without mocking the controller or clear hook.

**Success Criteria:** Inside-provider and outside-provider cases fail because the old owner/fence remains live; cancelled navigation preserves both.

**Tests:** Extend the existing coordinator integration fixture, whose real H1 controller and mocked external API already load owned native transcripts.

**Status:** Complete.

**Files:**

- Modify/test: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx`.
- Read: `apps/packages/ui/src/hooks/chat/useClearChat.ts`, `useHistorySelection.ts`, `apps/packages/ui/src/components/Option/Playground/Playground.tsx`.

**Interfaces:**

- Consume existing `useClearChat(): () => boolean`, `useHistorySelectionContext(): HistorySelectionController | null`, `controller.fence(): () => boolean`, `controller.getCurrent()` and existing `realLoader`/API fixtures.
- Produce tests for visible owner clearing and stale-fence invalidation, using only test-local probe components.

- [x] Add a `selectionProbe` fixture flag and reset it before each test. Render the following probe inside the existing mocked PlaygroundForm only when that flag is enabled. Render an additional `SharedClear` outside Playground for the outside-provider case.

```tsx
let observedSelection: HistorySelectionController | null = null
function SharedClear({ label }: { label: string }) {
  const clear = useClearChat()
  return <button onClick={() => clear()}>{label}</button>
}
function HistoryOwnerProbe() {
  const selection = useHistorySelectionContext()!
  observedSelection = selection
  return <>
    <output aria-label="Selected history owner">
      {selection.owner?.kind === "unavailable" ? "unavailable" : selection.owner?.conversation_id ?? "none"}
    </output>
    <SharedClear label="Clear inside view" />
  </>
}
```

- [x] Add the parameterized test below. Use existing native fixtures and literal expected values; do not infer expected owner state using production helpers.

```tsx
it.each(["inside", "outside"] as const)("releases the selected owner on shared clear %s the provider", async origin => {
  realLoader.enabled = true
  realLoader.selectionProbe = true
  useMessageOptionMock.mockImplementation(useRealServerConversation)
  useStoreMessageOption.setState({ serverChatId: "robot", historyId: null,
    messages: [], history: [], serverChatMetaLoaded: false,
    serverChatCharacterId: null, serverChatAssistantKind: null,
    serverChatAssistantId: null, temporaryChat: false, streaming: false,
    isProcessing: false })
  tldwClientState.getChat.mockResolvedValue({ id: "robot", title: "Robot chat",
    character_id: 5, assistant_kind: "character", assistant_id: "5",
    source: "webui-character-chat", scope_type: "global" })
  tldwClientState.listChatMessages.mockResolvedValue([
    { id: "answer", role: "assistant", content: "BEEP BOOP", version: 1 }
  ])
  const view = render(<><Playground /><SharedClear label="Clear outside view" /></>)
  try {
    await waitFor(() => expect(observedSelection?.status).toBe("ready"))
    expect(screen.getByLabelText("Selected history owner")).toHaveTextContent("robot")
    const previousFence = observedSelection!.fence()
    fireEvent.click(screen.getByRole("button", { name: `Clear ${origin} view` }))
    await waitFor(() => expect(screen.getByLabelText("Selected history owner")).toHaveTextContent("none"))
    expect(observedSelection!.getCurrent().status).toBe("idle")
    expect(previousFence()).toBe(false)
    expect(useStoreMessageOption.getState().serverChatId).toBeNull()
  } finally { view.unmount() }
})
```

- [x] Run `bun run test -- src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx -t 'releases the selected owner on shared clear'` in `apps/packages/ui`.

Expected: both cases fail on the retained owner or active old fence. An import/setup error is not the required red result; repair the test fixture before proceeding.

## Stage 2: Clear the ancestor controller after navigation accepts

**Goal:** Fix every clear-hook caller mounted inside a selected-history provider.

**Success Criteria:** Inside-provider case passes; declined navigation leaves the controller/fence unchanged.

**Tests:** Parameterized coordinator case and existing `useClearChat.settings-navigation.test.tsx`.

**Status:** Complete.

**Files:**

- Modify: `apps/packages/ui/src/hooks/chat/useClearChat.ts`.
- Test: coordinator integration fixture above and `apps/packages/ui/src/hooks/chat/__tests__/useClearChat.settings-navigation.test.tsx`.

**Interfaces:** Keep `useClearChat` options/return unchanged; consume existing optional history-selection context and its synchronous `reset(): void`.

- [x] Add the context import and retrieve the existing ancestor controller. Reset only after `requestSettingsNavigation(destination)` returns true, before publishing clear/navigation mutations. Include the stable reset callback in dependencies without introducing a new controller.

```ts
import { useHistorySelectionContext } from "@/hooks/chat/useHistorySelection"
// Inside useClearChat:
const resetHistorySelection = useHistorySelectionContext()?.reset
// Inside its clear callback, immediately after the navigation gate:
resetHistorySelection?.()
```

- [x] Run the inside-provider coordinator test and the entire existing clear-hook navigation suite. Expect green for inside and cancellation; outside remains red until Stage 3.
- [x] Add a cancellation assertion using the real observed controller: install `SETTINGS_NAVIGATION_REQUEST_EVENT` preventDefault, click shared clear, and assert owner `robot`, ready status, original fence true, and unchanged server ID. Remove the listener in `finally`.

## Stage 3: Handle outside-provider clear at the mounted owner

**Goal:** Fix controls such as the shared sidebar whose clear hook has no ancestor controller.

**Success Criteria:** Both clear paths release the active controller; unrelated/rejected/branch replacement events do not reset it; late capture cannot publish into the fresh view.

**Tests:** Both parameterized clear cases, a deferred capture, and existing route replacement boundary cases.

**Status:** Complete.

**Files:**

- Modify: `apps/packages/ui/src/components/Option/Playground/Playground.tsx`, existing `handleReplacement` effect.
- Test: existing coordinator integration file.

**Interfaces:** Consume qualified `ChatRouteReplacementDetail` and current captured href/history ID/server ID/restore revision. Only `characterId === undefined` means shared clear; defined character replacement and `nextChatId` branch semantics remain unchanged.

- [x] After the effect's existing owner/surface/revision checks, reset the mounted controller only in its shared-clear branch. Depend on its stable reset callback rather than the entire changing controller object.

```ts
if (detail.characterId === undefined) {
  resetHistorySelection()
  setCharacterModeIntentActive(false)
  void setChatWorkflowMode("standard")
}
```

- [x] Run both shared-clear cases and the existing `UAT068 rejects` boundary cases. Expect green; do not broaden reset to accepted character/branch events.
- [x] Add a late-capture regression: defer the external capture response after the native owner publishes loading, capture the controller fence, clear through the outside control, release the response, then assert owner none, status idle, old fence false, empty current messages and null server ID. The API double may ignore cancellation; the real controller must reject the stale result.
- [x] Verify compact isolation in the existing `useHistorySelection.test.tsx`: activate one logical view, bind it, activate another, reset that active view, and return to the first; its owned selection remains intact. Reuse the existing capture fixtures and do not reset all logical views.

## Stage 4: Verify, review and commit this delivery

**Goal:** Leave an independently reviewable fix with tested lifecycle boundaries and no accidental scope expansion.

**Success Criteria:** Relevant controller/coordinator/navigation suites pass; touched TypeScript compiles/lints with baseline failures distinguished; fresh code review has no unresolved important issue.

**Tests:** Existing selected-history and clear integration suites, targeted static checks, `git diff --check`, canonical Backlog normalization.

**Status:** In Progress.

- [x] Run the complete coordinator, `useClearChat.settings-navigation`, and `useHistorySelection` suites in one Vitest invocation with existing dependencies.
- [x] Run WebUI typecheck and the existing lint command for touched TS files. For any pre-existing failure, record the exact baseline evidence; do not disable checks or alter unrelated files.
- [x] Review the whole delivery using the two already authorized independent reviewers. Preserve controller security fences and verify no non-clear branch reset. Resolve important findings with a new red-first regression; do not dispatch additional implementers.
- [x] Record results in TASK-13454 through backlog-py. Bandit is not applicable to TS/doc-only changes; record that scope explicitly. Do not mark the entire Stage 1 task Done.
- [ ] Commit only these source/tests, this plan and the task record, with message `fix(chat): release selected history on shared new chat (TASK-13454)`.
- [ ] Keep this plan until all of its stages are complete, then remove only this plan per repository policy; the committed test/task record remains the evidence. Continue the remaining approved Stage 1 units with their own concrete plans after checking dev/overlap status.

## Self-review

The implementation uses existing `reset`, optional context, event qualification and external API fixtures. It adds no reset API or global ownership singleton. Scope intentionally covers shared clear lifecycle only; draft retention, supported greeting creation, complete readiness, saving status and replay safety remain explicit Stage 1 work. No application acceptance criterion is closed by merely preparing this plan.

## Verification record

Baseline: 127 tests passed. The initial inside/outside-owner regressions failed on the retained owner; the expanded outside clear and ignored-cancellation capture both failed before the mounted fix. Final coordinator/navigation/controller suites: 132 passed. Active-only logical-tab reset preservation passed. WebUI `bun run typecheck`: exit 0 after linking already installed Playwright types for shared test helpers. Touched-file ESLint: zero errors, no new warnings; baseline Playground 18 and controller-test 37 warnings remain. Prettier check flags all four files on both HEAD and the delivery; preserve their existing style without unrelated whole-file rewrites. `git diff --check`: clean. Independent design and evidence reviewers: no Critical/Important issues. Bandit does not apply to this TS/document-only scope. TASK-13454 stays In Progress with all broader acceptance criteria open.
