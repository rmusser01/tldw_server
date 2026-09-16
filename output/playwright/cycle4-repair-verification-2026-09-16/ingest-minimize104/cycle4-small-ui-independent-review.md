# Independent review — cycle4 Task8 small UI units

Read-only review of supplied `/private/tmp/uat104-review.diff`, `uat112-review.diff`, `uat102-review.diff` and their implementation reports, against `Docs/Design/2026-09-16-uat-cycle-4-repairs.md`. No production/test edits, commit/staging, native browser/runtime/inference, or subagents. Private Vite transform described below affects only the review process.

## UAT104 / TASK13260.45 — REVIEW CLEAR

No actionable defect found in the supplied patch.

`ProcessingStep.tsx:386` performs the existing context `minimize()` followed by the optional owner callback; `QuickIngestWizardModal.tsx:1731` supplies its existing `onClose`. This exactly matches the pre-existing close-confirmation minimize path at lines1575–1578. The actual owning QuickIngestButton closes via `hideSession` and resets transient preparation references, without canceling, deleting, replacing or restarting the session. Existing session authority/identity guard stays in place. No new asynchronous operation or stale-owner write is added.

The new test mounts the actual ProcessingStep, real wizard context and session-backed modal. It checks dialog dismissal, retained session id/tracking/running per-item progress, no cancel/start call, and normal same-session resume; the close-confirmation control is meaningful. Timer values are correctly excluded from strict snapshot identity. Existing standalone use remains compatible because the callback is optional.

Independent verification: entire session test suite, 45 tests passed. Retained implementation evidence also reports integration/floating widget/resume suites; these were inspected, not redundantly rerun.

Native limit: actual browser modal disappearance, focus return and continued real-job progress/resume still require targeted native verification. This is not a live acceptance pass.

## UAT112 / TASK13260.52 and stale UAT094 snapshot TASK13260.35 — CHANGES REQUESTED

### P3: decorative loading image remains exposed after pending ends

Locations:
- `apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx:427` and `:441` (new accessibility attributes at429–432 and443–444).
- `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx:1244` (new attributes1247–1250).

The explicit translated `aria-label` repairs each button's name and `aria-busy` reports real state. However, the AntD default loading icon still renders an exposed `role="img" aria-label="loading"` child during its leave animation after pending becomes false. The binding design expressly requires completed decorative loading indicators to be hidden from the accessibility tree. Naming the parent does not apply aria-hidden to that descendant.

Private behavioral probe extends the existing actual-AntD pending→false tests with:

```ts
if (!pending) {
  expect(button).toBeEnabled()
  expect(within(button).queryByRole('img', { name: 'loading' })).not.toBeInTheDocument()
}
```

The analogous Review assertion uses `!loading`. Both fail while existing name/busy/enabled assertions pass. Evidence:
- `/private/tmp/cycle4-small-ui-loading-review.config.ts` (read-only transform, no repo source changes)
- `/private/tmp/cycle4-small-ui-loading-review.log`: **2 failed**, 26 intentionally deselected cases,4.51s.
- AntD6.2.1 `button/DefaultLoadingIcon.js` wraps LoadingOutlined without aria-hidden; `@ant-design/icons` AntdIcon assigns role img/name loading.

Minimal repair: mark the local decorative loading icon/container aria-hidden using the supported button icon boundary, while retaining stable translated label, real busy state, pending disabled/click guard and original action handlers. Extend permanent tests to assert the absence of exposed loading descendants after completion, not only the button's computed accessible name. No stuck-mutation claim is made; this is an accessibility/transition-state gap.

Other aspects reviewed clear: floating Create card name is stable and translated; Create/Add Another and all-due handlers, validation, mutation/query state and draft reset behavior are unchanged. The snapshot delta removes only the stale automatic Study assistant tooltip subtree and matches the separately tracked094 repair; no new runtime tooltip removal is hidden in this unit. Actual ReviewTab suite including snapshot passed25tests.

Native limit: this probe establishes DOM/ARIA semantics through Testing Library, not an actual macOS/Chrome accessibility snapshot. Native keyboard/AX confirmation remains required after the correction.

## UAT102 / TASK13260.43 — REVIEW CLEAR

No actionable defect found in the supplied Admin/Prompt feedback patch.

ServerAdminPage uses existing `useAntdMessage`; usePromptSync uses existing `useAntdNotification`. Both hooks obtain the mounted AntD App APIs with existing compatibility fallbacks. Actual web AppProviders places route content inside AntdApp under ConfigProvider, so the production path has the required context. Notification compatibility translates legacy message/title fields and preserves descriptions. The changed callback dependency arrays include the contextual notification instance. No create, sync, query invalidation, local-save, mutation, owner-capture, conflict or draft logic changed.

The Admin test mounts actual App, verifies exact create payload plus user-list refresh, visible contextual success and zero static-success calls. Prompt test uses real App/ConfigProvider, controlled sync failure, verifies the original failed result and visible locally-saved warning with no static-warning call. Disabling motion only in the test wrapper is a reasonable deterministic jsdom harness choice. Existing owner/conflict tests retain stable original spies because the production compatibility utility wraps notification methods; they still assert real hook outcomes and captured owner boundaries. This is not a mock-only assertion of hook invocation.

Independent verification: Admin design-system12tests and Prompt owner-callers10tests passed (included in99test aggregate below).

Native limit: confirm absence of the original static-context console warning through real Admin creation and Prompt failure paths. Review does not certify every other static AntD caller.

## Independent test command/results

From `apps/packages/ui`:

```sh
bun run test \
  src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx \
  src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.tags.test.tsx \
  src/components/Flashcards/tabs/__tests__/ManageTab.empty-state.test.tsx \
  src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx \
  src/components/Option/Admin/__tests__/ServerAdminPage.design-system.test.tsx \
  src/components/Option/Prompt/__tests__/prompt-sync.owner-callers.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

**6 files,99 tests passed,41.75s**. Log `/private/tmp/cycle4-small-ui-independent-tests.log`. Existing Node localStorage warnings and a controlled Failed-to-fetch log remain; no clean-console claim.

Private additional probe:

```sh
bun run test --config /private/tmp/cycle4-small-ui-loading-review.config.ts
```

The probe selected only the two relevant transition tests and found the UAT112 gap above. No broad suites, new native runs or unrelated agent changes reviewed. Scoped `git diff --check` passes. Supplied104/112 ESLint comparison artifacts report unchanged baseline signatures; no TS-only Bandit claim.
