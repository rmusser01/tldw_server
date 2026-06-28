# Skills Seed Overwrite Confirmation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a confirmation step before the Skills manager overwrites existing built-in skills.

**Architecture:** This is a frontend-only safety affordance in the existing Skills manager. The overwrite menu item will open Ant Design `Modal.confirm`; the existing React Query seed mutation remains responsible for API calls, success UI, cache invalidation, and error handling.

**Tech Stack:** React 18, TypeScript, Ant Design `Modal.confirm`, TanStack Query, Vitest, Testing Library.

**Spec:** `Docs/superpowers/specs/2026-06-28-skills-seed-overwrite-confirmation-design.md`

---

## File Structure

- Modify `apps/packages/ui/src/components/Option/Skills/Manager.tsx`
  - Add `confirmSeedOverwrite()` near the existing mutation/action handlers.
  - Update only the `Seed and Overwrite Existing` dropdown item to use that handler.
  - Leave `Seed Missing Only` and the empty-state `Seed built-ins` button as immediate missing-only seed actions.
- Modify `apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx`
  - Keep existing missing-only coverage.
  - Replace the immediate-overwrite expectation with confirmation-first coverage.
  - Spy on `Modal.confirm` instead of relying on Ant Design static modal portal DOM.
- Update `backlog/tasks/task-530.9 - Implement-Skills-seed-overwrite-confirmation.md`
  - Record plan link, verification results, and final summary through Backlog MCP.

## Stage 1: Confirmation Tests

**Goal:** Lock the new behavior before changing the component.

**Success Criteria:** The new overwrite-confirmation tests fail against the current implementation because confirmation is missing, and the updated tests describe the confirmation contract.

**Tests:** `bunx vitest run src/components/Option/Skills/__tests__/Manager.test.tsx --reporter=dot`

**Status:** Not Started

### Task 1: Write Failing Confirmation Coverage

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx`

- [ ] **Step 1: Keep existing test imports unless the implementation requires otherwise**

The file already imports `Modal` from `antd` and uses Testing Library utilities. Follow existing repo patterns for `Modal.confirm` tests; do not add `act` unless Vitest reports a React state-update warning that is fixed by wrapping the handler call.

- [ ] **Step 2: Keep missing-only behavior unchanged**

Do not weaken the existing test named `seeds built-in skills via seedSkills action`. It must still click `Seed Missing Only` and assert:

```ts
expect(tldwClientMock.seedSkills).toHaveBeenCalledWith({ overwrite: false })
```

- [ ] **Step 3: Replace the immediate-overwrite test**

Replace `seeds built-in skills with overwrite via seedSkills action` with confirmation-first coverage:

```ts
it("opens a destructive confirmation before seeding built-in skills with overwrite", async () => {
  const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementation(
    () =>
      ({
        destroy: vi.fn(),
        update: vi.fn()
      }) as any
  )

  renderManager()

  await waitFor(() => {
    expect(tldwClientMock.listSkills).toHaveBeenCalled()
  })

  fireEvent.click(screen.getByRole("button", { name: "Seed Built-ins" }))
  fireEvent.click(await screen.findByText("Seed and Overwrite Existing"))

  expect(tldwClientMock.seedSkills).not.toHaveBeenCalled()
  expect(confirmSpy).toHaveBeenCalledTimes(1)

  const confirmConfig = confirmSpy.mock.calls[0][0]
  expect(confirmConfig.title).toBe("Overwrite existing built-in skills?")
  expect(confirmConfig.content).toBe(
    "This replaces existing skill copies that match built-in skill names. Custom skills with other names are not changed."
  )
  expect(confirmConfig.okText).toBe("Overwrite built-ins")
  expect(confirmConfig.cancelText).toBe("Cancel")
  expect(confirmConfig.okButtonProps).toMatchObject({ danger: true })
})
```

- [ ] **Step 4: Add cancel-path coverage**

Use the captured confirm config. Because the implementation does not need a custom `onCancel`, this unit test represents cancellation by opening the confirmation and not invoking `onOk`; that path must not call the seed endpoint:

```ts
it("does not seed built-in skills when overwrite confirmation is cancelled", async () => {
  const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementation(
    () =>
      ({
        destroy: vi.fn(),
        update: vi.fn()
      }) as any
  )

  renderManager()

  await waitFor(() => {
    expect(tldwClientMock.listSkills).toHaveBeenCalled()
  })

  fireEvent.click(screen.getByRole("button", { name: "Seed Built-ins" }))
  fireEvent.click(await screen.findByText("Seed and Overwrite Existing"))

  const confirmConfig = confirmSpy.mock.calls[0][0]
  expect(confirmConfig.onOk).toEqual(expect.any(Function))

  expect(tldwClientMock.seedSkills).not.toHaveBeenCalled()
})
```

- [ ] **Step 5: Add confirm-path coverage**

Invoke `onOk` and assert one overwrite call:

```ts
it("seeds built-in skills with overwrite after confirmation", async () => {
  const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementation(
    () =>
      ({
        destroy: vi.fn(),
        update: vi.fn()
      }) as any
  )

  renderManager()

  await waitFor(() => {
    expect(tldwClientMock.listSkills).toHaveBeenCalled()
  })

  fireEvent.click(screen.getByRole("button", { name: "Seed Built-ins" }))
  fireEvent.click(await screen.findByText("Seed and Overwrite Existing"))

  const confirmConfig = confirmSpy.mock.calls[0][0]
  await confirmConfig.onOk?.()

  await waitFor(() => {
    expect(tldwClientMock.seedSkills).toHaveBeenCalledTimes(1)
  })
  expect(tldwClientMock.seedSkills).toHaveBeenCalledWith({ overwrite: true })
})
```

- [ ] **Step 6: Run the focused test and verify failure**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Skills/__tests__/Manager.test.tsx --reporter=dot
```

Expected before implementation: FAIL because `Modal.confirm` is not called and overwrite still calls `seedSkills({ overwrite: true })` immediately.

## Stage 2: Component Implementation

**Goal:** Add the confirmation without changing seed endpoint semantics.

**Success Criteria:** Overwrite opens a destructive confirmation; confirming calls the existing mutation exactly once; missing-only seed paths are unchanged.

**Tests:** Same focused Manager Vitest file.

**Status:** Not Started

### Task 2: Add Confirm Handler And Wire Dropdown

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Skills/Manager.tsx`

- [ ] **Step 1: Add `confirmSeedOverwrite()` after `seedBuiltinsMutation`**

Add the handler near the existing action handlers:

```ts
const confirmSeedOverwrite = () => {
  Modal.confirm({
    title: t("option:skills.seedOverwriteConfirmTitle", {
      defaultValue: "Overwrite existing built-in skills?"
    }),
    content: t("option:skills.seedOverwriteConfirmContent", {
      defaultValue:
        "This replaces existing skill copies that match built-in skill names. Custom skills with other names are not changed."
    }),
    okText: t("option:skills.seedOverwriteConfirmOk", {
      defaultValue: "Overwrite built-ins"
    }),
    okButtonProps: { danger: true },
    cancelText: t("common:cancel", { defaultValue: "Cancel" }),
    onOk: () => seedBuiltinsMutation.mutateAsync(true)
  })
}
```

- [ ] **Step 2: Wire only the overwrite dropdown item**

Change only this menu item:

```ts
{
  key: "seed-overwrite-existing",
  label: t("option:skills.seedBuiltinsOverwrite", {
    defaultValue: "Seed and Overwrite Existing"
  }),
  onClick: confirmSeedOverwrite
}
```

- [ ] **Step 3: Confirm missing-only paths were not changed**

Leave both of these paths as-is:

```ts
onClick: () => seedBuiltinsMutation.mutate(false)
```

This applies to the dropdown `Seed Missing Only` item and the empty-state `Seed built-ins` button.

- [ ] **Step 4: Run focused test and verify pass**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Skills/__tests__/Manager.test.tsx --reporter=dot
```

Expected after implementation: PASS.

## Stage 3: Verification And Finalization

**Goal:** Verify the touched scope, update tracking, and commit a clean implementation.

**Success Criteria:** Focused tests pass, frontend-only Bandit skip is recorded, task metadata is current, and the worktree is clean.

**Tests:** Focused Manager Vitest file.

**Status:** Not Started

### Task 3: Verify, Update Backlog, Commit

**Files:**
- Modify through Backlog MCP: `backlog/tasks/task-530.9 - Implement-Skills-seed-overwrite-confirmation.md`
- Commit touched files.

- [ ] **Step 1: Run focused verification**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Skills/__tests__/Manager.test.tsx --reporter=dot
```

Expected: PASS.

- [ ] **Step 2: Record Bandit skip**

Do not run Bandit if the implementation remains frontend-only. Record: `Bandit skipped; touched scope is TypeScript/React only.`

- [ ] **Step 3: Review diff**

Run from repo root:

```bash
git diff -- apps/packages/ui/src/components/Option/Skills/Manager.tsx apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx Docs/superpowers/plans/2026-06-28-skills-seed-overwrite-confirmation.md
```

Expected: only planned files changed.

- [ ] **Step 4: Update task tracking**

Use Backlog MCP to add:

- Plan link: `Docs/superpowers/plans/2026-06-28-skills-seed-overwrite-confirmation.md`
- Verification result for the focused Vitest command.
- Bandit skip note for frontend-only scope.
- Final summary of behavior change.

- [ ] **Step 5: Commit implementation**

Run from repo root:

```bash
git add Docs/superpowers/plans/2026-06-28-skills-seed-overwrite-confirmation.md apps/packages/ui/src/components/Option/Skills/Manager.tsx apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx "backlog/tasks/task-530.9 - Implement-Skills-seed-overwrite-confirmation.md"
git commit -m "TASK-530.9 add skills seed overwrite confirmation"
```

Expected: commit succeeds without unrelated files.

- [ ] **Step 6: Confirm clean worktree**

Run:

```bash
git status --short --branch
```

Expected: branch ahead of `origin/dev`; no unstaged or staged changes.
