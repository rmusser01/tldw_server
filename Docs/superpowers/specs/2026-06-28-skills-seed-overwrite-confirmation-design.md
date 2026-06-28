# Skills Seed Overwrite Confirmation Design

## Context

`TASK-530.9` continues the `/skills` Safe Operations sequence after import review. The current Skills manager exposes a `Seed Built-ins` dropdown with two choices:

- `Seed Missing Only`, which calls `seedSkills({ overwrite: false })`
- `Seed and Overwrite Existing`, which immediately calls `seedSkills({ overwrite: true })`

The backend overwrite behavior already exists and has service/API coverage. The UX gap is that the destructive overwrite path is one menu click with no confirmation, while the page already treats delete and import overwrite as explicit safe-operation moments.

## Goal

Add a frontend confirmation before built-in skill overwrite seeding runs, without changing backend seed semantics or broader Skills workflows.

## Non-Goals

- No backend API changes.
- No preflight endpoint for counting overwritten skills.
- No version-aware delete.
- No bulk delete.
- No export feedback work.
- No permission/model/tool metadata panel work.
- No redesign of the Skills action bar or dropdown.

## UX Design

`Seed Missing Only` remains one click because it is additive and does not replace existing skills.

When the user chooses `Seed and Overwrite Existing`, the manager opens a confirmation dialog instead of firing the mutation immediately.

Dialog content:

- Title: `Overwrite existing built-in skills?`
- Body: `This replaces existing skill copies that match built-in skill names. Custom skills with other names are not changed.`
- Confirm action: `Overwrite built-ins`
- Cancel action: `Cancel`

The confirm action uses a destructive affordance. The dialog is appropriate here because this is a blocking destructive action, and the component already uses Ant Design `Modal.confirm` for delete confirmation. Avoid `Popconfirm` because dropdown-contained popovers are more fragile for focus, keyboard operation, and tests.

While the overwrite mutation is pending, the confirm action should remain in an async/loading state and must not allow duplicate submissions.

## Technical Design

Add a small handler near existing Skills manager action handlers:

- `confirmSeedOverwrite()`
- It calls `Modal.confirm(...)`
- `onOk` returns `seedBuiltinsMutation.mutateAsync(true)`

Update `seedMenuItems` so:

- `Seed Missing Only` keeps `onClick: () => seedBuiltinsMutation.mutate(false)`
- `Seed and Overwrite Existing` uses `onClick: confirmSeedOverwrite`

Keep the empty-state `Seed built-ins` button unchanged. It is another missing-only entry point and should continue to call `seedBuiltinsMutation.mutate(false)` directly.

Keep the existing `seedBuiltinsMutation` success and error handling. Do not add state unless the static Ant Design confirm path proves insufficient.

All new user-facing strings use `t(...)` with namespaced keys and `defaultValue`, matching the current component pattern.

## Accessibility And Interaction Notes

- The confirmation title should clearly identify the destructive operation.
- The copy must name the affected scope: existing skills with built-in names.
- The copy must name what is not affected: custom skills with other names.
- The confirm button must be visually destructive through `okButtonProps: { danger: true }`.
- Cancel must be available through the standard Ant Design modal controls.

## Testing

Update `apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx`.

Required coverage:

- `Seed Missing Only` still calls `seedSkills({ overwrite: false })` immediately.
- Clicking `Seed and Overwrite Existing` does not call `seedSkills` immediately.
- The confirmation dialog appears with the destructive title/copy.
- Cancelling the confirmation does not call `seedSkills`.
- Confirming the dialog calls `seedSkills({ overwrite: true })` exactly once.

Prefer spying on `Modal.confirm` for the overwrite-confirmation unit tests. Assert the confirm configuration includes the expected title, copy, cancel text, confirm text, and `okButtonProps: { danger: true }`, then invoke the captured `onCancel` and `onOk` handlers directly. Avoid relying on Ant Design's static modal portal DOM unless a role-based assertion is stable in the local test harness.

Focused verification:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Skills/__tests__/Manager.test.tsx --reporter=dot
```

No Bandit run is required if the implementation remains frontend-only. Record that as a non-Python scope skip in the task final summary.

## Risks And Mitigations

- **Risk:** Static `Modal.confirm` can leave modal state between tests.
  **Mitigation:** The existing test file already imports `Modal` and calls `Modal.destroyAll()` in cleanup.

- **Risk:** Async confirmation could allow duplicate overwrite submissions.
  **Mitigation:** Return `seedBuiltinsMutation.mutateAsync(true)` from `onOk` so Ant Design owns confirm loading behavior.

- **Risk:** Confirmation copy becomes vague and users cannot tell what will be overwritten.
  **Mitigation:** Copy names both affected and unaffected skill groups.

## Acceptance Criteria Mapping

- AC1: Dropdown overwrite item opens `Modal.confirm`.
- AC2: Cancel path leaves `seedSkills` untouched.
- AC3: Confirm path calls `seedSkills({ overwrite: true })` once and uses `danger`.
- AC4: Missing-only item keeps existing immediate mutation.
- AC5: Focused Manager Vitest coverage validates all of the above.
