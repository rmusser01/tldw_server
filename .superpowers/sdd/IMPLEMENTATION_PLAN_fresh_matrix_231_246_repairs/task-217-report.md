# TASK13260.217 / UAT276 — World Books catalogue containment

## Result

The World Books catalogue now contains its intrinsic-width table inside a
named horizontal scroll region. The same boundary is used in the collapsed
list path; selection, edit, and overflow handlers were not changed.

## Cause and minimal repair

The desktop manager gives the list a 35% flex pane, while the Ant Design table
has a wider intrinsic layout. The native baseline established that the Actions
column extended over the neighboring detail pane and its pointer events were
received by detail content. The established Prompt-list pattern is a local
`overflow-x-auto` wrapper. `WorldBookListPanel` now uses that bounded wrapper
with `min-w-0` and `max-w-full`, without changing the split ratio, table
columns, rows, callbacks, or responsive layout selection.

## Causal evidence

- The retained native baseline projection is
  `.tmp/uat-repairs-231-246/worldbooks276/baseline-geometry.json`. It records
  the 1200×953 pre-fix geometry, the pointer interception, and the keyboard
  control, with hashes back to the immutable native observations.
- Existing list selection/edit/action and manager desktop/mobile controls pass:
  **12 tests passed** in `focused-existing-controls.log`.
- The first-round `focused-red.log`, `focused-green.log`, and its test-file
  patch are retained as historical harness evidence only. JSDOM does not
  measure the relevant layout: the test assigned synthetic widths and a scroll
  position, then asserted that assignment. It is not causal pointer-layout
  validation and was removed; the test file is restored to its pre-task bytes.

## Validation and limits

- Scoped ESLint: exit 0, 0 errors. It reports 22 legacy panel warnings; none
  occur in the added wrapper hunk.
- `git diff --check`: exit 0.
- Bandit: exit 0, 0 findings. Bandit is Python-focused and is not evidence of
  TypeScript security coverage.
- jsdom does not calculate CSS geometry or real pointer hit testing. The
  retained native pre-fix geometry establishes the failure; no post-fix layout
  success is claimed here. Final acceptance requires a committed native run at
  the affected desktop size and an adjacent responsive size, with actual
  pointer hit-target checks.

## Frozen source hashes

- `WorldBookListPanel.tsx`: `082514ca18f5f38a413c12b1c04f63971ef8e0d715317d5806637b6115ca434f`

The complete command, receipt, and hash inventory is in
`.tmp/uat-repairs-231-246/worldbooks276/verification.json`.
