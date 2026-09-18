# TASK13260.217 / UAT276 — World Books catalogue containment

## Result

The World Books catalogue now contains its intrinsic-width table inside a named
horizontal scroll region. At the desktop split pane, the list can no longer
paint its `More actions` buttons over the detail panel. The same region is
used in the collapsed list path; selection, edit, and overflow handlers were
not changed.

## Cause and minimal repair

The desktop manager gives the list a 35% flex pane, while the Ant Design table
has a wider intrinsic layout. Without an overflow boundary, the Actions column
extended over the neighboring detail pane and its pointer events were received
by detail content. The established Prompt-list pattern is a local
`overflow-x-auto` wrapper. `WorldBookListPanel` now uses that bounded wrapper
with `min-w-0` and `max-w-full`, without changing the split ratio, table
columns, rows, callbacks, or responsive layout selection.

## Causal evidence

- The retained native baseline projection is
  `.tmp/uat-repairs-231-246/worldbooks276/baseline-geometry.json`. It records
  the 1200×953 pre-fix geometry, the pointer interception, and the keyboard
  control, with hashes back to the immutable native observations.
- The new regression first failed because no owned, bounded list scroll region
  existed: `focused-red.log` records the command, exit 1, and exact failed
  assertion.
- After the wrapper, the regression horizontally scrolls that region, opens
  `More actions`, and invokes `Manage Entries`. It passed together with the
  existing list selection/edit/action and manager desktop/mobile controls:
  **13 tests passed** in `focused-green.log`.

## Validation and limits

- Scoped ESLint: exit 0, 0 errors. It reports 22 legacy panel warnings; none
  occur in the added wrapper or test hunk.
- `git diff --check`: exit 0.
- Bandit: exit 0, 0 findings. Bandit is Python-focused and is not evidence of
  TypeScript security coverage.
- The frontend-wide typecheck currently exits 2 on unrelated existing paths;
  it has no diagnostics for either touched WorldBooks file.
- jsdom does not calculate CSS geometry or real pointer hit testing. The
  retained native pre-fix geometry establishes the failure; final acceptance
  still requires a committed native run at the affected desktop size and an
  adjacent responsive size.

## Frozen source hashes

- `WorldBookListPanel.tsx`: `082514ca18f5f38a413c12b1c04f63971ef8e0d715317d5806637b6115ca434f`
- `WorldBookListPanel.test.tsx`: `71cce836e23559fefe606d0e31d5abcd75c454365e7775ea992371c7f74f2edc`

The complete command, receipt, and hash inventory is in
`.tmp/uat-repairs-231-246/worldbooks276/verification.json`.
