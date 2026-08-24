# Task 4 Report: Output Profile Editor

## Scope

- Created `OutputProfileEditor.tsx` and focused behavior tests.
- Added English locale entries under `outputProfileEditor`.
- Preserved non-profile settings through `outputProfilesToSettings()`.

## TDD Evidence

### RED

Command:

```bash
bunx vitest run src/components/Option/Settings/__tests__/OutputProfileEditor.test.tsx
```

Result: expected failure before implementation. Vite could not resolve
`../OutputProfileEditor`; the suite reported zero tests because the module did
not exist.

### GREEN

Command:

```bash
bunx vitest run src/components/Option/Settings/__tests__/OutputProfileEditor.test.tsx
```

Result: passed, 1 test file and 8 tests. Coverage includes ordered section
editing, response format, branch inclusion, profile creation/deletion,
validation, and save-failure draft retention.

## Verification

- `bun -e 'JSON.parse(await Bun.file("src/assets/locale/en/settings.json").text())'`: passed.
- `git diff --check`: passed.
- `NODE_OPTIONS=--max-old-space-size=6144 bunx tsc --noEmit --pretty false`:
  exited 2 on existing package-wide diagnostics in unrelated UI tests and
  services. The captured diagnostics did not name `OutputProfileEditor.tsx`,
  its focused test, or `settings.json`. The default-heap attempt exhausted
  Node memory before producing diagnostics.
- Bandit skipped: this task changes TypeScript and JSON only.

## Accessibility And Responsive Review

- Labels are associated with the profile selector, profile-name input,
  section fields, and branch-output checkbox.
- Icon-only controls have both accessible names and Ant Design tooltips.
- Save status uses a polite live region; validation and save failures use an
  alert role while retaining drafts.
- Controls use semantic color tokens, visible focus rings, fixed icon-button
  dimensions, and minimum row heights. The profile controls stack at medium
  widths and section rows collapse to one column below the small breakpoint.

## Task 5 Boundary

`OutputProfileEditor` accepts already-loaded `settings` and performs no hidden
fetch. The existing prop contract cannot represent settings-load failure or a
retry action, so parent loading, failure, and retry integration coverage remain
for Task 5.

## Fix Round 1

### RED

Command:

```bash
bunx vitest run src/components/Option/Settings/__tests__/OutputProfileEditor.test.tsx
```

Result: 2 expected failures in 10 tests. Renaming `summary` to `overview`
saved an empty `section_titles` mapping, and the profile selector was enabled
while the update request was unresolved.

### GREEN

The same focused command passed all 10 tests after two narrow fixes:

- Renaming a section atomically transfers its custom heading from the previous
  section key to the new key.
- Every control that can mutate the draft is disabled while saving; the deferred
  save regression verifies that the server-normalized response becomes visible
  after resolution.

### Final Checks

- `bun -e 'JSON.parse(await Bun.file("src/assets/locale/en/settings.json").text())'`
  exited 0.
- `git diff --check` exited 0.
