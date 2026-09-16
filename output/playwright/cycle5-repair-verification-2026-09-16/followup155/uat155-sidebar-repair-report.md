# UAT155 / TASK13260.93 Sidebar capability repair

Base: codex/fresh-install-uat-fixes, HEAD 2d5ad06c86cf279fe0fcc6445009813d97ab1c1b.

## Implementation

The WebUI runtime bootstrap exposes its actual wxt-browser shim as global chrome. Its sidePanel.open is noopAsync, but the shared utility treats that function as supported. Settings now combines its existing Next runtime detector with actual sidepanel capability: `!isNextWebAppRuntime() && isSidepanelSupported()`.

The existing button is disabled in Next WebUI. Chrome and Firefox extension behavior stays available. No global shim, shared utility, Close/navigation, unsaved-change guard, runtime or browser changes.

Owned source/test files:

- apps/packages/ui/src/components/Layouts/SettingsOptionLayout.tsx (one-line production change)
- apps/packages/ui/src/components/Layouts/__tests__/settings-layout-sidebar-capability.test.tsx (new mounted suite)

Associated task updated through official Backlog CLI before code edits, then with RED/GREEN evidence. AC1/2 checked; task remains In Progress pending root-owned independent review/native acceptance/closure. No staging or commits performed.

## Validation

- RED before production edit: 1 failed / 3 passed. Real mounted SettingsLayout with actual WebUI wxt-browser shim exposed as chrome and __NEXT_DATA__; actual isSidepanelSupported reports true, and disabled assertion fails. `/private/tmp/uat155-red.log`.
- GREEN after change and test formatting: 5 files / 24 tests pass. New suite covers real WebUI shim with no preference or open writes; Chrome exact three preference requests plus active tab query/setOptions/open; Firefox three requests plus sidebarAction.open; unsupported extension negative. Existing exit-navigation, focus-order, labels and filter suites all pass. `/private/tmp/uat155-green.log`.
- ESLint: baseline production and final production/test both zero errors and warnings. JSON: `/private/tmp/uat155-eslint-baseline.json`, `/private/tmp/uat155-eslint-final.json`. Config prints an existing pages-directory notice when run from repository root.
- New suite Prettier check passes. Production source already fails Prettier at HEAD and remains otherwise unformatted; baseline stdin check recorded at `/private/tmp/uat155-prettier-baseline.log` (exit 1). No unrelated formatting edits.
- `git diff --check` passes.
- Bandit not applicable to TSX-only source/test change; recorded in Backlog.
- Node emits its existing experimental localStorage warning in tests; no test failures.

Focused command (cwd apps/packages/ui):

```sh
node_modules/.bin/vitest run src/components/Layouts/__tests__/settings-layout-sidebar-capability.test.tsx src/components/Layouts/__tests__/settings-layout-exit-navigation.test.tsx src/components/Layouts/__tests__/settings-layout-focus-order.test.tsx src/components/Layouts/__tests__/settings-layout-labels.test.tsx src/components/Layouts/__tests__/settings-layout-filter.test.tsx --maxWorkers=1 --no-file-parallelism
```

## Exact final SHA-256

```text
fae250e03181e7d5d9abaf7a236faa9f7f2e6693145703c32557703abde9e284  apps/packages/ui/src/components/Layouts/SettingsOptionLayout.tsx
4aae0c79221d873e67aa7621d7abf4817e4b4f09569b0166494533e1629535d0  apps/packages/ui/src/components/Layouts/__tests__/settings-layout-sidebar-capability.test.tsx
```

Machine-readable hashes: `/private/tmp/uat155-code-test-sha256.txt`.

Independent review, exact native Settings acceptance, evidence retention, staging and commit are delegated to root; no native acceptance is claimed here.
