# UAT155 / TASK-13260.93 independent review

Reviewed at 2026-09-16T19:19:35.134Z.
Repository: /Users/macbook-dev/Documents/GitHub/tldw_server2
HEAD: 05e0c5593ef79e1b9cacc4e741f7a4ee559a6f87
Production baseline: 2d5ad06c86cf279fe0fcc6445009813d97ab1c1b.
The committed SettingsOptionLayout source at HEAD is byte-identical to that production baseline; HEAD adds documentation/checkpoint changes.

## Verdict

No actionable code or regression-test findings in the reviewed change. The bounded repair satisfies the code-level requirement: the actual Next WebUI shim does not offer an enabled Switch to Sidebar action, supported Chrome/Firefox extension paths remain enabled, and unsupported extension paths remain disabled. Native acceptance is still pending and this review does not close TASK-13260.93 or UAT155.

## Scope and inspection

- The entire production diff is one line at SettingsOptionLayout.tsx:108: sidepanelSupported now requires both !isNextWebAppRuntime() and the existing isSidepanelSupported().
- The new mounted regression suite imports the actual WebUI wxt-browser shim and actual shared sidepanel utility. It does not mock isSidepanelSupported/openSidepanel. The WebUI test first proves the real utility reports true for the shim, then checks the rendered action is disabled and clicking it invokes neither setSetting nor sidePanel.open/setOptions.
- Runtime bootstrap imports that shim and exposes/merges it as global chrome (extension/shims/runtime-bootstrap.ts:1, 55-77); its sidePanel.open and setOptions are noopAsync (extension/shims/wxt-browser.ts:363-365). This is the same false capability positive reproduced by the test.
- The marker is appropriate for the actual Pages Router path: installed Next client initialize sets window.__NEXT_DATA__ before hydration (next/dist/client/index.js:139-140 and next/dist/client/next.js:16). pages/settings/tldw.tsx loads the SettingsRoute with ssr:false at line 14. The typeof window guard is safe outside the browser; the Settings route does not SSR-render the sidebar button. No new detector or timing-sensitive state/effect was introduced.
- In the absence of the Next marker, the existing utility and click handler remain unchanged. Chrome coverage verifies three preference requests, active/current-window tab query, tab-specific setOptions and open. Firefox coverage verifies the same three requests plus sidebarAction.open. Unsupported extension coverage proves disabled/no preference calls.
- Close, return-target normalization, unsaved-change navigation events, mobile navigation and link behavior are untouched. Existing exit-navigation, focus-order, label and filter suites pass in the fresh focused run.

## Fresh verification

### Current production: GREEN

Cwd: /Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui

```sh
node_modules/.bin/vitest run src/components/Layouts/__tests__/settings-layout-sidebar-capability.test.tsx src/components/Layouts/__tests__/settings-layout-exit-navigation.test.tsx src/components/Layouts/__tests__/settings-layout-focus-order.test.tsx src/components/Layouts/__tests__/settings-layout-labels.test.tsx src/components/Layouts/__tests__/settings-layout-filter.test.tsx --maxWorkers=1 --no-file-parallelism
```

Exit 0: 5 files passed, 24 tests passed. Log: /private/tmp/uat155-independent-green.log.

### Committed production: independent RED

A private pre-load Vite plugin serves exact git-show source from production commit 2d5ad06c86cf279fe0fcc6445009813d97ab1c1b only for SettingsOptionLayout.tsx. The author test remains unchanged; production files were never reverted or rewritten. The plugin logs its source SHA-256 when actually used.

```sh
node_modules/.bin/vitest run --config /private/tmp/uat155-independent-baseline.vitest.config.mts src/components/Layouts/__tests__/settings-layout-sidebar-capability.test.tsx --maxWorkers=1 --no-file-parallelism
```

Exit 1: 1 failed / 3 passed. The sole failure is the WebUI toBeDisabled assertion at test line 93; the original button is enabled. Chrome, Firefox and unsupported-extension cases pass. This confirms the new test detects the original defect rather than merely accepting the repair. Log: /private/tmp/uat155-independent-red.log.

### Other checks

- Scoped git diff --check: exit 0.
- Reviewed source/test SHA-256 verified both before and after all test runs; unchanged and identical to the author report.
- Node prints its existing experimental localStorage warning in both runs; no current-suite failures.

## Exact source/test SHA-256

```text
fae250e03181e7d5d9abaf7a236faa9f7f2e6693145703c32557703abde9e284  apps/packages/ui/src/components/Layouts/SettingsOptionLayout.tsx
4aae0c79221d873e67aa7621d7abf4817e4b4f09569b0166494533e1629535d0  apps/packages/ui/src/components/Layouts/__tests__/settings-layout-sidebar-capability.test.tsx
```

## Private evidence SHA-256

```text
6203e31c7ab6cb0d7c50ff0d5870988a8195774798ffd8da33e000fa625bc5dc  /private/tmp/uat155-independent-green.log
7ab012c2e5cfd1103457a4462011ed3430be0fd39976b0347ce47799c71c8900  /private/tmp/uat155-independent-red.log
7202f11585da5d61a17f864139eb9adb27c051b97924aea33838b26935855f8f  /private/tmp/uat155-independent-baseline-source.tsx
22101194b8d79703458ea5df9b6cee886f1a277437deb245b10320228a36a336  /private/tmp/uat155-independent-baseline.vitest.config.mts
```

## Limits and ownership

- No native browser/runtime was started or driven. No inference request, native extension API call, or real preference persistence was exercised. Preferences are observed as mocked setSetting calls; Chrome/Firefox API functions are test spies.
- No native acceptance, screenshots, console state, or lost private profiles are recertified. Exact native Settings acceptance remains root-owned and required before closure.
- No full application build/typecheck, full repository suite, or lint rerun was performed in this independent review. Author lint/format reports were read but are not presented as independently verified results here. Bandit does not analyze these TSX files; no Python was changed.
- No production, test, task, root-document, staging, or commit edits were made. Review artifacts and the baseline loader are confined to /private/tmp. Existing unrelated working-tree changes were left alone.
