# TASK13260.39 / UAT098 — mobile ChatHeader overlap

## Disposition
Implementation and focused static/interaction verification are complete and frozen. Independent review and actual native mobile/desktop geometry remain pending. This report does not claim a browser pass.

## Native RED and source cause
Parent retained actual 390x844 geometry at /private/tmp/uat097-reviewed-mobile-header-bounds.txt (checked23:20:25.609Z) and screenshot /private/tmp/uat097-reviewed-390-layout.png. Shortcuts spans x189.984–217.984; saved title spans x205.984–466.75:12px overlap and an oversized title hit area beyond390px. The inner group could not wrap, the brand could shrink while its children overflowed, and the title button retained its intrinsic260.766px width inside a clipped max220px wrapper. Document width alone did not detect the oversized hit area.

## Minimal change
Only apps/packages/ui/src/components/Layouts/ChatHeader.tsx: five CSS-class edits. Bound and wrap the left header group, keep sidebar/brand at intrinsic size, constrain the title button to its existing140–220px wrapper with block/w-full, and allow session badges to wrap. No controls are removed, no new breakpoints/dependencies, no changes to text, title attributes, edit callbacks, props, auth, identity, saved state or API behavior. Narrow screens may use another header line so controls fit. Existing desktop title limits remain.

Source before: 1e75ccb62ca303d38076efd42ca35c329a470b149bebfaa4a07c67a8f09fb0ab
Source after: 99ead6a0de84203afbe797b24d8da293a87fd668e3453fd5a245d42939fb996f
Freeze: /private/tmp/uat098-production-freeze.json (2026-09-15T23:30:56.674Z)
Exact diff: /private/tmp/uat098-source.diff

## Verification
- Existing Header/notification/shortcuts tests:59 in3 files before edit and59 in3 after.
- Existing actual Header title/rename/NextHead and SSR tests:16 in2 files after. Covers local/server rename, pending rename/newer draft, rejected rename retry, delayed history/account changes and same-owner rotation.
- Total post-edit distinct tests:75 in5 files. Baseline59 is overlapping evidence, not additional test coverage.
- Root scoped ESLint:0 errors,2 unchanged warnings;0 added/removed signatures. JSON coverage explicitly includes the changed file. Root Next ESLint emits its existing missing-root-pages advisory; no new lint finding.
- Scoped git diff --check clean.
- No new class-only test or altered test assertion. Existing behavior tests preserve interactions; JSDOM does not establish real layout or pointer geometry.
- Bandit not applicable to CSS-class-only TypeScript changes; no Python touched. No full compiler rerun for this class-only patch; no new type-level code.

Commands (existing executables, no install):
From apps/packages/ui:
    ./node_modules/.bin/vitest run src/components/Layouts/__tests__/ChatHeader.test.tsx src/components/Layouts/__tests__/ChatHeader.notifications.test.tsx src/components/Layouts/__tests__/HeaderShortcuts.test.tsx --maxWorkers=1 --no-file-parallelism
From apps/tldw-frontend:
    ./node_modules/.bin/vitest run __tests__/pages/chat-title.integration.test.tsx __tests__/pages/chat-title.ssr.test.tsx --maxWorkers=1 --no-file-parallelism
From repository root:
    apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/components/Layouts/ChatHeader.tsx -f json

Logs: /private/tmp/uat098-header-baseline-tests.log, /private/tmp/uat098-header-green-tests.log, /private/tmp/uat098-title-green-tests.log; lint baseline/current/comparison /private/tmp/uat098-eslint-{before,after,comparison}.json.

## Pending acceptance
Parent owns native390/1024/1280 geometry and pointer checks: all visible header controls must remain within viewport, nonoverlapping, and usable; full title remains accessible; rename input fits; Shortcuts/sidebar/current actions retain behavior. Independent reviewer owns source review. No browser, runtime, commit, global design or tracker edits made by implementer. Only source plus official TASK13260.39 notes changed; task remains In Progress until acceptance evidence is recorded.
