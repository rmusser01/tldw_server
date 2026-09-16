# TASK13260.41 / UAT100 — dismiss mobile Chat drawer on selection

## Disposition
Implementation frozen 2026-09-15T23:53:17.286Z. Focused interaction tests and lint passed; independent review and native390px mouse acceptance pending. No native pass claimed.

## Root cause and contract
Native evidence /private/tmp/uat098-final-mobile-robot-mouse.txt and /private/tmp/uat098-final-mobile-robot-snapshot.txt: Robot row click accepted the correct saved ID/title and restored2 messages/BEEPBOOP, but the drawer remained. Both layouts close the mobile drawer when pathname changes; saved conversation selection stays on /chat. ChatSidebar had no accepted-selection callback.

Root approved explicit target-handoff semantics: close once the user-selected server target is accepted synchronously by useSelectServerChat, without waiting for its separate transcript loader. Later HTTP load errors remain visible in Chat. Folder fallback lookup must succeed before target handoff/notification. No inference from background store hydration or global selected-ID effects.

## Minimal production change (5 paths)
- Common/ChatSidebar.tsx accepts optional onConversationSelected and forwards it to its actual server/folder tabs.
- Common/ChatSidebar/ServerChatList.tsx emits after normal explicit row handoff. Clicking already-current conversation emits without reloading. Bulk selection and Trash rows do not emit. A thrown handoff cannot reach the callback.
- Common/ChatSidebar/FolderChatList.tsx emits after cached or fetched target handoff. Pending lookup does not emit; lookup/handoff failure does not emit.
- Shared Layout.tsx and web WebLayout.tsx each provide the close callback only on their mobile Drawer ChatSidebar instance. Persistent desktop sidebar has no close callback. Existing normal route-change dismissal retained.

No auth/identity/loader mutation or source transport changes. LocalChatList is not a tab of this modern sidebar; no new local tab invented. Its existing awaited true/current-request onSelectChat callback and actual cancellation/unmount/failure tests remain unchanged. This unit does not add a loader-completion contract or alter legacy sidebar selection behavior.

## Test-first evidence
- /private/tmp/uat100-shared-red.log:5 expected product failures/22 passes (server accepted/current row, actual ChatSidebar server/folder forwarding, shared mobile Drawer).
- /private/tmp/uat100-folder-product-red.log:2 expected product failures/2 negative controls passed (cached/fetched accepted callbacks absent; failures do not emit).
- /private/tmp/uat100-web-product-red.log:1 expected product failure/23 passes (same-route accepted mobile selection remains open).
- Separate harness corrections are not product RED: initial folder creation used wrong cwd, then missing expect import; /private/tmp/uat100-folder-red.log records missing-import failure. Initial desktop assertion counted an existing mount-time collapse action; final test clears that mount call before measuring selection. Its initial attempt is /private/tmp/uat100-web-red.log.

Web baseline fixture repair: existing mock lacked getSessionAccessToken and getEffectiveStoredTldwConfig exported/used by production NotificationLifecycleProvider. Added just those exports with null values consistent with existing mocked API-key identity. Existing21 Web assertions retained; final all21 plus3 new controls pass. The original18 missing-export failures are independently retained in .40 evidence /private/tmp/uat099-web-shell-baseline.log. No tests disabled or auth code changed.

## Final verification
- Shared49 tests/7 files pass: /private/tmp/uat100-shared-final.log. Includes real ServerChatList/FolderChatList handlers with mocked transport/selector boundary; actual ChatSidebar callback forwarding; actual shared Layout drawer; existing local load13 cases and lazy/coordinator controls.
- Web24 tests/1 file pass: /private/tmp/uat100-web-green.log. Actual WebLayout owns Drawer state; ChatSidebar callback seam is mocked. Checks same-route dismissal, persistent desktop behavior, ordinary route-change dismissal, existing backend recovery/shell tests.
- Total73 distinct tests/8 files. Earlier44/5 shared green overlaps final and is not additional coverage.
- Correct repository-root ESLint covers all10 code/test files:0 errors,7 unchanged warnings,0 added/removed; only embedded line references normalized. Before/after/comparison /private/tmp/uat100-eslint-{before,after,comparison}.json. Existing root missing-pages advisory unchanged.
- Scoped git diff --check clean. Bandit N/A TypeScript callback/UI plumbing only; no Python. Parent owns combined compiler checkpoint; full tsc not rerun here.

Reproduce shared final (cwd apps/packages/ui):
    ./node_modules/.bin/vitest run src/components/Common/ChatSidebar/__tests__/ServerChatList.reliability.test.tsx src/components/Common/ChatSidebar/__tests__/ChatSidebar.tools-first.test.tsx src/components/Common/ChatSidebar/__tests__/FolderChatList.selection.test.tsx src/components/Common/ChatSidebar/__tests__/ChatSidebar.lazy-history.test.tsx src/components/Common/__tests__/ChatSidebar.coordinator.test.tsx src/components/Layouts/__tests__/Layout.shell-overrides.test.tsx src/hooks/__tests__/useLoadLocalConversation.test.tsx --maxWorkers=1 --no-file-parallelism
Reproduce Web (cwd apps/tldw-frontend):
    ./node_modules/.bin/vitest run __tests__/components/layout/WebLayout.chat-scroll-contract.test.tsx --maxWorkers=1 --no-file-parallelism
Lint (repo root):
    apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/components/Common/ChatSidebar.tsx apps/packages/ui/src/components/Common/ChatSidebar/ServerChatList.tsx apps/packages/ui/src/components/Common/ChatSidebar/FolderChatList.tsx apps/packages/ui/src/components/Layouts/Layout.tsx apps/tldw-frontend/components/layout/WebLayout.tsx apps/packages/ui/src/components/Common/ChatSidebar/__tests__/ServerChatList.reliability.test.tsx apps/packages/ui/src/components/Common/ChatSidebar/__tests__/ChatSidebar.tools-first.test.tsx apps/packages/ui/src/components/Layouts/__tests__/Layout.shell-overrides.test.tsx apps/tldw-frontend/__tests__/components/layout/WebLayout.chat-scroll-contract.test.tsx apps/packages/ui/src/components/Common/ChatSidebar/__tests__/FolderChatList.selection.test.tsx -f json

## Scope/limits and artifacts
Five product paths + five test paths + official task record; exact hashes /private/tmp/uat100-owned-paths.json. Production-only freeze /private/tmp/uat100-production-freeze.json; source-before copies/index /private/tmp/uat100-source-before.json; exact product diff /private/tmp/uat100-production.diff. Native pointer/geometry and selected transcript remain parent-owned and pending. Tests join actual action handlers, forwarding and layout owners across explicit seams; no single full JSX/browser integration claim. No browser/runtime/commit/globaldocs operations performed; no Media or ChatHeader edits. Task stays In Progress for independent review/native acceptance.
