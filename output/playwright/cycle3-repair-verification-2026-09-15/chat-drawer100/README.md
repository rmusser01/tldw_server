# Mobile Chat drawer selection — TASK13260.41 / UAT100

The bounded repair has a clear independent source review and parent-run native acceptance on the isolated multi-user UI at http://127.0.0.1:18281/chat. This bundle records those checks; full fresh single/multi acceptance remains separate. The original repair/review reports retain their then-pending native/compiler wording; the later evidence below updates that checkpoint without rewriting history.

## Native RED and acceptance

| Control | Evidence | Observed result |
|---|---|---|
| Original mobile failure, 23:43:40.525Z | uat098-final-mobile-robot-mouse.txt and robot-snapshot.txt | Actual Robot row click restores the correct saved identity and two messages, including BEEP BOOP, but one Chats dialog remains. |
| Mobile different/current target, 23:57:00.447Z | uat100-native-start.txt, mobile-selection.txt and mobile-closed.png | Both actual row clicks close the dialog. Saved Robot ID 2b88abd0-9bcb-4782-95cc-22077dc5145d, character5 and two messages remain; exact original question/answer and canonical title are returned. Screenshot is 390×844. |
| Desktop selection, 23:58:06.949Z | uat100-native-desktop-start.txt, desktop-selection.txt and desktop-open.png | At 1280×720, the actual Cedar row click leaves the sidebar visible with two collapse controls. Recorded command awaits original saved Cedar ID 812385d9-f19b-4bdf-82a0-1d850884a4b3, metadata loaded and two messages before returning. Canonical title and original answer are visible. |

The original command/result text is retained, with the three embedded results additionally parsed in RESULTS.json. Both PNGs were visually inspected during retention: mobile shows the unobscured Robot reply and desktop shows the open sidebar beside Cedar. No inference or new conversation was dispatched in these recorded selection commands. No separate network mutation audit is claimed. Native checks cover web server rows; folder/local cancellation/failure and shared/extension shell behavior are automated checks/source review, not native coverage. No claim of error-free unrelated routes or console is made.

## Regression and review evidence

- Product RED: shared5 failures/22 passes; folder2 failures/2 passing negative controls; web1 failure/23 passes. Those expected failures are retained in the three product-red logs.
- Implementer final: shared49 tests/7 suites plus web24/1 = 73 distinct tests/8 suites. Earlier intermediate runs overlap and are omitted.
- Independent review: shared45/6 plus web24/1 = 69 distinct tests/7 suites. This overlaps the implementer73; do not sum them as142 unique tests. Reviewer includes the real useSelectServerChat context-reset control. Exact commands are in the retained reports.
- Existing Web authStorage mock exports were repaired; all21 prior assertions remain, plus3 new cases. Initial missing-import/mount-count harness failures are described in the report but excluded as product RED. No tests were disabled.
- Test logs retain expected negative-control 409 diagnostics, Node localStorage experimental warnings and the existing mocked onPressEnter warning; passing suites do not mean warning-free output.
- Root-scoped ESLint: all10 code/test paths have0 errors and7 unchanged warnings, with0 added/removed signatures. Only embedded hook line-number references were normalized in the comparison. The independent five-production-path run reports the same0/7. No new lint run was needed for retention.
- Combined compiler checkpoint at 2026-09-16T00:00:15.339Z:90 current diagnostics versus90 merged baseline,0 added/removed. Both current log and comparison are retained. This is not a clean typecheck or an exit-zero claim; the comparison references the earlier baseline log outside this bundle. Bandit is inapplicable to this TypeScript-only unit.

## Contract and source provenance

The callback means explicit synchronous target acceptance, not eventual transcript-load completion. A later loader error remains visible in Chat. Cached/fetched folders notify only after target handoff; failures do not notify. Current-row clicks close without reloading; bulk/Trash do not notify. Only mobile drawer instances receive the close callback, and ordinary route-change dismissal remains. LocalChatList is not a modern sidebar tab and its existing accepted/current-request semantics were preserved. No new transport or account-isolation guarantees are implied.

The five production paths froze at 2026-09-15T23:53:17.286Z. source-verification.json freshly checks all10 code/test hashes against the original owned manifest; both independent before/after five-path hash records are retained. The task entry in the original owned manifest is a historical hash and may change through authorized CLI retention notes. source-before.json is a historical index: its nine private originals were hash-checked at packaging but are not copied into this bundle. These manifests do not claim unrelated Media/Header source was frozen.

## Retention, safety and limits

INDEX.json maps28 original artifacts to source/copied hashes, byte lengths, timestamps and transformations. JSON/PNG bytes are exact. Text/log copies remove trailing horizontal whitespace and excess final blank lines only; originals remain unchanged. SHA256SUMS indexes every payload except itself. SCAN.json records matching against14 known isolated-runtime secret values plus JWT/private-key patterns; private manifests and credential values are never copied. PNGs receive byte-pattern scanning plus visual inspection, not OCR. The private builder is /private/tmp/uat100-build-evidence.mjs, hash-bound in INDEX.json.

This retention task made no browser, runtime, production, test, global documentation or Git changes. Only this evidence bundle and official TASK13260.41 retention notes are repository edits. Parent owns final integration and full acceptance.
