# UAT134 / existing TASK13260.24

Frozen 2026-09-16T13:35:12.615Z.

## Correction

Readiness used apiSend without refreshAuth. Only captured readiness GET /api/v1/auth/sessions now opts into the existing scoped createDirectRuntime; its existing refresh implementation is exported without modification. Connection passes its captured config/generation; direct storage reads/writes and final config result reject obsolete generations. Scoped runtime retains target/principal checks, canonical token pair rotation/invalidation and its existing refresh singleflight key. Coalescing is bypassed for captured readiness so different generations cannot share a stale readiness response. Other requests/mutations use the old transport unchanged.

## Tests and static evidence

- Permanent original RED2 failures/6 controls: /private/tmp/cycle5-repair-ui-134-red.log. Actual store/apiSend/core with controlled HTTP/storage and actual canonical credential helpers.
- Final GREEN157/5: -green.log. New15-case boundary plus existing connection, api-send, background web-refresh and recipe uncertainty suites. Successive rotation uses raw original config plus real rotation storage, terminal401 clears effective credentials and subsequent checks issue no requests. Tests cover a concurrent existing scoped background request sharing one refresh, delayed refresh owner/server/A→B→A changes, old401, marker read invalidation,503/network/403, valid bearer, manual credential metadata, hosted cookies and mutation no-replay.
- Command from apps/tldw-frontend: bun run test ../packages/ui/src/store/__tests__/connection.readiness-refresh.test.ts ../packages/ui/src/store/__tests__/connection.test.ts ../packages/ui/src/services/__tests__/api-send.test.ts ../packages/ui/src/services/__tests__/background-proxy.web-refresh.test.ts ../packages/ui/src/services/__tests__/recipe-persistence-uncertainty.test.ts --maxWorkers=1 --no-file-parallelism
- Scope lint0errors33warnings, exact baseline; -lint-comparison.json and raw before/after. Existing connection mock assertions now account for captured client-only second argument.
- Full frontend typecheck90 existing diagnostics, zero added/removed after normalizing locations: -typecheck.log and -typecheck-comparison.json. This is not a clean compiler pass.
- git diff --check passed; Bandit not applicable TS-only.

## Validation limits / fixture corrections

No native browser/runtime/inference/staging/commit. TldwClient initialize/ragHealth are controlled, but its test getConfig uses actual direct credential resolver; runtime HTTP/storage seams are controlled, canonical rotation remains production. Concurrent test certifies existing scoped background refresh sharing; it does not claim consolidation of previously separate unscoped/scoped refresh maps. Existing connection suite requires WebUI shim; combined validation used WebUI config. During test setup, an unrelated disallowed Notes listing fixture was corrected to existing authorized exact note GET; manual credential fixture gained required metadata; hosted assertion allows fetch default same-origin. None required production changes. New generation marker read expects existing nested scope-error412 shape. Natural second expiry and revocation remain native acceptance work.

## Files

- apps/packages/ui/src/services/api-send.ts
- apps/packages/ui/src/services/background-proxy.ts
- apps/packages/ui/src/store/connection.tsx
- apps/packages/ui/src/store/__tests__/connection.test.ts
- apps/packages/ui/src/store/__tests__/connection.readiness-refresh.test.ts
- backlog/tasks/task-13260.24 - Preserve-authentication-across-transient-refresh-failures-and-recover-expired-sessions-cleanly.md

Hashes: /private/tmp/cycle5-repair-ui-134-manifest.json.
