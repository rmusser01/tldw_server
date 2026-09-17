# UAT250 / TASK13260.192 — transport test spy cleanup

One existing test-only path frozen in source-freeze.json. No production,248-owned source/tests, assertion, task, runtime or compiler changes.

## Cause and fix
The failed Chat completion transport group creates console.warn spies through setupTransport but restores only stubbed globals. Frontend default configuration does not automatically restore spies. Success/cancellation cases therefore see earlier failure-warning calls, and the final nested fixture leaks its warning spy into the later expected-status case. vi.resetModules and explicit module-mock resets do not restore that console spy.

Added one nested afterEach(vi.restoreAllMocks) hook and its import. This restores actual spy methods after every case, including failure paths, without suppressing production warnings or replacing assertions. Existing vi.fn transport/storage fixtures and their explicit resets remain intact. No global configuration workaround.

## Verification
- **red.log:7 failed/118 passed**, matching the independently identified baseline failure set.
- **green-baseline.log:125 passed/1 file, zero skips**, using the same retained original-production overlay config as RED.
- **green-current-adjacent.log:169 passed/4 files, zero skips**, using unmodified default frontend Vitest configuration and current frozen248 sources. Includes full transport suite, speech sanitizer249, configuration guidance and captured request scope.
- ESLint actual root configuration: **0 errors/80 warnings**, identical baseline0/80, no introduced warning signatures. Scoped diff-check exit0.
- Entire file is byte-identical to the original after removing exactly the added import symbol and cleanup hook; all289 original assertion lines preserved. assertion-preservation.json records this check.
- Exact commands in commands.json; configs hashed in config-inputs.json. No tests disabled, filters or skips used.
- Scoped project-venv Bandit was run on the touched TypeScript test: exit0,0 findings, but the file failed Python parsing (bandit.json). This provides no TypeScript security coverage. Exact command: source .venv/bin/activate && python -m bandit apps/packages/ui/src/services/__tests__/background-proxy.test.ts -f json -o .tmp/uat-repairs-231-246/transport-fixture250/bandit.json. Global compiler belongs to root and was not run. Mocked transport tests do not claim native provider acceptance.

Source is frozen for independent review. Root owns tracking and integration.
