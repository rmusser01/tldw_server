# Independent review — UAT154 / TASK13260.92

**Verdict: clear.** The bounded fixture correction preserves the original assertions and relevant behavior. Independent focused run: **36 tests / 3 files passed, zero skips, exit0, 9.74s**. No product change or native acceptance is claimed.

## Scope and integrity

Reviewed the three frozen files against HEAD by removing only the exact newly inserted `vi.mock("@plasmohq/storage", ...)` block. Each remainder is byte-for-byte identical to HEAD. All77 original `expect` call sites remain:12 auth-mode,16 form-lifecycle,49 cookie-logout. No test, assertion, scenario, timing, production helper or failure expectation was weakened, deleted or skipped.

Independent comparison: `/private/tmp/cycle5-uat154-independent-byte-comparison.json`.
Independent frozen hashes: `/private/tmp/cycle5-uat154-independent-hashes.json`; all3 match the author's17:56:28.747Z freeze.

- `tldw-auth-mode.form.test.tsx`: `026a49957d45e37b73b2f0c88f9cbc5669d3f913f7598bb31aa753e1c983af6c`
- `tldw.form-lifecycle.test.tsx`: `77d71831a5d0ea79154c3c9a30f5259c74edcfe2612699f11d1698129163d3e5`
- `tldw.cookie-logout.test.tsx`: `b9367428d439626d3da24a225057a29b5627d1c179c54e6466910d1345ced906`

Paths are under `apps/packages/ui/src/components/Option/Settings/__tests__/`.

## Why this fixture correction is appropriate

The import resolves to the existing `apps/tldw-frontend/extension/shims/plasmo-storage.ts`, the same adapter configured in Next and WebUI Vitest. It provides actual localStorage serialization, shared cross-instance watcher registration, same-tab notifications, native storage-event dispatch and callback-map unwatch. The original shared-UI tests had loaded extension Plasmo in a jsdom environment without its extension backend; that cannot represent their intended WebUI persistence/event boundary.

The correction changes storage environment rather than replacing `useSettingsLoginStatus`, effective credential projection or invalidation logic. Existing controls still exercise ordinary same-tab/cross-tab/config-event login/logout, exact-pair invalidation including storage-only writes, current rotation after an old invalidation, foreign server rejection, valid offline auth, delayed account/server ABA reads, edited-target guards, cookie ownership and StrictMode cleanup. The extension-style boolean-return test still calls the real watch registration, overrides only its return value to true, and checks marker cleanup/unmount behavior. Thus the portable watch/unwatch contract remains covered; this is not a claim of native extension end-to-end acceptance.

The actual AntD auth-mode and Form lifecycle controls remain intact. Their existing unrelated component/server mocks are unchanged. The prior28-failure baseline comparison was inspected: exact matching failure headings on unchanged1ea5402c83. This review did not rerun that expensive baseline or the already independently passed25 timeout/request controls.

## Independent command and result

Working directory `/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui`:

```sh
./node_modules/.bin/vitest run src/components/Option/Settings/__tests__/tldw.form-lifecycle.test.tsx src/components/Option/Settings/__tests__/tldw-auth-mode.form.test.tsx src/components/Option/Settings/__tests__/tldw.cookie-logout.test.tsx --maxWorkers=1 > /private/tmp/cycle5-uat154-independent-green.log 2>&1
```

36/36 tests pass across3files,zero skips,9.74s. Existing jsdom CSS parsing and Node localStorage notices remain visible. Inspected author ESLint comparison:0errors/0warnings,unchanged baseline. No independent full compiler run; parent owns combined verification. Bandit is not applicable to this TypeScript test-only correction.

No repository/source/task/runtime/browser/inference/staging/commit changes were made. Only private review evidence was written.
