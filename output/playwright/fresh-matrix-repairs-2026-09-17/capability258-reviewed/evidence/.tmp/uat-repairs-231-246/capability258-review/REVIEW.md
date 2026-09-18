# UAT258 independent source review — changes requested

Task: TASK13260.200. Reviewed the supplied brief, author report, diff, actual authentication normalization, and changed source/test bytes.

## P2 — Cookie marker alone does not establish a supported transport

At `apps/packages/ui/src/services/tldw/server-capabilities.ts:743–744`, the new guard accepts any non-invalidated `authSource: "cookie-session"`. The real request client requires an exact quickstart server URL, single-user auth mode, and an eligible browser transport (`TldwApiClient.ts:448`).

This is not only a mocked impossible configuration. `initialize()` passes a validated dedicated cookie record into `resolveEffectiveTldwConfig`, but that helper also returns ordinary persisted `tldwConfig` records after credential normalization without removing an existing cookie marker. The real `getConfig()` therefore preserves a foreign-origin cookie marker, or a same-origin multi-user cookie marker without a token. `ensureConfigForRequest(true)` correctly rejects both. The new capability guard nevertheless calls the protected endpoint once.

**Requested correction:** reuse the request client's active-cookie transport predicate, exposing a narrow shared helper if necessary, then retain runtime-key/token/manual-key fallback precedence. Do not duplicate the origin/browser policy inside capability discovery. Add the two rejected-cookie cases with actual config normalization, alongside the valid exact-origin cookie positive control.

## Independent verification

- Existing focused and adjacent tests: **117 passed** across capability discovery, quickstart authentication, credential normalization, and chat-scope cache keys. This includes all 48 author capability tests.
- Review-only integration diagnostic: **2 failed, 1 passed**. Both unsupported cookie cases successfully prove that real initialization/getConfig preserves the marker and real authenticated-request readiness rejects it; they fail only the final expectation that protected dispatch count is zero (actual count 1). The valid exact-origin single-user cookie case passes.
- Production source and author test hashes match the author's recorded evidence. Public discovery, unknown entitlement fallback, configured endpoint failure handling, manual-key recovery, and authority-scoped runtime-key handling otherwise look consistent with the brief.
- Author compiler comparison was inspected: 90 baseline/current diagnostics, zero added/removed. The author reports zero lint errors and 28 existing warnings. These static checks were not redundantly rerun. Bandit could not parse either TypeScript file; its zero findings provide no TypeScript security assurance.

## Scope and evidence

Only files under `.tmp/uat-repairs-231-246/capability258-review/` were authored. The diagnostic borrows the existing quickstart-auth test harness, mocks storage/network dispatch, and executes the actual client normalization and auth-readiness functions. It makes no native requests and changes no product code, repository tests, runtime, database, model, Git, or Backlog state. All three UAT259 diagnosis artifacts were checked and remain byte-identical.

`audit.mjs` records 29 hashed inputs, test commands/results, the finding, and the preserved UAT259 hashes. `focused.json` and `cookie-guard.json` retain the independent results; the corresponding logs stay on disk. Native fresh-context and authenticated positive verification remain required after this finding is fixed. This is not full-matrix acceptance.
