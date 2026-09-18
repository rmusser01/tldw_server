# UAT258 round-one review — tests-only changes requested

The production correction resolves the prior P2. `TldwApiClient` only exports its existing exact-origin active-cookie predicate; capability discovery reuses it. Hosted, active-cookie, runtime-key, multi-user-token, and manual-key precedence remains aligned with the request client. No further production change is requested.

Independent rerun of the actual-client diagnostic: **3 passed**. The foreign-origin and wrong-mode cookie configurations dispatch no protected request; the valid exact-origin single-user cookie dispatches one. The original two failures and initial review remain byte-identical.

## P2 — Keep the causal regression in maintained tests

`apps/packages/ui/src/services/__tests__/server-capabilities.test.ts:13` mocks `isActiveCookieSessionConfig`; the suite also mocks `getConfig`. Its inactive-cookie case proves mocked-helper wiring, but cannot exercise the real storage normalization and cookie transport policy that caused the issue. Existing `tldw-api-client.quickstart-auth.test.ts` never invokes capability discovery, and its legacy cookie-marker case only checks key scrubbing. Those adjacent tests do not retain the reproduced cross-module behavior.

The three causal actual-client cases currently exist only in `.tmp/uat-repairs-231-246/capability258-review/cookie-guard.review.test.ts`. Move or adapt them into the maintained quickstart-auth suite or a maintained real-client capability integration suite. Keep actual `initialize()`, `getConfig()`, and the active-cookie predicate; mocking storage/network I/O is appropriate. This is a **tests-only follow-up**, not another authentication-policy rewrite. Scratch review evidence must not be described as newly committed CI regression coverage.

## Verification and limits

Seven audit checks pass, including exact author hash parity and the one-line helper exposure. Author evidence records 117 adjacent passes and unchanged 90 compiler diagnostics; these were not redundantly rerun. Bandit cannot parse this TypeScript scope. No product, repository test, native, runtime, model, DB, Git, or Backlog state was changed.

Original review files are preserved. New evidence is in `ROUND1-cookie-guard.json`, `ROUND1-source.diff`, and `ROUND1-audit.json`. Native verification remains pending after the source/test gate.
