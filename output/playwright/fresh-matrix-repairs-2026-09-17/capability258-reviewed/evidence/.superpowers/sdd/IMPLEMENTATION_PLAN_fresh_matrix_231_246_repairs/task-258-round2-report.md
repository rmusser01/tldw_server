## UAT258 / TASK13260.200 maintained coverage — round 2

### Change

Added the three reviewed actual-client cookie-readiness cases to `apps/packages/ui/src/services/__tests__/tldw-api-client.quickstart-auth.test.ts`. The tests initialize the real `tldwClient` singleton against the existing storage harness, assert request readiness, then exercise `getServerCapabilities` with a forced refresh.

Foreign-origin and same-origin multi-user cookie markers keep public OpenAPI/docs-info discovery available while dispatching zero protected ingestion-capability requests. A valid exact-origin single-user cookie session is ready and dispatches one protected request. The nested harness clears storage, local/session state, runtime overrides, quickstart environment values, and reinitializes the singleton before and after each case.

No production source changed.

### Verification

- `bunx vitest run src/services/__tests__/tldw-api-client.quickstart-auth.test.ts --silent` from `apps/packages/ui`: 25 passed.
- Scoped adjacent suites (`server-capabilities`, quickstart auth, chat surface scope, single-user credential): 120 passed.
- Frontend ESLint configuration linted the exact changed test bytes: 0 errors and one pre-existing warning at the existing `any` cast.
- Bandit reported 0 findings and one TypeScript parser error. Bandit does not provide TypeScript security assurance.
- Compiler baseline comparison was not run because this test-only change did not trigger a diagnostics-change investigation, per task direction.

Evidence and the final source hash: `.tmp/uat-repairs-231-246/capability258/round2/evidence.json`.

### Limits

The maintained tests mock storage and transport I/O only. They do not run a native browser, server, model, database, provider, or runtime profile. Root retains those verification and checkpoint responsibilities.
