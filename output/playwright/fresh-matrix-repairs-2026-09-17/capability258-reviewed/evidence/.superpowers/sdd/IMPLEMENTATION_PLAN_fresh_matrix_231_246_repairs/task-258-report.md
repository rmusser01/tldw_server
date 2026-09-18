## UAT258 / TASK13260.200 author report

### Repair

`server-capabilities.ts` now separates public capability discovery from the optional, protected ingestion-source entitlement probe. The public OpenAPI and docs-info discovery remains available in a fresh browser context. The protected request is deferred when there is no usable credential or its cookie session was invalidated, leaving local-directory creation as unknown (`null`).

The protected probe still runs for manual single-user keys, multi-user tokens, valid runtime single-user overrides, active cookie sessions, and hosted session transport. A runtime override deliberately takes precedence over a stale multi-user config, matching `TldwApiClient` request semantics. Runtime-key cache scope uses the active override, so a later configured authority cannot reuse the fresh unconfigured entitlement state.

Existing configured endpoint failure behavior remains: generic ingestion source support stays true and the per-user entitlement stays unknown.

### Causal TDD evidence

1. RED: the new fresh-unconfigured dispatch test failed because `/api/v1/ingestion-sources/capabilities` was called (Vitest exit 1).
2. GREEN: after the guard, that focused test passed (Vitest exit 0).
3. RED: the runtime-override control failed with a stale multi-user config without a token (Vitest exit 1).
4. GREEN: after matching runtime precedence, that focused test passed (Vitest exit 0).
5. Final focused suite: `bunx vitest run src/services/__tests__/server-capabilities.test.ts --silent` — 48 passed, exit 0.

The suite covers fresh/unconfigured discovery, placeholder keys, invalidated cookies, manual keys, multi-user tokens, cookie and hosted sessions, runtime overrides, configured endpoint failure, and recovery after later credential configuration.

### Static and security checks

- `git diff --check`: exit 0.
- Scoped ESLint with the frontend config: exit 0, 0 errors, 28 pre-existing `no-explicit-any` warnings (no warnings added by this repair).
- Full frontend `bun run typecheck` completed with its known non-clean project diagnostics. A same-dependencies compiler API comparison using HEAD bytes only for the two owned files recorded 90 baseline and 90 current diagnostics, with 0 added and 0 removed. See `.tmp/uat-repairs-231-246/capability258/compiler-comparison.json`.
- Bandit: exit 0 with 0 findings, but it could not parse either TypeScript file. This is an unsupported-language result and is not TypeScript security assurance.

### Hashes and review handoff

Current SHA-256 values and red/green/static command summaries are in `.tmp/uat-repairs-231-246/capability258/evidence.json`. The author performed a clean scope/diff review against `TldwApiClient` request precedence and cache behavior. The brief reserves independent review and native verification for root; those remain the required handoff checks.

## Round 1 cookie predicate correction

Independent review found that checking `authSource: "cookie-session"` alone was too weak: `TldwApiClient` retains foreign-origin and wrong-auth-mode cookie markers in storage, but `ensureConfigForRequest(true)` does not treat either as an active browser transport. The capability probe could therefore still issue its protected request before any usable credential was present.

`isActiveCookieSessionConfig` is now exported from `TldwApiClient` and used directly by the capability guard. It checks the same quickstart server origin, single-user mode, and same-origin cookie transport conditions that client initialization uses. The remaining guard order preserves hosted transport, runtime single-user override, multi-user access tokens, and manual usable keys. Inactive cookie markers leave entitlement unknown while public discovery continues.

### Round 1 causal and adjacent evidence

1. RED reviewer diagnostic: exit 1, 2 failed / 1 passed. Foreign-origin and same-origin multi-user cookie markers each issued one protected capability request.
2. GREEN reviewer diagnostic: exit 0, 3 passed. Those markers dispatch no protected request; the exact-origin single-user cookie case dispatches one.
3. Adjacent coverage: exit 0, 117 passed across `server-capabilities`, quickstart auth, chat-surface scope, and single-user credential suites.
4. `git diff --check`: exit 0.
5. ESLint for the three touched files: exit 0, 0 errors, 560 warnings. The warnings are reported without treating the lint run as clean.
6. Compiler API comparison against HEAD for all three touched files: 90 baseline diagnostics, 90 current diagnostics, 0 added and 0 removed.
7. Bandit: exit 0, 0 findings, and 3 TypeScript parsing errors. Bandit does not provide TypeScript security assurance here.

The first combined revalidation used a repository-relative Vitest executable after changing into `apps/packages/ui` and exited 127. It made no source changes; both intended commands were rerun from their correct working directories and passed. The preserved round-one record, including final hashes, is `.tmp/uat-repairs-231-246/capability258/round1-evidence.json`.
