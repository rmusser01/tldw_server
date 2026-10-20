# C08 Admin status and role confirmation

Task: TASK-13260.278.18.83.45 (parent association recorded before edits).

ADR required: no. This repairs the shared client to honor existing audit-reason and acting-admin reauthentication guardrails; no API, authorization, persistence, or architecture rule changes. ADR index assessed.

Scope: ServerAdminPage, AdminUserUpdateRequest, existing admin domain request options, two new mounted test suites, and one selector-only correction in the existing design-system suite. Existing roles-management, password reset, create-user, and backend guards remain unchanged. User-role assignment offers only canonical user/admin values.

## Stage 1: Reproduce
**Goal**: Demonstrate premature unconfirmed mutation and missing audit/password fields through real mounted controls.
**Success Criteria**: Causal RED before production changes; inspect canonical service/schema and house forms.
**Tests**: Real AntD modal/form, delegated admin method, mocked transport only.
**Status**: Complete

## Stage 2: Repair and qualify
**Goal**: Explicit action/target confirmation with reason and existing admin password, validation, cancellation and visible errors.
**Success Criteria**: Affected UI/Web tests and configured types; actual lint baseline comparison; no new security findings. TS-only Bandit not applicable.
**Tests**: Active/role submit, invalid/blank input, Cancel/reopen, error/retry, pending duplicate protection, single-user compatibility, canonical role choices.
**Status**: Complete

## Stage 3: Independent and native acceptance
**Goal**: Independent source review followed by root-owned real Admin validation.
**Success Criteria**: Review clear and native status/role updates honor guardrails.
**Tests**: Root owns build/browser/runtime; no native acceptance inferred from fixtures.
**Status**: In Progress

Causal configured Node20 UI RED: 10 failed, natural exit1,19.24s. Actual premature PUT with is_active only and absent dialog. No setup/import failure. First edit script failed parsing before any write; its chained unchanged RED replay is not new coverage. No proof files created.

Self-review and independent review found P1: an unscoped request could forward the held acting-admin password after a connection change. Root approved canonical ownership remount plus existing configSnapshot/abortSignal domain options. Actual mounted real direct transport RED:3failed1same-owner-controlpassed,8.80s,natural1. Added only domains/admin.ts and one new private-lifetime test suite; original Admin test assertions preserved. Initial pre-scope types were UI0/Web0; final scoped checks below.

## Final qualification (source frozen, native pending)

- Canonical ownership remount clears held target/password when authority changes or is unresolved. Privileged requests bind the current owned configuration snapshot and abort signal; ignored-abort results cannot refresh/toast in the successor workspace. Same-owner credential refresh preserves the confirmation.
- Initial form fixture had two premature visibility assertions (8 passed/2 failed twice); awaiting actual visible dialog corrected both. A distinct whitespace-only password case reproduced one RED before the validation repair. Pre-authority qualification: UI23/2 and Web24/3 passed; those initial types are superseded below.
- Final authority-corrected UI run: 26 passed/1 failed (three suites); the sole failure was an existing global Empty selector now matching two legitimate panels. Root approved scoping that selector to its original budget panel, retaining both assertions. Focused final UI: 1 passed/11 deliberately filtered, natural exit0,1.73s. Thus27 distinct UI cases pass by split execution, not one combined green run.
- Final Web affected run:28 passed/4 files,0 skipped,natural exit0,22.30s (outer22.765s). Includes both new suites, existing design-system suite, and actual Next admin route.
- Final configured TypeScript: shared UI exit0/62.069s and Web exit0/41.980s, no diagnostics; noEmit, incremental false,8192MiB.
- Actual root-config ESLint inspected six TS/TSX paths:0 errors/723 inherited warnings (page13, client530, domain180, tests0). Exact pre-unit comparison: page15 (two obsolete console directives removed), client530 unchanged, domain180 unchanged with inverse baseline hash verified. Earlier Web-cwd ignored-file scan was rejected, not qualification. Existing root pages-directory advisory remains. TS-only Bandit is not applicable and is not security proof.
- Node20.19.5; configured default workers unchanged; every test/type command had600s outer cap. No runtime, browser, profile, build, installation, proof artifacts, or Git writes. Independent corrected-scope review and root live acceptance remain Stage3 gates.

### Reproducible command scope

All commands use `/Users/macbook-dev/.nvm/versions/node/v20.19.5/bin/node`.

- Shared UI cwd `apps/packages/ui`: `../../tldw-frontend/node_modules/vitest/vitest.mjs run src/components/Option/Admin/__tests__/ServerAdminPage.privileged-update.test.tsx src/components/Option/Admin/__tests__/ServerAdminPage.privileged-scope.test.tsx src/components/Option/Admin/__tests__/ServerAdminPage.design-system.test.tsx --reporter=dot`; final selector-only follow-up uses the design-system path plus `-t "uses an Empty state panel"`.
- Web cwd `apps/tldw-frontend`: `node_modules/vitest/vitest.mjs run ../packages/ui/src/components/Option/Admin/__tests__/ServerAdminPage.privileged-update.test.tsx ../packages/ui/src/components/Option/Admin/__tests__/ServerAdminPage.privileged-scope.test.tsx ../packages/ui/src/components/Option/Admin/__tests__/ServerAdminPage.design-system.test.tsx __tests__/pages/admin-server-route.test.tsx --reporter=dot`.
- Types in each configured project: `--max-old-space-size=8192 <Web node_modules>/typescript/bin/tsc --noEmit --incremental false --pretty false`.
- ESLint API from repository root with `overrideConfigFile: apps/tldw-frontend/eslint.config.mjs`, `lintText` and actual filePath for each changed TS/TSX file; baseline comparisons in memory only.
