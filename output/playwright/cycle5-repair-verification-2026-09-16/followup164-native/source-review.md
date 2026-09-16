# UAT164 / TASK-13260.101 independent review

## Final decision: clear after correction

No remaining actionable finding in the corrected six-file scope. Native final CTA acceptance remains root-owned.

## Initial frozen release: changes requested

Reviewed the six files in `owned-manifest.json`, baseline HEAD `ec28c34b7cf51f5e65100c0dba5f2bcd2b95711d`. All six current hashes matched the manifest and byte-for-byte review snapshots (`reviewer-hash-check.json`). No production, permanent test, browser, service, or git changes by reviewer.

### P2 — New setup CTA does not reach operator guidance after setup completion

`apps/packages/ui/src/components/Option/Models/index.tsx:572–576` promises provider configuration through server setup and routes to `/setup`. For a multi-user server whose setup status is `completed` or `skipped`, the actual destination renders **Connect your tldw server**, the single-user API-key form, and no operator-guide link. That existing form submits `authMode: "single-user"`. `MultiUserExitPanel` is reachable through the incomplete wizard, not these states.

This is a gap introduced by the new CTA's contract; the destination's old behavior is not claimed as newly introduced. Existing native incomplete-wizard evidence does not cover completed/skipped setups. Author confirmed their earlier route claim was too broad. Recommend a direct existing provider/operator guide with accurate copy rather than a setup-router rewrite. Root accepted the finding and assigned the bounded correction.

Private actual-route reproduction uses the existing OptionSetup test fixture, real route and entry-choice logic, synthetic completed/skipped multi-user hook data, and its existing wizard mock (which is not mounted in this branch). Assertions confirm API-key input, single-user guidance, and no wizard before the missing-guide assertion fails. It does not require runtime, browser, or production edits. Files: `reviewer-setup-probe.txt`, `reviewer-probes.config.ts`, `reviewer-probes.log`. Full probe run: **2 expected failures / 31 passed** (18 existing route controls, 12 existing Models tests, one extra refresh lifecycle control). The unchanged route counterexample is retained as evidence, not a gate after replacing the CTA with a direct guide.

## Verification of the initial release

- Fresh focused suite: **31 tests / 6 files passed**, `reviewer-current-tests.log`.
- Nonmutating baseline replay: current permanent tests with three original production sources loaded from the exact baseline commit through a private Vite `load` plugin. **9 failed / 22 passed**, `reviewer-baseline-tests.log`; failures cover new Home copy, catalog-empty copy, empty readiness, 401 distinction, 503 no-empty guidance, pending metadata, refresh recovery, and empty-to-ready refresh. Baseline sources retained under `reviewer-baseline/` and config in `reviewer-baseline.config.ts`.
- Additional actual Models control: successful Refresh → metadata503 Refresh → Retry. **Passed**. Last successful check string and saved defaults remain unchanged on failure; success notification count stays one; error recovery replaces setup guidance; Retry restores the ready model; exactly four metadata calls across initial load and three explicit operations. Injection in `reviewer-refresh-probe.txt`.
- Scoped ESLint: **0 errors / 0 warnings** in six owned files, `reviewer-eslint.json`. CLI emitted the existing repository-root pages-directory configuration notice; it was not a code diagnostic.
- Source confirms a single metadata query observer in ModelsBody; ModelsCatalog is a renderer and has no useQuery. Standalone AvailableModelsList retains one observer. Query key is unchanged. Actual shared query-client default retries at most once and disables focus/reconnect refetch. Query retry behavior is not expanded by this patch.
- Configured/defaults controls pass. Metadata200 empty is distinct from actual401/503. Home uses discovery success before showing its banner. Abort-like metadata handling was pre-existing and remains intentionally nonfatal; this review does not recast it as a new successful server receipt.
- Author compiler comparison (90 existing diagnostics unchanged) and Bandit TSX parsing limitation were inspected in their report; not independently rerun. Native acceptance belongs to root.

## Commands

From `apps/tldw-frontend`:

```sh
bunx vitest run ../packages/ui/src/components/Option/Models/__tests__ ../packages/ui/src/components/Option/CompanionHome/__tests__/CompanionHomeShell.providers.test.tsx ../packages/ui/src/components/Option/CompanionHome/__tests__/CompanionHomeShell.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/MultiUserExitPanel.test.tsx
bunx vitest run --config ../../.tmp/uat164-repair-20260916/reviewer-baseline.config.ts ../packages/ui/src/components/Option/Models/__tests__ ../packages/ui/src/components/Option/CompanionHome/__tests__/CompanionHomeShell.providers.test.tsx ../packages/ui/src/components/Option/CompanionHome/__tests__/CompanionHomeShell.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/MultiUserExitPanel.test.tsx
bunx vitest run --config ../../.tmp/uat164-repair-20260916/reviewer-probes.config.ts ../packages/ui/src/components/Option/Models/__tests__/ModelsBody.test.tsx ../packages/ui/src/routes/__tests__/option-setup-readiness.test.tsx
```

From repo root:

```sh
node apps/tldw-frontend/node_modules/eslint/bin/eslint.js --config apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/components/Option/Models/index.tsx apps/packages/ui/src/components/Option/Models/AvailableModelsList.tsx apps/packages/ui/src/components/Option/CompanionHome/CompanionHomeShell.tsx apps/packages/ui/src/components/Option/Models/__tests__/ModelsBody.test.tsx apps/packages/ui/src/components/Option/Models/__tests__/AvailableModelsList.test.tsx apps/packages/ui/src/components/Option/CompanionHome/__tests__/CompanionHomeShell.providers.test.tsx -f json
```

## Correction re-review

**P2 resolved; corrected release is clear.** Reviewed final manifest SHA256 `7e198f6b038faa5e48371dc0c1653d3256ffd8c88cb60fee8fd1cb491d5c44b2`. All six current files match their final manifest and review snapshots (`reviewer-final-hash-check.json`).

The empty-state CTA now opens the existing local-provider setup guide directly in a new tab, with `noopener noreferrer`. Guidance says to configure on the server or ask an administrator. It no longer promises that the frontend setup route provides a provider editor. The linked repository document exists and describes server-side provider configuration and readiness diagnostics. This is a guide for local endpoints; hosted credential controls remain available through the existing Provider Keys links. No setup, authentication, query, or permission behavior was expanded.

Fresh independent verification after freeze:

- **31 tests / 6 files PASS**, `reviewer-final-tests.log`, same main command above.
- Correction-specific nonmutating RED: the two final actual-render guide tests run against the exact initial Models/index.tsx snapshot both **FAIL as expected**; ten unrelated tests skipped. Config `reviewer-correction-red.config.ts`, output `reviewer-correction-red.log`. The initial release is retained under `reviewer-initial/` and `reviewer-initial-manifest.json`; this isolates the correction from the broader baseline.
- Private successful Refresh →503→Retry lifecycle control still **PASS**, `reviewer-final-refresh-probe.log`. Command uses `reviewer-probes.config.ts` with only ModelsBody.test.tsx and `-t review164`; unchanged route probes intentionally not rerun as a final gate.
- Scoped ESLint final **0 errors / 0 warnings**, `reviewer-final-eslint.json`, using repository-root command above. An initial recheck from frontend cwd ignored all six files as outside base path; it is retained as `reviewer-final-eslint-ignored-from-frontend.json` and is not counted as lint validation. The corrected root invocation parsed all six files.
- `git diff --check` on six owned paths: exit0.
- Author reports final compiler comparison90 baseline/90 current, no new diagnostics; report update only. Reviewer did not rerun the full compiler.

Final changed hashes for the correction:

- Models/index.tsx: `61cfd7a665bd5a2393ba247eb7a31022b08fe18f7086827a7e058c4002cb59a1`.
- ModelsBody.test.tsx: `50a5b1ce4c213ddb79c43b1a1c58101bd640fedc97c4d189c3fedd63a40b4789`.

No production/permanent-test changes, server/browser actions, staging, or commits by reviewer.
