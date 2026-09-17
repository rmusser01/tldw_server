# Independent review — UAT225 Stage A and UAT226

## Verdict

**CLEAR for these bounded changes. No remaining actionable source finding.** UAT225 remains open for its separately assigned complete inactive-Sync lifecycle and native acceptance. This review does not certify UAT227's separate endpoint-test fixture repair.

Frozen author manifests:

- Stage A: `.tmp/uat225-stage-a-20260917/owned-manifest.json`, SHA256 `94e8c564912da826a8efa1c97936474802b7774118a46530325f8c81a0bf92b9`.
- UAT226: `.tmp/uat226-repair-20260917/owned-manifest.json`, SHA256 `42ab50d7f67898f1e8ff94883bd94ccb4f40f8aeadb8ecd13abd6d7513a11b7b`.

All ten owned file hashes and corresponding snapshots matched before testing and again at completion. `source-before.json` and `source-after.json` bind the exact bytes. The baseline-relative production AST inventory is retained in `ast-compile.json`.

## Source assessment

### UAT225 Stage A

The existing scope helper remains strict by default. Only the exact server-derived `legacy:<selected DB owner>` can use the new read allowance, and only while that owner has no registered authority. A different owner's authority does not interfere; an existing conflicting binding denies the allowance. Reads preserve existing owner/dataset SQL predicates, byte/fingerprint/deleted checks, and transaction-local PostgreSQL dataset settings. No authority row is inserted or changed.

The five opt-ins are actual reads: source loading, FTS readiness inspection, suggestion listing, evidence listing, and rejection-set loading. The additional registration-presence read supports cancellation capability checks. Admission, decision mutation, run lifecycle and maintenance methods retain strict scope validation. The nested caller rollback control passes on both backends. This is not a concurrency fence for the future full lifecycle; Stage B owns that boundary.

The production factory is the sole application constructor call site and explicitly supplies actual decision-service readiness. Missing decisions disable generation and remove accept/reject/reset. Registered scopes retain their established cancellation path; the actual factory/coordinator tests exercise owner-scoped Jobs access and exact replay. An unregistered scope has no actions and cannot reach Jobs admission. Disabled-feature, worker, provider and FTS precedence is preserved. Registration/storage faults propagate as sanitized errors rather than being interpreted as absent authority.

The added restriction hashes the original provider revision together with effective availability, reason and actions. Identical restricted preflights retain identical ETags; authority/readiness transitions produce a different restricted revision. Existing ready/canonical revision semantics are unchanged. The frontend adds only the known safe unavailable reason and matching English disclosure. Actual parser/Inspector tests verify disabled unsupported actions and retained supported cancellation. Canonical/public English parity is verified for all ten unavailable messages.

### UAT226

Only endpoint `list_suggestions` changes. It explicitly projects the six existing evidence fields into the response shape, consistent with the surrounding item projection. It does not enable generic dataclass serialization, relax Pydantic models, widen accepted fields, or bypass evidence reconstruction/filtering.

Actual populated router tests cover registered/unbound scopes on PostgreSQL and SQLite; exact source/target excerpts survive serialization. Deleted, changed, oversized and out-of-range target evidence is omitted. Foreign source access remains404 and a changed source fingerprint yields no current suggestions. Strict response and request validation controls remain intact.

## Independent verification

- Official required-PostgreSQL/SQLite combined suite: **149 passed, zero skipped, 80.07s**, five baseline warnings. Includes Stage A60, UAT22624 and existing endpoint/API65 cases.
- Frontend parser, actual Inspector/i18n, hook and adjacent service suites: **65 passed, four files, zero skipped**, 1.13s.
- Ruff: zero findings across three production and three test Python files.
- Bandit: zero findings and zero errors on production; zero findings/errors on tests with only B101 assertions excluded.
- All six Python files compile from source without writing bytecode.
- Correctly scoped ESLint: zero errors/warnings. The first invocation from the frontend directory ignored the two shared files as outside its base path; `eslint.json` preserves that non-verification. The corrected repository-root invocation with the explicit existing frontend config is `eslint-scoped.json`; ESLint additionally printed its existing root Pages-directory discovery notice. No source rule was disabled.
- Fresh frontend TypeScript: **90 diagnostics**, identical normalized messages to the author's frozen candidate output, zero added/removed. The author separately retained its original-baseline comparison. This is not a clean full-build claim.
- Bandit does not provide meaningful TypeScript/TSX coverage; the author's two parse limitations remain qualified.

Backend command: activate `.venv`, then run the official `run-pg-tests-explicit-jobs.mjs` with label `uat225-226-sidebar-independent` and the five paths in `backend-command.json`. No fixture skipping or manual database setup was used. Only disposable official test databases were touched.

## Causal evidence and boundaries

Inspected retained Stage A clean RED34/control22, the corrected provider-expectation receipt, original fresh-route RED4/control4, frontend parser RED2, and UAT226 clean RED12/control12. The UAT226 baseline passes dataclass objects to the strict nested response model; the retained HTTP500/model_type traces directly match the repaired projection. Historical fixture mistakes remain documented rather than counted as product failures. No additional baseline replay was needed to interpret these causal receipts.

The existing endpoint helper's global FakeAPI leak is separately tracked as UAT227. Stage A's scoped real-factory fixture makes these actual-boundary tests reliable but does not itself repair that helper. Author227 changed the adjacent helper after this review's pytest process had imported/collected the original helper; all ten reviewed files stayed frozen. This run therefore establishes Stage A/226 behavior, not UAT227 isolation/order acceptance.

No source, tests, task records, tracker, git, native application process, browser, provider configuration or model inference was changed by this review. Native acceptance remains separate.
