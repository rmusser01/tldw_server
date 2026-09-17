# UAT225 Stage A — TASK13260.163

## Scope and status

Bounded read/capability repair. This stage does **not** close UAT225 or the promised inactive-Sync suggestion lifecycle. TASK13260.164 owns the required follow-on; no native restart or acceptance was performed here.

Five production paths plus three permanent test files are snapshotted separately from UAT226. The original fresh-read test/fixture was handed off by Retry. Its original source and causal packet remain untouched in the diagnosis folder (original run was 4 failures / 4 controls).

## Behavior

- Existing dataset helper remains strict by default. Five inspected read methods opt in: load_source_note, ensure_fts_ready, list_suggestions, list_suggestion_evidence, get_rejection_set. A missing binding is permitted only for exactly `legacy:<selected owner>`, and only while that owner has no authority row. An existing conflicting binding rejects the legacy read. No authority is inserted or replaced; the established transaction-local PostgreSQL dataset GUC behavior remains.
- One additional pure registration-presence read supports cancellation capability checks. It distinguishes unregistered eligible legacy reads from exact registered owner/dataset authority and propagates database errors. Authority-table RLS is owner-only; the absence check does not depend on a dataset-hidden authority row.
- The real factory explicitly passes whether a decision service exists. Missing decisions disable generation and remove accept/reject/reset; exact registered scopes retain cancellation. Fresh unbound scopes advertise no actions. Disabled-feature, worker, provider and FTS precedence remains. No mutation/admission/Jobs path is authorized by the read opt-in.
- The additional authority restriction derives a deterministic SHA256 revision from the original provider revision and effective generation availability, reason and actions. Canonical ready and existing worker-unavailable revision semantics remain unchanged. Repeated unchanged preflights have identical body and ETag.
- The frontend strict reason allowlist accepts the new existing error code. The actual Inspector renders the English disclosure, disables unsupported decisions, and preserves valid cancellation. Canonical English and the public extension-format English mirror have identical new copy; all ten graph-unavailable keys match. No other locale was rewritten.

## Causal record and verification

- Original fresh route diagnosis: 4 FAIL / 4 PASS across required PostgreSQL and SQLite.
- Expanded initial run: 24 FAIL / 16 PASS. Two populated fixture failures were a bound Python helper default source ID; retained as harness errors.
- Expanded clean run: 34 FAIL / 22 PASS. Two provider cases initially expected the wrong existing reason name; corrected four-case replay has 4 causal action failures. The other 32 failures were scope/readiness behavior. No production was changed before these receipts.
- First implementation: 56 PASS / 0 skips; adjacent store/API/capability/cancellation/endpoint/acceptance/lifecycle/retrieval/privacy suites: 193 PASS / 0 skips.
- Added four actual factory/coordinator registered cancellation controls passed. They exercise both canonical and legacy registered datasets and both backends, with owner-scoped Jobs interaction and idempotent replay.
- Strengthening populated evidence found a separate actual HTTP response bug, UAT226/TASK13260.165. Rejected suggestions deliberately erase evidence; a distinct pending target reaches real excerpts. The captured final candidate had 58 PASS / 2 HTTP failures. The failure was not discarded: UAT226 now has the actual populated router controls and the original raised-exception receipts. Stage A retains actual populated storage/facade reads, rejection metadata and stale-target evidence filtering. Combined final acceptance requires UAT226 GREEN.
- Frontend actual service/parser/Inspector/i18n and adjacent service/hook/Inspector tests: 65 PASS / 4 files. Original two new cases fail the actual parser on baseline.
- First mixed149 run:38 FAIL/111 PASS caused by an existing adjacent transport fixture directly assigning a FakeAPI to the shared endpoint factory without restoring it. This is recorded as harness pollution, not an application result. The owned real-DB fixture now explicitly pins the actual factory via scoped monkeypatch. The original endpoint test file is unchanged. Same149 rerun is the final gate.
- Final combined backend: **149 PASS / 0 skips**, 85.22s, five expected baseline warnings. This includes all60 Stage A cases, all24 UAT226 cases and65 existing endpoint/API cases.

## Static checks and limitations

Ruff: zero findings on owned Python source/tests; production baseline also zero. Bandit production zero findings/parse errors; tests zero with only the normal assertion rule B101 excluded. Initial test-only B608 table-loop and B106 synthetic completion-token findings were removed using fixed SQL statements and derived fixture tokens. Bandit cannot parse the two TS/TSX paths; its zero findings there are not security coverage. Scoped ESLint zero errors/warnings. Full frontend TypeScript retains the same 90 normalized diagnostics as the retained baseline, with zero additions/removals. No clean full-build claim. Both large Python production files already fail whole-file Ruff formatting in the baseline; no unrelated format rewrite was applied. New tests are formatted. Python files parse/compile.

No provider/model invocation, native UI, runtime process, credential/config change, raw DB provisioning, global authority/RLS change, git action or task edit occurred. Required PG tests use the official DB fixtures via the retained runner; they do not constitute a new raw-SQL tenant-security or native acceptance claim.

## Independent reproduction

From repository root, activate `.venv`, then use the existing official runner with a new evidence label:

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat225-226-independent node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_fresh_reads.py tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_unbound_scope.py tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_evidence_response.py tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_endpoints.py tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_api.py -q --tb=short
```

Frontend from `apps/tldw-frontend`:

```sh
bunx vitest run ../packages/ui/src/components/Notes/__tests__/NotesGraphSuggestions.unbound.test.tsx ../packages/ui/src/components/Notes/__tests__/NotesGraphInspector.suggestions.test.tsx ../packages/ui/src/components/Notes/__tests__/useNotesGraphSuggestions.test.tsx ../packages/ui/src/services/tldw/__tests__/note-graph-suggestions.test.ts
```

Exact author commands and sanitized receipts are `.tmp/fresh-uat-recovery-20260916/uat225-stage-a-*-command.json` / matching `.redacted.log`, and `uat225-226-isolated-final-green-command.json` / matching log. Do not retain private raw logs or connection configuration in durable packages.
