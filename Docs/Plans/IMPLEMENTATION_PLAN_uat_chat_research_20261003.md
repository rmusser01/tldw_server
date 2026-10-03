# Chat and Research Workspace root-cause repairs — TASK-13260.281.3

## Stage 1: Reproduce dispatch lease false positives
**Goal**: Prove that harmless MCP loading and unrelated model settings writes reject a saved send.
**Success Criteria**: Regression tests fail for identity-only writes and retain real model/auth invalidation.
**Tests**: Normal chat history-selection integration tests using production capture/admission/settlement.
**Status**: Complete

## Stage 2: Preserve semantic dispatch leases
**Goal**: Compare the effective generation settings and selected tools rather than whole store object identities.
**Success Criteria**: Harmless state publication permits send; material selected settings, tools, model and account changes reject it.
**Tests**: Focused chat integration and history-admission suites.
**Status**: Complete

## Stage 3: Restore server workspace after migration
**Goal**: Restore the server workspace identity and its sources before fresh workspace initialization after reload.
**Success Criteria**: A migrated workspace survives reload; failures never create a replacement; restoration uses the current authorized API scope.
**Tests**: Workspace initialization/component regressions and migration/storage regressions.
**Status**: Complete

## Stage 4: Prove and repair duplicate Research generation
**Goal**: Trace canonical workspace chat retrieval and remove unused server answer generation if confirmed.
**Success Criteria**: Source tests prove retrieval-only RAG followed by one chat answer; existing generation budgets stay intact.
**Tests**: RAG mode selected-source and canonical chat pipeline tests.
**Status**: Complete

## Stage 5: Verify touched source
**Goal**: Run focused tests, formatting/lint checks, type checks where feasible and Bandit on the touched scope.
**Success Criteria**: Report exact passing checks and material limitations without claiming resumed UAT.
**Tests**: Source-only checks from apps/tldw-frontend; no runtime, browser or provider replay.
**Status**: Complete

## Validation notes

- Harmless store publication and original generation-enabled retrieval were observed red in source regressions before production fixes.
- The exact normalized RAG POST and real chat pipeline now prove retrieval-only preflight, preserved selected source IDs and context, and one streamed answer.
- Nine source suites pass 220 tests, including migration persistence and account/workspace scope rejection. These are source checks, not live UAT.
- The original UAT request receipts are unavailable; `/api/v1/rag/search` is established for the canonical current Research Workspace source path, without claiming a recovered original runtime receipt.
- Standalone RAG generation APIs and generation time budgets remain unchanged.
- ESLint on touched files has no errors; existing warnings remain. Bandit was invoked with the primary venv but cannot parse TypeScript and does not validate this frontend patch.
- Final fixture and formatting rerun passes all 35 tests in the three changed suites; the two new files pass Prettier checks.
- Full frontend `NODE_OPTIONS=--max-old-space-size=8192 node_modules/.bin/tsc --noEmit --incremental false` completes with existing unrelated React and dependency type conflicts, with no diagnostics in the touched paths.
- Stage5 is complete after independent actual-source review and coordinator integration checks; the coordinator owns publication.
- Independent review's disabled-tool MCP refresh and foreign-account receipt findings were reproduced red and repaired. Disabled-tool discovery does not alter the effective request; enabled tool removal still invalidates it. Foreign bound receipts are excluded, and unbound legacy hints require a scoped current-account list match before restoration and local scope binding.
- After those review repairs, the five affected suites pass 158 tests. Final lint and new-file formatting checks pass; the final 8 GB frontend typecheck again reports only the same unrelated paths and no touched-path diagnostics.

Final coordinator integration passes241 frontend cases across22 suites, including all changed Chat/Research regressions. Independent source review has no unresolved blocking findings; full types remain inherited/unrelated with0touched diagnostics. Live budget, provider and workspace UAT remain unaccepted.
