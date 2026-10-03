# Final combined fix wave — I1, I2, I3, I4 and M1

Base: `5d24666688b13792884f027dfe8c3cb058ab8bb4`. Requirements: `final-fix-brief.md`, the exact whole-branch findings, and the later I4/Ruling14 Save-only server-principal authorization. TASK-13453.2 was reopened before edits; TASK-13453.4 tracks I4. This is one combined repair, with RED before each corresponding production change and no subagents or additional reviewer. Commit SHA is appended below.

## Changes and evidence

- **I1:** the existing bounded scope validator now accepts `keyword_filter: ""` as no filter; null collection remains valid. The canonical importer explicitly validates the new required checkpoint before POST/PUT, so invalid new provenance cannot silently reuse an old marker. Actual mounted AnswerPanel uses `DEFAULT_RAG_SETTINGS`, real builder/storage/importer and fresh versus older Review markers. Tests assert current import ID, completed/draftRetained checkpoint, original typed note/media excerpts/trust/scope, one marker, canonical readback, actual tombstoned server restore and remount with exactly one snapshot upload. Oversized/invalid filters remain rejected; the invalid-current-import case performs zero writes and does not duplicate its already acknowledged snapshot on retry.
- **I2:** explicit `metadata.media_id`/`mediaId` is authoritative for stored web content; the content subtype remains descriptive. Note results remain excluded from media identity, and an external web result's arbitrary numeric ID remains non-authoritative. Actual mounted MediaKnowledgeActions sends `web_document` and `web` through the real builder/importer and attaches the same canonical item with zero snapshots. A numeric external-web result colliding with unrelated stored media still creates its own snapshot and does not select the unrelated item.
- **I3:** canonical PUT merges `request.headers` with expected-version. A non-null principal fixture checks actual GET→PUT→readback headers. A simulated server principal change enforces the real optional expected-user contract: the old code mutated once before readback rejection; the repaired code rejects before mutation and leaves current draft state and incomplete checkpoint intact.
- **M1:** only the new shared Review Ask/Research controls, preparation error and fallback source title use the existing `review` namespace and `defaultValue` convention. Tests use an actual i18next provider to resolve translated singular/plural labels, title interpolation and error; English defaults work when resources are absent. No locale-file churn.
- **I4:** traced actual WebClipperPanel owner.requestScope → saveWebClip → requestScopeFields → shared path guard; the native canonical POST was missing from the guard. Allow only `/api/v1/web-clipper/save` and its resolved trailing-slash variant with POST. Real TldwApiClient/domain/bgRequest/request-core tests cover HTTP dispatch, captured config/header on extension messaging, direct principal/server changes before dispatch, malformed/expanded/absolute paths and wrong methods. Backend source inspection also found Save lacked the expected-user dependency; under Ruling14, only Save now uses the existing optional `require_expected_user`, matching Notes. Real FastAPI + existing SQLite/media/job fixture tests prove matching/absent-header saves and mismatched/changed principal 412 with no save-service invocation, canonical clip document, workspace source or job side effect. Status/enrichment routes and auth mechanism are unchanged.

The original importer ownership/edit/note/version/legacy/lost-ack/deletion/tombstone/manual-selection guards are unchanged. The strict provenance check is at the required import boundary, preserving the existing general editing helper's fallback behavior. No dependency, persistence owner, runtime or public endpoint family was added.

## RED and focused GREEN

Frontend commands run from `apps/packages/ui`; all data is synthetic.

```sh
./node_modules/.bin/vitest run src/utils/__tests__/research-workspace-import.test.tsx -t 'actual AnswerPanel default|invalid required current|actual Review caller|scoped principal through' --maxWorkers=1
```

`/private/tmp/knowledge-final-import-red.log`: **7 failed,40 skipped**, 4.452s test time. Two default-scope imports never completed, invalid required provenance still wrote old content, two stored-web cases attached nothing, successful canonical PUT omitted the expected user, and changed principal mutated once. These were behavior failures, before production edits.

```sh
./node_modules/.bin/vitest run src/components/Review/__tests__/MediaKnowledgeActions.test.tsx src/utils/__tests__/knowledge-note-provenance.test.ts -t 'translated Review|English defaults|bounded keyword' --maxWorkers=1
```

`/private/tmp/knowledge-final-i18n-validator-red.log`: **3 failed,8 passed,13 skipped**, 1.79s. Empty filter and two translated-label cases failed. A preliminary English-default fixture incorrectly required an exact accessible name after AntD's fading loading icon; the selector was corrected to tolerate that existing prefix before this recorded RED. A cwd-relative test-edit command initially addressed the wrong path and made no change; corrected before production work.

```sh
./node_modules/.bin/vitest run src/utils/__tests__/research-workspace-import.test.tsx src/utils/__tests__/knowledge-note-provenance.test.ts src/components/Review/__tests__/MediaKnowledgeActions.test.tsx -t 'actual AnswerPanel default|invalid required current|actual Review caller|scoped principal through|translated Review|English defaults|bounded keyword' --maxWorkers=1
```

`/private/tmp/knowledge-final-focused-green.log`: **18 passed,53 skipped**, 4.53s. First production repair passed. The external numeric-web fence was subsequently added and passed in covering checks.

```sh
./node_modules/.bin/vitest run src/services/__tests__/background-proxy.web-clipper-scope.test.ts --maxWorkers=1
```

I4 `/private/tmp/knowledge-final-i4-red.log`: **5 failed,14 passed**, 1.27s, because allowed canonical requests and owner-change cases stopped at the wrong path guard. `/private/tmp/knowledge-final-i4-green.log`: **19 passed**, 1.81s after the two-line route permission.

Original project venv was sourced before every Python/backlog/formatter/security/hook command:

```sh
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_NEW/integration/test_web_clipper_api.py -k expected_principal_boundary -q
```

I4 API `/private/tmp/knowledge-final-i4-api-red.log`: **2 failed,2 passed,8 deselected**,4.32s. Both mismatched/changed-principal cases returned200 and saved before the dependency repair. `/private/tmp/knowledge-final-i4-api-green.log`: **4 passed,8 deselected**,4.72s. Final typed fixture `/private/tmp/knowledge-final-i4-api-focused-final.log`: **4 passed,8 deselected**,3.81s.

## Covering checks

```sh
./node_modules/.bin/vitest run src/utils/__tests__/research-workspace-import.test.tsx src/utils/__tests__/research-workspace-prefill.test.ts src/utils/__tests__/knowledge-note-provenance.test.ts src/components/Review/__tests__/MediaKnowledgeActions.test.tsx src/components/Option/KnowledgeQA/__tests__/AnswerPanel.workspace-handoff.test.tsx src/store/__tests__/workspace.split-storage.test.ts src/components/Option/ResearchWorkspace/__tests__/workspace-server-restore.test.ts src/components/Option/ResearchWorkspace/__tests__/workspace-server-reconcile.test.ts src/components/Option/ResearchWorkspace/__tests__/QuickNotesSection.stage2.test.tsx src/components/Option/ResearchWorkspace/__tests__/QuickNotesSection.stage3.test.tsx src/components/Option/ResearchWorkspace/__tests__/QuickNotesSection.stage4.test.tsx src/components/Option/ResearchWorkspace/__tests__/QuickNotesSection.save-ownership.test.tsx --maxWorkers=1
```

`/private/tmp/knowledge-final-fix-covering.log`: **152 passed in12 files**,19.04s, after changed-block formatting. Includes all existing importer/Quick Notes race, UUID/legacy, lost-ack, provenance, restore/reconcile, deletion and manual-selection regressions.

```sh
./node_modules/.bin/vitest run src/services/__tests__/background-proxy.web-clipper-scope.test.ts src/services/tldw/__tests__/service-prompt-scope-error.test.ts src/services/tldw/__tests__/knowledge-qa-scope-policy.test.ts src/services/tldw/__tests__/chat-title-scope-policy.test.ts src/services/__tests__/background-proxy.monitoring-scope.test.ts src/services/__tests__/web-clipper-client.test.ts src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx --maxWorkers=1
```

`/private/tmp/knowledge-final-i4-covering.log`: **220 passed in7 files**,6.48s. These two frontend covering sets are disjoint: **372 tests in19 files** total.

```sh
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_NEW/integration/test_web_clipper_api.py tldw_Server_API/tests/Notes_NEW/integration/test_web_clipper_sync_contract.py tldw_Server_API/tests/Notes_NEW/unit/test_web_clipper_endpoint_error_mapping.py -q
```

`/private/tmp/knowledge-final-i4-api-covering.log`: **44 passed**,4 warnings,17.72s. Existing Node26 localStorage/fixture warnings and Python warnings are not a warning-free claim; no test skips or unresolved failures in covering runs. Actual native/live browser and final whole-branch builds remain controller-owned.

## Types, formatting, security and hooks

```sh
# repository root
NODE_OPTIONS=--max-old-space-size=8192 apps/packages/ui/node_modules/.bin/tsc --noEmit -p apps/packages/ui/tsconfig.json
# apps/tldw-frontend
bun run typecheck
# apps/extension
bun run compile
```

Final logs `/private/tmp/knowledge-final-fix-ui-types-complete.log` (exit2,352 existing diagnostics), `/private/tmp/knowledge-final-fix-webui-types-complete.log` (exit0), `/private/tmp/knowledge-final-fix-extension-types-complete.log` (exit0). Earlier pre-I4 type runs also had352/0/0; these repeats were required by the added I4 scope. Full diagnostic-block multiset comparison normalizes only line/column header positions: **0 added/0 removed** against `/private/tmp/task2-save-race-types-final.log` and `/private/tmp/task2-fix3-types.log`; **0 added/1 removed** against archived353 `/private/tmp/knowledge-types-baseline.log` (known history-selection fixture). `/private/tmp/knowledge-final-fix-types-comparison.txt`, script `/private/tmp/knowledge-final-fix-compare.cjs`.

```sh
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m black --check tldw_Server_API/app/api/v1/endpoints/web_clipper.py tldw_Server_API/tests/Notes_NEW/integration/test_web_clipper_api.py
python -m ruff check tldw_Server_API/app/api/v1/endpoints/web_clipper.py tldw_Server_API/tests/Notes_NEW/integration/test_web_clipper_api.py
python -m bandit -r tldw_Server_API/app/api/v1/endpoints/web_clipper.py -f json -o /private/tmp/knowledge-final-fix-bandit.json
```

Black/Ruff exit0: `/private/tmp/knowledge-final-fix-black-check.log`, `/private/tmp/knowledge-final-fix-ruff.log`. Black initially wrapped one preexisting long assertion in the touched API fixture; no broad Python rewrite. Changed TS blocks/new test use the already installed Prettier. Bandit exit0: **169 production Python LOC,0 findings,0 errors** (`knowledge-final-fix-bandit.json/.log`). This is real Python applicability; no TypeScript security-scan claim.

Explicit `python -m pre_commit run --files` on exactly the thirteen scoped paths below passed all applicable checks: `/private/tmp/knowledge-final-fix-precommit.log`. The hook configuration skips Ruff/Black for these paths, so they were explicitly run above. Final task-record hook output is `/private/tmp/knowledge-final-fix-precommit-final.log`. `git diff --check` passed. A final formatting-only cleanup restored unrelated receiver body wrapping; no semantic change followed covering/type checks.

## Scoped files, ownership and remaining gates

1. `apps/packages/ui/src/components/Review/MediaKnowledgeActions.tsx`
2. `apps/packages/ui/src/components/Review/__tests__/MediaKnowledgeActions.test.tsx`
3. `apps/packages/ui/src/utils/knowledge-note-provenance.ts`
4. `apps/packages/ui/src/utils/research-workspace-prefill.ts`
5. `apps/packages/ui/src/utils/use-research-workspace-prefill.ts`
6. `apps/packages/ui/src/utils/__tests__/knowledge-note-provenance.test.ts`
7. `apps/packages/ui/src/utils/__tests__/research-workspace-import.test.tsx`
8. `apps/packages/ui/src/services/tldw/service-prompt-scope-error.ts`
9. `apps/packages/ui/src/services/__tests__/background-proxy.web-clipper-scope.test.ts`
10. `tldw_Server_API/app/api/v1/endpoints/web_clipper.py`
11. `tldw_Server_API/tests/Notes_NEW/integration/test_web_clipper_api.py`
12. `backlog/tasks/task-13453.2 - Preserve-research-evidence-and-complete-saved-output-review.md`
13. `backlog/tasks/task-13453.4 - Complete-Knowledge-readiness-recovery-and-cross-surface-continuity.md`

Task2 implementation is Done. Task4 remains In Progress for the controller's fresh native Save→canonical Notes UUID→full-options Ask acceptance. No unresolved product concern was found in self-review. Root owns scoped independent review, real Research/stored-web/native browser checks, final builds/full suites, docs-only dev refresh, design/plan/review assets, and existing dependency/build outputs including Documents/. No private runtime descriptors/logs were read; no original-checkout product edits, dependency/runtime changes, fetch/rebase, push, merge, or subagents. Temporary verification artifacts are in `/private/tmp`; this report is in the existing ignored SDD folder.

## Final commit and handoff

Committed the exact thirteen scoped files above as `5a92c47a7ee4a08a7af7872d54656128424a2a7a` (`fix(knowledge): preserve continuation contracts and scoped clip saves`). Explicit final pre-commit output `/private/tmp/knowledge-final-fix-precommit-final.log` passed all applicable checks; no hook bypass was used. Post-commit status contains only the controller-owned design/plan changes, review assets, dependency symlink and build output listed above. Git emitted an existing automatic-gc/unreachable-object maintenance warning; no cleanup was attempted. No implementation concern remains; final independent review, live/native acceptance and whole-branch builds remain controller-owned.
