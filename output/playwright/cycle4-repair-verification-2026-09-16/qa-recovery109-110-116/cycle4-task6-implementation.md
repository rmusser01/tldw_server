# Cycle4 Task6 implementation — UAT109/110/116

Tasks TASK13260.50 and TASK13260.56 remain In Progress for independent review and native acceptance. All edits frozen/released to parent. No commit, staging, browser, live inference, runtime/config change or subagents.

## Diagnosis and bounded repair
- Direct background-proxy reader cancellation returned a rejected promise after native body abort. The existing synchronous catch did not own it. Await/catch cleanup now retains the primary timeout/caller-abort result and consumes cleanup rejection.
- QA and Analysis outer generation catches already contained errors but emitted console.error, intercepted as a Next development overlay. They now emit bounded warning codes and retain local error feedback/retry, cancellation, save and source handling.
- QA completedGenerationEnabled stores the completed request's boolean independently of current controls. Existing saved settings snapshots now persist/normalize that boolean; stored legacy results without it remain unknown. Live complete, thread/share/branch hydration and clear-results cover the new state.
- AnswerPanel distinguishes requested-but-empty output, deliberately disabled generation, and unknown legacy settings, retaining prior explicit insufficient-evidence and transport-error branches. Retry and generation-settings actions remain available; sources are retained.
- AnswerWorkspace resets assertive error text when the error clears, including retry and subsequent success.

## Permanent RED / GREEN evidence
- /private/tmp/cycle4-task6-red-ui.log: authoritative pre-fix RED, 10 failed / 200 passed across 5 files using apps/packages/ui Vitest config. Captures cleanup unhandled rejection, console.error interception boundary, completed-request semantics, misleading panel guidance and stale assertive announcement.
- /private/tmp/cycle4-task6-analysis-red-corrected.log: 2 Analysis failure handlers RED against expected console-error assertion after correcting test wording to existing timed-out copy.
- Added saved-result true/false/unknown hydration tests exposed generation boolean omission; the true/false cases were RED (211 passed / 2 failed) before adding the existing snapshot whitelist field. That intermediate log was superseded at green-final path; do not claim a separate retained RED log for this follow-up.
- /private/tmp/cycle4-task6-green-final.log: 213 passed / 5 files. Final broader run below also includes added history snapshot persistence assertion.
- /private/tmp/cycle4-task6-broader-final.log: 359 passed / 18 files, exit 0.

Exact broader command (cwd apps/packages/ui):
```sh
./node_modules/.bin/vitest run src/services/__tests__/background-proxy.test.ts src/services/__tests__/background-proxy.web-refresh.test.ts src/services/__tests__/background-proxy.monitoring-scope.test.ts src/components/Media/__tests__/AnalysisModal.stage3.regression.test.tsx src/components/Media/__tests__/AnalysisModal.stage1.cancel.test.tsx src/components/Media/__tests__/AnalysisModal.model-owner.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider*.test.tsx src/components/Option/KnowledgeQA/__tests__/AnswerPanel.states.test.tsx src/components/Option/KnowledgeQA/__tests__/AnswerWorkspace.a11y.test.tsx
```

Controls include native ReadableStream/Response error and cancellation, actual QA provider/Analysis handlers, Analysis cancel/model ownership, auth authority/isolation, QA follow-up/history/persistence/share, explicit insufficient evidence, normal/cited answers, proxy refresh and monitoring scope. HTTP/provider responses are mocked; these are component/service regressions, not real-model or browser acceptance. Analysis failures assert settled handler, no close/generated callback or version persistence, then a successful retry.

## Static checks
- Final exact-owned-scope ESLint exit 0: 0 errors / 120 existing warnings, identical file + rule + severity + message count signatures to pre-edit baseline. This is no new diagnostics, not lint-clean.
- Baseline /private/tmp/cycle4-task6-eslint-baseline.json; final /private/tmp/cycle4-task6-eslint-final.json; comparison /private/tmp/cycle4-task6-eslint-comparison.json.
- Command from repository root: ./apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs <all 11 manifest paths> --format json.
- Scoped git diff --check exit 0. No Python changed; Bandit not applicable to TypeScript-only unit. No full tsc requested/run.

## Harness distinction / limits
An initial run with apps/tldw-frontend's Vitest config produced 7 existing unrelated proxy warning-spy isolation failures, confirmed absent under the UI package's own config. /private/tmp/cycle4-task6-red.log is that nonauthoritative run; /private/tmp/cycle4-task6-proxy-baseline-check.log confirms existing 15 targeted controls pass with UI config. No unrelated tests or config changed.
Native Next overlay absence and real provider recovery require parent's coordinated verification. Generation intent was not historically saved, so old snapshots cannot be reconstructed and correctly use neutral guidance. No model/backend behavior or policy changed. Analysis persisted-source preservation is checked at the component callback/write boundary; no database or live browser exercised.

## Owned source/test manifest
- apps/packages/ui/src/services/background-proxy.ts
- apps/packages/ui/src/services/__tests__/background-proxy.test.ts
- apps/packages/ui/src/components/Media/AnalysisModal.tsx
- apps/packages/ui/src/components/Media/__tests__/AnalysisModal.stage3.regression.test.tsx
- apps/packages/ui/src/components/Option/KnowledgeQA/KnowledgeQAProvider.tsx
- apps/packages/ui/src/components/Option/KnowledgeQA/AnswerPanel.tsx
- apps/packages/ui/src/components/Option/KnowledgeQA/types.ts
- apps/packages/ui/src/components/Option/KnowledgeQA/panels/AnswerWorkspace.tsx
- apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.streaming.test.tsx
- apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/AnswerPanel.states.test.tsx
- apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/AnswerWorkspace.a11y.test.tsx

Official Backlog records for .50/.56 also updated via MCP; parent owns plan/tracker/commit.
