# Independent review: cycle4 Task6 (UAT109/110/116)

## Verdict
Review-clear in the frozen 11-path package. No actionable finding in the changed behavior. Reviewed the design's async-failure/generation-guidance contracts and adjacent stream, error mapping, settings snapshot, hydration, and ownership boundaries. No source/test edits or runtime/browser/inference actions.

## Behavior review
- `background-proxy.ts:1834` now awaits reader cancellation within its existing catch-protected cleanup. A native errored body can reject both `read()` and `cancel()`; the primary idle-timeout or caller-abort result survives, while cancellation's rejected promise is consumed. The idle timer is cleared and the caller abort listener removed. Existing transport/fallback policy is unchanged.
- The QA and Analysis outer generation catches retain their local error/retry behavior but emit bounded warning codes. QA uses the existing validated public error/status mapping; Analysis emits only timeout/provider_failure. Neither changed diagnostic forwards raw upstream errors. Cancellation remains separately handled. This is a review of those owned failure paths, not a claim that all unrelated diagnostics in these large files are sanitized.
- `completedGenerationEnabled` comes from the effective settings captured before the request, not subsequently edited controls. Results clearing resets it; the existing settings snapshot whitelist persists and normalizes the boolean, and thread/share/branch result hydration restores it. Missing legacy settings remain null. Existing owner/stale-request checks are preserved.
- Missing-answer guidance distinguishes explicitly disabled generation, requested-but-empty output, and unknown legacy intent. Existing transport-error and insufficient-evidence branches retain priority. Sources are not removed by presentation, and retry/settings actions remain available. Retry uses the current controls, allowing a changed answer model to take effect.
- The assertive announcement is cleared when the owned error clears. Analysis failure tests verify no generated/close callback or version write, then a successful retry; source/previous analysis persistence ownership is not changed.

## Independent verification
From `apps/packages/ui`, ran five changed behavior suites plus the separately reviewed five Review-pane suites, using the UI package Vitest config and one worker:
- background-proxy.test.ts
- AnalysisModal.stage3.regression.test.tsx
- KnowledgeQAProvider.streaming.test.tsx
- AnswerPanel.states.test.tsx
- AnswerWorkspace.a11y.test.tsx

Task6 portion: **213 passed / 5 files**. Combined run: **246 passed / 10 files**, exit 0, 11.52s. Log `/private/tmp/cycle4-task6-060-independent-tests.log`. No unhandled-error report.

Meaningful regressions use actual ReadableStream/Response rejection plus idle/caller cancellation, actual provider/component handlers with controlled HTTP/provider transport, a delayed request whose controls change while pending, saved true/false/unknown intent, sentinel diagnostics, local recovery, and stale announcement clearing. Panel-only source retention assertions are weaker than persistence tests, but provider/Analysis controls independently exercise their actual state/callback boundaries.

The author's broader 359/18 result, historical RED and exact lint baseline comparison were read as author evidence rather than independently claimed. No Python changed; Bandit is inapplicable to this TypeScript-only review.

## Limits
Mocks stand in for remote responses and inference. Actual Next overlay absence, native timeout/cancel recovery and actual database-preservation acceptance remain for the coordinated browser pass. Legacy generation settings cannot be reconstructed. This patch does not change provider generation semantics.

### Additional focused preservation check
To verify the adjacent cancellation and saved-result ownership contracts, separately ran AnalysisModal.stage1.cancel, KnowledgeQAProvider.persistence, KnowledgeQAProvider.branch-share and KnowledgeQAProvider.authority under the same package config: **35 passed / 4 files**, exit0 in4.50s. Log `/private/tmp/cycle4-task6-independent-preservation-tests.log`. Combined independently verified Task6 coverage is **248 passed / 9 distinct files**. This does not replace native acceptance.
