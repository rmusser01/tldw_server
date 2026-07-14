---
id: TASK-12113
title: Implement shared WebUI and extension per-video playlist ingestion
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-14 06:27'
labels:
  - webui
  - browser-extension
  - media-ingestion
  - implementation
dependencies: []
references:
  - TASK-12109
  - TASK-12111
  - TASK-12112
  - Docs/superpowers/specs/2026-07-12-youtube-playlist-per-item-ingest-design.md
documentation:
  - Docs/superpowers/plans/2026-07-12-youtube-playlist-ingest-shared-frontend.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved shared-frontend implementation plan after the backend version-2 contract is stable: mandatory playlist inspection, full virtualized preview, occurrence materialization, Review overrides, shared run/status transport, lifecycle UI, IndexedDB recovery, and WebUI/browser-extension parity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Complete all nine tasks and five stages in the approved shared-frontend plan using test-first red/green/refactor cycles.
- [ ] #2 Route every playlist entry path through mandatory fail-closed inspection and show every selected occurrence with bounded pagination and virtualization.
- [ ] #3 Use one shared WebUI/extension run controller and occurrence-aware queue/status/results model, with durable IndexedDB recovery and visible failure states.
- [ ] #4 Pass focused Vitest suites, TypeScript/lint gates, deterministic Playwright browser journeys, accessibility checks, and extension parity tests.
- [ ] #5 Complete per-task specification and code-quality reviews, then a final implementation review; record verification and final summary.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Begin only after TASK-12112 stabilizes the backend contract. Follow Docs/superpowers/plans/2026-07-12-youtube-playlist-ingest-shared-frontend.md sequentially, reusing the shared Quick Ingest package, TanStack Virtual, Dexie, and existing test infrastructure without new dependencies.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Backend dependency TASK-12112 is complete at commit bc20c306d2. Frontend execution began with Impeccable product-context preflight passed; the shared Quick Ingest design remains restrained, state-literate, accessible, and visually consistent across WebUI and extension. Task 1 will add the version-2 client models and truthful capability gate test-first.

Task 1 contract client and capability gate completed test-first. Initial RED: 11 failed / 49 passed; quality-remediation RED: 8 failed / 68 passed. Final focused Vitest: 76/76 passed. ESLint: exit 0 with zero errors; Prettier and git diff --check passed. Full TypeScript remains blocked by unrelated repository baseline after the three-attempt audit. Final specification re-review: compliant. Final code-quality re-review: approved. Touched scope: playlist-ingest client, media-domain methods, strict OpenAPI/capability gate, and focused tests. Bandit is not applicable to this TypeScript-only task.

Task 2 mandatory inspection controller completed test-first. It removes the direct playlist queue bypass, shares Add/Enter/extension-seed handling, keeps ordinary URLs staged while blocking proceed actions, limits concurrent v2 inspections, preserves first-page truncation and session duplicate evidence, serializes DELETE-before-replacement cleanup, uses the established 1200 ms polling cadence, retains only sanitized typed errors, and announces localized async status changes accessibly. Behavior RED: 13 failed / 7 passed; Strict Mode seed review RED: 1 failed / 25 passed; quality-remediation RED: 7 failed / 24 passed. Final focused Vitest: 31/31 passed. ESLint: exit 0 with zero errors; new-file frontend Prettier check and git diff --check passed. Final specification re-review: compliant. Final code-quality re-review: approved. Full TypeScript was not rerun after the Task 1 three-attempt baseline cap. Bandit is not applicable to this TypeScript-only task.

Task 3 complete: full immutable preflight snapshots now page sequentially by opaque cursor and publish atomically; malformed version/count/cursor responses fail closed; the 500-item panel is virtualized with stable occurrence identity, native-scroll focus recovery, position-aware accessible names, localized availability, safe duplicate/unknown evidence, batched selection, latest-queue session dedupe, and refresh reconciliation. RED cycles covered pager/panel behavior, hardening, spec findings, and quality findings. Final focused Vitest: 66/66. Repository-pinned ESLint: exit 0 except existing Next pages-directory informational output. Scoped Prettier and git diff --check: pass. Final spec review: compliant. Final quality review: approved. Full TypeScript not rerun after Task 1 baseline three-attempt cap. Bandit N/A (TypeScript only). Task 4 materialization/queue mutation remains intentionally absent.

Task 4 materialization, authoritative flat queue rows, playlist Review overrides, conference opt-in separation, atomic Start validation, virtual-list keyboard recovery, functional queue transactions, and bounded fail-closed persistence hardening completed test-first. Quality RED-to-GREEN regressions: atomic expiry (1), add/remove/clear materialization races plus updater collision (4), playlist conference opt-in (1), Review/playlist/conference virtual focus recovery (2), store reconstruction/consistency (5), and request identity/backend bounds (8). Final focused Vitest: 174/174 across five suites; store suite: 9/9 after final consistency cleanup. Repository-pinned production ESLint exited 0 with only the existing Next pages-directory informational output; git diff --check passed. Full TypeScript was not rerun due the existing Task 1 three-attempt baseline cap. Bandit N/A (TypeScript only). Task 5 shared transport remains deferred.

Task 4 quality rereview remediation completed: selected run inputs now use the shared 500-item bound with a visible Review block; persisted row 501 is retained and drafts above the 1000-source safety ceiling restore with an explicit invalid overflow sentinel; direct URL display caches derive from source authority and direct display titles are normalized and validated through the serializer helper; policy-only duplicate overrides bind to fresh server evidence while explicit stale or conflicting targets still return duplicate_target_changed; playlist candidates remain locked through passive commit verification; the Add queue uses roving virtual focus with nested-control Escape and filter/removal recovery; and Review source details sit outside truncation. Final verification after formatting cleanup: focused frontend Vitest 180/180 across five suites; backend playlist service 93/93 and endpoint 56/56; repository-pinned scoped ESLint exit 0 with zero rule findings (only the existing Next pages-directory informational message); Bandit on the touched backend service reported zero findings; git diff --check passed. Full TypeScript was not rerun under the existing Task 1 three-attempt baseline cap. No files were staged or committed.

Final Task 4 quality rereview remediation completed. Focused RED initially exposed 8 failures with 126 skips; one failure was corrected as a test-query scoping error, while the behavioral failures covered the requested blocked-flow, safe-policy, persistence-authority, stale-filter, and hydration gaps. Manual Quick and auto-process blocks now route deterministically to Review with visible processing-block state; auto-process retains its one-shot retry until Review resolves the block. Initial in-run duplicate policy choices are limited to skip/overwrite unless fresh server allowedDuplicatePolicies explicitly expands them, and request validation rejects unsupported seeded policies. Duplicate-only playlist recovery metadata no longer invalidates direct URL authority. Persisted and hydrated rows canonicalize kind/display fields from sourceRef, with file stubs always reattach-invalid and materialized URLs display-only. Removing the last playlist row clears stale queue filters. The two requested local formatting defects were corrected without a blanket formatter. Focused regression rerun: 8/8 passed. Final frontend Vitest: 188/188 across the five scoped suites. Repository-pinned scoped ESLint exited 0 with zero rule findings (only the existing Next pages-directory informational message); git diff --check passed. Backend files were unchanged in this rereview, so the immediately prior playlist service 93/93, endpoint 56/56, and zero-finding Bandit results remain applicable. Full TypeScript was not rerun under the existing Task 1 three-attempt baseline cap. No files were staged or committed.

Final release authority-cue remediation completed. Root cause: persistence sanitization and run serialization independently treated only materializationExpiresAt as evidence that a cached playlist URL required materialized source authority. Added one shared playlistHasMaterializationCues helper covering sourceUrl, playlistId, playlistTitle, ordinal, channelOrUploader, durationSeconds, normalizedSourceId, and materializationExpiresAt; both trust boundaries now fail closed when any meaningful cue survives without a materialized_playlist_item sourceRef. Duplicate-review-only direct_url rows remain valid because title and duplicateStatus are intentionally excluded. Store, modal hydration, and run-creation regressions were added, including the full eight-cue matrix and cached-URL non-submission control. Initial focused RED: 9 failed, 2 passed, 82 skipped; expiry and duplicate-only direct controls already passed while the seven non-expiry cues, store, and hydration cases exposed the gap. Focused GREEN: 11/11 passed. The first full run exposed one stale conference test fixture carrying materialization-only metadata without authority; the fixture now supplies its materialization ID and expiry. Final five-suite Vitest: 199/199 passed. Repository-pinned scoped ESLint exited 0 with zero rule findings (only the existing Next pages-directory informational message); git diff --check passed. TypeScript was not rerun under the existing Task 1 three-attempt baseline cap. Backend unchanged and Bandit not applicable to this TypeScript-only release fix. No files were staged or committed.

Final evidence-none Review recovery remediation completed test-first. Root cause: APPLY_PLAYLIST_REVIEW_REQUIRED correctly merged duplicate_no_longer_present/evidence kind none into a nonduplicate queue row, but then unconditionally latched a review_required processingBlock. Review requires processingBlock null, while the refreshed nonduplicate row exposes no duplicate controls, creating a dead end. Added a focused reducer regression that verifies fresh none evidence changes duplicateStatus to new, removes the obsolete policy, clears the block, and allows the materialized row to start normally. Focused RED: 1 failed / 53 skipped at the stale processingBlock assertion. Minimal fix computes the merged queue once and derives processingBlock from buildPlaylistIngestRunRequest on that fresh queue, preserving genuine duplicate, expiry, and invalid-request blocks. Focused GREEN control run: 2/2 passed, covering both evidence-none and existing duplicate-target-changed recovery. Final five-suite Task 4 Vitest: 200/200 passed. Repository-pinned production ESLint exited 0 with zero rule findings (only the existing Next pages-directory informational message); git diff --check passed. Files changed for this finding: apps/packages/ui/src/components/Common/QuickIngest/IngestWizardContext.tsx and its IngestWizardContext.test.tsx. TypeScript was not rerun under the existing Task 1 three-attempt baseline cap. Backend unchanged; Bandit not applicable to this TypeScript-only fix. No files were staged or committed.

Task 5 shared run client and WebUI delegate implemented through RED/GREEN. Added occurrence-only poll/SSE merge with full resync, authoritative bounded submissions, aligned URL/file arrays, same-attempt ambiguous retry, structured 207 handling, run-first cancel/retry/reattach, mixed-media field grouping, terminal no-job result mapping, and Review-required recovery. WizardModal passes the exact pending run request, never submits materialized cached URLs, keeps accepted jobs nonterminal until run status resolves, and surfaces globally stopped/rate-limited submission instead of false completion. Extension upload responses now preserve only sanitized Retry-After metadata through the background proxy. Expanded verification: 214/214 Vitest across nine suites; production ESLint exit 0 with zero errors and 149 existing legacy warnings; git diff --check passed. Full TypeScript remains skipped under the Task 1 three-attempt repository-baseline cap. Bandit N/A for TypeScript-only changes. Final reviews and root commit remain pending.

Supplemental hardening bounds persisted runId values to the backend 255-character identity limit and drops oversized values before session storage. Final fresh expanded gate after this change: 215/215 Vitest across nine suites; repository-pinned production ESLint --quiet and git diff --check both exit 0.

Task 5 reviewer remediation completed. The version-2 create request now carries normalized processing options, conference new-collection data, and per-occurrence conference metadata before run creation while excluding cached materialized URLs. Late-chunk rate limits preserve accepted-job monitoring, expose only unsent occurrence failures, and request cancellation for those unsent rows. Run cancellation reports non-legacy failures instead of silently succeeding; fallback is limited to unsupported old-server statuses. SSE reattachment reloads version-advanced items and skips the unchanged initial snapshot. Focused remediation gate: 102/102 across four suites. Final expanded gate: 222/222 across nine suites. Repository-pinned focused ESLint --quiet and git diff --check both exited 0. Full TypeScript remains skipped under the Task 1 three-attempt repository-baseline cap; Bandit is not applicable to TypeScript-only changes. Final code re-review and root commit remain pending; no files were staged or committed.

Final Task 5 cleanup-failure remediation completed test-first. Focused RED: 2 failed / 70 skipped, proving both the swallowed service cleanup failure and the Modal reattach race. The service now reports submissionCleanupFailed with an explicit recovery message when unsent occurrence cancellation is unconfirmed. The Modal defers initial reattachment until submit acknowledgement settles, records only unsent failures, preserves accepted work and run tracking, and persists an interrupted/error recovery state rather than polling staged rows forever. Targeted GREEN: 2/2. Full focused run/delegate gate: 104/104. Final expanded nine-suite gate: 224/224. Repository-pinned focused ESLint --quiet and git diff --check exited 0. Final reviewer re-check remains pending; no files were staged or committed.

Final Task 5 race remediation replaced the zero-delay timing heuristic with explicit run-submission acknowledgement state. Run-ID reattachment is gated while the version-2 submit promise is pending; a cleanup-failed acknowledgement keeps the gate closed and persists interrupted recovery, while an ordinary accepted acknowledgement enables reattachment. The strengthened Modal regression publishes runId, leaves submit pending past a timer turn, proves zero reattach, then resolves cleanup failure and proves interrupted/no reattach; the accepted-run control still proves post-ack reattachment. Targeted controls: 2/2. Full focused gate: 104/104. Expanded gate: 224/224. Focused ESLint --quiet and git diff --check pass. Same reviewer re-check pending; no files staged or committed.

Task 5 final code re-review: APPROVE / ready. Reviewer confirmed the explicit in-flight gate prevents unresolved-submission reattach; cleanup failure keeps interrupted/error state with runId/tracking, records only unsent failures, and does not terminalize accepted work; ordinary accepted acknowledgement re-enables reattachment. Root commit remains pending; no files staged or committed by this task agent.

Formal Task 5 blocker remediation completed test-first; final Task 5 approval and Step 5 remain open pending both requested formal re-reviews. Durable submission state is persisted before create, runId before upload, and cumulative batch/job mappings after each chunk; reload preserves version-2 materialized authority. Omitted occurrences now block and are cancelled without losing accepted work. Run item paging is bounded to 500 items and 4096-character cursors; stale SSE replay cannot regress state/progress. Run reattach falls back to legacy only for 404/405/501, keeps 429/503 retryable, and surfaces 401/403 authorization recovery. First-chunk cleanup failure stays interrupted with runId and no false terminalization. Cancellation during create and between chunks stops later uploads and cancels the server run. RED: 17 failed / 118 passed across five files. GREEN: 135/135. Expanded nine-file Task 5 gate: 241/241. Repository-pinned scoped ESLint --quiet and git diff --check exit 0 (existing Next pages-directory informational output only). Full TypeScript remains skipped under the Task 1 three-attempt baseline cap; Bandit N/A for TypeScript-only changes. No files staged or committed.

Task 5 second formal-review remediation completed test-first; Step 5 and final approval remain open pending both final re-reviews. Restored creating_run tracking now fails closed as interrupted and cannot reconstruct or restart a request, including after materialization expiry. Dedicated sanitized submissionOccurrenceIds preserves at most 500 occurrence identities without overloading collection planning metadata. Reload of run_created/submitting tracking cancels only authoritative unsent states (staged, awaiting_upload, submit_pending) and repolls; accepted/running work remains attached, while cleanup 503 remains retryable without false terminalization. Unknown-boundary SSE occurrence events reload authoritative summary/items so retained same- or higher-state events cannot regress metadata. Second formal RED: 8 failed / 132 passed across five files; final focused gate: 141/141; expanded nine-file gate: 247/247; repository-pinned scoped ESLint --quiet and git diff --check exit 0 with only the existing Next pages-directory informational output. Full TypeScript remains skipped under the Task 1 three-attempt baseline cap; Bandit N/A for TypeScript-only changes. No files staged or committed.

Task 5 bounded-transport remediation completed test-first; both final re-reviews remain pending. The 500-item unknown-cursor RED observed 1002 authoritative REST requests instead of the expected two because every retained occurrence event reloaded the complete run. Reattachment now returns the complete authoritative poll snapshot whenever no trustworthy event high-water mark exists and does not open SSE in that state; the stale-terminal control confirms retained events cannot weaken authoritative state. Final focused five-file gate: 142/142. Expanded nine-file gate: 248/248. Repository-pinned scoped ESLint --quiet exits 0 with only the existing Next pages-directory informational output. Full TypeScript remains skipped under the Task 1 three-attempt baseline cap; Bandit N/A for TypeScript-only changes. No files staged or committed.

Task 5 final formal specification/code-quality re-reviews approved with no actionable findings. Reviewers independently confirmed restored-create fail-closed recovery, bounded dedicated submission occurrence tracking, authoritative unsent-only cleanup with retryable cancellation failure, bounded unknown-cursor polling, stale-replay safety, and cursor-backed SSE correctness. One reviewer verified focused 142/142 and expanded 248/248; the other independently ran nine-file verification at 254/254. Both reported scoped ESLint --quiet and git diff --check clean. Step 5 remains open for the root-owned commit; no files staged or committed by this task agent.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
