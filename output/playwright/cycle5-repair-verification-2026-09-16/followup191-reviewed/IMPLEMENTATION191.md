# UAT191 / TASK13260.129 — frozen review handoff

**Scheduled Cram receives the active scheduler's interval previews through its existing card-list request.** The scheduler algorithm and ReviewTab are unchanged. [Design](DESIGN191.md), [causal RED](RED191.md), [owned patch](owned.patch), [source/test hashes](owned-manifest.json), and exact review snapshots are retained in this packet.

## Change

- `endpoints/flashcards.py` adds an optional `include_scheduler_preview=false` query argument. Opted-in rows use the same `_attach_scheduler_preview` helper as due-review. A response-local deck cache avoids repeated reads for the same deck. Default lists, filter arguments, row order, pagination/count/total and authorization dependency remain unchanged. Scheduler settings errors return the existing400 contract; DB errors retain500.
- `services/flashcards.ts` accepts and serializes the optional flag. `useFlashcardQueries.ts` sets it on the existing Cram list request. The query key, pagination loop, queue ownership and request count remain unchanged; no extra endpoint is called.
- The same Cram hook also supplies practice/availability data; it now returns previews there as well. Schedule-off behavior still performs no rating or session-end mutation. A configured scheduler error can fail this opted-in Cram request, while ordinary Manage/list requests remain independent of scheduler validity.

The native mismatch was a new-card generic1day fallback versus the correct SM-2+10minute result for Good3. The saved response's subsequent next_intervals.good=1day describes the next review of the now-learning card and is valid. No scheduler change was needed.

## RED → GREEN

Permanent real-router SQLite/official-PG RED:12 preview-null failures/4 ordinary-list controls. Additional opt-in errors/mixed-deck/pagination/cache RED:8 failures/2 empty-page controls. New mounted ReviewTab→real Cram hook→real service/URL serializer RED:4 failures/2 controls. Earlier fixture/setup issues, including separately tracked192 PostgreSQL tag mutation, are documented in RED191.md and preserved, not counted as191 reproductions.

**Final backend:58 passed,0 skipped** across three bounded runs:

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat191-route-green node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/Flashcards/test_cram_scheduler_preview_contract.py -q --tb=short
TLDW_UAT_EVIDENCE_LABEL=uat191-adjacent-green node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py tldw_Server_API/tests/Flashcards/test_scheduler_fsrs.py tldw_Server_API/tests/Flashcards/test_review_response_timestamp_contract.py -k 'review_next or fsrs_review or list_flashcard or flashcards_list or review_response or review_http' -q --tb=short
TLDW_UAT_EVIDENCE_LABEL=uat191-scheduler-adjacent node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/Flashcards/test_scheduler_fsrs.py tldw_Server_API/tests/Flashcards/test_flashcards_scheduler_schema.py tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py::test_get_next_review_card_returns_400_for_input_error tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py::test_get_next_review_card_returns_500_for_db_error -q --tb=short
```

Results26/20/12 pass respectively; the middle focused run deselects183 unrelated tests, not skips. Redacted logs are `../fresh-uat-recovery-20260916/uat191-{route-green,adjacent-green,scheduler-adjacent}.redacted.log`. The helper uses the official fixture cluster with required PG and no Docker autostart; escalation permits that test connection only. It does not touch native UAT data. New route tests cover all four rating values and committed gaps, customSM2 and FSRS, no mutation from preview reads, default omitted/false behavior, mixed-deck/page identity, one read per deck, settings400/DB500, and empty pages.

**Final frontend:39 passed/4 files/0 skipped**, default frontend config,3.12s. From `apps/tldw-frontend`:

```sh
bunx vitest run ../packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.cram-previews.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.cram-mode.test.tsx ../packages/ui/src/components/Flashcards/hooks/__tests__/useFlashcardQueries.cram-queue.test.tsx ../packages/ui/src/services/__tests__/flashcards.test.ts
```

[Log](frontend-final.log). The new six include returned interval labels, actual serialized opt-in, one request, all four mutation rating values, supplied-preview precedence and no practice-only mutation. The backend response is simulated only at the transport boundary; real Python HTTP tests separately prove that response/schedule contract. The first GREEN exposed an unreached test assertion using wire `card_uuid` at the hook boundary; it was corrected to the existing `cardUuid` hook field. No production workaround or weakened preview expectation was added. Existing20 Cram tests include separately frozen189/190 fixture updates owned by the other author.

## Static checks

- Ruff endpoint and new Python test:0 findings. Python test formatter check passes. Existing endpoint formatting retained.
- ESLint three owned TS/TSX paths:0 errors,2 unchanged service `no-explicit-any` warnings. Baseline checks use the actual logical filename via stdin; suggestion byte ranges shift with inserted lines, while rule/severity/message remain identical. [Comparison](lint-comparison.json).
- Full frontend `tsc --noEmit --incremental false`:exit2 with90 existing diagnostics, zero in owned files. All normalized error headers match the retained164 baseline's90 diagnostics. [Comparison](typecheck-comparison.json) names that retained source; this is not a clean application compiler result or a newly reconstructed baseline.
- Bandit endpoint:0 findings/errors. Python test:0 findings/errors with B101 excluded only for pytest assertions. Bandit cannot parse the three TS/TSX files; [report](bandit-typescript.json) retains all three parse errors and makes no TS security-pass claim.
- Owned patch has no added trailing whitespace; snapshots match all five owned hashes. No app/router/schema regeneration, browser, runtime/configuration, task/tracker or git mutations were performed by this author. Node localStorage/Browserslist test warnings are retained; no installs were performed.

## Remaining gates

Independent source/test review and native disposable-card preview/rating agreement remain root-owned. Root's D card is reset to version4; this author has not rated or altered it. Historical transcript and diagnostic artifacts are retained; the scheduler fixture intentionally excludes the separate192 tag setter defect rather than repairing another owner's ChaCha scope.
