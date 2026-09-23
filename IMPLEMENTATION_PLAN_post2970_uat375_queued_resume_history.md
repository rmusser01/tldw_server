# UAT375 queued conversation resume implementation plan

**Task:** TASK13260.277.25. Root approved the bounded design on 2026-09-20.
**Goal:** Resume an owned queued turn after an interrupted Character request without replaying canonical history or dispatching before conversation identity is known.
**Architecture:** Preserve canonical database rows and remove only a proven historical request prefix containing an assistant. Fence sidepanel metadata hydration with the existing verified request lease and lifecycle; hold both automatic and manual queued sends until hydration succeeds.
**Tech stack:** Python/FastAPI/CharactersRAGDB, SQLite and official PostgreSQL fixtures; React/Zustand/Vitest.
**Constraints:** No partial assistant reconstruction, arbitrary request-subsequence adoption, system-only matching, broad queue refactor, dependencies, runtime changes, frozen packet changes, global tracker edits, or git mutations. Do not edit useDraftPersistence or the UAT373 TldwApiClient work. Keep UAT374 Sidebar frozen. Root owns native acceptance and commits.

## Stage 1: Canonical historical prefix regression and repair
**Goal:** Reproduce the native five-to-ten-row preparation and save only the new queued user.
**Success Criteria:** Original row IDs and interrupted turn survive; final queued user/client ID survive even with deliberately repeated text. The provider context retains canonical gap rows. Changed instructions and unmatched prefixes are retained, insufficient windows are not guessed, and existing explicit Retry controls remain intact.
**Tests:** New real-DB test_chat_queued_resume_history.py on SQLite and official PostgreSQL: chronological/reverse windows, trimmed client prefix, repeated final user, changed system, non-prefix coincidence, system-only match, no evidence, and foreign owner. Existing retry system/image and operation lifecycle suites.
**Files:** tldw_Server_API/app/core/Chat/chat_service.py; tldw_Server_API/tests/DB_Management/test_chat_queued_resume_history.py.
**Status:** Complete
- [x] Write and run causal SQLite failures before production changes: six causal failures and ten controls after correcting two fixture assumptions.
- [x] Add the smallest bounded overlap check and verify the controls: thirty SQLite tests passed with the existing Retry suite.
- [x] Run the same tests with official pg_database_config against isolated fixture databases on the existing cluster: sixty SQLite/PostgreSQL tests passed, including thirty PostgreSQL cases using the official temporary databases on 127.0.0.1:55475. Expanded final controls remain part of Stage 3.

## Stage 2: Conversation identity readiness before queued dispatch
**Goal:** Restore authoritative assistant identity before queue replay; retain queued data on metadata failure or cancellation.
**Success Criteria:** Held metadata blocks auto/manual dispatch; success routes using the saved Character/persona; failure has concise retryable status; new conversations remain usable. Unmount, conversation replacement, account changes, and held asynchronous preflight cannot publish or send stale state.
**Tests:** Real Zustand metadata/queue harness with held scoped reads, failure/retry, plain/Character/persona controls, explicit Run Next/Now, unmount/account/conversation replacement, and delayed preparation. Existing sidepanel queue, useMessage service-prompt, account/privacy and route-ownership suites.
**Files:** apps/packages/ui/src/hooks/useMessage.tsx; hooks/chat/useSidepanelChatMetadata.ts and its real-store behavioral suite; hooks/__tests__/useMessage.service-prompts.test.tsx; components/Sidepanel/Chat/form.tsx; components/Common/ChatQueuePanel.tsx and its DOM suite; hooks/chat/useQueuedRequests.ts and components/Chat/composer/hooks/useComposerQueue.ts (optional completion guard only).
**Status:** Complete
- [x] Original useMessage metadata effect reproduced four failures (unmount, account replacement, conversation replacement, silent failure), then passed all twenty-four focused tests.
- [x] Reused serverChatMetaLoaded/loadState/error, verified scope leases, and existing guarded assistant selection in the bounded extracted hook. Fresh metadata reads and presentation requests carry the captured owner.
- [x] Three additional causal failures covered late model preflight, manual button availability, and a queue retained after conversation replacement; all forty-one focused cases then passed.
- [x] Two real-store remount failures proved that late success and failure callbacks could mutate the same restored queue ID. An optional queue completion guard now blocks both; fifty-four focused tests passed, including new-chat promotion and unchanged callers.
- [x] Gate auto/manual replay and show waiting/failure with retry, keeping queued payload/client identity intact. Partial Character output is never copied into canonical history.

## Stage 3: Verification and independent review handoff
**Goal:** Freeze reviewable source with exact receipts for root native acceptance.
**Success Criteria:** Causal and neighboring tests pass; no added scoped lint/type findings; touched Python Bandit clean; exact manifest and red/green evidence provided. Task remains In Progress until root completes native acceptance.
**Tests:** Scoped pytest/Vitest; matched Ruff/mypy and frontend lint/types; virtualenv Bandit on touched Python production scope; whitespace check.
**Status:** In Progress
- [x] Record SQLite and official PostgreSQL provenance without credentials.
- [ ] Complete final checks and update this plan and task through official CLI.
- [ ] Send root exact manifest, limits, and native queue-resume/readback recipe for independent review.

### Frozen review receipts

- Source/test manifest: `/private/tmp/uat375/owned-files.txt` (13 files including this plan and task); exact scoped patch `/private/tmp/uat375/review.diff`; eleven source/test hashes `/private/tmp/uat375/source-sha256.json`.
- Backend causal evidence: `/private/tmp/uat375/backend-causal-red.log` (6 failures/10 controls), `/private/tmp/uat375/backend-green.log` (30 SQLite passes), `/private/tmp/uat375/backend-postgres.log` (60 combined SQLite/official PostgreSQL passes). Final expanded run: `/private/tmp/uat375/backend-final.log`.
- Frontend causal evidence: `metadata-red.log`, `metadata-green.log`, `queue-preflight-red.log`, `queue-preflight-green.log`, `queue-remount-red.log`, `queue-remount-green.log` under `/private/tmp/uat375/`. Final `/private/tmp/uat375/frontend-final.log`: 14 suites, 154 passes, no unhandled errors.
- Matched frontend compiler: 93 baseline/current diagnostics, no additions/removals/touched diagnostics (`type-comparison.json`). ESLint: zero errors, 80 baseline/70 current warnings, no new normalized diagnostics (`lint-comparison.json`).
- Python Ruff: zero findings (`ruff-final.json`). Mypy: 83 baseline/current diagnostics, no additions/removals (`mypy-comparison.json`). Bandit: zero production findings; raw tests report 21 expected pytest assertion B101 findings, zero non-assertion findings (`bandit-production.json`, `bandit-all.json`, `bandit-tests-no-assert.json`). TypeScript files are outside Bandit's scope.
- Root received the frozen source for independent review. Keep task In Progress until root performs native SQLite and PostgreSQL extension acceptance against the next private packaged candidate.

### Native acceptance after review

Use a fresh root-owned private candidate and existing documented packaged extension procedure; do not reuse or modify the frozen failing evidence. With the same verified Alice owner, create a Character conversation with a complete first turn, start a long Character stream, queue a unique second request, switch to Bob, and return to Alice. Hold or observe saved conversation metadata restoration: the queue must remain visible and idle until identity is ready. Then it must use the restored Character transport. Verify canonical rows equal the original four-row prefix, the interrupted user, one queued user, and its reply (seven rows for the original five-row interrupted starting point), with original row IDs retained. Verify current queued client identity and parent linkage; there must be no duplicated prefix or Bob content. Repeat using separate real SQLite/PostgreSQL profiles. Inject or use a failed metadata read to confirm truthful retry status and unchanged queued payload before retry; no additional LLM call is needed merely to establish the failure UI.

### Root continuation after candidate11

Candidate12 is built and running on fresh SQLite18809 and official PostgreSQL18811: 25,045 source entries match, 0 mismatches, 88 frozen tests pass. SQLite native019 established saved Character seed chat4f0f7a1b-fd07-405c-a911-4d76f26bf694 with HTTP200/saved=true and model reply8. The PostgreSQL model-selection action did not execute because automatic approval review hit the weekly Codex usage limit. Interruption, reciprocal account return, metadata failure/retry, and canonical queue readback remain pending on both databases. Native actions must wait for approval-review availability; no database acceptance is waived. The separately tracked UAT388 sidebar model-validation repair can proceed locally.

UAT385/386/387 native acceptance is complete. Before candidate12 first includes375, the nine temporary real-store/message-wrapper ownership probes are now retained in the existing legacy caller suite. Opt-in real fixtures keep earlier assertions unchanged. All six queue/metadata/caller suites pass88 tests. Independent mutation checks fail9/9 when owned history publication is bypassed and1/9 when owned server linkage is bypassed. Initial typed adoption exposed an incomplete mock completion payload/return; it now uses the full typed payload and submitted result. Final matched UI TypeScript426→426 and lint0errors/10warnings add no findings; two existing Promise test diagnostics are unchanged. Independent11-file overlay review is accepted. Candidate12 freezes126overlays and is approved for build/startup; fresh archive preparation is underway and native acceptance remains pending. Prior candidates remain immutable.
