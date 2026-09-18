# H1 selected history and fork ownership: verification record

Status: **In progress. H1 is not qualified for release.** This record separates reviewed implementation evidence from the real-browser and owner checks that remain. Counts from different runs overlap and must not be added together.

## Scope and source

- Design: [H1 selected history and fork ownership](../Design/2026-09-16-chatbook-h1-history-selection-design.md).
- Execution plan: [H1 implementation plan](../../IMPLEMENTATION_PLAN_chatbook_h1_history_selection.md).
- Decisions: [chronological implementation decision history](CHATBOOK_H1_HISTORY_SELECTION_DECISIONS_2026_09_17.md), including superseded rulings and stated tradeoffs; this is not additional acceptance evidence.
- Backlog: TASK-13261.1, In Progress.
- Branch: `codex/chatbook-h1-history-selection`; isolated worktree `.worktrees/chatbook-h1-history-design`.
- Implementation base: `f00e12a5aa`, based on server dev `59049e094e0845a4611ea725ae19b7c1754ea709`.
- Most recent source refresh: Chatbook dev `c97a64eba54d18f88cecc77bf6233e208df8bf24`; server dev unchanged. The second delta concerns optional installation and batch-transcription readiness, with no identified H1 contract change. [Source delta review](../Design/2026-09-17-chatbook-chat-parity-source-refresh.md).
- Unrelated main-checkout UAT changes, processes and data are outside this work.

H1 is one part of the broader parity effort. Atomic native forks/receipts (H2), complete local and temporary rich-state recovery (H3), synchronization enrollment/transfer (H4), independent model connection/cold WebUI installation (F02) and browser execution hosts (D3) are separate work. `tldw-agent` remains an external OS/tool execution agent acting for the server; it is not a Chatbook runtime or chat storage/sync layer.

## Reviewed implementation checkpoints

| Task | Reviewed result | Evidence boundary |
|---|---|---|
| 1.1 — selected-path contract | Commits `28c7d1904e`, `2fed0871c4`, `77dde3d2b4`; independent review and two fix reviews resolved identified P1/P2. | 18 focused Vitest and 17 Python tests; focused types, Ruff, compile and Bandit. Owner-specific fork exclusions remain Task 4.1. |
| 2.1 — native snapshots and legacy CAS | Commits `8a293d1ef3`, `c11a8bd9e9`; native review and fix review clean. | Final affected SQLite/PostgreSQL scope: 32 passed, including 20,001-row review and accepted projection reopen. This is owner evidence, not browser evidence. |
| 2.2 — native admission and settlement | Commits `3f8144d505`, `db1c6feddf`; all six P2 findings addressed by independent fix review. | Final owner run: 67 passed, one intentionally SQLite-only parametrization skipped; endpoint/context: 69 passed; actual saved-character factory compatibility: 1 passed. Existing Ruff/third-party warnings are recorded, not hidden. |
| 2.3 — local owner and browser adapters | Commits `f93b78c5d9`, `0bcdc845ef`; all three P1/P2 findings addressed. | Local owner, bookmark, stable identity and scoped adapter tests. Real IndexedDB commit/abort/reopen remains Task 5. |
| 3.1 — mounted selection/review/restore | Commits `9a2c54ff7f`, `5881b02870`; independent fix review clean. | Mounted view/controller tests. Workspace/Research/Document/Model panel routing is not claimed browser-qualified. |
| 3.2 — ordinary/overlay send | Commits `0e71a732d4`, `16e911c79a`, `9213012401`; all original findings and the additional mounted before-first coverage gap resolved after two fix reviews. Milestone `6e8db34b0f`. | Original run 216 tests / 22 files; fix 1: 87 / 7; fix 2: 40 / 4. These overlap. Focused types pass; two proven baseline expanded prompt-sync diagnostics remain. Provider boundaries are mocked in mounted tests. |

The implementation reports, commands, logs and review findings currently live in the plan-owned SDD workspace. The tracked decision history preserves the current controller rulings; it must be refreshed after later decisions. The final record must retain the decisive evidence and review dispositions before any workflow-artifact cleanup.

## Reviewed native-send milestone

Task 3.3 commit `4822de5d716c5dc3e2c92144624670a94f3e4489`, based on `6e8db34b0f25345248a89c7f9552a6e6488634c1`, integrates supported tracked-character sends through native versioned completion in both mounted handlers. The server owns input/result persistence; ordinary/overlay sends keep client-managed stateless inference.

Final backend run `/tmp/h1-task33-backend-final.log`: **113 passed, 1 skipped, 10 warnings, 42.43s**. The skipped `TestCreateStreamingResponseWithTimeout.test_heartbeat_integration` is pre-existing and does not count as required H1 proof. Warnings include Starlette/httpx, Pydantic schema shadow/serialization, pytest config, passlib crypt, legacy HTTP422 naming and the existing character rate limiter. The new checks cover the native endpoint acknowledgement wiring, metadata-disabled saved identity, provider-field spoofing, save failure and cancellation.

The worker reports 222 UI tests across 12 files plus 53 mounted tests after a final local-owner gate adjustment; these are overlapping runs. Focused types/static/security pass; expanded types retain two known prompt-sync diagnostics. Root verified the full report, saved test/static/security summaries and exact commit/package. Independent review found one P2: admitted native recovery could say the response was not saved even though a missing acknowledgement leaves persistence unknown. Fix `5d62386395ad1899b1236df1a48af678f0f4b7f3` makes the presentation native-aware and preserves known admission plus inspectable text. Its focused run passed 64 tests across both mounted handlers and the renderer; focused types, formatting and locale generation passed. Fresh scoped re-review marked the P2 addressed with no new important breakage. Task 3.3 and Stage 3 are complete within their stated scope; real-browser/IndexedDB and full H1 acceptance remain outstanding.

## Current native qualification

Root ran the planned native checks against code at `6e9cce6f0ab930b40432da4c69592a999afcfed3` while Task4 client work continued. No native code changed during these runs.

| Check | Result | Scope and evidence |
|---|---|---|
| Actual owner/migration | 72 passed,1 skipped,4 warnings;83.14s;exit0 | 36 SQLite and36 PostgreSQL passes, including actual-owner RLS. Existing dedicated H1 fixture at localhost55476; no UAT55475 access. `/tmp/h1-final-native-owner.log`. |
| API and regression | 277 passed,1 skipped,11 warnings;128.65s;exit0 | History contract/context/API, continuation, multi-image, system-message, provider-parameter and streaming suites. `/tmp/h1-final-native-regression.log`. |
| Production security | 0 findings,0 errors;exit0 | Bandit across all16 changed production Python paths. `/tmp/bandit_chatbook_h1.json`, `/tmp/h1-final-python-paths.json`. |
| Compile | Passed;exit0 | `python -m compileall -q` on those same16paths. |
| Ruff | 2 existing diagnostics;exit1 | F401 and B904 in character_chat_sessions.py at3896/8513. Compared exact initial-base contents using Ruff stdin; offending code is unchanged. Baseline also had two I001 findings since fixed. `/tmp/h1-final-ruff.json`, `/tmp/h1-final-ruff-baseline.json`, `/tmp/h1-final-ruff-baseline-comparison.json`. |

The owner skip is `test_history_sqlite_function_follows_real_connection_lifetime[postgres]` (SQLite-only connection lifetime), not PostgreSQL unavailability. The API skip is pre-existing `TestCreateStreamingResponseWithTimeout.test_heartbeat_integration`. Neither provides required H1 proof. Owner warnings: Starlette/httpx, pytest plugins config, Pydantic schema shadow and passlib crypt. API warnings additionally include the HTTP422 alias and admission/tool-definition fixture serialization. No pristine-lint/warning claim is made. Results overlap prior task runs.

Both pytest commands ran from the H1 worktree with the root virtualenv activated, `PYTHONPATH="$PWD"`, cleared `JOBS_DB_URL`/`TEST_DATABASE_URL`/`DATABASE_URL`, dedicated `POSTGRES_TEST_DSN` on localhost55476, `TLDW_TEST_PG_CONTAINER_NAME=tldw_postgres_chatbook_h1`, `TLDW_TEST_POSTGRES_REQUIRED=1`, `TLDW_TEST_NO_DOCKER=0`, and options `-o addopts='' -v -rs`. The owner command covered `tests/DB_Management/test_history_selection_transactions.py` and `test_history_selection_migration.py` under tldw_Server_API. The API command covered:

```text
tldw_Server_API/tests/Chat/unit/test_history_selection.py
tldw_Server_API/tests/Chat/unit/test_history_context.py
tldw_Server_API/tests/Chat_NEW/integration/test_history_selection_api.py
tldw_Server_API/tests/Chat_NEW/unit/test_chat_continuation_controls.py
tldw_Server_API/tests/Chat_NEW/integration/test_chat_continuation_controls_integration.py
tldw_Server_API/tests/Chat/unit/test_chat_history_multi_image.py
tldw_Server_API/tests/Chat/unit/test_chat_service_system_messages.py
tldw_Server_API/tests/Chat/unit/test_chat_system_message_persistence.py
tldw_Server_API/tests/Chat/unit/test_chat_service_call_params.py
tldw_Server_API/tests/Chat/unit/test_streaming_utils.py
```

These checks can be retained while their native code/dependencies remain unchanged; later affected changes require revalidation. They do not qualify browser IndexedDB, UI actions, consumer compilation or broader parity. Full commands and evidence presently also live in the plan-owned `native-owner-qualification.md` record.

## Local-fork task accepted

Task 4.1 implementation `eb5e0683e5f7d1cd19f38289d99eee6a1e05e0c1` passed 220 tests across 12 files and focused WebUI/extension dependency checks. The original extension-named check accidentally used frontend aliases; its corrected check uses the installed extension packages and passes. These are focused checks, not full application qualification.

Independent review identified three P1 defects: successful fork display did not adopt the child in the history controller; local edit/delete checked message membership without current-profile ownership; required external chat settings could be omitted from a successful copy. Focused real-controller and production-mutation probes reproduced the first two. The existing external settings reader can also return null on failure, so it cannot certify absence. Independent fix reviews now confirm all three addressed. Task 4.1 is accepted within its bounded scope; the saved-copy/open-failure notification belongs to Task4.2, and real browser qualification remains Task5.

Fix round 1 is committed as `1a5492c5426e165014e4d9438b7959877aa1b9f1`, covering 25 code/test paths. Root read the complete report and checked `/tmp/h1-task41-fix1-final-vitest.log`: **307 passed / 19 files, 7.51s, exit 0**, with no skipped tests or warnings. Distinct focused WebUI and extension configs include the changed dependency/test closure and both produce empty diagnostic logs. The real-controller mounted cases now include normal and comparison fork → send → reopen, source isolation, held child loading and navigation; owner and settings cases include profile changes, migration, overlapping writers and import/delete guard preservation. The transaction boundary is still a serialized test double, not browser IndexedDB evidence. These results support the reviewed repairs but do not close F3's remaining availability defect.

While reconciling the re-review's storage-availability limit, root inspected the actual WebUI shim and identified its silent per-instance memory fallback. The same reviewer then ran a focused actual-shim/settings/projector probe: a required persona stayed unreadable in backing localStorage while bootstrap persisted an initialized-empty guard and committed a plain child. **One expected failing test** at `/private/tmp/h1-task41-fix1-storage-review/unavailable-storage.log` confirms an implementation defect, independent of real IndexedDB timing. The revised independent verdict is F1/F2 addressed, F3 **NOT ADDRESSED / P1**. Fix round 2 resumes the original implementer from `1a5492c542`; Task4.2 remains undispatched. A separate Minor misleading post-commit failure toast is assigned to Task4.2's outcome presentation.

Fix round 2 is committed as `28748d402b1a3e427f9f914b36fe21b9c58b33f9`, covering 17 code/test paths. The WebUI shim now exposes whether that exact instance selected persistent storage; the strict settings path also checks the installed extension client's captured backend and active routing. A fallback constructed while storage was unavailable cannot later certify persistence merely because a global API has recovered. Known canonical-backend unavailability rejects before initialization or a new pending guard, and legacy sync availability is required only when that area is actually read. If migration discovers unavailable legacy storage after publishing its token, the conservative pending guard remains; the fix does not claim automatic recovery for that abandoned migration. Root inspected the saved final **193 shared-UI tests / 13 files** and **20 WebUI tests / 3 files**, with no skips or warnings, plus distinct focused consumer configs and empty TypeScript diagnostic logs. Actual-shim and installed-extension negative tests supplement the existing serialized Dexie fixture; they do not establish browser transaction timing. The fresh independent review of `1a5492c542..28748d402b` found F3 **ADDRESSED**, with no new Critical/Important or Minor breakage. Root read the complete report and reconciled its remaining limits against Task5 and prior native/UAT evidence; no open in-scope P1/P2 remains. The focused configs extend shared UI tsconfig directly and carry distinct consumer resolution and expanded file lists, rather than literally extending the prior temporary config files.

The settings fix intentionally moves local per-conversation settings to browser-local storage, preserving the existing local value or copying the legacy browser-sync value once without deleting it. Later implicit browser-sync updates no longer overwrite that local baseline. Failed or uncertain settings effects retain pending guards; these block forks and conversation deletion/destructive import, while ordinary settings edits remain possible. Recovery of abandoned guards and broader synchronization remain separate work. No new runtime dependency, schema version or Python change is included in this fix.

## Native fork outcomes: task accepted

Task 4.2 is committed as `ec5f19816c15be5b901a4c4a2b7ee2abe96d46ca` and generated-locale completion `ac6919ae936729048e2e39fb1d9eb7d96b1a339d`, from exact task base `f836426ad49644cea352abc1c91bff5f1e44e56f`. The combined change touches 44 code, test and generated-locale paths. Root read the full report, final saved results and distinct consumer configurations. The independent review found two Important integration defects, both resolved by the independently reviewed fix1 below. Task 4.2 is accepted within its bounded scope; browser/storage/build qualification remains outstanding.

The implementation adds a durable client dispatch claim and retained fork outcomes, a limited frozen native copy, and uncached owner/workspace-scoped settings for known copied chats. It also repairs the overlooked timeline/Research stable-ID callers and Clear assistant's ambient settings callback, and uses plain-language outcome labels. Native copying remains a sequence of acknowledged requests; H2 atomic receipts/reconciliation and rich-state parity are not implemented.

| Check | Result | Evidence and limit |
|---|---|---|
| Combined client tests | 628 passed / 35 files / 19.29 s; exit 0 | `/tmp/h1-42-client-final.log`; actual mounted hooks/components/proxy plus serialized DB test doubles. No skipped tests. Expected negative-transport/share logging and existing router flags remain visible. |
| Native owner/API | 127 passed / 1 intentional skip / 9 warnings / 136.28 s; exit 0 | `/tmp/h1-42-native-final-authorized.log`; 43 SQLite and 43 PostgreSQL owner/migration passes, 21 contract-unit and 20 API passes. Actual-owner RLS included. |
| Focused core types | Passed; exit 0 | `/tmp/h1-42-core-tsconfig.json` and empty `/tmp/h1-42-core-types.log`; new fork store, history service, scoped path guard and dependencies. |
| Expanded consumer types | 11 existing diagnostics each; exit 2 | Distinct `/tmp/h1-42-extension-tsconfig.json` and `/tmp/h1-42-web-tsconfig.json`; the WebUI config resolves actual frontend shims. 63/66 included paths respectively. This is not a passing full-application check. |
| Python security/compile | Bandit 0 findings / 0 errors; compile passed | Four changed production paths: character_chat_sessions.py, history_selection_schemas.py, core/Chat/history_selection.py and chacha/message_store.py. `/tmp/bandit_h1_task42.json`, `/tmp/h1-42-python-static.json`. |
| Python lint | 2 existing diagnostics; exit 1 | F401 at 3896 and B904 at 8516 in character_chat_sessions.py; B904 shifted from 8513. Exact-base comparison has the same diagnostics. |
| Locale generation | Passed; exit 0 | Extension cwd `node scripts/sync-public-locales.js playground.json`; only 17 English generated entries changed, preserving 9 public-only keys. `/tmp/h1-42-locales.log`. |

The native command uses the same activated venv, dedicated H1 PostgreSQL configuration and `-o addopts='' -v -rs` options recorded above, targeting `test_history_selection_transactions.py`, `test_history_selection_migration.py`, `Chat/unit/test_history_selection.py` and `Chat_NEW/integration/test_history_selection_api.py`. It covers the optional plain-copy proof, settings/behavior row presence and context drift, malformed/required-state rejection, expected-user metadata/settings dependencies, workspace-default bypass and acknowledged child-chain reopen. The sole skip is PostgreSQL parametrization of the explicitly SQLite-only connection-lifetime check. The initial sandbox run could not reach the healthy dedicated PostgreSQL and produced 84 passes / 2 failures / 42 setup errors; root inspected both dynamic migration-fixture failures and confirmed no migration assertion was reached. The authorized same-scope run supersedes that environment failure without introducing availability skips or touching UAT 55475. Existing warning classes are Starlette/httpx, pytest config, Pydantic schema shadow/admission serialization, passlib crypt and the legacy character rate limiter.

The client command uses installed Vitest from `apps/packages/ui`, `NODE_OPTIONS=--no-experimental-webstorage`, the 35 targets in `/tmp/h1-42-client-command.json`, and coverage of the 18 touched production modules. Coverage of entire touched files, including large existing components, is 40.18% lines / 33.48% branches, not a passed coverage threshold. The new operation store is 92.66% lines / 81.11% branches; the history service 90.39% / 79.7%; controller 83.15% / 73.84%; settings hook 100% / 95%. The final report and saved coverage retain exact denominators. Earlier failing fixture runs are preserved and not combined with the final pass count.

<details>
<summary>Exact Task 4.2 client command retained for reproduction</summary>

```sh
cd /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design/apps/packages/ui
NODE_OPTIONS=--no-experimental-webstorage node_modules/.bin/vitest \
  run \
  src/components/Common/Playground/__tests__/HistorySelectionReview.test.tsx \
  src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.branch-share.test.tsx \
  src/components/Option/Playground/__tests__/Playground.search.integration.test.tsx \
  src/components/Option/Playground/__tests__/PlaygroundChat.per-model-routing.integration.test.tsx \
  src/components/Option/ResearchWorkspace/__tests__/ChatPane.stage2.test.tsx \
  src/components/Sidepanel/Chat/__tests__/body.dynamic-ui-fallback.test.tsx \
  src/db/dexie/__tests__/branch-projection.test.ts \
  src/db/dexie/__tests__/fork-operations.test.ts \
  src/db/dexie/__tests__/helpers.history-selection.test.ts \
  src/db/dexie/__tests__/history-selection.test.ts \
  src/db/dexie/__tests__/message-target-by-id.test.ts \
  src/hooks/__tests__/useLoadLocalConversation.history-selection.test.tsx \
  src/hooks/__tests__/useMessage.history-selection.test.tsx \
  src/hooks/__tests__/usePlaygroundSessionPersistence.history-selection.test.tsx \
  src/hooks/__tests__/useServerChatLoader.scope.test.tsx \
  src/hooks/__tests__/useServerChatLoader.test.ts \
  src/hooks/chat/__tests__/useChatActions.history-selection.test.tsx \
  src/hooks/chat/__tests__/useChatSettingsRecord.fork.test.tsx \
  src/hooks/chat/__tests__/useChatSettingsRecord.local.test.tsx \
  src/hooks/chat/__tests__/useHistorySelection.test.tsx \
  src/hooks/handlers/__tests__/messageHandlers.branch-controller.test.tsx \
  src/hooks/handlers/__tests__/messageHandlers.branch.test.ts \
  src/hooks/handlers/__tests__/messageHandlers.regenerate.test.ts \
  src/services/__tests__/background-proxy.test.ts \
  src/services/__tests__/chat-history-selection.test.ts \
  src/services/__tests__/chat-settings.deep-research-history.test.ts \
  src/services/__tests__/chat-settings.deep-research-pinned.test.ts \
  src/services/__tests__/chat-settings.deep-research.test.ts \
  src/services/__tests__/chat-settings.overlay.test.ts \
  src/services/__tests__/chat-settings.persistent-extension.test.ts \
  src/services/__tests__/chat-settings.sync.test.ts \
  src/services/tldw/__tests__/service-prompt-scope-error.test.ts \
  src/services/tldw/domains/__tests__/chat-rag.scope-error.test.ts \
  src/utils/__tests__/history-selection.test.ts \
  src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx \
  --coverage \
  --coverage.reportsDirectory=/tmp/h1-42-client-coverage \
  --coverage.include=src/components/Common/Playground/HistorySelectionReview.tsx \
  --coverage.include=src/components/Option/Playground/Playground.tsx \
  --coverage.include=src/components/Option/ResearchWorkspace/ChatPane/index.tsx \
  --coverage.include=src/components/Sidepanel/Chat/body.tsx \
  --coverage.include=src/db/dexie/schema.ts \
  --coverage.include=src/db/dexie/types.ts \
  --coverage.include=src/hooks/chat/chat-action-utils.ts \
  --coverage.include=src/hooks/chat/useChatActions.ts \
  --coverage.include=src/hooks/chat/useChatSettingsRecord.ts \
  --coverage.include=src/hooks/chat/useHistorySelection.ts \
  --coverage.include=src/hooks/chat/useServerChatLoader.ts \
  --coverage.include=src/hooks/handlers/messageHandlers.ts \
  --coverage.include=src/services/chat-history-selection.ts \
  --coverage.include=src/services/tldw/TldwApiClient.ts \
  --coverage.include=src/services/tldw/domains/chat-rag.ts \
  --coverage.include=src/services/tldw/service-prompt-scope-error.ts \
  --coverage.include=src/types/history-selection.ts \
  --coverage.include=src/db/dexie/fork-operations.ts
```

</details>

The 11 consumer diagnostics are 9 proxy mock typing errors (TS2493×3, TS2532×2, TS2345×4) plus 2 prompt-sync TS2339 errors. Root reconciled the proxy baseline with Task2.3's actual identical-config/exact-base comparison, `/tmp/h1-2.3-fix-baseline-types.log`, and the current unchanged-source proof `/tmp/h1-42-type-baseline-source-proof.json`. Current logs are `/tmp/h1-42-extension-types-final.log` and `/tmp/h1-42-web-types-final.log`; neither exits successfully. Bandit has no findings but emits existing nosec/comment warnings. No pristine-output, clean whole-package typing or real IndexedDB claim is made.

Both commits used normal hooks. Existing gc/unreachable-object warnings were left untouched. The exact two-commit review package is 192,771 bytes; root-owned documentation/tracking changes remain separate.

The independent review reproduced two failures through the actual mounted controller: changing account/config invalidates the native lease but leaves previous-owner fork outcomes displayed, and a verified ordinary legacy chat remains behind the settings-pending guard after qualification has completed. Its focused command was `NODE_OPTIONS=--no-experimental-webstorage apps/packages/ui/node_modules/.bin/vitest run --config /private/tmp/h1-42-review/vitest.config.mjs -t 'review probe'`, with **2 expected failing reproductions**, 20 unrelated tests filtered out and 430 ms duration; `/private/tmp/h1-42-review/probe.log` retains the exact assertions. Root read the complete review and inspected both relevant code paths. Fix round 1 resumes the original implementer from `ac6919ae93`, requiring callback/held-read isolation and ordinary legacy settings through the real controller, while preserving known-child uncached behavior and all history-admission gates. The independent fix review below confirms both findings addressed.

The review also records two Minor observations: disclosed baseline/test logging, and a new hard-coded saved-copy/opening-failed notification that can incorrectly name opening when only the client operation-record write failed. The old loader-throw result misclassification is fixed, but this distinct presentation case remains for final review. Real IndexedDB upgrade from populated v15, bookmark/send-recovery preservation, successful native fork → immediate send → reopen on both shells, full builds and broader integration preservation remain mandatory Task 5/final gates. No complete H1 verdict follows from this task's unit/mounted evidence.

### Task 4.2 fix1: review complete

Commit `c884e6ba25091f5b7d41bf33bd83fa55a4d3208b`, from exact fix base `ac6919ae936729048e2e39fb1d9eb7d96b1a339d`, changes two production TypeScript modules and six test files. Root read the complete fix appendix and saved final outputs, matched every one of the 16 requested test files to its passing output, and confirmed both consumer type logs exactly match the prior final logs. The scoped reviewer inspected the complete 49,503-byte one-commit package and confirmed F1/F2 addressed, with spec and quality PASS and no new Critical/Important breakage. Root read the full review and reconciled its browser/storage/build limits with the explicit Task 5 obligations. Static exit codes are from the implementer report with saved outputs; the reviewer did not rerun those commands.

The fix clears outcome presentation on invalidation and fences both successful and failed asynchronous reads by epoch, view, owner and live lease. Settings qualification is independent of history readiness in both the controller and native settings service; the send/fork capture gates are unchanged. Real-controller consumer tests exercise ordinary legacy settings, loader hydration, attachment restoration and Clear assistant. Known-child legacy settings remain scoped and editable. An additional self-review reproduction caught a same-view refresh that React batched through loading; the epoch dependency restores its outcome read without inventing a new selection.

| Check | Result | Evidence and limit |
|---|---|---|
| Initial permanent controller regressions | 5 expected failures, 2 already passing fail-closed cases, 20 filtered | `/tmp/h1-42-fix1-controller-red.log`; actual config subscription and ordinary/known-child legacy captures. |
| Actual service legacy settings regression | 1 expected failure, 50 filtered | `/tmp/h1-42-fix1-service-red.log`; owner handshake succeeds while ancestry remains unaccepted. |
| Same-view refresh reproduction | 3 expected failures, 24 filtered | `/tmp/h1-42-fix1-refresh-red.log`; subsequently repaired before final checks. |
| Final covering tests | 289 passed / 16 files / 10.66 s / exit 0; no skips | `/tmp/h1-42-fix1-final.log`; whole amended modules: 89.37% lines, 79.49% branches. Counts overlap earlier task runs. Existing model-fetch logging remains. |
| Core / consumer types | Core exit 0; each wider consumer exit 2 with the same 11 baseline diagnostics | `/tmp/h1-42-fix1-core-types.log`, `/tmp/h1-42-fix1-extension-types.log`, `/tmp/h1-42-fix1-web-types.log`; same distinct configs as the implementation check. |
| Format / whitespace | Focused Prettier check and `git diff --check` exit 0 | `/tmp/h1-42-fix1-format-check.log`; normal commit hooks, no bypass. |
| Native / Python | Unchanged by this fix | Prior 127-pass native result and production Bandit/compile evidence remain applicable; no redundant rerun or new security-scope claim. |

<details>
<summary>Exact fix1 covering command retained for reproduction</summary>

```bash
cd apps/packages/ui
NODE_OPTIONS=--no-experimental-webstorage node_modules/.bin/vitest run \
  src/hooks/chat/__tests__/useHistorySelection.test.tsx \
  src/hooks/chat/__tests__/useChatSettingsRecord.fork.test.tsx \
  src/hooks/chat/__tests__/useChatSettingsRecord.local.test.tsx \
  src/hooks/__tests__/useServerChatLoader.scope.test.tsx \
  src/hooks/__tests__/useServerChatLoader.test.ts \
  src/components/Option/Playground/__tests__/Playground.search.integration.test.tsx \
  src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx \
  src/hooks/chat/__tests__/useChatActions.history-selection.test.tsx \
  src/hooks/__tests__/useMessage.history-selection.test.tsx \
  src/hooks/handlers/__tests__/messageHandlers.branch-controller.test.tsx \
  src/hooks/handlers/__tests__/messageHandlers.branch.test.ts \
  src/db/dexie/__tests__/fork-operations.test.ts \
  src/services/__tests__/chat-history-selection.test.ts \
  src/components/Common/Playground/__tests__/HistorySelectionReview.test.tsx \
  src/hooks/__tests__/usePlaygroundSessionPersistence.history-selection.test.tsx \
  src/hooks/__tests__/useLoadLocalConversation.history-selection.test.tsx \
  --coverage \
  --coverage.include=src/hooks/chat/useHistorySelection.ts \
  --coverage.include=src/services/chat-history-selection.ts \
  --coverage.reportsDirectory=/tmp/h1-42-fix1-coverage
```

</details>

## Remaining qualification

- Native selected-message edit/delete require an owner-safe mutation adapter and remain capability-gated in this H1 control repair. Local scoped edits and leaf deletion are required; deletion must not orphan hidden descendants or cascade through alternatives.
- Adjacent regeneration and edit-and-resend remain capability-gated in H1. Task 4 must reject them before any copy/write/display truncation. Same-parent assistant admission and a read-only edited-input boundary override remain broader parity work; plain scoped edits/deletes are required here.
- Task 4.1 code/review is complete: exact allowlisted local copies, scoped stable mutation IDs, independent child files/parents, required-state exclusions and comparison materialization. Real browser isolation and source file-removal evidence remain Task5.
- Task 4.2 code/review is complete: atomic client dispatch claims, honest unknown/partial outcomes and no automatic retry or owner fallback. Real database migration, cross-view claims and native fork/send/reopen qualification remain Task 5.
- Task 5.1: actual WebUI, extension full page, compact sidepanel and extension expansion; real IndexedDB copy abort/reopen and concurrent bookmark/recovery updates; required native SQLite/PostgreSQL checks; consumer type checks, API generation and security checks.
- Final whole-branch review and fixes. No skipped required mode counts as parity or successful qualification.

## Acceptance status

| Criterion | Status |
|---|---|
| H1-A — independent views, exact selected path and late completion | Mounted/owner evidence exists; full-shell qualification pending. |
| H1-B — before-first, empty, stale bookmarks and reopen | Unit/mounted evidence exists; browser qualification pending. |
| H1-C — complete legacy review, CAS and independent interpretations | Reviewed native owner and local adapter evidence; real IndexedDB and full-shell review/reopen pending. |
| H1-D — accepted-parent admission and settlement with the existing composer | Ordinary/overlay and supported native tracked-character tasks reviewed; final regression and browser qualification pending. |
| H1-E — independent local child, files and supported context | Task 4.1 implementation and scoped reviews complete; real IndexedDB evidence pending. |
| H1-F — unknown/partial fork outcome and no automatic fallback/replay | Task 4.2 reviewed; real database/reload/browser evidence pending. |
| H1-G — model-qualified comparison child, shared controls and account/workspace scope | Local comparison and native outcome/controller tasks reviewed; full-shell qualification pending. |
| H1-H — capability rejection, unversioned compatibility and internal-field isolation | Existing API/adapter evidence; final client/native regression qualification pending. |

No passing release verdict, complete parity verdict, push or merge is recorded here.
