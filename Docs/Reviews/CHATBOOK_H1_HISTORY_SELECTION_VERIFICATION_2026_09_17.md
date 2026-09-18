# H1 selected history and fork ownership: verification record

Status: **In progress. H1 is not qualified for release.** This record separates reviewed implementation evidence from the real-browser and owner checks that remain. Counts from different runs overlap and must not be added together.

## Scope and source

- Design: [H1 selected history and fork ownership](../Design/2026-09-16-chatbook-h1-history-selection-design.md).
- Execution plan: [H1 implementation plan](../../IMPLEMENTATION_PLAN_chatbook_h1_history_selection.md).
- Decisions: [chronological implementation decision history](CHATBOOK_H1_HISTORY_SELECTION_DECISIONS_2026_09_17.md), including superseded rulings and stated tradeoffs; this is not additional acceptance evidence.
- Backlog: TASK-13261.1, In Progress.
- Branch: `codex/chatbook-h1-history-selection`; isolated worktree `.worktrees/chatbook-h1-history-design`.
- Implementation base: `f00e12a5aa`, based on server dev `59049e094e0845a4611ea725ae19b7c1754ea709`.
- Most recent reviewed source refresh: Chatbook dev `d8fb4053f9a27a799d5cdb8ee58f7fd1de91efce`; server dev `59049e094e0845a4611ea725ae19b7c1754ea709` unchanged. The third delta (PR2706) adds local Library STT failure diagnostic logging/tests/docs only; root inspected immutable source and identified no H1 contract change. No Chatbook tests were run. September18 02:16UTC read-only remote check confirmed these pins. Historical test pins below are retained. [Source delta review](../Design/2026-09-17-chatbook-chat-parity-source-refresh.md).
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

## Task 5.1 browser and storage qualification

Execution date: September17–18,2026; Task5 base `7b9bfed636a71b60e066b30d1776e64d38f1b440`, prior reviewed integration `c884e6ba25`. Task5 changes are pending independent task and whole-branch review; the successful checks below are not a release verdict. Root owns that disposition. No live provider, main/UAT database, dependency reconciliation, push or merge was used.

### Runtime and reproducible commands

All commands use this H1 worktree. Node26.0.0, Next16.1.4/Turbopack, installed Vitest4.0.18 and installed Playwright use Chromium build1208 at `/Users/macbook-dev/Library/Caches/ms-playwright/chromium-1208/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing`. The WebUI matrix mounted the actual development shell on isolated localhost18171; the extension mounted actual options and sidepanel pages with its production Chrome MV3 build and actual background transport. Extension launch failures throw; no `launchWithExtensionOrSkip`, optional required controls, retry-based passing classification or hidden test skip was used. The shared fixture is a create-capable ephemeral Node HTTP server; its native/capture/provider responses are deterministic mocks, not SQLite/PostgreSQL evidence.

```bash
# apps/tldw-frontend: dedicated running H1 dev process
NEXT_PUBLIC_API_URL=http://127.0.0.1:18172 NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=advanced NODE_OPTIONS=--max-old-space-size=8192 node_modules/.bin/next dev -p 18171
# Each test seeds its actual ephemeral fixture origin;18172 is initial bootstrap only.

# apps/tldw-frontend: full matrix
TLDW_WEB_URL=http://localhost:18171 TLDW_WEB_AUTOSTART=false PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH='/Users/macbook-dev/Library/Caches/ms-playwright/chromium-1208/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing' PLAYWRIGHT_JSON_OUTPUT_FILE=/tmp/chatbook_h1_web_final_matrix.json node_modules/.bin/playwright test e2e/workflows/chat-history-selection.spec.ts --project=chromium --workers=1 --retries=0 --reporter=line,json --output=/tmp/chatbook_h1_web_final_matrix_artifacts

# apps/extension: current production build, then full matrix
PATH="$PWD/node_modules/.bin:$PATH" TLDW_BUILD_PROFILE=production node scripts/build-with-profile.mjs --browser=chrome
TLDW_E2E_EXTENSION_HEADLESS=1 TLDW_E2E_SKIP_EXTENSION_BUILD=1 PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH='/Users/macbook-dev/Library/Caches/ms-playwright/chromium-1208/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing' PLAYWRIGHT_JSON_OUTPUT_FILE=/tmp/chatbook_h1_extension_final_matrix.json node_modules/.bin/playwright test tests/e2e/chat-history-selection.spec.ts tests/e2e/sidepanel-options-handoff.spec.ts --project=chromium-extension --workers=1 --retries=0 --reporter=line,json --output=/tmp/chatbook_h1_extension_final_matrix_artifacts
```

`/tmp/chatbook_h1_web_final_matrix.log`/JSON: **27 passed,141.075s,0 skipped,0 unexpected,0 flaky**. `/tmp/chatbook_h1_extension_final_matrix.log`/JSON: **27 passed,217.992s,0 skipped,0 unexpected,0 flaky** (24 history cases and3 handoff cases). The extension matrix used the successful35.9s build `/tmp/chatbook_h1_extension_build_session.log`. A subsequent self-review found the index-search reveal fence omission; the final-source rebuild and affected-only checks are recorded below, keeping this full-matrix provenance distinct.

### Mandatory behavior mapping

| Required boundary | Executed evidence |
|---|---|
| Two independent views over one owner DB | Both full shells select A/B separately, retain unique view identities/bookmarks, submit exactly their selected content and reopen. Real IndexedDB; provider requests captured by HTTP fixture. |
| Before-first/empty and legacy alternatives | Both shells confirm/reload before-first and empty boundaries and preserve alternative source rows. WebUI independently qualifies20001-row before-first in addition to both-shell full-tip coverage. |
| Complete large legacy review | Both full shells review20001 readable rows, confirm through19999, render the full20000 selected path without Chromium OOM, reopen, reach end row19999, search12345 and timeline-edit42; normal Cancel closes the editor and source remains unchanged. Owner/provider arrays remain complete; only rendering is windowed. |
| Real transaction commit/abort | Both shells invoke the production local fork through mounted controls, adopt child for immediate send/reopen; injected real IndexedDB transaction abort leaves no child/partial rows. Source messages, files and server/sync control identity remain unchanged. |
| Independent child mutation/file identity | Both full shells fork ordinary and canonical two-round comparison sources, remap common prompt ancestry from omitted B onto selected A, preserve model provenance and clear comparison markers, send/reopen child, edit/delete child via UI and remove copied file through production API. Copied file has a new identity; source rows/file remain byte-equivalent. |
| Comparison handoff and unsupported owner | Original selected URL/reload shows readable clusters while selected capture remains `unsupported_comparison_history`; model-qualified cluster fork is available. Both full shells complete the two-round chain. Actual390px sidepanel expansion opens its extension full page and completes the same chain. This does not implement ordinary comparison selected-send support. |
| Native fork adoption/settings | Full shells and compact sidepanel fork scoped native source, next send uses child, child reopens. Poisoned scratch/server-ID browser notes do not enter native settings; explicit native note patch is scoped, persists/reopens and permits a later nonplain child. Actual extension background allowlist is exercised. |
| Native owner failure/settings split | Malformed metadata scope retains failed native identity, invalidates controller, exposes readable error feedback and causes zero scratch/local/foreign/native settings writes. Real controller tests additionally cover null-ID hydration, absent metadata/messages, temporary/ordinary reset positives and late navigation. |
| No unsolicited native creates | Create-capable fixtures prove native and local selection/reopen/fork do not trigger legacy autosave creates; genuine fresh draft owns a local conversation. Unit execution covers old draft preparation/late response and positive unowned legacy save. |
| First native character receipt | Both full handlers, plus390px sidepanel, create once and dispatch versioned native completion, accept admission/result IDs, reopen and append a second native turn with exact4-row chain; no client message-copy POST. Missing/wrong ACK stays uncertain, retains native effects and never replays on reload. |
| Live and unload navigation | Both full shells distinguish unload/reload from actual same-page `tldw:open-history` product navigation. Held first-create never adopts/dispatches after local navigation;390px sidepanel also exercises this. Original live fork dispatcher refines retained result: create ACK becomes known partial child with zero copied rows after lease retirement; held settings result remains completed. No new view is retargeted. |
| Unknown fork/reload/two views | Headers-then-truncated-body fixture produces one application dispatch/one wire request, durable unknown outcome, no fallback/replay on reload or second view. Original empty-response duplicate delivery is separately retained below and did not pass. |
| Account/workspace filtering | Real IndexedDB retains a same-account foreign-workspace fork record; actual global H1 controls hide it and issue no foreign request/action. A second real page changes account config while capture is held; original view becomes unavailable, late result cannot redirect it, original records persist. Both final matrices use protocol-based consumer discrimination. Separate WorkspaceChatPanel is not H1-integrated; this is not workspace UI parity. |
| Populated schema upgrade | WebUI opens actual v15 IndexedDB then production v16 preserving legacy projection, bookmark and send-recovery records. This is real browser migration, not fake-indexeddb. |
| Settings migration/concurrency/guard lifecycle | Both consumer bundles invoke production APIs with real durable backends: migration and two writes; held migration overlapping later write; legitimate missing data versus SecurityError read unavailability; one failed writer preserves its guard while another independent view succeeds; guarded fork rejection; delete/undo/import/reload preserve pending tokens. |
| Compact controls and handoff | Strict unique review/header controls at390px, actual options expansion, explicit composer draft handoff in its supported Pro mode, and shared real extension settings. Default compact header tests retain casual mode. |
| Native SQLite/PostgreSQL/API/security | Reused unchanged native checkpoint above:36 SQLite+36 PostgreSQL owner passes,277 API/regression passes, exact documented non-required skips;16 production Python files Bandit0 findings/errors. HTTP fixtures never substitute for those checks. |

Browser storage helpers are `apps/extension/tests/e2e/utils/history-storage-entry.ts` and `history-storage.ts`. Installed esbuild0.27.2 is resolved through the existing WXT package; `injectHistoryStorage()` bundles the production entrypoint with `platform:'browser',format:'iife',globalName:'__h1Storage'`, write:false and browser alias/plugin configuration, then Playwright injects the bundle. Metafiles `/tmp/chatbook_h1_storage_webui_metafile.json` and `_extension_metafile.json` record exact resolutions. Both use shared Dexie PageAssistDatabase/v16; WebUI uses its actual localStorage-backed Plasmo/WXT shims, extension uses installed Plasmo and browser-local/legacy-sync. Operations under test call production settings/delete/import/undo APIs; raw seeding only establishes fixtures. Two independent page/module instances are required to avoid accidentally inheriting the first Dexie promise zone. No production export, test route, CSP weakening, alternate settings authority or installed dependency was added. Fault injection holds/fails the actual storage boundary and restores it afterward. This is browser transaction evidence distinct from mounted-control evidence.

### Defects exposed and bounded fixes

| Cause and repair | RED→GREEN evidence (runs overlap) |
|---|---|
| Stream transport mutated frozen prepared request; both installed domain method and maintained class copy now construct outgoing stream body | `/tmp/chatbook_h1_frozen_stream_red.log`2failed/8passed → `_green.log`28passed/2files; original fork→send browser passes. Scope/signal/native parser preserved. |
| Duplicate sidepanel review and overlapping header | Strict390px uniqueness/normal clicks exposed duplicate nonfunctional body control; route retains functional callbacks and spacing. `/tmp/chatbook_h1_sidepanel_layout_green.log`; final compact matrices. |
| Legacy autosave uploaded H1 capture during hydration; automatic mirror linking ran before coherent qualification | Actual pre-fix31 unsolicited creates `/tmp/chatbook_h1_web_native.log`; autosave RED4failed/9passed→GREEN13, create-capable browser positives. Qualified legacy-review mirror allowed without ancestry readiness; old-view async work fenced. |
| Full-tip20001-row review mounted20000 heavy cards and exhausted Chromium heap | `/tmp/chatbook_h1_web_storage.log` original OOM; installed virtualizer reused with actual outer scroller/stable blocks and ID/index reveal. `/tmp/chatbook_h1_fulltip_portrait_green.log` and final large cases. Full data retained. |
| Default portrait layout skipped edit listener as though absent bubble child owned it | `/tmp/chatbook_h1_portrait_edit_red.log`1failed/7passed→`_green.log`8passed; shared exact bubble predicate, actual portrait/bubble layouts and normal Cancel covered. |
| Metadata rejection cleared display ID but retained native controller, permitting scratch settings | `/tmp/chatbook_h1_rejected_owner_red.log`4failed/9passed→`_green.log`13passed; null native ID pending, matching loader unavailable state, failed display identity retained. Temporary/ordinary positive gate narrowed. |
| Error notification translated an object instead of string leaf, throwing ICU overlay | Original native negative browser failed; `common:error.label` leaf fixed; `/tmp/chatbook_h1_rejection_i18n_green.log`1pass and final negative. Real ICU resource/browser, not a defaultValue-only mock. |
| Unsupported comparison handoff rendered empty despite readable verified local rows | Guarded reuse of existing local display loader, exact owner/profile/view checks, no capture upgrade; helper5passed, integrated regression43/5files. Two-round canonical compare:reply fixture and actual cluster controls. |
| Feature preference temporarily false while storage hydrates destructively disabled comparison | `/tmp/chatbook_h1_compare_hydration_red_settled.log`1failed/2passed→`_green.log`9/3files. Real third storage metadata exposed; both comparison hooks defer destructive disable/persistence until loaded; settled false and force-enabled retained. |
| Shared session stale comparison replay raced owner compareStates hydration | `/tmp/chatbook_h1_comparison_session_both_red.log`2failed/5passed→`_final.log`21passed/3files. Verified local owner uses its existing per-conversation hydration; stale true/false/models, ordinary/native compatibility and held-read live navigation covered. |
| Sidepanel always supplied no-op context overrides, so native explicit-context guard rejected ordinary character send | `/tmp/chatbook_h1_context_noop_red_settled.log`2failed/8passed→`_green.log`12passed/2files. Only successfully resolved unchanged input with no optional selection/pieces/system context returns empty overrides; explicit/inherited unmatched context retained. Native guard unchanged. |
| Index search reveal captured but failed to recheck owner fence after await | Final self-review RED `/tmp/chatbook_h1_index_reveal_red.log`1failed/20passed. Add matching post-reveal check before DOM scroll; final-source GREEN and both-consumer affected runs below. ID/edit reveal already had the check. |

### Failed attempts and evidence corrections

- Canonical exporter initially failed importing workspace `tldw_profile_core`; no package install/stub/startup mutation. `/tmp/chatbook_h1_python` sets this exact worktree and `packages/tldw_profile_core/src` on PYTHONPATH, then executes root venv Python. Final `/tmp/chatbook_h1_generate_api.log`:2097paths/3230schemas; TypeScript generation/fingerprint succeeded. `/tmp/chatbook_h1_verify_api.log`:351paths/49fields/10documented exceptions. The first failed log was overwritten and is disclosed rather than represented as saved evidence.
- Comparison fixture initially used permissive unit tag `compare:assistant`; producer/renderer require `compare:reply`. Second round now matches canonical production shape with an omitted-model parent. Missing actual compare feature flag, then API-presence runtime discrimination and Plasmo serialization/model seeding mistakes were fixture causes. WebUI normally exposes chrome.storage through runtime-bootstrap; API presence is not extension identity. All shared helper branches now test `location.protocol==='chrome-extension:'`. Diagnostic `/tmp/chatbook_h1_comparison_flags.log` verified loaded current chunks/gates with effective flag false but wrong sync area true; corrected actual area `/tmp/chatbook_h1_comparison_backend.log` passed. This fixture correction is distinct from independently reproduced preference-hydration and shared-session product defects.
- Extension selected model uses JSON serialization and already-mounted Zustand precedence. Pre-mount provider-qualified `tldw:openai:h1-model` seed preserves actual gate. No provider relaxation. `/tmp/chatbook_h1_extension_character_premount.log`5pass/2sidepanel failures then exposed the no-op context defect. After that fix, positive sidepanel reached reopen but strict text locator also matched Rename conversation; final locator is scoped to actual message card, retaining exact native rows/parents.
- Original comparison repeated failures are retained (`comparison_hydration`, `comparison_writes`, `extension_comparison_backend` logs); no lucky retry accepted. Final matrices use retries0. Sidepanel expansion had passed while direct reload exposed the separate shared-session race.
- Large edit investigation had three failed timing variations before reassessment. The confirmed cause was missing listener under default portrait predicate; no added frame delay, changed preference or raised browser heap remains.
- Same-page storage concurrency initially inherited Dexie's zone and produced a test harness failure. Independent real page/module instances then qualified the actual separate-writer boundary (`/tmp/chatbook_h1_storage_web_two_views.log`, `_extension_two_views.log`).
- Sidepanel explicit draft handoff is a Pro ControlRow action. Casual mode has no such button. Fixture now sets the actual persisted Pro preference for that case only; no routing redesign.
- A diagnostic Playwright invocation from the wrong package root collected Vitest files (`/tmp/chatbook_h1_live_navigation.log`); that collection failure is not product evidence. A context unit setup briefly recreated primitives each render and remained loading; stable test dependency yielded the expected2 assertion failures before the repair. Wrong relative edit paths failed before any file mutation and were corrected.
- Pre-H1 winner-copy guard read the wrong source file. Root reproduced2 assertion failures at both initialH1base andTask5base with same guard SHA `a5f8baa8e8cf5ef315d3982b51bd8ee06aabb7a1ff05910f4b178df5ec534b9d`; all six strings already belong to PlaygroundCompareCluster. Only the source path changed; all six assertions retained, corrected guard1pass (`/tmp/chatbook_h1_winner_guard_green.log`). Root receipts `/tmp/chatbook_h1_winner_copy_baseline{,_receipt}.json`/.log.

### Explicit transport and capability limits

The original empty-response POST probe **failed**: `/tmp/chatbook_h1_unknown_empty.log` records1 browser request event versus6 received create requests, with a seventh late request observed. Fixture `nativeChats` allocates a distinct child on every received create, so this represents duplicate child creation, not merely duplicate logging. Root traced one application dispatch and extension uncertain-write fallback prohibition. Chromium primary source allows reused-socket resend before response headers: `https://chromium.googlesource.com/chromium/src/+/be49317a0e98dec0bd1330a46fa00abfe39eb0eb/net/http/http_network_transaction.cc`; this corroborates a transport possibility, not an exact installed Chromium revision audit. Headers/truncated-body no-replay passing cases do not erase this failed probe. H1 does not claim at-most-once native effects. Reliable native receipts/reconciliation/deduplication remain H2; independent review must assess this limitation explicitly.

Native HTTP fixture admissions/results do not establish native DB authority. Actual unchanged SQLite/PostgreSQL evidence above supplies that boundary. WorkspaceChatPanel remains a separate, non-H1 integrated UI. Unsupported rich context/persona/temporary/native plain-copy exclusions remain. Comparison raw display does not create accepted selected ancestry. No standalone/cold-install/local-model certification, H2/H3/H4 completion or whole-product parity is claimed.

### Final-source validation and static classifications

Final index-reveal RED1failed/20passed → GREEN21passed (`/tmp/chatbook_h1_index_reveal_green.log`). After that last production edit, extension production rebuild `/tmp/chatbook_h1_extension_build_release.log` passed39.9s,48.72MB; WebUI production rebuild `/tmp/chatbook_h1_web_production_build_release.log` compiled24.5s and generated154pages. The WebUI command is:

```bash
# apps/tldw-frontend; supported isolated dist avoids clobbering .next/dev
PATH="$PWD/node_modules/.bin:$PATH" TLDW_BUILD_PROFILE=production TLDW_NEXT_DIST_DIR=.next-live-tier-h1-production NODE_OPTIONS=--max-old-space-size=8192 node scripts/build-with-profile.mjs --bundler=turbopack
node scripts/verify-shared-token-sync.mjs --dir .next-live-tier-h1-production/static
node scripts/check-bundle-budget.mjs --dir .next-live-tier-h1-production
```

The normal production profile sets quickstart; browser WebUI matrix used the separately recorded advanced dev runtime. Existing Next config explicitly `typescript.ignoreBuildErrors:true`; build success is not a passing typecheck. Final token check `/tmp/chatbook_h1_web_release_tokens.log` verifies actual WebUI `static/chunks/2424eea334ee3a68.css`; bundle budget `/tmp/chatbook_h1_web_release_budget.log`578.2KB gzip shared app versus600KB budget,33files; heaviest route876.3KB. The first token check against the alternate dist root selected copied extension-profile CSS under standalone: the unchanged checker only limits traversal for literal basename `.next`. That first exit0 is not WebUI token proof; explicit `/static` corrects the evidence without modifying the checker. Generated outputs are retained locally and excluded from staging.

Final affected browser reruns use the exact matrix commands above, changing output/report paths to `/tmp/chatbook_h1_{web,extension}_final_navigation{.log,.json,_artifacts}` and `--grep '20001-row|timeline edit'` (WebUI) or `--grep '20001-row|isolate source: comparison|comparison sidepanel expansion'` (extension; only history spec). **WebUI4passed25.2s; extension3passed33.4s; each0skipped/0unexpected/0flaky.** These run on final source/current extension build and overlap the earlier complete matrices; counts are not additive. Unaffected matrix cases are deliberately reused. Installed `scripts/assert-playwright-no-skips.mjs` was executed against all four JSON files and reported executed/expected27/27,27/27,4/4,3/3 with every skipped/unexpected/flaky count0.

Affected shared Vitest `/tmp/chatbook_h1_release_focused_vitest.log`: **208passed/27files**,7.61s, before the last index-search test. Final index-search file21passed includes the new regression; final session lint-only test change8passed (`/tmp/chatbook_h1_session_lint_regression.log`). The full command, from apps/tldw-frontend, is:

```bash
NODE_OPTIONS=--no-experimental-webstorage node_modules/.bin/vitest run \
 ../packages/ui/src/hooks/__tests__/readable-local-comparison.test.ts \
 ../packages/ui/src/hooks/__tests__/useServerChatLoader.scope.test.tsx \
 ../packages/ui/src/hooks/chat/__tests__/useChatSettingsRecord.fork.test.tsx \
 ../packages/ui/src/hooks/__tests__/usePlaygroundSessionPersistence.history-selection.test.tsx \
 ../packages/ui/src/hooks/__tests__/usePlaygroundSessionPersistence.test.tsx \
 ../packages/ui/src/hooks/__tests__/useMessage.history-selection.test.tsx \
 ../packages/ui/src/hooks/handlers/__tests__/messageHandlers.branch-controller.test.tsx \
 ../packages/ui/src/components/Option/Playground/hooks/__tests__/usePlaygroundPersistence.test.tsx \
 ../packages/ui/src/components/Option/Playground/__tests__/Playground.search.integration.test.tsx \
 ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundChat \
 ../packages/ui/src/components/Common/Playground/__tests__/Message.error-recovery.integration.test.tsx \
 ../packages/ui/src/services/tldw/__tests__/TldwApiClient.request-scope.test.ts \
 ../packages/ui/src/services/tldw/__tests__/TldwApiClient.request-sanitization.test.ts \
 ../packages/ui/src/hooks/chat/__tests__/useConversationContextComposition.test.tsx \
 ../packages/ui/src/hooks/chat/__tests__/useCompareMode.hydration.test.tsx \
 ../packages/ui/src/hooks/__tests__/useFeatureFlags.rollout.test.tsx
```

Static checks are classified, not represented as pristine:

- Frontend `NODE_OPTIONS=--max-old-space-size=8192 node_modules/.bin/tsc --noEmit`: final `/tmp/chatbook_h1_frontend_tsc_final_source.log` **90diagnostics/124lines,exit2**, byte-identical to root's actual initial-base `/tmp/chatbook_h1_frontend_tsc_initial_base.log` (same installed compiler/dependencies/generated inputs). Intermediate release log differed only in two `HTMLElement | SVGElement` union orderings. All9diagnostic source files unchanged; root proved no current shared source leaked into baseline. Root qualification document/receipts remain in the SDD workspace; full typecheck is failing baseline, not passing.
- Extension `node_modules/.bin/tsc --noEmit -p tsconfig.compile.json`: passed `/tmp/chatbook_h1_extension_tsc_release.log`. This existing config excludes shared-UI/tests; it is not a replacement for full frontend tsc.
- Scoped ESLint uses installed `apps/tldw-frontend/node_modules/eslint/bin/eslint.js --config apps/tldw-frontend/eslint.config.mjs --format json` with all38 touched/new TS/TSX files enumerated in `/tmp/chatbook_h1_lint_release_paths.json`, from worktree root. Final `/tmp/chatbook_h1_eslint_final_source.json`: **0errors,1102warnings,exit0**. Immutable Task5BASE `git show BASE:path` was linted through the same ESLint API/config/filePath:30existing files,2errors/1041warnings (`/tmp/chatbook_h1_eslint_taskbase.json`). Two old test-hook lint errors were repaired while those tests were changed. Final net61extra warnings are primarily explicit-any test boundary typing; the new virtualizer has the installed-library React Compiler incompatibility warning, matching the existing sidepanel pattern. The compiler skips memoizing that component; actual browser navigation/rendering is covered. This is not a zero-warning claim and no rule was disabled. Baseline comparison and full per-file messages are retained.
- `git diff --check` passed. No Python changed in Task5, so unchanged16-file Bandit0findings/0errors, compile pass and explicitly classified2baseline Ruff diagnostics are reused from the native checkpoint above. No security finding is hidden by counting mocked HTTP as native DB proof.
- API generation used the root venv and `/tmp/chatbook_h1_python` wrapper (`PYTHONPATH=<exact-worktree>:<exact-worktree>/packages/tldw_profile_core/src`, exec root `.venv/bin/python`). Worktree-root commands: `PYTHON=/tmp/chatbook_h1_python node apps/tldw-frontend/scripts/generate-api-types.mjs`; `PYTHONPATH="$PWD:$PWD/packages/tldw_profile_core/src" node apps/extension/scripts/verify-openapi-client-paths.mjs`. Fingerprint is tracked; generated OpenAPI/schema outputs remain ignored. No dependency install or resolver reconciliation occurred.

Representative screenshots (absolute retained evidence paths):

- WebUI selected controls: `/tmp/chatbook_h1_web_final_matrix_artifacts/workflows-chat-history-sel-707f7-dependent-selected-variants-chromium/selected-history-review.png`.
- WebUI full-tip editor/end/search: `/tmp/chatbook_h1_web_final_navigation_artifacts/workflows-chat-history-sel-60ed4-001-row-full-tip-transcript-chromium/legacy-full-tip-navigation.png`; same directory has `legacy-20001-review.png` and `selected-history-review.png`. Subsequent normal Cancel/source-unchanged assertions pass.
- Extension full-tip: `/tmp/chatbook_h1_extension_final_navigation_artifacts/chat-history-selection-ful-4c5e1-001-row-full-tip-transcript-chromium-extension/legacy-full-tip-navigation.png`; same directory has the review screenshots.
-390px selected control and local fork outcome: `/tmp/chatbook_h1_extension_final_matrix_artifacts/chat-history-selection-com-ef0b5-r-immediate-send-and-reopen-chromium-extension/selected-history-review.png` and `local-fork-outcome.png`.
-390px native receipt: `/tmp/chatbook_h1_extension_final_matrix_artifacts/chat-history-selection-sid-c54f7-owledged-result-and-reopens-chromium-extension/native-character-receipt.png`.
- Native failure feedback with real translated notification: `/tmp/chatbook_h1_web_final_matrix_artifacts/workflows-chat-history-sel-1a94c-s-to-browser-mirror-storage-chromium/rejected-native-feedback.png`, captured immediately after notification assertion.
- Native settings edit and account invalidation screenshots remain under the corresponding final matrix artifact directories; every screenshot belongs to an executed test, not a separate synthetic UI.

Self-review checked stable owner identity, final post-await fences, full transcript/provider data retention, effective bubble predicate, qualified legacy mirror lifecycle, no-op context versus explicit/inherited optional context, candidate/settings ownership, real consumer storage aliasing, native mock provenance, generated outputs and exact staging scope. Outstanding release decisions are independent review, the explicit native duplicate-delivery limitation, baseline full frontend type failure, lint/compiler warnings and non-H1 WorkspaceChatPanel scope. No additional behavior or acceptance is implied by the passing counts.

Task5 implementation/evidence commit: `253874def16ab8ee5e46af04881ca9dc37a89d6d` (40 exact paths, normal `git commit`, no hook bypass). Commit output `/tmp/chatbook_h1_task5_commit.log` includes existing Git automatic-maintenance warnings about unreachable loose objects/gc.log; no prune/gc cleanup was attempted. Root-owned plan/spec/Backlog/parity/source/decision files and generated build artifacts were excluded.

Build warnings retained: final WebUI Turbopack reports two broad documentation tracing patterns at unchanged `apps/tldw-frontend/lib/documentation.ts:47` and`:56`, each matching333238files, with possible build-performance/over-bundling consequences; stale caniuse-lite/Browserslist data; Node's missing `--localstorage-file` experimental warning. Extension build retains duplicate-import warnings for `MediaNavigationFormat` and `estimateStorageCost`, plus stale Browserslist data. These checks passed with warnings; no dependency refresh or unrelated build-system repair was made. WebUI standalone output includes test-profile assets because of broad tracing and is local qualification output, not a published deployment package.

Source freshness boundary: historical qualification uses reviewed Chatbook `d8fb4053f9a27a799d5cdb8ee58f7fd1de91efce`. Root observed remote dev `e89f28d751bc8a5b4f4545b8894b87437252c657` at03:01:59UTC and is independently auditing that component-pattern-library delta. This task does not claim tests against that unreviewed source or expand implementation into it.
