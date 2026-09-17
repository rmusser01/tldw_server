# H1 selected history and fork ownership: verification record

Status: **In progress. H1 is not qualified for release.** This record separates reviewed implementation evidence from the real-browser and owner checks that remain. Counts from different runs overlap and must not be added together.

## Scope and source

- Design: [H1 selected history and fork ownership](../Design/2026-09-16-chatbook-h1-history-selection-design.md).
- Execution plan: [H1 implementation plan](../../IMPLEMENTATION_PLAN_chatbook_h1_history_selection.md).
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

The implementation reports, commands, logs and review findings currently live in the plan-owned SDD workspace. The final record must retain the decisive evidence and controller rulings before any workflow-artifact cleanup.

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

## Remaining qualification

- Native selected-message edit/delete require an owner-safe mutation adapter and remain capability-gated in this H1 control repair. Local scoped edits and leaf deletion are required; deletion must not orphan hidden descendants or cascade through alternatives.
- Adjacent regeneration and edit-and-resend remain capability-gated in H1. Task 4 must reject them before any copy/write/display truncation. Same-parent assistant admission and a read-only edited-input boundary override remain broader parity work; plain scoped edits/deletes are required here.
- Task 4.1 code/review is complete: exact allowlisted local copies, scoped stable mutation IDs, independent child files/parents, required-state exclusions and comparison materialization. Real browser isolation and source file-removal evidence remain Task5.
- Task 4.2: atomic client dispatch claims, honest unknown/partial outcomes and no automatic retry or owner fallback.
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
| H1-F — unknown/partial fork outcome and no automatic fallback/replay | Task 4.2 and reload/browser evidence pending. |
| H1-G — model-qualified comparison child, shared controls and account/workspace scope | Local comparison copy and mounted controller evidence exist; native outcomes and full-shell qualification pending. |
| H1-H — capability rejection, unversioned compatibility and internal-field isolation | Existing API/adapter evidence; final client/native regression qualification pending. |

No passing release verdict, complete parity verdict, push or merge is recorded here.
