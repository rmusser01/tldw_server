# Release Qodo remediation — 2026-09-20

Tracking: TASK-13263.1, under TASK-13263. Original PR: #2761 (published 0.1.42); candidate: #2972; PyPI recovery: #2973.

Published 0.1.42 source and grants remain immutable. Server repairs land in the 0.1.43 candidate; cross-repository client repairs use an isolated Chatbook checkout. A resolved review finding means a verified fix, an already-landed fix, or an evidence-backed rejection, never merely an unread thread.

## Stage 1: Inventory
**Goal**: Capture every Qodo finding.
**Success Criteria**: All 23 original threads represented; new and recovery PR reviews checked.
**Tests**: GitHub review/thread inventory.
**Status**: Complete

## Stage 2: Repairs and contract verification
**Goal**: Reproduce and repair valid findings; verify disputed claims against implementation and approved specifications.
**Success Criteria**: Each ledger row has evidence.
**Tests**: Focused regressions for affected behavior; scoped Bandit.
**Status**: In Progress

## Stage 3: Integration and release records
**Goal**: Review combined repairs and regenerate the candidate protected-source manifest.
**Success Criteria**: Tests, type checking and source verification pass; release plan and PRs reference final commits.
**Tests**: Release contracts, targeted suites, TypeScript, package build, Bandit.
**Status**: Not Started

## Stage 4: Review closeout
**Goal**: Reply to Qodo threads and check new PR review feedback.
**Success Criteria**: Every thread has an accurate disposition and link; remaining external release gates are explicit.
**Tests**: Fresh GitHub review and CI inventory.
**Status**: Not Started

## Findings

| # | Finding | Disposition and evidence |
|---|---|---|
| 1 | [Credentials can hijack upstream sessions](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308619) | Under verification; `apps/mcp-unified/src/mcp_unified/federation/http_transport.py`. |
| 2 | [Revoked recipients retain clone access](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308620) | Verified intentional contract. Owner-bound receipts and operation status survive revocation; they do not authorize new work. Added revoked/clone-disabled tests proving replay + polling succeed while a fresh key is denied and no second Job exists. Existing worker revocation tests pass. |
| 3 | [Users can read another dataset's links](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308621) | Not reproducible: the authority resolver permits exactly one canonical default personal dataset and rejects all other explicit IDs before lookup. Expanded route tests cover graph/list/detail rejection; real Sync authority tests pass. Adding dataset columns would contradict the current owner-bound product projection. |
| 4 | [Deleted links disappear from sync](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308623) | Fixed: include_deleted now includes deleted endpoints in link-store pagination. Real SQLite regression covers two tombstones across pages and default exclusion. |
| 5 | [Failed runs remain stuck as running](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308625) | Under verification; `tldw_Server_API/app/core/Scheduled_Tasks/agent_task_jobs.py`. |
| 6 | [One scheduled slot can run twice](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308626) | Under verification; `tldw_Server_API/app/core/Scheduled_Tasks/agent_task_jobs.py`. |
| 7 | [Malformed values can authorize purges](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308630) | Under verification; `tldw_Server_API/app/api/v1/schemas/personal_context.py`. |
| 8 | [Health checks reject connected clients](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308636) | Rejected security-weakening recommendation. The cited unauthenticated client is a mocked unit test; production TLDWAPIClient supports API keys/bearer tokens. Public /health is minimal liveness; diagnostics require system.logs. Added connected-client guidance to Long_Term_Admin_Guide. All 37 permission tests pass. |
| 9 | [Readiness becomes unknown to clients](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308641) | Fixed: /api/v1/health/ready retains ready, engine, db and timezone-aware time alongside sanitized operator fields. Both ready/not-ready regressions pass without weakening authorization. |
| 10 | [Workspace cloning rejects client requests](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308644) | Under verification; `tldw_Server_API/app/api/v1/endpoints/sharing.py`. |
| 11 | [Shared sources cannot be parsed](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308645) | Under verification; `tldw_Server_API/app/api/v1/endpoints/sharing.py`. |
| 12 | [Connected clients cannot delete links](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308646) | Under verification; `tldw_Server_API/app/api/v1/endpoints/notes_graph.py`. |
| 13 | [One websocket test lacks its marker](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308600) | Under verification; `tldw_Server_API/app/core/MCP_unified/tests/test_guarded_slides_websocket.py`. |
| 14 | [A shutdown test depends on internals](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308605) | Under verification; `tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py`. |
| 15 | [Draft recovery text cannot localize](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308610) | Fixed: presentation recovery, conflict, status and action text uses playground locale keys; pending messages update with locale changes. Included in 222 passing focused frontend tests. |
| 16 | [Cancelled refresh continues the request](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308612) | Fixed: propagate cancellation during refresh fetch or body read rather than consuming original 401. Both abort variants pass. |
| 17 | [Shared speech bypasses its wrapper](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308614) | Fixed: speech uses wxt/browser with an extension-runtime guard, preserving WebUI speechSynthesis fallback. Wrapper-only and fallback tests pass. |
| 18 | [Login logic remains in the page](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308617) | Already fixed in candidate: pages/login.tsx re-exports @web/routes/login; both login suites pass. Next-specific imports remain in the web package. |
| 19 | [Macro retries can duplicate replies](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308627) | Under verification; `tldw_Server_API/app/core/Chat_Macros/jobs.py`. |
| 20 | [Successful batch ingests look incomplete](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308632) | Fixed: scan all batch results and identifier aliases for first valid persisted ID. Failed/skipped first item and invalid ID regressions pass. |
| 21 | [One owner's jobs reorder another's](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308633) | Under verification; `tldw_Server_API/app/core/Jobs/operations/sqlite/lifecycle.py`. |
| 22 | [Malformed upstream replies appear valid](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308634) | Under verification; `apps/mcp-unified/src/mcp_unified/federation/http_transport.py`. |
| 23 | [Webhook error bypasses core module](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308597) | Under verification; `tldw_Server_API/app/core/Admin_Webhooks/crypto.py`. |

## Confirmed contract decisions

- Finding 2: The approved `2026-08-25-shared-workspace-clone-jobs-design.md` explicitly requires recipient-owned operation receipts/status to survive share revocation. Replays do not admit new work; workers recheck live share/clone authority before copy and publication. Verify endpoint and worker regressions before disposition.
- Finding 3: `resolve_notes_link_dataset_authority` selects exactly one active default personal Chatbook dataset and rejects any different explicit ID. Product link rows live in the owner-bound database. Verify real authority and route tests before disposition.
- Finding 8: The cited credential-free client is a unit test that mocks `_request`; the real client supports both X-API-KEY and bearer credentials. Operator diagnostics deliberately require SYSTEM_LOGS; `/health` is the public liveness route. Preserve this security boundary and document it for clients.
- Finding 18: Current candidate `pages/login.tsx` already delegates to `@web/routes/login`. Verify current login regressions.

## Release gates retained

PR2973 cannot merge or dispatch PyPI until the requester supplies the required human Change summary or explicitly waives it for that PR. Automatic approval rejected the ambiguous earlier reply. PR2972 legal dates remain proposed, and its final CI/review remain open. These gates do not prevent Qodo repairs.

## Verification evidence (in progress)

- 82 backend tests passed across clone endpoints/worker, Notes authority/links, and readiness; added disposition canaries: 5 passed. Parent final link-store/readiness run: 25 passed.
- Health control-plane/sanitizer tests: 13 passed after fixing test log capture to attach after application logging startup. The initial failure was a removed test log sink, not leaked diagnostic data.
- Health authorization matrix: 37 passed, including anonymous denial, unprivileged denial, and SYSTEM_LOGS/admin access.
- Frontend: 222 tests passed; full TypeScript check passed. Presentation test hook naming was corrected to satisfy React Hooks lint; its 111 tests passed again.
- Parent touched production Python Bandit: no findings. Test-only scan excludes B101 assertions; three unchanged B106 synthetic token/password fixtures remain (access/hash), with no new secret finding.
- Cross-repository findings 10–12 tracked in Chatbook TASK-32881 at isolated `/private/tmp/tldw-chatbook-release-compat`; implementation ongoing.
