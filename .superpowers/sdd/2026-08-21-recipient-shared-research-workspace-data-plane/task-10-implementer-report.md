# Task 10 Implementer Report

## Implementation Summary

- Added an integrated deterministic recipient security matrix using the real sharing repository, access service, owner media databases, recipient chat store, and recipient chat orchestrator. It covers owner/member/nonmember access, neutral missing/revoked/unauthorized `404`, `sharing.read` `403` ordering, source listing and preview, `all` and `include` chat scopes, replay and receipt conflicts, boundary-time revocation/membership/source/media changes, and no persisted failed chat.
- Extended the cross-user suite with a canonical authorization-order regression that rejects a nonmember before any owner database acquisition.
- Replaced the active removed-route request test with an OpenAPI assertion that recipient `GET` operations expose no owner `media_id` parameter.
- Added a shared-mode Playwright request ledger and page-object helpers. The ledger classifies local workspace, owner workspace, Studio, notes, MCP, ACP, sandbox, artifact, source mutation, extension writable destination, and removed full-media traffic as forbidden while `shared` is active.
- Added deterministic desktop/mobile recipient interaction coverage for navigation, source selection/search/filtering, preview, chat, citations, reload persistence, mobile tabs, revoked fail-closed state, and observed canonical operations.
- Updated the pagination matrix, both current sharing-guide copies, and the final UAT runner. The docs describe recipient-owned transcripts, provider disclosure limits, recipient credentials, frozen source scope, authorization/revocation boundaries, read/chat-only behavior, canonical routes, and Task 11 live-truth ownership.
- Regenerated the OpenAPI fingerprint from the actual FastAPI application. Generated schema artifacts contain the workspace/source/preview/history/chat recipient contracts and no `SharedMediaResponse` or removed recipient media path.
- No production defect was found and no production code was changed.

## TDD Evidence

Backend RED/GREEN:

```text
Initial focused matrix: 9 failed, 19 passed.
The failures identified test-harness contract mistakes in prompt-budget fixtures and dependency overrides; no production change was required.

Final focused matrix: 28 passed, 6 warnings in 27.26s.
```

The backend matrix seeds two shared sources plus unrelated owner media, owner note/chat, and recipient local-workspace sentinels. Assertions verify that no sentinel crosses the recipient boundary.

Browser RED/GREEN:

```text
Initial environment RED: 3 failed because the Playwright Chromium build was absent.
Chromium was installed through Playwright.

First completed Task 10 E2E run: 2 passed, 1 failed.
The failure was a test-only expectation that reload must call /chat/messages even though the UI hydrates persisted messages through the canonical workspace bootstrap.

Final exact E2E run: 3 passed in 25.5s.
```

The final ledger still requires observed canonical workspace, sources, preview, and chat operations, verifies the reloaded transcript, and fails on every prohibited destination category.

## OpenAPI Verification

The exact unactivated command was run first and failed before schema export because `generate-api-types` resolved the host `python3`, which cannot evaluate the project's PEP 604 type alias. With the repository `.venv` activated, generation completed:

```text
Python 3.11.13
2011 paths, 2961 schemas
fingerprint sha256: 9ae55783aa50...
openapi-typescript 7.13.0 completed
```

`bun run verify:openapi` passed:

```text
Verified 328 ClientPath entries; all paths present.
Verified 49 MEDIA_ADD_SCHEMA_FALLBACK fields.
10 unrelated reviewed OSS exception paths remain allowed by the verifier.
```

Static generated-schema checks found the five canonical recipient operation families and no `SharedMediaResponse`, removed recipient media path, or full-media path.

## Backend Gate

The first exact backend command attempted the repository fixture's Docker auto-start and timed out while polling because Docker was unavailable. The controller ruled that Docker must not be attempted again. A prior no-Docker run reached 58% before controller interruption; its stale process was terminated and it has no valid completion result.

The authoritative rerun used the unchanged required selection with `TLDW_TEST_NO_DOCKER=1`:

```text
519 collected
508 passed, 11 skipped, 19 warnings in 1178.33s (19m38s)
```

All 11 skips are standard repository fixture-unavailable signals. A supplemental `-rs` run recorded:

- 3 skips in `test_chacha_postgres_migration_v61.py`: `Postgres not reachable; skipping Postgres-backed tests`.
- 6 skips in `test_authnz_sharing_postgres.py`: `PostgreSQL not available; attempted docker start; skipping AuthNZ integration tests. Set TLDW_TEST_POSTGRES_REQUIRED=1 to enforce.` This is the fixture's fixed unavailable message; the ruled rerun had `TLDW_TEST_NO_DOCKER=1` and did not retry Docker.
- 2 skips in `test_shared_workspace_chat_store_postgres.py`: `Postgres not reachable; skipping Postgres-backed tests`.

The supplemental reason run completed `3 passed, 11 skipped`.

## Frontend Gates

Exact Vitest command:

```text
65 files: 62 passed, 3 failed
848 tests: 835 passed, 13 failed
```

The 13 failures exactly reproduce the documented untouched baseline: 7 in `SourceViewControls.test.tsx`, 5 in `ResearchWorkspace.stage12.source-list-view-state.test.tsx`, and 1 incomplete `SourcesPane.stage2.test.tsx` fixture. Shared recipient component, reducer, route-gate, safe Markdown, responsive/accessibility, domain-client, and locale-mirror tests passed. Task 10 changed none of the three failing files or their product implementation.

Exact frontend typecheck completed with existing diagnostics in `PromptDiff.tsx` and skills-certification scripts. No diagnostic references either Task 10 E2E file or a shared-recipient contract file.

The exact ESLint command is blocked by the repository ESLint 9 configuration because the external `../packages/ui/src/components/Option/ResearchWorkspace` directory argument is wholly ignored. Focused ESLint over both Task 10 TypeScript files exited 0 after the final edit.

The exact design-state verifier reported the existing repository blocked/stale baseline across Settings, Skills, Quiz, Watchlists, and previously introduced Research Workspace labels. Task 10 changes no product UI source. The touched E2E/page-object scope contains no product-state rendering.

The exact Playwright command passed all 3 Chromium tests in 25.5 seconds.

## Security And Final Gates

The exact required Bandit command completed successfully. `/tmp/bandit_task_12020_40.json` reports:

```text
46,442 lines scanned
0 findings
0 errors
```

Additional final checks:

```text
Ruff on the three changed backend tests: passed.
Focused backend matrix after import cleanup: 28 passed.
Focused ESLint after final E2E correction: passed.
git diff --check: passed.
Current and published sharing guides: byte-identical.
Generated OpenAPI obsolete-operation scan: no matches.
```

## Files Changed

- `.superpowers/sdd/2026-08-21-recipient-shared-research-workspace-data-plane/progress.md`
- `.superpowers/sdd/2026-08-21-recipient-shared-research-workspace-data-plane/task-10-implementer-report.md`
- `Docs/Design/Pagination_Completion_Matrix.md`
- `Docs/Development/Research_Workspace_Final_UAT_Runner.md`
- `Docs/Published/User_Guides/Server/Organizations_and_Sharing.md`
- `Docs/User_Guides/Server/Organizations_and_Sharing.md`
- `apps/tldw-frontend/e2e/utils/page-objects/ResearchWorkspacePage.ts`
- `apps/tldw-frontend/e2e/workflows/research-workspace.shared-recipient.spec.ts`
- `apps/tldw-frontend/lib/api/openapi.fingerprint.json`
- `backlog/tasks/task-12020.40 - Bind-recipient-shared-workspace-sources-and-chat-to-the-canonical-share.md`
- `tldw_Server_API/tests/Sharing/test_cross_user_access.py`
- `tldw_Server_API/tests/Sharing/test_shared_workspace_recipient_security_matrix.py`
- `tldw_Server_API/tests/Sharing/test_sharing_endpoints.py`

## Scope And Concerns

- Task 11 remains the live backend/provider/browser truth gate. This task's stubbed Playwright spec certifies interaction and request-ledger behavior only.
- Live PostgreSQL execution was unavailable. Deterministic PostgreSQL DDL/RLS contracts passed, while the 11 live cases skipped only through repository fixture-unavailable signals under the controller ruling.
- Repository-wide Vitest, typecheck, ESLint command composition, and design-state baselines remain non-green as classified above. No Task 10 touched-scope failure remains.
- `/research` remains separate. No redirect, alias, local fallback, owner sentinel, recipient local sentinel, source mutation, or extension writable-destination contract was added.
- The unrelated watchlist templates remain untouched and unstaged.

## Fix Round 1/5 - Reviewed Head `ba0b3e02f9a6`

This round addresses all six blocking findings in `task-10-review.md`. It supersedes the initial report's description of the backend matrix and browser ledger where that description implied production-boundary coverage that the reviewed implementation did not provide. No production code changed and no production defect was found.

### Production-boundary matrix

`test_shared_workspace_recipient_security_matrix.py` now drives the real FastAPI sharing routes and production recipient services. The fixture uses the production access-service builder, `SharedWorkspaceAccessService`, `SharedWorkspaceChatService`, real `CharactersRAGDB` owner/recipient stores, and real `MediaDatabase` source rows. Production helpers perform source list/status/default/history/preview reads, frozen snapshot resolution and revalidation, owner/recipient ChaCha selection, owner media selection, canonical target resolution, recipient credential resolution, receipt transitions, and chat persistence.

Only external retrieval/provider transport is deterministic. The fixture also supplies deterministic user and BYOK/default-model stores because the repository-global stores are unavailable in the isolated fixture; production target normalization, adapter validation, credential scoping, and route/service orchestration remain active. Assertions prove the recipient identity is used for credentials and recipient persistence, the owner identity is used for source/media retrieval, and owner-note/chat, unrelated-owner-media, and recipient-local-workspace sentinels never appear.

The matrix covers all/include source scope, completed replay, mismatched fingerprints, revocation, membership suspension, source deletion, and media hash/version changes after retrieval or generation. Each lifecycle mutation fails before persistence and leaves the receipt in the contractually correct retryable/conflicted state.

TDD evidence:

```text
Initial production-boundary GREEN attempt: 12 failed, 3 passed.
Cause: production builder reached the global AuthNZ user repository, which is incompatible with the deterministic fixture pool.

Second attempt: 6 failed, 9 passed.
Causes: external BYOK override-store resolution remained active and a direct media-hash update violated the production version trigger.

Third focused cycle exposed one remaining default-model store lookup; after constraining only that external store seam and using versioned media updates, the final focused matrix passed:
15 passed, 6 warnings in 30.23s.
```

The concurrency case now launches two matching requests that genuinely overlap. A deterministic provider barrier holds the first request after claim; the second arrives while the first lease is active. The clock and lease are current and controlled rather than fixed historical values. Assertions require one generation, one persisted user/assistant pair, one completed receipt, a conflict/active result for the loser, and a completed replay after the winner releases the barrier.

### Removed route runtime and schema absence

`test_sharing_endpoints.py` directly requests every removed recipient media URL with redirect following disabled:

```text
GET /api/v1/sharing/shared-with-me/12/media
GET /api/v1/sharing/shared-with-me/12/media/99
GET /api/v1/sharing/shared-with-me/12/full-media
GET /api/v1/sharing/shared-with-me/12/full-media/99
```

Each returns exact 404 with no `Location` header. The OpenAPI test separately asserts absence of all four templated paths, any `SharedMediaResponse` schema, and any operation ID containing that obsolete response name.

Focused verification:

```text
python -m pytest tldw_Server_API/tests/Sharing/test_sharing_endpoints.py -q -k 'openapi or removed'
5 passed, 52 deselected.
```

### Strict browser ledger and mobile/revoked flow

The page-object ledger now records from explicit `startRequestLedger()` until disposal even if navigation removes `shared` from the current URL. It detects removed recipient media paths independently of share ID and classifies the real production destinations, including `/api/v1/research-workspace/artifacts/generate` and `/api/v1/web-clipper/save`, plus artifact, web-clipper/capture, local workspace, Studio, notes, MCP, ACP, sandbox, source/media mutation, ingestion, and extension writable-destination families.

An explicit method/path allowlist is mandatory. Every undeclared API request, `requestfailed`, unexpected HTTP response at or above 400, page error, and console error fails the ledger. The revoked workspace 404 is declared explicitly; unknown stubs abort rather than quietly returning 404. A regression changes the address bar with `history.replaceState` and proves the ledger still rejects a real artifact-generation probe afterward.

The mobile project now performs source selection, search/filter/clear, preview sheet inspection, chat and citation interaction, reload transcript persistence, and revoked bootstrap. Revocation assertions require the exact canonical `/research-workspace?shared={share_id}` URL to remain, while shared shell/source/message content and local fallback/sentinels are absent and the request ledger remains clean.

Browser TDD and exact gate:

```text
RED: the initial production-path classifier did not reject the real artifact-generation URL.
Subsequent strict runs exposed undeclared ambient shell reads, aborted superseded source requests, notification-poll cancellation, and the expected revoked-404 console diagnostic; each was resolved without weakening unknown-request/error handling.

bunx playwright test e2e/workflows/research-workspace.shared-recipient.spec.ts --project=chromium --reporter=line --workers=1
4 passed in 32.6s.
```

This remains stubbed CI interaction coverage only. Task 11 remains live backend/provider/browser truth.

### Authoritative backend completion and Docker ruling

Docker was unavailable after the repository fixture's earlier auto-start timeout and was not attempted again. The controller-interrupted prior `TLDW_TEST_NO_DOCKER=1` run stopped at 58% and has no valid completion result.

The exact Task 10 backend selection was rerun once to completion with `TLDW_TEST_NO_DOCKER=1`:

```text
523 collected
512 passed, 11 skipped, 22 warnings in 2466.53s (41m06s)
```

All 11 skips are the standard repository fixture-unavailable signals. The exact supplemental reason audit completed `3 passed, 11 skipped`:

- 3 skips in `test_chacha_postgres_migration_v61.py`: `Postgres not reachable; skipping Postgres-backed tests`.
- 2 skips in `test_shared_workspace_chat_store_postgres.py`: `Postgres not reachable; skipping Postgres-backed tests`.
- 6 skips in `test_authnz_sharing_postgres.py`: `PostgreSQL not available; attempted docker start; skipping AuthNZ integration tests. Set TLDW_TEST_POSTGRES_REQUIRED=1 to enforce.` This is fixed fixture wording; the command set `TLDW_TEST_NO_DOCKER=1` and no Docker command was run.

### Exact and final gates

OpenAPI generation and verification:

```text
source ../../.venv/bin/activate && bun run generate:api-types
2011 paths, 2961 schemas; fingerprint sha256 9ae55783aa50...; passed.

bun run verify:openapi
328 ClientPath entries and 49 media-add fallback fields verified; passed.

Generated scan for SharedMediaResponse, full-media, and removed recipient media paths: no matches.
```

Frontend gates:

```text
Exact Vitest: 62 files passed, 3 failed; 835 tests passed, 13 failed.
The failures exactly reproduce the untouched baseline: 7 SourceViewControls, 5 Stage 12 source-list-view-state, and 1 incomplete SourcesPane Stage 2 fixture. Shared recipient, route gate, responsive/accessibility, safe Markdown, domain client, and locale tests passed.

Exact typecheck: existing PromptDiff missing `diff` module and skills-certification script diagnostics only; no touched-file diagnostic.

Exact ESLint: repository ESLint 9 rejects the external ../packages/ui ResearchWorkspace directory as wholly ignored.
Focused ESLint on the two changed E2E files: exit 0.

Exact design-state: existing blocked/stale baseline outside this round; no product UI source changed.
```

Security and final checks:

```text
Required Bandit scope: 46,442 LOC, 0 findings, 0 errors.
Ruff on both changed backend tests: passed.
git diff --check: passed.
Current contract scan: no active obsolete recipient operation.
OpenAPI regeneration caused no additional tracked artifact change.
```

### Fix-round files

- `.superpowers/sdd/2026-08-21-recipient-shared-research-workspace-data-plane/progress.md`
- `.superpowers/sdd/2026-08-21-recipient-shared-research-workspace-data-plane/task-10-implementer-report.md`
- `apps/tldw-frontend/e2e/utils/page-objects/ResearchWorkspacePage.ts`
- `apps/tldw-frontend/e2e/workflows/research-workspace.shared-recipient.spec.ts`
- `backlog/tasks/task-12020.40 - Bind-recipient-shared-workspace-sources-and-chat-to-the-canonical-share.md`
- `tldw_Server_API/tests/Sharing/test_shared_workspace_recipient_security_matrix.py`
- `tldw_Server_API/tests/Sharing/test_sharing_endpoints.py`

The unrelated watchlist templates remain untouched and unstaged.

## Fix Round 2/5 - Reviewed Head `a4e78b3ca2`

This round addresses every Important and both Minor findings in
`task-10-rereview-round-1.md`. No production backend or frontend source changed.
Docker was not attempted, and the prior authoritative Task 10 aggregate remains
`512 passed, 11 standard fixture-unavailable PostgreSQL skips, 22 warnings`.

### Production BYOK and lifecycle boundaries

The security fixture no longer replaces either imported
`resolve_byok_credentials` reference. It keeps the production resolver active
and replaces only deterministic user/shared credential repositories and config
lookups. Repository rows use the production encrypted BYOK envelope. The owner
has a distinct private user key; the recipient resolves user, then team, then
organization credentials under explicit recipient scope. Assertions require
every repository identity to be the recipient for chat, exact team/org scope
inputs, and provider transport propagation of the recipient team API key,
project config, and `user_identifier="2"`.

The lifecycle matrix now independently requires zero provider calls after every
post-retrieval authority/source mutation and exactly one provider call after
every post-generation mutation. Both groups retain empty persistence and
retryable/conflicted receipt assertions.

TDD evidence:

```text
RED production-resolver sentinel: 1 failed, 15 deselected; the active fixture
still replaced sharing.resolve_byok_credentials with its test double.
GREEN exact matrix: 17 passed, 7 warnings in 31.21s.
Focused lifecycle checkpoint selection: 8 passed, 9 deselected in 17.21s.
```

### Browser ledger, fallback, and history

The global 404 counter is gone. Expected Chromium diagnostics are accepted only
when the console source URL exactly equals a declared expected 404 response URL.
The regression observes both the API-sourced diagnostic and an unrelated generic
diagnostic with a different source, then requires `assertClean()` to fail.

Three initial pre-fix probe attempts did not produce a valid RED because
Chromium itself emitted the expected 404 diagnostic, leaving the injected error
visible even to the old counter. The probe series stopped at the required limit.
After removing the counter, the full browser run produced the useful integration
RED: `1 failed, 4 passed`, with the revoked expected 404 diagnostic rejected.
Exact URL/source correlation made that flow green. A later focused diagnostic
assertion RED (`1 failed`) recorded Chromium's injected-console source as the
empty source rather than the document URL; the corrected exact-source assertion
passed (`1 passed`).

The browser seed now uses the real `tldw-workspace` split-storage index and
workspace snapshot key with a recipient-local workspace, source, note, and
sentinel. Revoked bootstrap requires the canonical shared URL while excluding
the local workspace header, sources panel, chat main content, Studio panel,
workspace name, source/note sentinel, and every shared shell/message surface.

Bootstrap now supplies `next_before=older-history-cursor`. Desktop and mobile
both invoke **Load older messages**, receive a page with unique message IDs,
observe the older assistant message, and assert the canonical
`GET .../chat/messages` operation and cursor.

Final exact browser gate:

```text
bunx playwright test e2e/workflows/research-workspace.shared-recipient.spec.ts --project=chromium --reporter=line --workers=1
5 passed in 30.4s.
```

### Documentation and focused gates

Both sharing-guide copies now describe the table as the recipient shared
Research Workspace data-plane operation set and explicitly keep clone outside
that set under the share clone policy. `cmp` confirms byte identity.

```text
TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_recipient_security_matrix.py -q
17 passed, 7 warnings in 31.21s.

TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Sharing/test_sharing_endpoints.py -q -k 'openapi or removed'
5 passed, 52 deselected, 6 warnings in 7.00s.

python -m ruff check tldw_Server_API/tests/Sharing/test_shared_workspace_recipient_security_matrix.py
All checks passed.

bunx eslint e2e/workflows/research-workspace.shared-recipient.spec.ts e2e/utils/page-objects/ResearchWorkspacePage.ts
Exit 0.

cmp Docs/User_Guides/Server/Organizations_and_Sharing.md Docs/Published/User_Guides/Server/Organizations_and_Sharing.md
Exit 0.

Generated-client obsolete-operation scan for SharedMediaResponse and removed
recipient media/full-media paths: no matches.
```

Bandit was not rerun because this round changes no production Python. The exact
Task 10 production scope remains covered by the prior zero-finding, zero-error
46,442-LOC Bandit result. Task 11 remains the live PostgreSQL/provider/browser
truth gate. The two unrelated watchlist templates remain untouched and unstaged.
