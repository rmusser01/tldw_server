# UAT118 / TASK13260.58 — image-bearing failed Chat recovery

## Disposition

Implementation and tests are frozen for remaining independent service/frontend review and parent-owned native verification. No browser, model inference, runtime restart, staging, or commit was performed by this author. Task remains In Progress; full fresh UAT has not started.

Base: `045a953883b5f1809c61d7c171eda83e05450a9b`. Production/test freeze: **2026-09-16T06:26:58.001485Z**. Exact inventories/hashes:

- `/private/tmp/cycle4-uat118-paths.json`: ten production paths and seven test paths.
- `/private/tmp/cycle4-uat118-production-manifest.json`: ten production hashes.
- `/private/tmp/cycle4-uat118-owned-manifest.json`: production/test hashes, with the final task-record hash added after official CLI notes. Task-note timestamp does not change the production/test freeze.

## Cause and resulting behavior

Canonical listing previously exposed `has_image` without the image bytes. The normal loader therefore projected `images:[]`, which could not safely match an unacknowledged local image turn. Image-only persistence also stored a synthetic attachment label. Two adjacent actual-boundary failures compounded this: normal-mode local persistence relabeled qualified PNG input as JPEG, and Retry trimmed original text and rejected image-only input before checking its attachment.

The authorized existing listing now supports opt-in complete images. Default callers retain their existing response shape. The actual domain adapter and loader preserve the supported single image; the existing exact correlation check can then acknowledge only the matching owned local turn. Explicit Retry compares exact text and ordered image bytes/MIME for both saved tail and overlap, preserving prior successful image history and avoiding duplicated provider context. Changed/ambiguous data fails closed instead of text deduplication or attachment loss.

Image-only empty text is reconstructed only from the unchanged version-1 row, exact synthetic label/count, and server-authored placeholder metadata. Literal labels, edited rows, unknown provenance, or contradictory data do not gain this identity proof. Ordinary sends retain their existing repeated-turn semantics.

## Production scope

| Path | Bounded change |
| --- | --- |
| `tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py` | Optional images response; serializer omits it unless explicitly populated. |
| `tldw_Server_API/app/api/v1/endpoints/character_messages.py` | Existing authorized standard listing opts into complete validated images; default and completion formats unchanged; decoded cap checked before expansion. |
| `tldw_Server_API/app/api/v1/endpoints/chat.py` | Server metadata marks only an actual image-attachment placeholder during existing atomic user persistence. |
| `tldw_Server_API/app/core/Chat/chat_service.py` | Exact Retry image/text components, guarded placeholder projection, strict history completeness, and matching provider overlap. |
| `tldw_Server_API/app/core/DB_Management/chacha/message_store.py` | Opt-in strict single-statement page/blob read; ordinary reader unchanged. |
| `apps/packages/ui/src/services/tldw/TldwApiClient.ts` | Optional complete image array in the existing server-message type. |
| `apps/packages/ui/src/services/tldw/domains/chat-rag.ts` | Preserve complete opt-in arrays; reject invalid/subset/missing-image responses. |
| `apps/packages/ui/src/hooks/chat/useServerChatLoader.ts` | Bounded complete-image pagination, image-aware projection/history, and rejection of unsupported multi-image users before transcript/mirror mutations. |
| `apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts` | Preserve already-qualified supported image data URL via existing helper; retain raw-payload fallback. |
| `apps/packages/ui/src/hooks/handlers/messageHandlers.ts` | Image-only Retry allowed; preserve exact text, using trim only to detect emptiness. |

No migration, new auth route, request-schema relaxation, new global attachment framework, or provider metadata leakage was introduced. Existing captured authority, cancellation, history generation, canonical permission checks, and character-rendered listing behavior remain in place.

### Read safety

The strict DB path selects the exact message page, computes stored byte totals, and CASE-gates primary/ordered blobs in **one SQL statement**. It needs no repeatable-read transaction assumption or follow-up image queries. Over-budget results expose size information internally without returning blobs; order and limit/offset are unchanged. Ordered positions must be complete and contiguous. The page decoded cap is 32 MiB; the client aggregate encoded cap is 64 MiB and cancellation is checked between pages.

Legacy attachment presence uses non-None bytes or MIME evidence, rather than truthiness. Empty/missing bytes, absent or unsupported MIME, corrupt images, partial ordered rows, DB errors, and excessive size reject the entire opt-in read. Pillow verification is followed by actual pixel decoding to catch truncated JPEG data; existing per-image byte and decompression safeguards remain. No subset is presented as complete.

The frontend's existing history contract supports one image. A canonical user containing multiple attachments produces a clear unsupported-shape load error **before local history/message/mirror mutation**, retaining local work. The API/domain still return the complete array; this unit does not silently select its first image and continue an incomplete conversation.

## Permanent tests and RED evidence

Changed tests:

- `tldw_Server_API/tests/Chat/integration/test_chat_image_recovery.py` (new).
- `tldw_Server_API/tests/Chat/unit/test_chat_history_and_streaming.py` (existing fixture contract/valid image bytes).
- `apps/packages/ui/src/services/tldw/__tests__/TldwApiClient.request-scope.test.ts`.
- `apps/packages/ui/src/hooks/__tests__/useServerChatLoader.images.test.ts` (new).
- `apps/packages/ui/src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx`.
- `apps/packages/ui/src/hooks/handlers/__tests__/messageHandlers.regenerate.test.ts`.
- `apps/packages/ui/src/hooks/chat-modes/__tests__/normalChatMode.overlay.test.ts`.

Actual API/SQLite tests cover failed provider persistence, opt-in listing, explicit Retry and canonical acknowledgement in streaming/nonstreaming paths. Mounted tests exercise actual action pipeline, ChatTldw, domain mapping, loader, mirror and Retry/remount with a controlled transport and reactive local adapter. They cover text+image and image-only, each with/without prior successful image context, and retain the canonical final assistant on remount. They are separate from the Python API tests, not a cross-language/native end-to-end claim.

Meaningful retained RED records (overlapping development runs must not be summed):

- `cycle4-uat118-backend-red.log`: initial actual API run, 8 failed / 5 passed.
- `cycle4-uat118-regenerate-red.log`: exact whitespace and image-only regressions, 2 failed / 2 passed.
- `cycle4-uat118-missing-image-red.log`: actual client rejects neither missing nor empty images despite `has_image`, 2 failed.
- `cycle4-uat118-multi-image-red.log`: actual mounted preservation guard, 1 failed / 56 unselected.
- `cycle4-uat118-legacy-read-red.log`: empty/NULL primary and missing-MIME legacy controls, 3 failed.
- `cycle4-uat118-jpeg-red.log`: truncated JPEG failed while valid JPEG passed, 1 failed / 1 passed.
- `cycle4-uat118-mounted-mime-baseline-red.log`: final mounted tests with **only** pre-fix normalChatMode replayed from the base SHA, 4 failed / 53 unselected. This is a private post-implementation baseline replay, not a full-baseline run. Config: `cycle4-uat118-mime-baseline.config.mts`.

All paths above are under `/private/tmp/`. Earlier frontend/mounted development logs include fixture corrections and are retained as diagnostics; they are not all counted as product RED failures. The MIME replay avoids relying on an overwritten earlier debugging log.

## Final verification on frozen bytes

### Frontend: 186 unique passing tests across 12 files

From `apps/packages/ui`:

```sh
./node_modules/.bin/vitest run \
  src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx \
  src/hooks/__tests__/useServerChatLoader.images.test.ts \
  src/hooks/__tests__/useServerChatLoader.test.ts \
  src/hooks/__tests__/useServerChatLoader.scope.test.tsx \
  src/services/tldw/__tests__/TldwApiClient.request-scope.test.ts \
  src/hooks/handlers/__tests__/messageHandlers.regenerate.test.ts \
  src/hooks/chat-modes/__tests__/normalChatMode.overlay.test.ts \
  src/hooks/__tests__/useServerChatLoader.mirror.integration.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

`/private/tmp/cycle4-uat118-ui-final.log`: **167 passed / 8 files**.

```sh
./node_modules/.bin/vitest run \
  src/models/__tests__/ChatTldw.stream-metadata.test.ts \
  src/models/__tests__/ChatTldw.abort-signal.test.ts \
  src/models/__tests__/ChatTldw.stream-transport-interrupted.test.ts \
  src/services/tldw/__tests__/TldwChat.abort.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

`/private/tmp/cycle4-uat118-transport-compat.log`: **19 passed / 4 disjoint files**.

### Backend: 106 passed, one unavailable PostgreSQL case, across five files

From repository root:

```sh
source .venv/bin/activate && TLDW_TEST_NO_DOCKER=1 python -m pytest \
  tldw_Server_API/tests/Chat/integration/test_chat_image_recovery.py \
  tldw_Server_API/tests/Chat/unit/test_chat_history_and_streaming.py \
  tldw_Server_API/tests/Character_Chat/test_message_listing_raw_content.py -q -rs
```

`/private/tmp/cycle4-uat118-backend-final.log`: **98 passed / 1 skipped**, 78.27s. Official fixture skip: `Postgres not reachable; skipping Postgres-backed tests` at new integration test line264. Docker auto-start was disabled; no separate database was provisioned. This does not establish that a Docker-backed fixture could never be available. Parent owns any further official-fixture availability check.

```sh
source .venv/bin/activate && TLDW_TEST_NO_DOCKER=1 python -m pytest \
  tldw_Server_API/tests/Chat/integration/test_chat_endpoint_image_validation.py \
  tldw_Server_API/tests/Chat/unit/test_chat_history_multi_image.py -q -rs
```

`/private/tmp/cycle4-uat118-image-compat-backend.log`: **8 passed / 2 disjoint files**. Routine deprecation and existing pytest temporary-directory cleanup warnings remain in logs.

### Static/security

- Explicit repository-root frontend ESLint configuration covered all ten owned TS/TSX paths: **0 errors / 806 baseline warnings**, zero added/removed diagnostics. Evidence: `cycle4-uat118-eslint-{final,baseline,comparison}.json`; baseline comparison runner `cycle4-uat118-lint-compare.mjs`. Baseline source was linted with its actual pathname/config against the base SHA.
- Ruff on seven Python paths: **17 existing diagnostics versus 19 baseline**, zero added; two existing import-order findings removed. Evidence: `cycle4-uat118-ruff-{final,baseline,comparison}.json`; runner `cycle4-uat118-ruff-compare.py` using activated venv.
- Bandit on the five production Python paths: **0 findings / 0 errors**, `cycle4-uat118-bandit.json` and `.log`. Existing nosec commentary warnings are retained, not new findings.
- Owned diff whitespace check: exit0, `cycle4-uat118-diff-check.txt`.
- Whole-project compiler/combined validation belongs to parent; no clean whole-typecheck claim is made here.

All static artifacts are under `/private/tmp/`. Bandit used `source .venv/bin/activate && python -m bandit` with the five production Python paths from the manifest and JSON output. No test failure was disabled.

## Independent review

`/private/tmp/cycle4-uat118-read-independent-review.md` clears storage/schema/endpoint after three real legacy/corruption findings were reproduced and corrected. Its final selected run passed **24 tests / 12 deselected**, including four unchanged private probes, at `/private/tmp/cycle4-uat118-read-independent-final.log`. Those tests overlap author coverage and are not added to the unique totals. Storage/schema hashes stayed unchanged; corrected endpoint hash is `a43cd7606709d3ef1cbd886771fe871726551a036938a66182b50325a857850c`.

Remaining service/frontend source review, compiler and native acceptance are parent-owned.

## Limits / remaining acceptance

- No native image inference was performed. Parent's existing provider advertises `vision=false`; successful vision output remains unverified. Failure/reload/Retry preservation can be checked natively without changing providers.
- PostgreSQL SQL execution is unverified here; SQLite actual-query/blob-gating controls passed. The single-statement implementation avoids relying on a PostgreSQL transaction-level snapshot.
- Multiple canonical images per user are intentionally rejected by the current frontend loader before local mutations. They are not silently truncated. General multi-image frontend history support is outside this bounded correction.
- Damaged old image-only rows without trustworthy version/provenance/correlation remain unmatched and protected by existing actionable guards. There is no migration or equal-text deduplication.
- Incomplete/corrupt/over-budget reads fail the complete load and retain local work. Large conversations may require the user to choose another conversation rather than loading a partial transcript.
- Existing request models may discard unknown extra content-part fields before core validation; this patch does not claim a new strict HTTP-schema rejection for such ignored fields. Representable unsupported components/details fail exact Retry validation.
- Mounted frontend tests use controlled transport/local adapter; API persistence tests use temporary SQLite and mocked providers. Native browser/Dexie, live model output, and full fresh workflow acceptance remain separate.
- The new optional response field/query changes OpenAPI; parent owns final generated contract/fingerprint verification.
