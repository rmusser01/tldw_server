# UAT122 / TASK13260.62 — prevent silent image removal

## Status and freeze

Implementation is frozen for independent review and parent-owned native verification. No browser, inference, runtime restart, staging or commit was performed by this author. Task remains In Progress; the previous UAT118 native image acceptance remains blocked/pending, not passed.

Base revision: `6595bcfff4979d2495f34be39ebe18792d07a732`. Source/test/config freeze: **2026-09-16T07:16:41.108094Z**.

- `/private/tmp/cycle4-uat122-production-manifest.json`: three production paths/hashes.
- `/private/tmp/cycle4-uat122-owned-manifest.json`: seven source/test/config hashes, plus official task-record hash added after notes.
- `/private/tmp/cycle4-uat122-paths.json`: exact categorized inventory.
- `/private/tmp/cycle4-uat122-final-hash-check.json`: all seven frozen code/test/config files still match.

Approved design: `/private/tmp/cycle4-uat122-image-input-design.md`, retained by root as `Docs/Design/2026-09-16-uat-image-input-validation.md`. Parent approved the third production hunk after independent Retry evidence, described below.

## Root cause and repair

Native composer/local transcript retained a real PNG, but `ChatTldw.convertToTldwMessages` converted the HumanMessage to text whenever `supportsMultimodal` was false. The actual POST therefore lacked the image, and canonical user `9437ffc0-ce39-45bf-b120-fc2fc99f35eb` contained `images:[]`. This was attachment loss before transport, not a failed server restoration.

The actual model factory treats absent model information/capabilities and catalog failures as false. The new message therefore says **“Image support is not confirmed for this model.”** It does not claim the provider definitively lacks vision.

The model now refuses any outgoing HumanMessage image part without confirmed vision, before streaming or invoke transport. It checks the entire user history, so changing models cannot silently remove earlier images. Pure text and supported image payloads retain their existing behavior. The user sees the existing compact model-picker recovery action and guidance to choose a model supporting images or start a new text-only conversation. Original local text/image remains retained through failure and repeated Retry.

### Separate local identity from server Retry

Independent API/core evidence showed that a never-dispatched new turn can be identical to an older answered turn. Sending `retry_failed_turn=true` after the local refusal incorrectly asks the server to reuse that older turn and yields409.

The trusted local `ImageSupportUnconfirmedError` now captures the model's current server-retry intent. The existing friendly error envelope preserves an optional boolean `serverRetryRequired`. Only this class can mint the boolean during error mapping; matching provider text, named Error objects and arbitrary objects cannot establish that dispatch never occurred.

The pipeline still uses its existing local `retryFailedTurn` for user identity and ACK updates. Only the model-dispatch flag differs: an explicit stored false disables canonical failed-turn reuse. Missing/malformed provenance remains conservative. An initial refusal stores false; repeated refusal/remount retains false; a later actual transport failure has no such local proof, so its next Retry requires server reuse. A server/ambiguous failure followed by a local refusal keeps true. No inference is made from an absent ACK, and no backend correlation guard changed.

### Formatter and OCR limits

Actual formatter probes disproved a separate custom image-only defect: empty text falls through to the original image-containing content. No formatter production change was made.

Explicit OCR of the current message remains an intentional image-to-text conversion and is allowed. The prior implementation, however, stored original images in local history and silently removed them on a later text-only-model request, already losing extracted OCR context. Parent explicitly rejected automatic re-OCR/history exemptions. Such unconverted historical images now produce a truthful local refusal; the existing raw local image is retained. This compatibility boundary is permanent-tested and independently documented. Complete OCR-history reconstruction is not claimed.

## Owned files

Production:

1. `apps/packages/ui/src/models/ChatTldw.ts` — pretransport image guard.
2. `apps/packages/ui/src/utils/chat-error-message.ts` — trusted local error, actionable friendly message, optional boolean serialization.
3. `apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts` — separate server-dispatch Retry flag, preserving local user/ACK behavior.

Tests/config:

4. `apps/packages/ui/src/models/__tests__/ChatTldw.image-input.test.ts` — new actual formatter/model/factory controls.
5. `apps/packages/ui/src/utils/__tests__/chat-error-message.test.ts` — recovery/provenance/lookalike controls.
6. `apps/packages/ui/src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx` — actual image send/Retry/factory/mirror boundaries and existing owner/autosave controls.
7. `apps/packages/ui/vitest.config.ts` — one test-only alias to the already-installed frontend `pa-tesseract.js`, matching Next's existing resolution. No package/lockfile/install change; OCR worker remains controlled in tests.

Official task62 notes updated through Backlog CLI. No Playground, backend, schema, registry, normalChatMode production or shared tracker/plan edits.

## RED evidence

All artifacts below are under `/private/tmp/`:

- `cycle4-uat122-formatter-probe.log`: private unchanged-production actual formatter/model probe, **3 expected failures / 4 positives**. False/default capabilities lose images; supported and custom image-only formatting preserve them.
- `cycle4-uat122-unit-red.log`: permanent real formatter/factory/model plus friendly-error controls, **9 failed / 9 passed** before production change.
- `cycle4-uat122-mounted-red.log`: corrected mounted harness, **4 expected failures / 57 unselected** before the guard; transport was reached despite unsupported images, including delayed owner controls. An earlier fixture run lacked a mocked module export and is separately retained as `cycle4-uat122-mounted-fixture.log`; that fixture error is not product RED evidence.
- `cycle4-uat122-guard-green-provenance-red.log`: after guard/error mapping but before pipeline correction, **4 failed / 29 passed / 57 unselected**. Initial blocking and owner controls pass, but repeated refusal incorrectly changes explicit server-retry false to true.
- `cycle4-uat122-ambiguous-red.log`: additional transition run; the initial-local-refusal case reproduced wrong true dispatch. A companion case had an invalid synthetic PNG fixture that triggered the existing JPEG fallback; it was corrected to valid PNG bytes, not treated as a product MIME failure.

Do not add these overlapping development counts. The independent backend proof and false-flag control are `/private/tmp/cycle4-uat122-unacked-retry-independent.py` / `.log` (identical-prior409) and `cycle4-uat122-unacked-newturn-control.py` / `.log` (**3 passes** with only server Retry false). Author made no backend changes.

## Final verification — 150 unique tests / 14 files

Run from `apps/packages/ui` with `./node_modules/.bin/vitest run`, `--maxWorkers=1 --no-file-parallelism`.

### Core: 93 passed / 3 files

`/private/tmp/cycle4-uat122-core-green.log`:

```sh
./node_modules/.bin/vitest run \
  src/models/__tests__/ChatTldw.image-input.test.ts \
  src/utils/__tests__/chat-error-message.test.ts \
  src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

The 15 model tests use the real formatter and factory capability decision. The 12 error tests cover explicit false/true round-trip, missing/malformed provenance, provider lookalikes and existing model/empty-answer recovery. The 66 mounted tests include nine new cases and all 57 existing saved-normal controls.

Four mounted cases cover image-only/text+image, with/without an identical previously answered image turn: local block, persisted/remounted error, repeated local Retry, real factory switching to vision, exact payload and same local user ID, canonical acknowledgement and remount retaining the final user/assistant. Two further cases cover actual failure before ACK with/without earlier local refusal, subsequent unsupported-model refusal, and server Retry true recovery. A→B and A→B→A preserve the new owner's work after delayed catalog completion. Explicit current OCR plus historical-image refusal has a permanent two-turn case.

### Transport/factory compatibility: 27 passed / 5 disjoint files

`/private/tmp/cycle4-uat122-adjacent-green.log` executed:

- `src/models/__tests__/ChatTldw.abort-signal.test.ts`
- `src/models/__tests__/ChatTldw.stream-metadata.test.ts`
- `src/models/__tests__/ChatTldw.stream-transport-interrupted.test.ts`
- `src/models/__tests__/pageAssistModel.mcp-tools.test.ts`
- `src/hooks/handlers/__tests__/messageHandlers.regenerate.test.ts`

The initial command also named three nonexistent test filenames; Vitest matched/executed only these five. No pass is claimed for unmatched names. The six actual pipeline/history files below were then run explicitly.

### Pipeline compatibility: 30 passed / 6 disjoint files

`/private/tmp/cycle4-uat122-pipeline-green.log`:

```sh
./node_modules/.bin/vitest run \
  src/hooks/chat-modes/__tests__/chatModePipeline.error-recovery.guard.test.ts \
  src/hooks/chat-modes/__tests__/chatModePipeline.abort-lifecycle.test.ts \
  src/hooks/chat-modes/__tests__/chatModePipeline.provider-recovery.test.ts \
  src/hooks/chat-modes/__tests__/chatModePipeline.dynamic-ui.test.ts \
  src/hooks/chat-modes/__tests__/chatModePipeline.conversation-id.test.ts \
  src/utils/__tests__/generate-history.image-generation.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

### Static/security

- Repository-root explicit frontend ESLint config covered all seven TS/TSX/config files: **0 errors / 11 unchanged baseline warnings**, zero added/removed diagnostic signatures. `cycle4-uat122-eslint-{final,baseline,comparison}.json`, `.log`, and `cycle4-uat122-lint-compare.mjs`. Baseline uses frozen base SHA and actual filenames/config; no ignored-file pass substitution.
- Owned `git diff --check`: exit0, `cycle4-uat122-diff-check.txt`.
- Bandit: not applicable to this TypeScript-only correction; no Python/backend source changed. Parent retains prior backend security results independently.
- Parent reports final integrated **2593 tests / 95 files passed**, and full compiler **exactly 90 existing diagnostics, zero added/removed**. Parent also rechecked all seven frozen hashes. These broader tests overlap the author/reviewer runs and are not summed. There is no clean-typecheck claim; log retention belongs to parent.

## Independent review / remaining limits

**Independent final review is clear:** `/private/tmp/cycle4-uat122-independent-design-review.md`. Reviewer confirmed OCR before/after behavior and the actual backend identical-prior Retry issue, then independently passed all **93 tests / three core suites** on frozen bytes. All seven hashes match; audit is `/private/tmp/cycle4-uat122-independent-final-manifest.json`. These counts overlap author coverage and are not added to the 150 unique tests above.

Native verification is still required. The existing provider advertises no vision support; expected native result is an actionable refusal with no Chat POST and retained original text/image, including Retry. Successful actual vision output remains unverified and no model/provider change is authorized. Old already-damaged text-only canonical/image-local rows retain their mismatch guard; this patch does not guess missing attachments or migrate records.

Mounted tests use real action/pipeline/formatter/model factory/loader/mirror logic with controlled provider transport and a reactive local adapter. They are not real-browser/Dexie/live-vision acceptance. The current single-image frontend history limitation from UAT118 remains unchanged. Root owns stable-runtime native testing, final task acceptance, shared documentation and commits.
