# Cycle4 Task1 / UAT103 and UAT111 — independent review

## Final scoped re-review — both P2 findings resolved

**Result: clear for parent integration/native acceptance within the reviewed ACK scope. No remaining actionable finding in these two corrections.** This re-review is bounded to the two production and two adjacent test changes since the original review; the historical findings below are retained as evidence, not open issues.

- `server-chat-mirror.ts` now excludes only exact empty-string image sentinels before comparing arrays. The original `[""]` versus `[]` text-only case recovers the anchored user, while unrelated equal-text drafts remain unacknowledged. Permanent tests exercise both stored and visible reconciliation forms, and actual nonempty image mismatches, missing/conflicting anchors, existing canonical users, content edits, and owner guards remain covered. No broad text/image deduplication was added.
- `useChatActions.ts` now assigns the canonical assistant ID from both successful fallback `addChatMessage` and `saved:true` degraded outcomes before local persistence. Missing response IDs remain undefined; no extra server write or acknowledgement is fabricated. Permanent real-orchestration controls assert both canonical IDs, saved status, and exactly2 writes for user+fallback versus1 user write for a degraded already-saved result.

### Fresh independent evidence

The original private probe configs were rerun unchanged:

- `/private/tmp/cycle4-chat-ack-legacy-empty-image.config.ts`: **1 passed /24 filtered**; log `/private/tmp/cycle4-chat-ack-rereview-image.log`.
- `/private/tmp/cycle4-chat-ack-character-fallback.config.ts`: **2 passed /15 filtered**; log `/private/tmp/cycle4-chat-ack-rereview-character.log`.
- The author's exact focused command (mirror, Character orchestration, saved-normal integration, success/error persistence, loader mirror and actual Note menu) independently passed **126 tests /7 suites**, exit0,16.65 seconds; log `/private/tmp/cycle4-chat-ack-rereview-focused.log`. Counts overlap with the probes and prior runs.
- Scoped `git diff --check` passed for the four correction files.
- Refreshed22-path manifest matched except `chat_service.py`, whose separately authorized configured-model inventory change is explicitly excluded. Comparison to the pre-review manifest confirms only the two frontend production/two test files and official task notes changed in this correction.

No new full compiler, backend, lint or Bandit run is claimed for this rereview. Earlier independent251frontend/153backend+1existing skip evidence and author's static evidence remain the prior checkpoint. These corrections are frontend-only. Parent owns combined compiler/static/native acceptance. No repository edits, commits, runtime, browser, inference or subagent actions were performed.

The parentless captured legacy mirror remains conservatively unmodified and guarded; safe anchored recovery is the supported repair. Controlled table/API tests do not constitute native IndexedDB or browser reload acceptance. The ordinary saved send/reload and Character native checks remain parent-owned/pending. Pending Task5 UAT108/113/117 behavior and Task48 model inventory remain outside this review.

---

## Original review findings (resolved above)

**Result: two actionable P2 corrections.** Read-only review of the supplied manifest against f4e9f954d7; no repository edits, commits, runtime changes, inference, browser operations, or subagents. Scope excludes the configured-model inventory work and pending UAT108/113/117 behavior changes.

## P2 — Normalize empty image placeholders before anchored legacy matching

**Location:** apps/packages/ui/src/db/dexie/server-chat-mirror.ts:60–61.

The new recovery requires literal JSON equality of the image arrays. A real text-only local save passes `images: [image]` with `image === ""` (chat-helper/index.ts:573; the error paths use the same representation), and dexie/helpers.ts:175–181 persists that array unchanged. The real server adapter emits `images: []` for an ordinary text row (hooks/chat/useServerChatLoader.ts:440).

Consequently, an otherwise defensible legacy row with an exact text match and a unique canonically acknowledged reply pointing to its local parent fails recovery. Reconciliation inserts a separate canonical user instead; the original user remains unacknowledged, so the visible duplicate and unsaved-work backlink warning persist. The new test's `images: []` local fixture misses the actual prior save shape.

**Independent proof:** /private/tmp/cycle4-chat-ack-legacy-empty-image.config.ts injects a private case into the existing mirror test at its original module identity. It uses the actual reconciliation with the real text-only persisted shape, a retained acknowledged assistant parent, and an unrelated equal-text draft. Expected3 rows, actual4: local-q remains unacknowledged and `alice:server:question` is added. /private/tmp/cycle4-chat-ack-legacy-empty-image.log: **1 failed /20 filtered**.

**Bounded correction:** normalize empty image placeholders for this comparison while preserving actual image distinctions and every existing provenance/ambiguity guard. Exercise the real local save shape in permanent recovery coverage. This does not justify deleting parentless ambiguous legacy work.

## P2 — Forward canonical assistant IDs from successful Character fallback outcomes

**Location:** apps/packages/ui/src/hooks/chat/useChatActions.ts:2708 (the new `assistantServerMessageId` argument), with assignments missing in successful branches at2585–2607 and2609–2640.

The new `persistedAssistantServerMessageId` variable is assigned only when `persistCharacterCompletion` resolves normally. Two existing success paths obtain a canonical assistant ID but leave that variable undefined: a `persist_validation_degraded` error with `saved: true`, and the successful `addChatMessage` fallback. Both update the visible assistant and set `assistantPersistedToServer`, then save the local message with `assistantServerMessageId: undefined` and `serverMessagesAlreadyPersisted: true`.

The local persistence helper therefore loses the already-confirmed assistant identity. On reload its unacknowledged local assistant can coexist with the canonical server assistant, reproducing the mirror identity problem for these successfully saved turns. The user ID is correctly retained.

**Independent proof:** /private/tmp/cycle4-chat-ack-character-fallback.config.ts injects two cases into the real useChatActions Character orchestration test. Controlled API results cover both the successful fallback write and the saved=true degraded acknowledgement, and the actual orchestration's local save call receives undefined instead of `fallback-assistant`. /private/tmp/cycle4-chat-ack-character-fallback.log: **2 failed /13 filtered**. This probe controls the local persistence boundary, consistent with the existing Character integration fixture; the actual save helper's treatment of the argument was reviewed separately.

**Bounded correction:** assign the same canonical assistant variable in both successful outcome branches before forwarding it. This is acknowledgement propagation in Task1; no retry-policy, route restoration, or missing-answer behavior expansion is requested.

## Independent verification and source assessment

- All manifest source/test/task SHA-256 hashes matched at review start. Review used f4e9f954d7 as requested. Concurrent unrelated configured-model inventory work in chat_service.py is excluded from any findings.
- Supplied broader frontend command independently passed **251 tests /22 suites** (32.28 seconds): /private/tmp/cycle4-chat-ack-independent-frontend.log. It includes model metadata/abort/interruption, actual normal actions/autosave/pipeline/persistence integration, Character orchestration, mirror/loader ownership, and actual Note menu guards.
- Supplied backend command independently passed **153 tests /1 existing skipped**, exit0 (59.73 seconds): /private/tmp/cycle4-chat-ack-independent-backend.log. The existing asynchronous coordination skip and temporary-directory cleanup warnings remain. Provider inference is mocked; no live model call was made.
- Backend `build_context_and_messages` captures the last newly persisted user ID after overlap trimming and passes it through runtime context to both response modes. Streaming metadata and nonstream response payloads retain canonical IDs; real endpoint tests confirm each of two identical turns obtains its own user ID and the server keeps exactly5 rows. Ephemeral controls assert no acknowledgement or database writes.
- ChatTldw clears IDs per stream, ignores metadata after the captured signal aborts and for explicit ephemeral requests; pipeline forwarding updates the matching local user without changing its primary key. Scoped normal persistence and A→B/A→B→A controls pass. Primary Character success forwards both IDs; the two alternative success paths above are the exception.
- Reconciliation remains owner-bound and conservative about conflicting parents, already mirrored users, edits, missing anchors and ambiguous equal text. New normal send → local acknowledgement → five-row reconciliation and intentional identical-question/draft controls pass.
- Notes production guard remains unchanged. The real menu tests accept acknowledged normal turns and continue to warn/block a genuinely unsent equal-text draft. Neither finding calls for bypassing that guard.
- No independent full-workspace lint/typecheck/Bandit rerun is claimed. The implementer's reported TS90 baseline, ESLint0errors/71unchangedwarnings, Bandit0 and Ruff baseline comparison are supplied evidence; broad backend source is concurrently being edited outside this review scope.

## Honest legacy and native limits

The captured cycle4 damaged mirror already has null assistant parent links. Its saved-but-unacknowledged users cannot safely be separated from genuine equal-text drafts. Preserving those rows and retaining the warning is an explicit, justified limit; neither correction should broadly deduplicate text or claim to repair that exact old mirror automatically.

The mirror/loader tests replace IndexedDB with controlled tables, and normal pipeline tests control API/storage boundaries. They are meaningful interacting regressions but not a native browser reload. Character persistence coverage controls the local save function while asserting its canonical-ID arguments. The Note menu uses real handlers with controlled API/authority. Parent-owned freshly sent/reloaded normal and Character native checks remain pending; no live acceptance is claimed.

## Reproduction

From apps/packages/ui:

```sh
./node_modules/.bin/vitest run --config /private/tmp/cycle4-chat-ack-legacy-empty-image.config.ts -t 'independent legacy text-only' --maxWorkers=1 --no-file-parallelism
./node_modules/.bin/vitest run --config /private/tmp/cycle4-chat-ack-character-fallback.config.ts -t 'independent character fallback' --maxWorkers=1 --no-file-parallelism
```

The full frontend and backend commands are identical to /private/tmp/cycle4-chat-ack-implementation.md, with frontend workers limited to1. Private probe configs/logs are frozen for correction/re-review. Only temporary review artifacts were written.
