# Cycle4 Task1 — canonical Chat acknowledgements (UAT103/111)

Tasks: TASK-13260.44 / TASK-13260.51. Implementation baseline: dev merge `f4e9f954d7`. Source is frozen for independent review. No runtime/browser/model/inference calls, staging, commits, global plan/tracker edits, or subagents.

## Cause and repair

Normal saved Chat persisted user rows but discarded their returned database IDs. Responses/model/pipeline carried only the assistant acknowledgement; local user rows consequently looked unsaved. On reload the owned mirror preserved those unknown rows while adding canonical server rows. The Note backlink correctly refused to discard the resulting apparent unsaved work. Character orchestration already captured the server user ID but omitted it (and the distinct canonical assistant ID) from local persistence.

- Optional `tldw_user_message_id` is emitted alongside existing metadata for the last newly persisted user turn, both streamed and nonstreamed. Strict request validation and ownership are unchanged. Ephemeral responses have no acknowledgement. Existing positional optional streaming arguments remain compatible (the new argument is appended).
- ChatTldw resets/captures user acknowledgement per request, ignores ephemeral/late-aborted metadata; normal pipeline updates the matching local user ID and passes acknowledgement to local success/interruption/error persistence. Character completion forwards its existing canonical user and assistant IDs. Local primary keys stay unchanged.
- Owned mirror recovery recognizes a legacy user only when its local parent relationship is retained by exactly one canonically acknowledged saved reply, the preceding canonical server user matches content/images, and no conflicting parent/canonical user already exists. Unrelated identical text, edits, missing anchors and already duplicated canonical users remain intact. Missing remote parent metadata no longer erases the retained local parent relationship.
- Notes production guard is unchanged. Actual menu tests prove acknowledged neutral Chat opens and a genuinely unsent equal-text draft still warns and blocks.

## Important limitation

The captured cycle4 damaged local mirror already has null parent links on all rows, including its acknowledged assistants. It cannot safely identify original unacknowledged users versus genuine same-text drafts. This implementation does **not** delete those ambiguous rows or claim to repair that exact old mirror automatically. They remain visible as preserved local work; attempting the Note backlink retains the existing actionable “Finish or save the current chat before opening the linked conversation” warning. Forward saved sends and legacy mirrors with retained paired-turn anchors are repaired. Native verification on freshly sent/reloaded normal and character turns is pending with the parent; no live pass is claimed.

## Scope / test fidelity

Exact paths and SHA-256 hashes are in `/private/tmp/cycle4-chat-ack-owned-manifest.json` (20 code/test files + the two official task records). Six frontend production files, four backend production files; no Notes production changes.

The normal integration uses real useChatActions, autosave, ChatTldw, chatModePipeline, save helper and reconciliation; only transport/API/storage boundaries are controlled. It exercises two intentional identical questions, server system+2pairs, local acknowledgements before backlink eligibility, five merged rows and an additional equal-text unsent draft. Late A→B and A→B→A metadata is denied. Character tests use real character orchestration with a local-save spy asserting both IDs; the real persistence helper is tested separately. Mirror and loader suites use the existing in-memory table boundary, not native IndexedDB. The Note suite uses real menu/manager/selection handlers with controlled API/authority. Backend endpoint tests use real FastAPI/isolated user DB and mocked provider inference, checking exact user/assistant IDs and five rows for both streaming modes, plus ephemeral no-write controls.

## RED / GREEN

Counts overlap; do not add them together.

- Initial frontend four suites: **4 expected failures / 64 passed** before implementation. `/private/tmp/cycle4-chat-ack-frontend-red.log`.
- Actual backend streaming/nonstreaming endpoint acknowledgement: **2 expected failures** (empty user-ID acknowledgement versus actual persisted UUID). `/private/tmp/cycle4-chat-ack-backend-red.log`.
- Legacy mirror recovery + Note controls: **1 expected failure / 34 passed** before recovery implementation. `/private/tmp/cycle4-chat-ack-mirror-red.log`.
- Final permanent tests replayed unchanged against original production source `f4e9f954d7` via a private Vite load override: **7 expected failures / 97 filtered**. `/private/tmp/cycle4-chat-ack-baseline-replay.config.mjs`, `/private/tmp/cycle4-chat-ack-final-regressions-red.log`. This is a source-replay proof, not a repository checkout or browser run.
- Current broader frontend: **251 passed / 22 files**. `/private/tmp/cycle4-chat-ack-frontend-broader.log`.
- Final mirror projection/type correction: **30 passed / 2 files**, overlapping the 251. `/private/tmp/cycle4-chat-ack-mirror-final.log`.
- Final backend: **153 passed / 1 existing skipped** (the existing complex asynchronous coordination case), command exits0. `/private/tmp/cycle4-chat-ack-backend-final.log`. Test teardown prints existing pytest temporary-directory cleanup warnings. No new skip.

### Reproduce frontend (repo root)

```sh
cd apps/packages/ui
npm exec -- vitest run src/models/__tests__/ChatTldw.stream-metadata.test.ts src/models/__tests__/ChatTldw.stream-transport-interrupted.test.ts src/models/__tests__/ChatTldw.abort-signal.test.ts src/hooks/chat-helper/__tests__ src/hooks/chat-modes/__tests__ src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx src/hooks/chat/__tests__/useChatActions.persist-mirror.guard.test.ts src/hooks/chat/__tests__/useChatActions.service-prompts.test.tsx src/hooks/chat/__tests__/useChatActions.persona.integration.test.tsx src/hooks/__tests__/useServerChatLoader.mirror.integration.test.tsx src/hooks/__tests__/useServerChatLoader.scope.test.tsx src/db/dexie/__tests__/server-chat-mirror.test.ts src/components/Notes/__tests__/NotesManagerPage.stage26.backlink-labels.test.tsx
```

### Reproduce backend (repo root)

```sh
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py tldw_Server_API/tests/Chat/unit/test_chat_history_and_streaming.py tldw_Server_API/tests/Chat/unit/test_streaming_utils.py -q
python -m bandit tldw_Server_API/app/api/v1/endpoints/chat.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/core/Chat/streaming_pipeline.py tldw_Server_API/app/core/Chat/streaming_utils.py -f json -o /private/tmp/cycle4-chat-ack-bandit.json
```

## Static checks

- Root-invoked ESLint with explicit `apps/tldw-frontend/eslint.config.mjs` and all14 touched frontend paths: **0 errors, 71 unchanged warnings, zero added diagnostics**. `/private/tmp/cycle4-chat-ack-static-check.mjs` records exact command and original-source stdin comparison. Raw JSONs and comparison: `cycle4-chat-ack-lint-{baseline,current,comparison}.json`.
- `cd apps/tldw-frontend && npm run typecheck`: **90 baseline / 90 current; zero added/removed signatures**. `/private/tmp/cycle4-chat-ack-typecheck-final.log`, `...-typecheck-comparison.json`. Intermediate checks caught the mirror projection's required sources and narrowed role types; corrected, not suppressed.
- Bandit four touched production Python files: **zero findings/errors**. `/private/tmp/cycle4-chat-ack-bandit.json`.
- Ruff six touched Python files: four pre-existing import-order diagnostics, **zero added/removed**. `/private/tmp/cycle4-chat-ack-ruff-comparison.json`. Unrelated import blocks were left unchanged.
- Scoped `git diff --check`: clean.

## Review / handoff

Parent owns independent review, combined static checkpoint, native acceptance and commits. `chat_service.py` was released to the model-inventory task after these additive metadata hunks were frozen; any later parallel inventory hunks should not be attributed to this repair.

## Independent-review correction (two confirmed P2s only)

Read `/private/tmp/cycle4-chat-ack-independent-review.md` and reproduced both frozen reviewer probes without changing their assertions. No backend, auth, Notes production, Task5, runtime, browser or inference edits in this correction.

1. Anchored legacy matching now excludes **only empty-string image sentinels** from image-array comparison. Real text-only saves use `[""]`, whereas server-adapted text rows use `[]`. Content, unique reply acknowledgement, parent and ownership guards remain unchanged. Actual nonempty image differences remain distinct. The permanent positive fixture exercises both shapes in the persistent mirror and visible reconciliation; three additional substantive-image mismatch cases retain the unacknowledged local row.
2. Successful Character `addChatMessage` fallback and `saved:true` degraded outcomes now assign their already-confirmed assistant ID to the same variable passed to local persistence. No duplicate server write is introduced, and no ID is fabricated when the response supplies none. Two permanent real-orchestration cases assert both canonical IDs, saved status and the exact write count.

### Review correction evidence

- Unchanged independent image probe before edits: **1 expected failure**, `cycle4-chat-ack-review-image-reproduced-red.log`.
- Unchanged independent Character probe before edits: **2 expected failures**, `cycle4-chat-ack-review-character-reproduced-red.log`.
- Permanent two-suite RED before production edits: **3 failed /36 passed**, `cycle4-chat-ack-review-permanent-red.log`.
- Current focused regression: **126 passed /7 files**, `cycle4-chat-ack-review-focused-green.log`.
- Original unchanged image probe GREEN: **1 passed /24 filtered**, `cycle4-chat-ack-review-image-original-green.log`.
- Original unchanged Character probe GREEN: **2 passed /15 filtered**, `cycle4-chat-ack-review-character-original-green.log`.
- Final Character suite after replacing a new test-only `any` cast: **15 passed**, `cycle4-chat-ack-review-character-final.log` (overlaps the126).
- Root-config lint rerun covers all14 frontend paths; result is **0 errors /71 unchanged warnings /zero added or removed**, recorded in `cycle4-chat-ack-review-lint-summary.log` and the refreshed lint comparison. No new full TypeScript or backend run is claimed for this tiny frontend correction; the prior90/90 and backend153/1 remain the earlier checkpoint, with the parent owning the combined compiler run.
- Pre-review manifest retained at `cycle4-chat-ack-pre-review-owned-manifest.json`; `cycle4-chat-ack-owned-manifest.json` is refreshed after task notes and this correction. Only two production files and their two adjacent test files changed since that snapshot (plus task notes). Captured parentless legacy limitations and pending native acceptance remain unchanged.

Focused command (from `apps/packages/ui`):

```sh
./node_modules/.bin/vitest run src/db/dexie/__tests__/server-chat-mirror.test.ts src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx src/hooks/chat-helper/__tests__/saveMessageOnSuccess.scope.test.ts src/hooks/chat-helper/__tests__/saveMessageOnError.test.ts src/hooks/__tests__/useServerChatLoader.mirror.integration.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage26.backlink-labels.test.tsx
```
