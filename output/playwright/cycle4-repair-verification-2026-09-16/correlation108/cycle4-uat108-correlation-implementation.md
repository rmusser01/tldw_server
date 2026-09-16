# UAT108: failed-user correlation through the mounted loader

Task: TASK-13260.49. Baseline 7dcee3d72c; frozen 2026-09-16T05:21:13.630Z. Exact byte manifests: cycle4-uat108-correlation-owned-manifest.json and production-manifest.json in /private/tmp. Eight production paths, seven test paths, one task record. This follow-up supersedes the native acceptance of the prior retry-persistence repair; it does not claim native GREEN.

## Evidence and cause

The preserved native initial request contains one user and returns502. The UI already contains two identical users before Retry. Retry sends two users; after Retry, the canonical conversation contains two user IDs and its final assistant. Native captures did not include a pre-Retry canonical GET, so no pre-Retry server row-count claim is made. Evidence: /private/tmp/cycle4-uat108-native-multi-final-report.md and its21-artifact manifest.

Source and the controlled mounted reproduction establish the missing interaction: the server saves the user before provider execution; the early502 returns no ACK. The real server loader then sees a canonical user while the local row is still unacknowledged. Previous mirror identity recovery required an acknowledged assistant anchor, which cannot exist yet. It preserves the unmatched local row, resulting in two visible users. Explicit Retry then submitted both; the old overlap rule appended another canonical user.

## Correction

The already-generated local user ID travels through the existing pipeline/model/client options as app-only metadata tldw_client_message_id. Only the final user actually being persisted receives a validated, bounded client_message_id in its DB metadata, in the existing atomic transaction. Provider messages and arguments do not receive it. Correlation is added after content-placeholder classification, preserving empty-content behavior.

The mirror links only a single exact correlation claim to an unacknowledged same-history user with identical role, content and substantive images. Existing canonical identity, foreign-history and captured-authority checks remain. Equal text with another local ID remains separate. No text-only deduplication or deletion was added. Normal/persona message loads use the existing render_placeholders=false option so literal stored text can match; tracked-character loads retain rendered placeholders. Both listing shapes retain correlation metadata.

Explicit Retry rejects residual unmatched extra users409 before any new write/provider call. If saved correlation exists, a contradictory local identity also rejects409. Missing/null/malformed legacy metadata remains compatible with the pre-existing exact-tail retry validation. Ordinary repeated sends and successful-answer regeneration remain distinct from failed Retry.

## Verification and RED trail

- Permanent mounted actual action + useServerChatLoader + real ChatTldw regression reproduced1 expected failure before repair: /private/tmp/cycle4-uat108-mounted-real-model-red.log. It now verifies initial failure, one linked local user before Retry, one outbound user, successful canonical answer, and remount retaining both IDs/content. Transport/provider and storage adapters are controlled; this is not native IndexedDB or real inference.
- Correlation model/transport/mirror boundary RED:3 failed/51 passed across3 files, cycle4-uat108-correlation-ui-red.log. Real endpoint/SQLite DB RED:3 failed/23 deselected, cycle4-uat108-correlation-backend-red.log.
- Raw normal/persona versus tracked-character compatibility RED:3 failed/7 skipped, cycle4-uat108-placeholder-read-red.log. Character's missing explicit query flag was contract scaffolding; its existing default rendered behavior was already correct. Tests now also assert the resulting content.
- Additional review controls reproduced a null-extra error (1 failed/7 passed, cycle4-uat108-correlation-boundary-actual.log) and unrelated empty-content placeholder regression (1 failed, cycle4-uat108-empty-content-red.log) before their bounded corrections.
- Author expanded UI:117 passed/5 files, cycle4-uat108-correlation-ui-expanded.log. Final changed owner fixture:2 passed/50 skipped, cycle4-uat108-correlation-owner-final.log. Final loader scope suite including two actual mapper attachment-safety controls:12 passed, cycle4-uat108-correlation-mapper-safety-final.log. Counts overlap and must not be summed.
- Author backend:87 passed/2 files, cycle4-uat108-correlation-backend-expanded.log; final added/affected endpoint/DB controls:7 passed/24 deselected, cycle4-uat108-correlation-extra-boundaries.log. These overlap; final reviewer runs all92 tests across the same two files. Tests exercise transactional metadata failure rollback, actual pre-provider502 persistence, both listing shapes, metadata decoding failure, and no-partial-write409.
- Root-config frontend ESLint:0 errors/22 unchanged warnings over11 frontend paths. cycle4-uat108-correlation-lint-{current,baseline,comparison}.json. The initial test-only unreachable-yield finding was corrected with a behavior-equivalent rejecting async generator, then affected cases/lint rerun.
- Ruff:18 existing findings/18 current,0 added/removed across4 Python paths. cycle4-uat108-correlation-ruff-comparison.json.
- Bandit:0 findings/0 errors across2 touched production Python paths. cycle4-uat108-correlation-bandit.json.
- Owned diff-check clean. Whole TypeScript and combined broader runs are parent-owned and not claimed here before completion.

## Exact commands

UI from apps/packages/ui:
./node_modules/.bin/vitest run src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx src/db/dexie/__tests__/server-chat-mirror.test.ts src/models/__tests__/pageAssistModel.mcp-tools.test.ts src/services/tldw/__tests__/TldwChat.abort.test.ts src/hooks/__tests__/useServerChatLoader.scope.test.tsx --maxWorkers=1 --no-file-parallelism

Backend from repository root:
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/unit/test_chat_history_and_streaming.py tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py -q

Static exact commands/paths are retained in /private/tmp/cycle4-uat108-correlation-lint.mjs and cycle4-uat108-correlation-paths.json; ESLint uses explicit repository-root apps/tldw-frontend/eslint.config.mjs. Bandit: source .venv/bin/activate && python -m bandit tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/api/v1/endpoints/chat.py -f json -o /private/tmp/cycle4-uat108-correlation-bandit.json.

## Final review and formatting-only refresh

Independent review clear for ordinary text scope: /private/tmp/cycle4-uat108-correlation-independent-review.md. Independent backend92/2 and UI117/5 plus final loader12/1 passed (119 distinct UI cases across the same five suites). Root approved alignment of two ChatTldw property indents after its combined runners completed; actual git diff --no-index -w returned0 with no output, proving a whitespace-only delta. Exact before/after hashes: /private/tmp/cycle4-uat108-correlation-indent-verification.json. Tests were not repeated for whitespace. Refreshed manifest is authoritative; no additional behavioral edits.

## Limits and ownership

- Historically ambiguous local/server rows without the new marker are preserved. Existing acknowledged-reply recovery remains available; unresolved duplicate Retry fails409 rather than silently discarding work or adding another canonical user. No migration/deletion is attempted.
- The existing standard listing/client omits user attachment bytes and the loader maps ordinary images to[]. Image-only stored text can be a generated attachment placeholder. Therefore successful attachment recovery is NOT established by this text-workflow repair. The handcrafted mirror image case tests exact-equality safety only. Permanent actual mapper controls prove image-only/text+image local content and bytes remain unchanged and unacknowledged when identity cannot be safely established. This pre-existing limitation was explicitly scoped out by root; no weakened match was introduced.
- No browser/runtime/inference calls, staging, commit, global tracker or shared plan edits performed by this unit. Parent owns stable-runtime native recheck and full fresh UAT. Independent review is a separate report; final native108 is still pending.

## Owned paths

- apps/packages/ui/src/db/dexie/server-chat-mirror.ts (production)
- apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts (production)
- apps/packages/ui/src/hooks/chat/useServerChatLoader.ts (production)
- apps/packages/ui/src/models/ChatTldw.ts (production)
- apps/packages/ui/src/models/index.ts (production)
- apps/packages/ui/src/services/tldw/TldwChat.ts (production)
- apps/packages/ui/src/db/dexie/__tests__/server-chat-mirror.test.ts (test)
- apps/packages/ui/src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx (test)
- apps/packages/ui/src/hooks/__tests__/useServerChatLoader.scope.test.tsx (test)
- apps/packages/ui/src/models/__tests__/pageAssistModel.mcp-tools.test.ts (test)
- apps/packages/ui/src/services/tldw/__tests__/TldwChat.abort.test.ts (test)
- tldw_Server_API/app/core/Chat/chat_service.py (production)
- tldw_Server_API/app/api/v1/endpoints/chat.py (production)
- tldw_Server_API/tests/Chat/unit/test_chat_history_and_streaming.py (test)
- tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py (test)
- backlog/tasks/task-13260.49 - Exclude-failed-Chat-display-placeholders-from-retry-model-context.md (task)
