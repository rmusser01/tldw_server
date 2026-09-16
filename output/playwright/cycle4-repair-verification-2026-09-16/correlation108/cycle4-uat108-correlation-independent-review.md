# UAT108 correlation follow-up — independent review

## Verdict

No introduced blocking finding found in the bounded ordinary-text failed-send → canonical loader → Retry → reload correction. The prior native failure is reproduced by the mounted regression before correction and closed by the current automated control. Native acceptance remains pending; the previous native failure is not retroactively a pass.

Reviewed eight production paths against7dcee3d72c (the parent also identifies b4c72cb814 as the subsequent unrelated checkpoint): server-chat-mirror.ts, chatModePipeline.ts, useServerChatLoader.ts, models/ChatTldw.ts, models/index.ts, services/tldw/TldwChat.ts, backend core/Chat/chat_service.py, and endpoints/chat.py. No repository edits, browser/runtime actions, generation, staging, or commits by this reviewer.

## Evidence and reasoning

- Durable correlation attaches only to the final persisted user’s database copy. The existing database transaction writes user and metadata atomically; rollback is covered. Provider payloads do not receive the application marker.
- The existing local ID travels through the actual pipeline, model factory, ChatTldw, and both TldwChat transports. Successful regeneration remains outside failed-turn intent and does not re-ACK the old user.
- Owned mirror recovery requires one canonical claim for one local ID, user role, exact text and substantive images, and no contradictory prior ACK. Same-text distinct local IDs remain separate. Existing owner/history checks and captured authority cancellation guard the transaction and delayed UI commit.
- The mounted permanent test now runs actual actions, loader, ChatTldw and mirror with controlled persistence/transport boundaries. It simulates a persisted user before provider failure without ACK, hydrates one user, retries one question, saves canonical assistant metadata, and asserts that canonical answer after remount.
- Delayed A→B and A→B→A responses cannot ACK stale local work or replace the current draft. Backend rejects residual unmatched user messages before further writes/provider dispatch. Matching legacy tail metadata missing/null/invalid retains compatibility; ordinary equal-text send remains distinct.
- Normal/persona loader reads request raw canonical content so placeholder rendering does not break exact matching; tracked Character rendering is retained. No schema, authentication, or permission relaxation was introduced.
- Parent-found null-extra and blank-content classification edges are corrected and covered by actual database tests.

## Independent verification

1. Before correction, mounted regression: **1 expected failure /49 unselected**. Log /private/tmp/cycle4-uat108-hydration-independent-red.log. Actual failure: two visible user rows, one expected.
2. Final frontend: **117 passed /5 files**, exit0. Log /private/tmp/cycle4-uat108-correlation-independent-ui.log.
3. Final backend: **92 passed /2 files**, exit0, six existing warnings plus pytest temporary-directory cleanup warning. Log /private/tmp/cycle4-uat108-correlation-independent-backend.log.
4. Private actual canonical mapper→real reconciliation image recovery probe: **2 expected failures /10 unselected**. This establishes the pre-existing limitation below; it is not a new native failure. Log /private/tmp/cycle4-uat108-correlation-image-probe.log.
5. Separate private actual mapper image safety control: **2 passed /10 unselected**. Verifies unmatched image-bearing local content/images survive without false canonical ACK. Log /private/tmp/cycle4-uat108-correlation-image-probe-safety.log.
6. Final permanent loader safety delta: **12 passed /1 file**, exit0, in /private/tmp/cycle4-uat108-correlation-independent-loader-final.log. This supersedes the 10-test loader file within the117 run; final unique frontend total is119 across those same five suites. Only the two added mapper safety tests changed after the117 run.
7. Scoped production git diff --check: clean.

Counts overlap author and parent runs; do not add them. No whole compiler, broad UAT, external provider, or native IndexedDB coverage is claimed by this reviewer. Frontend tests use controlled database/transport adapters; backend tests use actual API/database with mocked provider.

## Explicit pre-existing limitation: image-bearing failed turns

The actual normal canonical listing exposes has_image, not image bytes, and useServerChatLoader.ts mapServerChatMessagesToPlaygroundMessages sets ordinary-user images to[] (around line440). Backend image-only persistence can store the <Image attachment x1> placeholder. Therefore strict correlation cannot match a local text+image or image-only turn through this real wire shape. The pure mirror image-only test verifies safety when images are supplied, not end-to-end recovery.

Root explicitly scoped this correction to observed ordinary-text108 and instructed preservation of fail-closed behavior, without expanding attachment serialization or weakening identity checks. The private safety controls pass: image-bearing local work remains and is not falsely ACKed. Image Retry recovery and native image coverage are not claimed. This limitation is reported for tracker retention separately from introduced regressions.

## Exact commands

Repository root:

    source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/unit/test_chat_history_and_streaming.py tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py -q

apps/packages/ui working directory:

    ./node_modules/.bin/vitest run src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx src/db/dexie/__tests__/server-chat-mirror.test.ts src/hooks/__tests__/useServerChatLoader.scope.test.tsx src/models/__tests__/pageAssistModel.mcp-tools.test.ts src/services/tldw/__tests__/TldwChat.abort.test.ts --maxWorkers=1 --no-file-parallelism

    ./node_modules/.bin/vitest run --config /private/tmp/cycle4-uat108-correlation-image-probe.config.mts -t 'private actual canonical image' --maxWorkers=1 --no-file-parallelism

    ./node_modules/.bin/vitest run --config /private/tmp/cycle4-uat108-correlation-image-probe-safety.config.mts -t 'private actual canonical image preserves' --maxWorkers=1 --no-file-parallelism

Private probe source files have the same prefixes ending .test.tsx. Both configs import the actual UI Vitest config and resolve packages for /private/tmp imports. One initial image-probe launch from repository root failed before tests because relative vitest.setup.ts resolved incorrectly; the retained final log is the correct apps/packages/ui execution.

## Freeze audit

Author manifest /private/tmp/cycle4-uat108-correlation-owned-manifest.json frozen2026-09-16T05:18:38.196Z: all16 entries match SHA256 (8 production,7 tests,1 official task). Reviewer audit: /private/tmp/cycle4-uat108-correlation-independent-hash-check.json. Author report: /private/tmp/cycle4-uat108-correlation-implementation.md. Two harmless ChatTldw property-indentation lines remain pending parent formatting disposition; any subsequent hash refresh must be verified as whitespace-only. This audit covers the exact current bytes.
