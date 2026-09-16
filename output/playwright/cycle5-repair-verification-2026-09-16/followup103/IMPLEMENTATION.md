# UAT103 selected-source local diagnostic repair — author handoff

Tracking: TASK-13260.44. Agent: retry031_repair. Approved design: option A in design.md. Status: implementation and scoped verification ready for independent review; native acceptance remains root-owned. No staging or commits by this agent.

## Result and root cause

Selected-source RAG preflight already returns saveToDb:false with generationInfo mode=rag, grounded=false and reason=selected_source_retrieval_failed or selected_source_evidence_not_found. It dispatches no normal Chat answer request. The shared pipeline appended this diagnostic pair to inference history and the action fallback also wrote it to the canonical conversation. Reload discarded its eligibility distinction, enabling later history saving and duplicate mirror rows.

The bounded repair keeps the pair visible and locally durable. It excludes only a known local diagnostic assistant and its exact undispatched parent from inference/rewrite history and automatic saving. The classifier requires the complete producer signature; parent exclusion additionally requires an unambiguous parent ID, the parent's matching provenance, no canonical receipt and no actual answered variant. There is no text deduplication or generic missing-ACK deletion. Canonical rows, real repeated turns, detached drafts, meaningful image rows and answered variants remain eligible.

Retry reuses the exact parent question, image (including no image), type and user ID. Because the diagnostic never dispatched a Chat user, it does not request backend failed-turn reuse. Temporary success restores eligibility without requiring a server receipt. Continue on an exact pending diagnostic follows this same Retry path; ordinary complete/partial Continue is unchanged. Character Retry preserves the existing conversation. Edits and deletion account for the difference between visible rows and eligible prompt rows. Deleting only the diagnostic assistant leaves its exact user as a retained local draft; immediate prompt eligibility now matches a real formatter reload. The existing latest-message ref supplies the rebuild, and explicit stored roles (including canonical system messages) are preserved before the legacy isBot fallback. The regeneration handler forwards its actual submission result, and diagnostic Continue forwards failure or reports skipped if validation prevents dispatch; direct Retry callers retain their existing behavior. The preparePrompt boundary filters ctx.messages in place, retaining context identity used by the RAG preflight WeakMap cache.

No backend, wire protocol or server-chat-mirror edits. Existing ownership/race guards remain in place. This prevents new promotion of the diagnosed pair; it does not safely remove already canonicalized duplicate prose whose original provenance was lost.

## Owned files

Eight production files:

- apps/packages/ui/src/utils/local-rag-diagnostic.ts
- apps/packages/ui/src/hooks/chat-modes/ragMode.ts
- apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts
- apps/packages/ui/src/hooks/handlers/messageHandlers.ts
- apps/packages/ui/src/db/dexie/helpers.ts
- apps/packages/ui/src/hooks/chat/useServerChatLoader.ts
- apps/packages/ui/src/hooks/chat/useChatActions.ts
- apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundPersistence.tsx

Six permanent test files:

- apps/packages/ui/src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx
- apps/packages/ui/src/hooks/__tests__/useServerChatLoader.mirror.integration.test.tsx
- apps/packages/ui/src/hooks/handlers/__tests__/messageHandlers.regenerate.test.ts
- apps/packages/ui/src/hooks/handlers/__tests__/messageHandlers.edit.test.ts
- apps/packages/ui/src/utils/__tests__/local-rag-diagnostic.test.ts
- apps/packages/ui/src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx

## Permanent RED evidence

- red-action.log: four original-production behavioral failures. Two actual RAG action cases made unwanted canonical add-message writes; two exact-parent Retry cases retained diagnostics in inference history.
- red-boundaries.log: two real loader/mirror/formatter/generateHistory failures. Equal genuine prose plus the diagnostic appeared twice in prompt history. Four earlier fixture-routing failures in the same log are not counted as product RED; red-action.log records the corrected fixture.
- red-retry-image.log: exact text-only diagnostic Retry borrowed an older genuine image/type.
- red-review-boundaries.log: three edit-index failures plus actual character-RAG Retry branching.
- red-delete-boundary.log: deleting a diagnostic removed identical genuine prompt prose.
- red-rewrite-boundary.log: actual later global-RAG question rewrite included the diagnostic through ctx.messages.
- red-continue-boundary.log: exact pending diagnostic Continue extended diagnostic prose. Both ordinary complete/partial Continue controls already passed.
- red-final-review-boundaries.log: assistant-only deletion gave different immediate/reloaded draft eligibility, and Continue reported submitted for a failed Retry. Its first validation fixture still inherited the store model; that branch is not counted as validation RED.
- red-continue-result-replay.log: after correcting the fixture to clear both model sources, a private Vitest transform replays only the original unconditional-success Continue wrapper. Both real validation-skip and provider-failure controls fail. Production source is untouched during this replay. The corrected controls pass in green-final-review-boundaries.log and the final combined suite.

- red-system-role.log: permanent post-mount loader-shaped canonical-system test fails before the explicit-role correction. Reviewer isolated the same actual deletion boundary and confirmed original baseline passes; copies of reviewer-system-delete-test.txt/current.log/baseline.log retain the comparison. A prefilled-mount version hit an older generic refresh behavior and was discarded as causal evidence.

All corresponding permanent regressions pass in the final combined run. The final character test additionally asserts that no branch is created or conversation ID setter called, and that the exact diagnostic reaches the RAG handoff; it does not invent a serverChatId parameter absent from that existing route.

## Final verification

- final-tests.log: **215 passed / 14 suites**, no skips (2026-09-16 13:26 local, 15.04 seconds). Includes actual action → pipeline → local save → empty-memory loader → generateHistory → next completion and settled reload, repeated saved/temporary Retry, exact receipts on promotion, ordinary handled-preflight saving, equal prose/drafts/images, provider failures, abort ownership, image events, successful/partial Continue, edits, deletion and character branching. Positive selected-source completion asserts only two retrieval calls across its two turns, preserving preflight-cache behavior.
- final-eslint.log, eslint-baseline.json and eslint-summary.json: **0 errors, 105 warnings** in both HEAD and current 14-path scope. Exact per-file rule/message/severity lists unchanged.
- final-diff-check.log: scoped git diff --check exits 0.
- bandit.json/log: project-venv Bandit invoked on eight production TS/TSX files. It reports **0 findings and 8 Python AST parse errors**. It did not analyze TypeScript, so this is not TypeScript security assurance. No Python source changed.
- final-typecheck.log/comparison.json: full compiler returns exit 2 with **90 baseline / 90 current diagnostics, zero added or removed**. Root found a new void-result narrowing error during independent review; explicit truthy-result narrowing corrected it and retained actual failed/skipped behavior. This is baseline equivalence, not a clean typecheck.
- Independent review closure and native acceptance remain root-owned; no native acceptance claim is made here.

## Exact commands and environment

The optional pa-tesseract.js package exists in the Bun cache but has no workspace node_modules link. Private vitest.config.ts merges the repository frontend config and aliases only that existing package. No dependency install or node_modules mutation was performed. OCR still uses the existing test mock; real human formatter and image controls execute unchanged. Earlier collection/image-control failures without this alias were environmental, not counted as passing repair evidence.

Combined tests, from apps/tldw-frontend:

```sh
bunx vitest run --config ../../.tmp/uat103-local-preflight-20260916/vitest.config.ts ../packages/ui/src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx ../packages/ui/src/hooks/__tests__/useServerChatLoader.mirror.integration.test.tsx ../packages/ui/src/hooks/handlers/__tests__/messageHandlers.regenerate.test.ts ../packages/ui/src/hooks/handlers/__tests__/messageHandlers.edit.test.ts ../packages/ui/src/utils/__tests__/local-rag-diagnostic.test.ts ../packages/ui/src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx ../packages/ui/src/hooks/chat-modes/__tests__/ragMode.sanitization.test.ts ../packages/ui/src/hooks/chat-modes/__tests__/chatModePipeline.provider-recovery.test.ts ../packages/ui/src/hooks/chat-modes/__tests__/chatModePipeline.error-variants.persistence.test.ts ../packages/ui/src/hooks/chat-modes/__tests__/chatModePipeline.abort-lifecycle.test.ts ../packages/ui/src/hooks/chat/__tests__/useChatActions.image-event-sync.integration.test.tsx ../packages/ui/src/components/Option/Playground/hooks/__tests__/usePlaygroundPersistence.test.tsx ../packages/ui/src/hooks/chat-helper/__tests__/saveMessageOnError.test.ts ../packages/ui/src/utils/__tests__/generate-history.image-generation.test.ts > ../../.tmp/uat103-local-preflight-20260916/final-tests.log 2>&1
```

Scoped checks from repo root:

```sh
apps/tldw-frontend/node_modules/.bin/eslint -c apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/utils/local-rag-diagnostic.ts apps/packages/ui/src/hooks/chat-modes/ragMode.ts apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts apps/packages/ui/src/hooks/handlers/messageHandlers.ts apps/packages/ui/src/db/dexie/helpers.ts apps/packages/ui/src/hooks/chat/useServerChatLoader.ts apps/packages/ui/src/hooks/chat/useChatActions.ts apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundPersistence.tsx apps/packages/ui/src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx apps/packages/ui/src/hooks/__tests__/useServerChatLoader.mirror.integration.test.tsx apps/packages/ui/src/hooks/handlers/__tests__/messageHandlers.regenerate.test.ts apps/packages/ui/src/hooks/handlers/__tests__/messageHandlers.edit.test.ts apps/packages/ui/src/utils/__tests__/local-rag-diagnostic.test.ts apps/packages/ui/src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx
git diff --check -- apps/packages/ui/src/utils/local-rag-diagnostic.ts apps/packages/ui/src/hooks/chat-modes/ragMode.ts apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts apps/packages/ui/src/hooks/handlers/messageHandlers.ts apps/packages/ui/src/db/dexie/helpers.ts apps/packages/ui/src/hooks/chat/useServerChatLoader.ts apps/packages/ui/src/hooks/chat/useChatActions.ts apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundPersistence.tsx apps/packages/ui/src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx apps/packages/ui/src/hooks/__tests__/useServerChatLoader.mirror.integration.test.tsx apps/packages/ui/src/hooks/handlers/__tests__/messageHandlers.regenerate.test.ts apps/packages/ui/src/hooks/handlers/__tests__/messageHandlers.edit.test.ts apps/packages/ui/src/utils/__tests__/local-rag-diagnostic.test.ts apps/packages/ui/src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx
source .venv/bin/activate
python -m bandit apps/packages/ui/src/utils/local-rag-diagnostic.ts apps/packages/ui/src/hooks/chat-modes/ragMode.ts apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts apps/packages/ui/src/hooks/handlers/messageHandlers.ts apps/packages/ui/src/db/dexie/helpers.ts apps/packages/ui/src/hooks/chat/useServerChatLoader.ts apps/packages/ui/src/hooks/chat/useChatActions.ts apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundPersistence.tsx -f json -o .tmp/uat103-local-preflight-20260916/bandit.json
```

Reproduce all final static receipts from the repo root with `source .venv/bin/activate` then `python .tmp/uat103-local-preflight-20260916/verify-static.py`.

Full compiler command from apps/tldw-frontend: `bun run typecheck > ../../.tmp/uat103-local-preflight-20260916/final-typecheck.log 2>&1`. Comparison strips only line/column positions from diagnostic headers and compares multisets with `.tmp/fresh-uat-recovery-20260916/typecheck-current.log`; final-typecheck-comparison.json records the result.

The ESLint baseline captures git show HEAD:<path> and supplies the returned text through the same ESLint binary's --stdin --stdin-filename <path> -f json arguments. New files use empty baseline text. No production rollback occurred.

## Review/native limits

Original latest native artifacts were lost; the original exact identities are transcript-only observations. Fresh permanent synthetic tests reproduce the actual frontend boundaries, but do not prove the lost native identities. Root owns final native acceptance; the full compiler baseline comparison now has no added diagnostics. UAT013 source-answer correctness and UAT157/TASK13260.95 genuine image-history duplication remain separate. Ordinary pre-dispatch failures followed by a different new Send and meaningful partial answers without receipts are broader history-identity cases; this local diagnostic policy does not claim to solve them.

owned-manifest.json records exact production, test, Backlog and evidence SHA-256 hashes at handoff. review-snapshot/ contains byte-identical copies of all 14 owned source/test files; owned.patch captures the change including three new files, for review after shared paths are released to the separate UAT157 repair. The earlier production-freeze-manifest.json is historical and superseded for review by owned-manifest.json.
