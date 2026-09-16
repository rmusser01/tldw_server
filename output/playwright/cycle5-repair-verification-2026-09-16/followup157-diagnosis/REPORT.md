# UAT157 / TASK13260.95 — saved completion duplicated by frontend fallback

## Conclusion

The first successful native image turn was saved once by `/chat/completions`, then a frontend `/chats/{id}/messages` fallback appended a text-only user and assistant pair. It happened immediately after the first completion, approximately52 seconds before the next Send. This is independent of the later failed image Retry, which root observed succeed.

The ordinary success-persistence wrapper uses `hasChatSaveToDb` to suppress fallback writes and ignores the current turn's explicit per-row server IDs. Actual-boundary tests reproduce duplicate writes even when **both ACKs are delivered and locally saved**. A missing native SSE ACK is not needed for this reproduction and is not proven by the unavailable SSE body.

## Native causal evidence

Canonical conversation: `9127b6f8-7ed8-49c5-9258-7dcb00110138`.

| UTC timestamp | Canonical row | Result |
|---|---|---|
|20:00:15.700|user `6cfcf0da-4d16-4728-be15-fb897b5a7c3d`|Original question, one image|
|20:00:21.412|assistant `23742a74-62e6-4722-94ba-ccd2e40c50ef`|Original correct answer|
|20:00:21.480|user `56a5dbe0-e761-4f9f-84d6-13329fe2da3b`|Identical question text, zero images|
|20:00:21.497|assistant `2d426942-16eb-4659-a252-4883448fb45f`|Identical answer text|
|20:01:13.556|user `7e6b415e-cc06-474a-b3bc-95a97b30ac34`|Later Send; cannot cause the earlier duplicate pair|

Backend access-log lines1506 and1515 record two `/api/v1/chats/9127.../messages` POST201 responses at local13:00:21.486 and13:00:21.500. Their timing matches the duplicate inserts exactly. The redacted receipt contains only timestamp, method/path/status/duration and source line; no headers, keys, or messages. This identifies the additional writes as separate frontend API calls, rather than a second backend save inside the first completion.

The first retained transport request has `save_to_db:true`, the same conversation ID, `metadata.tldw_client_message_id:pa_ae07-7a49-5d6-c4a8`, and a multimodal user containing text plus image_url. Model was `llama.cpp/gemma-4-26B-A4B-it`. Response status200, response-body capture unavailable. Root's later local receipts link original local user/assistant to the original canonical IDs; because those were captured after reload, they do not establish stream-time ACK delivery.

## Source chain

- `TldwApiClient.ts:2630-2641`: same-origin Quickstart server URL deliberately returns null for OpenAPI without requesting the schema.
- `server-capabilities.ts:110-201,322-328,774-810`: bundled fallback defines route names with empty operations; therefore no save_to_db request property exists and `hasChatSaveToDb` is false. An actual-module probe verifies this path without network I/O, with authoritative schema support as an independent positive control. Native Quickstart settings and absence of OpenAPI GET are consistent; native cached capability values were not captured by this subagent.
- `ChatTldw.ts:184-200`: received canonical user/assistant IDs populate separate current-turn model fields. The pipeline success payload forwards both IDs (`chatModePipeline.ts` ordinary success call).
- `useChatActions.ts:1101-1116`: global fallback suppression depends on capability support (or explicit `serverMessagesAlreadyPersisted`). Capability false/error permits fallback despite current-turn canonical IDs.
- `useChatActions.ts:1265-1295`: ordinary success fallback posts user text then assistant text. It omits user images and ignores the existing canonical IDs, matching the native duplicate signature.

Line numbers refer to the source hashes in `source-before-final-check.json`; all six hashes were unchanged across the final probe run.

## Actual-boundary regression evidence

Private probe reuses a frozen copy of the existing saved-normal integration fixture, with the production hook, pipeline, real human formatter, actual model factory/ChatTldw, success helper, and Dexie saveMessage. Downstream stream I/O and server/Dexie storage are mocked. The transport fixture explicitly saves only rows for which it emits a canonical ACK, allowing genuinely unpersisted-row controls. It never calls a provider.

| Case | Desired fallback roles | Actual | Result |
|---|---|---|---|
|Both ACKs, capability false|none|user + assistant|RED|
|Both ACKs, capability throws|none|user + assistant|RED|
|Both ACKs, capability true|none|none|PASS|
|User ACK only, assistant genuinely unsaved|assistant|user + assistant|RED|
|Assistant ACK only, user genuinely unsaved|user|user + assistant|RED|
|No ACKs, ordinary unsaved text, capability false|user + assistant|user + assistant|PASS|
|No ACKs, ordinary unsaved text, capability throws|user + assistant|user + assistant|PASS|

The both-ACK observations include local user/assistant canonical IDs **before assertions**, while mock canonical state already contains the duplicated text-only pair. Loader/reload is disabled for these cases, so reconciliation cannot supply the ACK evidence later.

Final command: `node apps/tldw-frontend/node_modules/vitest/vitest.mjs run --config .tmp/uat157-completion-ownership-20260916/vitest.config.ts -t UAT157`.

Result: **4 behavioral RED, 5 PASS** (seven persistence cases plus two capability cases);87 copied existing cases intentionally filtered out. Initial config merged broad include globs and began collecting unrelated skipped suites; it was stopped, then narrowed. Initial capability fixture lacked a mock serde export; that setup error is retained separately and corrected. Neither is counted as behavioral evidence.

## Smallest proposed repair and permanent regression

Keep the existing conversation/scope, capability, regeneration/Continue, image-generation-event, and exact local-RAG diagnostic guards. Inside the ordinary fallback branch, make each row conditional on its own current-turn canonical identity:

1. A non-empty current `userServerMessageId` suppresses the user POST only.
2. A non-empty current `assistantServerMessageId` suppresses the assistant POST only.
3. Missing user identity still permits the existing user fallback; missing assistant identity still permits the existing assistant fallback.

Do not blanket-skip the pair when only one ACK exists, and do not use text equality or image equality to deduplicate. Current model stream state already resets ACK fields per request; preserve that ownership boundary. Capabilities describe support, whereas a current-turn canonical ID is direct evidence that a particular row exists.

Permanent tests can add the seven cases to `useChatActions.saved-normal.integration.test.tsx` with a configurable capability mock (currently fixed true across that fixture). Include whitespace/empty-ID normalization and existing mismatched conversation/scope abort controls as appropriate. Keep the two actual Quickstart/capability tests separate from the per-row persistence assertions so they do not mask one another. No need to falsely advertise schema support in the bundled fallback.

This minimal fix does not guarantee idempotence when a server saves but its ACK is actually lost. That case needs correlation/reconciliation evidence and remains distinct. Legacy fallback's image omission and ignored returned IDs also remain separate behaviors; the proposed per-row fix avoids invoking that path for an acknowledged image row. Root must make the native acceptance decision after a fresh first image turn, full `/chat/` plus `/chats/` capture, and canonical reload counts.

## Coordination and limits

Root owns task13260.95 and all product edits. UAT103 author was notified; ordinary/image fallback is unchanged by that pending work, which only adds exact local RAG diagnostic suppression and prompt filtering. No production edits, browser actions, model calls, database resets or processes were started by this diagnosis. No closure claim for UAT013 wrong-answer behavior, UAT103 diagnostics, or the unrelated successful image Retry.
