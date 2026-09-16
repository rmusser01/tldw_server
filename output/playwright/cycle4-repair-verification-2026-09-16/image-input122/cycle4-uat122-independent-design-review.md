# UAT122 independent design and frozen implementation review

**Final outcome: clear within the approved bounded scope.** The Retry finding below is resolved by the final frozen implementation. Current OCR conversion is preserved; historical unconverted images deliberately fail closed as approved. No remaining actionable source finding.

Read-only scope: approved `/private/tmp/cycle4-uat122-image-input-design.md`, actual saved normal Chat action/pipeline, model/formatter, persisted Retry error/correlation, history projection, and backend failed-turn discriminator. No repository edits, browser, server/runtime changes, OCR worker, or inference.

## Actionable boundary found

**P2 — distinguish a never-dispatched local refusal from a failed server turn.** `chatModePipeline.ts:228` currently turns every decoded failed bubble into `retryFailedTurn=true`. After the proposed capability refusal, a new local image turn has never been sent. If an older answered canonical turn has the same text/image, the server rejects the later supported Retry with 409 “This turn already has an answer,” despite a different new `client_message_id` (`chat_service.py:4364–4368`). The existing exact-content safety guard should remain. Root approved a narrow pipeline distinction between local failed-turn identity/ACK handling and server failed-turn retry intent.

Evidence: `/private/tmp/cycle4-uat122-unacked-retry-independent.py` exercises the real `build_context_and_messages` with controlled DB records and persistence callback. `/private/tmp/cycle4-uat122-unacked-retry-independent.log`: **1 failure /2 passes**. Empty history and different prior question accept the new turn; identical prior answered image turn rejects it. This is an automated boundary finding, not a native inference/provider result.

Needed permanent transitions: initial local refusal→repeated local Retry→supported dispatch keeps server retry false and one local user/correlation; ambiguous real server failure without ACK→local refusal→supported Retry keeps server retry true. The same boolean must not disable local user identity or ACK: current `resolvedUserMessageId` and save helpers depend on the local failed-turn classification. Persist/remount the explicit provenance; missing or malformed provenance remains conservative. Do not infer dispatch from missing ACK.

## OCR history finding and approved decision

Mounted private two-turn probe uses real `useChatActions`, normal pipeline, formatter, `generateHistory` and `ChatTldw`; only OCR, storage/server boundaries and transport are controlled. `/private/tmp/cycle4-uat122-ocr-history-independent.test.tsx` and `.config.mts`; baseline log `/private/tmp/cycle4-uat122-ocr-history-before.log`: **1 passed**, 57 unrelated snapshot tests skipped by name filter.

Baseline first request includes `Read this receipt\n\n[IMAGE OCR TEXT]\nReceipt total 42`; local history retains original image and original question. Second text-only request succeeds but sends only original question, assistant answer, and follow-up: the extracted OCR text was **already lost before UAT122**. Current-turn formatting only occurs in `normalChatMode.ts:403–429`; history reconstruction at 526 uses `generateHistory`, which restores original image parts; default pipeline history at 926 retains original text/image.

Root decision: preserve explicit current OCR conversion, retain truthful refusal for historical image parts rather than silently stripping/exempting/re-OCRing them, and describe image-bearing history accurately. Guidance should offer an image-capable model or a new text-only chat; composer image removal does not remove historical attachments. Existing/unproven OCR history remains blocked until its canonical effective-text representation is complete. No backend/schema/snapshot framework expansion is authorized by this review.

The identical probe will be rerun after the actual guard lands to record the new refusal delta. This is a known compatibility boundary, not a claim that OCR history previously worked correctly.

## Other design checks

- Whole outgoing **user** image scan is the appropriate location before stream/invoke transport; omission/unknown catalog capabilities must fail closed with “not confirmed” wording, not claim a definitively unsupported model.
- `generateHistory` preserves ordinary user attachments and excludes only image-generation event rows plus recognized assistant error envelopes. Existing focused controls run independently: `/private/tmp/cycle4-uat122-history-exclusion-independent.log`.
- Existing saved-normal owner snapshots guard transport/persistence across account/target changes and A→B→A; the narrow model guard should not add global listeners or bypass those leases.
- Model selection remains reachable through the actual banner's existing Switch provider action and composer model selector. The encoded recovery action is overridden by the banner's provided Retry callback, so do not claim a new primary Choose model button without a UI change.
- Error details must be a fixed code/text with no attachment bytes. A no-ACK local refusal may have an empty saved conversation, which is acceptable if the local original user/image stays intact.

## Status / limitations

Design corrections have been sent to root and author. Final implementation/provenance hunks and unchanged OCR after-probe remain pending. No full native vision positive can be claimed: current runtime has no confirmed vision provider. These private tests use controlled transport/DB boundaries and do not certify native reload or real vision inference.

## Guard landed: independent before/after evidence

The unchanged mounted OCR probe was rerun against the actual author guard (not an injected implementation). `/private/tmp/cycle4-uat122-ocr-history-after.log`: its previous second-turn success expectation now fails exactly as the approved decision predicts. First explicit OCR remains `submitted` with the same extracted text; second is `failed` with `Image support is not confirmed for this model.`; dispatch count remains **1**, and the original local image remains retained. Before: **2** dispatches with OCR context already omitted from the second. Do not describe this expected probe RED as an unfixed silent-drop defect; it records the deliberate fail-closed compatibility change.

The actual new error is a fixed-message `ImageSupportUnconfirmedError` carrying `serverRetryRequired`. Only `instanceof` that local class can mint this provenance in `buildFriendlyErrorMessage`; provider strings and arbitrary response objects cannot establish “not sent.” Decoder preserves only actual booleans. Pipeline hunk remains pending review at this checkpoint.

A separate read-only backend new-turn control changes only request metadata `tldw_retry_failed_turn` from true to false: `/private/tmp/cycle4-uat122-unacked-newturn-control.py` and `.log`, **3/3 passed**. Existing context-builder behavior accepts empty, different prior, and identical prior answered histories as new correlated turns. This confirms the frontend-only intent distinction is viable without changing backend exact-match safeguards.


## Final frozen implementation review

Reviewed the seven-path freeze at 2026-09-16T07:16:41.108094+00:00, base 6595bcfff4979d2495f34be39ebe18792d07a732. Independent final audit: **7/7 hashes match**, including all three production paths, three test paths and the test dependency alias; audit saved in /private/tmp/cycle4-uat122-independent-final-manifest.json. Private immutable probe manifest: /private/tmp/cycle4-uat122-independent-probe-manifest.json.

Final pipeline keeps local failed-turn identity/ACK behavior unchanged and passes the separate serverRetryRequired value only to pageAssistModel. Initial trusted local refusal carries false; missing provenance is conservative true; repeated local refusal carries the captured server intent forward. The trusted error is thrown before stream or invoke transport and contains no image bytes. The final hint explicitly supports history-bearing requests: choose an image model or a new text-only conversation.

Independently reran the frozen author suites, ordinary shared-UI config:

./node_modules/.bin/vitest run src/models/__tests__/ChatTldw.image-input.test.ts src/utils/__tests__/chat-error-message.test.ts src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx --maxWorkers=1 --no-file-parallelism

**93 tests /3 suites passed**, exit0, /private/tmp/cycle4-uat122-core-independent-green.log. Cases cover unknown/default/confirmed capability through actual factory; real custom/ordinary formatter; exact PNG/JPEG/WebP parts and whitespace; stream/invoke; explicit OCR and approved history refusal; class-only provenance and malformed/missing controls; initial refused Retry across remount and identical previous canonical image turns; false→actual ambiguous transport failure→true local refusal→supported recovery; one local user and canonical ACK identity; delayed authority A→B/A→B→A controls. The ambiguity path controls transport and DB boundaries; actual backend metadata semantics are independently checked by the private context-builder probes above.

The shared-UI test alias resolves its existing WebUI pa-tesseract.js dependency for Vite import analysis; tests mock OCR and do not create workers. It does not change runtime aliases or production behavior.

### Final evidence limits

No native browser, runtime, OCR worker, real vision provider, inference, whole compiler, lint or security tool was run by this reviewer. Root/author own their additional static checks and native negative acceptance. No positive real vision/UAT118 roundtrip is claimed. The historical OCR before/after RED is expected compatibility evidence, not a pending defect: prior code had already discarded extracted context, and the approved guard now refuses unproven historical image content visibly. Repository files were not edited by this review.
