# UAT122 / TASK13260.62 — no silent image removal

## Read-only diagnosis

Native evidence on `e7bf2184d3` shows the PNG in the local user row, but the actual initial POST contains only a text string. Canonical user `9437ffc0-ce39-45bf-b120-fc2fc99f35eb` has `images:[]`. The earlier UAT118 image recovery check is blocked, not passed.

The source cause is `ChatTldw.convertToTldwMessages`: HumanMessage content is normalized as multimodal only when `supportsMultimodal` is true; otherwise `coerceTextContent` discards every image part. This applies to initial send, Retry, prior history images, streaming and invoke. `models/index.ts` defaults false and sets true only for a catalog model with `capabilities.includes('vision')`; missing model/capabilities or catalog failure remain false. ChatTldw also maps an omitted constructor option to false. False therefore conflates known unsupported and unconfirmed capability and must not produce an overconfident diagnostic.

The real `humanMessageFormatter` custom image-only path is **not independently broken**. With empty text its custom branch falls through and returns the original image-containing content. Ordinary and custom paths preserve it; ChatTldw then drops it when capability is false. A known-vision model preserves the formatter's image-only and text+image MIME/data URL exactly.

Private actual formatter/model probe:

- `/private/tmp/cycle4-uat122-formatter-probe.test.ts`
- `/private/tmp/cycle4-uat122-formatter-probe.config.mts`
- `/private/tmp/cycle4-uat122-formatter-probe.log`: **3 expected failures / 4 positive passes**.

Command from `apps/packages/ui`: `./node_modules/.bin/vitest run --config /private/tmp/cycle4-uat122-formatter-probe.config.mts --maxWorkers=1 --no-file-parallelism`.

It uses real formatter, actual isCustomModel function and real ChatTldw; only transport/OCR dependencies are controlled. Vite initially could not resolve the optional OCR package despite the no-OCR test; a private alias provides an unused throwing createWorker stub. No OCR worker, inference or browser ran. This harness adjustment is not a product failure or fix.

## Recommended contract

Fail fast before `tldwChat.streamMessage` / `sendMessage` if any user message has an image content part and `supportsMultimodal` is false. Do not dispatch a shortened request. Pure text remains unchanged. Scan the whole outgoing user history, not only the new turn, so switching to an unconfirmed model cannot silently drop earlier images either.

Use existing error persistence and the friendly error envelope to retain the original local text/image and provide a model-selection action. Suggested summary: “Image support is not confirmed for this model.” Hint: “Choose a model that supports images, or remove the image before sending.” The existing `open-model-selector` action is sufficient. The error must not include image bytes or model credentials. Do not label this an offline/server failure.

Known vision follows the current complete-payload path without MIME/byte changes. Unknown/default capability follows the same fail-closed behavior with truthful wording. Do not infer vision from a model name, invent capability, modify discovery/auth, or introduce a new capability framework.

Explicit OCR is a separate existing opt-in: formatter-converted text may be sent as text. The proposed guard observes actual post-formatter image parts, so it will not block an intentional successful OCR-to-text conversion. Preserve existing OCR behavior and test it; no automatic OCR fallback or image-only OCR feature is proposed. Original image absence after an explicit OCR conversion is not this silent-drop defect.

### Why not always dispatch complete images?

Removing the conditional would preserve the frontend payload but would submit images to models already not marked as supporting them. Provider rejection is not universal; an adapter or provider might still ignore input. Fail-fast guarantees truthful local behavior before transport and uses the existing model-selection recovery flow. It also avoids treating the current non-vision provider as an image acceptance control.

### Persistence / Retry boundary to prove

The saved conversation may already have been bootstrapped before the model rejects the input. Existing error persistence must keep the local user/image and display error without pretending a canonical user ACK exists. Repeated blocked Retry must not create another local user, mutate its image, or dispatch. Switching to a known-vision model and retrying must send the original image exactly once, retain correlation, and accept the resulting canonical user/assistant IDs. The existing backend Retry path can persist an unsent turn when no matching unanswered canonical user exists; prove the current actual action behavior rather than broadening that contract speculatively.

Old native rows already persisted text-only cannot safely be retroactively associated with a local image; retain the existing mismatch guard and local work. No migration or attachment guessing.

## Proposed bounded source/test scope

Production:

1. `apps/packages/ui/src/models/ChatTldw.ts`: reject instead of lossy HumanMessage coercion when an image is present without confirmed vision.
2. `apps/packages/ui/src/utils/chat-error-message.ts`: narrow actionable friendly-error mapping and model-selector action.

No formatter production edit is currently justified. No backend, schema, capability registry, loader, shared ownership abstraction, or pipeline production change is proposed. If real action regressions expose an additional necessary boundary, report it before extending scope.

Permanent tests before production edits:

1. Actual formatter→ChatTldw stream/invoke: false and omitted capability reject with no transport; true capability preserves exact text whitespace and PNG/JPEG/WebP data URLs; custom and ordinary image-only; pure text unchanged; explicit OCR conversion remains intentional text.
2. Actual pageAssistModel: catalog vision true, capability absent/false, missing model and catalog rejection; no factory replacement that forces vision true.
3. Mounted saved normal Chat with **real human-message formatter**: blocked send retains one local user/text/image + actionable error; Retry while unsupported remains local and retains identity; selected vision model Retry dispatches exact image and one user; final canonical reload/remount retains user/assistant/attachment. Include prior successful image history and the current single-image contract.
4. Captured owner/cancellation controls: deferred catalog/model preparation followed by A→B and A→B→A cannot dispatch or persist stale content; preserve current existing owner tests rather than add new watchers.
5. Friendly error mapping: open-model-selector action, no base64 in message, ordinary provider/connection/empty-answer errors unchanged.

Suggested files: new focused model/formatter test files, existing `pageAssistModel.mcp-tools.test.ts` or a focused adjacent factory suite, existing saved-normal integration suite, and existing chat-error-message tests. Replace only the formatter stub in the actual image-boundary controls; isolate the optional OCR package so tests run without installing/starting OCR.

## Verification and limits

TDD on actual boundaries, focused existing Chat suites, root-config ESLint baseline comparison; no whole compiler by this agent. Python is untouched, so Bandit is non-applicable to this TS-only source correction (parent may retain combined security checks). Parent owns independent review, stable-runtime restart and native negative verification.

Native validation with the current `vision=false` provider should verify visible actionable refusal, no Chat POST, retained local image/text and stable Retry. A successful real vision response remains unverified until an actually supported model is available; do not change model/runtime as a workaround. UAT118 cannot be marked fully passed from these negative controls alone.

No repository source/test edits have been made for this design. Source release is still parent-owned.
