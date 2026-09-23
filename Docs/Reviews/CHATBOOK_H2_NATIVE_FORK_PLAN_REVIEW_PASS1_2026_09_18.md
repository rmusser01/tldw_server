# H2 native fork implementation-plan review — pass 1

Date: 2026-09-18. Independent requesting-code-review seat.

Reviewed worktree: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design`, branch `codex/chatbook-h2-native-fork`, baseline `ac76c4bc5b035561bf816009c1326a114e87def9`.

**Verdict: ready to implement with fixes — no P1, three P2 findings.** These findings concern the proposed interfaces and qualification plan. H2 application code/tests are not implemented, and this review does not establish any runtime acceptance.

The line references below refer to the initial 410-line implementation plan and 183-line design read during this pass, before the coordinator's concurrent corrections. The coordinator has acknowledged P2-1/P2-2 and is preparing changes; that acknowledgment is not a rereview result.

## Review scope and method

Read the entire implementation plan and focused design; source audit; full character, asset and explicit-retention reports; first design-review report; approved D2 and retained D6 obligations. Inspected the current H1 source for selected-history admission, frozen character projection, accepted snapshot construction, settings reads/updates, summary persistence, client operation result/types, cold-load settings authority, document upload/removal, generated-image requests and reference resolution, and actual scoped browser transports.

No repository files, index, HEAD, branch state, applications, databases or running services were changed. No application tests or builds were run. Only this report was written outside the repository. Receipt/GC/quota lifecycle is assigned to the other review seat, so the prior R1–R3 closure requires that seat's assessment.

## Strengths

- Scope matches the approved split: native H2 has concrete A1–A8 work, while H3/H4/F02/persona/multi-participant/comparison and the extra WorkspaceChatPanel qualification remain explicit. There is no claim that H2 documentation proves parity.
- The plan names proposed new files as new and distinguishes planned commands from results. Stages generally respect dependency order: schemas/stores, retained bytes, frozen identity/composition, orchestration/clients, then real qualification.
- Snapshot-owned character identity closes the actual card FK cascade while keeping ordinary public character creation strict. The plan covers cold reopen without a local receipt, accepted-name display, child edits, first-send/no-greeting behavior and poisoned live readers.
- Rich behavior qualification checks actual provider payloads rather than only snapshot byte equality. Neutral chats and corrupt-required-state cases are explicit.
- Attachment promotion truthfully distinguishes extracted text from original binary, uses existing validated upload machinery, forbids external-ID ownership shortcuts, and recaptures after retention.
- Both full-page shells, mounted shared consumers, real scoped transport, real SQLite/PostgreSQL, lost responses and legacy-protocol separation are planned. Required skips do not count as passing gates.

## P2-1 — Carry the admitted history fence into child-summary persistence

**Plan:** `IMPLEMENTATION_PLAN_chatbook_h2_native_fork.md:79–83` and `:294`; related `:307`.

The proposed `NativeChildSummaryUpdate` contains `child_id`, `expected_settings_revision` and `summary_json`, and Task 3.2 explicitly describes persistence as a settings-revision CAS. The `FrozenHistoryBehaviorPlan` also has no admitted history revision. This loses a fence that the source implementation already requires.

**Source-backed failure scenario:** A send admits selected child history and computes its deterministic summary outside the transaction. Before its delayed summary update is applied, another view edits or deletes a retained child message. That mutation advances the history fence but need not change settings. The proposed settings-only CAS still succeeds and writes summary content/ranges derived from the old message state into the child's current canonical settings/materialized behavior. Depending on how the existing helper is reused, omitting its history version can instead cause every update to return early. Neither outcome implements the claimed child-only fenced summary policy.

**Evidence:**

- `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py:4209–4218` requires both `expected_settings_version` and `expected_history_version`, then compares each under a locked resume read.
- The existing caller supplies both at `character_chat_sessions.py:4370–4379`.
- `tldw_Server_API/app/core/DB_Management/chacha/message_store.py:932–943` advances history independently of settings for message mutations.
- `tldw_Server_API/app/core/Chat/chat_service.py:3965–3974` admits the selected history and appends current input in one transaction. The plan's composer runs after that transaction, creating the relevant race window.

**Bounded repair:** Add the correct admitted history version to the detached plan/update contract. Specify whether it is the post-input version and carry that version directly from admission; do not re-read a newer history version just to make the CAS succeed. Compare both settings and history under the protected child settings transaction. If a more precise selected-history token is chosen, it must still reject changed summarized content. Add deterministic held-composition tests in which a child message edit/delete wins before persistence, plus a no-race positive summary update and settings-edit race. Maintain the frozen envelope rebuild and no-live-reader rule.

## P2-2 — Define the server mutation that resolves an unavailable attachment

**Plan:** `IMPLEMENTATION_PLAN_chatbook_h2_native_fork.md:243–260` and `:360`. **Design:** `Docs/Design/2026-09-18-chatbook-h2-native-fork-design.md:46–48` and `:156`.

A6 promises that restore/replacement or explicit removal from context advances the canonical asset/context revision and unblocks dependent sends. The wire/interface map supplies upload retention, retention lookup and asset read, but no request/route/store operation for changing an existing context reference or removing an unavailable marker. Retention is defined as verified uploaded bytes being attached under the original source fence; a pure removal has no upload, and an additive retention alone does not specify which old required reference is superseded.

**Source-backed failure scenario:** Create the permitted degraded child with a required unavailable document/reference-image marker. Both clients correctly block dependent sends. Clicking the existing attachment Remove control updates browser presentation only; the authoritative server marker remains, so cold reopen or the next native admission still blocks. Conversely, allowing browser hiding to count as resolution violates the stated admission invariant. Uploading replacement bytes without an explicit expected-reference transition can leave both the old marker and new claim required or replace a reference changed by another view.

**Evidence:**

- `apps/packages/ui/src/hooks/chat/useFileUpload.ts:173–186` only optionally cancels processing and filters `uploadedFilesRef`; it has no native context mutation.
- `apps/packages/ui/src/services/chat-document-processing.ts:760–819` implements cancellation against Media jobs/batches/upload drafts, not an independently owned native manifest.
- The plan's listed service methods (`prepare_retention`, `retain_assets`, `read_owned_asset`) and design route table do not define removal/replacement semantics or a result carrying the new canonical revision.

**Bounded repair:** Specify an authenticated, scoped, projection-gated owner/revision-fenced native context update request and response. It should identify the exact reference/old asset revision and expected context fence; support pure remove without source-job cancellation; and support restore/replace only with an already verified owned native claim or the explicitly defined combined retention transaction. Describe how context changes advance the canonical fence, return the updated manifest, invalidate stale fork captures/sends, and preserve bytes still used by other references. Name the actual shared attachment consumer files and route/proxy wiring. Test degraded cold reopen → remove → accepted send, verified replacement → accepted send, wrong owner, stale reference CAS, response recovery as appropriate, and absence of source cancellation in both shells.

## P2-3 — Specify the native reference-image consumer wire contract

**Plan:** `IMPLEMENTATION_PLAN_chatbook_h2_native_fork.md:308`, `:342` and `:360`. **Design:** `Docs/Design/2026-09-18-chatbook-h2-native-fork-design.md:142`, `:156`.

The plan promises to remap executable refine/reference-image links to native claims and wire them into existing image consumers, but its interfaces/file map only cover retention/read plus chat completion admission. The existing image-generation consumer has a different request and namespace. A typed native claim cannot pass through it merely by changing the attachment renderer or replacing an ID.

**Source-backed failure scenario:** Retain and fork a generated revision with the reference image needed for its next image-generation action; delete the external Media original. The child displays its independently retained image successfully. Reusing its generation/reference action either sends the old `referenceFileId` (which now fails or refers to a different live resource) or drops the native reference and generates without it. The proposed ordinary chat admission/header guard does not intercept `/api/v1/files/create`, which has no child/claim identity in the current request.

**Evidence:**

- `apps/packages/ui/src/utils/image-generation-chat.ts:19–32` exposes only `referenceFileId?: number` in `ImageGenerationRequestSnapshot`.
- `apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts:337–365` forwards that foreign numeric reference.
- `apps/packages/ui/src/services/tldw/domains/media.ts:1524–1566` emits only numeric `reference_file_id` to `/api/v1/files/create`.
- `tldw_Server_API/app/core/File_Artifacts/adapters/image_adapter.py:113–160` normalizes known fields and recognizes only `reference_file_id`; an invented native-claim payload field is not consumed.
- `image_adapter.py:219–245` resolves that ID through `core/Image_Generation/reference_images.py:191–211`; its repository query at `reference_images.py:64–85` requires a live MediaFiles/Media original.
- The visible “Refine with LLM” helper is separately implemented at `components/Option/Playground/hooks/usePlaygroundImageGen.ts:350–384`; it sends prompt text without a conversation identity. Name which image/refine actions require native asset admission instead of assuming every such action traverses the chat history controller.

**Bounded repair:** Add the exact typed native owner/conversation/asset/revision reference and authenticated resolver/admission seam through the image request, serializer, FileArtifacts boundary and image adapter; preserve ordinary numeric Media reference behavior for unrelated requests. Keep native claims and foreign provenance disjoint. Require missing/stale/unauthorized native references to fail before artifact/backend dispatch, and carry the frozen owner through both client transports. Alternatively explicitly gate this action as an unsupported capability and leave its rich acceptance cell open; do not claim successful rendering establishes independent reference-image use. Add a production-handler test with an intercepted image backend proving exact retained reference bytes after deleting the Media original, plus missing/stale/wrong-owner negatives and both-shell dispatch assertions.

## Minor corrections

1. **Actual extension transport path:** Plan `:342` calls `apps/extension/entrypoints/background.ts` an exact proxy/header modification path, but that file contains only `export { default } from "@tldw/ui/entries/background"`. The actual handler is `apps/packages/ui/src/entries/background.ts` (`:1420` upload path, `:1684` request path), and direct/WebUI upload runs through `services/background-proxy.ts:2231–2440` and shared request-core. Name those real seams and the precise WebUI shim if it actually requires a change; the instruction to test both transports is already sound.
2. **Result discriminator example:** Plan `:164` asserts `result.status == "gone"`, while its `:92` requires exact H1 committed-result names and H1 `types/history-selection.ts:183–208` discriminates fork results with `state`. Pick and state the new union discriminator explicitly, then make the sample test consistent. This is a doc/test-example mismatch, not runtime proof of broken recovery.

## Overall assessment

A1–A8 all have named implementing tasks, and no unjustified broader H3/H4/F02/persona/routing parity claim was found. The principal missing precision is concentrated in new effect/asset interfaces, not a reason to redesign the owner/receipt architecture. Fix P2-1 through P2-3, rereview the resulting exact contracts and added qualification obligations, and retain the lifecycle review's separate R1–R3 disposition before declaring the specification complete.
