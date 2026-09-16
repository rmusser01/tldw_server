# Chatbook chat parity design review

Date: 2026-09-16. Tracking: TASK-13261. Scope: the [call-path audit/design](2026-09-16-chatbook-chat-parity-call-path-audit.md) and [90-row inventory](2026-09-16-chatbook-console-parity-matrix.md).

The [closure addendum](2026-09-16-chatbook-chat-parity-review-closure.md) records subsequent iterative review and current decisions. These findings describe this historical pass; deferred choices are superseded by the addendum.

This is the first review record. The [second review](2026-09-16-chatbook-chat-parity-design-review-2.md) adds five findings and refines the copy rules, consent/queue/voice lifetimes and H1–H4 gates. Use the current audit for the resulting design.

**Verdict: revise before implementation.** Nine issues were found. The audit and inventory have been corrected, including the delivery sequence and acceptance contracts. Those document corrections do not mean the underlying application defects have been fixed or that the implementation specification is approved.

The review used the same pinned source: Chatbook `24094f23d59c7a9d3cfac964c19fd263bc0393b2` and server `59049e094e0845a4611ea725ae19b7c1754ea709`, including its bundled `tools/tldw-agent`. Three independent read-only reviews covered fork persistence, local/sync architecture, and runtime/UI evidence; the primary reviewer checked the findings against the source and reviewed scope and qualification.

## Findings

| ID | Priority | Issue | Design correction |
|---|---|---|---|
| R1 | P1 | A local fork can retain source mutation identities and mis-own attachments. | Explicit copy/remap/drop contract and source-isolation tests. |
| R2 | P1 | Fork precedes the history/selection contract it depends on. | Separate revision fences, legacy interpretation and authoritative selected path first. |
| R3 | P1 | Pinned Chatbook and server chat sync contracts are incompatible. | Explicit versioned adaptation through public schemas and reverse apply. |
| R4 | P1 | Enrolled sync changes server chat behavior and drops required fork fields. | Per-mode retention, enrollment scope and supported-operation gates. |
| R5 | P1 | The proposed atomic operation spans multiple stores and lacks ambiguous-result recovery. | Owner transaction, durable receipt, bounded publication and character snapshot contract. |
| R6 | P1 | Independent local generation and cross-origin continuity are unspecified. | Independent model connection, WebUI availability and explicit transfer semantics. |
| R7 | P1 | ACP has a second serial reader that can block other sessions. | Both receive layers, prompt ownership and pending-decision cancellation in the control repair. |
| R8 | P2 | Queue analysis misses a different failure path in the sidepanel. | Propagate structured terminal outcomes and test direct versus queued turns separately. |
| R9 | P2 | The first delivery and validation claims are too broad. | H1–H4 delivery gates and behavior fixtures that reach their assertions. |

### R1. Fresh local IDs alone do not establish independent ownership

Audit E1 and matrix B02 treated Dexie spread-copying as richer reuse without describing its hazards. The source conversation/message objects can include `server_chat_id`, `serverMessageId`, versions, parents and runtime metadata. Branching changes local IDs but retains those values; rendering preserves them and Delete routes to the server when `serverMessageId` is present. A fork of mirrored history can therefore retain source write authority. The file copy also writes `history_id` while the table is keyed by `sessionId`, leaving the new branch without its intended file association. [Copy](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/db/dexie/branch.ts#L27), [rendered identity](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/db/dexie/helpers.ts#L298), [delete routing](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/hooks/chat/useChatActions.ts#L4436), [file key](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/db/dexie/schema.ts#L376).

**Corrected:** require new mutation identities, an old-to-new message map, remapped parents/selection/pins, correctly owned file references and independent settings. Retain ancestry as provenance only. Clear source server/sync bindings and exclude live work. Acceptance now exercises actions on a reopened child and checks the source remains unchanged. This is a demonstrated code-level risk; no destructive live test was performed.

### R2. “Source revision” cannot mean only the conversation version

Audit E1's generic revision and the previous fork-before-variants sequence were insufficient. Message changes advance `history_version`; settings have another version. The selected variant changes only the normal full-page display, while generation uses separate history. Existing user-message writes can omit parents, and resume selects a timestamp tail. Walking parents from a selected node would not recover all existing conversations. [History version](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/tldw_Server_API/app/core/DB_Management/chacha/message_store.py#L65), [resume snapshot](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/tldw_Server_API/app/core/DB_Management/chacha/conversation_resume_store.py#L482), [user-message payload](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/hooks/chat/useChatActions.ts#L2130), [variant selection](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/components/Option/Playground/PlaygroundChat.tsx#L1149).

**Corrected:** establish the minimum selected-path contract before fork, covering conversation/history/settings/selection/asset changes. Specify a deterministic interpretation for provably linear legacy history and reject ambiguous graphs without inventing ancestry. Use the same chosen path for normal provider requests and fork. Comparison mode has a separate projection and must retain its existing behavior. The later full edit/rewind project remains separate.

### R3. Existing Sync v2 foundations do not establish chat wire compatibility

Audit E4 and matrix F04/F06 previously presented the producer, adapters and materializers as one usable chat protocol. Chatbook emits legacy `chat/upsert` or `chat/delete` with encrypted role-based content. Its API sends that model directly. Server public chat domains are `chat.conversation` and `chat.message`; messages use `append/tombstone`, and M1 permits `server_trusted_v1`. The local Chatbook apply registry lacks those two public chat domains. Identity/cursor aliases do not translate these semantics. [Producer](https://github.com/rmusser01/tldw_chatbook/blob/24094f23d59c7a9d3cfac964c19fd263bc0393b2/tldw_chatbook/Sync_Interop/envelope_builder.py#L329), [direct serialization](https://github.com/rmusser01/tldw_chatbook/blob/24094f23d59c7a9d3cfac964c19fd263bc0393b2/tldw_chatbook/tldw_api/client.py#L16453), [server operations](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/tldw_Server_API/app/core/Sync/v2/models.py#L111), [public validation](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/tldw_Server_API/app/api/v1/schemas/sync_v2_models.py#L1951), [reverse apply](https://github.com/rmusser01/tldw_chatbook/blob/24094f23d59c7a9d3cfac964c19fd263bc0393b2/tldw_chatbook/Sync_Interop/envelope_applier.py#L76).

**Corrected:** use the registered public domain identities as the integration target, and explicitly adapt operations, payloads, parent dependencies, identities, revisions/hashes and reverse apply. Retain existing outbox/receipt/recovery owners. The version/encryption design remains to settle: silently decrypting private data into M1 is not an acceptable adapter. A schema-only probe confirmed four specific incompatibilities below; a complete cross-repository round trip remains required. The matrix now links the production materializer rather than an unregistered legacy server adapter.

### R4. Active sync is a materially different persistence mode

The session endpoint validates exact fork ancestry, but its active-sync serializer and materializer omit parent conversation and fork boundary. The inspected active-sync message route rejects binary images and edits; streamed character-completion persistence is also unsupported. Activation is selected by the user's default personal Chatbook dataset, not an individual conversation's enrollment. Thus a successful native-mode fork test says nothing about these enrolled-mode cases. [Lossy payload](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py#L832), [projection](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/tldw_Server_API/app/core/Sync/v2/materializers/chat.py#L68), [image restriction](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/tldw_Server_API/app/api/v1/endpoints/character_messages.py#L455), [edit restriction](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/tldw_Server_API/app/api/v1/endpoints/character_messages.py#L988), [stream restriction](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py#L8205), [activation](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/tldw_Server_API/app/core/Sync/v2/server_origin.py#L293).

**Corrected:** name native-server and enrolled-sync paths separately, define explicit conversation/dataset consent, and require retained-field/capability checks with a genuinely enrolled dataset. An unsupported required mode stays incomplete. A fork must not bypass active sync through a direct DB write that is absent from publication. Browser reachability-driven auto-save also cannot supply consent.

### R5. One transaction is insufficient across settings, blobs and sync

The existing server DB can transact conversation/message/settings work together, but browser settings live outside Dexie, and server SyncDatabase is separate from the chat DB. A sync mutation group appends envelopes atomically but materializes them one by one, with a possible return after a later failure. This does not establish all-or-nothing visible chat creation. The current action may create a local fallback after an unknown server result. [Browser settings](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/services/chat-settings.ts#L622), [separate server stores](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/tldw_Server_API/app/core/Sync/v2/factory.py#L433), [individual materialization](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/tldw_Server_API/app/core/Sync/v2/server_origin_batch.py#L917), [fallback](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/hooks/handlers/messageHandlers.ts#L508).

There are also hidden retained-state dependencies: the cited message endpoint supports one image, not arbitrary documents; character reopen requires a valid behavior snapshot and matching settings, while ordinary creation rematerializes current definitions. [Message schema](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py#L400), [resume admission](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/tldw_Server_API/app/core/DB_Management/chacha/conversation_resume_store.py#L549), [character materialization](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/tldw_Server_API/app/core/Character_Chat/character_conversation_factory.py#L2391).

**Corrected:** commit the owned graph, settings/snapshot, mapping and operation receipt coherently. Make separate settings-cache, blob and sync publication recoverable; all list/load/send readers must honor the completion boundary. Reuse the same operation ID/digest after a lost response, return its committed child before rejecting a now-stale source, and reject different-input key reuse. Remove automatic owner-changing fallback for an unknown outcome. The fork projection explicitly handles accepted character snapshots, greeting exclusion, branch-bound summaries, document representation and selected asset revisions. The implementation specification must choose the concrete receipt/publication storage and prove rollback/replay on both DB backends.

### R6. Local persistence, model availability and surface identity are separate contracts

Audit E4 specified local storage/sync but left the required generation path unresolved. The ordinary model factory always constructs `ChatTldw`, which sends through the configured `tldwClient`. Submission queues on server readiness. A cached model name plus Dexie history does not provide generation if that connection is down. WebUI and extension local storage also live in different origins; matrix F01's unqualified “same conversation in each surface” could silently demand upload from local-only mode. [Factory](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/models/index.ts#L246), [transport](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/services/tldw/TldwChat.ts#L539), [admission](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundSubmit.ts#L293).

**Corrected:** independently bind storage owner, model endpoint/credentials/capabilities, sync enrollment, and execution target. A supported local model route must work while the synchronization server is down. A local deployment of the existing server is one possible endpoint; direct provider support is another bounded adapter decision. Specify WebUI shell availability for cold reopen and preserve drafts on missing capability. Local-only behavior qualifies independently in each origin; cross-origin continuation needs explicit transfer or enabled sync. Extension sidepanel expansion shares its extension owner. None of this gives `tldw-agent` a chat-hosting, storage or sync role.

### R7. ACP responsiveness requires fixing both receiving layers

Audit E7 correctly identified the WebSocket loop awaiting a whole prompt while the same socket carries decisions. The review found a second dependency: the normal shared stdio reader awaits a permission handler, and that handler waits for a user decision. One session's pending permission can therefore block consumption of another session's progress/results. Cancel sends a notification but does not settle the pending permission future. [WebSocket receiver](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py#L1314), [stdio receiver](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/tldw_Server_API/app/core/Agent_Client_Protocol/stdio_client.py#L194), [decision wait](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/tldw_Server_API/app/core/Agent_Client_Protocol/runner_client.py#L1112), [cancel](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/tldw_Server_API/app/core/Agent_Client_Protocol/runner_client.py#L848).

**Corrected:** design responsive readers, bounded session-owned work, admission across REST/WebSocket, disconnect policy and cancellation settlement together. The decisive fixture uses the actual stdio reader: A waits for permission, B continues, A is cancelled, and late approval cannot act. A mock whose prompt returns immediately cannot establish this property. Existing durable ACP/orchestration owners remain appropriate.

### R8. Sidepanel queue outcomes differ from full Playground

The earlier queue wording generalized idle auto-drain. Full Playground converts failed/skipped submit outcomes into a blocked queue head. Sidepanel drops the chat mode outcome in `useMessage.onSubmit`, then drops the mutation result in its queue sender, allowing failure/cancellation to appear successful. Directly submitted Stop still needs explicit pause on both surfaces. [Full-page guard](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundQueueManagement.ts#L435), [sidepanel submit](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/hooks/useMessage.tsx#L2901), [sidepanel queue sender](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/components/Sidepanel/Chat/form.tsx#L2630), [resolved dispatch removal](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/hooks/chat/useQueuedRequests.ts#L159).

**Corrected:** propagate terminal results through both sidepanel layers, preserve existing full-page failure behavior, and add explicit pause/resume. Test direct/queued origins, Stop/failure, and cancellation while dispatch is starting on both surfaces. This is a bounded repair that can ship independently of history work.

### R9. Delivery readiness needs smaller units and real behavioral proof

The proposed first fork included ordinary/character chats, legacy/variant selection, image/document assets, local/temporary/server/sync ownership, two full browser surfaces and cross-store recovery. That is several reviewable units. Passing existing branch tests proves current behavior, while several source/UI suites reported in the audit failed before their intended assertions. They cannot be acceptance evidence for the revised contract.

**Corrected:** the audit now separates H1 selection/ownership/recovery, H2 native server fork, H3 browser local/temporary completion, and H4 compatible synchronized publication. H2/H3 can proceed independently after their shared contract; H4 gates any claim of synchronized parity. Queue and ACP corrections are independent. All required modes remain in B01–B03 until qualified.

For each delivery, use provider-free behavior fixtures that exercise its public boundary, then the appropriate live surface journey. Include unauthorized/stale identities, history/settings changes, active/partial generation, precommit failure, postcommit lost response, restart/replay, missing assets, child mutation isolation, and both supported database backends. Repair only harness issues that block the selected delivery; do not turn unrelated suite cleanup into a new universal prerequisite. Existing mixed test results remain unchanged and explicitly labeled as earlier investigation.

## Verification in this review

Static findings were checked against the pinned implementations above. No application code was edited, no provider/external-host journey was run, and the earlier application suites were not rerun.

A fresh probe imported the pinned public `SyncV2Envelope` schema using the activated server virtual environment and synthetic data:

| Case | Result |
|---|---|
| M1 `chat.message/append` control | Accepted by schema |
| Legacy `chat/upsert` | Rejected domain |
| Rename only to `chat.message/upsert` | Rejected operation |
| Correct domain/operation with `client_private_v1` | Rejected encryption policy |
| Legacy `delete` operation | Rejected operation |

The command exited 0 with assertions for all five outcomes. This isolates schema incompatibility; it does not prove payload materialization, HTTP authorization or full sync interoperability. Its original JSON record was `/private/tmp/tldw-chat-parity-audit.Tq0UXo/logs/design-review-sync-schema.json`. The temporary directory was no longer present at the subsequent H1 handoff; the pinned reproduction below remains available, but that original log does not.

Reproduce from the pinned server snapshot after activating the project's virtual environment:

```python
from pydantic import ValidationError
from tldw_Server_API.app.api.v1.schemas.sync_v2_models import SyncV2Envelope

base = dict(
    client_envelope_id="review-envelope", dataset_id="review-dataset",
    domain="chat.message", operation="append", object_id="review-message",
    payload_hash="review-hash",
    payload=dict(conversation_id="review-chat", sender="user", content="review"),
)
cases = [
    ({}, True),
    (dict(domain="chat", operation="upsert"), False),
    (dict(operation="upsert"), False),
    (dict(encryption_metadata=dict(policy="client_private_v1")), False),
    (dict(operation="delete"), False),
]
for changes, expected in cases:
    try:
        SyncV2Envelope.model_validate({**base, **changes})
        accepted = True
    except ValidationError:
        accepted = False
    assert accepted is expected
```

Document verification passed: 227 pinned file/line targets across 163 source files, six local link targets, all 90 unique ordered inventory IDs, reference labels and whitespace. All nine review findings and E1–E12 sections are present. Inventory counts remain 70 Partial, nine Missing in the inspected path and 11 Unverified; none is marked Equivalent. Bandit does not apply to these documentation/tracking-only changes. TASK-13261 remains In Progress for the selected sub-project's specification and design approval.

## Remaining design decisions before implementation

The corrected architecture still requires concrete choices in the selected specification: authoritative selection/legacy interpretation; typed retained-field projection and character snapshot rebinding; the exact receipt/publication transaction boundary; supported independent model connection and WebUI shell availability; and version/encryption adaptation wherever sync is enabled. These choices must be reviewable before implementation proceeds. They do not require reopening the user's settled surface scope, independent-local/optional-sync requirement, or `tldw-agent` role.
