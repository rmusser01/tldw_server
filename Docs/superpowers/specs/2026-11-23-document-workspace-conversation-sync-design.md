# Cross-device conversations in Document Workspace

TASK-15586 lets a signed-in user open the same PDF or EPUB on a second browser connected to the same tldw server and resume its saved conversations in the Chat sidebar. A document can have multiple conversations. The existing browser tabs remain the entry point. The Public Dev release depends on the tabs in TASK-15584. [Conversation tabs design](2026-11-20-document-workspace-conversation-tabs-design.md) · [Public Dev PR #3534](https://github.com/rmusser01/tldw_server/pull/3534)

## What follows the account

The sidebar combines server-saved conversation summaries with eligible histories saved only in this browser. Selecting a server-saved chat opens its latest saved transcript. The list refreshes on document open, when the browser tab returns to the foreground with the sidebar open, and after a confirmed save. A refresh preserves the current selection and unsent draft. Empty tabs, draft text, the selected tab, and in-progress reply streams belong to the current browser session. The existing chat persistence choice governs new chats, including browser-only and temporary chats.

When opening a document with no current selection, choose its most recently active saved conversation or open a fresh draft if the list is empty. Preserve the tabs’ keyboard controls. Reuse Chat’s save and retry feedback so users can tell whether a conversation is available on the server or only on this device.

## Server association and access

Store one document association for each linked conversation in ChaCha, with the authenticated owner, canonical Media UUID, and native Chat identity. Bind to the original conversation record so deleting and recreating the same chat ID cannot inherit its link. Index by owner and document UUID for listing. The native Chat record supplies the title, history, and lifecycle.

Expose document-scoped operations to list conversation summaries, open a linked chat, and attach a saved conversation. Resolve the route’s media ID through the request’s authorized Media session to get the UUID. Verify the live document, current Chat owner, and matching link at the server operation that opens or admits a turn in a linked chat. Use the same unavailable response for inaccessible or mismatched records. Browser-only and temporary chats use the existing Media permission checks when retrieving document content. Bind requests and pending saves to the captured server and account using the existing expected-user guard.

Attach a new server-saved conversation before admitting its first saved turn. Reuse Chat’s existing recovery for an unconfirmed create. An attach writes atomically under the Chat owner and rechecks the original record and document UUID on retry. An identical retry returns the established association. Attaching the same chat to a different document returns a conflict. If Chat confirms the save and the link write fails, retain the confirmed target under the captured server, account, and document so a retry after reload can attach it. Publish the tab to other devices once the link and a saved message exist.

List live, owned chats by last activity with a stable ID tie-breaker and pagination. A trashed or deleted chat is omitted. Restoring the original chat can reveal its link again if the user still has document access. Removal or loss of Media access blocks Document Workspace discovery, resume, and retrieval on the next request. Saved transcripts and passages already included in them follow ordinary Chat ownership and retention. Opening a source link requires current Media access. A newly ingested document with a different UUID has its own list.

## Continuing a saved document chat

Bind the selected tab to Chat’s canonical history selection. The shared history pipeline currently rejects native chats in document-grounded mode. Qualify the document send path by reusing native admission and reply settlement with document authorization, and constrain retrieval to the verified Media record. Keep **Use document content** available for a linked chat. Require an end-to-end check of a document-grounded reply after resuming on a second device before enabling cross-device sends. [Shared history guard](../../../apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts)

For a regular send, use Chat’s latest-history precondition so an intervening message causes a refusal before appending a turn or calling the answer model. Preserve the draft, refresh the transcript, and let the user retry against the latest turn. Once Chat admits an input, settle its reply only against that input under the existing ownership checks. Use the current role and capability checks for the account, including active Sync v2 profiles.

## Existing browser conversations

Record the authorized Media UUID in new browser document chat histories. On the originating browser, reconcile owned histories when the user reopens the document. Automatically attach a verified server copy when the saved Media UUID matches the current document under the same server and account. Verify the original Chat record or a matching persisted message on that record. Legacy rows only recorded a numeric document ID. If that is the available document evidence, show the current document title and let the user confirm the link for that chat. A chat whose owner or server copy cannot be verified stays out of the server list. A confirmed mirror and its server copy appear as one tab. Keep browser-only turns accessible until their save is confirmed. [Browser history fields](../../../apps/packages/ui/src/db/dexie/types.ts)

Offer **Save to server** on each browser-only saved conversation. Confirm the intended document for a numeric-only legacy history before uploading. Native import requires a UUID, while browser histories use `pa_` IDs. Reserve one import UUID and its captured transcript per local conversation in a browser transaction scoped to the server, account, and document. Concurrent browser tabs and retries after a lost response or reload reuse that save. Reconcile the browser history only after the server confirms the import and link and accounts for the saved turns. If another browser tab added turns after capture, keep those turns accessible in the browser. [Browser ID creation](../../../apps/packages/ui/src/db/dexie/helpers.ts) · [Import request contract](../../../tldw_Server_API/app/api/v1/schemas/chat_import_schemas.py)

Preserve text, roles, timestamps, verified reply order, supported assets, and visible document citations when saving or importing a conversation. Use Chat’s legacy history review when the saved reply order is ambiguous. The current native import schema accepts a full message graph and limited generation metadata. Define one bounded citation representation for document chat saves and imports, with the same Media access rule on citation links. If retained data cannot be represented safely or the account cannot import, explain why and keep the original conversation available. The current native import refuses accounts with an active Sync v2 profile.

## Refresh and recovery

Apply asynchronous results only while the captured server, account, document, and tab selection are current. Account or server changes clear the old visible state and cancel pending work. Keep transcript visibility and the ability to send as separate states so a connection failure can leave a local transcript and its draft readable within the verified account scope.

Use server capability discovery to select browser-only behavior on older servers. For a supported server, a transient failure keeps the last verified list available for retry in the already authorized document session. Opening another server chat requires a fresh authorized read. A document access denial clears its server list and transcript. A chat access denial clears that chat from the list and view. Show the current availability and preserve the draft when a write fails.

## Verification

- Open the same PDF and EPUB in two separately signed-in browsers. Create a saved chat on one, find it on the other, and continue it with **Use document content** enabled and disabled. Verify the current transcript and citations on both devices, independent drafts, and one tab per server chat.
- Verify isolation across accounts and documents, including a shared document with two owners of separate chats. Revoke access between list and open, and between open and send. Cover account switching during a list request, transcript load, link, and send.
- Create and link a chat, then fail its first send before a message is saved. Verify that the second browser sees the chat once there is a saved message. Exercise a concurrent send from the earlier transcript and verify the conflict keeps the draft and saves no extra turn.
- Reopen a numeric-only legacy history whose ID now points to a different Media UUID. Verify that linking or uploading waits for document confirmation. Import a browser-only conversation with citations from concurrent browser tabs. Retry after a lost response or a failed link and confirm the same chat is published once. Verify unsupported imports and turns added during import remain readable in the original browser.
- Exercise a supported server with network failure and explicit access denial, plus an older server. Cover trash, restore, and a recreated chat with the same ID. Check persistence and ownership against SQLite and PostgreSQL, plus sidebar loading, retry, and keyboard switching.

## ADR check

**ADR required:** Yes. Document discovery adds a durable association and an authorization boundary across Media and Chat. [Proposed ADR-100](../../ADR/100-document-workspace-chat-association.md) records that rule. [ADR-049](../../ADR/049-chat-history-selection-ownership.md) governs history selection and continuation.
