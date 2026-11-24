# Cross-device conversations in Document Workspace

TASK-15586 lets a signed-in user open the same PDF or EPUB on a second browser connected to the same tldw server and resume its saved conversations in the Chat sidebar. A document can have multiple conversations. The existing browser tabs remain the entry point. The Public Dev release depends on the tabs in TASK-15584. [Conversation tabs design](2026-11-20-document-workspace-conversation-tabs-design.md) · [Public Dev PR #3534](https://github.com/rmusser01/tldw_server/pull/3534)

## What follows the account

The sidebar combines server-saved conversation summaries with eligible histories saved only in this browser. Selecting a server-saved chat opens its latest saved transcript. The list refreshes on document open, on returning focus to the sidebar, and after a confirmed save. A refresh preserves the current selection and unsent draft. Empty tabs, draft text, the selected tab, and in-progress reply streams belong to the current browser session. The existing chat persistence choice governs new chats, including browser-only and temporary chats.

When opening a document with no current selection, choose its most recently active saved conversation or open a fresh draft if the list is empty. Preserve the tabs’ keyboard controls. Reuse Chat’s save and retry feedback so users can tell whether a conversation is available on the server or only on this device.

## Server association and access

Store one document association for each linked conversation in ChaCha, with the authenticated owner, canonical Media UUID, and native Chat identity. Bind to the original conversation record so deleting and recreating the same chat ID cannot inherit its link. Index by owner and document UUID for listing. The native Chat record supplies the title, history, and lifecycle.

Expose document-scoped operations to list conversation summaries, open a linked chat, and attach a saved conversation. Resolve the route’s media ID through the request’s authorized Media session to get the UUID. Verify the live document, current Chat owner, and matching link before opening a chat and before admitting a document-scoped send. Derive retrieval scope from the verified document. Use the same unavailable response for inaccessible or mismatched records. The association grants no document permissions.

Attach a new server-saved conversation before admitting its first saved turn. Reuse Chat’s retryable create and import identities, then attach with an idempotent write. An identical retry returns the established association. A request to attach that conversation to a different document returns a conflict. If Chat confirms the save and the link write fails, preserve the confirmed server chat ID in the account-scoped save recovery state so a retry can attach it after a reload. Show the conversation as synced once the server confirms both steps.

List live, owned chats by last activity with a stable ID tie-breaker and pagination. A trashed or deleted chat is omitted. Restoring the original chat can reveal its link again if the user still has document access. A document that is removed or becomes inaccessible cannot be used to discover or resume its chats from Document Workspace. A newly ingested document with a different UUID has its own list.

## Existing browser conversations

On the originating browser, reconcile owned document histories when the user reopens the document. For a history with a server copy, resolve its recorded document ID in the original server and account scope. Require that it identifies the open, accessible document and matches any recorded Media UUID. Confirm the live Chat record before attaching it. For a legacy mirror that lacks the original chat record identity, verify a persisted server message ID belongs to the current conversation. Leave ambiguous histories readable on their original browser. A confirmed local mirror and its server copy appear as a single tab. Preserve any browser-only turns until their save is confirmed.

Offer **Save to server** on each browser-only saved conversation. It saves only that conversation using the supported native import path and attaches it to the current verified document. Keep the original browser history available during the save. Once the server confirms the import and link, reconcile that history with the server copy. A lost response or repeated click reuses the captured transcript and the same import and link identities. Preserve later local turns until Chat confirms those writes. If the server cannot import the conversation, show the reason and keep the chat available on this device. The current native import rejects accounts with an active Sync v2 profile, so those browser-only chats remain readable on their original browser.

## Refresh and recovery

Load the selected transcript through Chat’s owner-validated history path. If another device advanced that history, preserve the draft, refresh the selected transcript, and let the user retry against the latest turn. Use the existing chat history error and branch behavior. Apply asynchronous results only while the captured server, account, document, and tab selection are current. Account or server changes clear the old visible state and cancel pending work.

If the server is unreachable or lacks the document-chat operations, retain accessible browser conversations and show the list’s current availability. A failed server read keeps the last verified list for the current scope with a retry action. A denied or changed scope removes that list. Keep the draft if a server write fails.

## Verification

- Open the same PDF and EPUB in two separately signed-in browsers. Create a saved chat on one, find it on the other, and continue it there. Return focus to the first browser and verify the saved transcript on selection, independent drafts, and one tab per server chat.
- Verify that a second account and a second document each load their own eligible conversations. Revoke access to a document and verify that discovery and resume are denied. Cover account switching during a list request, transcript load, link, and send.
- Exercise new chat creation, legacy server-copy linking, repeated imports, a lost save response, and a successful import followed by a failed link. Verify that retries return the same chat and that unsupported imports leave the local history accessible.
- Exercise a concurrent send, a network failure, trash and restore, and a recreated chat with the same ID. Cover access and link persistence against SQLite and PostgreSQL, plus sidebar loading, retry, and keyboard switching.

## ADR check

**ADR required:** Yes. Document discovery adds a durable association and an authorization boundary across Media and Chat. [Proposed ADR-098](../../ADR/098-document-workspace-chat-association.md) records that rule. [ADR-049](../../ADR/049-chat-history-selection-ownership.md) governs history selection and continuation.
