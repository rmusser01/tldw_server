# Browser-scoped conversation tabs in Document Workspace

Document Workspace currently restores the newest chat for a document. Add conversation tabs to the Chat sidebar so a person can keep separate threads while reading the same document. The browser's document-to-chat index supplies the list for the signed-in account. TASK-15586 tracks cross-device discovery.

## Sidebar behavior

- Show the document's saved conversations by title in the Chat sidebar. Use the existing tab and overflow patterns so the selected title remains identifiable in the narrow drawer.
- **New chat** opens a blank composer for the current document. The draft becomes a saved conversation on its first saved turn, and the tab takes the saved title. A new draft starts with its own transcript.
- Selecting a tab restores its transcript and its unsent draft for the current workspace session. Opening a document without an active selection chooses the account's most recently used saved conversation. A document with no chats opens a draft.
- Mark the selected tab for assistive technology. Support keyboard switching, a visible focus indicator, and an accessible name for the conversation list and **New chat**.

## Conversation identity and switching

Reuse the saved history identifier, document association, and verified account scope. List histories by the current document and account, then verify that a selected history belongs to that pair before displaying it. Open the exact selected history and its linked server chat. A new draft gets an independent history identifier when its first turn is saved. Adopt that identifier in the same tab after the save to preserve the active composer and reply.

Restore the active tab's transcript, draft, and document-content choice together when switching tabs. A saved chat loaded from browser history starts with the existing composer defaults. Use the open document as the retrieval scope and restore saved messages with their original identities, including in server-backed transcripts.

Scope pending work to the account, document, and conversation captured at its start. Switching tabs stops an in-progress reply using the workspace's existing cancellation behavior. Any retained partial belongs to its source conversation. A late read or response may update the visible panel only while its captured selection is current. During a conversation load, show progress and enable sends once the requested history and account are ready. On a load error, keep the tabs available and let the user retry the selected conversation.

## Verification

- Open a document with an existing chat, create a second chat, exchange messages in both, and reopen the document. Each title restores its own saved transcript.
- Switch tabs rapidly during a history load and switch again during a streamed answer. Verify that each completed or retained partial turn remains attached to its source and that the selected tab responds only to its own drafts and messages.
- Verify isolation across documents and signed-in accounts, including conversation lists, restored transcripts, and late responses. Check that document retrieval uses the current document in both conversations.
- Verify that repeated tab switches and reloads render one entry per saved message identity.
- Exercise keyboard switching and overflow at sidebar and drawer widths.

## ADR check

**ADR required: no**

The browser list and selection reuse the existing document index, saved conversation identities, and account ownership contract. The accepted owner-bound history decision governs the restore and send paths. [ADR-049](../../ADR/049-chat-history-selection-ownership.md)
