# Document Workspace chat send repair

Document Workspace supplies no account-bound scene to the shared send path. The send guard refuses the question before contacting the model. The workspace already owns a private per-document transcript, so use the private scene capture supported by the shared composer and exercised in Research Workspace.

Mounting the ordinary Chat history provider here would introduce a second owner for the workspace's transcript. Allowing an unverified send would weaken the shared guard. A private scene keeps scene reads scoped to the account and document while preserving the workspace's existing transcript and retrieval options.

A send may capture the scene once the document session and account are ready. Switching documents, clearing chat, leaving the workspace, or changing accounts invalidates pending work. Reuse the account authorization already held by the workspace. Existing document chats still restore from the browser, and completed turns retain the document index used to reopen their transcript. A public regression covers this behavior and the same production-source change is backported to the hosted pin.

ADR required: no. This follows [ADR-049](../../ADR/049-chat-history-selection-ownership.md) and the existing private scene path without changing a protocol or storage contract.
