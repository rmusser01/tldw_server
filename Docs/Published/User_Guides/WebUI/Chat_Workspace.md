# Chat Workspace

Chat Workspace (`/chat-workspace`) is a focused conversation view for the active workspace. It combines workspace sources, a message composer with explicit context staging, and a read-only inspector. It is available in the WebUI and extension options, not the compact extension sidepanel.

## Choose The Right Page

| Page | Use it for | Difference from Chat Workspace |
| --- | --- | --- |
| Chat (`/chat`) | General conversations, model and assistant selection, history, and the full chat controls. | The place to choose a model or explicit assistant before using the focused workspace view. |
| Chat Workspace (`/chat-workspace`) | Conversation in the active workspace, with sources staged for the next send. | Shows source staging and runtime information; it has no workspace picker, general model picker, or persona editor. A failed-turn recovery action can select another model for that retry. |
| Knowledge QA (`/knowledge`) | Questions over selected knowledge sources, with retrieval results and cited answers. | A dedicated search-and-answer workflow, not this workspace-scoped chat composer. |
| Research Workspace (`/research-workspace`) | Select a workspace, organize and inspect sources, use its research chat, and create Studio outputs. | The broader research environment. Its chat pane is separate from the Chat Workspace page. |
| Document Workspace (`/document-workspace`) | Read and analyze a document with document-centered chat. | Focuses on the document being read rather than a staged set of workspace sources. |

See [Chat, characters, and assistants](Chat_Characters_Assistants.md), [Knowledge, media, and sources](Knowledge_Media_Sources.md), and the [Knowledge QA guide](../WebUI_Extension/Knowledge_QA_Guide.md) for those workflows.

## Choose Workspace, Model, And Persona

Set these up before composing an unsent message in Chat Workspace. Leaving the page to change settings can discard that message and its staged sources.

### Workspace

1. Open Research Workspace and select or create the workspace there. Confirm the workspace name in its header.
2. Add or organize the sources you need in Research Workspace.
3. Open Chat Workspace and confirm the same name in the Sources pane and the inspector's **Scope** section.

Chat Workspace reads the active workspace from the shared workspace store. After successful hydration, it initializes an empty local workspace only when no active workspace identity exists. It does not select an existing server workspace for you. Local-only state shows **Workspace not connected** with **Open workspaces**; create or choose a server workspace there and use **Open**. Before mounting the chat, the existing activation guard verifies the current account and loads the canonical workspace, sources, artifacts, and native notes. A non-empty local identity alone is not sufficient to send. A storage-loading failure shows **Retry workspace recovery** and does not initialize over the stored data.

**Workspaces manager:** **Open** on a Research Workspace uses the canonical `?workspace=<id>` route. The route verifies the current account and loads the target workspace, sources, artifacts, and native notes before activating it. An unavailable, unauthorized, or incompletely loaded target is not silently activated. Other workspace profiles can use a provenance link rather than Research Workspace activation. Confirm the intended workspace name after opening it.

### Model

Choose a model using the model control in Chat (`/chat`), then return to Chat Workspace. Its inspector displays the shared chat model selection: the current in-memory selection takes precedence, with the saved browser selection as a fallback. Chat Workspace does not apply a separate workspace model default.

**No model selected** and **Select a model** are status messages, not clickable pickers. Choose a configured, usable model before sending. A visible Send button or a ready connection status does not establish that a model is configured.

The **Switch model** failed-turn recovery action is the exception: it opens a picker for that retry. It does not replace the shared selected-model value shown in the inspector.

### Persona And Inheritance

A persona is optional. To choose an explicit assistant, use the assistant selection controls in Chat; manage persona profiles on `/persona`. To configure a workspace fallback, use **Default assistant** in Research Workspace's workspace menu. That dialog selects a Persona and its **Memory access**, with **Read only** as the default. **Read and write** requires the dialog's explicit memory-write confirmation and permission to save the setting.

Workspace inheritance applies to a new, empty conversation when no assistant is explicitly selected and no existing chat identity or history is present. An explicit assistant selection takes precedence over this fallback. An existing chat's tracked assistant or saved assistant overlay is not simply replaced by changing the workspace default. Inheriting a Persona does not change the global assistant selection.

Check **Model / Persona** in the inspector for the effective state:

- **Inherited from workspace** identifies the workspace Persona in use.
- **Explicit persona** identifies an explicit or restored chat assistant rather than a newly applied workspace default.
- **No persona selected** means a persona is not in use; ordinary model chat remains possible.
- **Workspace default unavailable** can include a reason such as permission denied, deleted or unavailable Persona, disabled Persona support, an invalid default, or an unsupported assistant kind. It does not mean that Persona was applied. Repair or clear the default in Research Workspace, or choose an explicit assistant in Chat.

There is no persona selector, memory-access editor, or model editor in Chat Workspace's inspector.

## Browse And Stage Sources

The Sources pane lists sources from the active workspace. **Filter sources** filters their titles; it does not retrieve document passages or add context to a message.

- **Browse** opens the canonical bounded read-only preview and marks the source as **Browsing**. The preview can show content, snippets, loading, unavailable, or error states; long content is explicitly truncated. It is a modal, not a full source-content inspector. Browsing does not stage a source, insert its content, send a message, or clear other staged sources.
- **Stage** adds a ready source to the local **Context staged - not sent** list. Browsing one source does not remove other staged sources.
- **Unstage** in Sources or **Remove** in the staging card removes only that staged entry. **Clear staged context** empties staging without deleting sources or clearing the typed message.
- **Add source** opens Research Workspace's Sources pane. **Open library** opens `/media`. Use those pages to add or inspect material; Chat Workspace itself is not an ingestion or source-processing editor.

Staging the same source again does not create a duplicate entry. Each entry shows its title, workspace scope, type, availability, and any status message. The inspector lists the staged sources, not everything in the workspace and not just the source marked Browsing.

### Processing And Source Errors

**Stage** is disabled for sources marked **processing**, **error**, or **unavailable**. **Browse** remains available. Read the source's status message and return to Research Workspace or the media library to inspect or resolve ingestion problems; there is no source-processing retry button here.

The pane distinguishes **Loading workspace sources**, **Refreshing workspace sources**, **No workspace sources yet**, and **No sources match the filter**. A source-loading error is shown even when previously loaded sources remain visible. Chat Workspace displays the shared store's source state; it is not a separate source-status polling or repair screen.

Staged entries capture source information when staged. They are not a live source-content snapshot or a promise of current backend availability. After correcting a source problem, check its current state and remove/re-stage it as needed.

## What A Send Includes

Staging is separate from both browsing and the text in the composer. Workspace membership alone does not send every source to the model.

| Staged sources at send time | How this page submits them |
| --- | --- |
| Ready sources with valid positive integer media IDs | Sends unique media IDs as structured retrieval context (`ragMediaIds`), enables file retrieval, and requests RAG mode. This references media for retrieval; it does not paste each source's full text into the message. |
| Sources without a usable media ID, or entries not marked ready | Cannot carry those entries as retrieval IDs. Uses a formatted source-list text fallback, not their contents. |
| A mix of usable media IDs and other entries | Sends the ready media IDs and appends the formatted list of all staged sources to the typed message so the remaining entries are not silently omitted from the text. Only the usable IDs supply structured retrieval context. |
| No staged sources | Sends the typed message in normal mode, with no staged media IDs and file retrieval disabled by this page. |

If you send staged sources without typing a question, the formatted source list becomes the message. Ready media IDs still accompany it where available. Writing a question makes the intended task clearer.

### Insert Context Summary Is Text Only

**Insert context summary** appends a `Context sources:` list to the composer and clears staging. Each row contains the source title, type, workspace scope, and a non-ready status when applicable. It is not an AI-generated summary and contains no source passages, transcript, or full document content.

After insertion, the list is ordinary editable draft text. Sending that text alone does not carry the previously staged media IDs. Re-stage ready sources if you also want structured retrieval context. Insertion does not send anything to the server.

## Send, Stop, And Retry

1. Check the workspace name, selected model, connection state, and staged list.
2. Type your question. Use **Send message**, **Send with staged context**, or **Ctrl+Enter** (**Cmd+Enter** on macOS). Both send buttons use the same draft and staged context. Plain Enter adds a newline.
3. During a request, sending again is disabled. While the response is streaming, **Stop generating** stops the active request. Stopping is not an undo or a deletion of the conversation; review any response already received before retrying.

Sending is also disabled while the server is unavailable, while workspace identity is loading, or when there is neither a typed message nor staged context. The composer remains editable while disconnected, but there is no offline send queue or automatic resend on reconnect.

When the send operation reports success, the draft and staging are cleared. A failed or skipped send preserves them while this page stays open in the same workspace. On failure, the composer displays an error and the inspector/status strip reports **Send failed**. Resolve the reported connection, model, authorization, or request problem. **Retry same model** and **Switch model** retry the captured failed turn: after durable admission they retain the same saved user turn and create a new assistant attempt. Typing and sending a new message is a separate turn. Inspect the transcript if an earlier attempt received a partial response.

## Local State And Lifetime

| State | Lifetime in this view |
| --- | --- |
| Typed draft and inserted source-list text | Local to the mounted chat panel. Preserved on failed/skipped sends and disconnection; cleared on successful send, workspace change, page unmount, or reload. Not an autosaved draft. |
| Staged sources | Local to the mounted Chat Workspace page and current workspace identity. Preserved on failed/skipped sends and disconnection; cleared by successful send, Insert context summary, Clear staged context, workspace change, page unmount, or reload. |
| Browsing marker | Local to this page and workspace identity. Browsing also updates the shared source-focus target, but does not create durable staged context. |
| Workspace identity, sources, and conversation history | Managed separately by the shared workspace/chat stores and their persistence. Saved workspace or chat state does not restore this page's unsent draft or staging. |

Switching the narrow-screen **Chat**, **Sources**, and **Inspector** tabs only hides or shows panes; it does not unmount the chat panel or clear the draft and staging. Navigating to another page is different: do not rely on returning to restore unsent work.

## Inspector, Status, And Narrow Screens

On wide screens, Sources, Chat, and Inspector appear side by side. On narrower screens, use the **Chat**, **Sources**, and **Inspector** tabs above the panes. Start on Chat, switch to Sources to stage material, then return to Chat to review the staging card and send. The controls are reachable with Tab and have visible keyboard focus; long lists and messages scroll within their panes.

The read-only inspector shows **Scope**, staged **Sources**, **Model / Persona**, and **Runtime**. The bottom status strip summarizes connection/runtime state, staged context, and missing model or persona selection. Its recovery labels are guidance, not buttons. Runtime prioritizes server unavailability, then loading workspace identity, then send failure, then streaming or ready state. A ready runtime does not certify that a model or usable source context is selected.

There is no tool-approval queue, approve/reject control, runtime configuration editor, or source-content inspector in these rails. The **Study** section currently reports **No generated study set**; use the broader research/study pages for those workflows.

## In-App Help

In both the WebUI and extension options, open **Quick Chat Helper > Browse Guides** for the **Chat with staged workspace sources** workflow guide. From Chat Workspace, use **Tutorials for this page** there to start **Chat Workspace Basics**.

The extension also has Page Help. In the WebUI, `?` opens keyboard-shortcut help, not the tutorial picker; use Quick Chat Helper for the guide and tour.

## Related Guides

- [Chat, characters, and assistants](Chat_Characters_Assistants.md)
- [Buddy and Persona management](Buddy_And_Persona_Management.md)
- [Knowledge, media, and sources](Knowledge_Media_Sources.md)
- [Knowledge QA](../WebUI_Extension/Knowledge_QA_Guide.md)
- [Start, account, and settings](Start_Account_Settings.md)
- [Page and feature index](Page_Feature_Index.md)
