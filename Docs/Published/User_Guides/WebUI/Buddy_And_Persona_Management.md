# Buddy And Persona Management

A Buddy is a visible companion for an existing conversation or workspace. Its artwork is independent of a Persona: you can use a Buddy with no Persona, and choosing artwork does not change a conversation's identity or behavior. Deleting the source Persona does not delete artwork already copied into an independent Buddy.

## Choose A Buddy And Target

1. Open **Buddy & Persona** from the chat composer, a workspace header, or an active workspace's list entry. The current conversation or workspace is selected when available.
2. Choose from **Your Buddies**, or expand **Choose a ready-made Buddy** to preview artwork before selecting it. No Persona setup is required for ready-made artwork.
3. Choose **One conversation** or **Workspace**. For a conversation, choose its location and named conversation. Workspace mode lets you select an individual conversation when you reply.
4. Select **Apply** to save, or **Cancel** to discard unapplied selections. Apply becomes available after the choices and target have loaded.

If Apply reports an error, read the message before retrying: a new Buddy or workspace default may already have been saved even if attachment failed. Cancel closes the editor; it does not undo saves already confirmed by the server.

The composer entry shows the conversation's current Persona and offers **Edit conversation Persona & behavior**. Finish or cancel staged Buddy changes before opening those settings. Artwork selection itself keeps the existing conversation identity.

For a workspace, **Default Persona for new conversations** applies only when a new conversation has no explicit assistant choice. Explicit **None** and other explicit choices win. Existing conversations keep their identity. Changing the default here uses read-only Persona memory; advanced memory choices remain in workspace settings.

## Read And Reply Without Leaving Your Page

One active Buddy stays in the application shell while you move between pages. Its label identifies the attached conversation or workspace; navigation does not silently switch its target. The saved attachment is restored from the server when you return. Unsaved reply drafts and speech controls belong to the current application session.

Select the Buddy to open its interaction panel. Read the conversation history and check **Reply to [conversation name]** before pressing **Send**. In workspace mode, choose a conversation first, or select a named **New response** entry. **Mark read** acknowledges that exact result; it cannot clear a newer response. Conversations with the same title show their creation time first; a short identifier distinguishes missing or matching timestamps. The same label appears in the picker, results, reply context, and speech introductions. Saved titles stay unchanged.

Replies use the conversation's configured provider and model, shown above Send after the target is checked. An explicit model selection in an ordinary workspace Chat is saved for later Buddy replies. If either setting is missing, **Reply model settings** opens with the required fields and Send stays unavailable until both are supplied. Your draft stays in place. You can also set optional overrides for one Buddy reply; those overrides do not replace the conversation's saved defaults. Queued and Working describe accepted Buddy replies. Approvals remain in the originating workflow's existing controls.

## Stop And Detach Mean Different Things

**Stop** beside a queued or working reply asks the server to stop that turn. A reply already saved remains in the conversation. Closing the panel, changing pages, or choosing **Detach Buddy** removes no accepted work: the server continues to own it. To stop before detaching, use Stop first.

This continuity depends on the running server process. A server restart does not replay unfinished replies. If a turn reports failure or interruption, inspect its conversation before intentionally retrying; an earlier message or effect may already have been saved.

If a target is deleted or becomes unavailable, the attachment needs attention. Open management and select an available target; a stale saved selection does not grant access to another user's conversations.

## Motion And Position

Open **Buddy options** and choose **Expressions → Static** or **Dynamic**. Your system's reduced-motion preference keeps expressions static even when Dynamic is selected.

Drag the movement handle below the Buddy with your pointer, or focus it and use the arrow keys. Hold **Shift** for larger keyboard steps. **Home**, or **Reset position** in Buddy options, returns it to the upper-right starting position below the navigation, clear of the bottom composer. Previously saved placements are preserved. The position stays within the application viewport.

## Optional Speech And Dictation

**Read new responses aloud** is off until you enable it. Newly detected responses join a single speech queue and are introduced by their conversation name. Use **Pause speech**, **Resume speech**, or **Skip** to control playback. Enabling speech does not replay the initial history or mark results read. Playback uses the application's configured speech service.

**Dictate a reply** is available only for a Buddy attached to one conversation. Start dictation explicitly, select **Finish dictation**, review or edit the transcript, then press **Send** yourself. Dictation does not send automatically. Starting capture pauses speech playback; changing the selected target or closing the panel stops capture. Workspace mode has no microphone input.

## Persona Live Is A Separate Workflow

The full Persona Live session remains available in Persona Garden for its existing conversation and voice controls. It retains its connection-owned behavior: a Persona Live disconnect can cancel its active work. The navigation continuity described above applies to accepted Buddy replies, not to legacy Persona Live sessions.

## Related Guides

- [Persona user guide](../Server/Personas_User_Guide.md)
- [Chat, characters, and assistants](Chat_Characters_Assistants.md)
- [Speech setup](../WebUI_Extension/Getting-Started-STT_and_TTS.md)
- [Independent Buddy API](https://github.com/rmusser01/tldw_server/blob/dev/Docs/API/Buddies.md)
- [Accepted Buddy turns: operation, failures, and restart](https://github.com/rmusser01/tldw_server/blob/dev/Docs/Operations/Buddy_Turns.md)


## Downloaded Buddy packs and credits

Downloaded `.tldw-persona-vpack` files currently use the Persona import API;
the Buddy manager offers existing Buddies and ready-made artwork, without a
file-upload control. After importing and reviewing a pack, create an independent
copy through the [Buddy API](https://github.com/rmusser01/tldw_server/blob/dev/Docs/API/Buddies.md), leave the optional Persona
unset if desired, then select the new Buddy and apply its target in management.

Credited packs keep their embedded creator, license, source URL and notices
through new imports, independent copies and native exports. Older imports may
have lost those fields; recover them by re-importing the original download and
making a new Buddy copy. Keep the accompanying license files when the original
archive did not embed notices. Import jobs currently require SQLite storage.
