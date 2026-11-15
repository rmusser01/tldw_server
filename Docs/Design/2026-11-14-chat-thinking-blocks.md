# View and edit chat thinking blocks

Use the existing disclosure on assistant replies and show **Edit thinking** inside each expanded block. One block editor per reply uses the message editor's acknowledgement and stale-source protection. Save replaces the selected text span and preserves the original delimiters and every byte outside the span.

Persist provider-supplied reasoning text in native streamed replies using the same tagged representation already used by client-managed chat. The WebUI negotiates support when it captures native history and requests retention on its completion. The answer and reasoning remain separate in the SSE response. Native prompt history uses the visible answer, while reload returns the full saved message for rendering and editing. When the combined reply exceeds a server message limit or thinking fails the save policy, retain the approved answer.

ADR required: no. Existing tagged content, chat history ownership ([ADR-049](../ADR/049-chat-history-selection-ownership.md)), and scoped message edits already cover the change.
