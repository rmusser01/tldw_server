# Document Workspace chat repair implementation plan

**Goal** Restore sending from Document Workspace with verified document and account context.

**Spec** [Document Workspace chat send repair](../specs/2026-11-19-document-workspace-chat-design.md).

**ADR check** ADR required: no. Reuse the existing owner-validation rule in [ADR-049](../../ADR/049-chat-history-selection-ownership.md).

1. Extend the Document Workspace ownership tests to read the real scene and assert that the originating document and account remain current. Test a send while an existing transcript is being restored, and verify that a switch, clear, sign-in change or unmount revokes the old capture.
2. In `useDocumentChat.ts`, reuse the workspace's verified account operation with `captureActorSettingsTarget` and `captureActorSceneSendForTarget`. Persist the document key in the existing owner-scoped history transaction. Make a capture available after the current document session is applied. Pass it through `DocumentChat.tsx` to `useMessageOption`, and enable Send and suggested questions once ready.
3. Run the ownership, Document Workspace and shared send tests. Review and publish to public Dev. Backport the scoped WebUI fix to hosted, qualify the immutable WebUI, and verify the private beta path with a regular test account and synthetic document.
