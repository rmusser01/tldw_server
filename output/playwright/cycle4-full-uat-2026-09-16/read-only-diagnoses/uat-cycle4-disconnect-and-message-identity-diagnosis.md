# Cycle4 final read-only observations: Disconnect and Chat identities

## Manual-key Disconnect classification

**Classify the captures as disconnected single-user local-cache visibility, not a demonstrated cross-account leak or a recurrence of the multi-user UAT048 masking repair.** This matches the current single-user connection-only behavior. The existing UAT045/047 requirements are credential removal and cessation of private polling; neither states that Disconnect deletes or locks locally retained Chat history. UAT048's protected-page unmount contract and tests concern a multi-user logout/expired-session boundary. No account change or unauthorized request after Disconnect was exercised in these captures.

That is a contract-limited conclusion: the implementation does not mask already rendered server-linked Chat rows after single-user manual-key Disconnect. A product requirement that Disconnect lock those local copies would require an explicit extension to the current contract and a focused regression; it should not be claimed as already implemented. Preserving local data on disk and hiding it while disconnected are separate choices. Do not erase cached drafts/history merely to change this presentation.

### Evidence and source

- `/private/tmp/uat-cycle4-single-offline-disconnected.txt`: after 1 second, the key is cleared, Media displays a credentials gate, and the old Chat tab shows seven cached rows plus Offline. The browser context is offline.
- `/private/tmp/uat-cycle4-single-online-disconnected-settled.txt`: after 4 seconds online, the same connection-only outcome holds; Chat has eleven cached rows and Offline, Settings has the empty key form, and Media asks for credentials. Therefore this is not only a slow network check or one-second transient.
- Task13260.5 AC3 and tracker UAT045 explicitly require forgetting the active manual key; UAT047 requires stopping private polling until verified reconnection. The tracker UAT048 concerns multi-user offline logout, retained queued drafts, signed-out page, and Alice/Bob isolation.
- `services/tldw/TldwAuth.ts:220-234`: manual single-user logout clears manual credentials and emits a logout boundary. It does not delete Chat storage.
- `services/tldw/single-user-credential.ts:456-505`: removes manual/session credential fields while preserving the server connection configuration. No Chat transcript mutation occurs.
- `store/connection.tsx:1357-1363` invalidates verified connection authority on the logout event; the composer checks readiness (`PlaygroundForm.tsx:819,4049-4055`), matching the visible Offline state. These captures alone do not certify every private polling/action path.
- `apps/tldw-frontend/pages/_app.tsx:225-229` considers single-user authentication false without a key/cookie. At `291-299`, however, `requiresLogin` is explicitly limited to unauthenticated **multi-user** configuration. Only `requiresLogin` triggers the protected-page unmount at `479-491`. `hideShellNav` still hides single-user shell controls at `426`.
- The existing app tests reflect that distinction: missing/manual single-user credentials test hidden shell controls (`app-layout.test.tsx:666,780`), while the offline private-content removal tests explicitly use multi-user Alice configuration (`989-1028`). No single-user transcript-lock expectation is asserted.
- `usePlaygroundSessionPersistence.tsx:151-174` refreshes the scope key without clearing the active message store. `useLoadLocalConversation.ts:53-60` invalidates pending loads on principal events, rather than removing already rendered rows. This explains why local continuity remains despite request authority being invalidated.

**Recommendation for the current matrix:** retain the exact observation and the connection/re-entry results. Do not call same-user local cache an account-isolation failure, and do not overstate it as proof of cross-account-safe Chat state. No code change is justified solely by the old045/047/048 acceptance wording.

## UAT103: tracked character user acknowledgment also needs coverage

The three-row server/four-row UI observation extends the mirror defect beyond normal Chat. There is a concrete tracked-path identity loss:

1. `hooks/chat/useChatActions.ts:2216-2245` explicitly saves the tracked user row, captures `createdUser.id` in `persistedUserServerMessageId`, and updates the in-memory row with its server ID.
2. The success call at `2692-2712` passes local `userMessageId` and `assistantMessageId` into `saveMessageOnSuccess`, but does not pass `persistedUserServerMessageId`.
3. `hooks/chat-helper/index.ts:475-515,558-582` supports an assistant server ID, while the user row persists only its local ID and lacks a user server acknowledgment field. Thus the mapping known during the live turn can be lost in the local mirror and later mistaken for genuinely unsynced content.

Task13260.44 should cover tracked greeting/user/assistant plus normal Chat, success/failure, mirror reload and actual subsequent model request. Preserve true unsent rows and the existing owned-history/authority guards.

The actual `/private/tmp/uat-cycle4-single-prompt-chat-request.json` contains equal **user** entries at zero-based indexes1/3 and4/6. These are the duplicated user turns. Assistant entries2/5 both contain the requested short phrase and are legitimate two-turn replies; equal assistant text is not evidence of duplicate persistence. Never repair this by globally deduplicating equal text.

## Saved-card provenance: no ID mismatch

The saved-card response in `/private/tmp/uat-cycle4-single-chat-card-saved.json` identifies:

- conversation: `ce67520e-883b-42ff-b565-c2872616c048`
- message: `pa_baec-8ac1-caa-6966`
- flashcard: `7f077a43-b4ba-487e-8dd9-edf1a4c4a3dc`

`/private/tmp/uat-cycle4-single-character-server-messages.json` explicitly contains that **same message ID** for the `Cycle4 Aster Guide` assistant row. The UUID ce67520e… is the conversation ID, not a competing message ID. A `pa_` prefix does not make an ID noncanonical once the server stores and returns it. `Message.tsx:1176-1180` sends `props.serverMessageId` for this operation. No additional provenance failure is established here.

## Limits

Read-only source/evidence inspection only. No browser, network/model, runtime, application, test-source, or configuration changes; no new runtime acceptance claimed. The card `.json` artifact includes a Playwright `### Result` wrapper; the structured result was parsed from that wrapper, not treated as raw JSON. A broad temporary-directory filename search was stopped and narrowed; no runtime process was interrupted.
