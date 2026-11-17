# Header account menu

Add a no-photo current-account popup at the existing notification-control position for a resolved multi-user JWT session. Anonymous, logged-out, single-user and rehydrating configurations keep their existing notification control. Shared `Header` and `ChatHeader` compose the same behavior for the WebUI and extension.

On `dev`, the conversation header and the other shared header layout both render the same account control. The existing composer already keeps the model picker, attachments and Send available with Tools closed on phone and desktop. It continues to honor the saved disclosure preference.

The popup loads username and email through the existing `tldwAuth.getCurrentUser()` service when opened. It does not use cached user data. Existing `watchChatAccountChanges` invalidates pending identity responses, clears identity and closes the popup when credentials, principal or server authority changes. Navigation and sign out remain available if identity loading fails.

Actions:

- **Account settings** opens `/settings/tldw`, the existing working account/connection surface. `/profile` and `/account` are placeholders in the open-source client, so this change does not present them as a functional profile editor.
- **Settings** opens `/settings`.
- **Notifications** retains the caller's existing destination, unread count, lifecycle messages and retry action where supported.
- **Sign out** uses `tldwAuth.logout()`, disables duplicate submissions, and displays failures without removing navigation. It preserves existing auth-gate routing and does not add remote-revocation guarantees.

The popup remains a nonmodal dialog. Native keyboard activation opens it, the first action receives focus, Escape restores focus to the trigger, and outside pointer interaction or tabbing away dismisses it. Width and height stay within the viewport; rows retain 44-pixel touch targets and existing design tokens.

## Implementation plan

1. Extend real header keyboard and notification regressions before changing the popup shell.
2. Add the shared menu and gate it with `useCanonicalConnectionConfig`; test identity failure, stale responses, sign out and anonymous/single-user behavior.
3. Run focused header/composer tests, WebUI typecheck and touched-file lint; obtain independent review before updating the draft PR.

ADR required: **no**. This UI composes existing authentication, authority watchers, routes and persistence without changing their contracts.

This scope adds no profile editor, authentication protocol, cookie-session implementation, storage key or dependency.
