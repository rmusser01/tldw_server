# Cycle5 New saved reset — UAT068 recurrence, not a new UAT123 failure

## Evidence conclusion

`/private/tmp/cycle5-single-native-237-ordinary-new.txt` records the actual header New saved chat click. The immediate result is `/chat`, empty transcript, Character setup with no selected character. `/private/tmp/cycle5-single-native-238-ordinary-settled.txt` then restores E2E-TestBot, original title and two saved rows (including BEEP BOOP), still at `/chat`. This is a real reset failure and belongs to the reopened UAT068 / TASK13260.15 route-transition work.

The separate sequence `/private/tmp/cycle5-single-native-244-roleplay-setup.txt`, `-245-roleplay-setup-state.txt`, `-246-explicit-clear-identity.txt`, `-247-clear-identity-settled.txt`, `-248-after-clear-new-saved.txt`, `-249-standard-ready.txt` clears identity through the real drawer, then starts a second new saved chat. The settled result is empty, title Chat, Character Chat / Choose character. That empty mode is consistent with the deliberately retained new-chat preference, not regression of UAT123's saved ordinary-conversation rule. The artifact filename “standard-ready” does not prove Standard mode.

## Causal source

- `apps/packages/ui/src/components/Layouts/Header.tsx:139`: New saved clears selected Character and invokes `clearChat`.
- `apps/packages/ui/src/hooks/chat/useClearChat.ts:97`: it calls `navigate('/chat')`, then immediately resets stores/serverChatId and clears session persistence. It does not wait for the route to commit.
- `apps/tldw-frontend/extension/shims/react-router-dom.tsx:158`: navigation schedules a React transition and asynchronous Next `router.push`; it returns immediately.
- `apps/packages/ui/src/components/Option/Playground/Playground.tsx:722,963`: cached current route intent still contains the old canonical chatId during that intermediate reset. The route owner reasserts that saved ID and the canonical loader restores its messages/identity. Once navigation settles to `/chat`, absence of route intent does not clear the already-reasserted chat.
- This is the same route-ownership conflict as the current-character picker recurrence. Unlike the picker, header reset does request navigation, but the request is not an atomic accepted transition with state invalidation.
- Playground lines828–837 deliberately use canonical metadata for an owned saved conversation and use route/local/new-chat preference when no saved target exists. Lines947–950 persist Character preference for a Character route. Neither `useClearChat` nor drawer Clear identity/Apply clears this preference. `/private/tmp/cycle4-uat123-mode-implementation.md` explicitly retains it for future new chats; coordinator tests include fresh/unsaved preference controls. Do not reclassify this intentional empty-state behavior as UAT123 based on the header reset failure.

## Private production-boundary proof

`/private/tmp/cycle5-new-saved-reset-probe.config.ts` transforms only the existing coordinator test in memory, mounting actual Playground, actual `useClearChat`, real selected-assistant storage and canonical loader with controlled transport. A private button performs the header reset sequence. The route seam commits either immediately or after70ms (models pending Next navigation; does not execute Next itself).

Command from `apps/packages/ui`: `bun run test --config /private/tmp/cycle5-new-saved-reset-probe.config.ts`.

Final `/private/tmp/cycle5-new-saved-reset-probe.log`: **1 RED / 1 GREEN**, no unhandled errors. Immediate route commit: `/chat`, null serverChatId, no rows/selection. Delayed route commit: `/chat`, `robot` serverChatId, BEEP BOOP row, Robot selection restored. Initial fixture run lacked model-settings reset; corrected only the private transform and retained that setup-error log separately. No product finding is based on that fixture exception.

## Post-freeze scope / controls

Include header New saved/temporary accepted reset in UAT068's deliberate route transition owner, retiring the old canonical intent/generation before state clearing can reactivate it. Preserve stored server history and existing navigation/dirty confirmation cancellation. Avoid global preference resets or weakening canonical loader authority.

Permanent controls: actual header→clear hook→Playground with delayed Next route commit, immediate route control, extension hash, late old profile/messages/reply/mirror, A→B→A and account/target changes, cancelled settings/unsaved-draft guard, old saved chat still retrievable, UAT113 saved Character reload, UAT123 ordinary History/Note routes. Preserve explicit Character new-chat preference unless a separate product decision changes it.

Read-only diagnosis: no repo/source/test/task/browser/runtime edits or inference. Source/test/probe hashes: `/private/tmp/cycle5-controller-additional-diagnosis-hashes.txt`.
