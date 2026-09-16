# UAT068 recurrence / existing TASK13260.15 — saved Character URL wins over explicit replacement

## Outcome

**Confirmed interacting route/selection defect; keep the existing task, no duplicate.** An explicit tracked Character replacement detaches the old saved chat in state, but the still-current saved Character URL reasserts its old chatId. The server loader then restores the old character/transcript. A private actual-picker/loader probe reproduces this with the supported canonical saved URL; the existing one-time Settings handoff URL control passes.

No wrong-identity inference was sent in the native observation or this diagnosis. No repository/source/test files, browser state or running servers were changed.

## Native evidence and click/confirmation qualification

- `/private/tmp/cycle5-multi-native-character-replacement-before.txt`: saved TestBot4, route `/chat?mode=character&characterId=4&chatId=edaa65df-dd08-4668-a4cd-affadd033995`, original question and BEEP BOOP reply.
- `...-selector.txt`: visible **Default Assistant** button `f7e467` alongside the active TestBot; it is a real character choice, not the favorite-star control.
- `...-result.txt` and `...-settled.txt`: dropdown closes, but route, header, context and transcript settle back to TestBot4.

The four snapshots do not embed a click receipt, but the subsequently retained `/private/tmp/cycle5-multi-native-replacement-receipt.txt` **does verify a second exact native click**. Its Playwright code resolves the exact button name `Default Assistant`, reads its label, awaits `button.click()`, waits two seconds, and captures settled state. `clickedAt` is 08:55:05.145Z (recorded immediately before label lookup/click), `observedAt` is 08:55:07.293Z. The clicked label is `Default Assistant\nCharacter`; the URL still has character4/old chatId and the active mode remains `Character Chat\nCycle5 Alice TestBot`, with `dialogs: []`. The receipt's `selected` text is empty, so that field alone cannot prove the selected label; the active mode, URL and other settled snapshots supply the result evidence. Default Assistant's numeric server ID remains pending a normal catalog read and is not inferred from this receipt. The recurrence is now action-backed native evidence as well as an independent actual-component reproduction.

No confirmation/unsaved dialog appears in these snapshots. The production composer uses `ComposerToolbar.tsx:560` → **AssistantSelect**, whose tracked-selection handler directly detaches a different saved target. It has no switch confirmation on this path. The separate legacy **CharacterSelect** does have a “Switch character?” confirmation (`:787–825`), and its Cancel semantics must be retained. The native composer appears empty; this diagnosis does not establish a lost unsaved draft or an incorrectly rejected confirmation.

## Causal boundary

Paths relative to `/Users/macbook-dev/Documents/GitHub/tldw_server2`:

1. `apps/packages/ui/src/components/Common/AssistantSelect.tsx:546–563`: the real choice produces tracked nextEntry; for a different character with serverChatId, `clearActiveServerChat()` clears the current chat/history/metadata before awaiting canonical `setSelectedAssistant(nextEntry)`. No route update retires the old target.
2. `apps/packages/ui/src/components/Option/Playground/Playground.tsx:963–971`: whenever current serverChatId differs from routeCharacterIntentChatId, the effect assigns the URL's old chat ID again. It depends on serverChatId, so explicit detachment retriggers it.
3. Same file `:973–976`: fresh-character application is skipped when the route already carries chatId. Consequently saved-route entry does not establish the fresh-character applied marker used by the later synchronization guard.
4. Same file `:1093–1097`: route synchronization returns until that character marker is applied; it cannot retire the stale saved chatId in response to the manual replacement.
5. Existing server loader correctly treats the restored saved target as authoritative and restores its metadata/messages; this also replaces the draft selection. UAT123's saved-metadata workflow derivation then correctly displays the restored old Character mode. Removing that authority rule would hide the symptom and risk ordinary-chat/backlink behavior.

The old greeting is not the writer proven here. UAT113 made the saved Character URL a normal reload destination, exposing a gap absent from the old one-time handoff picker test. No historical pre113 run was performed, so this report identifies the interaction without claiming a bisected introducing commit.

## Independent private reproduction

Artifacts:

- `/private/tmp/cycle5-uat068-saved-route-picker-probe.config.ts`
- `/private/tmp/cycle5-uat068-saved-route-picker-probe.log`
- `/private/tmp/cycle5-uat068-source-hashes.txt`

The temporary Vite transform reuses `Playground.coordinator.integration.test.tsx`'s real picker test. It mounts actual Playground, AssistantSelect, useSelectedAssistant, WebUI storage and production server loader; the existing controlled backend/authority, router shim and presentation boundaries remain. Six real loaders reproduce the existing app-owner fixture. The test waits for a fully loaded old chat/profile, clicks the real current-character button and then the real replacement button, and observes settled route/store/storage.

From `apps/packages/ui`:

```sh
bun run test --config /private/tmp/cycle5-uat068-saved-route-picker-probe.config.ts
```

**1 passed, 1 failed, 47 deselected**, 5.91s. Four soft expectations fail within the single regression case.

| Initial route | Settled after replacement click | Server reads |
| --- | --- | ---: |
| `/chat?settingsServerChatId=robot` control | Route `/chat`; selection7 New choice; serverChatId null; empty new draft | 1 getChat |
| `/chat?mode=character&characterId=5&chatId=robot` | Same old URL; selection5 Robot; serverChatId robot; original BEEP BOOP restored | 7 getChat |

Seven reads is this six-loader test fixture's count, not a native request-count claim. The real picker and loader run, but the backend is controlled and the broader composer is mocked. The test is causal evidence for selection → route → loader, not proof of live inference or actual canonical server writes.

## Minimum post-freeze correction

Treat an **explicit accepted tracked replacement** as a route transition as well as a canonical selection change. Retire the old saved chatId before an intermediate clear can cause the URL effect to restore it; publish the requested new Character route/intent and new draft through the existing navigation/selection owner. Use the existing selection callback boundary if necessary to connect the composer picker to Playground's route owner. Keep route params unrelated to chat identity, supported extension hash routing and current replace-history semantics.

Do not merely remove the saved-chat authority check, ignore every chatId URL, or respond to arbitrary shared-preference/storage changes as if they were a deliberate replacement. A raw cross-tab preference must not detach an owned saved conversation. Likewise, only marking the old route “applied” is insufficient if the intermediate old selection can rewrite a fresh old-character URL before the new selection commits. The transition needs the **explicit requested identity**, not whichever draft value a later effect happens to see.

Expected bounded production ownership: Playground route owner and, if required, the narrow actual ComposerToolbar/AssistantSelect callback connection. Keep shared assistant storage, server loader merge rules, greeting code, backend, inference and DB schema unchanged unless a permanent regression proves another necessary edit. No existing saved server transcript should be deleted or rewritten; replacement begins the intended fresh conversation while the original remains recoverable.

## Permanent regressions and no-loss controls

1. Retain this actual picker + actual route + real loader regression with a fully loaded saved URL, both search and extension hash forms. Assert selected identity, header/context, current route, detached serverChatId and intended fresh draft agree; no stale old transcript or old-target mutation returns.
2. Same-character selection remains on the saved conversation. A different explicit selection creates only the intended draft, and subsequent send uses the new character ID through controlled transport; no inference is needed to assert payload. Preserve original saved messages in DB and verify selecting/backlinking the original still restores them.
3. Hold old profile/messages/reply/mirror completion across replacement. Old completions cannot republish identity/transcript or overwrite the new route/draft. Test A→B→A selection generations, account/server changes, unmount and late route hydration; preserve existing owner/abort checks.
4. Add an unsynced draft/active stream case and the existing legacy confirmation controls. Cancel leaves route, conversation, selection and draft untouched; accepted replacement follows the existing protected action contract. Do not implement a fix by eagerly clearing before a required confirmation or by losing unsynced rows. The current native evidence did not exercise a dirty composer, so this remains a required negative control.
5. Preserve UAT113 creation → acknowledged save → canonical route → reload/remount (one saved transcript, no duplicate fresh chat), including explicit new same/different Character entry. Preserve UAT123 ordinary saved History/Note navigation clearing stale Character mode/gating while loaded Character/persona chats retain their own mode.
6. Preserve raw cross-tab selectedAssistant preference updates: they must not replace the active saved conversation. Preserve fresh-entry route priority and late principal/restore guards. Existing picker tests use the consumed settings handoff URL, while cockpit route tests cover fresh entry without saved chatId; neither tested this combination.
7. Run coordinator, actual AssistantSelect/CharacterSelect, canonical selection/loader, session persistence and relevant Notes backlink tests; scoped lint/type baseline and independent review. After freeze, repeat the retained exact native saved TestBot → Default Assistant click and capture settled route/header/context and intended new-chat behavior, resolving the target's real numeric ID from the catalog. No wrong-identity generation is necessary to demonstrate the original defect.

## Evidence limits and hashes

This is RED diagnosis and design only; no correction or GREEN claim. The native fresh Character entry, BEEP BOOP completion/version2 and reload remain useful passing controls, and this recurrence reopens only the replacement acceptance under existing task15. No duplicate UAT131 is needed.

Five SHA256 records were verified unchanged. Playground source: `5966f5ec847afb087f3448d6cb50b4d3e4035ba7e6c8f7da85607419425f104d`; full paths/hashes in `/private/tmp/cycle5-uat068-source-hashes.txt`. No full typecheck, browser access or inference was run.
