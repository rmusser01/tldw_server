# UAT113 / TASK13260.53 — character entry URL replays after save

## Finding

Confirmed route/session ownership failure. A newly saved character conversation retains its character-only creation URL. A normal reload treats that URL as a new character entry, clears the saved identity and skips session restoration. The stored server conversation remains recoverable through the ordinary sidebar.

No source or repository tests were changed. No browser, runtime, provider, or other network calls were made for this diagnosis. Task notes are the sole permitted repository write. All five inspected production/test files still match frozen product revision `7c9409`; see `/private/tmp/uat113-frozen-source-comparison.json`.

## Native evidence supplied by the cycle4 runner

- Conversation `bd564030-70fe-41bc-9c91-4b661408f8e3`, character 4 / Cycle4 Cedar Guide, real complete-v2 and message persistence both returned 200 (runner request IDs 1828/1838).
- `output/playwright/cycle4-full-uat-2026-09-16/multi/character-cedar-reloaded.txt`: normal reload remains `/chat?mode=character&characterId=4`; title is `Chat | tldw`; only the greeting remains, `Chat 1 message`.
- Runner-owned API transcript `/private/tmp/uat-cycle4-multi-character-messages.json` remains 200. No destructive server call is implicated.
- `output/playwright/cycle4-full-uat-2026-09-16/multi/character-history-reopened.txt`: actual Recent conversations → Server row `Cycle4 Cedar Guide Chat (20260916_010755)` restores the exact title, greeting, user, and answer (3 messages), with URL `/chat`. The answer includes the correct Project Cedar date/coordinator. This is a positive recovery control, not closure of reload UAT113.

## Causal code boundary

Paths below are relative to `/Users/macbook-dev/Documents/GitHub/tldw_server2`.

1. `apps/packages/ui/src/components/Option/Playground/Playground.tsx:715–735` derives an entry command from the current query/hash. Its applied/in-flight refs are component-lifetime only.
2. `Playground.tsx:959–978` treats any characterId without chatId as a fresh entry on mount. It cancels restoration, clears local history/messages and server identity, and calls `clearPersistedSession()`.
3. `apps/packages/ui/src/hooks/usePlaygroundSessionPersistence.tsx:717–725` actually cancels the save timer and clears the session; `apps/packages/ui/src/store/playground-session.tsx:93–98` resets its identity and advances the restore revision. The reset is not merely a rendered greeting overlay.
4. `Playground.tsx:1655–1656` returns before normal session restoration whenever this character-only command exists. The safe, scoped restore path at `usePlaygroundSessionPersistence.tsx:535–549` never gets the chance to restore the saved ID.
5. `Playground.tsx:1080–1084` leaves a matching selected character's character-only URL untouched. The synchronization effect has no acknowledged serverChatId dependency; its helper at `Playground.tsx:206–225` explicitly removes chatId aliases. Consequently, creating/saving the conversation does not consume or replace the original new-entry command.
6. Explicit saved chatId routes select their saved target instead (`Playground.tsx:949–956`, `1649–1653`), consistent with the native sidebar recovery at neutral `/chat`.

The failure does not justify globally skipping character entry resets: existing tests intentionally require an explicit character-only entry to start a new draft even over an active saved conversation. A consumed creation intent and a newly requested same-character draft need distinct ownership.

## Private interacting probe

Files:

- `/private/tmp/uat113-route-session.config.ts`
- `/private/tmp/uat113-route-session-cases.txt`
- `/private/tmp/uat113-route-session-red.log`

The Vite transform changes only the existing coordinator test fixture: removes its session-hook mock, supplies synthetic scoped configuration and adds private cases. Production files are untransformed. The mounted owner is actual Playground, with actual session hook/store, actual message-option Zustand state, and actual Next WebUI storage hook/shim. Unrelated views, transport, routing navigation and messages loader remain fixture mocks. A valid saved character session is seeded; this probe isolates the reload decision, and does not itself run a provider/completion or model the entire save acknowledgment.

Command, from `apps/packages/ui`:

```sh
./node_modules/.bin/vitest run --config /private/tmp/uat113-route-session.config.ts --testNamePattern UAT113 --maxWorkers=1 --no-file-parallelism
```

Result: **1 expected RED, 3 passing controls; 37 unrelated tests filtered**.

| Reload URL | Actual active ID | Actual persisted ID | Result |
| --- | --- | --- | --- |
| `/chat` | saved-cedar | saved-cedar | positive restore |
| `/chat?mode=character&chatId=saved-cedar&characterId=4` | saved-cedar | saved-cedar | positive canonical target |
| `/chat?mode=character&characterId=4` | null | null | RED for the native post-save reload state |
| `/chat?mode=character&characterId=5` | null | null | positive explicit replacement |

An initial fixture-only attempt left the original test's abbreviated store branch enabled, causing an undefined queuedMessages error. It is retained separately in `uat113-route-session-fixture-initial.log`, is not product evidence, and was resolved by enabling the real store fixture branch. The final log contains the actual assertion failure above.

## Bounded repair proposal and regression scope

Consume the original character-entry command when it is accepted, or promote it to the owned canonical saved target at a successful save acknowledgment. Use the router's replace path, including hash-route parity. Preserve workflow mode/character UI and other query parameters. Choose one explicit ownership transition; do not add a general restore exception based only on character-ID equality, and do not revive stale account sessions to solve it.

Before implementation, agree whether accepted entry normalizes immediately to neutral `/chat` (allowing existing scoped persistence) or saved acknowledgment replaces it with a canonical chatId URL. In either choice, a late completion from a superseded draft must not rewrite the current route. No backend or transcript-delete change is needed by the observed finding.

Required regressions:

1. Real character entry → owned first-save acknowledgment → URL consumption/canonicalization → unmount/remount with actual session/WebUI storage → same canonical identity and transcript. Do not seed the saved ID alone for the permanent end-to-end owner regression.
2. Ordinary `/chat` and explicit saved chatId restoration remain positive; unavailable saved target remains honest.
3. Explicit different-character and intentional same-character new draft still clear the prior target. Cancellation/newer picker/new route must suppress late former-save URL promotion.
4. Account/server changes and A→B→A restoration must remain scoped; keep current restoreRevision and request lifetime guards.
5. Query and extension hash variants preserve unrelated parameters. StrictMode must consume the entry once without discarding a successful owned save.
6. Native post-fix create/save/reload plus saved-sidebar reopen controls; current native recovery is only a positive control, not a repair acceptance.

Status remains To Do / implementation deferred during frozen cycle4. No compiler/lint/security result is claimed for this read-only diagnosis.
