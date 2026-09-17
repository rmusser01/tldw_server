# SQLite multi-user Character mode audit

## Disposition

**Related residual of UAT123, with a different trigger; the post-send presentation mismatch is real.** The initially empty “Character Chat / Choose a character” state alone does not establish a defect: UAT123 deliberately preserves the global new-chat preference and fresh/unsaved Character behavior. After Bob’s actual ordinary exchange, retaining that header and setup panel misrepresents the conversation. Track that bounded fresh/account-switch transition if a new finding is needed; do not reopen UAT123’s already accepted History/Note scenario as though it failed again.

## Retained observation

- At 16:18:23.754Z the empty chat has Character Chat / Choose a character.
- At 16:18:49.956Z the captured ordinary `POST /api/v1/chat/completions` carries conversation `e24e6232-aae1-472e-98c5-3c29906ababf`, with no character ID.
- The `bobChat` snapshot retained in the 16:19:46.368Z receipt contains the BIRCH-913 user/assistant exchange, yet keeps Character Chat / Choose a character. Modes simultaneously shows Character / Scene Off. The outer receipt time is not a separately timestamped DOM event.

The parent’s scenario identifies Bob3 after account switch. These three inputs do not independently include a fresh identity response. No foreign content/character disclosure is demonstrated here.

## Frozen source explanation

All eight inspected source/history files match the SQLite-multi archive at `8f8774e6c868b304a96d95ab82e28389c129a78b`.

- `Playground.tsx:312,457` uses unscoped `playgroundChatWorkflowMode`; `501,918–954` also holds local Character intent. `839–852` prefers canonical saved-chat kind only when session readiness, ID, metadata and route-intent guards all pass; otherwise preference, local intent or explicit route can enable Character UI.
- `PlaygroundModeLauncher.tsx:117–125` says Off solely when `selectedCharacterName` is absent. `PlaygroundForm.tsx:4474` supplies the selected character name. It does not report the same workflow flag as the header.
- `Playground.tsx:3530–3553` blocks Send for a missing chat model, not a missing character. Thus a Choose-character panel can coexist with ordinary Send.
- `useChatActions.ts:3827–3865` takes the ordinary route without a selected character. Its new-chat path at `1663–1669` normally publishes non-character metadata and `serverChatMetaLoaded=true`. Therefore **the global preference alone is not a proven complete cause of the post-send state**: if all saved-workflow guards passed, UAT123 would override it. Route intent or unresolved/replaced metadata must be distinguished by a causal reproduction.
- Frozen task `13260.63:17–41` limits UAT123 to saved History/Note/cold restoration. `Playground.coordinator.integration.test.tsx:804–844` asserts both ordinary override and preserved Character preference/fresh/pending behavior. Those controls do not exercise this real account-switch→new ordinary-send sequence.

## Minimal follow-up

First capture/reproduce the actual route intent and session/metadata state across account switch and ordinary creation. Then make the displayed workflow follow the authoritative current conversation/send mode at that transition, preserving deliberate fresh Character entry and existing pending/unsaved safeguards. Do not blanket-reset unrelated preferences. A mounted real-hook regression should cover prior Character→new principal→ordinary create/send, with the existing UAT123 controls retained.

Read-only audit only: no tests, browser, runtime or database actions; no repair or native acceptance claim. Exact evidence/source hashes are in the companion JSON (SHA-256 `dee5e8370b51fba3cabf7815b789ed4d9ea7d577623dc2ecc96c3d9ee0ee6897`).
