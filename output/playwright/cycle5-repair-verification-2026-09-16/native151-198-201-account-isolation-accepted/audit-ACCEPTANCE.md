# Independent native acceptance — UAT151 / UAT198 / UAT201

## Disposition

| Finding / task | Verdict | Acceptance action |
| --- | --- | --- |
| UAT151 / TASK13260.90 | **Native acceptance supported after added literal Back capture** | AC3 can close with durable retention of both the earlier empty-catalogue return and the exact history Back evidence below. |
| UAT198 / TASK13260.136 | **Native acceptance supported** | AC3 can close with durable retention of this evidence and the already retained source/test review. |
| UAT201 / TASK13260.139 | **Native acceptance supported** | AC3 can close: Bob creates an independently owned deck with Alice's existing name through the ordinary UI. |

This is a bounded PostgreSQL multi-user acceptance audit, not a fresh full matrix, all-domain isolation audit, or raw-SQL/RLS certification. No task status, product file, browser, runtime, database, or git state was changed by this reviewer.

## Evidence handling and source boundary

I read the exact task criteria through the official CLI and the tracker entries, parsed each CLI JSON result between `### Result` and `### Ran`, independently inspected the safe observer code, and visually viewed `bob-cleared-draft.png`, `alice-manage-return.png`, and the added `bob-literal-history-back.png`. The last screenshot shows the cleared text field; the two deck selectors are verified from its matching settled accessibility snapshot. The observer records actual request/response bodies and only id/username from auth/me; it neither rewrites responses nor fabricates app state. No headers, login credentials, cookies, session files, or browser profiles are included in the audit packet.

`input-manifest.json` records exact SHA256/size for 30 allowlisted original inputs. `asserted-native-summary.json` records checked identities, timestamps, resource IDs, and outcomes. Original artifacts remain at their existing private paths for parent-owned durable retention.

The pre-start source receipt is at **05:29:59.276 UTC**, revision `a130b8e5507ff2f1ec6a930a7aa12183a44b5d44`. It includes reviewed ChaCha source SHA256 `6089c5e0cac45fd0dd2c5253ad906b20108746e35a9189bf3934fc1eca9506b7` and unchanged Flashcards endpoint SHA256 `7d6933c5ed7ff06e55e2ff335c48ccf5fb93d0ff209dd98a10e92a140f5c6e1c`. Health receipt reports200 at05:31:06.702. The source receipt covers backend files only. The parent identifies the same running recovery frontend18583; this audit does not claim a fresh frontend bundle/source manifest.

## UAT198 — original private-read failure is repaired

1. Alice identity2 is recorded at05:33:59.206/.436/.678. Her deck response at05:33:59.698 contains owner2 decks including deck1 named `Alice UAT151 Citrine Deck`.
2. After the ordinary account change, auth/me identifies Bob3 at05:37:55 and05:39:10. At **05:39:10.715**, the actual card list returns200 with `items:[]`, total0. At **05:39:10.716/.746**, the actual private deck catalogue returns200 with `[]`. The full500-limit Bob list at05:40:26.372 is also empty. This directly repeats the original backend catalogue failure with the opposite result.
3. Bob's ordinary card Create sends `deck_id:7`. The response at **05:45:01.454** returns UUID `3556231e-ebb5-43df-81d2-f1b07be1b493`, owner3, deck7, version1. After actual `page.reload()`, auth/me again reports Bob3; the full list at **05:45:39.773** contains exactly that one card. UUID, owner, deck, front/back, version, repetitions, timestamps, state, and deleted flag agree with creation. This is an owned manual-card save/reload control, not a generated-model-card claim.
4. After the normal switch back, auth/me reports Alice2 at05:51:10 and05:51:28. Catalogue responses return only decks **1/4/6**, all owner2. At **05:52:09.726**, the full list has five owner2 cards and excludes Bob's UUID. The corresponding screenshot visibly shows those five cards.

The original Citrine card `37b10bd7-edf4-4f35-83c1-1490115d8c55` is exactly equal as a parsed row between the05:33:59 first-page response and the final full response: version2, repetitions1. The final disposable card `f2a61712-8d3e-45df-bfb9-0527f86cda21` is version4/repetitions0. The other three final UUIDs and versions are recorded in the summary. I do not claim all five rows are byte-identical before/after: no pre-switch full-five response was required or reconstructed for that claim.

The native service-role qualification remains the accepted privileged/BYPASSRLS setup. The retained implementation independently verifies application persistence under both privileged and restricted roles, but it installs no Flashcards RLS policies. Arbitrary raw SQL with table grants remains outside this acceptance. No native foreign mutation was attempted or is needed to establish the original read-repair result.

## UAT201 — same name, separate owners and IDs

`bob-create-own-same-name-deck.txt` records an ordinary click on `flashcards-inline-create-deck-submit`. The POST at **05:43:44.849** names the deck `Alice UAT151 Citrine Deck`; its actual200 response at **05:43:44.881** creates **deck7 / owner3**. Alice's existing **deck1 / owner2** has the identical name. Bob's subsequent catalogue and saved card use7; Alice's later catalogue retains1 and excludes7.

The name intentionally contains “Alice”; that label alone does not imply cross-owner leakage. The response IDs and client_ids distinguish the two resources. Together with the already reviewed historical68→69 catalog/data/rollback/concurrency tests and security checks, this satisfies the native per-owner creation criterion. Native same-owner duplicate-conflict or destructive restore controls are not claimed; UAT202 is outside this audit.

## UAT151 — original history boundary now verified

The first evidence batch provided valid **Close-return** controls: Alice's selected deck and StudyPack Note cleared after switching to Bob; Bob's empty catalogue yielded Create new deck in both generation and image-occlusion selectors. Bob's separately primed pasted draft cleared after switching to Alice. These captures used Settings Close, whose current Next implementation calls `window.location.assign`, so the preliminary audit correctly held AC3 open. That earlier disposition is preserved in `PRELIMINARY-close-return-only.md` and is superseded by the following added literal history capture.

`alice-literal-back-primed.txt` shows Alice's unique **deck6 / owner2** label, `Alice UAT196 concise recovery 20260917`, in **both** selectors, plus the private text `Alice literal browser Back private draft UAT151.`. After the normal logout/login flow, safe auth/me events identify Bob3 at **05:59:21.567/.639** and **05:59:27.718**. `bob-literal-history-back.txt` explicitly records **`await page.goBack();`** and the resulting `/flashcards?tab=importExport` URL.

At **05:59:27.751/.782**, Bob's actual catalogue is200 with only **deck7 / owner3**, named `Alice UAT151 Citrine Deck`. The card list at05:59:27.752 has exactly Bob's UUID and total1. On the **first settled** snapshot, both selectors show that current catalogue label, Alice's unique deck6 label is absent, private text is blank, and Generate is disabled. The matching screenshot visually corroborates the blank text. The subsequent StudyPack dialog has blank fields and zero sources.

This establishes the original Alice→Bob history-Back failure boundary: a selected foreign deck absent from the new account's catalogue is not retained. Bob now owns one deck instead of the original empty catalogue; this is an explicit data variation, supported by the earlier native empty-catalogue control and the retained actual mounted empty-catalogue regressions. No extra Forward/Back remount, synthetic response, or page reload was used to obtain the settled literal-Back result. The original task requires the Alice→Bob scenario; a new reverse literal-Back test is not imposed. Reverse Close-return evidence remains useful but is not relabeled as browser Back.

Combined with already retained independent200/14 regressions, static checks, and review, the added capture satisfies151 AC3. No native generated-model save or late occlusion continuation is claimed; those write-authority cases remain covered by the relevant automated controls. No private login receipt or credential field is needed in durable acceptance evidence: safe identity events and the history action establish the tested boundary.

## Review limits

- Identical selected labels cannot alone prove the reverse transition's underlying selected ID. Native wire evidence proves Bob's manual save used7; automated scoped-selector tests cover invalid selection/write authority.
- No raw SSE, internal app state, invented request IDs, full clean-console claim, or all-domain authorization claim is made. The bounded observer retained no pageerror event, but it is not a general console auditor; an isolated auth/me401 elsewhere in the captured sequence is not suppressed.
- Create versus list response shapes differ in scheduler_type; persistence equality is asserted only for the explicit stable fields above, not entire Bob response payload identity.
- Parent must preserve the exact safe inputs and this report durably before final task closure, given the earlier private-artifact loss. No credentials or session material should be copied with them.
