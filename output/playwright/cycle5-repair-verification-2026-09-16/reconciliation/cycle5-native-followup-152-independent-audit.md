# Independent native follow-up checkpoint audit — UAT001–152

Date: 2026-09-16. Reviewed staged checkpoint bytes for the targeted native run recorded on `1ea5402c83`. Read-only review: no browser, runtime, inference, tests, source, task or repository documentation writes.

## Outcome

The acceptance dispositions are defensible: **140 verified, 8 implemented awaiting acceptance, 2 blocked, 2 unresolved; 152 unique contiguous IDs (001–152)**. UAT064 and UAT140 have the original-scenario evidence needed for their bounded acceptance. UAT013 remains awaiting a positive source answer; UAT151 and UAT152 remain unresolved in this historical checkpoint. UAT114 and UAT118 remain blocked. Another full UAT remains blocked.

Follow-up staged-index verification confirms both requested documentation corrections are now applied:

1. **UAT151 sequence corrected:** the tracker now limits the stale Alice label to the settled first Back and explicitly records that subsequent Forward/Back shows `Create new deck`. The multi README also qualifies URL observation, screenshot cropping and reload causality. The original overclaim was that the label remained “including after settling and Forward/Back.” `followup-native-multi/uat064-bob-back-settled.md` does show both stale Alice deck labels, but `uat064-bob-forward-back.md` shows `Create new deck` twice. Retain the confirmed settled-Back failure and state the repeated-history control accurately. Do not attribute clearing uniquely to reload: the retained Forward/Back capture already shows cleared labels. This correction does not negate the confirmed initial leak or UAT064 source clearing.
2. **UAT013 evidence corrected:** `original_scenario_native_acceptance` now records actual Cedar restoration, request434 with `include_media_ids:[2]`, the browser timeout, backend200 in19.689seconds and pending positive152 acceptance. It links the new README/request. The disposition remains `implemented-awaiting-acceptance`, appropriately.

The additional stale UAT013 `current_acceptance.summary` was also corrected in the staged index and now agrees with the new scoped-dispatch/pending-positive evidence. UAT140's copied current-acceptance field was updated as well. **Final correction status: clear; no remaining acceptance contradiction identified.** Counts and dispositions are unchanged.

## Evidence review

All relative paths below are under `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle5-repair-verification-2026-09-16/`.

### UAT064 / TASK13260.13

- `followup-native-multi/uat064-alice-hydrated.md` and `uat064-alice-fresh-positive.md`: exact five-fact Biology source and attached Note provenance are visible on initial and fresh-return transfer.
- `uat064-bob-hydrated.md`: Bob's exact `BOB-BROWSER-ONLY-CYCLE5 violet orchard has 17 bells.` source and Bob Note provenance are visible.
- `uat064-bob-back.md`, `uat064-bob-back-settled.md`, `uat064-bob-forward-back.md`, `uat064-alice-return-back.md`, and `uat064-alice-return-forward-back.md`: source textarea is empty and attached source provenance is absent after the recorded account-history transitions.
- `uat064-bob-auth-me.json` identifies Bob, ID3; `uat064-alice-fresh-auth-me.json` identifies Alice, ID2. The retained README records the normal Logout/Login and clean `/flashcards?tab=importExport` navigation. These AX captures do not themselves include an address-bar receipt, so clean URL remains the controller's recorded native observation, corroborated by the reviewed transfer implementation/regressions rather than an independently visible URL in these captures.
- The task explicitly combines this original history acceptance with prior real Bob generation/save and reviewed delayed-generation/save/one-time-consumption controls. It does not claim native acceptance for the other producers or every race. That scope is appropriate. UAT151's independent cached deck label leak is not evidence that the Note body/provenance clearing failed.

### UAT140 / TASK13260.79

- `followup-native-single/uat140-conversation-settings.md`: actual `Current Chat Model Settings` dialog, `Conversation` selected, labelled numeric controls present in the accessibility tree.
- `uat140-console.txt`: 6 messages, 0 errors, 0 warnings. No addonBefore warning.
- `uat140-conversation-settings.png` inspected: visible actual Conversation modal. The viewport only shows its upper portion; the AX capture is the evidence for controls below the fold.
- No native editing/persistence claim is made. Existing actual-AntD permanent tests cover those task criteria. This matches the task's original warning scenario and bounded native acceptance requirement.

### UAT013 / UAT152

- `followup-native-single/uat013-cedar-restored-actual.md` contains the saved Cedar transcript; `uat013-home-source.md` retains the Home source action context.
- `uat013-rag-request.json`: actual query `Chat with this media: cycle5-home-handoff-proof-20260916.md\n\nSummarize this source.` and `include_media_ids: [2]` establish the correct source-scoped RAG dispatch.
- `uat013-after-send.md`: settled response says evidence retrieval failed and that it did not send as general Chat. `uat013-console.txt` records the RAG timeout warning. This proves the negative presentation and correct dispatch, not a delivered positive answer or a comprehensive independent count of all network activity.
- `uat152-timeout-settings-expanded.md`: Balanced selected; generic request, Chat request/startup and RAG request values are 10 seconds; idle values are 15 seconds.
- `uat152-backend-rag.redacted.log:195`: POST `/api/v1/rag/search` returns HTTP200 in **19689ms**. Server completion does not establish browser consumption after its timeout.
- Retained harness mistakes are correctly disclosed: `uat013-cedar-restored.md` shows a new-chat state from an unsuccessful deep link; it is not restoration proof. `uat152-timeout-settings.md` contains only an empty tabpanel; the expanded artifact supplies actual values. Neither is a product failure.

### UAT151

- `followup-native-multi/uat064-bob-back-settled.md` contains both `Cycle5 Alice Biology` selectors while the private source textarea is empty.
- `uat064-bob-decks-response.json` is `[]`; `uat064-bob-auth-me.json` is Bob3. Together they support the bounded stale-client-label finding. No cross-account generation/save was attempted or established.
- `uat151-bob-stale-alice-deck.png` was inspected but is cropped above the deck selectors; use the settled AX snapshot as direct proof of the labels, not that PNG alone.
- `uat151-bob-reload-control.md` shows `Create new deck` twice but includes a loading-time capture qualification. The README's later locator observation is narrative. As noted above, the earlier Forward/Back control also clears the selectors, so a unique reload-causality claim is unwarranted.

## Staged integrity and secret-scan record

- Independently hashed the **staged bytes** of all manifest-listed files via `git show :path`.
- Single bundle: **13/13 files**, 232346 bytes, all SHA256 hashes and sizes match.
- Multi bundle: **15/15 files**, 178289 bytes, all SHA256 hashes and sizes match.
- Every staged evidence file in both directories is covered, excluding each manifest itself.
- Both manifests identify native source `1ea5402c83` and record `secretScan: { knownValues: 62, findings: 0 }`. This audit confirms the recorded scan and exact byte binding; it did not access secret needles or independently rerun that scan.
- Reconciliation checkpoint SHA256 matches its staged manifest: `613cdf500e5359ac69948e3509e47108afb12de0ee1f95534247d233fdb80099`.
- The controller regenerated the affected multi-bundle and reconciliation manifests; the final corrected staged reconciliation hash matches. The 28 evidence hashes matched after the README correction, with no further bundle changes. This report does not certify later modified bytes.

## Limits

This is evidence reconciliation, not a fresh run, test execution, security-wide audit or PostgreSQL native matrix certification. Native image recovery UAT118 and hidden-tab/starvation UAT114 are still explicit gaps. Subsequent UAT153 belongs to a later checkpoint and does not alter the historical 152-row count reviewed here. No changes were made to UAT151 implementation work.
