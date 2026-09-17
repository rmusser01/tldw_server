# UAT244 — saved ingest detail with stale empty catalogue

Task: `TASK13260.186`. Read-only diagnosis against frozen `8f8774e6c868b304a96d95ab82e28389c129a78b`. All 14 inspected source/history files match both the archive manifest and controller Git bytes at that revision.

## Conclusion

**High-confidence missing catalogue refresh in the active Quick Ingest wizard path.** The selected detail and catalogue have separate state owners. Open in Media hydrates the new source successfully but leaves the already-mounted catalogue’s old empty result unchanged. No persistence failure or cross-account disclosure is shown.

## Evidence window

| Retained event | Result |
|---|---|
| Bob job6, 16:23:49.008Z | Completed, owner3, media1 |
| Bob detail, 16:25:10.234Z | 200, 1906 source characters; following snapshot still Results0/0 and first-ingest onboarding |
| Admin catalogue, 16:42:49.892Z | 200, items0/total0 before ingest |
| Admin job7, 16:43:41.470Z | Completed, owner1, media1 |
| Admin detail, 16:43:42.035Z | 200, 1914 characters; catalogue stays0/0 |
| Admin snapshot, 16:44:14.447Z | Still0/0 and onboarding,32.412 seconds after successful detail read |
| Admin delete / restore | Delete204 at16:45:14.733Z; restore200 at16:46:02.481Z |
| Subsequent navigation catalogue, 16:46:03.054Z | 200, items1/total1; final snapshot1/1 |

The cumulative admin capture contains no catalogue request between completion and deletion, while detail polling remains200. Bob’s plain `resume-1628-snapshot.txt` corroborates stale presentation but has no embedded capture timestamp; its filename is not used as timing proof.

## Source boundary

1. `QuickIngestButton.tsx:24–28` loads **QuickIngestWizardModal**, which renders **WizardResultsStep**. Those active components never emit `tldw:quick-ingest-complete`. The only production emitter found in frozen `apps` is legacy `ResultsPanel.tsx:117–131`. The wizard’s completion effect (`QuickIngestWizardModal.tsx:1040–1065`) updates recent-document metadata for DocumentPicker only.
2. `ViewMediaPage.tsx:317–335` listens for that event and explicitly refetches after1500ms. `useMediaSearch.ts:655–683` sets `enabled:false`; `803–835` drives searches on mount/criteria/page changes. A new selected media ID is not a catalogue query input.
3. Wizard `handleOpenMedia` (`1686–1698`) checks current operation, navigates to `/media?id=…`, and closes. `useMediaNavigationState.ts:162–282` independently loads missing permalink detail and publishes selection/content; it does not refresh the catalogue.
4. `ResultsList.tsx:308–340` renders first-ingest guidance from empty results. Thus the left pane can remain empty while the right pane has a valid newly saved source. The page’s whole-library empty guard already excludes selected/permalink detail; this is not simply that guard overlooking selection.

Existing `ingest-complete-event.test.ts` manually dispatches an event and tests its payload. `WizardResultsStep.navigation.test.tsx` checks callbacks. Neither inspected control proves active wizard completion reaches a mounted catalogue.

## Prior findings and repair boundary

- **030/.9:** owned-source full-text/Knowledge QA retrieval and worker ownership. This window shows no post-ingest catalogue read to fail; it is a different boundary.
- **082/.26:** private ingest authority/session leakage. Keep its current-operation and late-callback protections intact; no stale foreign result is alleged here.
- **071/.17:** hidden Trash after last deletion. Trash works in this evidence. Historical **081/.25** concerns saved Flashcard source destinations, not empty Trash; here the Media destination correctly hydrates detail.

For a later repair, first reproduce actual wizard completion against already-empty Media and verify one current-owner refresh reconciles items, counts and onboarding. Cover same-route handoff, completion before Media mounts, repeated/resumed batches, process-only/failure outcomes, and account/server replacement. Use the existing explicit refresh contract; generic invalidation alone need not execute an `enabled:false` query. Preserve filters/pagination rather than inserting detail blindly into the list.

## Limits

No causal tests, live requests or runtime inspection were run. A fresh backend catalogue response immediately after ingest is absent, so that exact backend state is not independently proved. Restore/navigation is a later recovery observation, not proof that deletion is necessary. Only the two authorized ignored audit files were written.

Companion JSON binds all five inputs and14 inspected files: SHA-256 `e3950d54b0da88378aaf743961f6094ab6d343b6935a4b36c005195c3109e9d0`.
