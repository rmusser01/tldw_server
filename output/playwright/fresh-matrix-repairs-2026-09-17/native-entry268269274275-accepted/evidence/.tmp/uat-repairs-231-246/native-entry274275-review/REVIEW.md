# Independent native acceptance — UAT268/269/274/275

## Verdict

**CLEAR for the bounded lifecycle and count repairs: UAT268, UAT269, UAT274 and UAT275.** The offline verifier passes **33 checks across 62 hashed inputs**. **UAT276 (overlapping row controls) and UAT277 (misleading permanent-deletion wording) remain open.** Neither workaround nor this lifecycle acceptance closes those findings.

This uses the original PostgreSQL multi-user Alice2 profile, preserved book3 and original entry1, on committed source `ac713a4e238a51d548f12722126854e88c79b1c9`. The native interval is **2026-09-18 04:29:46–04:42:02 UTC**. Browser control was explicitly returned to root after removing only this review's passive listeners. No model request occurred.

## Accepted behavior

| Finding | Evidence and result |
| --- | --- |
| UAT274 | The original blue entry1, created at `03:39:54.428625`, was edited through its existing dialog. A single `PUT /api/v1/characters/world-books/entries/1` returned200 at04:30:38–39. Canonical ID remains1 in book3; creation timestamp is unchanged. Normal reload at04:31:05 returned and displayed the exact green replacement content. The original failure was not avoided by replacing the fixture. |
| UAT275 | A second disposable entry2 returned201 at04:31:53. Table, detail header and Entries total all moved1→2 without reload. Deleting2 returned200 at04:33:03 and all three returned to1 without reload. Normal reload agreed. Deleting original1 returned200 at04:34:16, with live0 and canonical/UI0 after normal reload. |
| UAT268/269 | Earlier accepted source270682 evidence proves this same book3 and entry1 creation, character7 attachment, reload readback, detachment and reload-empty association. Current source completes entry edit, both entry deletions, and book deletion. Normal supported Delete ran after its ten-second undo window: `DELETE /api/v1/characters/world-books/3` at04:37:21.417 returned200, explicitly **soft-deleted**. Normal catalogue reload at04:41:10 contained only books1/2; book3 was absent. |

The two earlier failures remain intact in `native-final-worldbook-sources-review`: the request to `/entries/undefined` returned422 without changing the original blue entry, and the parent count stayed0 while Entries showed1 until reload. Current observations supersede those specific failures; the historical evidence is not rewritten.

The separately reviewed backend lifecycle source and maintained test bytes still match this runtime exactly. `worldbook-lifecycle210-review` establishes **12 real SQLite/official PostgreSQL lifecycle passes, zero skips**, and the reviewed combined fixture receipt records57 passes, zero skips. Those controls cover caller rollback, wrapper portability, cache invalidation, soft/hard deletion, attachment idempotency and optimistic conflict. This review uses that established evidence; it does not claim to have exercised rollback, restore or hard deletion in the browser.

## Preserved fixtures and identity

- Final normal catalogue response objects for books1 and2 equal the initial response objects, including descriptions, versions and timestamps.
- Character4 remains version2, character6 version1; their original response objects are unchanged. Both have book1 attached and enabled. The original book's UI shows **Attached Characters (2)** after settling.
- Dedicated character7 remains version1 and detached, with canonical association array `[]`. It was not deleted.
- Normal application `/auth/me` observations remain Alice2. There was no login, credential entry, auth/header/storage injection or privilege change.
- Passive capture contains **460 API requests and460 responses**, no HTTP status≥400 and no inference request. This is a bounded capture count, not a claim about every startup console message or the whole product.

## Open UAT276: supported row action is covered by detail pane

The normal pointer click on `getByRole('button', {name: 'More actions for UAT268269 disposable public lifecycle 20260918', exact: true})` timed out after10 seconds. The capture records the detail pane intercepting pointer events. This was a real layout overlap, not a stale selector or offscreen control.

At CSS viewport **1200×953**, DPR2, scrollY178.5:

- Action button rectangle: x565.953125, y464.5, width24, height24.
- List rectangle: x80, width380.796875, right460.796875.
- Detail rectangle: x476.796875, width691.203125.
- The action is visibly drawn beyond the list boundary inside the detail region; the center hit test returns the detail's `DIV.space-y-3`. The screenshot also shows list columns and detail text overlapping.

Proof: `10-zero-reloaded-book-actions.txt`, `11-zero-pointer-observed.txt`, `11-pointer-overlap.png`, `12-book-menu-keyboard.txt`. Ordinary keyboard Enter on the same button opened the supported menu, allowing the independent deletion lifecycle to finish. **No force click was used. The keyboard workaround does not resolve the pointer-access defect.**

## Open UAT277: confirmation promises permanent removal

`13-book-delete-confirmation.txt` shows **“This will permanently remove:”** plus the ten-second pending-delete undo explanation. `14-book-deleted.txt` records the actual200 response as **soft-deleted**. The runtime backend source matches the separately reviewed soft-delete default. This is a wording discrepancy; it does not establish or request a post-deletion restore feature. The supported pending-delete timer completed normally, and the book disappeared from the library after reload.

## Source, runtime and evidence integrity

The startup receipt records API18703/Web18783 healthy200 on the exact committed source. Original profile, initialization and official PostgreSQL holder hashes match the immutable binding; the runtime role remains non-superuser without BYPASSRLS/CREATEDB/CREATEROLE. Prepared gate, completion, source manifest and reused-dependency proof hashes agree.

The API84240 and Next84458 receipts were observed still started at04:42:16, with hashes identical to startup. `process-observed-safe.json` preserves that observation before handback-related runtime changes. The audit separately records any later receipt status/hash, allowing a legitimate root-owned stop after acceptance without pretending mutable process receipts never change.

All sampled runtime sources match the prepared manifest, including the entry manager, canonical-ID mapping, parent manager/detail component, World Book domain methods, backend lifecycle manager and actual lifecycle test. The entry manager hash is `312053fec35ecfb9a35c5a438b91e812b436486ffe8999bdd3077c1a342cc791`; ID mapping hash is `57eebc59045faf27825541f0b56785df8326ae39e70670dce29dff85be178e9e`; backend lifecycle hash is `c6ae32d056d79c4fb5a529fae5e99d749ca855c0a975fc4d485fe4d72abca186`.

`audit.mjs` reads evidence offline, projects safe facts and hashes every reviewed input. Private records remain hash-only; six known credential values were checked in memory with zero matches in the safe audit. No private log, credential value or provider reasoning is copied into this report.

## Limits and unsuccessful attempts

- This is the preserved original PostgreSQL multi-user configuration, with reused isolated dependencies. It is not a clean OS install, fresh full48 matrix, SQLite/single-user native acceptance, or provider reliability result.
- Partial/failed bulk mutations remain covered only to the extent stated in the existing source review; this native sequence exercises ordinary successful add/edit/delete.
- Entry response timestamps are naive strings; this review does not claim entry chronology rendering acceptance.
- The pointer timeout is retained as UAT276. Transient “Character attachments unavailable”/loading text settled to correct canonical associations; no persistent attachment failure was observed.
- Root preparation notes retain two corrected command mistakes and a frontend-not-ready probe before successful startup. The final startup proof is the readiness evidence.
- Offline audit attempts initially assumed the private accounts collection was an array and pluralized “1 entry” incorrectly. Both verifier-only mistakes are recorded in `audit-attempts.json`; neither modified native evidence. The briefly suspected response-timing issue was not present: both response bodies already reported1.
- Only authorized browser UI mutations occurred: original entry1 edit, disposable entry2 add/delete, original entry1 delete, dedicated book3 delete. There were no direct API writes, interception, model calls, new fixtures/profiles, runtime/DB provisioning, product/test/Git/Backlog edits or forced clicks.
- Raw captures and screenshot remain locally hash-addressed. A retained packet containing safe audit/report only is not a standalone replay of the raw browser session.
