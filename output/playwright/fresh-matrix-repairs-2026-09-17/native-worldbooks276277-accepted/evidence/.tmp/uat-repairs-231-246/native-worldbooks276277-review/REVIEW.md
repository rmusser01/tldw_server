# Independent native acceptance — UAT276 / UAT277

## Verdict

**CLEAR for bounded native UAT276 and UAT277.** The offline verifier passes **26 checks across54 hashed inputs**. The original PGmulti Alice2 browser used committed source `c13701938e86ed8e9c83cacebdcd5d7fb3abe7e9`. Both original books and their associations remain unchanged. **No DELETE or model request occurred.** Browser control was explicitly returned after removing only this review's passive listeners.

## UAT276: real containment and pointer access

At **CSS1200×953**, the table scroll region was380.796875px wide, with381px client width and517px scroll width. Ordinary horizontal mouse-wheel input moved scrollLeft from0 to136. The original book1 More actions button then occupied x421.0234375–445.0234375, inside the list's x80–460.796875 boundary. The detail pane began atx476.796875. The button's visible center hit its own icon descendant; a normal pointer click opened the supported menu.

At **CSS900×953**, the selected catalogue uses the existing collapsible tablet presentation. After a normal click on its visible summary, the table region was802px wide with no horizontal overflow; the action atx834.9921875–858.9921875 remained inside the list. The list ended aty516 and the detail began aty533. The center again hit the button's own descendant, and a normal pointer click opened its menu. No force click, keyboard-only workaround, CSS injection or synthetic dimension assignment was used.

The actual screenshots and geometry show the table contained separately from the detail pane. Dropdowns intentionally overlay nearby content while open; that ordinary popup behavior is distinct from the original table columns painting over the detail pane.

Current viewport sizing reports **DPR1**, whereas the original failure image wasDPR2. These checks match the required CSS viewport sizes and measure actual browser layout/hit targets. They are not a device-pixel-identical image comparison. Desktop actions can require ordinary horizontal scrolling; the fix does not claim every column fits initially.

Evidence: `02-desktop-pointer-corrected.txt`, `02-desktop-pointer-absolute.png`, `07-tablet-pointer-expanded.txt` and `07-tablet-pointer-expanded.png`.

## UAT277: native confirmation wording, cancelled

The single-book dialog displays **“This world book will be removed from your library. It contains:”**, followed by0 entries and the two existing attached characters. It retains the ten-second undo explanation and the warning that refresh/navigation cancels the local pending timer. The dialog was cancelled. The settled element screenshot `08-single-settled.png` visibly establishes the text and controls; `08-single-settled-cancelled.txt` records Cancel and subsequent hidden state.

After restoring1200×953, normal checkboxes selected the two original books. The bulk dialog displays **“This will remove 2 world books from your library.”** It was cancelled and selection cleared. `10-bulk-settled.png` and `10-bulk-cancelled.txt` provide visual and interaction evidence. Neither dialog promises permanent erasure or post-deletion restore.

No deletion was submitted or scheduled. This copy check did not repeat the previously accepted ten-second deletion lifecycle. The bulk count1 branch is source-reviewed only; this native check uses count2.

## Preservation and recovered background auth

Normal reload at **05:18:15 UTC** returns only original books1/2, with whole canonical objects equal to the initial observations and preceding lifecycle review. Descriptions, versions, timestamps and counts remain intact. Character4/6 metadata is unchanged; both full book1 association objects remain enabled. Character7 remains detached. Book1 shows Attached Characters(2).

The capture contains270 API requests and270 responses. The sole non-GET request is one automatic `/api/v1/auth/refresh` POST. At05:15:39.237/.238, background buddies and attachment GETs returned401. Refresh returned200 at.260; automatic `/auth/me` returned Alice2 at.291; both automatic retry GETs returned200 at.305. No user action failed or identity loss was observed. Error bodies were intentionally not recorded, so the exact cause is unestablished. This recovered sequence alone does not establish a new authentication defect or prove token expiry.

## Source and runtime binding

Healthy startup, immutable source manifest, completion/gate/dependency proofs and runtime bindings agree on c13701938. Original profile, initialization, config and official PostgreSQL holder hashes are unchanged; the restricted role remains non-superuser without BYPASSRLS/CREATEDB/CREATEROLE.

Serving panel hash `082514ca18f5f38a413c12b1c04f63971ef8e0d715317d5806637b6115ca434f` and hook hash `2522236f2596586aee7b6ed8ca358f53e2c8b00736fdad1347b5526ca0e1252b` match the independent source reviews. Related manager/domain/backend/auth sources match the prepared manifest.

API97657 and Next97812 were observed still started at **05:19:18 UTC**, with hashes equal to startup and coverage of the native interval. Mutable receipts may subsequently reflect a root-owned stop; the audit preserves the observed snapshot separately. Root's visual review is retained as an additional input, independent of this audit's assertions.

## Retained harness limits

- A mistyped run label failed setup before any browser action. Its output remains `02-desktop-pointer.txt`; the corrected run is separately retained.
- The first relative screenshot was written under the original browser profile's sourceRoot `.tmp` directory. It is preserved and hashed; no product file changed. A separate absolute-path screenshot exists in this review directory.
- `04-single-dialog.png` caught the preceding menu frame and is **not** dialog-copy image evidence. The associated text records the actual dialog/Cancel. The settled08 screenshot supersedes the image for visual proof.
- `07-tablet-pointer-expanded.png` catches the popup during animation. Its real containment geometry and pointer result, followed by the settled single-dialog image, establish the relevant interaction.
- The first900px helper timed out awaiting the hidden table before recognizing the intended collapsed catalogue. It was corrected by normal summary expansion. A premature file/image lookup while that helper was pending returned no usable evidence; only completed captures feed the audit.
- This is one preserved PostgreSQL multi-user configuration with reused dependencies, not a clean installation, full48 matrix or all-mode claim. No direct API writes, auth injection, model/settings changes, fixture mutation, product/test/Git/Backlog edit or forced click occurred.
- Private records remain hash-only. Known credentials are checked in memory; no secrets, private logs or provider reasoning are published. Safe-only retention does not independently replay omitted local screenshots/captures.
