# Root visual review — UAT276/277

Inspected the reviewer-owned native screenshots without operating the browser:

- `02-desktop-pointer-absolute.png`: the horizontally scrolled catalogue ends before the detail panel; the row action button is inside the list. Its ordinary popup may extend over the detail panel as an overlay.
- `07-tablet-pointer-expanded.png`: expanded catalogue rows sit above the detail panel at CSS900×953. The popup is captured during its animation; use native pointer/action facts for interaction acceptance.
- `04-single-dialog.png`: shows the preceding row menu, not the confirmation. It is retained as a stale compositor frame and is not dialog-copy evidence.
- `08-single-settled.png`: single confirmation describes library removal, lists0entries/2attachments and retains10second undo/tab-local timer guidance. Cancel and Delete are visible.
- `10-bulk-settled.png`: bulk confirmation says it removes2world books from the library, with Cancel and Delete controls.

Screenshots establish the visible layout/copy only. Cancellation, absence of DELETE, normal reload and original data preservation require the separate native audit. Current acceptance captures useDPR1; the original failure usedDPR2, with the same affected desktop CSS viewport.
