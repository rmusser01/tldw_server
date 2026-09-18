# Native acceptance: UAT260 / UAT265 / UAT266

**CLEAR for these three bounded native criteria.** All 30 audit checks pass. Separate Character edit UAT271 and reciprocal attachment-count candidate UAT272 remain open; this is not acceptance of the whole World Book UI.

## Verified native outcomes

| Finding | Actual result on original Alice PostgreSQL profile |
| --- | --- |
| UAT260 / TASK13260.202 | Existing browser timezone remained `America/Los_Angeles`. Original book1 read now includes `Z` and shows “2 hours ago.” New public book2 was created201 at 02:19:07 with `2026-09-18T02:19:07.412074Z` for creation and modification. Immediate UI and normal reload show “a few seconds ago”; canonical fields survive unchanged. |
| UAT265 / TASK13260.207 | Repeated the exact original description edit that previously returned500. Normal PUT book1 with `expected_version=1` returned200, version2, at 02:16:59. Original creation instant stayed unchanged; modification became `2026-09-18T02:16:59.615253Z`. Normal reload at 02:18:06 retained the exact description and displayed “a minute ago.” |
| UAT266 / TASK13260.208 | A new public Character6 was created through Alice's normal form,201 at 02:29:17.127. Its selected original book1 produced POST `/characters/6/world-books`200 at 02:29:17.195. Normal page reload and editor readback finished02:29:58.432 with canonical GET200, one book1 link, enabled=true, priority0, version2, and a visibly retained selection. |

All times are UTC on 2026-09-18. Book1 is the original **UAT239 Alice public fictional context**; book2 is **UAT260 Alice timestamp fixture 20260918**. The added description sentence is `UAT265 verified edit.` Character6 is **UAT266 Alice public attachment fixture 20260918**, with manually entered public fictional instructions. No inference was requested; the all-API passive observer records zero completion/RAG/image-generation dispatches.

## Failures retained separately

**UAT271: existing Character4 Save remains blocked.** Before creating Character6, the normal original character editor selected book1 and submitted PUT `/api/v1/characters/4` at02:26:14.915. It returned422 at02:26:14.940:

- Validation: `type=missing`, `loc=[query, expected_version]`, `message=Field required`.
- The request had no query and only body keys `name` and `system_prompt`.
- Canonical character catalogue responses exposed Character4 version1 before and after the failed edit. No Character4 attachment POST occurred.
- The public character response does not expose an owner column. The review verifies Alice2 against the original private fixture and normal scoped UI operations; it does not claim a direct database owner-row check.

The original failure was reported immediately. Character6 is a separately authorized path proving the repaired attachment endpoint and persistence; it does not turn the failed original-card operation into a pass.

**Candidate UAT272: reciprocal World Book UI still displays a false zero.** After Character6's canonical saved link, normal navigation at02:30:39 to `/world-books`, selected book1, Attachments tab, still showed `Attached Characters (0)` and `No characters attached.` The observed navigation fetched the book catalogue/config200; no `/world-books/1/characters` request occurred. This is a separate unclassified UI/data-flow issue, not attributed to UAT268/269. Character-side canonical persistence is established; reciprocal UI correctness is not.

## Source and fixture binding

Native source was **0d7f2a23c95e9fc301959610721f7e1175e5cafa**, WebUI18783/API18703. The immutable upgrade binding, preparation gate/completion/manifest and exact frozen implementation/test hashes agree with the independent source review. Production `world_book_manager.py` SHA-256:

`1ffb3839115e34daa162abae2f9b6648741597814744798d435c70960010c870`

The original profile, initialization and official fixture-holder bytes match the upgrade binding. The held runtime role is neither superuser nor BYPASSRLS and cannot create roles/databases. Both app receipts were still byte-identical to startup at the safe pre-handback observation (API36086, Next35312). Their immutable binding and interval cover the native evidence. Later root-owned process shutdown may update receipt status/hash; the pre-handback observation is retained separately.

The existing independent source review supplies 45 actual SQLite/official PostgreSQL tests with no skips, nine permission tests and 26 frontend formatter tests. This native evidence review did not rerun them or change product/tests. The timestamp repair retains its declared stable writer/reader session-timezone policy; unknown historical PostgreSQL writer zones remain ambiguous.

## Evidence and limits

`audit.json` records 30 checks and 61 hashed inputs, including private hash-only provenance. `audit.mjs` recomputes the checks using only local files. The decisive completed captures are 04/05/07/09/10, 19, 22/23, and25. `runtime-observed-safe.json` preserves the process observation before handback.

Five unsuccessful helper captures remain unchanged: wrong display-name wait (02), unavailable CLI `URL` constructor (06), hidden accessibility option click (16), wait prevented by actual422 (17), and waiting for an absent reciprocal endpoint (24). A premature local parse of the last running helper also failed; only its completed capture was evaluated. The corrected keyboard selector, actual responses and normal canonical reloads provide the successful proof.

Initial auth/me401 recovered normally to Alice2. Character-list429 polling and unsupported visual-identity metadata were also observed; no zero-error startup/console claim is made. No private logs, credentials or provider reasoning are included in this report/audit. Raw local captures stay local. This is acceptance on one preserved PG multi-user profile, not a clean OS install, new PG single/SQLite native run, model-reliability result, or full48 matrix acceptance.

Browser lease was returned at **02:31:24.190**, `/world-books` → original book1 → Attachments, with no unsaved edit. Only this review's passive listeners were removed; the original observer remained installed. No runtime, fixture, product, Git, Backlog or model mutations were made.
