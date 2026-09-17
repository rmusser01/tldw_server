# Independent UAT191 / TASK13260.129 review

## Verdict

**Clear within the bounded scheduler-preview repair.** No material correctness or regression defect found. The three production and two test files match the frozen manifest and snapshots. I independently reran the specified backend and frontend controls; native disposable-card acceptance remains the parent's gate.

## Fresh verification

- Real route/SQLite/required-PG contract: **26 passed,0 skipped**,28.12s.
- Focused existing list/review/FSRS controls: **20 passed,0 skipped**,183 unrelated cases deselected,12.89s.
- Scheduler/schema and due-next error controls: **12 passed,0 skipped**,3.63s.
- Frontend actual component/query/service and adjacent controls: **39 passed,4 files,0 skipped**,3.05s.

These are58 backend test executions across the three requested commands, plus39 frontend tests; no claim of58 distinct backend cases is needed because the adjacent selections can overlap. Exact commands and redacted receipts are retained beside this report and hash-listed in `verification.json`. PostgreSQL runs used the existing official fixture runner with PG required and Docker autostart disabled, with explicit sandbox escalation for test connectivity. No native UAT database, browser or provider was used.

## Source and contract review

The endpoint opts in with `include_scheduler_preview=false` by default. Enrichment happens after the existing filtered list/count reads, changes only the response row's scheduler metadata and uses the unchanged `_attach_scheduler_preview` helper already used for due review. The response-local cache reads each non-null deck once, including a missing-deck result; it does not cache across requests. A deckless card uses the same helper's existing default. Filter arguments, ordering, pagination totals and the authenticated DB dependency remain unchanged. Existing scheduler settings errors map to400 only in the opted-in path; DB errors retain the established mapping.

The service adds an optional field; the unchanged URL serializer skips undefined values. The Cram hook sets true within the existing pagination loop, preserving query keys, limits, residue filtering and request sequencing. It adds no parallel endpoint request. The mounted regression retains the actual ReviewTab, Cram query, service and serializer, with a simulated response only at transport. Its companion Python tests exercise actual HTTP/router and saved scheduling, so the frontend fixture is not being mistaken for server validation.

The route regressions compare all four supported rating labels against committed due-minus-last-reviewed gaps and check version/state before and after. Custom SM-2+ and configured FSRS use the same active-deck settings as due review. Ordinary omitted/false requests remain unenriched and readable even when opted-in scheduler validation fails. Mixed-deck pagination, one lookup per deck, empty pages and no mutation from preview reads are exercised. Existing Cram controls preserve schedule-off behavior, recovery, identity progression, re-rate and scope reset.

The same Cram hook also supplies practice and availability data. Those requests now receive previews and may fail for invalid scheduler settings even with scheduling off. This behavior is explicit in the approved design and retains existing queue-error recovery; it does not cause practice-only rating or session writes. The fix leaves the scheduler and ReviewTab preview selection unchanged. The post-rating response previews the *next* review of the updated card; that is distinct from the interval just committed.

## RED and static evidence

I inspected the retained causal route12-fail/4-control receipt and mounted4-fail/2-control receipt. Route failures reach the intended null-preview comparison after successful real ratings; frontend failures lack the supplied interval labels. The extra opt-in error/cache controls and earlier fixture problems are described in the author packet. I did not rerun old-source RED or count setup failures as product reproductions.

I read the author's static artifacts: endpoint Bandit0 findings/errors, Python-test Bandit0 with only pytest B101 excluded, and explicit3 TS/TSX parse errors with no claimed JavaScript security coverage. The author's scoped lint and90-diagnostic compiler comparison are retained; I did not rerun those checks or claim a globally clean compiler. The small production diff adds no SQL, provider call, permission bypass or raw sensitive data reporting.

## Provenance and limits

Frozen manifest SHA256: `4cd119b828905a84d3223c5ed71a12100bab6eaaad9a3808d78d441a81a94f1f`. Author report SHA256: `29c29ce67d609a9f5235ab39851c87ba0fae9100f11c3351955423c72dd58649`.

Production hashes:

- endpoint `flashcards.py`: `7d6933c5ed7ff06e55e2ff335c48ccf5fb93d0ff209dd98a10e92a140f5c6e1c`
- service `flashcards.ts`: `923f290dce3b692d9d1d5160f43dc0fdd03fb4678cb1c34fffc221f762068aaa`
- Cram hook `useFlashcardQueries.ts`: `8a0f75914c299b4fbd27fa700835330738f2872b6a82a0b3785f114b23259e4b`

`source-before.json`, `source-after-route.json` and `source-after.json` record relevant boundaries. All five owned files stayed hash-identical. External keyword/note/ReviewTab/Cram-control files also stayed unchanged. The shared `ChaChaNotes_DB.py` matched the starting hash through the26-case route run, then changed during the remaining review window to `9725bc3772edee0b770bff906a28c83fa08b76bb4dbd8ca366b51e7b15a3af9b`; the parent had explicitly allowed concurrent unrelated ChaCha ownership work. The file hashes do not identify which version a later already-running process had imported. Thus the26-case route contract has stable observed source boundaries, while the two later adjacent runs have this external-source attribution limit. This does not establish a191 defect, and no clean full-HEAD or native-runtime claim is made. Selected copied test receipts passed an exact known-PG-password/DSN scan without printing either value. No production/test/task/tracker/git edits or runtime/browser actions were performed by this reviewer.
