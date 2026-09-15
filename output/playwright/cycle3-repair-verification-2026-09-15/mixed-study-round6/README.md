# Targeted mixed-deck Study verification

Existing isolated multi-user Bob runtime, Study repairs backend982b03a940 and frontend536ad461e9; checkout f9b0dd2f53. Production changes were paused through this check. No mocked responses, seeded database rows, clock changes or credential-store manipulation.

## Result

Two cards created through real Note generation/save were moved out of Biology using each card's normal Move to deck → Clear → Move action. Independent GET verifies five Biology learning cards plus two undecked new cards. The initial dashboard shows Biology Due5/New0/Learning5 and Review5ready; the all-decks due queue shows7.

The runner reports revealing and rating seven visible cards Good between19:47:01 and19:48:54UTC. Individual action captures are retained through rating4; the last three are corroborated by the terminal UI, per-card timestamps/versions, session accounting and filtered review/end responses below. The first acknowledged server session is id2, due:global, deck_id null, one reviewed. The final independent read shows the same id2 completed with seven reviewed, and no other new session. All seven card versions advanced once and review timestamps changed within this run; five remain decked and two undecked. Repetition counters are scheduler state, not event counts: only the two new cards incremented that field. Filtered API access lines corroborate seven review200 responses and one automatic review-sessions/end200.

The native UI shows All caught up,7 cards reviewed this session, and Recent study sessions → All decks → Completed →7. Actual reload followed by opening Study preserves that completed7-card row. Next review shows two cards due within the hour following the next due time.

## Evidence limits

The normal queue-empty transition automatically ended this run. No explicit End Session button was clicked; explicit early End and cancellation remain covered by focused regressions and require separate live checks where specified.

The immediate first-rating capture shows the previous queue count while refetch was pending. The settled next-card capture correctly shows6 remaining/1reviewed; subsequent retained steps show5/2,4/3,3/4. Runner-reported diagnostic: the final scripted wait expected a zero-remaining status, but the terminal screen replaces it with the completed summary, causing an automation timeout after seven successful ratings. The failed command emitted its error to the tool output, leaving its redirected file empty; a raw final-loop/error capture is not retained. No repeated ratings were sent. The terminal snapshot and independent API comparison establish completion.

This is targeted multi-user acceptance for mixed-session accounting and related visible counts. It is not a fresh install, PostgreSQL runtime pass, practice/Undo/early-End check, or completion of remaining single/multi matrices. No raw credentials, headers or unfiltered network captures are retained.
