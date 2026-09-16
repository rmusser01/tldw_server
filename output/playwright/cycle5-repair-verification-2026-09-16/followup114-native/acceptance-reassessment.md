# UAT114 AC3 reassessment

**Disposition: AC3 is supported; TASK-13260.54 can be Done.** This corrects the earlier retention note that kept the task open for a fresh pre-fix native stalled replay. That demand was stronger than the actual criterion and approved implementation contract.

The unchanged criterion reads: “Regression demonstrates the observed stalled-request boundary and targeted native recovery passes.” It requires regression coverage plus native recovery; it does not explicitly require a pre-fix native rerun or proof of socket exhaustion.

## Evidence mapped to the contract

1. The task's original implementation notes define the bounded repair as actual `NotificationLifecycleProvider` ownership: hidden tabs release SSE, active reads and polling; activation catches up with the captured current owner/credentials. This is the controllable resource-retention boundary implicated by the original six-tab stalls.
2. Retained `../../cycle4-repair-verification-2026-09-16/hidden-tabs114/uat114-permanent-red.txt` records four failing baseline visibility regressions and two passing terminal-state controls. The reviewed diff exercises initial hidden startup/retry; stream unsubscribe and120-second poll pause; pending bootstrap abort/late result; terminal401/403 preservation; hidden account replacement; and fresh unread/cursor catch-up. These regressions demonstrate the faulty hidden-tab transport ownership, rather than asserting a hypothetical browser socket count.
3. `cycle4-uat114-independent-tests.log` and its review record111/4 passing tests with no skips. All nine files listed in that package's evidence manifest were freshly hash-verified during this reassessment. These are retained executions, not a new test run.
4. The fresh native package in this directory independently demonstrates genuine hidden/visible pages, six same-origin application tabs, aborted hidden notification streams18/94, sole remaining visible stream361 at `after=1`, a persisted hidden-interval notification caught up in the badge/inbox, and actual prompt writes317/318 each201 in16ms. This supplies the previously missing targeted native recovery half.

The baseline resource-release regression plus passing preservation coverage and genuine six-tab native result satisfy AC3 for the approved repair. AC1 is directly supported by ordinary writes and working notifications; AC2 retains its prior preservation-test support. No stronger acceptance criterion is added and none is rewritten.

## Limits retained

Original HTTP connection exhaustion remains an **inference**, not socket-level proof. There is no fresh pre-fix native stalled rerun. The native receipt does not claim all hidden application traffic stops: auxiliary buddy/health polling completes200, and one early unread-count lifecycle lacks a terminal event. No new persistent stall is demonstrated. These limits remain in `independent-audit.md` and do not contradict the bounded ownership/native-recovery acceptance.

The prior In Progress note remains historical; this document and the appended task correction supersede its disposition. No source/test/browser/runtime/git actions were performed for reassessment.
