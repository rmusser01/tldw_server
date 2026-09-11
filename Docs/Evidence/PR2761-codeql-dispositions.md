# PR2761 reviewed CodeQL dispositions

Tracking: TASK-13013.3.1. Verified against GitHub on 2026-09-11 UTC.

The requester explicitly approved the 416 remaining, individually reviewed
repository-wide dispositions after automatic approval review requested that
authorization. Those 416 updates completed successfully. Together with the
previously reviewed alert 2671, the verified total is **417**: **405 false
positives and 12 synthetic-test findings**.

| Scope | False positive | Used in tests |
| --- | ---: | ---: |
| Actions | 241 | 0 |
| Python | 154 | 2 |
| JavaScript | 10 | 10 |

Two subsequent PR-only request alerts, **2675 and 2676**, were independently
reviewed against all eight SARIF flows and resolved individually after their
proof was committed in `86a09da9e1`. Integer normalization precedes the fixed
job-run API path; initial/retry origin and credential guards remain enforced.
Both GitHub instance lists contain only PR2761. The ledger therefore now has
**419** dispositions, including these two additional false positives. These
were separate verified actions, not additions to the original approval list.

The [per-alert ledger](PR2761-codeql-dispositions.csv) records the actual GitHub
rule, reason, timestamp and submitted explanation. Before each update, the
executor checked the alert identity, rule, source path and effective state;
afterward it checked the returned disposition and exact comment. A separate
read of both default-branch and PR-specific dismissed-alert lists reconciled
all 417 IDs and reasons. PR-only alerts are omitted from the default-branch
list, so both scopes are required for this verification.

Trace-specific proofs, source hashes, cross-branch reviews and behavioral tests
are in the [Actions](PR2761-codeql-actions.md),
[Python paths](PR2761-codeql-python-paths.md),
[Python boundaries](PR2761-codeql-python-boundaries.md) and
[JavaScript](PR2761-codeql-javascript.md) evidence. Shared guard behavior does
not replace individual source/sink review.

No query, threshold or workflow gate was disabled. Actual source deficiencies
were excluded from these dispositions. In particular, main's checkpoint
ownership defects associated with alerts 2281 and 2282 remain undismissed.
New rescan alerts are assessed separately; this approval does not authorize
dismissing an accurate operator-credential or path-boundary finding as a test
fixture or accepting its risk.

The 417 dispositions are not an assertion that the release is ready. The
current-source rescan still requires reconciliation of repaired and newly
identified flows, and the separate release gates remain in force.
