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

Whisper alert **2679** was then independently resolved after proof and regression
tests were committed in `6b671f22bc`. Its four reported relative-Hub-ID flows
cannot pass the absolute-path guard before the existence probe. Its only active
instance is PR2761. See [Whisper rescan evidence](PR2761-codeql-whisper-rescan.md).
The ledger now totals **420 dispositions: 408 false positives and 12 synthetic
test findings**. The accurate account-lookup finding 2678 was repaired, not
dismissed.

The `009505c415` JavaScript rescan then cleared real storage findings 2673/2674
and identified regression-test writes **2680/2681**. Independent exact-flow and
instance review confirmed fixed synthetic literals, installed memory facades,
native-empty assertions and PR-only scope. Both were individually resolved as
used in tests; 12 seed tests pass. The ledger now totals **422 dispositions:
408 false positives and 14 synthetic-test findings**. See the final section of
the [JavaScript evidence](PR2761-codeql-javascript.md).

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

The reviewed dispositions are not an assertion that the release is ready. The
current-source rescan still requires reconciliation of repaired and newly
identified flows, and the separate release gates remain in force.

Two final PR-only shared-guard alerts, **2686/2687**, were independently
reviewed against all four flows and actual endpoint/root-policy behavior.
Their base is deliberately selected by an administrator; no generic tainted
base sanitization is asserted. Both individual false-positive updates were
verified against fresh identity, source path, PR-only instance scope, returned
state and exact comment. The ledger now totals **424 dispositions: 410 false
positives and 14 synthetic-test findings**. See the
[workspace-root trace proof](PR2761-codeql-workspace-root-traces.json).

Python analysis **1759364372** on `49cce1c853` cleared the ten previous snapshot,
Research and checkpoint alerts. Its three remaining Whisper reports were
repaired in `2e037be445`; hosted confirmation is pending. No additional real
finding was dismissed.

## Final verified result

All three hosted analyses completed on **58070a0fea5e636d5426b25ced16b66e7147f397**:
Python **1759451338**, JavaScript/TypeScript **1759415368**, Actions **1759389568**.
The PR has **zero open CodeQL alerts** and its CodeQL check passed. The production
source is **2e037be4452ddae74807ab672fec94a88c030cc0**; later commits contain
release metadata and evidence. The three Whisper repairs are confirmed by this
scan. Raw SARIF result counts include dismissed findings and are not open-alert
counts. [Machine-readable result](PR2761-codeql-final-result.json).

An independent GitHub reconciliation checked all **424** unique ledger rows
against rule, reason, exact comment, timestamp and effective dismissed state,
using both default-branch and PR-specific lists. Main findings **2281/2282**
remain open and undismissed. At this verification checkpoint, CI had **73
passing checks, zero failures, 38 skips**, with frontend/coverage aggregates
still running. The CodeQL task is complete; the separate release gates and
human Change summary remain open. No merge, tag or publication was performed.
