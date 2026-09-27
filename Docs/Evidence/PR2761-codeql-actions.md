# PR 2761 Actions CodeQL review

Tracking: TASK-13013.3.1. Reviewed 241 individual alerts from the Actions analysis of
`28797892b1e021dad55cbc736532e702f0963ae8`. The companion
`PR2761-codeql-actions.json` maps every alert ID to its exact sink, workflow/job,
event/checkout proof, and inspected file hashes. No workflow or CodeQL query was
weakened to produce these dispositions.

## Cache poisoning: 240 false positives

Five direct-cache alerts and 235 poisonable-step alerts reduce to 52 distinct
checkout/event combinations. Every reported cache-write event is
`workflow_dispatch` or `schedule`. In those contexts, neither the
`pull_request` nor `workflow_run` event object exists. Where a checkout also uses
`needs.admission.outputs.head_sha`, the admission job has an exact conjunction
requiring `workflow_run`, successful completion, and the enabled license gate;
it is skipped for both reported cache-write events. Every checkout therefore
resolves to the event's `github.sha`. None takes a dispatched input as a revision.
All reviewed checkouts disable persisted credentials.

The query identifies a PR field anywhere in the checkout expression, then joins
it with any cache-writing trigger on the job. That loses the event correlation
in these fallback expressions. The pinned implementation explicitly excludes
`workflow_run` from default-branch cache-write events. This is a finding about
an impossible combination of event and revision source, rather than proof that
all workflow execution is universally trusted.

References: [CodeQL cache-write event implementation](https://github.com/github/codeql/blob/eb3ddb87306141a681e11b3cca65b653231bb3f2/actions/ql/lib/codeql/actions/security/CachePoisoningQuery.qll),
[GitHub event contexts](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows),
[cache access restrictions](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching).

Validation: `test_codeql_cache_event_boundaries.py` has **59 passing cases**,
including all 52 combinations and negative controls that reject dispatched
revision inputs, missing event-commit fallbacks, or an unguarded admission job.
The test evaluates only the reviewed OR-chain grammar and rejects unknown forms;
it is not an interpreter for arbitrary GitHub expressions.

## Untrusted checkout: alert 2359 is a false positive

`frontend-license-gate.yml` checks out the trusted base `github.sha` with no
persisted credentials. Its evaluator fetches PR objects into a separate remote
ref, compares both fetched revisions with immutable event SHAs, and passes only
NUL-delimited changed filenames to the trusted classifier. Fetch does not change
HEAD, the index, or the checked-out classifier. The diff disables external diff
programs and text conversion. No PR file is executed or checked out.

The [pinned query heuristic](https://github.com/github/codeql/blob/eb3ddb87306141a681e11b3cca65b653231bb3f2/actions/ql/lib/codeql/actions/security/UntrustedCheckoutQuery.qll)
classifies `git fetch` and `git pull` with PR metadata as checkouts. Here the
fetch is a data-only operation and the subsequent sink consumes names, not code.

Validation: `test_license_gate_fetch_isolation.py` executes the unmodified
workflow evaluator with a local Git remote. An external PR that replaces the
classifier with code that writes an attacker marker is rejected; the marker is
absent and HEAD, classifier bytes, and worktree remain unchanged. Wrong base and
head SHAs also fail closed; an allowed documentation change succeeds. Together
with the existing workflow surface contracts, **17 tests pass in Linux Bash 5**.
The repository mount was read-only. The local remote is selected through Git's
test-only URL rewrite; no production network or credentials are used.

macOS Bash 3.2 ignores `errexit` on the workflow's `[[ ... ]]` guards. The test
therefore requires Bash >=4 rather than rewriting the workflow or claiming that
macOS execution matches the Ubuntu runner. All four behavioral cases were
executed successfully in an ephemeral Linux container. Logs:
`/tmp/pr2761-actions-linux-isolation.log` and
`/tmp/pr2761-actions-cache-contracts.log`.

Bandit subprocess warnings in the integration fixture are annotated individually:
fixed executable paths, literal version probe, and the trusted base script are
intentional. This adds no production suppression. Pytest assertions are excluded
from test-only Bandit inspection in the usual way.

Final combined Linux run: **76 passed**, including all new event/checkout and
hostile-ref cases plus existing workflow contracts; log
`/tmp/pr2761-actions-linux-final.log`. Independent review reconciled all 241
IDs and 52 proofs against SARIF, without mapping collisions. The only shared
open main-branch Actions instance is 2359: its fetch, diff, trusted checkout and
classifier boundary are the same. The candidate separates `readonly` from two
command substitutions, which does not turn main's fetch into code execution.
Other shared instances belong to already-merged PRs 2618, 2691 and 2745.
