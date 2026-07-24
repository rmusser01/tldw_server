# License-First PR CI with `workflow_run`

- **Status:** Requester approved the strict license-first design and independent
  specification review approved it on 2026-07-24; written specification pending
  final requester review
- **Backlog task:** TASK-12986
- **Repository:** `rmusser01/tldw_server`

## Problem

The trusted frontend-license audit and 28 ordinary pull-request workflows are
currently triggered independently. GitHub provides no priority or cross-
workflow `needs` setting, so ordinary CI can occupy hosted runners before the
required license audit starts.

The required behavior is strict ordering:

1. a pull-request activity starts the trusted license audit;
2. no ordinary PR CI job starts while that audit is queued or running; and
3. ordinary PR CI is released only after that exact audit succeeds.

An earlier polling design was rejected because waiters could consume the runner
pool ahead of the audit. A later status-bypass design was a misunderstanding of
the goal and will not be implemented. No live ruleset was changed.

## GitHub Constraints

- `workflow_run` fires only after the named upstream workflow reaches the
  requested activity state.
- A completed event is emitted for every conclusion, so downstream jobs must
  explicitly require `conclusion == "success"`.
- A `workflow_run` workflow executes from the repository default branch and
  can receive a write-capable token and secrets even when the upstream workflow
  did not.
- `workflow_run` has no native `paths` or `paths-ignore` filter and always
  executes the workflow definition from the default `main` branch.
- Live successful license runs normally include one associated pull request
  with its number, base, repository IDs, and exact head SHA. Some historical
  run records have an empty pull-request list, so absence or ambiguity must
  fail closed.

## Goals

1. Make the trusted license audit the only workflow directly triggered by
   `main` and `dev` pull-request activity.
2. Start ordinary PR CI only after a successful completion of
   `Frontend License Gate Audit`.
3. Bind every downstream run to the exact audited PR head and current base
   branch/repository identity without treating normal base-branch advancement
   as a stale-head failure.
4. Preserve existing non-PR triggers and workflow-specific path selection.
5. Prevent the privileged `workflow_run` context from exposing secrets, write
   authority, or trusted caches to untrusted PR code.
6. Use no PAT, GitHub App, label, manually forged status, or polling waiter.

## Non-Goals

- Do not change the required status contexts or repository rulesets.
- Do not make informational CI results required for merging.
- Do not execute PR-controlled workflow definitions.
- Do not combine all CI into one monolithic workflow.
- Do not prioritize jobs after the license audit succeeds.
- Do not preserve a check-run identity if GitHub inherently changes it when
  the trigger moves from `pull_request` to `workflow_run`; document observed
  UI behavior during rollout instead.

## Selected Architecture

### Trusted license workflow

Keep `.github/workflows/frontend-license-gate.yml` source-controlled,
`pull_request_target`-triggered, and source-bound exactly as it is. It remains
the only direct `main`/`dev` PR trigger and the only publisher of:

- `frontend-license-policy/trusted/main`; and
- `frontend-license-policy/trusted/dev`.

The trusted workflow does not check out or execute PR code and does not call
the downstream workflows.

### Shared admission workflow

Add one reusable workflow callable through `workflow_call`. Each ordinary CI
workflow invokes it as its first job when the caller event is `workflow_run`.
The admission workflow owns no write permission, accepts no secrets, and
checks out only trusted default-branch helper code.

It validates:

1. caller event name is `workflow_run`;
2. `github.event.workflow_run.name`, `.path`, and `.workflow_id` identify the
   live default-branch workflow named `Frontend License Gate Audit` at
   `.github/workflows/frontend-license-gate.yml`;
3. upstream event is `pull_request_target`;
4. exactly one entry exists at
   `github.event.workflow_run.pull_requests[0]`;
5. that entry's `number`, `head.sha`, `head.repo.id`, `base.sha`, `base.ref`,
   and `base.repo.id` are well formed;
6. the current Pulls API response has the same PR number, head SHA, head
   repository ID, base ref, and base repository ID;
7. current PR state is open and base ref is exactly `main` or `dev`; and
8. the exact branch-qualified trusted status is successful on that head and
   was created no earlier than the upstream run's `run_started_at`.

The current Pulls API base SHA is intentionally not required to equal the
audited event's base SHA. The target branch can advance without a PR
`synchronize` event; failing admission on that routine movement would suppress
CI with no guaranteed event to release it. The audited base SHA remains the
immutable diff/test baseline for that run, while current base ref and repository
identity still detect retargeting.

When the current base SHA differs from the audited base SHA, admission remains
valid but path-based skipping is disabled. Every path-filtered workflow receives
`should_run == "true"` because the current Pull Files API response is no longer
guaranteed to describe the same diff as the audited base/head pair.

Admission returns immutable outputs for PR number, head SHA, audited base SHA,
base ref, and a workflow-specific `should_run` path decision. Missing, stale,
ambiguous, or malformed metadata fails closed before PR code is checked out.

For existing non-PR triggers, callers bypass PR admission and keep their
current event semantics.

Every caller grants the admission job only `actions: read`, `contents: read`,
`pull-requests: read`, and `statuses: read`. It does not use
`secrets: inherit`.

### Ordinary workflow triggers

For every ordinary workflow currently using `pull_request` on `main` or `dev`:

- remove `main` and `dev` from the direct `pull_request` trigger;
- preserve direct PR behavior for every unsupported base:
  - CodeQL retains `pull_request.branches: [master]`;
  - formerly unrestricted PR workflows use
    `pull_request.branches-ignore: [main, dev]`; and
  - each residual trigger retains its original activity types and path filters;
- add:

  ```yaml
  workflow_run:
    workflows: [Frontend License Gate Audit]
    types: [completed]
  ```

- retain existing `push`, `release`, `schedule`, and `workflow_dispatch`
  triggers;
- add the shared admission job with this server-evaluated condition:

  ```yaml
  if: >-
    vars.LICENSE_FIRST_CI_ENABLED == 'true' &&
    github.event_name == 'workflow_run' &&
    github.event.workflow_run.conclusion == 'success'
  ```

- make every PR-path root job require successful admission and
  `should_run == "true"`;
- preserve existing downstream dependency graphs; and
- make every `always()` rollup depend directly on admission and reject a
  failed, skipped, or unauthorized admission.

The trigger itself creates a workflow record for unsuccessful upstream runs,
but the admission call is skipped before runner allocation and no ordinary
runner-owning job may start. An unset, false, or differently valued cutover
variable also skips admission. Every runner-owning root and rollup condition
includes `!cancelled()`; where dependency results must be inspected it uses
`always() && !cancelled()` to distinguish:

- admitted `workflow_run`: require `needs.admission.result == "success"` and
  the expected admission output;
- rejected or unsuccessful `workflow_run`: skip; and
- an existing non-PR or unsupported-base PR event: accept the intentionally
  skipped admission and preserve the original job condition.

A truth-table contract covers enabled/disabled cutover state plus successful,
failed, cancelled, skipped, malformed and non-PR cases so a skipped admission
cannot accidentally skip or release the wrong execution path. It also proves
that cancellation after admission stops stale runner-owning jobs.

### PR context translation

`workflow_run` does not populate normal `github.event.pull_request`,
`github.head_ref`, or `github.base_ref` values for downstream jobs. Replace
every PR-specific use with admission outputs or the validated upstream payload.

In particular:

- existing workflow-level concurrency remains workflow-level and uses the
  GitHub-owned pre-admission expression:

  ```yaml
  ${{ github.event.workflow_run.pull_requests[0].number ||
      github.event.pull_request.number || github.ref || github.run_id }}
  ```

  Admission later validates the same workflow-run PR number, and malformed or
  empty associations cannot release a job;
- every checkout of PR code uses the immutable admitted head SHA;
- base-diff logic uses the admitted audited base SHA;
- PR/non-PR conditionals treat an admitted `workflow_run` as the PR execution
  mode; and
- no bare `actions/checkout` in the admitted path may implicitly check out the
  default branch when PR code was intended.

All existing expressions referencing pull-request event data must be
enumerated by a static contract test. Untranslated references are a failure.

### Path-filter compatibility

Eighteen ordinary workflows currently use `paths` or `paths-ignore`, which
`workflow_run` cannot express for `main`/`dev`. Preserve their include/exclude
intent in one reviewed routing manifest keyed by workflow filename while
retaining the original filters unchanged on residual direct PR triggers for
unsupported bases.

The admission helper reads the current PR file list through the API and applies
the corresponding ordered include/exclude patterns. Its matcher implements
GitHub-compatible slash-aware `*`, recursive `**`, `?`, character classes, and
ordered `!` re-inclusion/exclusion semantics. It follows pagination completely
but does not claim that Pull Files API ordering is identical to GitHub Actions'
internal path-filter ordering.

Path filtering is an execution optimization, not an authorization boundary.
GitHub's current canonical workflow-syntax and troubleshooting pages disagree
about whether internal path evaluation stops at 3,000 or 300 files, while the
Pull Files API itself caps responses at 3,000. Therefore the router preserves
normal-case selection but makes no byte-for-byte equivalence claim. If more
than 300 files changed, enumeration is incomplete, ordering matters beyond
enumerated data, pattern syntax is unsupported, or equivalence is otherwise
uncertain, admission sets `should_run == "true"` so CI runs rather than silently
skipping a relevant workflow. A current base SHA that differs from the audited
base SHA is also uncertain: admission skips path evaluation and sets
`should_run == "true"` for every formerly path-filtered workflow.

Contract tests compare the routing manifest with the pre-cutover trigger
filters so the migration cannot silently narrow a workflow. Tests cover
`paths-ignore`, ordered negation, rename old/new names, deletion paths, empty
diffs, pagination, the conservative greater-than-300-file fallback, and the
3,000-file API ceiling. A workflow without a prior path filter receives
`should_run == "true"`.

### Privilege and cache boundary

Every downstream job declares explicit minimum permissions. No admitted PR job
receives or references a secret, uses `secrets: inherit`, or receives
`contents: write`, `actions: write`, `packages: write`, `id-token: write`, or
`security-events: write`.

Every checkout of admitted PR code sets `persist-credentials: false`. CodeQL
still analyzes the admitted head but disables SARIF upload for this PR path;
trusted `push` and scheduled CodeQL runs retain their existing
`security-events: write` upload behavior.

Because `workflow_run` shares privileged default-branch cache scope, admitted
PR jobs may restore only explicitly safe caches and may not save caches.
Replace `actions/cache` with `actions/cache/restore` on the admitted path and
disable automatic setup-action cache writes for admitted PR runs. Trusted
default-branch `push` jobs remain responsible for populating caches.

No admitted workflow references repository or environment secrets. The
contract rejects future secret references, credential-persisting checkouts, or
write permissions on the admitted path.

## Data Flow

1. A PR activity targets `main` or `dev`.
2. Only `Frontend License Gate Audit` starts.
3. The audit publishes `pending`, evaluates trusted metadata, and publishes its
   exact branch-qualified terminal status.
4. GitHub emits a completed `workflow_run` event.
5. Each ordinary workflow receives that event. For any non-success conclusion,
   its admission job is skipped before runner allocation.
6. On success, admission re-reads the current PR, proves the audited SHA is
   still current, and applies the workflow's original path filter.
7. Relevant workflows check out and test only the admitted immutable head SHA.
8. A later synchronize event starts a new license audit. Existing ordinary runs
   remain permanently bound to the previously audited immutable head and cannot
   produce results for the new head. New-head CI is not released until the new
   head's own audit succeeds; when released, existing workflow-specific
   PR-number concurrency supersedes older work.

The ordering invariant is head-scoped: no ordinary job for head `H` may start
until the trusted audit for `H` succeeds. A job already admitted for old head
`H1` may finish while new head `H2` is being audited, but it continues to use
only `H1`, and its result cannot satisfy or represent `H2`.

## Failure Handling

- **License queued/running:** no ordinary workflow-run event exists yet.
- **License failure, error, cancellation, or timeout:** the server-side caller
  condition skips admission, and downstream workflow records contain no
  eligible runner-owning jobs.
- **Cutover variable unset or not exactly `true`:** skip workflow-run admission
  before runner allocation while residual direct/non-PR triggers keep their
  explicitly preserved behavior.
- **Missing or multiple PR associations:** fail admission.
- **Closed, retargeted, or changed-head PR:** fail admission. Normal movement
  of the same base branch does not fail admission, but it disables path-based
  skipping.
- **Status missing or not successful:** fail admission.
- **Path API or matcher uncertainty:** run the workflow rather than risk a
  false skip.
- **Admission failure:** all ordinary jobs remain skipped; rollups must not turn
  the skip into success.
- **New PR head after admission:** jobs continue only on the immutable old SHA;
  the new head gets a separate audit and admission. Old results are not treated
  as results for the new head, and no new-head job starts before new-head
  admission.

## Testing

Use test-first contract changes to prove:

- the trusted gate remains the only direct `main`/`dev` PR trigger;
- every migrated workflow names only the trusted upstream workflow and
  `completed` activity;
- no ordinary job can run for a non-success conclusion;
- a truth table covers enabled/disabled cutover state, preserves non-PR and
  unsupported-base PR behavior, and rejects failed, cancelled, skipped,
  malformed, or concurrency-cancelled workflow runs;
- admission validates exact workflow name/path/ID, one current open PR, the
  exact current head identity, current base branch/repository identity, the
  audited base SHA's shape, and a fresh branch-qualified status success;
- base-branch advancement without retargeting remains admissible and forces
  every path-filtered workflow to run, while a changed head, base ref, or base
  repository fails closed;
- missing, stale, retargeted, closed, and ambiguous payloads fail closed;
- the path manifest preserves every old path filter's normal-case behavior and
  fails open for uncertain ordering, more than 300 files, incomplete
  pagination, or the 3,000-file API ceiling;
- pre/post direct PR base coverage is identical outside `main`/`dev`;
- existing non-PR triggers remain unchanged;
- every PR-event expression and checkout is translated;
- every workflow-level concurrency expression uses only event-time context and
  its workflow-run PR number is corroborated by admission;
- every runner-owning job has a successful admission dependency;
- `always()` jobs also require `!cancelled()` and reject unsuccessful
  admission;
- admitted paths contain no secret reference, inherited secret,
  credential-persisting checkout, or forbidden permission;
- admitted paths cannot save caches; and
- the actionlint contract covers every touched workflow.

Run focused pytest contract suites, matcher unit/property tests, actionlint,
Bandit on the admission helper, YAML parsing, and `git diff --check`.

## Rollout

`workflow_run` workflows must exist on the default branch, so the default
`main` branch is the activation boundary.

1. Re-read the live rulesets and prove no required context depends on any of the
   28 ordinary workflow checks.
2. Declare `main` the canonical CI-definition branch for both `main`- and
   `dev`-targeted PRs after cutover. Before canarying, make every migrated
   workflow, admission helper, and routing manifest byte-identical on `main`
   and `dev`, and add a contract that blocks cutover on drift. After cutover,
   CI-definition changes land on `main` first; `dev` may not carry an
   ahead-of-`main` CI definition that its PRs cannot execute.
3. Land an inert preparation state on both branches: the admission workflow,
   helper, routing manifest, translated event logic, and dual-trigger-capable
   ordinary workflows. Keep existing direct PR triggers active and guard the
   `workflow_run` path with repository variable
   `LICENSE_FIRST_CI_ENABLED` unset or not equal to `true`.
4. Briefly enable that variable for controlled canary PRs, then disable it
   again. Record:
   - license run ID, conclusion, and completion time;
   - admission and ordinary job start times;
   - exact head SHA/base observed by each;
   - path-filter behavior; and
   - downstream `workflow_sha`, check-suite `head_sha`, PR UI association,
     cache behavior, secret access, and effective permissions.
5. Require the canary to prove ordering and acceptable check visibility before
   the mass cutover.
6. Establish a short `main`/`dev` CI-definition and author-activity freeze,
   verify no preparation drift, and enable `LICENSE_FIRST_CI_ENABLED` **before**
   removing either branch's direct PR triggers. This creates a temporary
   duplicate-CI window but no event-loss window.
7. While the freeze remains active, generate a fresh supported PR activity for
   every open `main`/`dev` PR whose current head's license event completed
   before the variable was enabled. Use a controlled `edited` activity, such as
   a recorded title/body marker, and wait until its fresh license event produces
   a successful admitted downstream run. Do **not** use the Actions rerun API:
   a rerun retains the old event payload and therefore cannot establish a fresh
   audited base after the target branch advances.
8. Remove the direct `main`/`dev` PR triggers on `dev`, then land the same
   cutover on `main`. The `workflow_run` definitions and reusable workflow now
   execute canonically from `main`.
9. Restore any controlled PR metadata marker, which emits another fresh
   supported `edited` activity, and wait for every open PR's resulting license
   and admitted downstream runs. This ensures no head depends on an event
   emitted during the transition. Then end the freeze.
10. Trigger one post-cutover canary per base and prove every ordinary
   runner-owning job starts after the matching successful license completion.

Strict ordering is not claimed during the preparation/canary stage because
direct triggers still exist. During the final frozen cutover, a missing
required license status, unadmitted PR-code execution, or ordinary job starting
before license success stops rollout and triggers rollback.

## Rollback

Restore the previous direct `pull_request` triggers, event expressions,
checkouts, and cache behavior from the saved pre-cutover source state on
`main` and `dev` **while the repository variable remains enabled**. This
temporarily permits duplicate CI but prevents a no-trigger window. After both
branches contain working direct triggers, rerun or explicitly supersede any
open-PR head that lacks ordinary CI, then disable the variable. Remove the
admission workflow, path manifest, helper, and variable only after direct PR CI
is restored and open PRs have a current execution path.

The trusted license workflow and source-bound required statuses remain
unchanged throughout rollback.
