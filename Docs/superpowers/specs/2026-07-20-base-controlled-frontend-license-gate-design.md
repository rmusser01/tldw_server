# Base-Controlled Frontend License Gate Design

- **Status:** Approved by Robert Benjamin Jake Musser on 2026-07-20
- **Date:** 2026-07-20
- **Backlog task:** TASK-12977
- **Related execution task:** TASK-12976

## Problem

The first implementation of the temporary contribution freeze placed a
classifier step in `.github/workflows/frontend-required.yml`, a workflow
triggered by `pull_request`. That does not form a trust boundary: the pull
request can change the workflow that is supposed to inspect it. Loading the
Python classifier from the base revision is insufficient when the pull request
can delete or replace the shell step that loads it.

The original line-oriented `git diff --name-only` pipe also C-quotes or splits
unusual filenames. A protected path containing Unicode, tabs, or newlines could
therefore be classified incorrectly.

The live repository state confirms that enforcement cannot be delegated to an
existing branch rule:

- `dev` has no branch protection or required status checks.
- Active ruleset `5653432` (`safee`) targets only the default branch and
  requires a pull request, deletion protection, and non-fast-forward
  protection; it has no required status checks.
- `rmusser01` is the only direct collaborator and has the `admin` role.

## Goals

1. Run the temporary licensing policy from code controlled by the default
   branch, not by the pull request.
2. Never check out or execute pull-request code in the privileged workflow.
3. Evaluate exact old and new paths for every change, including adversarial
   filenames and renames.
4. Allow Robert Benjamin Jake Musser's changes while blocking other authors'
   changes to protected frontend, license-governance, and conservative API
   declaration paths.
5. Make the trusted result required on both `dev` and `main` without weakening
   existing repository rules.
6. Bootstrap safely through a small `main` PR before the licensing cutoff PR is
   merged to `dev`.

## Non-Goals

- This gate does not replace the later contributor agreement or counsel review.
- It does not run frontend tests or replace `frontend-required`.
- It does not execute, build, import, or check out pull-request code.
- It does not grant commercial, community-fork, customer, trademark, or patent
  rights.
- It is not a general policy engine or a new hosted service.

## Alternatives Considered

### 1. Base-controlled workflow plus trusted commit status — selected

Use `pull_request_target` from `main`, evaluate only Git metadata, and publish a
dedicated commit status with a write-capable token that fork pull-request
workflows do not receive. Require that status through repository rulesets.

This is the smallest approach that keeps the policy code outside pull-request
control and works for the repository's current single-admin model.

### 2. Dedicated GitHub App

A separate GitHub App could evaluate the policy and provide an expected-source
check that repository workflows cannot impersonate. This is the strongest
long-term boundary but requires a new application, credentials, deployment,
monitoring, and operational ownership. It is disproportionate for the
two-week pre-counsel freeze.

### 3. Manual review or CODEOWNERS only

Manual refusal of protected changes is legally workable but does not provide
the requested automatic fail-closed signal. CODEOWNERS also depends on branch
review configuration and does not by itself distinguish the pull-request
author at merge time.

## Trust Model

The trusted inputs are:

- the `pull_request_target` workflow file from the default branch;
- the classifier from the exact default-branch commit used by that run;
- immutable pull-request metadata supplied by GitHub;
- Git object names fetched from the base repository's pull-request ref; and
- the default-branch workflow's `GITHUB_TOKEN` with explicitly limited
  permissions.

The untrusted inputs are:

- every file and Git object supplied by the pull-request head;
- the pull-request author login and branch name;
- filenames, including control characters and Unicode; and
- workflow or classifier changes proposed by the pull request.

The workflow may fetch untrusted Git objects for metadata comparison, but it
must never check out the pull-request head, run its actions, import its Python,
load its configuration, use its caches, or execute its build commands.

The temporary status boundary assumes that no third party has write access to
the base repository. That matches the verified 2026-07-20 collaborator list.
Before granting any non-owner write role, replace this status publisher with a
dedicated GitHub App or another identity-isolated policy service.

## Components

### Trusted workflow

Create `.github/workflows/frontend-license-gate.yml` on `main` with:

- trigger `pull_request_target` for pull requests targeting `main` or `dev`;
- workflow permissions limited to `contents: read` and `statuses: write`;
- no secrets other than the ephemeral `GITHUB_TOKEN`;
- no cache, artifact download, dependency install, or pull-request checkout;
- a pinned `actions/checkout` revision checking out `${{ github.sha }}`, which
  is the trusted default-branch revision for this event;
- `persist-credentials: false`; and
- one audit job named `frontend-license-gate-audit`.

The job's ordinary Actions check is diagnostic only. The ruleset trust signal
is the commit-status context `frontend-license-policy/trusted` posted to the
pull-request head SHA through the base repository's statuses API.

Using a commit status rather than requiring the ordinary job check matters:
a fork pull request can declare a GitHub Actions job with the same check name,
but its read-only token cannot write a commit status in the base repository.
If a check and status share a name, GitHub requires both, so the trusted status
cannot be replaced by a same-named pull-request check. The workflow and status
names will nevertheless remain distinct to avoid ambiguity.

### Classifier

Keep `Helper_Scripts/ci/check_frontend_license_gate.py` as a standard-library
module with these interfaces:

```python
def blocked_changes(paths: Iterable[str]) -> list[str]: ...

def evaluate(*, author: str, owner: str, paths: Iterable[str]) -> list[str]: ...

def read_nul_paths(stream: BinaryIO, *, max_bytes: int = 8 * 1024 * 1024) -> list[str]: ...
```

`evaluate` allows the owner case-insensitively. Other authors are blocked for:

- `admin-ui/`
- `apps/tldw-frontend/`
- `apps/extension/`
- `apps/packages/ui/`
- `LICENSES/`
- `LICENSE`
- `THIRD_PARTY_NOTICES.txt`
- `Helper_Scripts/ci/check_frontend_license_gate.py`
- `.github/workflows/frontend-license-gate.yml`
- `.github/workflows/frontend-required.yml`
- `tldw_Server_API/app/main.py`
- `tldw_Server_API/app/api/v1/`

The CLI accepts a `--null` mode and reads raw bytes from `sys.stdin.buffer`.
It reads at most 8 MiB plus one sentinel byte, fails closed if the limit is
exceeded, splits only on NUL, decodes with UTF-8 plus `surrogateescape`, and
does not trim whitespace. Diagnostic filenames are escaped with `ascii()` or
an equivalent control-character-safe representation before logging.

### Exact changed-path collection

The workflow validates the event's base SHA, head SHA, and pull-request number
before passing them to Git. It fetches the pull-request head ref without
checking it out and verifies that the fetched commit equals the event head SHA.

It then runs the equivalent of:

```bash
git --no-pager diff \
  --name-only \
  -z \
  --no-renames \
  --no-ext-diff \
  --no-textconv \
  "${BASE_SHA}" "${HEAD_SHA}" --
```

`--no-renames` deliberately reports a rename as a deletion plus an addition,
so moving a protected file out of a protected directory cannot hide the old
path. `-z` preserves every permitted filename byte except NUL, which Git itself
does not allow in a path.

## Status Publication and Failure Handling

The workflow posts `frontend-license-policy/trusted` as `pending` before
evaluation. It posts `success` only for an explicit allow result. A blocked
path, SHA mismatch, fetch failure, missing trusted classifier, malformed input,
oversized path stream, API failure, or unexpected exception results in
`failure` or leaves the status pending. Both outcomes block the ruleset.

The final status-publishing step may use `if: always()` only to publish the
already-computed result. It must not fetch or execute source. Missing or unknown
evaluation output maps to failure, never success.

Owner-authored changes may short-circuit path fetching after the trusted
classifier confirms the case-insensitive owner match. External unrelated
changes succeed. External protected or governance changes fail with escaped
path diagnostics.

## Rollout

### Phase 1: Bootstrap `main`

Create a small PR to `main` containing only:

- the trusted workflow;
- the hardened classifier and tests;
- workflow contract tests;
- this design and its implementation plan; and
- TASK-12977 tracking updates.

The new `pull_request_target` workflow cannot protect the PR that first adds it
because the default branch does not contain it yet. The existing `main`
ruleset still requires a PR. Repository policy also requires Robert to write
the AI-generated PR's `Change summary` in his own words before merge.

### Phase 2: Observe the trusted status

After the bootstrap merges to `main`, open or synchronize the licensing cutoff
PR targeting `dev`. Verify that:

- the workflow run source is the default branch;
- the trusted status is posted to the current head SHA;
- the owner-authored PR succeeds; and
- the committed event/workflow harness proves that an external author with a
  protected path fails without executing head code.

Required statuses must have completed recently before GitHub permits selecting
them as ruleset requirements.

### Phase 3: Activate rulesets

Preserve ruleset `5653432` exactly and add a required-status-check rule for
`frontend-license-policy/trusted`, bound to the observed GitHub Actions source,
without changing its existing pull-request, deletion, or non-fast-forward
rules.

Create a separate active `dev` ruleset targeting only `refs/heads/dev`. It
requires a pull request and the same trusted status, with no bypass actors.
Record the complete before/after JSON and ruleset IDs in TASK-12977.

If GitHub cannot bind the commit status to the expected source, keep the
ruleset disabled and stop. Do not silently fall back to `any source` without a
new user decision.

### Phase 4: Resume the licensing cutoff

Replace the rejected PR-controlled gate commit in the `dev` licensing branch
with the NUL-safe classifier contract expected by the trusted workflow. Then
complete Tasks 5 and 6 of TASK-12976 and open the license-only PR to `dev`.

## Testing

Classifier tests cover:

- owner casefolding;
- all four protected prefixes;
- governance and API boundaries;
- unrelated and near-prefix paths;
- leading/trailing whitespace;
- Unicode, tabs, and newlines;
- rename behavior through old/new path pairs;
- NUL parsing without trimming;
- oversized input failure; and
- escaped diagnostics.

Workflow contract tests cover:

- `pull_request_target` on `main` and `dev`;
- exact permissions;
- pinned trusted checkout at `github.sha`;
- absence of pull-request checkout, caches, installs, or head-code execution;
- validated event values;
- NUL-delimited, no-rename, no-external-diff collection;
- pending then fail-closed final status publication;
- distinct audit-job and trusted-status names; and
- actionlint coverage.

Live verification records the workflow run URL, check/status source, head SHA,
ruleset JSON, and owner allow result. The external protected-path deny result
is recorded from the deterministic event/workflow harness; the first real
external protected-path PR is also audited when one occurs, but a second
GitHub identity is not an activation prerequisite.

## Rollback

If the trusted workflow malfunctions after activation:

1. disable, but do not delete, the new `dev` ruleset;
2. remove only the added required-status rule from `5653432`, preserving its
   other rules byte-for-byte;
3. record the before/after API responses in TASK-12977;
4. keep the contribution freeze documented and enforce it manually; and
5. do not resume protected contribution intake until a corrected trusted gate
   is reviewed and active.

## Accepted Limitations

- The first bootstrap PR is protected by human review and the existing main
  pull-request rule, not by the workflow it introduces.
- The temporary status publisher relies on the verified fact that Robert is
  the only direct collaborator. Adding another write-capable collaborator is a
  stop condition.
- GitHub administrators retain ultimate control over workflows and rulesets.
- This gate prevents accidental acceptance through configured repository
  controls; it cannot prevent an administrator from intentionally disabling
  those controls.

## References

- GitHub Actions events: https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows
- Secure use reference: https://docs.github.com/en/actions/reference/security/secure-use
- Ruleset rules: https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets
- Required-status troubleshooting: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-quality-features/troubleshooting-required-status-checks
