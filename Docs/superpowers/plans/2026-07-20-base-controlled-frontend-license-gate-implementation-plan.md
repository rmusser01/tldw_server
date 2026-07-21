# Base-Controlled Frontend License Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to execute this plan task by task. Use
> `superpowers:test-driven-development` for Tasks 2 and 3,
> `superpowers:requesting-code-review` after each implementation task, and
> `superpowers:verification-before-completion` before any completion claim.

**Goal:** Bootstrap a default-branch-controlled, metadata-only frontend
licensing gate on `main`; publish a source-bound trusted status for pull
requests to `main` and `dev`; activate fail-closed rulesets; then replace the
rejected PR-controlled gate on the licensing branch.

**Architecture:** A `pull_request_target` workflow stored on `main` checks out
only `${{ github.sha }}`, reads GitHub-supplied PR metadata, fetches base and PR
Git objects without checking out the head, streams `git diff --name-only -z
--no-renames` into a standard-library classifier, and publishes the distinct
commit-status context `frontend-license-policy/trusted`. Rulesets require that
context from the observed GitHub Actions App integration. The workflow never
imports, builds, caches, or executes pull-request content.

**Tech Stack:** Git, GitHub Actions, GitHub REST API through `gh`, Python 3,
pytest, PyYAML, actionlint, Bandit, Backlog.md CLI/MCP.

**Backlog task:** `TASK-12977`

**Approved design:**
`Docs/superpowers/specs/2026-07-20-base-controlled-frontend-license-gate-design.md`

## Global Constraints

- Execute the bootstrap in a new worktree based on current `origin/main`; do
  not perform it directly in the licensing worktree or the user's dirty root
  worktree.
- Keep the bootstrap PR limited to the trusted workflow, classifier, focused
  tests, actionlint target, approved design/plan, and TASK-12977 records.
- Never check out the PR head, run a local action from the PR, install its
  dependencies, restore its caches, or execute any file from it in the
  privileged workflow.
- Treat PR metadata, refs, SHAs, author login, and filenames as untrusted until
  validated. A validation, fetch, diff, classifier, or API error must never
  publish success.
- Keep `frontend-license-gate-audit` and
  `frontend-license-policy/trusted` distinct. Only the commit status is a
  ruleset requirement.
- Do not activate any required-status rule until the workflow is merged to
  `main`, a real trusted status has been observed, and its source integration
  has been identified.
- Bind every required status to the observed integration ID. If GitHub rejects
  or drops the binding, roll back/leave disabled and stop; do not use an
  any-source requirement without a new user decision.
- Re-read the direct collaborator list immediately before activation. If any
  write-capable person other than `rmusser01` exists, stop and migrate the
  publisher to a dedicated GitHub App before activation.
- Preserve all existing rules, conditions, enforcement, and bypass actors in
  main ruleset `5653432`; append only the new required-status rule.
- Robert Benjamin Jake Musser must write the bootstrap PR's `Change summary`
  in his own words before merge. Agent-generated text cannot satisfy that
  repository policy.
- Do not touch the unrelated untracked watchlist template files in the
  licensing worktree.

## Stage Tracking

| Stage | Goal | Success criteria | Tests | Status |
|---|---|---|---|---|
| 1 | Prepare isolated `main` bootstrap | Exact current refs/state captured; clean bootstrap worktree contains the approved design, plan, and task | Baseline focused CI-policy tests | Not Started |
| 2 | Harden the classifier | Exact protected boundaries and NUL-safe parsing fail closed | Focused pytest + Bandit | Not Started |
| 3 | Add the trusted workflow | Base-controlled workflow publishes only a trusted, fail-closed status and passes contract linting | Workflow contract pytest + actionlint | Not Started |
| 4 | Land and activate | Human-reviewed bootstrap merges, trusted status is observed, and source-bound rules protect `main` and `dev` | Live run/status/ruleset evidence | Not Started |
| 5 | Reconcile licensing branch | Rejected PR-controlled gate is replaced and TASK-12976 resumes on the trusted contract | Focused policy/workflow tests + Bandit + diff review | Not Started |

## Task 1: Prepare the `main` Bootstrap Worktree

**Files:**

- Create worktree:
  `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/frontend-license-gate-bootstrap`
- Carry:
  `Docs/superpowers/specs/2026-07-20-base-controlled-frontend-license-gate-design.md`
- Carry:
  `Docs/superpowers/plans/2026-07-20-base-controlled-frontend-license-gate-implementation-plan.md`
- Carry:
  `backlog/tasks/task-12977 - Bootstrap-base-controlled-frontend-license-gate.md`

### Step 1: Re-read the worktree skill and capture immutable starting state

From the licensing worktree:

```bash
git fetch origin main dev
git rev-parse origin/main
git rev-parse origin/dev
gh pr view 2727 --repo rmusser01/tldw_server \
  --json number,state,isDraft,baseRefName,headRefName,headRefOid,url
gh api repos/rmusser01/tldw_server/collaborators?affiliation=direct \
  --jq '.[] | {login, role_name, permissions}'
gh api repos/rmusser01/tldw_server/rulesets/5653432
```

Expected preconditions:

- PR `#2727` remains open and draft.
- `rmusser01` is the only direct collaborator.
- ruleset `5653432` remains active on the default branch and has no
  `required_status_checks` rule.

If any precondition has changed, update the design/task evidence and reassess
before creating or mutating branches.

### Step 2: Create the isolated bootstrap branch

Use `superpowers:using-git-worktrees`, verify the target directory does not
already exist, then run:

```bash
git worktree add \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/frontend-license-gate-bootstrap \
  -b codex/frontend-license-gate-bootstrap origin/main
```

In the bootstrap worktree, carry the approved records without bringing over
the licensing implementation:

```bash
git cherry-pick eb69feeea4
git checkout codex/frontend-licensing-cutoff -- \
  Docs/superpowers/specs/2026-07-20-base-controlled-frontend-license-gate-design.md \
  Docs/superpowers/plans/2026-07-20-base-controlled-frontend-license-gate-implementation-plan.md \
  'backlog/tasks/task-12977 - Bootstrap-base-controlled-frontend-license-gate.md'
```

The checkout is deliberately path-scoped. Confirm the branch contains no
frontend license corpus or package metadata changes from TASK-12976:

```bash
git status --short
git diff --stat origin/main...HEAD
git diff --name-only origin/main...HEAD
```

### Step 3: Establish a baseline

Activate the project environment before Python commands:

```bash
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tldw_Server_API/tests/CI/test_frontend_required_workflow.py
```

If `.venv` is not present in the new worktree, use the repository's existing
shared environment only after verifying its interpreter points to the same
project dependencies; do not install or rewrite dependencies as part of this
bootstrap.

### Step 4: Update tracking and commit carried records

Use the Backlog.md MCP/CLI to set the Stage 1 plan status to `Complete`, record
the exact `origin/main`, `origin/dev`, PR head, collaborator result, and
ruleset ID, then commit only the carried records:

```bash
git add \
  Docs/superpowers/specs/2026-07-20-base-controlled-frontend-license-gate-design.md \
  Docs/superpowers/plans/2026-07-20-base-controlled-frontend-license-gate-implementation-plan.md \
  'backlog/tasks/task-12977 - Bootstrap-base-controlled-frontend-license-gate.md'
git diff --cached --check
git commit -m 'docs: stage trusted frontend gate bootstrap (TASK-12977)'
```

## Task 2: Implement the NUL-Safe Classifier with TDD

**Files:**

- Create: `Helper_Scripts/ci/check_frontend_license_gate.py`
- Create: `tldw_Server_API/tests/CI/test_frontend_license_gate.py`
- Modify: TASK-12977 task record

### Step 1: Write failing boundary and byte-transport tests

Create focused tests that import the classifier module by path and assert:

```python
PROTECTED_PREFIXES = (
    "admin-ui/",
    "apps/tldw-frontend/",
    "apps/extension/",
    "apps/packages/ui/",
    "LICENSES/",
    "tldw_Server_API/app/api/v1/",
)

PROTECTED_EXACT = (
    "LICENSE",
    "THIRD_PARTY_NOTICES.txt",
    "Helper_Scripts/ci/check_frontend_license_gate.py",
    ".github/workflows/frontend-license-gate.yml",
    ".github/workflows/frontend-required.yml",
    "tldw_Server_API/app/main.py",
)
```

Required test cases:

1. `evaluate(author="RMUSSER01", owner="rmusser01", paths=[...]) == []`.
2. Every exact path and a child of every prefix is blocked for `contributor`.
3. Near-prefixes such as `admin-ui-copy/x`, `LICENSE.md`, `LICENSES-copy/x`,
   `tldw_Server_API/app/main.py.bak`, and
   `tldw_Server_API/app/api/v10/x.py` are allowed.
4. Old/new rename pairs are both examined.
5. `read_nul_paths(BytesIO(...))` preserves leading/trailing spaces, tabs,
   newlines, Unicode, and undecodable bytes through `surrogateescape`.
6. A test stream that returns short reads is consumed through EOF or the
   sentinel limit; one short read is never mistaken for EOF.
7. Empty NUL fields are ignored without stripping non-empty fields.
8. More than `8 * 1024 * 1024` bytes raises `ValueError`.
9. The real CLI with `--null` exits `1` for an external protected path,
   exits `0` for unrelated paths/owner paths, and exits `2` for oversized or
   malformed invocation input.
10. CLI diagnostics escape newline, tab, and surrogate bytes instead of writing
   them raw.

Run the new suite and confirm RED because the implementation does not exist:

```bash
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tldw_Server_API/tests/CI/test_frontend_license_gate.py
```

### Step 2: Implement the minimal standard-library classifier

Implement these public interfaces exactly:

```python
from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from typing import BinaryIO, Sequence

MAX_INPUT_BYTES = 8 * 1024 * 1024

PROTECTED_PREFIXES = (
    "admin-ui/",
    "apps/tldw-frontend/",
    "apps/extension/",
    "apps/packages/ui/",
    "LICENSES/",
    "tldw_Server_API/app/api/v1/",
)

PROTECTED_EXACT = frozenset(
    {
        "LICENSE",
        "THIRD_PARTY_NOTICES.txt",
        "Helper_Scripts/ci/check_frontend_license_gate.py",
        ".github/workflows/frontend-license-gate.yml",
        ".github/workflows/frontend-required.yml",
        "tldw_Server_API/app/main.py",
    }
)


def blocked_changes(paths: Iterable[str]) -> list[str]:
    return [
        path
        for path in paths
        if path in PROTECTED_EXACT
        or any(path.startswith(prefix) for prefix in PROTECTED_PREFIXES)
    ]


def evaluate(*, author: str, owner: str, paths: Iterable[str]) -> list[str]:
    if author.casefold() == owner.casefold():
        return []
    return blocked_changes(paths)


def read_nul_paths(
    stream: BinaryIO,
    *,
    max_bytes: int = MAX_INPUT_BYTES,
) -> list[str]:
    chunks: list[bytes] = []
    remaining = max_bytes + 1
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b"".join(chunks)
    if len(data) > max_bytes:
        raise ValueError(f"changed-path input exceeds {max_bytes} bytes")
    return [
        value.decode("utf-8", errors="surrogateescape")
        for value in data.split(b"\0")
        if value
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--null", action="store_true", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        paths = read_nul_paths(sys.stdin.buffer)
    except (OSError, ValueError) as exc:
        print(f"frontend license gate input error: {exc}", file=sys.stderr)
        return 2

    blocked = evaluate(author=args.author, owner=args.owner, paths=paths)
    if not blocked:
        print("frontend license gate: allowed")
        return 0

    print("frontend license gate: protected changes are frozen", file=sys.stderr)
    for path in blocked:
        print(f"- {ascii(path)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

Do not add a dependency, configuration framework, generic policy abstraction,
or fallback line parser.

### Step 3: Run GREEN and security checks

```bash
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tldw_Server_API/tests/CI/test_frontend_license_gate.py
python -m bandit -r Helper_Scripts/ci/check_frontend_license_gate.py \
  -f json -o /tmp/bandit_task-12977-classifier.json
git diff --check
```

Use `superpowers:requesting-code-review`; fix every valid correctness or
security finding, rerun the commands, update TASK-12977 with results, and
commit:

```bash
git add \
  Helper_Scripts/ci/check_frontend_license_gate.py \
  tldw_Server_API/tests/CI/test_frontend_license_gate.py \
  'backlog/tasks/task-12977 - Bootstrap-base-controlled-frontend-license-gate.md'
git commit -m 'test: harden frontend license path classifier (TASK-12977)'
```

## Task 3: Add the Base-Controlled Workflow with Contract Tests

**Files:**

- Create: `.github/workflows/frontend-license-gate.yml`
- Create: `tldw_Server_API/tests/CI/test_frontend_license_gate_workflow.py`
- Modify: `.github/workflows/actionlint.yml`
- Modify: TASK-12977 task record

### Step 1: Write the workflow contract tests first

The tests must load YAML with:

```python
triggers = data.get("on", data.get(True))
```

This avoids PyYAML 1.1 treating `on` as Boolean `True`. Assert all of the
following before the workflow exists:

- only `pull_request_target` is used for PR automation and its branches are
  exactly `main` and `dev`;
- top-level permissions equal `{"contents": "read", "statuses": "write"}`;
- the only job ID is `frontend-license-gate-audit`;
- checkout is pinned to
  `actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd`, uses
  `ref: ${{ github.sha }}`, `fetch-depth: 0`, and
  `persist-credentials: false`;
- no step checks out the head SHA or PR ref, uses a local action, installs
  dependencies, restores a cache, downloads artifacts, or executes a path
  fetched from the head;
- pending status is posted before checkout/evaluation;
- the evaluator validates PR number, base ref, 40-hex base/head SHAs, fetches
  explicit base/head refs, and verifies both fetched object IDs;
- diff invocation contains `--name-only`, `-z`, `--no-renames`,
  `--no-ext-diff`, `--no-textconv`, and final `--`;
- the classifier is invoked from the trusted checkout with `--null`;
- unknown/missing evaluation output maps to `failure`;
- the final status publisher uses `if: always()` and success is possible only
  when the evaluator output is exactly `success`;
- the status context is `frontend-license-policy/trusted`, distinct from the
  job ID; and
- actionlint's explicit target list includes the new workflow.

Run RED:

```bash
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tldw_Server_API/tests/CI/test_frontend_license_gate_workflow.py
```

### Step 2: Add the trusted workflow

Create `.github/workflows/frontend-license-gate.yml` with this control flow:

```yaml
name: Frontend License Gate Audit

on:
  pull_request_target:
    branches: [main, dev]
    types: [opened, reopened, synchronize, ready_for_review]

permissions:
  contents: read
  statuses: write

concurrency:
  group: frontend-license-gate-${{ github.event.pull_request.number }}
  cancel-in-progress: true

jobs:
  frontend-license-gate-audit:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    env:
      GH_TOKEN: ${{ github.token }}
      STATUS_CONTEXT: frontend-license-policy/trusted
      STATUS_REPOSITORY: ${{ github.repository }}
      HEAD_SHA: ${{ github.event.pull_request.head.sha }}
      BASE_SHA: ${{ github.event.pull_request.base.sha }}
      BASE_REF: ${{ github.event.pull_request.base.ref }}
      PR_NUMBER: ${{ github.event.pull_request.number }}
      PR_AUTHOR: ${{ github.event.pull_request.user.login }}
      REPOSITORY_OWNER: ${{ github.repository_owner }}
    steps:
      - name: Mark trusted policy pending
        shell: bash
        run: |
          set -euo pipefail
          gh api --method POST \
            "repos/${STATUS_REPOSITORY}/statuses/${HEAD_SHA}" \
            -f state=pending \
            -f context="${STATUS_CONTEXT}" \
            -f description='Trusted frontend license policy is evaluating'

      - name: Checkout trusted policy
        uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd
        with:
          ref: ${{ github.sha }}
          fetch-depth: 0
          persist-credentials: false

      - id: evaluate
        name: Evaluate immutable pull request metadata
        shell: bash
        run: |
          set -euo pipefail
          verdict=failure
          emit_verdict() {
            printf 'verdict=%s\n' "${verdict}" >> "${GITHUB_OUTPUT}"
          }
          trap emit_verdict EXIT

          [[ "${PR_NUMBER}" =~ ^[1-9][0-9]*$ ]]
          [[ "${BASE_SHA}" =~ ^[0-9a-f]{40}$ ]]
          [[ "${HEAD_SHA}" =~ ^[0-9a-f]{40}$ ]]
          [[ "${BASE_REF}" == main || "${BASE_REF}" == dev ]]
          [[ -n "${PR_AUTHOR}" && -n "${REPOSITORY_OWNER}" ]]

          if [[ "${PR_AUTHOR,,}" == "${REPOSITORY_OWNER,,}" ]]; then
            python3 Helper_Scripts/ci/check_frontend_license_gate.py \
              --author "${PR_AUTHOR}" --owner "${REPOSITORY_OWNER}" --null \
              </dev/null
            verdict=success
            exit 0
          fi

          readonly public_remote="https://github.com/${STATUS_REPOSITORY}.git"
          git fetch --no-tags --depth=1 "${public_remote}" \
            "+refs/heads/${BASE_REF}:refs/remotes/license-gate/base"
          git fetch --no-tags --depth=1 "${public_remote}" \
            "+refs/pull/${PR_NUMBER}/head:refs/remotes/license-gate/pr-head"

          readonly fetched_base="$(git rev-parse refs/remotes/license-gate/base)"
          readonly fetched_head="$(git rev-parse refs/remotes/license-gate/pr-head)"
          [[ "${fetched_base}" == "${BASE_SHA}" ]]
          [[ "${fetched_head}" == "${HEAD_SHA}" ]]

          set +e
          git --no-pager diff --name-only -z --no-renames \
            --no-ext-diff --no-textconv "${BASE_SHA}" "${HEAD_SHA}" -- | \
            python3 Helper_Scripts/ci/check_frontend_license_gate.py \
              --author "${PR_AUTHOR}" --owner "${REPOSITORY_OWNER}" --null
          readonly pipeline_status=("${PIPESTATUS[@]}")
          set -e

          [[ "${pipeline_status[0]}" -eq 0 ]]
          [[ "${pipeline_status[1]}" -eq 0 ]]
          verdict=success

      - name: Publish trusted policy result
        if: always()
        shell: bash
        env:
          VERDICT: ${{ steps.evaluate.outputs.verdict }}
        run: |
          set -euo pipefail
          state=failure
          description='Trusted frontend license policy failed closed'
          if [[ "${VERDICT}" == success ]]; then
            state=success
            description='Trusted frontend license policy allowed this change'
          fi
          gh api --method POST \
            "repos/${STATUS_REPOSITORY}/statuses/${HEAD_SHA}" \
            -f state="${state}" \
            -f context="${STATUS_CONTEXT}" \
            -f description="${description}"
          [[ "${state}" == success ]]
```

Security review must specifically verify that `${{ github.sha }}` is the only
checkout ref and that the public fetch provides Git objects only; no command
uses the fetched worktree content.

Add `.github/workflows/frontend-license-gate.yml` to the explicit invocation
in `.github/workflows/actionlint.yml` without opportunistically changing other
workflow targets.

### Step 3: Run GREEN, lint, and focused regression tests

```bash
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tldw_Server_API/tests/CI/test_frontend_license_gate.py \
  tldw_Server_API/tests/CI/test_frontend_license_gate_workflow.py \
  tldw_Server_API/tests/CI/test_frontend_required_workflow.py
actionlint -color -config-file .github/actionlint.yaml \
  .github/workflows/frontend-license-gate.yml \
  .github/workflows/actionlint.yml
python -m bandit -r Helper_Scripts/ci/check_frontend_license_gate.py \
  -f json -o /tmp/bandit_task-12977-workflow.json
git diff --check
```

If local `actionlint` is unavailable, run the repository's pinned 1.7.12
binary using the same download procedure as `actionlint.yml` and record that
fact; do not substitute an unpinned release.

Use `superpowers:requesting-code-review` with the approved design and trust
model. Resolve every valid finding, rerun all commands, update TASK-12977, and
commit:

```bash
git add \
  .github/workflows/frontend-license-gate.yml \
  .github/workflows/actionlint.yml \
  tldw_Server_API/tests/CI/test_frontend_license_gate_workflow.py \
  'backlog/tasks/task-12977 - Bootstrap-base-controlled-frontend-license-gate.md'
git commit -m 'ci: add base-controlled frontend license gate (TASK-12977)'
```

## Task 4: Open, Merge, Observe, and Activate

**Files:**

- Modify after live verification:
  `backlog/tasks/task-12977 - Bootstrap-base-controlled-frontend-license-gate.md`
- Create on the licensing branch after activation:
  `Docs/superpowers/evidence/TASK-12977/main-ruleset-before.json`
- Create on the licensing branch after activation:
  `Docs/superpowers/evidence/TASK-12977/main-ruleset-after.json`
- Create on the licensing branch after activation:
  `Docs/superpowers/evidence/TASK-12977/dev-ruleset.json`

### Step 1: Verify and open the bootstrap PR

Use `superpowers:verification-before-completion`:

```bash
git status --short
git diff --check origin/main...HEAD
git diff --name-status origin/main...HEAD
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tldw_Server_API/tests/CI/test_frontend_license_gate.py \
  tldw_Server_API/tests/CI/test_frontend_license_gate_workflow.py \
  tldw_Server_API/tests/CI/test_frontend_required_workflow.py
actionlint -color -config-file .github/actionlint.yaml \
  .github/workflows/frontend-license-gate.yml \
  .github/workflows/actionlint.yml
python -m bandit -r Helper_Scripts/ci/check_frontend_license_gate.py \
  -f json -o /tmp/bandit_task-12977-final.json
git push -u origin codex/frontend-license-gate-bootstrap
gh pr create --repo rmusser01/tldw_server \
  --base main \
  --head codex/frontend-license-gate-bootstrap \
  --draft \
  --title 'ci: bootstrap trusted frontend license gate' \
  --body $'## Summary\n\nBootstrap the default-branch-controlled frontend licensing gate described in TASK-12977.\n\n## Verification\n\nFocused pytest, actionlint, Bandit, and diff checks are recorded in TASK-12977.\n\n## Human Change summary required before merge\n\n<!-- Robert: write what changed and why these choices were made, in your own words. -->'
```

The PR body may contain an agent-authored factual test/diff summary, but it
must contain a clearly empty section headed `## Human Change summary required
before merge`. Stop and ask Robert to write that section in his own words.
Do not merge, ready, or activate rules until Robert confirms the human summary
is present and the PR review/checks are acceptable.

### Step 2: Merge the bootstrap and produce a real status

After Robert's confirmation, merge through the existing main ruleset. Then
open or synchronize an owner-authored PR targeting `dev`—normally the licensing
cutoff PR—to produce the trusted status from the now-base-controlled workflow.
Record:

```bash
gh pr view codex/frontend-licensing-cutoff \
  --repo rmusser01/tldw_server \
  --json number,headRefOid,url \
  > /tmp/task-12977-licensing-pr.json
task12977_pr_number="$(jq -er '.number' /tmp/task-12977-licensing-pr.json)"
task12977_head_sha="$(jq -er '.headRefOid' /tmp/task-12977-licensing-pr.json)"
gh pr checks "${task12977_pr_number}" --repo rmusser01/tldw_server
gh api \
  "repos/rmusser01/tldw_server/commits/${task12977_head_sha}/status" \
  --jq '.statuses[] | select(.context == "frontend-license-policy/trusted")'
gh api \
  "repos/rmusser01/tldw_server/commits/${task12977_head_sha}/check-runs" \
  --jq '.check_runs[] | select(.name == "frontend-license-gate-audit") | {id, name, conclusion, app}'
task12977_integration_id="$(gh api \
  "repos/rmusser01/tldw_server/commits/${task12977_head_sha}/check-runs" \
  --jq '[.check_runs[] | select(.name == "frontend-license-gate-audit" and .conclusion == "success") | .app.id] | unique | if length == 1 then .[0] else error("expected exactly one successful source app") end')"
[[ "${task12977_integration_id}" =~ ^[1-9][0-9]*$ ]]
```

Never infer the PR number, head SHA, or source from a stale local branch.
Confirm:

- the status SHA equals the current PR head SHA;
- status state is `success`;
- status creator is the GitHub Actions bot/source;
- the audit check's `app.id` is a positive integer; and
- the workflow run references the merged default-branch workflow.

Keep the observed `app.id` in `task12977_integration_id`. Do not assume a
hard-coded GitHub Actions App ID.

### Step 3: Build a source-bound main ruleset payload from live state

Immediately recheck direct collaborators. Then save the complete live main
ruleset response and derive an update payload from it:

```bash
gh api repos/rmusser01/tldw_server/rulesets/5653432 \
  > /tmp/task-12977-main-ruleset-before.json
jq --arg context 'frontend-license-policy/trusted' \
  --argjson integration_id "${task12977_integration_id}" '
    if any(.rules[]; .type == "required_status_checks") then
      error("main ruleset already has required_status_checks; stop")
    else
      {
        name,
        target,
        enforcement,
        bypass_actors,
        conditions,
        rules: (.rules + [{
          type: "required_status_checks",
          parameters: {
            do_not_enforce_on_create: false,
            required_status_checks: [{
              context: $context,
              integration_id: $integration_id
            }],
            strict_required_status_checks_policy: false
          }
        }])
      }
    end
  ' /tmp/task-12977-main-ruleset-before.json \
  > /tmp/task-12977-main-ruleset-update.json
```

`strict_required_status_checks_policy: false` preserves the current merge
model instead of introducing an unrelated up-to-date-branch requirement.
`do_not_enforce_on_create: false` prevents branch creation from bypassing the
rule.

Apply and re-read:

```bash
gh api --method PUT repos/rmusser01/tldw_server/rulesets/5653432 \
  --input /tmp/task-12977-main-ruleset-update.json
gh api repos/rmusser01/tldw_server/rulesets/5653432 \
  > /tmp/task-12977-main-ruleset-after.json
```

Use `jq` to assert that every old rule, condition, bypass actor, target,
enforcement value, and name remains present and unchanged, with exactly one
new status rule whose context and `integration_id` match the observed values.
If the PUT or assertion fails, PUT a sanitized payload derived from the saved
before JSON, verify restoration, and stop.

### Step 4: Create and verify the `dev` ruleset

Create a payload with no bypass actors, targeting only `refs/heads/dev`. Copy
the existing main pull-request rule from the saved live response instead of
retyping its evolving parameter schema:

```bash
jq --arg context 'frontend-license-policy/trusted' \
  --argjson integration_id "${task12977_integration_id}" '
    [.rules[] | select(.type == "pull_request")] as $pull_request_rules
    | if ($pull_request_rules | length) != 1 then
        error("expected exactly one live main pull_request rule")
      else
        {
          name: "frontend-license-gate-dev",
          target: "branch",
          enforcement: "active",
          bypass_actors: [],
          conditions: {
            ref_name: {
              include: ["refs/heads/dev"],
              exclude: []
            }
          },
          rules: [
            $pull_request_rules[0],
            {
              type: "required_status_checks",
              parameters: {
                do_not_enforce_on_create: false,
                required_status_checks: [{
                  context: $context,
                  integration_id: $integration_id
                }],
                strict_required_status_checks_policy: false
              }
            }
          ]
        }
      end
  ' /tmp/task-12977-main-ruleset-before.json \
  > /tmp/task-12977-dev-ruleset-create.json
```

Create it through:

```bash
gh api --method POST repos/rmusser01/tldw_server/rulesets \
  --input /tmp/task-12977-dev-ruleset-create.json
```

Read the returned ruleset by its new ID and assert:

- target is `branch`, enforcement is `active`, and include is exactly
  `refs/heads/dev`;
- bypass actors are empty;
- PR merge methods and zero-approval policy match the current repository
  choice; and
- required status context and `integration_id` match the observed status.

If the expected-source binding is absent or differs, disable the new dev
ruleset, restore main from the saved before payload, verify both changes, and
stop.

### Step 5: Record evidence without secrets

Copy the three public JSON responses into the evidence directory on the
licensing branch, removing only volatile `_links` fields if they make review
noisy. Record workflow run URL, PR number, exact head SHA, status creator,
integration ID, main ruleset ID, dev ruleset ID, and activation timestamps in
TASK-12977. Never record tokens or request headers.

## Task 5: Reconcile and Resume TASK-12976

**Files:**

- Modify: `Helper_Scripts/ci/check_frontend_license_gate.py`
- Modify: `tldw_Server_API/tests/CI/test_frontend_license_gate.py`
- Revert/remove the rejected edit to: `.github/workflows/frontend-required.yml`
- Carry from `main`: `.github/workflows/frontend-license-gate.yml`
- Carry from `main`:
  `tldw_Server_API/tests/CI/test_frontend_license_gate_workflow.py`
- Modify: `.github/workflows/actionlint.yml`
- Modify: TASK-12976 and TASK-12977 records
- Modify: `.superpowers/sdd/progress.md` by appending only
- Create: `Docs/superpowers/evidence/TASK-12977/*.json`

### Step 1: Integrate the trusted bootstrap into the licensing branch

Fetch the merged `main` and rebase/merge only in the isolated licensing
worktree using the least disruptive method compatible with its published PR.
Do not rewrite PR `#2727`. The license-only branch is separate and may be
rebased only after verifying it has not been published or after explicit user
approval.

Bring the trusted workflow/classifier/tests from the merged main commit. Then
remove the rejected `pull_request` classifier step from
`.github/workflows/frontend-required.yml`; that workflow must return to its
pre-TASK-12976 behavior.

### Step 2: Re-run the Task 4 review from TASK-12976

Verify the effective diff against current `origin/dev`:

```bash
git diff --name-status origin/dev...HEAD
git diff -- .github/workflows/frontend-required.yml
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tldw_Server_API/tests/CI/test_frontend_license_gate.py \
  tldw_Server_API/tests/CI/test_frontend_license_gate_workflow.py \
  tldw_Server_API/tests/CI/test_frontend_required_workflow.py \
  tldw_Server_API/tests/CI/test_licensing_policy.py
actionlint -color -config-file .github/actionlint.yaml \
  .github/workflows/frontend-license-gate.yml \
  .github/workflows/frontend-required.yml \
  .github/workflows/actionlint.yml
python -m bandit -r Helper_Scripts/ci/check_frontend_license_gate.py \
  -f json -o /tmp/bandit_task-12976-gate-replacement.json
git diff --check
```

Use `superpowers:requesting-code-review` specifically against the two rejected
findings:

1. the PR cannot remove or replace the workflow/classifier that evaluates it;
2. every path byte is transported by NUL with no whitespace trimming, and
   renames expose both old and new paths.

### Step 3: Close TASK-12977 only after live enforcement is proven

Complete acceptance criteria 2–5 and the Definition of Done only when:

- the bootstrap commit is on `main`;
- a real owner-authored dev PR has a successful source-bound trusted status;
- main and dev rulesets return the expected `integration_id`;
- external protected/unprotected deterministic cases pass locally;
- focused tests, actionlint, Bandit, and diff checks are recorded; and
- rollback artifacts/IDs are documented.

Append the completion entry to `.superpowers/sdd/progress.md`; do not rewrite
existing entries. Commit evidence/tracking with the reconciled gate changes.

### Step 4: Return to the licensing cutoff

Mark Task 4 of TASK-12976 complete only after the new independent review is
clean. Continue Tasks 5 and 6 from the conservative cutoff plan, then open the
license-only PR to `dev`. The merge order remains:

1. trusted bootstrap to `main`;
2. source-bound status/ruleset activation;
3. license-only cutoff to `dev`;
4. architecture PR `#2727` only after the cutoff is merged and verified.

## Final Verification Checklist

- [ ] Bootstrap diff contains only TASK-12977 trust-root files.
- [ ] Robert supplied the bootstrap PR `Change summary` in his own words.
- [ ] Workflow exists on `main` before ruleset activation.
- [ ] Privileged workflow checks out only `${{ github.sha }}`.
- [ ] PR head is fetched as Git objects and never checked out/executed.
- [ ] Classifier input is NUL-delimited, bounded, and decoded with
      `surrogateescape` without trimming.
- [ ] Owner allow and external protected deny cases pass.
- [ ] Pending is posted first; only an explicit allow publishes success.
- [ ] `frontend-license-policy/trusted` is bound to the observed integration.
- [ ] Main ruleset `5653432` retains every prior rule and condition.
- [ ] Dev ruleset targets only `refs/heads/dev`, requires PR + trusted status,
      and has no bypass actors.
- [ ] Focused pytest suites, actionlint, Bandit, and `git diff --check` pass.
- [ ] Public before/after ruleset evidence and rollback IDs are recorded.
- [ ] TASK-12977 and TASK-12976 statuses accurately reflect live state.

## Primary References

- GitHub `pull_request_target` semantics:
  https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows
- GitHub secure-use guidance:
  https://docs.github.com/en/actions/reference/security/secure-use
- Available ruleset rules and expected sources:
  https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets
- Repository ruleset REST schema:
  https://docs.github.com/en/rest/repos/rules
- Commit-status REST endpoint:
  https://docs.github.com/en/rest/commits/statuses
