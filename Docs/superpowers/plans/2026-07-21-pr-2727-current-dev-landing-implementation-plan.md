# PR #2727 Current-Dev Landing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the reproduced exact-head gates, obtain the requester-authored release rationale, and merge PR #2727 into the then-current protected `dev` without repeating the already completed `dev` merge or disturbing unrelated worktree files.

**Architecture:** Treat pushed PR head `6065c64ab4a06687cc10e938eb0bd1cc5b6fd031` and merge commit `0e8eadc55f48ff50f55525b8996140cbad43630c` as the completed integration baseline. Make two independently reviewable corrections on top: an actionlint-safe declaration change that preserves successful trusted-gate behavior while making failed SHA resolution terminate fail closed, and an acknowledged OpenAPI fingerprint refresh whose semantic delta is proven to be exactly four top-level repository/contact/license fields merged from `dev`. Push once, require every ordinary and trusted exact-head gate, collect the human-only Change summary, recheck `dev` freshness, merge with an enforced up-to-date base plus head-SHA protection, and verify the actual merge lineage before opening the separate private-pilot task.

**Tech Stack:** Git and clean Git worktrees, GitHub CLI and GitHub Actions, Bash, actionlint 1.7.12, the project virtual environment plus an isolated local Python 3.12 environment with the hosted OpenAPI job's schema-affecting dependency versions, pytest, FastAPI OpenAPI export tooling, Bun/openapi-typescript, Bandit evidence, and Backlog.md CLI.

## Global Constraints

- This plan implements only TASK-12982. TASK-12983, deployment, customer access, images, browser-extension packages, protected release records, and all other pilot work remain out of scope.
- `dev` was already merged into PR #2727. Do not repeat that merge while `origin/dev` remains `8ed612c7e0335ab922b6abd5f5c11ba1407d552d`; merge again only if a final fetch proves that `dev` advanced.
- Preserve history. Do not rebase, squash, force-push, bypass a gate, use `--admin`, or merge into `main`.
- Keep these unrelated untracked files untouched and unstaged: `server-ux-smoke.pid`, `tldw_Server_API/Config_Files/templates/watchlists/cti_osint_report_markdown.md`, and `tldw_Server_API/Config_Files/templates/watchlists/news_briefing_markdown.md`.
- Do not run reset, stash, checkout-discard, clean, or broad recursive deletion against the working tree.
- The validated revision is always one exact remote PR head. Any new commit, review fix, or `dev` refresh invalidates earlier head results and restarts the exact-head gate loop.
- `frontend-license-policy/trusted/dev` is necessary but not sufficient. Also require the ordinary aggregate gates documented in `Docs/Development/CI_REQUIRED_GATES.md` and every triggered current-head check to reach a non-blocking terminal state.
- A canceled current-head job is not success. Accept a canceled old-head job only when a newer exact-head run contains an equivalent successful replacement; otherwise rerun the current-head job or workflow and record the result.
- The requester, Robert Benjamin Jake Musser, must write the PR's **Change summary** in his own words. An agent may verify its presence but must not draft, rewrite, paste, or substitute that text.
- Use the project environment before Python, pytest, or Bandit commands: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate`.
- The current project environment is Python 3.11/Pydantic 2.11 and emits a different schema from hosted `backend-required`, which used Python 3.12.13, FastAPI 0.136.3, Pydantic 2.13.4, pydantic-core 2.46.4, and pydantic-settings 2.14.2. Never write the checked-in fingerprint from the ambient project environment. Prepare a local Python 3.12 environment in Task 3, assert its major/minor and the exact schema-affecting dependency versions, reproduce the hosted hash, and pass that interpreter explicitly to the supported Make targets. Do not claim the local Python patch version is identical to hosted CI unless it is.
- Dependency resolution is not locked by this repository, so the canonical exporter is environment-canonical but not dependency-canonical. Do not broaden this landing PR into dependency-policy work; retain a follow-up recommendation to pin or otherwise record schema-generator dependencies after #2727 lands.
- Launch every fenced Bash block with its process working directory set to `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/provider-credential-runtime` unless that block explicitly changes to the owned OpenAPI or closeout worktree. This is an execution precondition, not shell state inherited from an earlier block.
- Treat every fenced Bash command as a fresh shell. Start every multi-command execution with `set -euo pipefail`; never rely on a prior `cd`, variable, export, or virtual-environment activation.
- Use the official Backlog.md CLI for TASK-12982 updates. Do not manually edit its task file.
- Before executing Task 1, this reviewed plan and TASK-12982's official plan link must already be committed. That preparation removes the plan as a fourth untracked path, so Task 1 can enforce the invariant that only the three named user files remain untracked.
- Stop after three unsuccessful attempts at the same failure, record the attempts and exact error, and reassess before another change.

## File Map

- Modify `.github/workflows/frontend-license-gate.yml`: separate two command substitutions from their `readonly` declarations without changing fetch, comparison, classifier, or fail-closed behavior.
- Modify `tldw_Server_API/tests/CI/test_frontend_license_gate_workflow.py`: add the declaration-order regression and update the reviewed evaluator-script digest.
- Modify `apps/tldw-frontend/lib/api/openapi.fingerprint.json`: acknowledge the hosted-CI-reproduced OpenAPI ownership/license metadata change after proving its exact semantic source.
- Read but do not modify `Helper_Scripts/export_openapi_schema.py`: canonical exporter and fingerprint implementation.
- Read but do not modify `apps/tldw-frontend/scripts/generate-api-types.mjs`: generated-type verification path.
- Read but do not modify `tldw_Server_API/tests/Services/test_openapi_contracts.py`: executable API/server license contract.
- Update TASK-12982 through `backlog task edit`: plan link, evidence, acceptance criteria, and final closeout.

---

### Task 1: Re-establish the immutable landing baseline

**Files:**
- Read: `Docs/superpowers/specs/2026-07-21-pr-2727-landing-private-pilot-design.md`
- Read: `backlog/tasks/task-12982 - Land-PR-2727-on-current-dev.md` through `backlog task TASK-12982 --plain`
- Read: remote PR #2727 and Git refs
- Modify: TASK-12982 implementation notes through the Backlog.md CLI

**Interfaces:**
- Consumes: completed merge `0e8eadc55f48ff50f55525b8996140cbad43630c`, pushed baseline `6065c64ab4a06687cc10e938eb0bd1cc5b6fd031`, and protected `dev` tip `8ed612c7e0335ab922b6abd5f5c11ba1407d552d`.
- Produces: a recorded baseline showing that local work descends from the remote PR head, both earlier heads are ancestors, and only the three named unrelated files are untracked.

- [ ] **Step 1: Fetch without changing the checked-out branch**

```bash
set -euo pipefail
cd /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/provider-credential-runtime
git fetch --no-tags origin dev codex/provider-credential-runtime-dev
```

Expected: fetch succeeds without a merge, rebase, checkout, or force update.

- [ ] **Step 2: Verify local and remote lineage**

```bash
set -euo pipefail
git rev-parse origin/dev
git rev-parse origin/codex/provider-credential-runtime-dev
git merge-base --is-ancestor 8ed612c7e0335ab922b6abd5f5c11ba1407d552d 6065c64ab4a06687cc10e938eb0bd1cc5b6fd031
git merge-base --is-ancestor e8bcc4c8b705df50a5f7e6299335ba8001ff4811 6065c64ab4a06687cc10e938eb0bd1cc5b6fd031
git merge-base --is-ancestor 6065c64ab4a06687cc10e938eb0bd1cc5b6fd031 HEAD
git show -s --format='%H%n%P%n%s' 0e8eadc55f48ff50f55525b8996140cbad43630c
```

Expected: `origin/dev` is `8ed612c7e0335ab922b6abd5f5c11ba1407d552d`; the remote PR branch is `6065c64ab4a06687cc10e938eb0bd1cc5b6fd031`; all three ancestry commands exit 0; and the merge parents are `7d76bdfcc0c467c779596cd6b92d2f078aa8529e 8ed612c7e0335ab922b6abd5f5c11ba1407d552d` in that order. If a remote ref differs, stop and reconcile it before editing.

- [ ] **Step 3: Prove the unrelated worktree state is unchanged**

```bash
set -euo pipefail
git status --short --branch
git diff --name-only
git diff --cached --name-only
```

Expected: no tracked unstaged or staged path; exactly the three globally listed untracked files; the local branch is ahead only by the already reviewed design/plan commits.

- [ ] **Step 4: Capture current remote PR and check state**

```bash
set -euo pipefail
gh pr view 2727 --repo rmusser01/tldw_server \
  --json headRefOid,baseRefOid,isDraft,mergeable,mergeStateStatus,reviewDecision,url
gh pr checks 2727 --repo rmusser01/tldw_server \
  --json bucket,name,state,workflow,link,startedAt,completedAt \
  --jq 'sort_by(.bucket,.workflow,.name)[] | [.bucket,.state,.workflow,.name,.link] | @tsv'
```

Expected before the corrective push: head `6065c64ab4a06687cc10e938eb0bd1cc5b6fd031`, base `8ed612c7e0335ab922b6abd5f5c11ba1407d552d`, draft and mergeable/unstable; trusted license status passes; the reproduced underlying failures are actionlint SC2155 and the OpenAPI fingerprint mismatch. Pending checks remain pending evidence, not failures to cancel.

- [ ] **Step 5: Record the immutable baseline in TASK-12982**

```bash
backlog task edit TASK-12982 --append-notes \
  'Execution baseline reconfirmed: PR head 6065c64ab4 contains original head e8bcc4c8b and protected dev 8ed612c7e0 through merge 0e8eadc55f (parents 7d76bdfcc0 and 8ed612c7e0). Local work descends from that head; no tracked dirty paths exist; server-ux-smoke.pid and the two named watchlist templates remain unrelated and unstaged. Known current-head corrections are actionlint SC2155 and OpenAPI fingerprint drift.' \
  --plain
```

Expected: TASK-12982 remains In Progress and the note is appended rather than replacing earlier evidence.

- [ ] **Step 6: Commit the baseline note before code changes**

```bash
set -euo pipefail
git add 'backlog/tasks/task-12982 - Land-PR-2727-on-current-dev.md'
git diff --cached --check
git diff --cached --name-only
git commit -m "chore(backlog): record PR 2727 landing baseline (TASK-12982)"
```

Expected: the commit contains only the TASK-12982 record, leaving a clean tracked tree for the correction commits.

---

### Task 2: Repair actionlint while hardening trusted-gate failure behavior

**Files:**
- Modify: `.github/workflows/frontend-license-gate.yml:79-82`
- Modify: `tldw_Server_API/tests/CI/test_frontend_license_gate_workflow.py:25-29,130-160`
- Test: `tldw_Server_API/tests/CI/test_frontend_license_gate_workflow.py`

**Interfaces:**
- Consumes: evaluator script from the base-controlled `pull_request_target` workflow and its SHA-256 integrity contract.
- Produces: the same successful `fetched_base` and `fetched_head` values and fail-closed comparisons, while a failed `git rev-parse` is no longer masked by the `readonly` builtin; actionlint 1.7.12 is clean and the evaluator digest is `2e31c8d430e2e636f770de9d869c9bff364c6a1ceb37ca8d16bed83b6326f243`.

- [ ] **Step 1: Add the failing declaration-order regression**

Add this test immediately after `assert_success_follows_prerequisites` and before the first `test_...` function:

```python
def test_evaluator_separates_readonly_declarations_from_command_substitution() -> None:
    steps = load_yaml(WORKFLOW_PATH)["jobs"][JOB_ID]["steps"]
    script = next(step for step in steps if step.get("id") == "evaluate")["run"]

    expected_assignments = {
        "fetched_base": 'fetched_base="$(git rev-parse refs/remotes/license-gate/base)"',
        "fetched_head": 'fetched_head="$(git rev-parse refs/remotes/license-gate/pr-head)"',
    }
    for variable, assignment in expected_assignments.items():
        declaration = f"readonly {variable}"
        assert script.count(assignment) == 1
        assert script.count(declaration) == 1
        assert f"readonly {variable}=" not in script
        assert script.index(assignment) < script.index(declaration)
```

- [ ] **Step 2: Run the new test and confirm RED**

```bash
set -euo pipefail
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
TLDW_RED_LOG="$(mktemp)"
cleanup_red_log() {
  rm -f -- "${TLDW_RED_LOG}"
}
trap cleanup_red_log EXIT
if python -m pytest -q \
  tldw_Server_API/tests/CI/test_frontend_license_gate_workflow.py::test_evaluator_separates_readonly_declarations_from_command_substitution \
  >"${TLDW_RED_LOG}" 2>&1; then
  cat "${TLDW_RED_LOG}"
  echo 'Expected the new declaration-order regression to fail before the workflow edit' >&2
  exit 1
fi
cat "${TLDW_RED_LOG}"
rg -F 'test_evaluator_separates_readonly_declarations_from_command_substitution' \
  "${TLDW_RED_LOG}"
rg -F 'AssertionError' "${TLDW_RED_LOG}"
cleanup_red_log
trap - EXIT
```

Expected: pytest reaches the named test and fails with an assertion because the current workflow uses inline `readonly` assignments. Activation, import, collection, or infrastructure failures do not satisfy RED.

- [ ] **Step 3: Make the minimal workflow change**

Replace only the two inline declarations with:

```bash
fetched_base="$(git rev-parse refs/remotes/license-gate/base)"
readonly fetched_base
fetched_head="$(git rev-parse refs/remotes/license-gate/pr-head)"
readonly fetched_head
```

Do not change the remote URLs, refs, SHA comparisons, pipeline-status handling, classifier call, verdict trap, permissions, event, or status publisher.

- [ ] **Step 4: Run the new regression and the full contract file**

```bash
set -euo pipefail
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/CI/test_frontend_license_gate_workflow.py::test_evaluator_separates_readonly_declarations_from_command_substitution
python -m pytest -q tldw_Server_API/tests/CI/test_frontend_license_gate_workflow.py
```

Expected: the new test passes; the full file fails only because `TRUSTED_RUN_SHA256["Evaluate immutable pull request metadata"]` still has the reviewed old digest. Any other failure means the workflow behavior changed unexpectedly and must be corrected before updating the digest.

- [ ] **Step 5: Review and update the exact evaluator digest**

First inspect the workflow/test diff:

```bash
git diff -- .github/workflows/frontend-license-gate.yml \
  tldw_Server_API/tests/CI/test_frontend_license_gate_workflow.py
```

Then replace only this entry in `TRUSTED_RUN_SHA256`:

```python
"Evaluate immutable pull request metadata": "2e31c8d430e2e636f770de9d869c9bff364c6a1ceb37ca8d16bed83b6326f243",
```

Expected: the only script change is declaration/assignment order; successful SHA resolution is unchanged, failed resolution now terminates under `set -e`, and the digest matches the exact YAML-loaded run body.

- [ ] **Step 6: Run the trusted workflow contracts GREEN**

```bash
set -euo pipefail
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/CI/test_frontend_license_gate_workflow.py
```

Expected: all tests in the file pass.

- [ ] **Step 7: Reproduce CI's pinned actionlint command**

```bash
set -euo pipefail
TLDW_ACTIONLINT_TMP="$(mktemp -d)"
cleanup_actionlint() {
  rm -rf -- "${TLDW_ACTIONLINT_TMP}"
}
trap cleanup_actionlint EXIT
gh release download v1.7.12 \
  --repo rhysd/actionlint \
  --pattern 'actionlint_1.7.12_darwin_arm64.tar.gz' \
  --output "${TLDW_ACTIONLINT_TMP}/actionlint.tar.gz"
tar -xzf "${TLDW_ACTIONLINT_TMP}/actionlint.tar.gz" -C "${TLDW_ACTIONLINT_TMP}" actionlint
"${TLDW_ACTIONLINT_TMP}/actionlint" -color -config-file .github/actionlint.yaml \
  .github/workflows/actionlint.yml \
  .github/workflows/codeql.yml \
  .github/workflows/container-build-check.yml \
  .github/workflows/frontend-license-gate.yml \
  .github/workflows/frontend-ux-gates.yml \
  .github/workflows/publish-ghcr-main.yml \
  .github/workflows/publish-docker.yml \
  .github/workflows/sbom.yml \
  .github/workflows/vz-linux-host-gated.yml
cleanup_actionlint
trap - EXIT
```

Expected on this Apple Silicon worktree: actionlint 1.7.12 exits 0 with no SC2155 finding. If the host architecture differs, select the matching official 1.7.12 asset but keep the version and target list identical.

- [ ] **Step 8: Commit the independently reviewable workflow correction**

```bash
set -euo pipefail
git add .github/workflows/frontend-license-gate.yml \
  tldw_Server_API/tests/CI/test_frontend_license_gate_workflow.py
git diff --cached --check
git diff --cached --name-only
git commit -m "ci: satisfy actionlint for trusted license gate (TASK-12982)"
```

Expected: only the workflow and its contract test are committed; the three unrelated files remain untracked.

---

### Task 3: Acknowledge the OpenAPI ownership/license-metadata fingerprint

**Files:**
- Modify: `apps/tldw-frontend/lib/api/openapi.fingerprint.json:6`
- Read: `Helper_Scripts/export_openapi_schema.py`
- Read: `apps/tldw-frontend/scripts/generate-api-types.mjs`
- Test: `tldw_Server_API/tests/Services/test_openapi_contracts.py::test_openapi_contract_declares_contract_and_code_licenses`

**Interfaces:**
- Consumes: the canonical exporter, old fingerprint `f78a5c19071191c6667574a8a03c32bd964d7dd7dbcdc00ec03b16fc9c75b370`, hosted failure run `29865544853` / job `88776726144`, pre-`dev`-merge commit `7d76bdfcc0c467c779596cd6b92d2f078aa8529e`, and four newly emitted top-level `/info` properties from `tldw_Server_API/app/main.py`: `termsOfService`, `contact`, `license`, and `x-server-code-license`.
- Produces: hosted-CI-reproduced fingerprint `9a07fa34479c3fd6fcff06026295123117fee8d40dacb7c1537ecc21dbf7a4b1`, unchanged path/schema counts, whole-schema equality after deleting exactly those four properties, and byte-identical generated TypeScript before and after the metadata additions.

- [ ] **Step 1: Confirm the hosted schema-generator versions and the ambient mismatch**

```bash
set -euo pipefail
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
gh run view 29865544853 --repo rmusser01/tldw_server \
  --job 88776726144 --log | \
  rg 'pythonLocation:|fastapi==0\.136\.3|pydantic==2\.13\.4|pydantic-core==2\.46\.4|pydantic-settings==2\.14\.2'
python -c 'import platform,fastapi,pydantic,pydantic_core,pydantic_settings; print(platform.python_version(),fastapi.__version__,pydantic.__version__,pydantic_core.__version__,pydantic_settings.__version__)'
```

Expected: the immutable hosted log records Python 3.12.13, FastAPI 0.136.3, Pydantic 2.13.4, pydantic-core 2.46.4, and pydantic-settings 2.14.2. The ambient project environment is recorded separately; at planning time it was Python 3.11.13/Pydantic 2.11.7 and emitted 2,911 schemas, so it is not allowed to write this fingerprint.

- [ ] **Step 2: Create a disposable environment matching hosted schema generation**

```bash
set -euo pipefail
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
TLDW_OPENAPI_ROOT='/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/task-12982-openapi-env'
[[ ! -e "${TLDW_OPENAPI_ROOT}" && ! -L "${TLDW_OPENAPI_ROOT}" ]]
mkdir "${TLDW_OPENAPI_ROOT}"
[[ -d "${TLDW_OPENAPI_ROOT}" && ! -L "${TLDW_OPENAPI_ROOT}" && -O "${TLDW_OPENAPI_ROOT}" ]]
TLDW_OPENAPI_SETUP_COMPLETE=false
cleanup_incomplete_openapi_setup() {
  if [[ "${TLDW_OPENAPI_SETUP_COMPLETE}" != true ]]; then
    rm -rf -- "${TLDW_OPENAPI_ROOT}"
  fi
}
trap cleanup_incomplete_openapi_setup EXIT
uv venv --python /Users/macbook-dev/.local/bin/python3.12 \
  "${TLDW_OPENAPI_ROOT}/venv"
uv pip install --python "${TLDW_OPENAPI_ROOT}/venv/bin/python" \
  '/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/provider-credential-runtime[dev]' \
  'fastapi==0.136.3' \
  'pydantic==2.13.4' \
  'pydantic-core==2.46.4' \
  'pydantic-settings==2.14.2'
"${TLDW_OPENAPI_ROOT}/venv/bin/python" - <<'PY'
import sys
import fastapi
import pydantic
import pydantic_core
import pydantic_settings

assert sys.version_info[:2] == (3, 12)
assert fastapi.__version__ == "0.136.3"
assert pydantic.__version__ == "2.13.4"
assert pydantic_core.__version__ == "2.46.4"
assert pydantic_settings.__version__ == "2.14.2"
print(sys.version, fastapi.__version__, pydantic.__version__, pydantic_core.__version__, pydantic_settings.__version__)
PY
TLDW_OPENAPI_SETUP_COMPLETE=true
trap - EXIT
```

Expected: the exact non-symlink sibling directory `.worktrees/task-12982-openapi-env` is created, the interpreter is Python 3.12, and all four schema-affecting dependency versions match hosted CI. At planning time the available local patch was Python 3.12.11; the exact hosted fingerprint in Step 3 is the behavioral equivalence guard. A failed setup removes only that exact new directory. Do not install or upgrade packages in the project `.venv`.

- [ ] **Step 3: Reproduce the hosted drift gate RED in the CI-compatible environment**

```bash
set -euo pipefail
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
TLDW_OPENAPI_ROOT='/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/task-12982-openapi-env'
[[ -d "${TLDW_OPENAPI_ROOT}" && ! -L "${TLDW_OPENAPI_ROOT}" && -O "${TLDW_OPENAPI_ROOT}" ]]
TLDW_CI_PYTHON="${TLDW_OPENAPI_ROOT}/venv/bin/python"
if "${TLDW_CI_PYTHON}" Helper_Scripts/export_openapi_schema.py \
  --check apps/tldw-frontend/lib/api/openapi.fingerprint.json \
  2>&1 | tee "${TLDW_OPENAPI_ROOT}/drift-red.log"; then
  echo 'Expected the pre-refresh fingerprint check to fail' >&2
  exit 1
fi
rg -F 'checked-in sha256: f78a5c19071191c6667574a8a03c32bd964d7dd7dbcdc00ec03b16fc9c75b370' \
  "${TLDW_OPENAPI_ROOT}/drift-red.log"
rg -F 'current    sha256: 9a07fa34479c3fd6fcff06026295123117fee8d40dacb7c1537ecc21dbf7a4b1' \
  "${TLDW_OPENAPI_ROOT}/drift-red.log"
rg -F 'checked-in counts: paths=1999 schemas=2909' \
  "${TLDW_OPENAPI_ROOT}/drift-red.log"
rg -F 'current    counts: paths=1999 schemas=2909' \
  "${TLDW_OPENAPI_ROOT}/drift-red.log"
```

Expected: the check fails for exactly the hosted hash change, with both sides reporting 1,999 paths and 2,909 schemas.

- [ ] **Step 4: Prove the complete schema delta is exactly four newly added `/info` properties**

```bash
set -euo pipefail
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
TLDW_OPENAPI_ROOT='/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/task-12982-openapi-env'
[[ -d "${TLDW_OPENAPI_ROOT}" && ! -L "${TLDW_OPENAPI_ROOT}" && -O "${TLDW_OPENAPI_ROOT}" ]]
TLDW_CI_PYTHON="${TLDW_OPENAPI_ROOT}/venv/bin/python"
TLDW_PRE_DEV_WORKTREE="${TLDW_OPENAPI_ROOT}/pre-dev-worktree"
cleanup_pre_dev_on_exit() {
  git worktree remove --force "${TLDW_PRE_DEV_WORKTREE}" >/dev/null 2>&1 || true
}
trap cleanup_pre_dev_on_exit EXIT
git worktree add --detach "${TLDW_PRE_DEV_WORKTREE}" \
  7d76bdfcc0c467c779596cd6b92d2f078aa8529e
"${TLDW_CI_PYTHON}" \
  "${TLDW_PRE_DEV_WORKTREE}/Helper_Scripts/export_openapi_schema.py" \
  --out "${TLDW_OPENAPI_ROOT}/pre-dev.json" \
  --fingerprint "${TLDW_OPENAPI_ROOT}/pre-dev.fingerprint.json"
"${TLDW_CI_PYTHON}" Helper_Scripts/export_openapi_schema.py \
  --out "${TLDW_OPENAPI_ROOT}/current.json" \
  --fingerprint "${TLDW_OPENAPI_ROOT}/current.fingerprint.json"
git worktree remove --force "${TLDW_PRE_DEV_WORKTREE}"
trap - EXIT
export TLDW_OPENAPI_ROOT
"${TLDW_CI_PYTHON}" - <<'PY'
from copy import deepcopy
import json
import os
from pathlib import Path

root = Path(os.environ["TLDW_OPENAPI_ROOT"])
before = json.loads((root / "pre-dev.json").read_text())
after = json.loads((root / "current.json").read_text())
before_fp = json.loads((root / "pre-dev.fingerprint.json").read_text())
after_fp = json.loads((root / "current.fingerprint.json").read_text())
keys = ("termsOfService", "contact", "license", "x-server-code-license")
expected = {
    "termsOfService": "https://github.com/rmusser01/tldw_server",
    "contact": {
        "name": "tldw_server Maintainers",
        "url": "https://github.com/rmusser01/tldw_server/issues",
    },
    "license": {
        "name": "Apache License 2.0 (OpenAPI contract only)",
        "identifier": "Apache-2.0",
    },
    "x-server-code-license": "GPL-3.0-only",
}
assert all(key not in before["info"] for key in keys), before["info"]
assert {key: after["info"].get(key) for key in keys} == expected
stripped = deepcopy(after)
for key in keys:
    stripped["info"].pop(key)
assert stripped == before, "schema changed outside the four reviewed /info properties"
assert before_fp["sha256"] == "f78a5c19071191c6667574a8a03c32bd964d7dd7dbcdc00ec03b16fc9c75b370"
assert after_fp["sha256"] == "9a07fa34479c3fd6fcff06026295123117fee8d40dacb7c1537ecc21dbf7a4b1"
assert (before_fp["path_count"], before_fp["schema_count"]) == (1999, 2909)
assert (after_fp["path_count"], after_fp["schema_count"]) == (1999, 2909)
print(before_fp["sha256"], after_fp["sha256"], expected)
PY
```

Expected: the pre-merge schema lacks all four properties, the current schema has exactly the approved values, deleting them makes the complete schemas equal, and the fingerprints match the checked-in old and hosted new hashes. The old emitted schema did **not** contain cpacker/GPLv2 values; introducing those values would be a false reconstruction and must fail this proof.

- [ ] **Step 5: Verify the executable license contract in the matched environment**

```bash
set -euo pipefail
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
TLDW_OPENAPI_ROOT='/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/task-12982-openapi-env'
[[ -d "${TLDW_OPENAPI_ROOT}" && ! -L "${TLDW_OPENAPI_ROOT}" && -O "${TLDW_OPENAPI_ROOT}" ]]
"${TLDW_OPENAPI_ROOT}/venv/bin/python" -m pytest -q \
  tldw_Server_API/tests/Services/test_openapi_contracts.py::test_openapi_contract_declares_contract_and_code_licenses
```

Expected: PASS.

- [ ] **Step 6: Prove generated frontend declarations are unchanged by the four metadata additions**

```bash
set -euo pipefail
TLDW_OPENAPI_ROOT='/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/task-12982-openapi-env'
[[ -d "${TLDW_OPENAPI_ROOT}" && ! -L "${TLDW_OPENAPI_ROOT}" && -O "${TLDW_OPENAPI_ROOT}" ]]
cd apps/tldw-frontend
bun x openapi-typescript "${TLDW_OPENAPI_ROOT}/current.json" \
  -o "${TLDW_OPENAPI_ROOT}/current.d.ts"
bun x openapi-typescript "${TLDW_OPENAPI_ROOT}/pre-dev.json" \
  -o "${TLDW_OPENAPI_ROOT}/pre-dev.d.ts"
cmp "${TLDW_OPENAPI_ROOT}/current.d.ts" \
  "${TLDW_OPENAPI_ROOT}/pre-dev.d.ts"
shasum -a 256 "${TLDW_OPENAPI_ROOT}/current.d.ts" \
  "${TLDW_OPENAPI_ROOT}/pre-dev.d.ts"
cd ../..
```

Expected: `cmp` exits 0 and both declaration files have the same SHA-256. Therefore the four top-level metadata additions do not justify a hand-edited or committed frontend type change.

- [ ] **Step 7: Regenerate through the supported project command with the matched interpreter**

```bash
set -euo pipefail
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
TLDW_OPENAPI_ROOT='/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/task-12982-openapi-env'
[[ -d "${TLDW_OPENAPI_ROOT}" && ! -L "${TLDW_OPENAPI_ROOT}" && -O "${TLDW_OPENAPI_ROOT}" ]]
TLDW_CI_PYTHON="${TLDW_OPENAPI_ROOT}/venv/bin/python"
make CI_LOCAL_PYTHON="${TLDW_CI_PYTHON}" openapi-fingerprint
git diff -- apps/tldw-frontend/lib/api/openapi.fingerprint.json
"${TLDW_CI_PYTHON}" - <<'PY'
import json
from pathlib import Path

fingerprint = json.loads(Path("apps/tldw-frontend/lib/api/openapi.fingerprint.json").read_text())
assert fingerprint == {
    "note": "Regenerate with `make openapi-fingerprint`. A change here means the backend API contract drifted; regenerate the frontend types (`bun run generate:api-types` in apps/tldw-frontend) and review.",
    "openapi_version": "3.1.0",
    "path_count": 1999,
    "schema_count": 2909,
    "sha256": "9a07fa34479c3fd6fcff06026295123117fee8d40dacb7c1537ecc21dbf7a4b1",
}
PY
```

Expected: only `sha256` changes, from `f78a5c19071191c6667574a8a03c32bd964d7dd7dbcdc00ec03b16fc9c75b370` to `9a07fa34479c3fd6fcff06026295123117fee8d40dacb7c1537ecc21dbf7a4b1`; OpenAPI version 3.1.0, 1,999 paths, 2,909 schemas, and the note remain unchanged.

- [ ] **Step 8: Run the full supported type-generation command and confirm no tracked type artifact appears**

```bash
set -euo pipefail
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
TLDW_OPENAPI_ROOT='/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/task-12982-openapi-env'
[[ -d "${TLDW_OPENAPI_ROOT}" && ! -L "${TLDW_OPENAPI_ROOT}" && -O "${TLDW_OPENAPI_ROOT}" ]]
TLDW_CI_PYTHON="${TLDW_OPENAPI_ROOT}/venv/bin/python"
cd apps/tldw-frontend
PYTHON="${TLDW_CI_PYTHON}" bun run generate:api-types
cd ../..
git status --short -- apps/tldw-frontend/lib/api
make CI_LOCAL_PYTHON="${TLDW_CI_PYTHON}" openapi-drift-check
```

Expected: only `apps/tldw-frontend/lib/api/openapi.fingerprint.json` is tracked as modified; generated OpenAPI/TypeScript files remain ignored build artifacts; the drift check exits 0.

- [ ] **Step 9: Commit the independently reviewable fingerprint acknowledgement**

```bash
set -euo pipefail
git add apps/tldw-frontend/lib/api/openapi.fingerprint.json
git diff --cached --check
git diff --cached --name-only
git commit -m "chore: refresh OpenAPI licensing fingerprint (TASK-12982)"
```

Expected: the commit contains only the fingerprint JSON.

---

### Task 4: Perform focused local verification and independent review

**Files:**
- Test: `.github/workflows/frontend-license-gate.yml`
- Test: `tldw_Server_API/tests/CI/test_frontend_license_gate_workflow.py`
- Test: `tldw_Server_API/tests/CI/test_licensing_policy.py`
- Test: `tldw_Server_API/tests/Services/test_openapi_contracts.py`
- Read: all commits after `6065c64ab4a06687cc10e938eb0bd1cc5b6fd031`
- Modify: TASK-12982 implementation notes through Backlog.md CLI

**Interfaces:**
- Consumes: the two corrective commits and post-merge high-risk evidence already recorded at commit `6065c64ab4`.
- Produces: a clean local gate, an independent no-blocker review, a documented Bandit disposition, and a push-ready branch whose tracked tree contains no incidental files.

- [ ] **Step 1: Run the combined focused regression gate**

```bash
set -euo pipefail
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
TLDW_OPENAPI_ROOT='/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/task-12982-openapi-env'
[[ -d "${TLDW_OPENAPI_ROOT}" && ! -L "${TLDW_OPENAPI_ROOT}" && -O "${TLDW_OPENAPI_ROOT}" ]]
TLDW_CI_PYTHON="${TLDW_OPENAPI_ROOT}/venv/bin/python"
python -m pytest -q \
  tldw_Server_API/tests/CI/test_frontend_license_gate_workflow.py \
  tldw_Server_API/tests/CI/test_licensing_policy.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py::test_openapi_contract_declares_contract_and_code_licenses
make CI_LOCAL_PYTHON="${TLDW_CI_PYTHON}" openapi-drift-check
git diff --check 6065c64ab4a06687cc10e938eb0bd1cc5b6fd031..HEAD
```

Expected: all tests pass, the fingerprint matches, and diff hygiene is clean.

- [ ] **Step 2: Record the proportionate Bandit disposition**

Do not fabricate a new production scan for a YAML/test/JSON-only correction. Confirm the changed files first:

```bash
git diff --name-only 6065c64ab4a06687cc10e938eb0bd1cc5b6fd031..HEAD
```

Expected corrective scope: one workflow YAML, one CI contract test, one fingerprint JSON, approved design/plan/task records, and no production Python. Record that Bandit is not applicable to the landing corrections; retain the already committed post-merge result of 0 findings/0 errors across 8,898 production LOC and require the fresh `security-required` hosted gate.

- [ ] **Step 3: Request an independent correctness/security review**

Use `superpowers:requesting-code-review`. Give the reviewer the approved design, this plan, and exact range `6065c64ab4a06687cc10e938eb0bd1cc5b6fd031..HEAD`. Require explicit answers to:

1. Does the workflow remain fail closed and preserve its immutable-ref/classifier/status behavior?
2. Does the digest update match only the reviewed declaration change?
3. Is the OpenAPI fingerprint change proven to be exactly the intended repository terms, contact, API-contract license, and server-code license metadata?
4. Are any protected-path, unrelated-file, CI-bypass, or release-boundary changes present?

Expected: no unresolved Critical or Important issue. Reproduce and correct valid findings with focused RED/GREEN coverage; reject stale or incorrect findings with evidence.

- [ ] **Step 4: Recheck the exact local tree**

```bash
set -euo pipefail
git status --short --branch
git log --oneline --decorate 6065c64ab4a06687cc10e938eb0bd1cc5b6fd031..HEAD
git diff --stat 6065c64ab4a06687cc10e938eb0bd1cc5b6fd031..HEAD
git diff --name-only 6065c64ab4a06687cc10e938eb0bd1cc5b6fd031..HEAD
```

Expected: no tracked dirty path; only the three unrelated untracked files; all corrective and planning commits are explicit and reviewable.

- [ ] **Step 5: Append verification and review evidence to TASK-12982**

```bash
backlog task edit TASK-12982 --append-notes \
  'Landing corrections verified locally: trusted workflow contract and licensing policy tests pass; pinned actionlint 1.7.12 passes; the OpenAPI drift check passes in an isolated environment matching hosted Python 3.12/FastAPI/Pydantic versions. The pre-dev schema lacks termsOfService, contact, license, and x-server-code-license; deleting exactly those four newly emitted /info properties from the current schema makes the complete schemas equal, recovers fingerprint f78a5c190711, and leaves generated TypeScript byte-identical. Fingerprint 9a07fa34479c was reproduced independently from the failed hosted job. The ambient project venv is dependency-stale for fingerprint generation; separately pinning schema-generator dependencies remains recommended after landing. No production Python changed in this correction, so a new local Bandit scan is not applicable; prior post-merge Bandit evidence remains 0 findings/0 errors over 8,898 LOC and fresh security-required remains mandatory. Independent correction-range review found no unresolved blocker.' \
  --plain
```

Expected: evidence is appended; TASK-12982 remains In Progress.

- [ ] **Step 6: Commit the local verification record before the corrective push**

```bash
set -euo pipefail
git add 'backlog/tasks/task-12982 - Land-PR-2727-on-current-dev.md'
git diff --cached --check
git diff --cached --name-only
git commit -m "chore(backlog): record PR 2727 correction checks (TASK-12982)"
```

Expected: the commit contains only the updated TASK-12982 record. The tracked worktree is clean before Task 5.

Retain the disposable OpenAPI environment through Tasks 5-7. A review fix or newly advanced `dev` can require the Task 4 gate to run again; Task 8 removes the exact sibling directory only after the actual merge is verified.

---

### Task 5: Push once and drive fresh exact-head CI to terminal success

**Files:**
- Read: PR #2727 remote head and all current-head GitHub Actions checks
- Read: failed-job logs when applicable
- Modify: remote branch `codex/provider-credential-runtime-dev` by ordinary non-force push
- Modify: PR #2727 with a clearly machine-labeled CI evidence comment that does not alter the head

**Interfaces:**
- Consumes: a locally clean, independently reviewed branch descending from remote head `6065c64ab4`.
- Produces: one new immutable validated-head ref with trusted and ordinary hosted checks all terminal and non-blocking.

- [ ] **Step 1: Confirm a non-force push is safe**

```bash
set -euo pipefail
git fetch --no-tags origin codex/provider-credential-runtime-dev dev
git merge-base --is-ancestor origin/codex/provider-credential-runtime-dev HEAD
git status --short --branch
```

Expected: ancestry exits 0; no tracked dirty path; the three unrelated untracked files remain untouched. If the remote branch advanced independently, stop and reconcile rather than force-pushing.

- [ ] **Step 2: Push the complete reviewed head once**

```bash
set -euo pipefail
TLDW_CANDIDATE_HEAD="$(git rev-parse HEAD)"
[[ "${TLDW_CANDIDATE_HEAD}" =~ ^[0-9a-f]{40}$ ]]
git update-ref refs/tldw/task-12982/candidate "${TLDW_CANDIDATE_HEAD}"
git push origin "${TLDW_CANDIDATE_HEAD}":codex/provider-credential-runtime-dev
TLDW_REMOTE_HEAD="$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json headRefOid --jq .headRefOid)"
test "${TLDW_REMOTE_HEAD}" = "${TLDW_CANDIDATE_HEAD}"
git fetch --no-tags origin codex/provider-credential-runtime-dev
test "$(git rev-parse origin/codex/provider-credential-runtime-dev)" = \
  "${TLDW_CANDIDATE_HEAD}"
```

Expected: ordinary push succeeds; the local candidate ref, remote PR head, and fetched remote branch are the same 40-character SHA. Do not rely on any check from `6065c64ab4` afterward. Any concurrent push causes an equality assertion to fail rather than silently changing the reviewed candidate.

- [ ] **Step 3: Poll current-head checks without canceling other work**

At intervals no longer than 60 seconds, run:

```bash
set -euo pipefail
TLDW_CANDIDATE_HEAD="$(git rev-parse refs/tldw/task-12982/candidate)"
test "$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json headRefOid --jq .headRefOid)" = "${TLDW_CANDIDATE_HEAD}"
gh pr checks 2727 --repo rmusser01/tldw_server \
  --json bucket,name,state,workflow,link,startedAt,completedAt \
  --jq 'sort_by(.bucket,.workflow,.name)[] | [.bucket,.state,.workflow,.name,.link] | @tsv'
```

Expected while running: pending rows are allowed. Do not cancel unrelated runs to accelerate this PR. Continue until no current-head row is pending.

- [ ] **Step 4: Enforce the full check contract**

Require all of the following on the same head:

- `frontend-license-policy/trusted/dev` is success for the exact head.
- `backend-required`, `security-required`, `coverage-required`, `frontend-required`, `e2e-required`, and `container-build-check` are success when triggered, or an explicitly designed no-op success.
- actionlint is success.
- Every other current-head check is pass or intentional skip; none is fail, canceled, pending, or timed out.
- The old-head canceled Windows `research-websearch` job needs no action if the new head contains a successful equivalent. If the current head has no equivalent terminal success, rerun its current-head workflow/job and record the result.

Use this machine-readable rejection check:

```bash
set -euo pipefail
TLDW_CANDIDATE_HEAD="$(git rev-parse refs/tldw/task-12982/candidate)"
test "$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json headRefOid --jq .headRefOid)" = "${TLDW_CANDIDATE_HEAD}"
TLDW_CHECKS="$(gh pr checks 2727 --repo rmusser01/tldw_server \
  --json bucket,name,state,workflow,link)"
test "$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json headRefOid --jq .headRefOid)" = "${TLDW_CANDIDATE_HEAD}"
TLDW_CHECK_BLOCKERS="$(jq -c \
  '[.[] | select(.bucket == "fail" or .bucket == "cancel" or .bucket == "pending")]' \
  <<< "${TLDW_CHECKS}")"
printf '%s\n' "${TLDW_CHECK_BLOCKERS}"
test "${TLDW_CHECK_BLOCKERS}" = '[]'
for TLDW_REQUIRED_CHECK in \
  'frontend-license-policy/trusted/dev' \
  'backend-required' \
  'security-required' \
  'coverage-required' \
  'frontend-required' \
  'e2e-required' \
  'container-build-check' \
  'actionlint'; do
  jq -e --arg required "${TLDW_REQUIRED_CHECK}" \
    'any(.[]; .name == $required and .bucket == "pass")' \
    <<< "${TLDW_CHECKS}" >/dev/null
done
```

Expected final output: `[]`, and every named ordinary/trusted gate has at least one passing current-head row.

- [ ] **Step 5: Diagnose rather than bypass any current-head failure**

For each failed GitHub Actions check, derive its run ID from the check link and inspect the underlying failed jobs before an aggregate failure:

```bash
set -euo pipefail
TLDW_CANDIDATE_HEAD="$(git rev-parse refs/tldw/task-12982/candidate)"
while IFS=$'\t' read -r TLDW_FAILED_LINK TLDW_FAILED_NAME; do
  TLDW_FAILED_RUN_ID="$(printf '%s\n' "${TLDW_FAILED_LINK}" | \
    sed -E 's#.*actions/runs/([0-9]+).*#\1#')"
  case "${TLDW_FAILED_RUN_ID}" in
    ''|*[!0-9]*)
      printf 'External or non-Actions check requires provider-specific inspection: %s %s\n' \
        "${TLDW_FAILED_NAME}" "${TLDW_FAILED_LINK}"
      continue
      ;;
  esac
  TLDW_FAILED_RUN_HEAD="$(gh run view "${TLDW_FAILED_RUN_ID}" \
    --repo rmusser01/tldw_server --json headSha --jq .headSha)"
  test "${TLDW_FAILED_RUN_HEAD}" = "${TLDW_CANDIDATE_HEAD}"
  gh run view "${TLDW_FAILED_RUN_ID}" --repo rmusser01/tldw_server \
    --json headSha,status,conclusion,jobs
  gh run view "${TLDW_FAILED_RUN_ID}" --repo rmusser01/tldw_server --log-failed
done < <(
  gh pr checks 2727 --repo rmusser01/tldw_server \
    --json bucket,name,link \
    --jq '.[] | select(.bucket == "fail") | [.link,.name] | @tsv'
)
```

Expected: `headSha` equals the candidate exact head. Reproduce attributable failures locally, add a focused failing test, implement the smallest correction, review it, commit it, push non-force, and restart Tasks 4-5 for the new head. A reproduced protected-`dev` baseline failure still blocks this PR: correct it separately on `dev`, merge that fix into the PR through a clean integration worktree, and restart all exact-head gates.

For a current-head transient cancellation or infrastructure failure only after recording its evidence, require exactly one classified canceled Actions row before rerunning it:

```bash
set -euo pipefail
TLDW_TRANSIENT_ROW="$(gh pr checks 2727 --repo rmusser01/tldw_server \
  --json bucket,link,name \
  --jq '([.[] | select(.bucket == "cancel")] | sort_by(.name,.link) | first) as $row | if $row == null then empty else [$row.link,$row.name] | @tsv end')"
test -n "${TLDW_TRANSIENT_ROW}"
IFS=$'\t' read -r TLDW_TRANSIENT_LINK TLDW_TRANSIENT_NAME <<< \
  "${TLDW_TRANSIENT_ROW}"
printf 'Selected one canceled current-head row for classified rerun: %s %s\n' \
  "${TLDW_TRANSIENT_NAME}" "${TLDW_TRANSIENT_LINK}"
TLDW_TRANSIENT_RUN_ID="$(printf '%s\n' "${TLDW_TRANSIENT_LINK}" | \
  sed -E 's#.*actions/runs/([0-9]+).*#\1#')"
case "${TLDW_TRANSIENT_RUN_ID}" in
  ''|*[!0-9]*)
    printf 'Cannot rerun a non-Actions check with gh run: %s\n' "${TLDW_TRANSIENT_LINK}"
    exit 1
    ;;
esac
TLDW_CANDIDATE_HEAD="$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json headRefOid --jq .headRefOid)"
test "${TLDW_CANDIDATE_HEAD}" = \
  "$(git rev-parse refs/tldw/task-12982/candidate)"
TLDW_RUN_HEAD="$(gh run view "${TLDW_TRANSIENT_RUN_ID}" \
  --repo rmusser01/tldw_server --json headSha --jq .headSha)"
test "${TLDW_RUN_HEAD}" = "${TLDW_CANDIDATE_HEAD}"
export TLDW_TRANSIENT_LINK
TLDW_TRANSIENT_JOB_ID="$(gh run view "${TLDW_TRANSIENT_RUN_ID}" \
  --repo rmusser01/tldw_server --json jobs | \
  jq -r --arg link "${TLDW_TRANSIENT_LINK}" \
    '.jobs[] | select(.url == $link) | .databaseId')"
[[ "${TLDW_TRANSIENT_JOB_ID}" =~ ^[1-9][0-9]*$ ]]
gh run rerun "${TLDW_TRANSIENT_RUN_ID}" --repo rmusser01/tldw_server \
  --job "${TLDW_TRANSIENT_JOB_ID}"
```

Expected: use rerun only after the deterministically selected first canceled row has been explicitly classified as transient, needs no source correction, and lacks a successful equivalent. Its job URL resolves to one numeric `databaseId`, its run head equals the candidate PR head, and only that job plus GitHub-required dependencies reruns. When there are multiple canceled rows, wait for this rerun to finish, reevaluate the complete rollup, and execute the block again for at most one remaining row; never bulk-rerun them.

- [ ] **Step 6: Record exact-head hosted evidence without changing the head**

```bash
set -euo pipefail
git fetch --no-tags origin codex/provider-credential-runtime-dev
TLDW_CANDIDATE_HEAD="$(git rev-parse refs/tldw/task-12982/candidate)"
TLDW_VALIDATED_HEAD="$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json headRefOid --jq .headRefOid)"
[[ "${TLDW_VALIDATED_HEAD}" =~ ^[0-9a-f]{40}$ ]]
test "${TLDW_VALIDATED_HEAD}" = "${TLDW_CANDIDATE_HEAD}"
test "${TLDW_VALIDATED_HEAD}" = \
  "$(git rev-parse origin/codex/provider-credential-runtime-dev)"
git update-ref refs/tldw/task-12982/hosted-green "${TLDW_VALIDATED_HEAD}"
gh pr comment 2727 --repo rmusser01/tldw_server \
  --body "Machine-recorded TASK-12982 landing evidence: fresh hosted validation completed for exact PR head ${TLDW_VALIDATED_HEAD}. The trusted frontend-license status and all ordinary/current-head checks reached terminal non-blocking results. Underlying failures and any reruns were diagnosed at this exact head; no gate was bypassed and no unrelated run was canceled. The PR check rollup retains the immutable job URLs. This is CI evidence, not the requester-authored Change summary."
```

Expected: the local evidence ref and PR comment contain the actual 40-character validated SHA while the Git tree and PR head remain unchanged. Add a separate clearly machine-labeled PR comment with exact failure/rerun dispositions when Step 5 encountered any such event; durable Backlog evidence is added after merge in Task 8.

---

### Task 6: Satisfy the human-authorship and review gates, then mark ready

**Files:**
- Modify by requester only: PR #2727 `Change summary (human-authored)` section
- Read: PR body, reviews, review threads, head/base SHAs, and ready state
- Modify: PR #2727 draft state after all prerequisites pass

**Interfaces:**
- Consumes: one exact CI-green head and the repository's AI-generated-PR merge policy.
- Produces: a ready-for-review PR with requester-authored rationale, no active change request, no unresolved actionable thread, and successful ready-event checks on the same head.

- [ ] **Step 1: Stop for Robert's own Change summary**

Ask Robert Benjamin Jake Musser to edit the PR body directly and replace the instruction block under `## Change summary (human-authored)` with his own explanation of:

1. what changed at a meaningful level;
2. why Chat and Knowledge QA/RAG share one server-side credential runtime; and
3. why invalid/unavailable BYOK fails closed, credentials remain server-side, and unsafe stream replay is prohibited.

Do not propose wording, write the section, transform AI prose, or call `gh pr edit` on his behalf.

- [ ] **Step 2: Verify the human section is no longer the instruction template**

```bash
set -euo pipefail
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
TLDW_PR_BODY="$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json body --jq .body)"
export TLDW_PR_BODY
python - <<'PY'
import os
import re

body = os.environ["TLDW_PR_BODY"]
heading = "## Change summary (human-authored)"
assert body.count(heading) == 1, "required human-summary heading is missing or duplicated"
match = re.search(
    r"(?ms)^## Change summary \(human-authored\)\s*\n(.*?)(?=^##\s|\Z)",
    body,
)
assert match is not None
section = match.group(1)
visible = re.sub(r"<!--.*?-->", "", section, flags=re.S).strip()
assert visible, "human-summary section has no visible content"
for instruction in (
    "In your own words, explain:",
    "Required before this PR is marked ready or merged",
):
    assert instruction not in section, "instruction template is still present"
print(visible)
PY
```

Expected: one exact heading exists, its section contains visible non-template content, and the command prints that content for manual verification. Human ownership and substantive rationale remain policy judgments, not word-count loopholes.

- [ ] **Step 3: Recheck exact head and current `dev` before readying**

```bash
set -euo pipefail
git fetch --no-tags origin dev codex/provider-credential-runtime-dev
TLDW_PR_STATE="$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json headRefOid,baseRefOid,isDraft,mergeable,mergeStateStatus)"
printf '%s\n' "${TLDW_PR_STATE}"
TLDW_PR_HEAD="$(jq -r .headRefOid <<< "${TLDW_PR_STATE}")"
TLDW_PR_BASE="$(jq -r .baseRefOid <<< "${TLDW_PR_STATE}")"
TLDW_PR_DRAFT="$(jq -r .isDraft <<< "${TLDW_PR_STATE}")"
TLDW_PR_MERGEABLE="$(jq -r .mergeable <<< "${TLDW_PR_STATE}")"
TLDW_HOSTED_GREEN="$(git rev-parse refs/tldw/task-12982/hosted-green)"
TLDW_CANDIDATE_HEAD="$(git rev-parse refs/tldw/task-12982/candidate)"
[[ "${TLDW_PR_HEAD}" =~ ^[0-9a-f]{40}$ ]]
test "${TLDW_HOSTED_GREEN}" = "${TLDW_CANDIDATE_HEAD}"
test "${TLDW_PR_HEAD}" = "${TLDW_HOSTED_GREEN}"
test "$(git rev-parse origin/codex/provider-credential-runtime-dev)" = \
  "${TLDW_HOSTED_GREEN}"
test "${TLDW_PR_BASE}" = "$(git rev-parse origin/dev)"
test "${TLDW_PR_DRAFT}" = true
test "${TLDW_PR_MERGEABLE}" = MERGEABLE
git merge-base --is-ancestor origin/dev origin/codex/provider-credential-runtime-dev
```

Expected: PR head and the remote branch both equal the exact `hosted-green` ref; `baseRefOid` equals `origin/dev`; ancestry exits 0; PR remains draft and mergeable. If `dev` advanced or head changed, execute Task 7's clean freshness integration and then restart Tasks 4-6 before marking ready, updating `hosted-green` only after the replacement head completes the full hosted gate.

- [ ] **Step 4: Mark the PR ready without changing its head**

```bash
set -euo pipefail
TLDW_HOSTED_GREEN="$(git rev-parse refs/tldw/task-12982/hosted-green)"
test "${TLDW_HOSTED_GREEN}" = \
  "$(git rev-parse refs/tldw/task-12982/candidate)"
test "$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json headRefOid --jq .headRefOid)" = "${TLDW_HOSTED_GREEN}"
gh pr ready 2727 --repo rmusser01/tldw_server
TLDW_READY_STATE="$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json headRefOid,isDraft)"
printf '%s\n' "${TLDW_READY_STATE}"
if [[ "$(jq -r .headRefOid <<< "${TLDW_READY_STATE}")" != "${TLDW_HOSTED_GREEN}" ]]; then
  gh pr ready 2727 --repo rmusser01/tldw_server --undo
  echo 'PR head changed while marking ready; restored draft state' >&2
  exit 1
fi
test "$(jq -r .isDraft <<< "${TLDW_READY_STATE}")" = false
```

Expected: `isDraft` is false and the head SHA equals both candidate and `hosted-green`. A concurrent head change is detected and the command restores draft state before stopping.

- [ ] **Step 5: Wait for ready-event checks and reviews**

The trusted workflow includes `ready_for_review`, and review bots may start only after draft removal. Poll checks as in Task 5 and require a clean terminal rollup again on the same head.

Inspect review state:

```bash
set -euo pipefail
TLDW_HOSTED_GREEN="$(git rev-parse refs/tldw/task-12982/hosted-green)"
TLDW_REVIEW_STATE="$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json reviewDecision,reviews,headRefOid,isDraft,mergeable,mergeStateStatus)"
printf '%s\n' "${TLDW_REVIEW_STATE}"
test "$(jq -r .headRefOid <<< "${TLDW_REVIEW_STATE}")" = \
  "${TLDW_HOSTED_GREEN}"
test "$(jq -r .isDraft <<< "${TLDW_REVIEW_STATE}")" = false
test "$(jq -r .reviewDecision <<< "${TLDW_REVIEW_STATE}")" != \
  CHANGES_REQUESTED
```

Expected: no `CHANGES_REQUESTED`, no unresolved required review, same exact head, and no new blocking check.

- [ ] **Step 6: Inspect unresolved current review threads**

```bash
gh api graphql --paginate --slurp \
  -f owner='rmusser01' \
  -f name='tldw_server' \
  -F number=2727 \
  -f query='query($owner:String!,$name:String!,$number:Int!,$endCursor:String){repository(owner:$owner,name:$name){pullRequest(number:$number){reviewThreads(first:100,after:$endCursor){nodes{isResolved isOutdated comments(first:1){nodes{author{login}body path line}}}pageInfo{hasNextPage endCursor}}}}}' \
  --jq '[.[].data.repository.pullRequest.reviewThreads.nodes[] | select((.isResolved|not) and (.isOutdated|not))]'
```

Expected: `[]`, or only threads independently demonstrated non-actionable and explicitly dispositioned. Address valid findings with focused tests and a new head, then repeat Tasks 4-6. Do not resolve a thread merely to make the count zero.

- [ ] **Step 7: Restore draft state before any post-ready head change**

If a ready-event check, review, or thread requires a corrective commit, or Task 7 detects that `dev` advanced, run this guard before creating or pushing any new head:

```bash
set -euo pipefail
TLDW_PRE_CHANGE_HEAD="$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json headRefOid --jq .headRefOid)"
test "${TLDW_PRE_CHANGE_HEAD}" = \
  "$(git rev-parse refs/tldw/task-12982/candidate)"
TLDW_PRE_CHANGE_DRAFT="$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json isDraft --jq .isDraft)"
if test "${TLDW_PRE_CHANGE_DRAFT}" = false; then
  gh pr ready 2727 --repo rmusser01/tldw_server --undo
fi
TLDW_DRAFT_STATE="$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json headRefOid,isDraft)"
test "$(jq -r .headRefOid <<< "${TLDW_DRAFT_STATE}")" = \
  "${TLDW_PRE_CHANGE_HEAD}"
test "$(jq -r .isDraft <<< "${TLDW_DRAFT_STATE}")" = true
```

Expected: the same previously reviewed head is draft again before any correction or freshness push. After the new head is pushed, restart Tasks 4-6; no previous hosted, human-summary verification, review, ready-event, or check sample carries forward to the replacement head.

---

### Task 7: Reconcile any last `dev` advance and merge with exact-head protection

**Files:**
- Read: remote `dev`, PR head, PR base, required checks, and merge state
- Modify only if `dev` advanced: a clean temporary integration branch/worktree and the PR branch
- Modify: PR #2727 merge state
- Read: resulting merge commit and ancestry

**Interfaces:**
- Consumes: the ready, reviewed, human-documented, exact-head-green PR.
- Produces: a GitHub merge into `dev` whose actual merge commit contains both the final validated PR head and the exact protected `dev` tip used for final freshness.

- [ ] **Step 1: Require an atomic up-to-date-base rule before merging**

The active `frontend-license-gate-dev` repository ruleset is ID `19362594`. At planning time its required-status-check rule has `strict_required_status_checks_policy: false`, so `--match-head-commit` alone cannot prevent `dev` from advancing between final validation and merge. Do not rewrite this live multi-rule policy: GitHub does not support conditional requests for unsafe REST methods unless an endpoint explicitly documents them, so a GET→PUT update could overwrite a concurrent administrator change. Instead, use GitHub's documented ruleset layering—applicable rules are aggregated and the most restrictive form wins—to add one dedicated, permanent strict layer. Read and validate the existing rule first:

```bash
set -euo pipefail
TLDW_EXISTING_RULESET="$(gh api repos/rmusser01/tldw_server/rulesets/19362594)"
printf '%s\n' "${TLDW_EXISTING_RULESET}" | jq \
  '{id,name,target,enforcement,conditions,rules,bypass_actors}'
jq -e '
  .id == 19362594 and
  .name == "frontend-license-gate-dev" and
  .target == "branch" and
  .enforcement == "active" and
  .bypass_actors == [] and
  .conditions == {"ref_name":{"exclude":[],"include":["refs/heads/dev"]}} and
  ([.rules[] | select(.type == "pull_request")] | length) == 1 and
  ([.rules[] | select(
    .type == "required_status_checks" and
    .parameters.required_status_checks == [{"context":"frontend-license-policy/trusted/dev","integration_id":15368}]
  )] | length) == 1
' <<< "${TLDW_EXISTING_RULESET}" >/dev/null
```

Expected: active ruleset `frontend-license-gate-dev`, target `refs/heads/dev`, no bypass actors, its pull-request rule, and exactly one required `frontend-license-policy/trusted/dev` status bound to integration `15368`. If it differs, stop and review the live policy; this plan never overwrites it.

Stop and obtain Robert's explicit approval to create or retain the additive permanent ruleset `dev-up-to-date-required`. After approval, create it only when absent, then validate its complete policy. If a same-named rule already exists but is not byte-for-byte policy-equivalent after JSON normalization, stop rather than modifying it:

```bash
set -euo pipefail
TLDW_STRICT_PAYLOAD='{
  "name":"dev-up-to-date-required",
  "target":"branch",
  "enforcement":"active",
  "bypass_actors":[],
  "conditions":{"ref_name":{"exclude":[],"include":["refs/heads/dev"]}},
  "rules":[{
    "type":"required_status_checks",
    "parameters":{
      "strict_required_status_checks_policy":true,
      "do_not_enforce_on_create":false,
      "required_status_checks":[{
        "context":"frontend-license-policy/trusted/dev",
        "integration_id":15368
      }]
    }
  }]
}'
TLDW_STRICT_RULESET_IDS="$(gh api --paginate \
  repos/rmusser01/tldw_server/rulesets \
  --jq '.[] | select(.name == "dev-up-to-date-required") | .id')"
if [[ -z "${TLDW_STRICT_RULESET_IDS}" ]]; then
  TLDW_STRICT_RULESET_ID="$(printf '%s\n' "${TLDW_STRICT_PAYLOAD}" | \
    gh api --method POST repos/rmusser01/tldw_server/rulesets \
      --input - --jq .id)"
else
  [[ "${TLDW_STRICT_RULESET_IDS}" =~ ^[1-9][0-9]*$ ]]
  TLDW_STRICT_RULESET_ID="${TLDW_STRICT_RULESET_IDS}"
fi
[[ "${TLDW_STRICT_RULESET_ID}" =~ ^[1-9][0-9]*$ ]]
TLDW_STRICT_RULESET="$(gh api \
  "repos/rmusser01/tldw_server/rulesets/${TLDW_STRICT_RULESET_ID}")"
TLDW_EXPECTED_STRICT_POLICY="$(jq -Sc . <<< "${TLDW_STRICT_PAYLOAD}")"
TLDW_ACTUAL_STRICT_POLICY="$(jq -Sc \
  '{name,target,enforcement,bypass_actors,conditions,rules}' \
  <<< "${TLDW_STRICT_RULESET}")"
test "${TLDW_ACTUAL_STRICT_POLICY}" = "${TLDW_EXPECTED_STRICT_POLICY}"
printf 'strict-layer-ruleset-id=%s\n' "${TLDW_STRICT_RULESET_ID}"
```

Expected: one active, no-bypass ruleset targets only `dev`, requires the trusted status from integration `15368`, and sets strict up-to-date checking. The existing ruleset remains unchanged. Keep the additive strict layer enabled after landing; GitHub aggregates both policies and applies the more restrictive required-status rule.

- [ ] **Step 2: Capture the final candidate identities**

```bash
set -euo pipefail
git fetch --no-tags origin dev codex/provider-credential-runtime-dev
TLDW_IS_DRAFT="$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json isDraft --jq .isDraft)"
TLDW_MERGEABLE="$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json mergeable --jq .mergeable)"
TLDW_REVIEW_DECISION="$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json reviewDecision --jq .reviewDecision)"
test "${TLDW_IS_DRAFT}" = false
test "${TLDW_MERGEABLE}" = MERGEABLE
test "${TLDW_REVIEW_DECISION}" != CHANGES_REQUESTED
git rev-parse origin/dev
TLDW_CHECK_BLOCKERS="$(gh pr checks 2727 --repo rmusser01/tldw_server \
  --json bucket,name,state,workflow,link \
  --jq '[.[] | select(.bucket == "fail" or .bucket == "cancel" or .bucket == "pending")]')"
printf '%s\n' "${TLDW_CHECK_BLOCKERS}"
test "${TLDW_CHECK_BLOCKERS}" = '[]'
```

Expected: not draft, mergeable, no active change request, rejection array `[]`. Persist the exact identities in local task-specific refs before merge:

```bash
set -euo pipefail
TLDW_VALIDATED_HEAD="$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json headRefOid --jq .headRefOid)"
TLDW_VALIDATED_BASE="$(git rev-parse origin/dev)"
TLDW_HOSTED_GREEN="$(git rev-parse refs/tldw/task-12982/hosted-green)"
TLDW_CANDIDATE_HEAD="$(git rev-parse refs/tldw/task-12982/candidate)"
TLDW_PR_BASE="$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json baseRefOid --jq .baseRefOid)"
[[ "${TLDW_VALIDATED_HEAD}" =~ ^[0-9a-f]{40}$ ]]
[[ "${TLDW_VALIDATED_BASE}" =~ ^[0-9a-f]{40}$ ]]
test "${TLDW_HOSTED_GREEN}" = "${TLDW_CANDIDATE_HEAD}"
test "${TLDW_VALIDATED_HEAD}" = "${TLDW_HOSTED_GREEN}"
test "${TLDW_PR_BASE}" = "${TLDW_VALIDATED_BASE}"
git update-ref refs/tldw/task-12982/validated-head "${TLDW_VALIDATED_HEAD}"
git update-ref refs/tldw/task-12982/validated-base "${TLDW_VALIDATED_BASE}"
```

Expected: base equality and both `git update-ref` commands succeed. The refs preserve both full SHAs without creating a new PR head after exact-head validation; Task 8 writes them into the durable Backlog closeout.

- [ ] **Step 3: If and only if `dev` advanced, integrate it in a clean worktree**

If `git merge-base --is-ancestor origin/dev origin/codex/provider-credential-runtime-dev` exits 0, skip this step. Otherwise use `superpowers:using-git-worktrees` and create a clean temporary branch from the exact remote PR head:

```bash
set -euo pipefail
git fetch --no-tags origin dev codex/provider-credential-runtime-dev
TLDW_PRE_REFRESH_HEAD="$(git rev-parse \
  origin/codex/provider-credential-runtime-dev)"
test "$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json headRefOid --jq .headRefOid)" = "${TLDW_PRE_REFRESH_HEAD}"
TLDW_PRE_REFRESH_DRAFT="$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json isDraft --jq .isDraft)"
if test "${TLDW_PRE_REFRESH_DRAFT}" = false; then
  gh pr ready 2727 --repo rmusser01/tldw_server --undo
fi
TLDW_PRE_REFRESH_STATE="$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json headRefOid,isDraft)"
test "$(jq -r .headRefOid <<< "${TLDW_PRE_REFRESH_STATE}")" = \
  "${TLDW_PRE_REFRESH_HEAD}"
test "$(jq -r .isDraft <<< "${TLDW_PRE_REFRESH_STATE}")" = true
TLDW_FRESH_ROOT="$(mktemp -d)"
TLDW_FRESH_WORKTREE="${TLDW_FRESH_ROOT}/worktree"
TLDW_FRESH_COMPLETE=false
report_preserved_freshness_worktree() {
  if [[ "${TLDW_FRESH_COMPLETE}" != true ]]; then
    printf 'Freshness integration stopped; preserve and inspect exact temp root: %s\n' \
      "${TLDW_FRESH_ROOT}" >&2
  fi
}
trap report_preserved_freshness_worktree EXIT
git worktree add --detach "${TLDW_FRESH_WORKTREE}" origin/codex/provider-credential-runtime-dev
cd "${TLDW_FRESH_WORKTREE}"
git status --porcelain=v1
git merge --no-ff --no-edit origin/dev
git show -s --format='%H%n%P%n%s' HEAD
git diff --check HEAD^1..HEAD
git push origin HEAD:codex/provider-credential-runtime-dev
git status --porcelain=v1
cd /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/provider-credential-runtime
git fetch --no-tags origin codex/provider-credential-runtime-dev
git merge --ff-only origin/codex/provider-credential-runtime-dev
test "$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json headRefOid --jq .headRefOid)" = "$(git rev-parse HEAD)"
test "$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json isDraft --jq .isDraft)" = true
git worktree remove "${TLDW_FRESH_WORKTREE}"
rmdir "${TLDW_FRESH_ROOT}"
TLDW_FRESH_COMPLETE=true
trap - EXIT
```

Expected: the PR is restored to draft before the head changes; the worktree starts empty; merge first parent is the prior PR head and second parent is the newly fetched `origin/dev`; no conflict or incidental path is present; push is non-force. The successful temporary worktree and empty root are removed. A failed/conflicted integration deliberately preserves and prints its exact temporary root for diagnosis rather than deleting evidence; record and reconcile it before a retry. Then return to the main PR worktree, fetch, fast-forward its tracked branch only if safe around the three unrelated untracked files, and restart Tasks 4-7. Never merge immediately on stale checks.

- [ ] **Step 4: Submit the merge with base- and head-SHA guards**

After the final freshness check and all exact-head/ready-event gates pass:

```bash
set -euo pipefail
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
TLDW_VALIDATED_HEAD="$(git rev-parse refs/tldw/task-12982/validated-head)"
TLDW_VALIDATED_BASE="$(git rev-parse refs/tldw/task-12982/validated-base)"
TLDW_HOSTED_GREEN="$(git rev-parse refs/tldw/task-12982/hosted-green)"
TLDW_CANDIDATE_HEAD="$(git rev-parse refs/tldw/task-12982/candidate)"
test "${TLDW_HOSTED_GREEN}" = "${TLDW_CANDIDATE_HEAD}"
test "${TLDW_VALIDATED_HEAD}" = "${TLDW_HOSTED_GREEN}"
git fetch --no-tags origin dev codex/provider-credential-runtime-dev
test "$(git rev-parse origin/dev)" = "${TLDW_VALIDATED_BASE}"
test "$(gh pr view 2727 --repo rmusser01/tldw_server --json headRefOid --jq .headRefOid)" = \
  "${TLDW_VALIDATED_HEAD}"
test "$(gh pr view 2727 --repo rmusser01/tldw_server --json baseRefOid --jq .baseRefOid)" = \
  "${TLDW_VALIDATED_BASE}"
TLDW_STRICT_RULESET_IDS="$(gh api --paginate \
  repos/rmusser01/tldw_server/rulesets \
  --jq '.[] | select(.name == "dev-up-to-date-required") | .id')"
[[ "${TLDW_STRICT_RULESET_IDS}" =~ ^[1-9][0-9]*$ ]]
TLDW_STRICT_RULESET="$(gh api \
  "repos/rmusser01/tldw_server/rulesets/${TLDW_STRICT_RULESET_IDS}")"
jq -e '
  .name == "dev-up-to-date-required" and
  .target == "branch" and
  .enforcement == "active" and
  .bypass_actors == [] and
  .conditions == {"ref_name":{"exclude":[],"include":["refs/heads/dev"]}} and
  .rules == [{
    "type":"required_status_checks",
    "parameters":{
      "strict_required_status_checks_policy":true,
      "do_not_enforce_on_create":false,
      "required_status_checks":[{
        "context":"frontend-license-policy/trusted/dev",
        "integration_id":15368
      }]
    }
  }]
' <<< "${TLDW_STRICT_RULESET}" >/dev/null

TLDW_FINAL_CHECKS="$(gh pr checks 2727 --repo rmusser01/tldw_server \
  --json bucket,name,state,workflow,link)"
test "$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json headRefOid --jq .headRefOid)" = "${TLDW_VALIDATED_HEAD}"
TLDW_FINAL_CHECK_BLOCKERS="$(jq -c \
  '[.[] | select(.bucket == "fail" or .bucket == "cancel" or .bucket == "pending")]' \
  <<< "${TLDW_FINAL_CHECKS}")"
printf '%s\n' "${TLDW_FINAL_CHECK_BLOCKERS}"
test "${TLDW_FINAL_CHECK_BLOCKERS}" = '[]'
for TLDW_REQUIRED_CHECK in \
  'frontend-license-policy/trusted/dev' \
  'backend-required' \
  'security-required' \
  'coverage-required' \
  'frontend-required' \
  'e2e-required' \
  'container-build-check' \
  'actionlint'; do
  jq -e --arg required "${TLDW_REQUIRED_CHECK}" \
    'any(.[]; .name == $required and .bucket == "pass")' \
    <<< "${TLDW_FINAL_CHECKS}" >/dev/null
done

TLDW_FINAL_THREADS="$(gh api graphql --paginate --slurp \
  -f owner='rmusser01' \
  -f name='tldw_server' \
  -F number=2727 \
  -f query='query($owner:String!,$name:String!,$number:Int!,$endCursor:String){repository(owner:$owner,name:$name){pullRequest(number:$number){reviewThreads(first:100,after:$endCursor){nodes{isResolved isOutdated}pageInfo{hasNextPage endCursor}}}}}' \
  --jq '[.[].data.repository.pullRequest.reviewThreads.nodes[] | select((.isResolved|not) and (.isOutdated|not))]')"
printf '%s\n' "${TLDW_FINAL_THREADS}"
test "${TLDW_FINAL_THREADS}" = '[]'

TLDW_FINAL_PR_STATE="$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json headRefOid,baseRefOid,isDraft,mergeable,reviewDecision,body)"
test "$(jq -r .headRefOid <<< "${TLDW_FINAL_PR_STATE}")" = \
  "${TLDW_VALIDATED_HEAD}"
test "$(jq -r .baseRefOid <<< "${TLDW_FINAL_PR_STATE}")" = \
  "${TLDW_VALIDATED_BASE}"
test "$(jq -r .isDraft <<< "${TLDW_FINAL_PR_STATE}")" = false
test "$(jq -r .mergeable <<< "${TLDW_FINAL_PR_STATE}")" = MERGEABLE
test "$(jq -r .reviewDecision <<< "${TLDW_FINAL_PR_STATE}")" != \
  CHANGES_REQUESTED
TLDW_FINAL_PR_BODY="$(jq -r .body <<< "${TLDW_FINAL_PR_STATE}")"
export TLDW_FINAL_PR_BODY
python - <<'PY'
import os
import re

body = os.environ["TLDW_FINAL_PR_BODY"]
heading = "## Change summary (human-authored)"
assert body.count(heading) == 1, "required human-summary heading is missing or duplicated"
match = re.search(
    r"(?ms)^## Change summary \(human-authored\)\s*\n(.*?)(?=^##\s|\Z)",
    body,
)
assert match is not None
section = match.group(1)
visible = re.sub(r"<!--.*?-->", "", section, flags=re.S).strip()
assert visible, "human-summary section has no visible content"
for instruction in (
    "In your own words, explain:",
    "Required before this PR is marked ready or merged",
):
    assert instruction not in section, "instruction template is still present"
print(visible)
PY
test "$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json headRefOid --jq .headRefOid)" = "${TLDW_VALIDATED_HEAD}"
test "$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json baseRefOid --jq .baseRefOid)" = "${TLDW_VALIDATED_BASE}"
gh pr merge 2727 --repo rmusser01/tldw_server \
  --merge \
  --match-head-commit "${TLDW_VALIDATED_HEAD}"
```

Do not add `--admin`, `--squash`, or `--rebase`.

Expected: immediately before submission, the exact head/base, human-written summary, ready/review state, zero current unresolved threads, full named check contract, and additive strict ruleset all pass again. GitHub then performs an ordinary merge only while the PR is up to date with the captured base and the head still matches. If GitHub reports a merge queue instead, stop: capture the merge-group head and add its exact queue-check verification before continuing rather than assuming submission equals merge.

- [ ] **Step 5: Read the actual merge result from GitHub**

```bash
set -euo pipefail
TLDW_MERGE_RESULT="$(gh pr view 2727 --repo rmusser01/tldw_server \
  --json state,mergedAt,mergeCommit,headRefOid,baseRefOid,url)"
printf '%s\n' "${TLDW_MERGE_RESULT}"
test "$(jq -r .state <<< "${TLDW_MERGE_RESULT}")" = MERGED
git fetch --no-tags origin dev
TLDW_MERGE_SHA="$(jq -r .mergeCommit.oid <<< "${TLDW_MERGE_RESULT}")"
[[ "${TLDW_MERGE_SHA}" =~ ^[0-9a-f]{40}$ ]]
git merge-base --is-ancestor "${TLDW_MERGE_SHA}" origin/dev
git update-ref refs/tldw/task-12982/merge "${TLDW_MERGE_SHA}"
```

Expected: state `MERGED`, a non-null `mergeCommit.oid`, and `origin/dev` contains that commit. The exact actual merge is persisted at `refs/tldw/task-12982/merge`; never infer it from a local simulation.

- [ ] **Step 6: Verify exact ordinary-merge parents and ancestry**

```bash
set -euo pipefail
TLDW_VALIDATED_HEAD="$(git rev-parse refs/tldw/task-12982/validated-head)"
TLDW_VALIDATED_BASE="$(git rev-parse refs/tldw/task-12982/validated-base)"
TLDW_MERGE_SHA="$(git rev-parse refs/tldw/task-12982/merge)"
git merge-base --is-ancestor "${TLDW_VALIDATED_HEAD}" "${TLDW_MERGE_SHA}"
git merge-base --is-ancestor "${TLDW_VALIDATED_BASE}" "${TLDW_MERGE_SHA}"
git merge-base --is-ancestor "${TLDW_MERGE_SHA}" origin/dev
TLDW_MERGE_PARENTS="$(git show -s --format='%P' "${TLDW_MERGE_SHA}")"
test "${TLDW_MERGE_PARENTS}" = "${TLDW_VALIDATED_BASE} ${TLDW_VALIDATED_HEAD}"
git show -s --format='%H%n%P%n%s' "${TLDW_MERGE_SHA}"
```

Expected: all ancestry checks exit 0 and the actual GitHub merge's first parent is exactly the validated `dev` base while its second parent is exactly the validated PR head. A mismatch is a landing failure and blocks TASK-12982 closeout.

---

### Task 8: Verify licensing metadata at the merge and close TASK-12982

**Files:**
- Read from the actual merge ref: `LICENSE`, `LICENSES/README.md`, `LICENSES/releases/README.md`, `LICENSES/PolyForm-Perimeter-1.0.1.txt`, `LICENSES/AGPL-3.0-only.txt`, `.github/workflows/frontend-license-gate.yml`, `Helper_Scripts/ci/check_frontend_license_gate.py`, and `tldw_Server_API/app/main.py`
- Modify through Backlog.md CLI: TASK-12982 status, evidence, acceptance criteria, Definition of Done, and final summary
- Create after merge: a narrow TASK-12982 closeout branch/PR if protected `dev` does not accept the tracking commit directly

**Interfaces:**
- Consumes: verified local refs `refs/tldw/task-12982/merge`, `refs/tldw/task-12982/validated-head`, and `refs/tldw/task-12982/validated-base`.
- Produces: durable proof that the merged tree retains the approved license boundaries and trusted gate, a completed TASK-12982 record, and an explicit handoff that allows planning—not deployment—for TASK-12983.

- [ ] **Step 1: Verify the merged license scope map and corpus**

```bash
set -euo pipefail
TLDW_MERGE_SHA="$(git rev-parse refs/tldw/task-12982/merge)"
TLDW_LICENSE_SCOPE="$(git show "${TLDW_MERGE_SHA}:LICENSE")"
for TLDW_REQUIRED_SCOPE in \
  'PolyForm Perimeter License 1.0.1' \
  'AGPL-3.0-only' \
  'GPL-3.0-only' \
  'Apache-2.0' \
  'admin-ui/**' \
  'apps/tldw-frontend/**' \
  'apps/extension/**' \
  'apps/packages/ui/**'; do
  [[ "${TLDW_LICENSE_SCOPE}" == *"${TLDW_REQUIRED_SCOPE}"* ]]
done
git cat-file -e "${TLDW_MERGE_SHA}:LICENSES/PolyForm-Perimeter-1.0.1.txt"
git cat-file -e "${TLDW_MERGE_SHA}:LICENSES/AGPL-3.0-only.txt"
git cat-file -e "${TLDW_MERGE_SHA}:LICENSES/GPL-3.0-only.txt"
git cat-file -e "${TLDW_MERGE_SHA}:LICENSES/Apache-2.0.txt"
TLDW_RELEASE_POLICY="$(git show "${TLDW_MERGE_SHA}:LICENSES/releases/README.md")"
[[ "${TLDW_RELEASE_POLICY}" == *'No protected frontend release may be published'* ]]
[[ "${TLDW_RELEASE_POLICY}" == *'uncompleted template or this README is not a Countdown grant'* ]]
TLDW_RELEASE_FILES="$(git ls-tree --name-only "${TLDW_MERGE_SHA}:LICENSES/releases")"
test "${TLDW_RELEASE_FILES}" = README.md
```

Expected: all scoped licenses and protected paths are named; all legal texts exist; `LICENSES/releases` contains only `README.md`, so this landing did not create a protected release or Countdown grant.

- [ ] **Step 2: Verify the merged trusted gate and OpenAPI license metadata**

```bash
set -euo pipefail
TLDW_MERGE_SHA="$(git rev-parse refs/tldw/task-12982/merge)"
git cat-file -e "${TLDW_MERGE_SHA}:.github/workflows/frontend-license-gate.yml"
git cat-file -e "${TLDW_MERGE_SHA}:Helper_Scripts/ci/check_frontend_license_gate.py"
TLDW_MAIN_SOURCE="$(git show "${TLDW_MERGE_SHA}:tldw_Server_API/app/main.py")"
for TLDW_REQUIRED_OPENAPI_TEXT in \
  'https://github.com/rmusser01/tldw_server' \
  'https://github.com/rmusser01/tldw_server/issues' \
  'Apache License 2.0 (OpenAPI contract only)' \
  'x-server-code-license' \
  'GPL-3.0-only'; do
  [[ "${TLDW_MAIN_SOURCE}" == *"${TLDW_REQUIRED_OPENAPI_TEXT}"* ]]
done
TLDW_FINGERPRINT="$(git show "${TLDW_MERGE_SHA}:apps/tldw-frontend/lib/api/openapi.fingerprint.json")"
[[ "${TLDW_FINGERPRINT}" == *'9a07fa34479c3fd6fcff06026295123117fee8d40dacb7c1537ecc21dbf7a4b1'* ]]
```

Expected: trusted-policy files exist, the canonical OpenAPI document advertises Apache-2.0 for the contract and GPL-3.0-only for server code, and the reviewed fingerprint is merged.

- [ ] **Step 3: Remove only the disposable OpenAPI environment after merge verification**

```bash
set -euo pipefail
TLDW_OPENAPI_ROOT='/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/task-12982-openapi-env'
test "${TLDW_OPENAPI_ROOT}" = \
  '/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/task-12982-openapi-env'
[[ -d "${TLDW_OPENAPI_ROOT}" && ! -L "${TLDW_OPENAPI_ROOT}" && -O "${TLDW_OPENAPI_ROOT}" ]]
[[ -x "${TLDW_OPENAPI_ROOT}/venv/bin/python" ]]
if git worktree list --porcelain | rg -Fq "worktree ${TLDW_OPENAPI_ROOT}/"; then
  echo 'A temporary child worktree remains registered; remove or preserve it explicitly first' >&2
  exit 1
fi
rm -rf -- "${TLDW_OPENAPI_ROOT}"
[[ ! -e "${TLDW_OPENAPI_ROOT}" && ! -L "${TLDW_OPENAPI_ROOT}" ]]
```

Expected: only the exact owned, non-symlink sibling directory created in Task 3 is removed; the repository worktree and three unrelated files are untouched.

- [ ] **Step 4: Create a clean closeout worktree from merged `dev`**

Use `superpowers:using-git-worktrees` so the original untracked files remain untouched:

```bash
set -euo pipefail
git fetch --no-tags origin dev
TLDW_CLOSEOUT_WORKTREE='/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/task-12982-closeout'
test ! -e "${TLDW_CLOSEOUT_WORKTREE}"
if git show-ref --verify --quiet refs/heads/codex/task-12982-closeout; then
  printf 'Local closeout branch already exists; reconcile it before continuing.\n'
  exit 1
fi
if git show-ref --verify --quiet refs/tldw/task-12982/closeout-candidate; then
  printf 'Closeout candidate ref already exists; reconcile it before continuing.\n'
  exit 1
fi
git worktree add -b codex/task-12982-closeout "${TLDW_CLOSEOUT_WORKTREE}" origin/dev
cd "${TLDW_CLOSEOUT_WORKTREE}"
git status --porcelain=v1
```

Expected: clean closeout worktree based on the verified merged `dev`.

- [ ] **Step 5: Finalize TASK-12982 with exact evidence**

```bash
set -euo pipefail
cd /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/task-12982-closeout
TLDW_VALIDATED_HEAD="$(git rev-parse refs/tldw/task-12982/validated-head)"
TLDW_VALIDATED_BASE="$(git rev-parse refs/tldw/task-12982/validated-base)"
TLDW_MERGE_SHA="$(git rev-parse refs/tldw/task-12982/merge)"
TLDW_STRICT_RULESET_ID="$(gh api --paginate \
  repos/rmusser01/tldw_server/rulesets \
  --jq '.[] | select(.name == "dev-up-to-date-required") | .id')"
[[ "${TLDW_STRICT_RULESET_ID}" =~ ^[1-9][0-9]*$ ]]
backlog task edit TASK-12982 \
  --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --append-notes "Final hosted and merge verification: exact head ${TLDW_VALIDATED_HEAD} passed the trusted frontend-license status and every ordinary/current-head gate before merge; validated protected-dev base ${TLDW_VALIDATED_BASE} and that head are ancestors of actual merge ${TLDW_MERGE_SHA}, which is present on origin/dev with those exact two parents. Additive ruleset dev-up-to-date-required (${TLDW_STRICT_RULESET_ID}) permanently requires strict up-to-date trusted status checks without rewriting existing ruleset 19362594. The merged tree retains the root multi-license scope map, exact license corpus, empty release-record directory except README, trusted frontend license workflow/classifier, current repository/contact metadata, Apache-2.0 OpenAPI contract declaration, GPL-3.0-only server declaration, and reviewed OpenAPI fingerprint. No protected artifact or customer deployment was published." \
  --final-summary 'PR #2727 was brought onto current dev, repaired only for reproduced actionlint and OpenAPI fingerprint gates, validated on one exact final head and strict up-to-date base, supplied with the requester-authored Change summary, reviewed, and merged into dev. The actual two-parent merge lineage and licensing/trusted-policy files were verified. TASK-12983 may now be planned separately; no pilot deployment occurred under TASK-12982.' \
  --status Done \
  --plain
```

Expected: the command reads all three exact identities from the preserved task refs, checks every acceptance/Done item, and marks TASK-12982 Done.

- [ ] **Step 6: Commit the durable closeout record**

```bash
set -euo pipefail
cd /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/task-12982-closeout
git add 'backlog/tasks/task-12982 - Land-PR-2727-on-current-dev.md'
git diff --cached --check
git diff --cached --name-only
git commit -m "chore(backlog): close PR 2727 landing task (TASK-12982)"
TLDW_CLOSEOUT_CANDIDATE="$(git rev-parse HEAD)"
[[ "${TLDW_CLOSEOUT_CANDIDATE}" =~ ^[0-9a-f]{40}$ ]]
test "$(git diff-tree --no-commit-id --name-only -r "${TLDW_CLOSEOUT_CANDIDATE}")" = \
  'backlog/tasks/task-12982 - Land-PR-2727-on-current-dev.md'
git update-ref refs/tldw/task-12982/closeout-candidate \
  "${TLDW_CLOSEOUT_CANDIDATE}" \
  0000000000000000000000000000000000000000
git push -u origin codex/task-12982-closeout
git fetch --no-tags origin codex/task-12982-closeout
test "$(git rev-parse origin/codex/task-12982-closeout)" = \
  "${TLDW_CLOSEOUT_CANDIDATE}"
```

Expected: one task-record file, an immutable local ref pins the reviewed commit, the ordinary push resolves to that exact commit, and no unrelated artifact is included.

- [ ] **Step 7: Land the tracking-only closeout through normal protection**

```bash
set -euo pipefail
gh pr create --repo rmusser01/tldw_server \
  --base dev \
  --head codex/task-12982-closeout \
  --draft \
  --title 'chore(backlog): close PR 2727 landing task (TASK-12982)' \
  --body '## Closeout scope

Records the exact validated head, protected dev base, actual merge SHA, gate results, and merged licensing-policy verification for completed TASK-12982. No runtime, frontend, artifact, or deployment behavior changes.'
TLDW_CLOSEOUT_CANDIDATE="$(git rev-parse \
  refs/tldw/task-12982/closeout-candidate)"
TLDW_CLOSEOUT_PRS="$(gh pr list --repo rmusser01/tldw_server \
  --base dev --head codex/task-12982-closeout --state open \
  --json number,baseRefName)"
test "$(jq 'length' <<< "${TLDW_CLOSEOUT_PRS}")" -eq 1
test "$(jq -r '.[0].baseRefName' <<< "${TLDW_CLOSEOUT_PRS}")" = dev
TLDW_CLOSEOUT_PR="$(jq -r '.[0].number' <<< "${TLDW_CLOSEOUT_PRS}")"
[[ "${TLDW_CLOSEOUT_PR}" =~ ^[1-9][0-9]*$ ]]
test "$(gh pr view "${TLDW_CLOSEOUT_PR}" --repo rmusser01/tldw_server \
  --json headRefOid --jq .headRefOid)" = "${TLDW_CLOSEOUT_CANDIDATE}"
```

Expected: a narrow one-file draft closeout PR whose remote head is the pinned candidate. Preserve the original human Change summary in #2727 as the authoritative product rationale.

- [ ] **Step 8: Satisfy the closeout PR's human-authorship gate**

Because the closeout record and PR were agent-produced, ask Robert to add a concise `## Change summary (human-authored)` section in his own words explaining that this PR records the verified #2727 merge and why the durable Backlog closeout is needed. The agent must not supply or edit that language.

Verify the section exists and is substantive, then mark the closeout ready:

```bash
set -euo pipefail
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
TLDW_CLOSEOUT_PRS="$(gh pr list --repo rmusser01/tldw_server \
  --base dev --head codex/task-12982-closeout --state open \
  --json number,baseRefName)"
test "$(jq 'length' <<< "${TLDW_CLOSEOUT_PRS}")" -eq 1
test "$(jq -r '.[0].baseRefName' <<< "${TLDW_CLOSEOUT_PRS}")" = dev
TLDW_CLOSEOUT_PR="$(jq -r '.[0].number' <<< "${TLDW_CLOSEOUT_PRS}")"
[[ "${TLDW_CLOSEOUT_PR}" =~ ^[1-9][0-9]*$ ]]
TLDW_CLOSEOUT_CANDIDATE="$(git rev-parse \
  refs/tldw/task-12982/closeout-candidate)"
test "$(gh pr view "${TLDW_CLOSEOUT_PR}" --repo rmusser01/tldw_server \
  --json headRefOid --jq .headRefOid)" = "${TLDW_CLOSEOUT_CANDIDATE}"
TLDW_CLOSEOUT_BODY="$(gh pr view "${TLDW_CLOSEOUT_PR}" \
  --repo rmusser01/tldw_server --json body --jq .body)"
export TLDW_CLOSEOUT_BODY
python - <<'PY'
import os
import re

body = os.environ["TLDW_CLOSEOUT_BODY"]
heading = "## Change summary (human-authored)"
assert body.count(heading) == 1, "required human-summary heading is missing or duplicated"
match = re.search(
    r"(?ms)^## Change summary \(human-authored\)\s*\n(.*?)(?=^##\s|\Z)",
    body,
)
assert match is not None
section = match.group(1)
visible = re.sub(r"<!--.*?-->", "", section, flags=re.S).strip()
assert visible, "human-summary section has no visible content"
for instruction in (
    "In your own words, explain:",
    "Required before this PR is marked ready or merged",
):
    assert instruction not in section, "instruction template is still present"
print(visible)
PY
test "$(gh pr view "${TLDW_CLOSEOUT_PR}" --repo rmusser01/tldw_server \
  --json baseRefName --jq .baseRefName)" = dev
gh pr ready "${TLDW_CLOSEOUT_PR}" --repo rmusser01/tldw_server
if test "$(gh pr view "${TLDW_CLOSEOUT_PR}" --repo rmusser01/tldw_server \
  --json headRefOid --jq .headRefOid)" != "${TLDW_CLOSEOUT_CANDIDATE}"; then
  gh pr ready "${TLDW_CLOSEOUT_PR}" --repo rmusser01/tldw_server --undo
  printf 'Closeout head changed while marking ready; draft status restored.\n' >&2
  exit 1
fi
```

Expected: the body contains Robert's own summary in addition to the factual closeout scope, the PR still points to the pinned candidate, and the PR is no longer draft.

- [ ] **Step 9: Run normal protection and merge the closeout record**

```bash
set -euo pipefail
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
cd /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/task-12982-closeout
git fetch --no-tags origin dev codex/task-12982-closeout
TLDW_CLOSEOUT_PRS="$(gh pr list --repo rmusser01/tldw_server \
  --base dev --head codex/task-12982-closeout --state open \
  --json number,baseRefName)"
test "$(jq 'length' <<< "${TLDW_CLOSEOUT_PRS}")" -eq 1
test "$(jq -r '.[0].baseRefName' <<< "${TLDW_CLOSEOUT_PRS}")" = dev
TLDW_CLOSEOUT_PR="$(jq -r '.[0].number' <<< "${TLDW_CLOSEOUT_PRS}")"
[[ "${TLDW_CLOSEOUT_PR}" =~ ^[1-9][0-9]*$ ]]
TLDW_CLOSEOUT_CANDIDATE="$(git rev-parse \
  refs/tldw/task-12982/closeout-candidate)"
[[ "${TLDW_CLOSEOUT_CANDIDATE}" =~ ^[0-9a-f]{40}$ ]]
test "$(git rev-parse HEAD)" = "${TLDW_CLOSEOUT_CANDIDATE}"
test "$(git rev-parse origin/codex/task-12982-closeout)" = \
  "${TLDW_CLOSEOUT_CANDIDATE}"
test "$(gh pr view "${TLDW_CLOSEOUT_PR}" --repo rmusser01/tldw_server \
  --json headRefOid --jq .headRefOid)" = "${TLDW_CLOSEOUT_CANDIDATE}"
test -z "$(git status --porcelain=v1)"

if ! git merge-base --is-ancestor origin/dev "${TLDW_CLOSEOUT_CANDIDATE}"; then
  TLDW_PREVIOUS_CLOSEOUT_CANDIDATE="${TLDW_CLOSEOUT_CANDIDATE}"
  TLDW_CLOSEOUT_PRE_REFRESH_DRAFT="$(gh pr view "${TLDW_CLOSEOUT_PR}" \
    --repo rmusser01/tldw_server --json isDraft --jq .isDraft)"
  if test "${TLDW_CLOSEOUT_PRE_REFRESH_DRAFT}" = false; then
    gh pr ready "${TLDW_CLOSEOUT_PR}" --repo rmusser01/tldw_server --undo
  fi
  TLDW_CLOSEOUT_PRE_REFRESH_STATE="$(gh pr view "${TLDW_CLOSEOUT_PR}" \
    --repo rmusser01/tldw_server --json headRefOid,isDraft)"
  test "$(jq -r .headRefOid <<< "${TLDW_CLOSEOUT_PRE_REFRESH_STATE}")" = \
    "${TLDW_PREVIOUS_CLOSEOUT_CANDIDATE}"
  test "$(jq -r .isDraft <<< "${TLDW_CLOSEOUT_PRE_REFRESH_STATE}")" = true
  git merge --no-ff --no-edit origin/dev
  TLDW_CLOSEOUT_CANDIDATE="$(git rev-parse HEAD)"
  test "$(git rev-parse HEAD^1)" = "${TLDW_PREVIOUS_CLOSEOUT_CANDIDATE}"
  test "$(git rev-parse HEAD^2)" = "$(git rev-parse origin/dev)"
  git diff --check origin/dev..."${TLDW_CLOSEOUT_CANDIDATE}"
  test "$(git diff --name-only origin/dev..."${TLDW_CLOSEOUT_CANDIDATE}")" = \
    'backlog/tasks/task-12982 - Land-PR-2727-on-current-dev.md'
  git update-ref refs/tldw/task-12982/closeout-candidate \
    "${TLDW_CLOSEOUT_CANDIDATE}" \
    "${TLDW_PREVIOUS_CLOSEOUT_CANDIDATE}"
  git push origin codex/task-12982-closeout
  git fetch --no-tags origin codex/task-12982-closeout
  test "$(git rev-parse origin/codex/task-12982-closeout)" = \
    "${TLDW_CLOSEOUT_CANDIDATE}"
  test "$(gh pr view "${TLDW_CLOSEOUT_PR}" --repo rmusser01/tldw_server \
    --json headRefOid --jq .headRefOid)" = "${TLDW_CLOSEOUT_CANDIDATE}"
  test "$(gh pr view "${TLDW_CLOSEOUT_PR}" --repo rmusser01/tldw_server \
    --json isDraft --jq .isDraft)" = true
  printf 'Closeout head refreshed onto current dev in draft; rerun Steps 8-9 after checks finish.\n' >&2
  exit 1
fi

git diff --check origin/dev..."${TLDW_CLOSEOUT_CANDIDATE}"
test "$(git diff --name-only origin/dev..."${TLDW_CLOSEOUT_CANDIDATE}")" = \
  'backlog/tasks/task-12982 - Land-PR-2727-on-current-dev.md'
TLDW_CLOSEOUT_HEAD="$(gh pr view "${TLDW_CLOSEOUT_PR}" \
  --repo rmusser01/tldw_server --json headRefOid --jq .headRefOid)"
[[ "${TLDW_CLOSEOUT_HEAD}" =~ ^[0-9a-f]{40}$ ]]
test "${TLDW_CLOSEOUT_HEAD}" = "${TLDW_CLOSEOUT_CANDIDATE}"
TLDW_CLOSEOUT_DRAFT="$(gh pr view "${TLDW_CLOSEOUT_PR}" \
  --repo rmusser01/tldw_server --json isDraft --jq .isDraft)"
TLDW_CLOSEOUT_REVIEW="$(gh pr view "${TLDW_CLOSEOUT_PR}" \
  --repo rmusser01/tldw_server --json reviewDecision --jq .reviewDecision)"
test "${TLDW_CLOSEOUT_DRAFT}" = false
test "${TLDW_CLOSEOUT_REVIEW}" != CHANGES_REQUESTED
TLDW_CLOSEOUT_CHECKS="$(gh pr checks "${TLDW_CLOSEOUT_PR}" \
  --repo rmusser01/tldw_server \
  --json bucket,name,state,workflow,link)"
TLDW_CLOSEOUT_BLOCKERS="$(jq -c \
  '[.[] | select(.bucket == "fail" or .bucket == "cancel" or .bucket == "pending")]' \
  <<< "${TLDW_CLOSEOUT_CHECKS}")"
printf '%s\n' "${TLDW_CLOSEOUT_BLOCKERS}"
test "${TLDW_CLOSEOUT_BLOCKERS}" = '[]'
jq -e '
  any(.[];
    .name == "frontend-license-policy/trusted/dev" and
    .bucket == "pass"
  )
' <<< "${TLDW_CLOSEOUT_CHECKS}" >/dev/null

TLDW_CLOSEOUT_THREADS="$(gh api graphql --paginate --slurp \
  -f owner='rmusser01' \
  -f name='tldw_server' \
  -F number="${TLDW_CLOSEOUT_PR}" \
  -f query='query($owner:String!,$name:String!,$number:Int!,$endCursor:String){repository(owner:$owner,name:$name){pullRequest(number:$number){reviewThreads(first:100,after:$endCursor){nodes{isResolved isOutdated}pageInfo{hasNextPage endCursor}}}}}' \
  --jq '[.[].data.repository.pullRequest.reviewThreads.nodes[] | select((.isResolved|not) and (.isOutdated|not))]')"
printf '%s\n' "${TLDW_CLOSEOUT_THREADS}"
test "${TLDW_CLOSEOUT_THREADS}" = '[]'

TLDW_CLOSEOUT_FINAL_STATE="$(gh pr view "${TLDW_CLOSEOUT_PR}" \
  --repo rmusser01/tldw_server \
  --json headRefOid,baseRefName,isDraft,reviewDecision,body)"
test "$(jq -r .headRefOid <<< "${TLDW_CLOSEOUT_FINAL_STATE}")" = \
  "${TLDW_CLOSEOUT_HEAD}"
test "$(jq -r .baseRefName <<< "${TLDW_CLOSEOUT_FINAL_STATE}")" = dev
test "$(jq -r .isDraft <<< "${TLDW_CLOSEOUT_FINAL_STATE}")" = false
test "$(jq -r .reviewDecision <<< "${TLDW_CLOSEOUT_FINAL_STATE}")" != \
  CHANGES_REQUESTED
TLDW_CLOSEOUT_FINAL_BODY="$(jq -r .body <<< "${TLDW_CLOSEOUT_FINAL_STATE}")"
export TLDW_CLOSEOUT_FINAL_BODY
python - <<'PY'
import os
import re

body = os.environ["TLDW_CLOSEOUT_FINAL_BODY"]
heading = "## Change summary (human-authored)"
assert body.count(heading) == 1, "required human-summary heading is missing or duplicated"
match = re.search(
    r"(?ms)^## Change summary \(human-authored\)\s*\n(.*?)(?=^##\s|\Z)",
    body,
)
assert match is not None
section = match.group(1)
visible = re.sub(r"<!--.*?-->", "", section, flags=re.S).strip()
assert visible, "human-summary section has no visible content"
for instruction in (
    "In your own words, explain:",
    "Required before this PR is marked ready or merged",
):
    assert instruction not in section, "instruction template is still present"
print(visible)
PY
test "$(gh pr view "${TLDW_CLOSEOUT_PR}" --repo rmusser01/tldw_server \
  --json headRefOid --jq .headRefOid)" = "${TLDW_CLOSEOUT_HEAD}"
test "$(gh pr view "${TLDW_CLOSEOUT_PR}" --repo rmusser01/tldw_server \
  --json baseRefName --jq .baseRefName)" = dev
gh pr merge "${TLDW_CLOSEOUT_PR}" --repo rmusser01/tldw_server \
  --merge --match-head-commit "${TLDW_CLOSEOUT_HEAD}"
git fetch --no-tags origin dev
git merge-base --is-ancestor "${TLDW_CLOSEOUT_HEAD}" origin/dev
TLDW_CLOSEOUT_WORKTREE='/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/task-12982-closeout'
test -z "$(git -C "${TLDW_CLOSEOUT_WORKTREE}" status --porcelain=v1)"
cd /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/provider-credential-runtime
git worktree remove "${TLDW_CLOSEOUT_WORKTREE}"
```

Expected: the exact pinned one-file candidate is current with `dev`; if `dev` advanced, the PR returns to draft, a reviewed ordinary merge refreshes and re-pins the candidate, and the step stops for fresh checks plus Steps 8-9. Immediately before merge, the human summary, review/thread state, exact head/base, and check rollup are resampled. Once all gates pass, no admin bypass is used and that exact candidate reaches `origin/dev`.

- [ ] **Step 10: Hand off without starting deployment**

Report TASK-12982 complete with `VALIDATED_HEAD`, `VALIDATED_BASE`, `MERGE_SHA`, the PR #2727 URL, and the closeout PR URL. State explicitly that TASK-12983 is now unblocked for its own design/implementation plan and that no service, image, extension, customer data, or protected release was deployed or published by this plan.
