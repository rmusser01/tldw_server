# `task renumber` refuses ids held on origin/dev or in open PRs (TASK-13961) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `task renumber` runs `task create`'s origin/dev and open-PR id check, which `renumber_task(..., check_remote=False)` skips and the duplicate-id generator's `apply` turns off.

**Architecture:** `renumber_task` gains `check_remote: bool = True` and, after its local check and before it reads the file, calls #3263's `MutableRepository._reject_remote_id(self, task_id: str) -> None` (`tools/backlog-py/src/backlog_py/core/repository.py:96-110`). That method skips child ids, returns when no source is reachable, and raises `TaskMutationError` naming the source, so there is no second copy of the check. The CLI (`cli/main.py:127`) passes no flag, so `task renumber` checks. The generator's `apply` passes `check_remote=False`: on a re-run backlog-py's live check lists the renumbering PR's own block, while the generator's allocation already leaves its own PR out (`renumber_duplicate_task_ids.py:316`) and keeps every other PR's ids out of the block (`:466-468`).

**Tech Stack:** Python 3.12, pytest, click `CliRunner`, backlog-py.

**Spec:** `backlog/tasks/task-13961 - Make-task-renumber-refuse-ids-held-on-origin-dev-or-in-open-PRs-once-3263-lands.md` on `origin/dev` (3 ACs). Its draft code is the Appendix of `plan_dupids_v3.md` (plans dir), written before #3263's review fixes. This plan updates it to the final interface: it calls `_reject_remote_id` rather than copying it, `check_remote` defaults to True, and the generator passes False. Brief: `brief_plan_13961.md` (plans dir).

**Measured on:** #3263's head `origin/codex/backlog-py-remote-id-allocation` at `33f24dad94` (it merges dev `cb70f0c5d1`), detached, reverted afterwards. `origin/dev` is now `7a020da14a`, 6 commits past that merge. None of them touches a file this plan changes (`git diff --stat cb70f0c5d1 origin/dev -- <files>`: empty). `proto_13961.patch` applies to `git merge-tree --write-tree origin/dev 33f24dad94`, checked with `git apply --cached --check` against a throwaway index. `file:line` citations are for `33f24dad94`. The independent review re-ran every step on `17e9098a63` (#3263 rebased; same tree). #3263 merged on 2026-11-10 as `4f6ec4c2d7`; Task 0 Step 3 re-measures the baseline there, and a different count is recorded, not chased.

**Prototype:** `proto_13961.patch` (plans dir), 6 files, +69/-7. It holds every code, test and doc change, but not the plan copy or the task file.

---

## Decisions

1. **ADR-004 Change summary: answered.** The owner said on 2026-11-07, "Yes, waive for all", and that includes the TASK-13961 PR. Record it in the PR body as a full `## Change summary` section (Task 3 Step 5), not as a one-line note.
2. **No CLI flag to skip the check (ruling, not an owner question).** `task create` has none either. The only caller that needs the skip, the generator, uses the Python API. A person whose own open PR still shows the id they want can pick another id, or push first (Review Focus 1). Add `--no-remote-check` when a second caller needs it.
3. **The current branch's own PR stays a remote source (ruling).** Leaving it out would change #3263's `collect_remote_ids` for `create` too, and is outside this task. The refusal names the source, and the message says to pick a higher id.

## Global Constraints

Verbatim from the brief:

- Tests make no real outbound network requests and never run the real `git` against a remote or the real `gh`: they stub `collect_remote_ids` / the subprocess layer (#3263's conftest does it for the backlog-py suite; the generator's tests live in `tldw_Server_API/tests/Helper_Scripts/` and need their own stub — check).
- Backlog: edit tasks with backlog-py only (`PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit ...`); `--append-notes`, never `--notes`; stage with `git add -A backlog/tasks`; never hand-edit task files.
- Branch from `origin/dev`; the PR targets `dev`. Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Never `--no-verify`, never force-push, never `git stash`.
- Mirrored docs (`Docs/Published`) need `bash Helper_Scripts/refresh_docs_published.sh` in the same PR (e.g. if ADR-059 or the backlog-py README changes).
- CI checks before the PR, in the last task: backlog-py's tests; `tldw_Server_API/tests/Helper_Scripts`; `tests/Docs`, `tests/lint`, `tests/CI`; `test_quality_triage.py --enforce`; shard coverage; `check_backlog_task_ids.py` exit 0; Bandit on touched source; `git diff --check`.
- Agent shells don't keep the venv `PATH`: prefix commands that spawn `python` with `PATH="$(dirname $PY):$PATH"`.

Plan-specific (each checked on the prototype):

- **The stub, checked.** The generator's tests have no conftest. Only the new `apply` test reaches backlog-py's remote layer, and it sets `repository_module.collect_remote_ids` itself. The other 7 either stub the generator's `run`/`git` or call pure functions. With `check_remote=False`, `apply` never calls `collect_remote_ids`, and `edit_task` never does. So no autouse fixture is added. The offline probe (Task 3 Step 1) makes `remote_ids._run`, the generator's `run` and its `git` raise: the two suites pass under it.
- **Docs.** `AGENTS.md:434` (the Renumber bullet) and `tools/backlog-py/README.md:51` change. Neither is mirrored: `refresh_docs_published.sh` copies `Docs/{Site,Wiki,API-related,ADR,Code_Documentation,Deployment,Evaluations,Getting_Started,User_Guides}`, `Architecture.md`, `Env_Vars.md` and `Feature_Status.md`. ADR-084:31 ("`task renumber PATH --id NEW_ID` gives one file a new id") and ADR-059 say nothing about what renumber refuses. So no refresh is needed. `CLAUDE.md` has no Renumber bullet.
- **Baselines.** No touched file is in `Helper_Scripts/ci/scope_predicate_baseline.txt`, and none is in `tldw_Server_API/tests/lint/function_size_baseline.json` (grep: nothing). backend-required runs `tools/backlog-py/tests` and `test_renumber_duplicate_task_ids.py` in one pytest call (`.github/workflows/backend-required.yml:327-328`). They pass together: 189.
- **`$PY`** is `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`. `<plans dir>` is `/Users/macbook-dev/.claude/projects/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/plans`. `<pytest>` is `PATH="$(dirname $PY):$PATH" PYTHONPATH=tools/backlog-py/src $PY -m pytest -q -p no:cacheprovider`.
- **Git.** Stage named files. `git diff --cached --check` prints nothing before each commit.

## Review Focus

1. **Your own open PR still shows the id.** For example, you renumber a file away from TASK-N and, before pushing, renumber another file to TASK-N. The pushed PR still lists TASK-N, so the renumber is refused, naming "open pull requests", as `task create` is. Expected: refused (Decision 3). Pick another id, or push first. Pinned by the refusal test: an id an open PR shows is refused.
2. **Offline, no `gh`, or not authenticated.** Each source that cannot be read warns on stderr, and the renumber goes ahead on the local check, as `create` does (`_reject_remote_id` returns when no source is reachable). Pinned by the 11 existing renumber tests, which run under #3263's autouse empty `RemoteIdState()` (`tests/conftest.py`) and pass unchanged.
3. **A subtask id.** `TASK-1.1` is taken while a source holds `1`, because remote sources list top-level ids only. The ceiling, from #3263: a PR holding `TASK-N.m` counts as `N` (`remote_ids.py:20`), so a child-id collision with a PR is not caught, for `create` or `renumber`. Pinned by the refusal test's last assertion.
4. **A lower-case id** (`--id task-15000`). It is normalized before the check, so the message names `TASK-15000`, exactly as `create`'s does. Pinned: the CLI call in the refusal test uses `task-15000`.
5. **A refused renumber changes nothing.** The check runs before the file is read, so nothing is renamed, written or noted. Pinned: the refusal test asserts the original file still exists.

Known cost, not a defect: every `task renumber` now runs a fetch of origin/dev, a 5 s `ls-tree`, `gh pr list`, and one `gh api` call per open PR over 100 files, each network call up to 30 s (`remote_ids.py:23`). The generator skips all of them.

---

## Task 0: Branch after #3263, start the task, draft PR, baseline

**Files:** `backlog/tasks/task-13961 - ...md` (status, notes); `Docs/superpowers/plans/2026-11-07-renumber-remote-id-check.md` (this plan).

- [ ] **Step 1: Wait for #3263.** This PR branches from `dev` after #3263 merges, not from #3263's branch.
  - `gh pr view 3263 --json state --jq .state` must print `MERGED`.
  - `git fetch origin dev --quiet`.
  - `git grep -n "def _reject_remote_id(self, task_id: str) -> None" origin/dev -- tools/backlog-py/src/backlog_py/core/repository.py` must print one line. If the name or signature differs, stop: the interface moved after this plan.
  - `git switch -c fix/backlog-py-renumber-remote-check-13961 origin/dev`.
  - `git apply --check <plans dir>/proto_13961.patch` must print nothing (a sanity check; apply nothing yet).
- [ ] **Step 2: Start TASK-13961.**

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13961 -s "In Progress" \
  --append-notes "Plan: Docs/superpowers/plans/2026-11-07-renumber-remote-id-check.md. #3263 (TASK-13854) is on dev; renumber_task calls its _reject_remote_id (no second copy). ADR-004 Change summary: waived by the owner on 2026-11-07 ('Yes, waive for all')."
```

- [ ] **Step 3: Baseline.** Run `<pytest> tools/backlog-py/tests`, then `<pytest> tldw_Server_API/tests/Helper_Scripts/test_renumber_duplicate_task_ids.py`, then `<pytest> tldw_Server_API/tests/Helper_Scripts tools/backlog-py/tests`.
  Expected: `179 passed`, `7 passed` and `461 passed, 1 skipped` (all measured on `33f24dad94`).
- [ ] **Step 4: Plan copy, commit 0, draft PR.**
  - Copy this plan, unchanged, to `Docs/superpowers/plans/2026-11-07-renumber-remote-id-check.md`.
  - `git add Docs/superpowers/plans/2026-11-07-renumber-remote-id-check.md && git add -A backlog/tasks && git diff --cached --check`. It must print nothing.
  - `PATH="$(dirname $PY):$PATH" TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q -n 8 tldw_Server_API/tests/Docs`. Expected: `212 passed` (measured with the copy in place).
  - Commit with message 0, then `git push -u origin fix/backlog-py-renumber-remote-check-13961`.
  - `gh pr create --draft --base dev --title "fix(backlog-py): task renumber refuses ids held on origin/dev or in open PRs (TASK-13961)"`. Give it a one-line body naming TASK-13961 and #3263, ending with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

## Task 1: `renumber_task` runs `_reject_remote_id` unless `check_remote=False` (AC #1, #2, #3)

**Files:**
- Modify: `tools/backlog-py/src/backlog_py/core/repository.py:277-298` (`renumber_task`)
- Modify: `AGENTS.md:434`, `tools/backlog-py/README.md:51`
- Test: `tools/backlog-py/tests/test_remote_id_allocation.py` (append; it already has `_copy_fixture`, `_state` and `repository_module`)

**Interfaces:**
- Consumes: `MutableRepository._reject_remote_id(self, task_id: str) -> None` (#3263).
- Produces: `MutableRepository.renumber_task(self, path: Path, new_id: str, *, note: str | Sequence[str] | None = None, check_remote: bool = True) -> TaskRecord`.

- [ ] **Step 1: Write the failing tests.** Add `from backlog_py.cli.main import main` and `from click.testing import CliRunner` to the imports, in isort order. Append:

```python
TASK_1 = Path("backlog/tasks/task-1 - Example-task.md")


def test_renumber_refuses_a_top_level_id_held_remotely_with_the_create_message(tmp_path, monkeypatch):
    repo = _copy_fixture(tmp_path)
    monkeypatch.setattr(
        repository_module, "collect_remote_ids", lambda _root: _state({"open pull requests": {1, 15000}})
    )
    with pytest.raises(TaskMutationError) as created:
        MutableRepository.from_path(repo).create_task(title="New", task_id="TASK-15000")

    args = ["--cwd", str(repo), "task", "renumber", str(TASK_1), "--id", "task-15000"]
    result = CliRunner().invoke(main, args)

    assert str(result.exception) == str(created.value)
    assert "open pull requests: TASK-15000" in str(created.value)
    assert (repo / TASK_1).is_file()
    # Remote sources list top-level ids only, so a subtask id is checked locally, as in create.
    assert MutableRepository.from_path(repo).renumber_task(TASK_1, "TASK-1.1").id == "TASK-1.1"


def test_renumber_skips_the_remote_check_when_told_to(tmp_path, monkeypatch):
    repo = _copy_fixture(tmp_path)
    monkeypatch.setattr(
        repository_module, "collect_remote_ids", lambda _root: _state({"open pull requests": {15000}})
    )
    # The generator's apply passes check_remote=False: its own open PR lists the block it writes.
    moved = MutableRepository.from_path(repo).renumber_task(TASK_1, "TASK-15000", check_remote=False)
    assert moved.id == "TASK-15000"
```

- [ ] **Step 2: Run them.** `<pytest> tools/backlog-py/tests/test_remote_id_allocation.py -k renumber`
  Expected: `2 failed` (measured). The refusal test fails with `AssertionError: assert 'None' == 'Task id alre...and open PRs)'`: the renumber went through. The skip test fails with `TypeError: MutableRepository.renumber_task() got an unexpected keyword argument 'check_remote'`.
- [ ] **Step 3: Implement.**
  - In `renumber_task`, use the signature in Interfaces, one parameter per line (on one line it is over 120 characters).
  - Directly after `raise TaskMutationError(f"Task id already exists: {normalized_id}")`, add `if check_remote: self._reject_remote_id(normalized_id)` on two lines. It must come before `current_path.open(...)`.
  - Docstring: the sentence that starts "Refuses an id" becomes this, rewrapped to the existing width: `Refuses an id that any task file in backlog/tasks, backlog/completed or backlog/archive/tasks holds, and, like create_task, a top-level id that origin/dev or an open PR holds, unless check_remote is False.` Keep the names in double backticks, as the docstring already does.
  - `AGENTS.md:434` (the Renumber bullet): insert the text below directly before the period that ends "or `backlog/archive/tasks` holds".

```text
, and, as `task create` does, a top-level id that `origin/dev` or an open PR holds (your own pushed PR counts too)
```

  - `tools/backlog-py/README.md:51`: the line `` Child ids (`P.n`) skip the remote check.`` becomes these two lines, so the child-id sentence covers both commands (review Minor 1):

```text
`task renumber` runs the same check on its new id. Child ids (`P.n`) skip the
remote check.
```
- [ ] **Step 4: Run them.** The same command. Expected: `2 passed`. Then `<pytest> tools/backlog-py/tests`. Expected: `181 passed` (measured).
- [ ] **Step 5: Mutation probes** (revert each).
  - With `check_remote: bool = False`, the Step 2 command gives `1 failed, 1 passed`: the refusal test fails with `assert 'None' == ...`.
  - With `if True:` for `if check_remote:`, the skip test fails with `TaskMutationError: Task id already exists on open pull requests: TASK-15000`.
  - Both measured.
- [ ] **Step 6: Commit** (message 1): `git add tools/backlog-py/src/backlog_py/core/repository.py tools/backlog-py/tests/test_remote_id_allocation.py AGENTS.md tools/backlog-py/README.md && git diff --cached --check`.

## Task 2: The generator's `apply` passes `check_remote=False` (AC #2, #3)

**Files:**
- Modify: `Helper_Scripts/renumber_duplicate_task_ids.py:597-598` (`apply`)
- Test: `tldw_Server_API/tests/Helper_Scripts/test_renumber_duplicate_task_ids.py` (append)

**Interfaces:**
- Consumes: `renumber_task(..., check_remote: bool = True)` (Task 1); the module's `_copy(directory, name, origin_index, **evidence) -> Task` test helper.
- Produces: nothing new. `apply(p, root, task_ref, today)` keeps its signature.

- [ ] **Step 1: Write the failing test.** Add `import shutil`, `from backlog_py.core import repository as repository_module` and `from backlog_py.core.remote_ids import RemoteIdState` to the imports, in isort order. After `pytestmark`, add a blank line and then:
  `BACKLOG_FIXTURE = Path(__file__).resolve().parents[3] / "tools/backlog-py/tests/fixtures/repos/basic"`. Then append:

```python
def test_apply_takes_the_ids_its_own_open_pr_shows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A re-run while this PR is open: backlog-py's live check lists the PR's own block (TASK-13961).
    shutil.copytree(BACKLOG_FIXTURE, tmp_path, dirs_exist_ok=True)
    monkeypatch.setattr(
        repository_module, "collect_remote_ids", lambda _root: RemoteIdState({"open pull requests": {15000}})
    )
    row = {"copy": _copy("tasks", "task-1 - Example-task.md", 0), "old": "TASK-1", "new": "TASK-15000",
           "keeper": _copy("completed", "task-1 - Kept.md", 0)}

    renumber.apply({"rows": [row], "tasks": [], "edits": {}, "rewrites": {}}, tmp_path, "TASK-13960", "2026-11-07")

    assert [path.name for path in (tmp_path / "backlog/tasks").iterdir()] == ["task-15000 - Example-task.md"]
```

  The `monkeypatch` is this file's stub: no other test here reaches backlog-py's remote layer (Global Constraints, "The stub, checked").
- [ ] **Step 2: Run it.** `<pytest> tldw_Server_API/tests/Helper_Scripts/test_renumber_duplicate_task_ids.py::test_apply_takes_the_ids_its_own_open_pr_shows`
  Expected: `1 failed` (measured), with `TaskMutationError: Task id already exists on open pull requests: TASK-15000 (concurrent-session collision guard; ...)` raised through `apply` → `renumber_task` → `_reject_remote_id`. Before Task 1 it passes: renumber had no remote check.
- [ ] **Step 3: Implement.** In `apply`, the `renumber_task` call gains `check_remote=False`. Its loop line gets the comment `# check_remote=False: this PR's own block shows as an open PR's ids (TASK-13961)`, so no line number shifts.
- [ ] **Step 4: Run it.** The same file without the node id. Expected: `8 passed` (measured).
- [ ] **Step 5: Commit** (message 2): `git add Helper_Scripts/renumber_duplicate_task_ids.py tldw_Server_API/tests/Helper_Scripts/test_renumber_duplicate_task_ids.py && git diff --cached --check`.

## Task 3: Verify, close, review, PR

- [ ] **Step 1: Suites and the offline probe.**
  - `<pytest> tldw_Server_API/tests/Helper_Scripts tools/backlog-py/tests`. Expected: `464 passed, 1 skipped` (measured): the base's 461 plus this plan's 3 tests.
  - Offline probe: save as `<scratchpad>/offline_13961.py`, and run `PATH="$(dirname $PY):$PATH" PYTHONPATH=tools/backlog-py/src:. $PY <scratchpad>/offline_13961.py tools/backlog-py/tests tldw_Server_API/tests/Helper_Scripts/test_renumber_duplicate_task_ids.py`. Expected: `189 passed` (measured).

```python
import sys

import pytest
from backlog_py.core import remote_ids

from Helper_Scripts import renumber_duplicate_task_ids as gen


def _boom(*args, **kwargs):
    raise AssertionError(f"real git/gh reached: {args[:1]}")


remote_ids._run = _boom
gen.run = _boom
gen.git = _boom
sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", *sys.argv[1:]]))
```

- [ ] **Step 2: CI checks.** Each result below was measured on the prototype.
  - `PATH="$(dirname $PY):$PATH" TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q -n 8 tldw_Server_API/tests/Docs tldw_Server_API/tests/lint tldw_Server_API/tests/CI`: `1497 passed, 6 skipped`, about 3 minutes. That was measured without the plan copy. The copy was then checked on its own (Task 0 Step 4).
  - `PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/check_shard_coverage.py --ci-file .github/workflows/ci.yml`: `new_uncovered=0`, "OK — no newly-unshared test files."
  - `PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/test_quality_triage.py --enforce`: exit 0, "OK — no offenses beyond the baseline."
  - `/usr/bin/python3 Helper_Scripts/ci/check_backlog_task_ids.py .`: exit 0.
  - `/opt/homebrew/bin/uvx bandit -ll tools/backlog-py/src/backlog_py/core/repository.py Helper_Scripts/renumber_duplicate_task_ids.py`: "No issues identified."
  - `$PY -m ruff check` on the four touched `.py` files: "All checks passed!"
  - `git diff origin/dev --check`: nothing.
- [ ] **Step 3: Close TASK-13961** (fill `<n>` from Step 2 first).

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13961 \
  --append-notes "RED/GREEN: Task 1 2 failed (the refusal: assert 'None' == 'Task id already exists on open pull requests: TASK-15000 ...'; the skip: TypeError, unexpected keyword argument 'check_remote') -> 2 passed, backlog-py 179 -> 181 passed; mutations (default False; the guard dropped) fail one test each. Task 2 1 failed (TaskMutationError from apply) -> 8 passed. Offline probe (remote_ids._run and the generator's run/git raise): 189 passed. tests/Docs + lint + CI <n>; shard coverage, test-quality ratchet, check_backlog_task_ids unchanged; Bandit: no issues; ruff clean." \
  --append-notes "Docs: AGENTS.md's Renumber bullet and the backlog-py README's Task ids section say renumber refuses a remote-held id; neither is mirrored to Docs/Published. No CLI flag for the skip (Decision 2)." \
  --check-ac 1 --check-ac 2 --check-ac 3 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "task renumber refuses a top-level id that origin/dev or an open PR holds, with task create's message: renumber_task calls #3263's _reject_remote_id after its local check, before it reads the file. Child ids skip it, as in create, and an unreachable source falls back to the local check. renumber_task(..., check_remote=False) skips it, and Helper_Scripts/renumber_duplicate_task_ids.py's apply passes that, since a re-run's live check would list the renumbering PR's own block. Tests cover the refusal (through the CLI, compared with create's message), the skip, and apply taking an id its own open PR shows. AGENTS.md and the backlog-py README updated." \
  -s Done
```

  Then `git grep -n "<n>" -- 'backlog/tasks/task-13961 *'` must print nothing. Then `git add -A backlog/tasks && git diff --cached --check`, and commit with message 3.
- [ ] **Step 4: Review before the PR.** Run `/superpowers:requesting-code-review` in a clean subagent on `origin/dev..HEAD`, and give it this plan's Review Focus. Address every finding before the PR leaves draft, and record each ruling in the task notes (`--append-notes`).
- [ ] **Step 5: Finish the PR and merge.**
  - `git push`. Replace the body with `gh pr edit <n> --body-file <file>`, then `gh pr ready <n>`.
  - The body has: the three ACs and how each is met; Decisions 2-3; the Review Focus items; the measured counts. It also has this section:

```markdown
## Change summary

`task renumber` now refuses a new top-level id that `origin/dev` or an open PR already holds, with the same message as `task create`, by calling #3263's `_reject_remote_id`. `renumber_task(..., check_remote=False)` skips that check, and `Helper_Scripts/renumber_duplicate_task_ids.py`'s `apply` passes it, because on a re-run the live check would list the renumbering PR's own block. AGENTS.md and the backlog-py README say so.

ADR-004: the owner waived a user-written Change summary for this PR on 2026-11-07 ("Yes, waive for all").
```

  - The body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
  - Merging: run `gh variable get MERGE_QUEUE` (unset when this plan was written; dev requires 8 statuses). When reviews are settled, run `QUEUE="<n>:fix/backlog-py-renumber-remote-check-13961" sh <plans dir>/pr_queue3.sh` in the background. Each body edit re-triggers the license status; wait for it before merging. Confirm `MERGED` before deleting the branch.
  - The last `git status --short` is empty.

## Commit messages

Each ends with a blank line and `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

- 0: `chore(backlog): start TASK-13961 and add its plan`
- 1: `fix(backlog-py): task renumber refuses an id held on origin/dev or in an open PR (TASK-13961)`
- 2: `fix(backlog): the duplicate-id generator's apply skips the remote id check (TASK-13961)`. The body says why: its own open PR lists the block it writes.
- 3: `chore(backlog): close TASK-13961`

## Conflicts with open PRs

Only #3263, the dependency. 41 open PRs target `dev` (2026-11-07). Checked with `gh api repos/rmusser01/tldw_server/pulls/<n>/files --paginate` per PR. #3263 touches `repository.py`, `test_remote_id_allocation.py`, `tests/conftest.py`, the backlog-py README and `AGENTS.md`. No other PR touches any file this plan changes, the generator, its test, or TASK-13961's file.

## Not measured

- **The patch on dev after #3263 merges.** It was checked against `merge-tree(origin/dev 7a020da14a, 33f24dad94)`. If #3263's head moves before it merges, Task 0 Step 1's grep and `git apply --check` catch the drift.
- **A live `task renumber` against GitHub.** It was not run, by design: every check is stubbed.
