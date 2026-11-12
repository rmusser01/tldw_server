# Re-parent the orphaned subtasks and finish the renumbering's leftovers (TASK-13962) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One PR that closes TASK-13962:
- AC 1: the 43 subtasks that 3a660d4f1d (2026-07-05) left under a reused parent id get ids under their real parents, with a matching `parent_task_id`;
- AC 2: the two dependency retargets that open PRs hold are applied where a PR has closed, and otherwise handed to the owner and to a small follow-up task;
- AC 3: TASK-15183 gets its title, status and labels, and its leftover block is removed (TASK-13534, the one other file with the same defect, gets the same repair);
- AC 4: every bare-id `references:` entry that names a renumbered id is checked, and the 45 that mean a renumbered copy, in files `task edit` can reach and no open PR changes, name its new id.

**Architecture:**
- **backlog-py, three small changes,** RED-first, in the functions every caller routes through:
  - `renumber_task`: a subtask id `P.n` also sets `parent_task_id: P`. Today it accepts `TASK-12295.1` but leaves `parent_task_id: TASK-407` (measured). It never refused a subtask id under another parent, so no refusal needs lifting.
  - `task edit --ref X` (repeatable): replaces `references:`, one entry per flag. backlog-py had no way to change references.
  - `task edit --remove-preamble`: drops the body text above the first `## ` heading, which no section owns, and refuses a task without a heading. backlog-py had no way to remove body text outside a section.
- **One-off drivers in the plans dir, not the repo.** They write task files only through backlog-py's CLI:
  - `reparent_13962.py` (AC 1);
  - `scan_refs_13962.py` (AC 4: classify, write the Docs table, apply).
  - Two read-only checks: `parent_check_13962.py`, `fm_diff_13962.py`.
- **Docs:**
  - a follow-up section in the record `Docs/Development/Backlog_Task_Id_Renumbering.md`;
  - a generated table `Docs/Development/Backlog_Task_Id_References.md`, one row per AC 4 entry;
  - one clause each in `AGENTS.md` and `tools/backlog-py/README.md`.
  - ADR-084 does not change: its decision (unique ids, checked on every PR) is untouched. Neither the record nor the table is mirrored.

**Tech Stack:** Python 3.12, click, PyYAML, pytest; git; `gh` (read-only queries).

**Spec:**
- TASK-13962's four ACs and its Implementation Notes;
- the planner brief `brief_plan_13962.md` (plans dir);
- the record `Docs/Development/Backlog_Task_Id_Renumbering.md` ("Subtasks attributed by topic", "Dependency retargets left to open PRs") and ADR-084.

**Measured on:** `origin/dev` at `a960463a26` (2026-11-03, the merge of #3408). The open-PR data was read on 2026-11-03 between 01:00 and 01:20 PST. Everything was run in a detached worktree, which was reverted afterwards.

**Prototype:** `proto_13962.patch` in the plans dir: 100 files, +638/-297. It holds every change in this plan except the plan copy and the follow-up task file. It is the result of running Tasks 1-4's steps on `a960463a26` with the date 2026-11-03 and `TASK-15419` left as a placeholder. A second run of those steps from a clean `a960463a26` (the plans dir's drivers, then the record and `AGENTS.md` text) reproduced it byte for byte (`cmp`). Apply it with `git apply --index` to compare; never commit it in place of running the steps.

**Placeholders.** `<plans dir>` is `/Users/macbook-dev/.claude/projects/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/plans`. `<scratchpad>` is the executing session's scratchpad. `$PY` is `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`. `TASK-15419` is the follow-up task's id (Task 0). `<date>` is the execution date (`date +%F`). Write each literally.

---

## What was verified (on `a960463a26`)

| # | Item | Evidence | Verdict |
|---|---|---|---|
| V1 | The 43 subtasks and their new parents | All 43 files exist, each with `parent_task_id` = the old parent: 407.1-.6, 410.1, 463.1-.5, 481.1-.19, 528.1-.9 in `backlog/tasks`, and 418.15, 418.16, 418.17 in `backlog/completed`. 418.17 is the completed 418.14, renumbered on 2026-10-31. `git show --name-status -M 3a660d4f1d` renames (R099) `task-407 - Implement-main-chat-role-play-preset-remediation.md` → `task-12295`, `task-410` → `task-12303`, `task-418 - Plan-llama.cpp-...` → `task-12318`, `task-463` → `task-12418`, `task-481` → `task-12446`, `task-528` → `task-12561`: each the record's "parent's new id". | **re-parent all 43** |
| V2 | Child numbers | No task file in the three directories, and no file in any open PR, has an id under TASK-12295, 12303, 12318, 12418, 12446 or 12561. So each set takes `.1`-`.n` in its old order. For 418 the old order is 418.14 (now .17; added 8e23caecaa, 2026-05-18 17:03), then 418.15 and 418.16 (8cad8f4af3, 18:38). | `.1`-`.n` |
| V3 | Does `task renumber` do it? | It accepts `--id TASK-12295.1` (`test_renumber_takes_a_subtask_id`) and refuses only an id a file holds. But it rewrites only `id`, so the file keeps `parent_task_id: TASK-407`. Measured: the new test fails with `assert 'TASK-2' == 'TASK-8'`. `task edit` cannot set `parent_task_id`. | **backlog-py change** |
| V4 | What else names the old subtask ids | `git grep -P` over the whole tree (108 lines):<br>- **Frontmatter:** no `dependencies:` or `references:` entry anywhere names one.<br>- **Their own files:** the `id:` lines, notes such as "TASK-528.6 complete", and self-links (418.15, 418.16).<br>- **Parents' notes:** TASK-12295, 12318, 12446 and 12561 (×3).<br>- **Docs:** `Docs/Plans/2026-06-07-knowledge-qa-follow-on-gap-matrix.md` and nine `Docs/superpowers/plans/2026-06-07-knowledge-*` plans (TASK-528.x). One file-name link: a `git add` line of `Docs/superpowers/plans/2026-05-17-llamacpp-managed-runtime-implementation-plan.md:543`, whose neighbour line already names task-418's pre-July path.<br>- **False hits:** the hits in `task-418.14` (the WebUI keeper) are its own id. | **leave the text** (as TASK-13960 did); append a note to each moved file and each new parent |
| V5 | AC 2's PRs | #2855 (task-13145: `TASK-13111` → `TASK-15132`) and #2421 (task-2281: `TASK-2387` → `TASK-15254`) are OPEN. Neither PR's diff changes that entry yet. | **left to the PRs** unless one has closed by execution |
| V6 | TASK-15183 | It has no opening `---`, so its frontmatter is `{id: TASK-15183}`, and the old block (`id: TASK-13511`, title, `status: Done`, three labels, a stray `---`) sits in the body above `## Description`. backlog-py reads it as untitled, "To Do". Of the 4,437 task files, one other has no opening `---`: `task-13534` (the #3210 fast-follow, the same author session), whose whole frontmatter block is read as body. backlog-py has no way to remove body text outside a section. | **backlog-py flag**; repair both |
| V7 | AC 4's entries | 157 bare-id `references:` entries (TASK-13962's notes say 155) in 126 files name one of the 235 renumbered ids: 138 in `backlog/tasks`, 18 in `backlog/completed`, 1 in `backlog/archive/tasks`. Classified with `scan_refs_13962.py` (rules below): keeper 73, neither 32, renumbered copy 50, unclear 2. | **change 45**, leave 112 |
| V8 | Other parent mismatches | `parent_check_13962.py` lists files whose `parent_task_id` disagrees with their id's prefix or names no task file. It finds 12 on `a960463a26`, the same 12 after the prototype. (The 43 do not show there: their id and `parent_task_id` agree, and both name the reused id.) The 12 are outside TASK-13962's list: TASK-12020.46-.52 (they name a sibling), TASK-12116.1 (→ TASK-13140), TASK-12178 and TASK-12179 (→ TASK-45.44.x), TASK-12570 (→ TASK-530), and the archived TASK-18.82.1 (→ a missing TASK-18.82). | **out of scope**; Decision 4 |

**AC 4's rules** (`scan_refs_13962.py`). The first rule that applies wins:
- **self:** the citing file is one of the copies. 0 rows.
- **parent:** the citing file is a subtask of that id. Subtasks stayed with the keeper, by the renumbering's rule (b). 34 rows.
- **decided:** a reviewer's call from the citing file's own text, listed in `DECISIONS`. It covers rows the next rule leaves open, and two where the only file on the writing commit contradicts the text (TASK-13001.2, TASK-12945). 42 rows.
- **written:** take the commit that first wrote the entry (`git log` over the citing file's renames). The files that held the id on that commit are what its author could mean. The rule picks the only one, or the one whose title shares the most words with the citing title (at least 2, and twice the runner-up). That file is now the keeper, a renumbered copy, or **neither**: a file 3a660d4f1d renumbered in July, which the AC does not cover. 79 rows.
- **unclear:** the rest. They stay. 2 rows, both TASK-504: TASK-12520 and TASK-12524 cite it, and none of the four files holding it then is about onboarding.

**The 50 renumbered-copy rows and the 2 unclear ones** (all 157 are in the generated `Docs/Development/Backlog_Task_Id_References.md`):

| Citing task | Old id | Verdict | Becomes | Action | Evidence (rule) |
|---|---|---|---|---|---|
| TASK-12301 | TASK-407 | renumbered copy | TASK-15005 | changed | of 4 files holding it, the one whose title shares playground, remediation, workspace (78f3f460cc) is now TASK-15005 (written) |
| TASK-12520 | TASK-504 | unclear | TASK-504 | left as is | 4 files held TASK-504 when the entry was written (efa22622a3): Impeccable design context, StrictBool, API boundary Stage 3, worker lifecycle (unclear) |
| TASK-12524 | TASK-504 | unclear | TASK-504 | left as is | same 4 files (efa22622a3) (unclear) |
| TASK-12905 | TASK-12164 | renumbered copy | TASK-15084 | changed | character expressions: the character emotes follow-up, not the Backlog id cleanup (decided) |
| TASK-12906 | TASK-12164 | renumbered copy | TASK-15084 | changed | character expressions: the character emotes follow-up, not the Backlog id cleanup (decided) |
| TASK-12948 | TASK-12106 | renumbered copy | TASK-15050 | changed | PR 2713 auth persistence: single-user API key device persistence (decided) |
| TASK-12948 | TASK-12108 | renumbered copy | TASK-15052 | changed | PR 2713 auth persistence: persistent HttpOnly sessions (decided) |
| TASK-12953 | TASK-12106 | renumbered copy | TASK-15050 | changed | of 2 files holding it, the one whose title shares api, key, single, user (b500259583) is now TASK-15050 (written) |
| TASK-12968.1 | TASK-12971 | renumbered copy | TASK-15121 | changed | its notes: "TASK-12971's connected-peer-verified one-hop primitive" (decided) |
| TASK-12968.2 | TASK-12971 | renumbered copy | TASK-15121 | changed | its dependency was retargeted to TASK-15121 by the renumbering (decided) |
| TASK-12970 | TASK-12969 | renumbered copy | TASK-15261 | changed | of 4 files holding it, the one whose title shares governed, preflight, scraping, web (92612d49bf) is now TASK-15261 (written) |
| TASK-12970.1-.14 (14 rows) | TASK-12969 | renumbered copy | TASK-15261 | changed | a Phase 3 task "from the approved plan": Plan Web_Scraping Phase 3 governed preflight (decided) |
| TASK-12989 | TASK-12988 | renumbered copy | TASK-15122 | changed | title shares article, extraction, orchestration, scraping, web (f1b46e5902) (written) |
| TASK-12993 | TASK-12989 | renumbered copy | TASK-15125 | changed | title shares analytics, claims, exports, jobs (5b12fe5baa) (written) |
| TASK-12993 | TASK-12990 | renumbered copy | TASK-15126 | changed | title shares analytics, claims, exports, jobs (5b12fe5baa) (written) |
| TASK-13001 | TASK-13000 | renumbered copy | TASK-15127 | changed | the only file holding it (ee974a3316) (written) |
| TASK-13001.2 | TASK-13000 | renumbered copy | TASK-15127 | changed | Work Package 2 of the UserProfiles Stage 2 plan, whose design TASK-13001 and TASK-13001.1 cite as TASK-15127 (decided) |
| TASK-13138 | TASK-13134 | renumbered copy | TASK-15135 | changed | title shares graph, notes (7fcdc6c45c) (written) |
| TASK-13138 | TASK-13135 | renumbered copy | TASK-15136 | changed | its notes: "TASK-13135 automatic background organization" (decided) |
| TASK-13139.2 | TASK-12971 | renumbered copy | TASK-15121 | changed | unpinned browser DNS and unverified peers: the HTTP hop primitive (decided) |
| TASK-13210 | TASK-13208 | renumbered copy | TASK-15147 | changed | Persona speech decoding, not Study Assistant prompts (decided) |
| TASK-13421 | TASK-13215 | renumbered copy | TASK-15149 | left: open PR #3323 changes this file | title shares atomicity, completion, jobs (34f5cd8e08) (written) |
| TASK-15031 | TASK-9936 | renumbered copy | TASK-15030 | changed | title shares claims, jobs (97cdfc10a5) (written) |
| TASK-15033 | TASK-9937 | renumbered copy | TASK-15031 | changed | title shares claims, jobs (6a91f3220c) (written) |
| TASK-15050 | TASK-12108 | renumbered copy | TASK-15052 | changed | title shares single, user (07ea4c43a3) (written) |
| TASK-15052 | TASK-12106 | renumbered copy | TASK-15050 | changed | title shares single, user (07ea4c43a3) (written) |
| TASK-15070 | TASK-12146 | renumbered copy | TASK-15069 | changed | slide claim verification: Plan Research Workspace artifact claim verification (decided) |
| TASK-15084 | TASK-12163 | renumbered copy | TASK-15083 | changed | "Follow-up after v1 character emote directives" (decided) |
| TASK-15099 | TASK-12173 | renumbered copy | TASK-15096 | changed | title shares media, research, workspace, wp2 (ee2e1e2f7c) (written) |
| TASK-15125 | TASK-9937 | renumbered copy | TASK-15031 | changed | title shares claims, jobs (f9f77e65ec) (written) |
| TASK-15126 | TASK-12989 | renumbered copy | TASK-15125 | changed | title shares analytics, claims, exports, jobs (26fd141999) (written) |
| TASK-15149 | TASK-13216 | renumbered copy | TASK-15150 | changed | Jobs completion: the UUID-precondition caller migration (decided) |
| TASK-15149 | TASK-13217 | renumbered copy | TASK-15151 | changed | title shares completion, jobs (524cb55dcf) (written) |
| TASK-15150 | TASK-13215 | renumbered copy | TASK-15149 | changed | its description: "After TASK-13215 establishes the optional complete_job expected_uuid contract" (decided) |
| TASK-15151 | TASK-13215 | renumbered copy | TASK-15149 | changed | title shares completion, jobs (524cb55dcf) (written) |
| TASK-2281 | TASK-2387 | renumbered copy | TASK-15254 | left: open PR #2421 changes this file | its description: "the MCP smoke client harness from TASK-2387" (decided) |
| TASK-13001.1 | TASK-13000 | renumbered copy | TASK-15127 | left: task edit reaches only backlog/tasks | the only file holding it (8d9495289c) (written) |
| TASK-15226 | TASK-407 | renumbered copy | TASK-15005 | left: task edit reaches only backlog/tasks | title shares playground, remediation, workspace (4aa89ab42d) (written) |
| TASK-15262 | TASK-12969 | renumbered copy | TASK-15260 | left: task edit reaches only backlog/tasks | the only file holding it (5845211cc0) (written) |

Actions: changed 45 (in 41 files), left as is 107 (keeper, neither, unclear), left to open PRs 2, left in `backlog/completed` 3.

## Open-PR overlaps

The full file list of every open PR to `dev` (34 PRs) was read on 2026-11-03. The 11 PRs over `gh pr list`'s 100-file cap were paged through the REST API.

| PR | Overlap | Handling |
|---|---|---|
| #2855 | `backlog/tasks/task-13145 - Implement-canonical-admin-webhook-durable-producers-and-final-activation.md` (AC 2 dependency) | left; the owner changes it (Task 3 Step 1) |
| #2421 | `backlog/tasks/task-2281 - Add-LSP-backed-code-intelligence-MCP-tools.md` (AC 2 dependency, AC 4 references entry) | left |
| #3323 | `backlog/tasks/task-13421 - Assess-Jobs-completion-SQL-ownership-after-atomicity-fix.md` (AC 4 references entry) | left (`--skip`) |
| #2738 | its own `backlog/tasks/task-12970 - Harden-playlist-ingest-review-boundaries.md`: same id as one of the 41 AC 4 files, a different file | **harmless**; the record already tells #2738 to renumber it (plan review M9) |
| #3263 | `tools/backlog-py/src/backlog_py/core/repository.py` | **harmless.** Its hunks are in `__init__`, `create_task` and `_next_task_id`; this plan's are in `edit_task`, `renumber_task` and a new `_remove_preamble`. #3263 is based before #3392 (it still calls `_task_exists`), so it must be rebased anyway. |

No open PR touches any of the 43 subtasks, the 6 new parents, TASK-15183, TASK-13534, TASK-13962, the 41 AC 4 files, the record, `AGENTS.md` or the README. #3071 and #3090 add other `task-13421 ...` files; this plan does not touch them.

---

## Decisions

1. **ADR-004 Change summary.** This PR is AI-authored. The waivers on record cover named PRs, not this one.
   - **Recommendation:** ask the owner to extend the waiver to TASK-13962's PR, as for #3408. Otherwise the owner writes the `## Change summary` before merge.
2. **AC 2's home.** Both PRs are the owner's and open. TASK-13962 cannot finish AC 2 while they are.
   - **Recommendation:** a small follow-up task, TASK-15419, carries AC 2. It also carries the two AC 4 entries held by #2421 and #3323. TASK-13962 removes AC 2 (`--remove-ac 2`, with a note naming TASK-15419) and closes Done.
   - **Alternative:** TASK-13962 keeps AC 2 unchecked and stays In Progress after the merge, until the PRs land.
   - **Execution:** this plan executes the recommendation unless the owner answers otherwise. If all three PRs have closed by Task 0 Step 1, no follow-up is filed and Task 3 Step 1 applies the edits.
3. **TASK-13534, beyond the ACs.** It has the same defect as TASK-15183: it was hand-written without an opening `---`, so backlog-py shows it untitled and "To Do". The same command repairs it. Its frontmatter then has no `id:` line; backlog-py and `check_backlog_task_ids.py` read the id from its file name.
   - **Recommendation:** include it (Task 2 Step 5).
   - **Alternative:** drop that step to keep to the ACs' letter.
4. **The 12 other parent mismatches (V8)** and **the 32 "neither" references** (stale since the July renumber, not since #3392) are outside TASK-13962's ACs.
   - **Recommendation:** leave both. The References table names each July id, so a reader can resolve them. File a task only if the owner wants the subtask tree consistent.

Not owner questions (rulings for the reviewer, in Review Focus):
- `--ref` replaces the list and does not split on commas;
- the subtask order for 418;
- notes on the moved files and parents;
- no ADR change.

---

## Global Constraints

Carried verbatim from the brief:

- Backlog:
  - edit task files with backlog-py only (`PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task ...`);
  - `--append-notes`, never `--notes`; stage with `git add -A backlog/tasks backlog/completed` (renames);
  - never hand-edit task files;
  - if a new task must be filed, its explicit id comes from `zsh <plans dir>/task_next.sh` (prints the HIGHEST id in
    use; the new id is that plus one), after `mkdir <scratchpad>/id_lock`; release it once the PR's file list shows it.
- Branch from `origin/dev`; the PR targets `dev`. Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
  PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Never `--no-verify`, never force-push, never `git stash`. Never push to, comment on, or edit another PR.
- Mirrored docs (`Docs/Published`) need `bash Helper_Scripts/refresh_docs_published.sh` in the same PR (check whether the
  record or ADR-084 is mirrored).
- CI checks before the PR, in the last task:
  - `tests/Docs`, `tests/lint`, `tests/CI`, and backlog-py's tests;
  - shard coverage (`Helper_Scripts/ci/check_shard_coverage.py`) and `Helper_Scripts/ci/test_quality_triage.py --enforce`;
  - Bandit (`/opt/homebrew/bin/uvx bandit -ll`) on touched source, if any;
  - `/usr/bin/python3 Helper_Scripts/ci/check_backlog_task_ids.py .` exits 0.
  - Commands: see `Docs/superpowers/plans/2026-11-02-deferred-minors-3391-3392.md`'s Task 3.
- Agent shells don't keep the venv `PATH`: prefix commands that spawn `python` with `PATH="$(dirname $PY):$PATH"`.

Plan-specific:

- **`check_backlog_task_ids.py` exits 0 after every commit.** Each task's commit step runs it.
- **Run the drivers from the repository root** with `PYTHONPATH=tools/backlog-py/src`. They call `python -m backlog_py` through `sys.executable`, so the venv python must run them.
- **Run the AC 4 scan once, before any reference edit.** An applied entry no longer names a renumbered id, so a second run would write a shorter table. If a run fails part-way, reset the reference edits (`git checkout -- backlog/tasks` on the files it changed; never `git stash`) and run it again from the start.
- **Git.** Stage named paths. `git diff --cached --check` must print nothing before each commit.
- **Test runs** can leave a read-only Go module cache under `tldw_Server_API/Config_Files/acp_runner_home`. If it appears, run `chmod -R u+w tldw_Server_API/Config_Files/acp_runner_home && git clean -ffdx -- tldw_Server_API/Config_Files/acp_runner_home`; never `rm -rf` the directory (it holds a tracked file).
- **This PR's file list is over 100** (100 in the prototype, plus the plan copy and the follow-up task: 102). `gh pr view --json files` and `gh pr list` show 100. The reviewer and anyone checking the list use `git diff --name-status -M origin/dev...HEAD` or `gh api --paginate repos/rmusser01/tldw_server/pulls/<n>/files --jq '.[].filename'`. The id lock is released in Task 0, when the draft PR has three files, so `gh pr view <n> --json files` shows the new task file then.

## Review Focus

1. **`renumber_task` sets `parent_task_id` for every dotted new id.**
   - A same-parent renumber writes the parent the file already names. It adds the key to a subtask file that lacked it, which the old code left without one. The 2026-10-31 renumbering only ever did same-parent renumbers: 394.x → 394.8-.14, 45.44.21 → .23, 13260.60 → .282, 13261.10 → .12.
   - Renumbering to a top-level id leaves `parent_task_id` as it is. No caller does that, and there is no rule for which parent it should then name. Task 1's renumber test pins both top-level cases (plan review M3).
   - No parent-exists guard. The id itself already names the parent; a guard would also break `test_renumber_takes_a_subtask_id`, whose TASK-5 does not exist.
2. **`--ref` replaces `references:` and does not split on commas,** unlike `-l` and `--dep`. Entries are URLs, paths and free text. The test pins a URL with a comma. One flag per entry.
3. **`--remove-preamble` deletes text.** It removes only what sits between the frontmatter and the first `## ` heading. It refuses a file without a heading, and a preamble that holds a section marker (a section block needs no heading of its own; plan review I1), each with a test. It was run on two files, each checked in the diff. TASK-15183's dropped `id: TASK-13511` is already named in its notes.
4. **AC 4's verdicts.**
   - Only `renumbered copy` rows change; keeper, neither and unclear stay. A wrong `keeper` or `neither` label therefore costs a missed change, never a wrong one.
   - All 50 copy rows were read against the citing file (table above). The reviewer should read the 29 `decided` copy rows: those are judgement.
   - The "written" rule needs at least 2 shared title words, and twice the runner-up's.
5. **Frontmatter re-dump.** Every `task edit` with a frontmatter change re-serializes the frontmatter with `yaml.safe_dump`: list indent, quotes, `updated_date: '...'` unquoted. That is existing behaviour. `fm_diff_13962.py` shows the parsed values change only as intended (Task 3 Step 3).
6. **418's order:** 418.17 (the old 418.14, created first) → 12318.1, 418.15 → .2, 418.16 → .3.
7. **No ADR change.** ADR-084's decision is unchanged. The new renumber behaviour is one sentence in `AGENTS.md` and the README.

## Files

| File | Task | Change |
|---|---|---|
| `tools/backlog-py/src/backlog_py/core/repository.py` | 1 | `edit_task(references=, remove_preamble=)`, `renumber_task` sets `parent_task_id`, new `_remove_preamble` (+28/-2) |
| `tools/backlog-py/src/backlog_py/cli/main.py` | 1 | `--ref`, `--remove-preamble` (+8) |
| `tools/backlog-py/tests/test_task_mutations.py` | 1 | 5 tests (+75) |
| 43 subtask files (40 in `backlog/tasks`, 3 in `backlog/completed`) | 2 | renamed by `task renumber`: `id`, `parent_task_id`, one note |
| 6 parent files (`task-12295`, `12303`, `12318`, `12418`, `12446`, `12561`) | 2 | one note each |
| `task-15183 ...`, `task-13534 ...` | 2 | title, status, labels; preamble removed |
| 41 files in `backlog/tasks` | 3 | `references:` entries (45) |
| `Docs/Development/Backlog_Task_Id_References.md` | 3 | new, generated (157 rows) |
| `Docs/Development/Backlog_Task_Id_Renumbering.md` | 4 | follow-up section (+70) |
| `AGENTS.md`, `tools/backlog-py/README.md` | 4 | one clause each |
| `task-13962 ...`, `task-15419 ...`, `Docs/superpowers/plans/2026-11-03-reparent-orphaned-subtasks-13962.md` | 0, 4 | the tasks; this plan |

- **Ratchets.** No touched file is in `Helper_Scripts/ci/scope_predicate_baseline.txt` or `function_size_baseline.json` (grep: no `backlog` keys).
- **Mirror.** `Docs/Development/` is not mirrored (`refresh_docs_published.sh` copies `ADR`, the guides and the published sections), and ADR-084 does not change. `bash Helper_Scripts/refresh_docs_published.sh` must leave `git status --short Docs/Published` empty (Task 4 Step 4).

## Test commands

- **backlog-py:** `PATH="$(dirname $PY):$PATH" PYTHONPATH=tools/backlog-py/src $PY -m pytest -p no:cacheprovider -q tools/backlog-py/tests`. Measured: `158 passed` on `a960463a26`; `5 failed, 158 passed` with Task 1's tests on unchanged code; `163 passed` after (with the I1 fold, measured by the controller).
- **The CI suite** (the brief's list, plus the renumber generator's tests, which import backlog-py):
  `PATH="$(dirname $PY):$PATH" $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Docs tldw_Server_API/tests/lint tldw_Server_API/tests/CI tools/backlog-py/tests tldw_Server_API/tests/Helper_Scripts/test_renumber_duplicate_task_ids.py`
  Measured: `1611 passed, 5 skipped` on `a960463a26` (about 6 minutes); `1615 passed, 5 skipped` on the prototype (about 9 minutes), exactly its 4 new tests; with the I1 fold's fifth test, expect `1616 passed, 5 skipped` (the plan review's figure). It includes the task-format ratchet, which read the prototype's changed task files from a local commit, and the id check's tests. No other test names a touched symbol (`git grep -l -E "renumber_task|edit_task|backlog_py|remove_preamble"` over test files: backlog-py's own, `CI/test_backlog_task_format_ratchet.py`, `CI/test_path_classifier.py`, `lint/test_first_party_source_pinning.py`, all in the suite).

---

## Task 0: Branch, start the task, file the follow-up, draft PR, baseline

- [ ] **Step 1: Re-check the PRs and the overlaps.**
  - `git fetch origin dev --quiet`, then `for n in 2855 2421 3323; do gh pr view $n --json number,state,mergedAt --jq '"\(.number) \(.state) \(.mergedAt)"'; done`. Measured on 2026-11-03: all three `OPEN`.
  - Re-read the open PRs' full file lists (as in "Open-PR overlaps"), and confirm nothing new touches the files in "Files". A new overlap on a task file: leave that file out and add it to Task 3's `--skip` list (AC 4) or skip its step (AC 1, 3), and say so in the PR body.
- [ ] **Step 2: Branch.** `git switch -c fix/reparent-orphaned-subtasks-13962 origin/dev`.
- [ ] **Step 3: File the follow-up** (Decision 2), only if at least one of the three PRs is still open:
  - `mkdir <scratchpad>/id_lock` (if it exists, wait; never break another session's lock). Run `zsh <plans dir>/task_next.sh` from the repository root. `<FU>` is the printed value + 1 (dev's highest was 15418 on 2026-11-03). `ls backlog/tasks backlog/completed backlog/archive/tasks | grep "^task-15419 "` must print nothing.
  - Leave out the AC of any PR that has closed (Task 3 Step 1 applies its edit instead):

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task create "Apply the backlog edits TASK-13962 left to open PRs #2855, #2421 and #3323" \
  --id TASK-15419 -l backlog --dep TASK-13962 \
  -d "TASK-13962 re-parented the subtasks 3a660d4f1d orphaned and pointed bare-id references at the renumbered copies they mean, but could not change three task files that open PRs change. Each PR should make its file's change; if a PR closes without it, apply it here with task edit. Docs/Development/Backlog_Task_Id_Renumbering.md ('Left to open PRs') lists the lines." \
  --ac "task-13145 (canonical admin webhook producers) depends on TASK-15132, not TASK-13111: by #2855, or with task edit --dep once #2855 closes without the change" \
  --ac "task-2281 (LSP-backed MCP tools) depends on and references TASK-15254, not TASK-2387: by #2421, or with task edit --dep and --ref once #2421 closes without the change" \
  --ac "task-13421 (Jobs completion SQL ownership) references TASK-15149, not TASK-13215: by #3323, or with task edit --ref once #3323 closes without the change"
```

- [ ] **Step 4: Start TASK-13962.**
  `PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13962 -s "In Progress" --append-notes "Plan: Docs/superpowers/plans/2026-11-03-reparent-orphaned-subtasks-13962.md. Follow-up for AC 2: TASK-15419. Owner answers: Decision 1 (Change summary) <answer>; Decision 2 (AC 2's home) <answer>; Decision 3 (TASK-13534) <answer>."`
  - `<answer>` is the owner's recorded answer to that decision (the controller asks before execution and passes the answers in the dispatch). Never write `<answer>` literally: notes are append-only (plan review M1).
  - If Step 3 filed no follow-up, drop `Follow-up for AC 2: TASK-15419.` from the note.
  Measured: `TASK-13962 - Re-parent ... (In Progress)`.
- [ ] **Step 5: Plan copy, commit, push, draft PR, release the lock.**
  - Copy this plan to `Docs/superpowers/plans/2026-11-03-reparent-orphaned-subtasks-13962.md`, with `TASK-15419` and `task-15419` both replaced by the real id (case-insensitively; plan review M6). Keep `<date>`, `<answer>` and the other placeholders.
  - `git add Docs/superpowers/plans/2026-11-03-reparent-orphaned-subtasks-13962.md && git add -A backlog/tasks backlog/completed && git diff --cached --check` (nothing). Then `/usr/bin/python3 Helper_Scripts/ci/check_backlog_task_ids.py .` must exit 0. Commit (message 0).
  - `git push -u origin fix/reparent-orphaned-subtasks-13962`.
  - `gh pr create --draft --base dev --title "chore(backlog): re-parent the subtasks 3a660d4f1d orphaned; repair TASK-15183; retarget references (TASK-13962)"`. The body is one line naming TASK-13962 and TASK-15419, and ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
  - When `gh pr view <n> --json files --jq '.files[].path'` lists `backlog/tasks/task-15419 - ...md`, `rmdir <scratchpad>/id_lock`.
- [ ] **Step 6: Baseline.** On this commit, run backlog-py's tests (expect `158 passed`) and the CI suite (Test commands; expect `1611 passed, 5 skipped`, measured on `a960463a26`). Record the counts and every failing id in the ledger. Later failures are compared with these; never dismiss one as environment-only without them.

## Task 1: backlog-py: `task renumber` re-parents, `task edit --ref`, `task edit --remove-preamble`

**Files:** `tools/backlog-py/src/backlog_py/core/repository.py`; `tools/backlog-py/src/backlog_py/cli/main.py`; `tools/backlog-py/tests/test_task_mutations.py`.

**Interfaces:**
- `MutableRepository.edit_task(..., references: Sequence[str] | None = None, ..., remove_preamble: bool = False)`. `references=None` leaves the list; a list replaces it.
- `MutableRepository.renumber_task(path, new_id, *, note=None)`: signature unchanged; a dotted `new_id` also sets `parent_task_id`.
- New private `_remove_preamble(source: str, parsed: ParsedTaskMarkdown) -> str`, raising `TaskMutationError` without a `## ` heading or when the preamble holds a section marker.
- CLI: `task edit <id> --ref X [--ref Y ...]`, `task edit <id> --remove-preamble`.

- [ ] **Step 1: Write the tests.** Append to the end of `tools/backlog-py/tests/test_task_mutations.py`:

```python


def test_renumber_to_a_subtask_id_points_parent_task_id_at_the_new_parent(tmp_path: Path) -> None:
    # TASK-13962: 3a660d4f1d renumbered parents without their subtasks; a renumber re-parents them.
    repo = _copy_fixture(tmp_path)
    repository = _repository(repo)
    repository.create_task(title="New parent", task_id="TASK-8")
    child = repository.create_task(title="Orphan", task_id="TASK-2.1").path
    source = child.read_text(encoding="utf-8")
    child.write_text(source.replace("id: TASK-2.1\n", "id: TASK-2.1\nparent_task_id: TASK-2\n"), encoding="utf-8")

    moved = repository.renumber_task(child.relative_to(repo), "TASK-8.1")

    assert moved.parsed.frontmatter["parent_task_id"] == "TASK-8"
    # A top-level id names no parent: a top-level task gains none, and a subtask keeps the one it has.
    top = repository.create_task(title="Top", task_id="TASK-9").path.relative_to(repo)
    assert "parent_task_id" not in repository.renumber_task(top, "TASK-10").parsed.frontmatter
    lifted = repository.renumber_task(moved.path.relative_to(repo), "TASK-11")
    assert lifted.parsed.frontmatter["parent_task_id"] == "TASK-8"


def test_cli_edit_ref_replaces_the_references_one_entry_per_flag(tmp_path: Path) -> None:
    repo = _copy_fixture(tmp_path)
    # A comma stays inside its entry: references hold URLs, so --ref does not split like -l and --dep.
    refs = ["TASK-15005", "https://example.test/a,b"]
    args = ["--cwd", str(repo), "task", "edit", "TASK-1", "--ref", refs[0], "--ref", refs[1]]

    result = CliRunner().invoke(main, args)

    assert result.exit_code == 0, result.output
    assert _frontmatter(_task_file(repo))["references"] == refs


_STRAY = "id: TASK-13511\ntitle: Old title\nstatus: Done\nlabels:\n- frontend\n---\n\n"


def test_cli_edit_remove_preamble_drops_the_text_above_the_first_heading(tmp_path: Path) -> None:
    # TASK-15183 kept its hand-written frontmatter, which had no opening ---, as body text.
    repo = _copy_fixture(tmp_path)
    path = _task_file(repo)
    source = path.read_text(encoding="utf-8")
    path.write_text(source.replace("---\n\n## Description", f"---\n{_STRAY}## Description", 1), encoding="utf-8")

    result = CliRunner().invoke(main, ["--cwd", str(repo), "task", "edit", "TASK-1", "--remove-preamble"])

    assert result.exit_code == 0, result.output
    written = _task_file(repo).read_text(encoding="utf-8")
    assert "TASK-13511" not in written
    assert "custom_field: preserve-me\nnested_unknown:\n  source: fixture\n---\n\n## Description\n" in written


def test_remove_preamble_refuses_a_task_with_no_section_heading(tmp_path: Path) -> None:
    repo = _copy_fixture(tmp_path)
    path = _task_file(repo)
    path.write_text("---\nid: TASK-1\ntitle: Bare\n---\nonly text\n", encoding="utf-8")
    before = _snapshot_tasks(repo)

    with pytest.raises(TaskMutationError, match="no section heading"):
        _repository(repo).edit_task("TASK-1", remove_preamble=True)
    assert _snapshot_tasks(repo) == before


def test_remove_preamble_refuses_a_preamble_that_holds_a_section(tmp_path: Path) -> None:
    # A section block needs no heading of its own; removing the preamble must never delete one.
    repo = _copy_fixture(tmp_path)
    path = _task_file(repo)
    plan = "<!-- SECTION:PLAN:BEGIN -->\nkeep\n<!-- SECTION:PLAN:END -->\n\n"
    source = path.read_text(encoding="utf-8")
    path.write_text(source.replace("---\n\n## Description", f"---\n\n{plan}## Description", 1), encoding="utf-8")
    before = _snapshot_tasks(repo)

    with pytest.raises(TaskMutationError, match="section marker"):
        _repository(repo).edit_task("TASK-1", remove_preamble=True)
    assert _snapshot_tasks(repo) == before
```

- [ ] **Step 2: Run, and see it fail.** `PATH="$(dirname $PY):$PATH" PYTHONPATH=tools/backlog-py/src $PY -m pytest -p no:cacheprovider -q tools/backlog-py/tests`
  Expected: `5 failed, 158 passed` (measured by the controller with the folded tests). The failures:
  - `test_renumber_to_a_subtask_id_points_parent_task_id_at_the_new_parent`: `AssertionError: assert 'TASK-2' == 'TASK-8'`;
  - `test_cli_edit_ref_replaces_the_references_one_entry_per_flag`: `Error: No such option '--ref'. Did you mean '--dep'?` (`assert 2 == 0`);
  - `test_cli_edit_remove_preamble_drops_the_text_above_the_first_heading`: `Error: No such option '--remove-preamble'. Did you mean '--remove-ac'?`;
  - `test_remove_preamble_refuses_a_task_with_no_section_heading` and `test_remove_preamble_refuses_a_preamble_that_holds_a_section`: `TypeError: MutableRepository.edit_task() got an unexpected keyword argument 'remove_preamble'`.
- [ ] **Step 3: Implement** in `tools/backlog-py/src/backlog_py/core/repository.py`.

  (a) `edit_task`'s signature: after `labels: Sequence[str] | None = None,` add `references: Sequence[str] | None = None,`; after `on_status_change: bool | None = None,` add `remove_preamble: bool = False,`. Its docstring's second paragraph becomes:

```python
        ``append_notes`` may be one text or several, appended in order. Indexes in
        check/uncheck refer to criteria after removals and additions. ``references``
        replaces the references list. ``remove_preamble`` drops the body text above
        the first ``## `` heading, and refuses when that text holds a section.
```

  (b) Directly after the `parsed = parse_task_markdown(source)` that follows the normalize `try`/`except`, and before `if description is not None:`, add:

```python
        if remove_preamble:
            source = _remove_preamble(source, parsed)
            parsed = parse_task_markdown(source)
```

  (c) In the `updates` block, after `if labels is not None: updates["labels"] = list(labels)`, add:

```python
        if references is not None:
            updates["references"] = list(references)
```

  (d) In `renumber_task`, add one docstring line before the `note` line: ``        A subtask id (``P.n``) also sets ``parent_task_id`` to ``P``.``. Then replace `source = _replace_frontmatter_values(source, parsed, {"id": normalized_id})` with:

```python
        updates: dict[str, object] = {"id": normalized_id}
        parent = normalized_id.rpartition(".")[0]
        if parent:  # a subtask id names its parent; keep parent_task_id in step (TASK-13962)
            updates["parent_task_id"] = parent
        source = _replace_frontmatter_values(source, parsed, updates)
```

  (e) Add this function directly above `def _replace_section(`:

```python
def _remove_preamble(source: str, parsed: ParsedTaskMarkdown) -> str:
    """Drop the body text above the first ``## `` heading; one blank line then separates it from the frontmatter."""
    match = re.search(r"^## ", parsed.body, re.MULTILINE)
    if match is None:
        raise TaskMutationError("Cannot remove the preamble: the task has no section heading")
    if re.search(r"^<!-- [A-Z0-9_ :-]+:(?:BEGIN|END) -->", parsed.body[: match.start()], re.MULTILINE):
        raise TaskMutationError("Cannot remove the preamble: it holds a section marker")
    head = source[: len(source) - len(parsed.body)]
    newline = head[len(head.rstrip("\r\n")):] or "\n"
    return f"{head}{newline}{parsed.body[match.start():]}"
```

- [ ] **Step 4: Implement** in `tools/backlog-py/src/backlog_py/cli/main.py`:
  - Directly above `@click.option("--ac", "add_ac", ...)`, add:

```python
@click.option(
    "--ref", "references", multiple=True, help="(edit) One reference (URL, path or task id) per flag; replaces them."
)
@click.option("--remove-preamble", is_flag=True, help="Drop the body text above the first section heading (edit).")
```

  - In `task_command`'s parameters, after `dependencies: tuple[str, ...],`, add `references: tuple[str, ...],` and `remove_preamble: bool,`.
  - In the `edit` branch's `edit_task(...)` call, after `labels=...`, add `references=list(references) if references else None,` and `remove_preamble=remove_preamble,`.
- [ ] **Step 5: Run, and see it pass.** The Step 2 command. Expected: `163 passed` (measured by the controller with the folds; the folded diff is `proto_13962_backlogpy_folded.patch` in the plans dir). Then:
  - `$PY -m ruff check tools/backlog-py/src/backlog_py/core/repository.py tools/backlog-py/src/backlog_py/cli/main.py tools/backlog-py/tests/test_task_mutations.py`: "All checks passed!" (measured). No new line is over 120 characters.
  - `git diff tools/backlog-py` shows only Steps 1, 3 and 4.
- [ ] **Step 6: Commit** (message 1): `git add tools/backlog-py/src/backlog_py/core/repository.py tools/backlog-py/src/backlog_py/cli/main.py tools/backlog-py/tests/test_task_mutations.py && git diff --cached --check`. Then `/usr/bin/python3 Helper_Scripts/ci/check_backlog_task_ids.py .` exits 0.

## Task 2: Re-parent the 43 subtasks (AC 1); repair TASK-15183 and TASK-13534 (AC 3)

**Files:** the 43 subtask files, the 6 new parents, `task-15183 ...`, `task-13534 ...` (backlog-py only).

- [ ] **Step 1: Before.**
  - `PYTHONPATH=tools/backlog-py/src $PY <plans dir>/parent_check_13962.py | head -1`. Expected: `12` (V8; measured).
  - `PYTHONPATH=tools/backlog-py/src $PY <plans dir>/reparent_13962.py --verify`. Expected: `verified 0 of 43` and 86 `PROBLEM` lines (each old id still has a file; each new id has none), exit 2 (measured).
- [ ] **Step 2: Dry run.** `PYTHONPATH=tools/backlog-py/src $PY <plans dir>/reparent_13962.py`.
  - Expected on stderr: `43 subtasks under 6 parents`. Then the 43-row table (`| Old id | New id | File |`), exit 0.
  - A `PROBLEM` line (exit 2) means the base moved: a new child under a new parent, a missing file, or a changed `parent_task_id`. Stop and re-plan that set.
- [ ] **Step 3: Apply.** `PYTHONPATH=tools/backlog-py/src $PY <plans dir>/reparent_13962.py --apply > <scratchpad>/reparent_table.md`. That runs 43 `task renumber "<path>" --id <new> --append-notes "<note>"` and 6 `task edit <parent> --append-notes "<note>"`. Their output lines go to stderr, so stdout (the file) holds only the 45-line table.
  - The subtask note reads: `Re-parented from TASK-481.13 to TASK-12446.13 on <date> (TASK-13962): its parent TASK-481 became TASK-12446 when 3a660d4f1d (2026-07-05) renumbered it without its subtasks, and TASK-481 now names only "Design Watchlists durable audio artifact projection" (backlog/tasks/task-481 - Design-Watchlists-durable-audio-artifact-projection.md). Docs/Development/Backlog_Task_Id_Renumbering.md maps every moved id.`
  - TASK-418.17's note says `TASK-418.17 (TASK-418.14 before 2026-10-31)`.
  - The parent note reads: `Its subtasks TASK-481.1 to TASK-481.19 were re-parented as TASK-12446.1 to TASK-12446.19 on <date> (TASK-13962), in that order: 3a660d4f1d (2026-07-05) renumbered this task from TASK-481 without them. ...`. For 418 and 410 it lists the ids. For 410's single subtask it reads `Its subtask TASK-410.1 was re-parented as TASK-12303.1 on <date> (TASK-13962): 3a660d4f1d (2026-07-05) renumbered this task from TASK-410 without it. ...` (plan review M4; the script is fixed).
  - It ends with its own check: `verified 43 of 43` on stderr, exit 0 (measured). Any `PROBLEM` line: stop.
  - Keep `<scratchpad>/reparent_table.md`: Task 4 puts it in the record.
- [ ] **Step 4: Check.**
  - `git add -A backlog/tasks backlog/completed && git status --short | grep -c '^R '`: `43`. `git status --short | grep -c '^M '`: `6` (the new parents).
  - `PYTHONPATH=tools/backlog-py/src $PY <plans dir>/parent_check_13962.py | head -1`: still `12` (measured): no new mismatch.
  - `/usr/bin/python3 Helper_Scripts/ci/check_backlog_task_ids.py .`: exit 0.
  - Spot-check one rename: `git diff --cached -M -- 'backlog/tasks/task-481.13 - PR-4---Notes-navigation-and-first-time-UX.md' 'backlog/tasks/task-12446.13 - PR-4---Notes-navigation-and-first-time-UX.md'` shows `similarity index 85%`, `-id: TASK-481.13` / `+id: TASK-12446.13`, `-parent_task_id: TASK-481` / `+parent_task_id: TASK-12446`, and one added note line.
  - Commit (message 2) after `git diff --cached --check`.
- [ ] **Step 5: TASK-15183 and TASK-13534** (AC 3; the second is Decision 3):

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-15183 \
  -t "Remediate WebUI and extension performance review findings" -s Done -l frontend,performance,architecture --remove-preamble
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13534 \
  -t "Fast-follow PR #3210 review findings (storage index safety, researchActions staleness, sidebar debounce)" \
  -s Done -l frontend,review-followup --remove-preamble
```

  Expected: `TASK-15183 - Remediate WebUI and extension performance review findings (Done)` and `TASK-13534 - Fast-follow PR #3210 review findings (...) (Done)`. Neither file is renamed: each title slugs to its current file-name tail.
  - `git diff -- 'backlog/tasks/task-15183 - Remediate-WebUI-and-extension-performance-review-findings.md'`: the second `---`, `id: TASK-13511` and the old `title:` line go; `title:` is added after `labels`; nothing below `---` changes.
  - In `task-13534`, the opening `---` is added and `id: TASK-13534` goes. `status`, `labels` and a re-wrapped `title` form the frontmatter.
  - `PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task TASK-15183 --plain | head -1`: `TASK-15183 [Done] Remediate WebUI and extension performance review findings`.
  - `git add -A backlog/tasks && git diff --cached --check`; `check_backlog_task_ids.py` exit 0; commit (message 3).

## Task 3: References (AC 4) and the PR-held retargets (AC 2)

**Files:** 41 files in `backlog/tasks` (backlog-py only); `Docs/Development/Backlog_Task_Id_References.md` (generated).

- [ ] **Step 1: AC 2.** For each of #2855, #2421 and #3323, run `gh pr view <n> --json state,mergedAt`.
  - **OPEN:** leave its file. Its `--skip` stays in Step 2.
  - **MERGED or CLOSED:** `git fetch origin dev` and merge `origin/dev` into the branch. Then read the file's entry:
    - if the PR already made the change, check that TASK-15419's AC (`task edit TASK-15419 --check-ac N --append-notes "Done by #<n>."`);
    - if not, apply it here and drop its `--skip`:
      - #2855: `task edit TASK-13145 --dep TASK-15132`;
      - #2421: `task edit TASK-2281 --dep TASK-15254 --ref https://code.claude.com/docs/en/tools-reference --ref TASK-15254`;
      - #3323: `task edit TASK-13421 --ref TASK-15149 --ref "https://github.com/rmusser01/tldw_server/pull/3092#discussion_r4171067327"`.
      - Re-read each file's current list first: `--dep` and `--ref` replace the whole list. Then check TASK-15419's AC with a note.
  - Measured on 2026-11-03: all three OPEN, so nothing is applied here.
- [ ] **Step 2: Scan, write the table, apply.** Run once (Global Constraints):

```bash
PYTHONPATH=tools/backlog-py/src $PY <plans dir>/scan_refs_13962.py \
  --table Docs/Development/Backlog_Task_Id_References.md --json <scratchpad>/refs_13962.json \
  --skip TASK-13421=3323 --skip TASK-2281=2421 --apply
```

  Expected on stderr: `157 entries in 126 files. Verdicts: keeper 73, neither 32, renumbered copy 50, unclear 2. Actions: changed 45, left as is 107, left: open PR #2421 changes this file 1, left: open PR #3323 changes this file 1, left: task edit reaches only backlog/tasks 3.`, then 41 `TASK-... - ...` lines, one per `task edit --ref`, exit 0 (measured: about 4 minutes).
  - Each `task edit` call is checked: the script re-reads the file and exits with `... references are ..., expected ...` if the list differs.
  - The table's header names the branch's merge base with `origin/dev` "plus that PR's re-parenting commits", never a branch commit, which a server-side rebase would orphan (plan review M2; the script is fixed).
  - A different count means the base moved. Read the new or changed rows in the JSON before going on; add a `DECISIONS` entry only from the citing file's own text.
- [ ] **Step 3: Check.**
  - `git status --short backlog | grep -c '^ M'`: `41`.
  - `PYTHONPATH=tools/backlog-py/src $PY <plans dir>/fm_diff_13962.py` (it compares with `origin/dev` after `git add -A backlog/tasks backlog/completed`). Expected (measured, with the other tasks' commits):

```text
  6 M - body changed
  2 M labels,status,title body changed
 41 M references body same
  1 M status body same
 43 R id,parent_task_id body changed
```

    plus `1 A new file` for TASK-15419. TASK-13962's line is `status`; it also gets a note, so it may read `status body changed`.
  - `grep -c '^| \`' Docs/Development/Backlog_Task_Id_References.md`: `157`.
  - `/usr/bin/python3 Helper_Scripts/ci/check_backlog_task_ids.py .`: exit 0.
- [ ] **Step 4: Commit** (message 4): `git add -A backlog/tasks && git add Docs/Development/Backlog_Task_Id_References.md && git diff --cached --check`.

## Task 4: Docs, verify, close, review, PR

- [ ] **Step 1: The record.** Append to `Docs/Development/Backlog_Task_Id_Renumbering.md` (a doc, not a task file):
  - this head, with `<date>` filled;
  - then the 45 table lines of `<scratchpad>/reparent_table.md` (`| Old id | New id | File |`, the separator, 43 rows);
  - then the tail, with `TASK-15419` filled. If Task 3 Step 1 applied an edit, drop that PR's bullet and say so.

```markdown

## Follow-up (TASK-13962, <date>)

### Subtasks re-parented

The 43 subtasks that "Subtasks attributed by topic" gives to no copy named a parent id that 3a660d4f1d (2026-07-05) had moved to another task. `task renumber` moved each under its real parent, in their original order; for a subtask id it now also sets `parent_task_id`. Each moved file's notes name its old id, and each new parent's notes name the range.

```

```markdown

No dependency or `references:` entry names an old subtask id. The old ids stay, as history, in the moved files' own text, in their parents' notes, in `Docs/Plans/2026-06-07-knowledge-qa-follow-on-gap-matrix.md` and the nine `Docs/superpowers/plans/2026-06-07-knowledge-*` plans (TASK-528.1-.9), in one `git add` line of `Docs/superpowers/plans/2026-05-17-llamacpp-managed-runtime-implementation-plan.md` (task-418.15's path, next to task-418's pre-July path), and in `TOPIC_EXCEPTIONS` of the one-off generator `Helper_Scripts/renumber_duplicate_task_ids.py` (the 418.14-.16 paths).

### TASK-15183 and TASK-13534

Both were written by hand without an opening `---`. `task edit --title -s -l --remove-preamble` gave each its title, status and labels in the frontmatter and dropped the old block from the body (TASK-15183's `id: TASK-13511`, which its notes already name). TASK-13534's frontmatter has no `id:` line: backlog-py and `check_backlog_task_ids.py` read its id from the file name.

### References

[Backlog_Task_Id_References.md](Backlog_Task_Id_References.md) checks each bare-id `references:` entry that names a renumbered id: 157 entries in 126 files. The 45 that meant a renumbered copy, in `backlog/tasks` files no open PR changes, now name its new id (`task edit --ref`). Three more are in `backlog/completed`, which `task edit` cannot reach, and stay as they are; two are left to open PRs (below).

### Left to open PRs

An open PR changes each of these files, so TASK-13962 leaves them. The PR should make the change; if it closes without it, TASK-15419 applies it with `task edit`.

- #2855 (rmusser01): in `backlog/tasks/task-13145 - Implement-canonical-admin-webhook-durable-producers-and-final-activation.md`, change the dependency `TASK-13111` to `TASK-15132`.
- #2421 (rmusser01): in `backlog/tasks/task-2281 - Add-LSP-backed-code-intelligence-MCP-tools.md`, change the dependency and the `references:` entry `TASK-2387` to `TASK-15254`.
- #3323 (rmusser01): in `backlog/tasks/task-13421 - Assess-Jobs-completion-SQL-ownership-after-atomicity-fix.md`, change the `references:` entry `TASK-13215` to `TASK-15149`.
```

  Measured: the section is 70 lines; `git diff --check` passes.
- [ ] **Step 2: `AGENTS.md`** (Backlog.md Task Tracking, the command list):
  - In the `- Edit:` line (`:433`), replace the ending `` `--dep TASK-1`. `` with:

```markdown
`--dep TASK-1`, `--ref X` (one reference per flag; replaces them), `--remove-preamble` (drops body text above the first `## ` heading; refuses if that text holds a section).
```

  - At the end of the `- Renumber:` line (`:434`), after `` `backlog/archive/tasks` holds. ``, append ` A subtask id (`P.n`) also sets `parent_task_id` to `P`.`
- [ ] **Step 3: `tools/backlog-py/README.md`** ("Task ids"). Replace

```markdown
file-name tail, and `--append-notes` records why. The `changes` job of
```

  with

```markdown
file-name tail, and `--append-notes` records why. A subtask id (`P.n`) also sets
`parent_task_id` to `P`, which re-parents a subtask. The `changes` job of
```

- [ ] **Step 4: Mirror check and commit** (message 5).
  - `bash Helper_Scripts/refresh_docs_published.sh`, then `git status --short Docs/Published`: empty (nothing touched is mirrored).
  - `git add Docs/Development/Backlog_Task_Id_Renumbering.md AGENTS.md tools/backlog-py/README.md && git diff --cached --check`; commit.
- [ ] **Step 5: Verify.** Run each, and compare with Task 0 Step 6's baseline:
  - backlog-py: `163 passed`.
  - The CI suite (Test commands): `1616 passed, 5 skipped` (the prototype's 1615 plus the I1 test). It must be Task 0's baseline plus exactly the 5 new tests. Any other new failure is this plan's.
  - `PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/check_shard_coverage.py`: `new_uncovered=0`, "OK — no newly-unshared test files." (measured).
  - `PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/test_quality_triage.py --enforce`: exit 0, "OK — no offenses beyond the baseline." (measured).
  - `/usr/bin/python3 Helper_Scripts/ci/check_backlog_task_ids.py .`: exit 0 (measured).
  - `PYTHONPATH=tools/backlog-py/src $PY <plans dir>/parent_check_13962.py | head -1`: `12`; `PYTHONPATH=tools/backlog-py/src $PY <plans dir>/reparent_13962.py --verify`: `verified 43 of 43`, exit 0.
  - `git diff origin/dev --check`: nothing.
  - `git grep -n "TASK-15419" -- ':!Docs/superpowers/plans'`: nothing.
  - `git diff --name-status -M origin/dev...HEAD | wc -l`: `102` (100 in the prototype, plus the plan and TASK-15419).
- [ ] **Step 6: Bandit,** on the touched source: `/opt/homebrew/bin/uvx bandit -ll tools/backlog-py/src/backlog_py/core/repository.py tools/backlog-py/src/backlog_py/cli/main.py`. Measured: "No issues identified." (Medium 0, High 0).
- [ ] **Step 7: Close TASK-13962** (Decision 2's recommendation; fill every `<...>` first):

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13962 \
  --remove-ac 2 -t "Re-parent the subtasks orphaned by 3a660d4f1d and finish the renumbering's leftovers" \
  --append-notes "AC 2 (the dependency retargets left to #2855 and #2421) moved to TASK-15419 on <date>, with the references entries left to #2421 and #3323: all three PRs were open at execution. The record's 'Left to open PRs' names the line each PR must change." \
  --append-notes "AC 1: 43 subtasks re-parented with task renumber (TASK-12295.1-.6, TASK-12303.1, TASK-12318.1-.3, TASK-12418.1-.5, TASK-12446.1-.19, TASK-12561.1-.9), in their old order; 418.17 (the old 418.14) first. backlog-py's renumber now sets parent_task_id for a subtask id. Each moved file and each new parent has a note. No dependency or references entry named an old subtask id. reparent_13962.py --verify: 43 of 43. parent_check: 12 before and after; those 12 are outside this task (TASK-12020.46-.52, 12116.1, 12178, 12179, 12570, 18.82.1)." \
  --append-notes "AC 3: TASK-15183 and TASK-13534 (same missing opening ---) got title, status and labels with task edit, and the old block removed with the new task edit --remove-preamble." \
  --append-notes "AC 4: 157 bare-id references entries in 126 files checked (Docs/Development/Backlog_Task_Id_References.md): keeper 73, neither 32, renumbered copy 50, unclear 2. 45 changed with the new task edit --ref; 3 in backlog/completed left (task edit cannot reach them); 2 left to #2421 and #3323 (TASK-15419)." \
  --append-notes "Verification: backlog-py 158 -> 5 failed, 158 passed (RED) -> 163 passed; CI suite <base> -> <after>; shard coverage new_uncovered=0; test-quality ratchet OK; check_backlog_task_ids exit 0; Bandit <result>." \
  --check-ac 1 --check-ac 2 --check-ac 3 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "Re-parented the 43 subtasks 3a660d4f1d left under reused parent ids (now TASK-12295.x, 12303.1, 12318.1-.3, 12418.x, 12446.x, 12561.x) with task renumber, which now also sets parent_task_id. Gave TASK-15183 and TASK-13534 their frontmatter and removed the stray block with the new task edit --remove-preamble. Checked all 157 bare-id references entries that name a renumbered id and pointed the 45 that mean a renumbered copy at its new id with the new task edit --ref (table: Docs/Development/Backlog_Task_Id_References.md). The two dependency retargets and two reference retargets held by open PRs #2855, #2421 and #3323 moved to TASK-15419; the record names each line." \
  -s Done
```

  - `--remove-ac 2` renumbers the old ACs 3 and 4 to 2 and 3; `--check-ac` counts after removals.
  - `-t` retitles the task so it stops promising AC 2 (plan review M10). It renames the file; no open PR touches it, and `git add -A backlog/tasks` stages the rename.
  - If Decision 2 is answered "keep": drop `--remove-ac 2` and `-t`, use `--check-ac 1 --check-ac 3 --check-ac 4`, check DoD 2-6, and keep `-s "In Progress"`.
  - If no follow-up was filed (all PRs closed): drop `--remove-ac 2`, `-t` and the first note, and check ACs 1-4.
  - `git diff origin/dev -- backlog/tasks | grep -E '^\+.*<(answer|base|after|result|date|FU)>'` must print nothing: it checks only this PR's added lines (plan review M1; dev's task-13921 legitimately holds `<base>/<id>`). Then `git add -A backlog/tasks && git diff --cached --check`, and commit (message 6).
- [ ] **Step 8: Review before the PR.** Run `/superpowers:requesting-code-review` in a clean subagent on `origin/dev..HEAD`, with this plan's Review Focus. Its file list comes from `git diff --name-status -M origin/dev...HEAD` (over 100 files). Address the findings before the PR is ready, and record each ruling in the ledger.
- [ ] **Step 9: Finish the PR and merge.**
  - `git push`. Replace the draft's body with `gh pr edit <n> --body-file <file>`, then `gh pr ready <n>`.
  - The body states:
    - what each AC got, with the counts; AC 4 is met "for every file backlog-py can edit" (three copy-meaning entries in `backlog/completed` stay; plan review M7);
    - the AC 4 rules and where the table is;
    - **for the owner, the three lines in "Left to open PRs"** (this PR comments on no other PR);
    - the overlaps;
    - how to list all 102 files (`gh api --paginate .../pulls/<n>/files`);
    - the Review Focus;
    - the `## Change summary` or the waiver (Decision 1).
  - The body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
  - **Merging.** Run `gh variable get MERGE_QUEUE` (unset when this plan was written). When reviews are settled and the Change summary is in, run `QUEUE="<n>:fix/reparent-orphaned-subtasks-13962" sh <plans dir>/pr_queue3.sh` in the background. It reads dev's required checks (8) from the ruleset, rebases server-side when BEHIND, and merges when all pass. Wait for each body edit's re-triggered license status before merging. Confirm MERGED before deleting the branch.

## Commit messages

Each ends with a blank line and `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

- 0: `chore(backlog): start TASK-13962, file TASK-15419 for the PR-held retargets, and add the plan`
- 1: `feat(backlog-py): renumber sets parent_task_id; task edit --ref and --remove-preamble (TASK-13962)`. Body: why each exists (re-parenting, references, the stray TASK-15183 block); `--ref` does not split commas.
- 2: `chore(backlog): re-parent the 43 subtasks 3a660d4f1d orphaned (TASK-13962)`
- 3: `chore(backlog): give TASK-15183 and TASK-13534 their frontmatter (TASK-13962)`
- 4: `chore(backlog): point 45 references at the renumbered copies they mean (TASK-13962)`. Body: the counts and the table's path.
- 5: `docs(backlog): record TASK-13962's re-parenting, repairs and reference check`
- 6: `chore(backlog): close TASK-13962`

## Not measured

- **Open PRs' content.** A task file an open PR adds may cite an old subtask id or one of the 45 retargeted entries' old ids. Only the PRs' file paths were read, not their files' text.
- **`task_next.sh`** was not run: it needs the lock, and the id is taken at execution.
- **The PRs' CI, and GitHub's rendering of the 157-row table.**
- **The verdicts' ground truth.** The `written` rule's picks were read row by row, but only the authors know what each entry meant. That is why only `renumbered copy` rows change.

## Folded after the plan review

The independent review and dry run (`review_plan_13962.md`, plans dir) ran every task from the plan text on `a960463a26`
with placeholder id TASK-99999. Every `Expected:` line matched except Task 4 Step 7's placeholder grep (M1), and the
result matched `proto_13962.patch` apart from the References header SHA (M2). It re-derived the 157 references
independently and spot-checked 32 verdicts (25 "renumbered copy"); none was wrong. 0 Blocker, 1 Important, 10 Minor.
Folded:
- **I1:** `_remove_preamble` refuses a preamble that holds a section marker (a section block needs no heading), with a
  fifth Task 1 test. The controller measured, on `a960463a26` with only the backlog-py hunks applied: RED `5 failed,
  158 passed`; GREEN `163 passed`; ruff clean. The folded backlog-py diff is `proto_13962_backlogpy_folded.patch`
  (plans dir); `proto_13962.patch` predates the folds. The docstring wording in Task 1 Step 3(a) was changed after that
  measurement (text only).
- **M1:** Task 0 Step 4 defines `<answer>` (never written literally) and drops the follow-up clause when none is filed;
  Task 4 Step 7's placeholder check covers only this PR's added lines.
- **M2:** `scan_refs_13962.py` names the merge base with `origin/dev` in the table header, not HEAD.
- **M3:** the renumber test pins both top-level cases.
- **M4:** `reparent_13962.py` writes a singular note for 410's one subtask.
- **M5:** `--ref`'s help text says it is an edit flag.
- **M6:** the plan copy replaces `task-15419` as well as `TASK-15419`.
- **M7:** the PR body says AC 4 is met for every file backlog-py can edit.
- **M8:** the record names the generator's `TOPIC_EXCEPTIONS` among the places the old ids stay.
- **M9:** #2738 is in the overlap table.
- **M10:** the close-out retitles TASK-13962 so it no longer promises AC 2.
