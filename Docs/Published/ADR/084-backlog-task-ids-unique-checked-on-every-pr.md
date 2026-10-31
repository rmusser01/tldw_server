# ADR-084: Backlog task ids are unique, checked on every PR

**Status:** Accepted
**Date:** 2026-10-30
**Backfilled from:** not backfilled
**Decision owner:** Repository owner (@rmusser01); accepted with plan v2 on 2026-10-18, as ADR-075 (a number since taken)
**Related task:** TASK-13960
**Related spec/plan:** `Docs/Development/Backlog_Task_Id_Renumbering.md`; [ADR-059](059-backlog-py-task-editor-cutover.md)

## Decision

Every task file under `backlog/tasks`, `backlog/completed` and `backlog/archive/tasks` has its own id, and `backend-required`'s change-detection job, which runs on every PR, fails a PR that leaves two task files with one id.

## Context

- On 2026-10-30, 235 ids on `dev` were held by more than one file (528 files), so `task <id>`, dependencies and parent links were ambiguous.
- ADR-059 left id collisions to the allocation recipe in AGENTS.md ("Id collisions are not solved by the tool"). The recipe reads at most 100 files per open PR, and nothing checked its result.
- ADR-059 also declined a required format check on backlog-only PRs, because it would cost a full `backend-required` run on each one.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Keep the allocation recipe as the only guard | It is what produced the 235 duplicates. |
| Run the check in `backend-required`'s own steps | Those steps skip when no backend path changed, so backlog-only PRs, where most duplicates arrive, would pass. |
| Refuse only ids a PR adds, with a baseline of today's duplicates | A permanent exception list, and `task <id>` stays ambiguous for every id on it. |

## Consequences

- `Helper_Scripts/ci/check_backlog_task_ids.py` is the last step of the `changes` job. That job runs on every PR and installs nothing, so the check adds seconds, not a backend run. ADR-059's choice not to run the format check on backlog-only PRs stands.
- backlog-py's `task create` refuses an id held in any of the three directories and, without `--id`, allocates above all of them; `task renumber PATH --id NEW_ID` gives one file a new id.
- A branch that adds a task file with a taken id fails until that file is renumbered or dropped. TASK-13960 renumbered the 293 duplicates on `dev` and told each open PR that adds one.
- ADR-059 stays Accepted: its decision, backlog-py as the only task editor, is unchanged. Its note that id collisions are not solved by the tool describes backlog-py before TASK-13960.
