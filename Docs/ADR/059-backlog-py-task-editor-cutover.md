# ADR-059: backlog-py is the task editor

**Status:** Accepted
**Date:** 2026-10-03
**Backfilled from:** not backfilled
**Decision owner:** Repository owner (decision of 2026-10-03)
**Related task:** TASK-13440, TASK-13441
**Related spec/plan:** `Docs/superpowers/specs/2026-05-10-backlog-md-python-compatibility-clone-design.md`; `tools/backlog-py/README.md`

## Decision

Create and edit `backlog/` task files only with backlog-py (`tools/backlog-py`), never with the Node Backlog.md CLI or MCP server. [ADR-002](002-backlog-md-task-tracking.md) still requires a task before repo-changing work; this ADR changes only the tool used to write it.

## Context

Two editors wrote the same files, and each mangled the other's output.

- The Node CLI writes implementation notes as `SECTION:NOTES`; backlog-py writes `SECTION:IMPLEMENTATION_NOTES`.
- When the Node CLI edits a backlog-py task, even to change a label, it wraps the existing IMPLEMENTATION_NOTES block in a new NOTES block and duplicates the `SECTION:FINAL_SUMMARY` markers.
- When backlog-py appended notes to a Node task, it did not recognize NOTES. It added a second `## Implementation Notes` section at the end of the file instead.
- On 2026-10-03, 332 of the 3,689 files in `backlog/tasks/` held both kinds of notes block. 375 held those or duplicated FINAL_SUMMARY markers, and 2,308 differed from the single canonical form.
- The Node CLI picks the next task id from local branches only, so concurrent sessions took the same id (two different `TASK-12171` files exist).

backlog-py already had the safe-mutation core, path containment, and Definition of Done defaults. TASK-13440 added what agents were missing: labels, criteria on create, adding and removing criteria, retitling, replacing notes, and dependencies on create.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Keep the Node CLI and retire backlog-py | Its edits caused the corruption, its id selection still collides across sessions, and it needs a Node/Bun runtime in every agent environment. The backlog-py design spec targets a Node-free runtime. |
| Keep both editors and teach each to read the other's format | Coexistence caused the corruption. Two writers drift again with every upstream release. |
| Hand-edit task files | Hand edits bring back the inconsistencies that tooled edits prevent. AGENTS.md already forbids them without explicit approval. |
| Install backlog-py as `backlog` on PATH | A shadowed `backlog` silently changes what scripts and old instructions run. The separate `backlog-py` command makes the switch visible and rollback trivial. |

## Consequences

- AGENTS.md names backlog-py as the editor: `PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd <repo> ...`, or the `backlog-py` command after `pip install -e tools/backlog-py`. It forbids mutating task files with the Node CLI or MCP.
- The canonical task format has one `SECTION:IMPLEMENTATION_NOTES` block and one BEGIN/END pair per section. `backlog-py task normalize [--check]` rewrites older files into it without dropping text, and every backlog-py edit normalizes the file it touches.
- `tldw_Server_API/tests/CI/test_backlog_task_format_ratchet.py` runs in `backend-required`. It checks that every task file the PR adds or edits parses with backlog-py and is canonical. It reads only the files changed since the merge base with the PR's base commit, so a PR is never failed for a file it did not touch. A global count with a baseline was tried first and dropped in review: a backlog-only PR, which this gate does not run on, could raise the count and fail the next unrelated backend PR. The 2,308 non-canonical files present at cutover need no baseline; TASK-13441 normalizes them. backlog-py's own tests run in the same step, and changes under `tools/backlog-py/` trigger the gate.
- Accepted gaps:
  - backlog-py has no browser UI, no `--plan`, priority, assignee, or parent options, and no MCP server adapter (its MCP tools are pure functions). Agents use the CLI.
  - The ratchet runs only when `backend-required` sees backend changes. Backlog-only PRs are covered by the `backlog-task-format` pre-commit hook instead: it runs `backlog-py task normalize --check` on each changed `backlog/tasks/*.md` file, locally and in the (not required) `run-pre-commit` CI job, so a Node-format task file shows up red on the PR that adds it. Making that check required, or putting `backlog/tasks/**` in the backend gate, costs a full `backend-required` run on every task-only PR and was not chosen. Because the ratchet is scoped to each PR's own changes, a file that slips through this way does not fail later PRs; the next backlog-py edit of it normalizes it.
- Id collisions are not solved by the tool. backlog-py's default next id sees only the local checkout, so AGENTS.md has agents check `origin/dev` and every open PR branch before choosing an id.

## Follow-up

- TASK-13441: normalize every task file.
- Close the remaining parity gaps (`--plan`, priority, assignee, parent, MCP server adapter) when agents need them.
