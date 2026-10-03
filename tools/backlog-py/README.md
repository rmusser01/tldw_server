# backlog-py

Python compatibility clone of Backlog.md.

## Status: standard task editor

Since 2026-10-03 backlog-py is this repository's task editor, replacing the Node
Backlog.md CLI and MCP server for every change to `backlog/` task files
([ADR-059](../../Docs/ADR/059-backlog-py-task-editor-cutover.md), TASK-13440).
The two editors write different notes markers and corrupt each other's files, so
do not use the Node `backlog` CLI or MCP to create or edit tasks.

Run it from the repository root:

```bash
PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task list --plain
# or, once: pip install -e tools/backlog-py, then:
backlog-py --cwd . task list --plain
```

The console script is `backlog-py`. Do not install or alias it as `backlog`.
See the Backlog.md section of the root `AGENTS.md` for the agent workflow,
including how to pick a task id that does not collide with other branches.

## Canonical task format

backlog-py stores implementation notes in `SECTION:IMPLEMENTATION_NOTES` and
keeps one BEGIN/END pair per section. `task normalize [--check] [path...]`
rewrites files into that form: a Node `SECTION:NOTES` block is renamed, NOTES
wrapping IMPLEMENTATION_NOTES unwraps with the inner text first, a repeated
section merges into the first one, and orphaned or duplicated END markers go.
No text is dropped and frontmatter bytes are kept. `--check` only lists the
files that would change and exits 1 if there are any. Every `task edit`
normalizes the file it edits first.

`tldw_Server_API/tests/CI/test_backlog_task_format_ratchet.py`, run by the
`backend-required` gate, checks that every task file parses and that the count
of files still needing normalization never rises.

## Oracle Fixtures

Compatibility fixtures are pinned to explicit upstream Backlog.md release
metadata. The initial oracle manifest records `backlog.md@1.44.0`, source kind,
source reference, package metadata hash, generation date, and the agent-critical
commands/resources/tools that future golden fixtures must cover.

Upstream Backlog.md and its Node/Bun toolchain are only allowed in fixture
generation or refresh jobs. Normal `backlog-py` runtime, regular tests, and
future repository cutover paths must remain Node/Bun-free.

## Agent Cutover Gate

Agent-critical parity is tracked in `docs/agent-critical-parity.md`. The matrix
enumerates every CLI command, MCP resource, and pure MCP helper that blocks the
first local-file agent cutover candidate, plus the browser, interactive,
completion, hook, and git behaviors that are explicitly deferred.

The gate is enforced by `tests/test_agent_critical_matrix.py`: every
`golden-required` inventory item must have a matching oracle manifest fixture,
and the matrix document must mention every implemented or deferred item. Run it
with the rest of the suite:

```bash
source .venv/bin/activate
python -m pytest tools/backlog-py/tests -v
```

The root `pyproject.toml` puts `tools/backlog-py/src` on the pytest path, and
`backend-required` runs this suite.

Mutation smoke tests in this package run against temporary copies of the
fixture repositories, never the live repository backlog.

Browser and interactive behavior is tracked separately from the first agent
cutover candidate:

- `docs/browser-parity.md` records browser requirements such as drag-and-drop,
  service mode, rich Markdown editing, and mobile behavior.
- `docs/interactive-deferrals.md` records CLI/TUI, `onStatusChange`,
  auto-commit, hook bypass, and remote-operation deferrals.
