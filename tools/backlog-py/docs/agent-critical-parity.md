# Agent-Critical Parity Gate

This matrix is the cutover gate for agent workflows that currently depend on
Backlog.md. Every agent-critical CLI or MCP operation must be represented in the
built-in compatibility inventory and, when implemented, in the pinned oracle
manifest. Deferred capabilities are explicit blockers for later milestones and
do not block the first local-file agent cutover candidate.

## Implemented Golden Requirements

| Inventory item | Status | Expected command, resource, or tool | Fixture |
| --- | --- | --- | --- |
| cli:help | implemented | backlog --help | cli:help |
| cli:task-list-plain | implemented | backlog task list --plain | cli:task-list-plain |
| cli:task-view-plain | implemented | backlog task <id> --plain | cli:task-view-plain |
| cli:search-plain | implemented | backlog search <query> --plain | cli:search-plain |
| cli:board | implemented | backlog board | cli:board |
| cli:config-list | implemented | backlog config list | cli:config-list |
| cli:task-create | implemented | backlog task create <title> --status <status> --plain | cli:task-create |
| cli:task-edit | implemented | backlog task edit <id> --append-notes <text> --plain | cli:task-edit |
| cli:doc-list | implemented | backlog doc list | cli:doc-list |
| cli:doc-view | implemented | backlog doc view <path-or-id> | cli:doc-view |
| cli:doc-create | implemented | backlog doc create <path> --title <title> --content <body> | cli:doc-create |
| cli:doc-update | implemented | backlog doc update <path-or-id> --title <title> | cli:doc-update |
| cli:milestone-list | implemented | backlog milestone list | cli:milestone-list |
| cli:milestone-add | implemented | backlog milestone add <name> | cli:milestone-add |
| cli:milestone-rename | implemented | backlog milestone rename <old> <new> | cli:milestone-rename |
| cli:milestone-remove | implemented | backlog milestone remove <name> | cli:milestone-remove |
| cli:milestone-archive | implemented | backlog milestone archive <name> | cli:milestone-archive |
| cli:config-dod-defaults-get | implemented | backlog config dod-defaults-get | cli:config-dod-defaults-get |
| cli:config-dod-defaults-upsert | implemented | backlog config dod-defaults-upsert [item...] | cli:config-dod-defaults-upsert |
| mcp:workflow-overview | implemented | backlog://workflow/overview | mcp:workflow-overview |
| mcp:task-workflow-alias | implemented | backlog://docs/task-workflow | mcp:task-workflow-alias |
| mcp:task-search | implemented | task_search(project, query, limit=10) | mcp:task-search |
| mcp:task-view | implemented | task_view(project, task_id) | mcp:task-view |
| mcp:task-create | implemented | task_create(project, **kwargs) | mcp:task-create |
| mcp:task-edit | implemented | task_edit(project, task_id, **kwargs) | mcp:task-edit |
| mcp:document-list | implemented | document_list(project, query=None, limit=100) | mcp:document-list |
| mcp:document-search | implemented | document_list(project, query=<query>, limit=100) | mcp:document-search |
| mcp:document-view | implemented | document_view(project, path_or_id) | mcp:document-view |
| mcp:document-create | implemented | document_create(project, **kwargs) | mcp:document-create |
| mcp:document-update | implemented | document_update(project, path_or_id, **kwargs) | mcp:document-update |
| mcp:milestone-list | implemented | milestone_list(project) | mcp:milestone-list |
| mcp:milestone-add | implemented | milestone_add(project, name, description='') | mcp:milestone-add |
| mcp:milestone-rename | implemented | milestone_rename(project, old_name, new_name, update_tasks=False) | mcp:milestone-rename |
| mcp:milestone-remove | implemented | milestone_remove(project, name, clear_tasks=False) | mcp:milestone-remove |
| mcp:milestone-archive | implemented | milestone_archive(project, name) | mcp:milestone-archive |
| mcp:definition-of-done-defaults-get | implemented | definition_of_done_defaults_get(project) | mcp:definition-of-done-defaults-get |
| mcp:definition-of-done-defaults-upsert | implemented | definition_of_done_defaults_upsert(project, items) | mcp:definition-of-done-defaults-upsert |

## Explicit Deferred Blockers

| Inventory item | Status | Expected behavior | Deferred reason |
| --- | --- | --- | --- |
| browser:kanban-drag-drop | deferred | backlog browser | Browser UI parity is tracked in the browser deferral milestone. |
| cli:interactive-board | deferred | backlog board interactive controls | Interactive terminal controls are deferred behind non-interactive agent workflows. |
| cli:rich-colored-output | deferred | ANSI-rich terminal rendering | Plain output is the cutover blocker; rich color is later polish. |
| cli:shell-completion-install | deferred | backlog completion install | Shell completion installation is not needed for agent runtime cutover. |
| core:on-status-change | deferred | onStatusChange hooks | Hook execution remains disabled until a separate safety review. |
| git:remote-operations | deferred | remote git operations | Remote git behavior is outside the first local-file compatibility gate. |
| git:auto-commit | deferred | autoCommit | Automatic commits are deferred to keep mutation review explicit. |
| git:hook-bypass | deferred | bypassGitHooks | Hook bypass remains unsupported for safety. |

## Validation Commands

Run the matrix test with:

```bash
source .venv/bin/activate
python -m pytest tools/backlog-py/tests/test_agent_critical_matrix.py -v
```

Run the full local cutover validation with:

```bash
source .venv/bin/activate
python -m pytest tools/backlog-py/tests -v
python -m bandit -r tools/backlog-py/src -f json -o /tmp/bandit_backlog_py.json
git diff --check
```

Mutation smoke tests must run only against a copied fixture or temporary
repository, never against the live project backlog.
