# Native CodeGraph Context Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Improve `codegraph.context` so selected context is ordered by task relevance and nearby graph relationships while preserving the existing response shape and safety bounds.

**Architecture:** Keep ranking local to the CodeGraph context flow. Add a deterministic ranking helper under `app/core/CodeGraph/context.py`, call it from the MCP module before relationship collection and snippet assembly, and leave repository search, workspace resolution, and public tool schemas unchanged.

**Tech Stack:** Python 3.11, existing CodeGraph repository/model objects, Unified MCP module tests, pytest, Ruff, Bandit.

---

### Task 1: Add Context Ranking Helper

**Files:**
- Modify: `tldw_Server_API/app/core/CodeGraph/context.py`
- Test: `tldw_Server_API/tests/CodeGraph/test_codegraph_context.py`

- [x] **Step 1: Write RED ranking tests**

Add tests proving a node with direct task-token matches is ordered before a weaker search result, and a node connected by a selected relationship is boosted ahead of an unrelated node when `max_files` would otherwise exclude it.

- [x] **Step 2: Run focused context tests and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_context.py -q
```

Expected: the new tests fail because the helper does not exist or preserves input order.

- [x] **Step 3: Implement minimal ranking helper**

Add a small exported helper, `rank_context_nodes(task, nodes, relationships)`, that:

- tokenizes the task into lowercase alphanumeric terms;
- scores matches in `name`, `qualified_name`, and `file_path`;
- adds a relationship boost for nodes that appear in selected relationships;
- preserves deterministic tie-breaking by original order, file path, line number, and qualified name.

- [x] **Step 4: Run focused context tests and verify GREEN**

Run the same context test command. Expected: all context tests pass.

### Task 2: Wire Ranking Into MCP Context

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py`

- [x] **Step 1: Write RED MCP context test**

Add a regression showing `codegraph.context` chooses a related caller/callee pair within tight `max_nodes` or `max_files` bounds instead of returning an unrelated search result first.

- [x] **Step 2: Run the focused MCP test and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py::test_codegraph_context_ranks_task_and_relationship_relevance -q
```

Expected: failure because MCP context does not yet apply the ranking helper.

- [x] **Step 3: Apply helper in `_build_context`**

Fetch a small over-selection of candidates, collect one-hop relationships, rank nodes, clamp to `max_nodes`, then rebuild the relationship neighborhood for the ranked subset. Preserve existing missing-index, truncation, and response keys.

- [x] **Step 4: Run focused MCP test and verify GREEN**

Run the same focused MCP test. Expected: pass.

### Task 3: Verification And Closeout

**Files:**
- Modify: `backlog/tasks/task-78 - Improve-CodeGraph-context-ranking.md`

- [x] **Step 1: Run focused regression suite**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/CodeGraph tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py -q
```

- [x] **Step 2: Run style/security checks on touched scope**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check tldw_Server_API/app/core/CodeGraph/context.py tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py tldw_Server_API/tests/CodeGraph/test_codegraph_context.py tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/CodeGraph/context.py tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py -f json -o /tmp/bandit_codegraph_context_ranking.json
git diff --check
```

- [x] **Step 3: Update Backlog task**

Check all completed acceptance criteria, record verification, and add the final summary.

- [x] **Step 4: Commit and open PR**

```bash
git add Docs/superpowers/plans/2026-05-05-native-codegraph-context-ranking-implementation-plan.md \
  'backlog/tasks/task-78 - Improve-CodeGraph-context-ranking.md' \
  tldw_Server_API/app/core/CodeGraph/context.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py \
  tldw_Server_API/tests/CodeGraph/test_codegraph_context.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py
git commit -m "feat: rank codegraph context by relevance"
git push -u origin codex/codegraph-context-ranking
gh pr create --base dev --head codex/codegraph-context-ranking
```
