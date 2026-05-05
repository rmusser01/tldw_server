---
id: TASK-70
title: Add CodeGraph Jobs-backed indexing mode
status: Done
assignee:
  - '@Codex'
created_date: '2026-05-05 14:15'
updated_date: '2026-05-05 14:24'
labels:
  - codegraph
  - mcp
  - jobs
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the deferred Jobs-backed execution path for native CodeGraph indexing so large workspaces can enqueue index/sync work instead of relying only on bounded foreground MCP calls. Scope this first Jobs slice to job payload helpers, MCP job-mode enqueue responses, a CodeGraph Jobs worker entrypoint that runs the existing indexer against trusted workspace/index paths, and focused tests. Exclude file watching, Scheduler integration, automatic in-process startup, and changing existing foreground behavior unless explicitly covered by tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 codegraph.index and codegraph.sync accept a job/background mode that creates a core Jobs entry with domain, queue, job_type, owner, and JSON-safe workspace/index payload
- [x] #2 Foreground mode keeps the current synchronous behavior and existing CodeGraph tests continue to pass
- [x] #3 A CodeGraph Jobs worker handler validates job payloads, runs index or sync through the existing CodeGraphIndexer, returns serialized result data, and rejects unsupported job types or unsafe paths
- [x] #4 Focused tests cover job enqueueing, worker success and validation failures, MCP job-mode responses, and existing foreground regressions
- [x] #5 Ruff, Bandit on touched production scope, focused pytest, and git diff --check pass before PR
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created implementation plan at Docs/superpowers/plans/2026-05-05-native-codegraph-jobs-indexing-implementation-plan.md. Scope is limited to Jobs payload helpers, MCP job/background enqueue mode, and a CodeGraph Jobs worker entrypoint; no file watching, Scheduler integration, or automatic worker startup in this slice.

Implemented CodeGraph Jobs payload helpers, a non-retryable validation worker handler, and MCP job/background mode enqueueing for codegraph.index and codegraph.sync. The worker validates job_type, operation, payload shape, workspace key, settings, language filters, max_files, and index_db_path containment before opening the CodeGraph repository. Foreground mode remains unchanged and still offloads blocking repository work through asyncio.to_thread.

Verification passed:

- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_jobs.py tldw_Server_API/tests/CodeGraph/test_codegraph_jobs_worker.py tldw_Server_API/tests/CodeGraph/test_codegraph_indexer.py tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py -q (58 passed)
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check tldw_Server_API/app/core/CodeGraph/jobs.py tldw_Server_API/app/core/CodeGraph/jobs_worker.py tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py tldw_Server_API/tests/CodeGraph/test_codegraph_jobs.py tldw_Server_API/tests/CodeGraph/test_codegraph_jobs_worker.py tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/CodeGraph/jobs.py tldw_Server_API/app/core/CodeGraph/jobs_worker.py tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py -f json -o /tmp/bandit_codegraph_jobs_indexing.json (0 findings)
- git diff --check
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a Jobs-backed execution path for native CodeGraph index and sync work while preserving bounded foreground behavior. codegraph.index and codegraph.sync now accept mode="job" or mode="background" to enqueue core Jobs rows with a JSON-safe workspace/index payload and owner id, and a new CodeGraph Jobs worker entrypoint validates and executes those jobs through the existing CodeGraphIndexer. No file watching, Scheduler integration, or automatic worker startup was added in this slice.

No known blockers. Automatic worker deployment/startup remains intentionally out of scope for this task.
<!-- SECTION:FINAL_SUMMARY:END -->
