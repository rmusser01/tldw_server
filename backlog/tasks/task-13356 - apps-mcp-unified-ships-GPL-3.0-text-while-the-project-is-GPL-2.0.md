---
id: TASK-13356
title: apps/mcp-unified ships GPL-3.0 text while the project is GPL-2.0
status: To Do
assignee: []
created_date: '2026-09-23 15:02'
labels:
  - licensing
  - mcp
  - packaging
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`apps/mcp-unified/LICENSE` contains the full **GNU GPL version 3** text. The project is GPL-2.0 (CLAUDE.md, README), relicensed in `9a34da262a` ("also now licensed GPLv2.0 for the forseeable future"). The repo root `LICENSE` was then replaced by a licensing-boundary document in `da0ec87d7d` (TASK-12976), titled "# tldw_server Licensing".

So the standalone package's LICENSE matches neither the project licence nor the root file, and that file is what gets shipped in the published `mcp_unified` wheel and sdist.

`test_runtime_package_boundary.py::test_mcp_unified_package_license_file_is_local_to_project` asserts the two files are byte-identical and has been red since the root LICENSE changed. It went unnoticed because `app/core/MCP_unified/tests` runs in no CI job -- see TASK-13291, which this is a concrete instance of.

**Owner decision required.** Three shapes, and they are not equivalent:
1. Copy the new root licensing document into the package -- makes the test pass, but ships a document that points at repo structure rather than a licence.
2. Put the GPL-2.0 text in the package LICENSE and relax the test to assert the licence identity rather than byte-equality with a root file that is no longer a licence.
3. Decide the standalone package is intentionally GPL-3.0 and record why, then change the test's premise.

Do not resolve this by making the assertion pass. Which text a published artifact carries is a licensing question.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The licence text shipped in apps/mcp-unified is the one the owner intends, recorded explicitly
- [ ] #2 test_mcp_unified_package_license_file_is_local_to_project asserts the intended relationship rather than byte-equality with a non-licence file
- [ ] #3 The published wheel and sdist are checked to carry that text
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
