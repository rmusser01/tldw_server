---
id: TASK-12158
title: Plan tldw-frontend static PyPI bundle
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-06 03:50'
labels:
  - packaging
  - pypi
  - webui
  - frontend
dependencies: []
documentation:
  - Docs/Product/WebUI/TLDW_Frontend_Static_PyPI_Bundle_PRD.md
  - Docs/ADR/029-tldw-frontend-static-pypi-bundle.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the PRD and ADR for bundling a clean static tldw-frontend export inside the tldw-server PyPI package, and seed implementation tracking without changing runtime code in this planning slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PRD documents scope, requirements, non-goals, risks, release gates, and follow-up implementation work for tldw-frontend static PyPI bundling.
- [x] #2 ADR records the durable release-boundary decision for allowing static WebUI assets in the backend wheel while forbidding source/build/cache/server artifacts.
- [x] #3 Implementation follow-up tasks are identified for static export mode, FastAPI static serving, package content guards, and install smoke validation.
- [x] #4 Planning docs clearly exclude admin-ui from this first slice.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created the PRD and ADR for the tldw-frontend-only static PyPI bundle planning slice. Updated the ADR index so ADR-029 is discoverable. No runtime implementation files were changed; admin-ui remains explicitly out of scope for this first slice. Verification: git diff --check passed for the PRD, ADR, and ADR index. Bandit skipped because this is docs/task-only planning work.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created `Docs/Product/WebUI/TLDW_Frontend_Static_PyPI_Bundle_PRD.md` and `Docs/ADR/029-tldw-frontend-static-pypi-bundle.md`, and updated `Docs/ADR/README.md` with ADR-029. The PRD scopes a guarded static `apps/tldw-frontend` export served at `/ui` from the PyPI-installed backend, lists required follow-up implementation slices, and excludes `admin-ui`. The ADR records the release-boundary change: clean static WebUI assets are allowed in `tldw-server` wheels, while source, `.next`, `node_modules`, standalone traces, caches, models, databases, and admin UI artifacts remain forbidden. Verification: doc whitespace check passed; Bandit skipped as not applicable to docs-only work.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Backlog task references the PRD and ADR paths.
- [x] #2 Docs are written using existing repository conventions.
- [x] #3 No runtime implementation files are changed in this planning slice.
- [x] #4 Verification records doc/task creation and known skips.
<!-- DOD:END -->
