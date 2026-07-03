---
id: TASK-12123
title: Auto-publish backend PyPI releases from main version bumps
status: Done
labels:
- packaging
- pypi
- ci
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update the backend/API PyPI publish workflow so pushes to main publish tldw-server only when pyproject.toml contains a version that is not already present on PyPI. Keep manual workflow_dispatch publishing available.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pushes to main can trigger backend PyPI publishing without manual dispatch.
- [x] #2 Workflow detects the pyproject.toml project.version and skips publishing when that version already exists on PyPI.
- [x] #3 Workflow fails closed on PyPI lookup errors instead of publishing blindly.
- [x] #4 Manual workflow_dispatch target=testpypi and target=pypi behavior remains available.
- [x] #5 Validation and final summary are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- 2026-07-03: Design review adjustments before implementation:
  - Keep manual TestPyPI/PyPI workflow_dispatch behavior.
  - Add push-to-main publishing only for versions missing from PyPI.
  - Fail closed when the PyPI version lookup errors instead of assuming publish is safe.
  - Serialize publish workflow runs per ref to avoid duplicate publish races.
- 2026-07-03: Updated `.github/workflows/publish-pypi.yml` to add a `push` trigger on `main` scoped to `pyproject.toml`, a `detect-version` job that reads `pyproject.toml` and queries the PyPI JSON API, and publish gating so automatic production publishing only runs when the pushed version is absent from PyPI. Manual TestPyPI/PyPI dispatch paths still build and publish via the existing inputs.
- 2026-07-03: Verification:
  - `python -c 'import yaml; yaml.safe_load(open(".github/workflows/publish-pypi.yml")); print("yaml ok")'` passed.
  - Manual-dispatch local detect script simulation produced `version=0.1.34`, `published=false`, `should_publish=false`, `reason=manual_dispatch`.
  - Push-event local detect script simulation against PyPI produced `version=0.1.34`, `published=false`, `should_publish=true`, `reason=missing_from_pypi`.
  - `git diff --check` passed.
  - Bandit skipped because only GitHub Actions YAML and Backlog markdown changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added automatic backend/API PyPI publishing for new versions pushed to `main`. The workflow now detects the current `pyproject.toml` package version, queries PyPI, skips already-published versions, fails closed on lookup errors, and preserves manual `workflow_dispatch` publishing to TestPyPI and PyPI. Validation covered YAML parsing, local manual/push detection simulations, and whitespace checks.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
