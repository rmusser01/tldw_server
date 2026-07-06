---
id: TASK-12162
title: Cut 0.1.38 release with publishing workflow fixes
status: In Progress
priority: High
modified_files:
- .github/workflows/ci.yml
- CHANGELOG.md
- Docs/API-related/API_Tags_Index.md
- Docs/mkdocs.yml
- README.md
- pyproject.toml
- tldw_Server_API/app/main.py
- tldw_Server_API/tests/Web_Scraping/test_phase2_article_runtime_boundary.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a new 0.1.38 release that includes the GHCR-only Docker release workflow fix and PyPI PortAudio setup fix, then publish it through GitHub release flow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] Version metadata is bumped to 0.1.38.
- [ ] Changelog and README describe the corrective release.
- [ ] The release PR carries the publishing workflow fixes from dev to main.
- [ ] Relevant release/doc workflow tests pass locally.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Release is cut from the current `dev` release train into `main`.
- This release supersedes the failed 0.1.37 publish attempt by including workflow fixes for PyPI and GHCR-only Docker publishing.
- Local verification before pushing:
  - `git diff --check`
  - `python -m pytest -q tldw_Server_API/tests/CI/test_release_workflow_contracts.py`
  - `python -m pytest -q tldw_Server_API/tests/Docs/test_chatbook_openwebui_import_docs.py::test_openwebui_import_is_discoverable_from_api_docs`
  - `python Helper_Scripts/ci/check_shard_coverage.py --ci-file .github/workflows/ci.yml`
  - `python -m pytest -q tldw_Server_API/tests/Workflows/test_workflows_config_defaults.py`
  - `python Helper_Scripts/checks/guard_http_client_patching.py`
  - `python -m pytest -q tldw_Server_API/tests/Web_Scraping/test_phase2_article_runtime_boundary.py`
  - `python -m black --check tldw_Server_API/tests/Web_Scraping/test_phase2_article_runtime_boundary.py`
  - `pre-commit run --files tldw_Server_API/tests/Web_Scraping/test_phase2_article_runtime_boundary.py .github/workflows/ci.yml`
  - `python -m bandit -r tldw_Server_API/app/main.py -f json -o /tmp/bandit_release_0_1_38_main.json`
  - `python -m bandit -r tldw_Server_API/tests/Web_Scraping/test_phase2_article_runtime_boundary.py -f json -o /tmp/bandit_phase2_article_runtime_boundary.json` (pytest `assert` B101 findings only)
- CI follow-up: PR #2677 initially failed `Shard coverage guard` because `tldw_Server_API/tests/Workflows/test_workflows_config_defaults.py` had not been assigned to a full-suite shard. Added it to the existing `product-workflows-api` shard entries.
- CI follow-up: PR #2677 then failed push `run-pre-commit` because one existing Web Scraping test helper call had `monkeypatch` and `backend="httpx"` on the same line, matching the raw HTTP patch guard. Split the call over multiple lines without changing behavior.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
