---
id: TASK-12054
title: Address CATS fuzzing code review findings
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-27 20:48'
labels:
  - testing
  - security
  - api
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2538'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the blocking and important findings from the CATS fuzzing PR review: harden runtime config/credential isolation, keep OpenAPI export on the selected Python interpreter, reject non-loopback existing-server URLs for local-only blocks, and return clean CLI usage errors for invalid no-start-server combinations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CATS child/server environments use generated inert config and do not inherit local credentials or provider endpoint overrides.
- [x] #2 OpenAPI export uses the current Python interpreter and invalid runtime CLI combinations fail with usage errors before subprocess work.
- [x] #3 Built-in local-only runtime blocks reject non-loopback existing-server URLs.
- [x] #4 Live CATS contract block exports OpenAPI and reports failure_class=ok.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Generated an inert runtime config/env for CATS subprocesses, forced TLDW_CONFIG_FILE/TLDW_CONFIG_PATH/TLDW_CONFIG_DIR, set PYTHON_DOTENV_DISABLED=true, blanked known secrets plus provider endpoint/base-url overrides, used sys.executable for OpenAPI export, rejected non-loopback --server-url values for local-only runtime blocks, and moved invalid --no-start-server usage to argparse errors. Added route_key=setup to the minimal optional setup router spec so generated CATS configs can disable setup API routes before OpenAPI export.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed CATS fuzzing review findings and validated the harness. Focused tests pass, production-path Bandit reports zero findings, and a live contract block now exports OpenAPI and reports failure_class=ok with setup UI and setup API routes disabled by policy. Verification: python -m pytest tldw_Server_API/tests/VectorStores/test_vector_stores_openapi_examples.py tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_manifest.py tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_env.py tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_cats_cli.py tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_summary.py tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_runner.py tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_cli.py tldw_Server_API/tests/Services/test_router_groups_contract.py::test_minimal_optional_setup_router_participates_in_route_policy -q (60 passed); python -m Helper_Scripts.cats_fuzz --block contract --output /tmp/tldw-cats-review-fix-contract-v4 (exit 0, failure_class ok); python -m bandit -r Helper_Scripts/cats_fuzz tldw_Server_API/app/api/v1/router_groups/minimal.py -f json -o /tmp/bandit_cats_fuzz_review_fix_prod_final.json (0 findings); git diff --check (clean).
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
