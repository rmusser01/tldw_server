# Dev integration verification

Fresh dev59049e094e merged at f4e9f954d7 after frozen cycle4 checkpoint b2595a4edb. Zero dev-only commits at this checkpoint. The13 imported files concern macOS sandbox drills/docs/tests.

Activated project .venv; ran pytest for sandbox/test_vz_linux_runner.py, sandbox/test_vz_linux_workspace_host_gated.py and macos-vz-helper Tests/test_failure_drill.py, test_failure_orchestration.py, test_failure_workflow_paths.py. Initial170pass/1host-gated skip/2fail because the restricted shell denied ps. Re-ran exactly the two test_logged_stops_process_tree_before_unwinding cases with process-inspection permission:2pass. Existing pytest cleanup warnings are retained. This covers172 passing cases and one explicit host-gated skip, not a new application UAT pass.

Bandit is not a gate for importing already-reviewed upstream files unchanged; each new Python repair will run its touched-scope Bandit gate.
