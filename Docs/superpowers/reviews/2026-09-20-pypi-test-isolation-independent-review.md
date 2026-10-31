# Independent review: PyPI 0.1.42 test collection isolation

Reviewed the current uncommitted test changes against `cd2dbc792b8888555abac5c5c9eafa7a43d9b0e4` in the recovery worktree. Scope: nine modified test/fixture files and two new Infrastructure regression files. No actionable findings.

- Prompt Studio collection no longer imports/reloads shared `app.main` or sets its four profile/auth environment variables. All former global-app consumers now request the function-scoped fixture.
- The fixture obtains the actual production app through `reload_app_main()`, preserving its routes, middleware, and lifespan. Existing TestClient context-manager usage remains intact; no assertions, routes, tests, or lifecycle behavior were replaced with a reduced app.
- `app_main_isolated()` restores the previous module object and package binding in `finally`, including setup failures. Client/dependency fixtures tear down before the app fixture; monkeypatch restores environment afterward. The original shared app object is not reloaded in place.
- The three route-policy fixes remove only collection-time `ROUTES_DISABLE` mutation. Caller route policy remains untouched. The new regression executes each target module and verifies an existing policy survives, with `patch.dict` restoring all environment mutations.
- The Prompt Studio regression executes its conftest and checks environment, app identity, and route identity preservation. Its minimal-profile sentinel exercises the old collection-time reload trigger.

Independent checks: all 11 changed/new Python test files parsed successfully; Ruff baseline/current comparison introduced no diagnostics; `git diff --check` passed. Used the source checkout's activated virtual environment because the recovery worktree has no `.venv`. No pytest processes, repository edits, or commits were performed.

Runtime evidence supplied by the implementer: 161 owning Prompt Studio tests and the independent three-test Prompt Studio/artifact run pass, and the original collection-related artifact 404 is fixed. These were not rerun during this review. Owning route-policy validation remains the implementer's active check. This review does not establish that the release gate timeout is resolved and does not review packaging or workflows.
