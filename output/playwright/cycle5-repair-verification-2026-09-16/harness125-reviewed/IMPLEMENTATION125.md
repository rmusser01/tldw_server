# TASK13260.125: frozen private-harness repair

Parent-approved bounded repair; no product files changed. [Design](DESIGN125.md), [original draft snapshots](original-draft/), [original hashes](original-draft-manifest.json), [RED](RED125.md), [owned patch](owned.patch), [frozen hashes](owned-manifest.json).

## Result

- Preparation records reserve each canonical source root before helper writes. A different cell/run cannot reuse that binding; a failed preparation retains its record for inspection. Completed preparation is required before initialization/startup.
- The new private Python wrapper calls the existing frozen initializer function with non_interactive=True. Only normal return writes its attempt-specific mode0600 proof. SystemExit(0), KeyboardInterrupt and exceptions produce no proof. The wrapper checks the initializer module's frozen-root origin before calling its main function.
- Launcher success requires exit0, no forwarded interruption and matching normal-return proof tied to the exact prepared record. Backend/frontend require the matching completed init marker. Missing, legacy, stale and mismatched receipts fail before spawn/port use. Failed initialization preserves partial state/logs and is explicitly retryable with a new proof identity.
- Process receipts distinguish raw child code from launcher resultCode. Rapid retry controls also exposed same-millisecond log filename collisions; adding a UUID preserves each attempt's log without overwriting.
- Official PG holder/fixtures, PROTOCOL.md and pytest-holder.ini are byte-identical to the draft. Docs preserve four serial frozen archives, copied Python/Bun dependencies, strict private paths/ports and reused-dependency disclosure. The two actual copied editable entries are tldw_server (maps app+MCP) and backlog_py.

## Non-destructive verification

**23 Node tests pass,0 skipped;5 Python tests pass,0 skipped.** The final same Node test file replayed against the preserved original launcher produces **19 failures/4 passes/0 skips**. Original failures include actual raw-exit-zero markers, missing startup gates, source rebinding and lost failed-preparation state; receipt feature controls also fail because the original has no attempt request. Python's initial5 REDs were missing-wrapper contract tests, not extra native reproductions.

```sh
node --experimental-vm-modules --test .tmp/uat-next-matrix-20260916/matrix-launcher.test.mjs
MATRIX_TEST_SOURCE=.tmp/uat-next-matrix-20260916/repair125/original-draft/matrix-launcher.mjs node --experimental-vm-modules --test .tmp/uat-next-matrix-20260916/matrix-launcher.test.mjs
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest --confcutdir=.tmp/uat-next-matrix-20260916 -c .tmp/uat-next-matrix-20260916/pytest-holder.ini .tmp/uat-next-matrix-20260916/test_initialize_cell.py -q --tb=short
```

The second command intentionally exits1 against the original. [Exact results/log paths](../launcher-static-validation.json). Tests evaluate launcher action control flow only inside a VM: inspection, runtime helpers, environment, networking and spawning are replaced with fakes. Real filesystem operations are limited to new disposable test directories, cleaned by the test fixture. The Python tests replace import_module with a fake initializer and isolate pytest from repository conftest/autoload. No actual launcher CLI, real profile helper, app import, PG fixture, database, browser, server, provider, source archive or dependency copy action occurred. Real profiles/holders directories remain absent.

Static checks: Node syntax, wrapper/test/embedded-probe AST parse, Ruff and format checks pass. Wrapper Bandit has0 findings/errors; test Bandit has0 findings/errors with B101 excluded for pytest assertions. Initial B105 warnings on synthetic token literals were removed by generating fixture nonces, without suppressing findings. Node's experimental VM warning is expected and applies only to guard tests. The initial PG fake receipt control exposed /var versus /private/var fixture aliasing; its temporary root was canonicalized, then the original replay's positive PG ownership control passed.

## Remaining boundaries

The normal-return wrapper certifies the existing initializer's control flow, not successful future authentication/inference. Multi-user admin bootstrap remains the documented operator step. Full source archive manifests and runtime/module/port/storage receipts remain required later; the nine source fingerprints are not a complete source freeze. The documented serial controller owns scheduling; this repair does not add a parallel preparation framework. Failed preparation is preserved, never automatically deleted; another attempt needs a fresh archive/run after inspection.

Independent review by a different author is still required. Full-matrix execution remains gated on all targeted acceptance. The original self-review remains historical evidence of the unfixed draft. No task/tracker/git/runtime changes were made by this agent.
