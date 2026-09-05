# Final combined snapshot fix report

Status: implemented and targeted verification passed; awaiting controller independent re-review. Review base: `786b46e9ef`. TASK-13162 remains under controller tracking; TASK-13163 live acceptance is still open.

ADR required: yes. ADR path: `Docs/ADR/043-managed-llamacpp-manual-slot-snapshots.md`. Reason: these fixes enforce the existing native-control, ownership and fail-closed storage boundaries; no new ADR or feature redesign was needed. Read TASK-13162, the approved design, ADR-043, repository guidance, prior task reports and the testing-evidence lessons. Used TDD and verification-before-completion skills. No subagents, push or merge.

## Changes

- `llamacpp_process_runner.py`: snapshot-enabled launches require a numeric loopback IP before port selection, private working-directory creation or spawn. Both IPv4 and IPv6 loopback literals work; wildcard/LAN/public binds and DNS names fail with an instruction to use `127.0.0.1` or `::1`. Snapshot-disabled bind and Windows lifecycle behavior is preserved.
- `llamacpp_supervisor_service.py`: a global cleanup lock serializes disk cleanup passes across profiles. Each pass iterates a tuple snapshot and removes only a successfully cleaned, proven-dead entry from the current ledger. Synchronous launch registration may append while cleanup awaits disk work, and no stale list replacement can erase it. Failed termination retains the real storage ownership fence through async shutdown and synchronous fallback.
- `llamacpp_snapshot_store.py`: `fcntl` is optional at import time, but absent POSIX locking raises `SnapshotStorageUnavailableError` before storage initialization creates any directory. Ordinary runner/API imports remain usable on platforms without `fcntl`; Windows snapshot support is not claimed.
- `Docs/Guides/llamacpp-manual-snapshots.md`: documents the numeric loopback requirement, rejected DNS aliases, unsupported platform storage, and trusted-local-host limitation. Native routes are still available to local callers even with the empty production build gate; never forward/proxy them to users.

## Regression coverage and RED evidence

All Python commands used the activated existing environment:

```sh
source /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_process_runner.py tldw_Server_API/tests/LLM_Local/test_llamacpp_supervisor_service.py -q -k 'snapshot_launch or without_fcntl or overlapping_cleanup'
```

Initial RED: **10 failed, 9 passed, 58 deselected, 6 warnings**. Log: `/private/tmp/snapshot-final-red.log`. The eight bind cases initially reached the unavailable-store error rather than loopback validation, so the test was strengthened with a real SnapshotStore and fake child before production edits:

```sh
python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_process_runner.py -q -k rejects_non_loopback
```

Strengthened bind RED: **8 failed, 29 deselected, 6 warnings**; all eight launched the fake child instead of raising `ServerError`. Log: `/private/tmp/snapshot-final-bind-red.log`.

Changed tests:

- `test_snapshot_launch_generations_and_private_working_path` now covers `127.0.0.1`, `127.0.0.2`, `::1` and `[::1]` with real storage, generations and private argument behavior.
- `test_snapshot_launch_rejects_non_loopback_before_spawn` covers `0.0.0.0`, `::`, `[::]`, `192.168.1.10`, `8.8.8.8`, `2001:db8::1`, `localhost` and `example.com`. Requires no child spawn.
- `test_runner_and_api_import_without_fcntl` starts a fresh interpreter with `sys.modules['fcntl'] = None`, imports both ordinary runner and actual API endpoint, then requires a storage capability failure with no directory created. RED was the transitive import's `ModuleNotFoundError`.
- `test_overlapping_cleanup_preserves_new_live_child_and_owner_on_failed_stop` deterministically interleaves two cleanup passes with registration through actual `start_profile`. Its child refuses stop. After both shutdown and `cleanup_sync`, a second real SnapshotStore must fail to acquire the same root, and the live child remains tracked. RED was `DID NOT RAISE SnapshotStoreError`, demonstrating actual premature ownership release.
- The existing ordinary Windows two-profile lifecycle test now also runs with a snapshot-disabled wildcard bind and verifies the child command retains that address.

The cleanup test initially reproduced registration by direct ledger append, then was strengthened to use actual `start_profile`. A temporary pytest plugin loaded only the baseline cleanup method from `git show 786b46e9ef:.../llamacpp_supervisor_service.py` into the test process to verify the stronger test still detects the original bug:

```sh
PYTHONPATH=/private/tmp python -m pytest -p snapshot_cleanup_old_regression tldw_Server_API/tests/LLM_Local/test_llamacpp_supervisor_service.py -q -k overlapping_cleanup --timeout=15
```

Mutation RED: **1 failed, 39 deselected, 6 warnings**, again at the real ownership-lock assertion. Log: `/private/tmp/snapshot-final-race-red.log`. No production source was reverted for this mutation check. The first temporary plugin used copied globals, preventing the disk barrier patch from reaching the method; corrected it to use module globals before the recorded mutation result.

## GREEN and static/security evidence

Focused runner/supervisor GREEN before the final test strengthening: **77 passed, 7 warnings** (`/private/tmp/snapshot-final-green.log`). Final surrounding regression command:

```sh
PYTHONPATH=/Users/macbook-dev/.cache/uv/archive-v0/iTu2xL1Afi-adAWUqtN_p:/Users/macbook-dev/.cache/uv/archive-v0/i-VcGRId6asg3bZg python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_snapshot_operations.py tldw_Server_API/tests/LLM_Local/test_llamacpp_snapshot_api.py tldw_Server_API/tests/LLM_Local/test_llamacpp_snapshot_store.py tldw_Server_API/tests/LLM_Local/test_llamacpp_snapshot_compatibility.py tldw_Server_API/tests/LLM_Local/test_llamacpp_process_runner.py tldw_Server_API/tests/LLM_Local/test_llamacpp_supervisor_service.py tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_api.py tldw_Server_API/tests/LLM_Local/test_llamacpp_admin_config_api.py tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_reconciler.py -q
```

Result: **213 passed, 6 warnings in 5.29s**, `/private/tmp/snapshot-final-combined.log`. Cached Hypothesis 6.138.2 and sortedcontainers were used without installation. Two initial collection attempts used an unsuitable cached Hypothesis path (missing sortedcontainers, then missing native extension); the final command above resolves both. Existing bootstrap/dependency warning qualification from the earlier reports applies; no full suite or live inference was run.

For the following commands, `production` means the three changed modules and `tests` means the two changed test modules listed above (all passed as explicit paths):

- `ruff format --check <production> <tests>` → **5 files already formatted**.
- `ruff check --per-file-ignores 'llamacpp_process_runner.py:BLE001' <production> <tests>` → **All checks passed!** The existing runner-only broad-exception exemption is unchanged; no new blanket suppressions.
- `python -m compileall -q <production>` → exit **0**.
- `PYTHONPATH=/Users/macbook-dev/.cache/uv/archive-v0/BD4794DBIFvoMXJ8bxCnA:/Users/macbook-dev/.cache/uv/archive-v0/wNOip7lQRcXR49p3z9hCF python -m bandit <production> -f json -o /private/tmp/snapshot-final-bandit.json` → exit **0**, Bandit **1.9.4**, **no findings** and no skipped rules.
- `git diff --check` → exit **0**.

Self-review confirmed only the three requested production boundaries changed. Controller-owned backlog/lesson edits and the untracked node_modules symlink and `.llamacpp.lock` are excluded from the scoped commit and preserved.

## Limitations

`TESTED_TEXT_BUILD_SHA256` remains empty. No executable/model was available, no native runtime test was performed, and no live cache reuse, real browser-to-runtime flow, Chatbook state invariance or real Pause/Resume proof is claimed. The fresh-interpreter test simulates absent `fcntl` on macOS; existing fake-process Windows lifecycle tests are not a real Windows host run. Loopback blocks remote binds but does not authenticate untrusted local callers. The controller owns independent final re-review and task acceptance decisions.
