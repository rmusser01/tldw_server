# Task 1 implementation report

Status: implementation complete; awaiting controller independent review. TASK-13161 remains In Progress.

## Scope delivered

- Added strict, frozen fingerprint and snapshot metadata models plus path-free request/receipt models with bounded IDs, strings, counts, UTC timestamps, lowercase SHA-256 values, and forbidden extras.
- Added fail-closed fingerprint comparison and stable content hashing for the model, executable, optional projector, canonical effective options, and canonical adapter descriptors. File identity is checked before and after hashing.
- Added a process-owned filesystem store with `flock`, owner-only directories/files, no-follow reads, exclusive temporary creation, chunked copy/hash, staged identity checks, fsync boundaries, binary-first/manifest-last publication, valid-manifest catalog recovery, hash-verified restore staging, monotonic-sequence retention, path-free deletion, and durable receipts.
- Added persisted profile defaults (`snapshots_enabled=False`, `snapshot_retention=10`, range 1..1000) to runtime and admin request/response models.
- ADR required: yes. Existing ADR: `Docs/ADR/043-managed-llamacpp-manual-slot-snapshots.md`; no new ADR was needed.

## TDD evidence

- RED: `python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_snapshot_compatibility.py -q` exited 2 during collection because `llamacpp_snapshot_compatibility`/`llamacpp_snapshot_models` did not exist.
- GREEN after strict models/comparison: the required compatibility test passed (`1 passed`).
- RED store: `python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_snapshot_store.py -q` exited 2 during collection because `llamacpp_snapshot_store` did not exist.
- GREEN store: initial store run passed (`13 passed`); interruption-boundary cases were then added and included in the final combined run.
- RED profile compatibility: profile-store run failed with `AttributeError: 'LlamaCppProfile' object has no attribute 'snapshots_enabled'` (`1 failed, 10 passed`).
- Final targeted verification: `python -m pytest ...test_llamacpp_snapshot_compatibility.py ...test_llamacpp_snapshot_store.py ...test_llamacpp_profile_store.py -q` -> `34 passed, 7 warnings in 0.83s`.

The tests use real temporary files/directories and cover known-hash publication, corrupted restore cleanup, traversal identifiers, symlink roots/sources, oversized/malformed manifests, orphan binaries, disk-full writes, interrupted binary/manifest publication, preservation of previous snapshots, private permissions, ownership fencing, durable receipts, monotonic pruning under clock rollback, partial prune failures, individual compatibility mismatches, canonical configuration hashing, and old-profile/default round trips.

## Static and security evidence

- `ruff format --check` on all eight touched source/test files: `8 files already formatted`.
- `ruff check --ignore UP037` on all eight touched source/test files: `All checks passed`. `UP037` is an unrelated pre-existing quoted forward annotation in the touched admin schema.
- `python -m compileall -q` on the five touched production modules: exit 0.
- Bandit 1.9.4 via the controller-provided installed module path, scanning the five touched production modules: exit 0, no findings.
- `git diff --check`: exit 0.

## Review notes and concerns

- Blocking filesystem/hash methods are deliberately synchronous storage primitives; Stage 2 supervisor operations must call them with `asyncio.to_thread` so the event loop is not blocked.
- Launch working-directory quarantine cleanup requires a proven-dead child and belongs to the Stage 2 lifecycle owner; this store only creates integrity-verified restore staging files and never attempts lifecycle cleanup itself.
- Hypothesis is declared by the project but unavailable in the mandated existing venv, so the delivered boundary matrix uses parametrized tests rather than Hypothesis-generated cases. No dependency was installed, per instruction.
- Profile create/update API propagation and runtime/profile deletion guards remain Stage 2 concerns; this stage supplies the persisted fields and storage primitives only.
- Existing controller edits to the implementation-plan heading and TASK-13161 tracking file were preserved and are intentionally excluded from this stage's code commit.
