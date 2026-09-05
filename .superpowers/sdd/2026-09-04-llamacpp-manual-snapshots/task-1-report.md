# Task 1 implementation report

Status: implementation complete; awaiting controller independent review. TASK-13186 remains In Progress.

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
- Hypothesis was initially unavailable on the venv's default import path. The controller supplied an existing cached package path, so the round-one fix adds and runs a generated invalid-digest property test without installing or changing dependencies.
- Profile create/update API propagation and runtime/profile deletion guards remain Stage 2 concerns; this stage supplies the persisted fields and storage primitives only.
- Existing controller edits to the implementation-plan heading and TASK-13186 tracking file were preserved and are intentionally excluded from this stage's code commit.

## Round-one independent-review fixes

- RED evidence: the focused store/compatibility collection failed because the newly specified `SnapshotStorageUnavailableError` did not exist. The added regressions also named the previously unguarded ancestor-symlink, closed-owner, and pathname-replacement behaviors before production edits.
- Every existing path component is now checked with `lstat`; any symlink ancestor is rejected. `O_NOFOLLOW` is mandatory rather than silently falling back to zero, and directory fsync opens use both no-follow and `O_DIRECTORY` when supported. Root, staged-file, and restore-working ancestor tests prove an outside marker remains unchanged.
- Catalog reads now ignore only missing/malformed/incomplete entries. Directory enumeration and metadata `EIO` raise `SnapshotStorageUnavailableError`; receipt `EIO` is no longer translated to not-found.
- Stable file hashing now compares the open descriptor with the pathname both before and after reading, including device, inode, size, mtime, and ctime. A deterministic atomic pathname-replacement test raises `UnstableFingerprintError`.
- All public operations reject a closed store, including after another instance acquires the root lock.
- Malformed/oversized manifest fixtures now have private `0600` permissions, proving parsing/size recovery rather than permission rejection. Fault injection covers copy, file fsync, binary rename, binary-directory fsync, manifest write, manifest fsync, manifest rename, and manifest-directory fsync while preserving the prior committed entry.
- Final round-one test command used cached Hypothesis 6.138.2 via the supplied `PYTHONPATH`: `49 passed, 7 warnings in 1.54s`. The seven warnings are existing test-bootstrap/dependency warnings: Starlette/httpx deprecation, the documented unknown pytest `plugins` option, two legacy Pydantic class-config warnings, the conftest event-loop deprecation, Python `crypt` deprecation through passlib, and an existing Pydantic field-shadow warning.
- Round-one static/security evidence: Ruff format check and Ruff check passed for the four amended files; compileall passed for both amended production modules; Bandit 1.9.4 reported no findings; `git diff --check` passed.

## Round-two independent-review fix

- RED evidence: the deterministic ancestor-swap test failed against the round-one implementation because pathname `mkdir` followed an ancestor replaced by a symlink after validation.
- Directory traversal now opens every component relative to a held parent descriptor with mandatory `O_DIRECTORY|O_NOFOLLOW`. Directory creation uses `mkdir(..., dir_fd=...)`; confinement correction uses `fchmod`; file open/create/unlink, atomic replace, directory listing, and directory fsync all operate relative to verified directory descriptors.
- The regression swaps the snapshot root ancestor immediately before directory creation. The operation fails closed on the subsequent pathname walk, and the outside target receives neither the snapshot directory nor ownership lock.
- Final round-two targeted run, with cached Hypothesis 6.138.2: `50 passed, 7 warnings in 1.59s`. The warnings are the same seven pre-existing bootstrap/dependency warnings itemized above.
- Round-two static/security evidence: Ruff format check and Ruff check passed; compileall passed; Bandit 1.9.4 reported no findings; `git diff --check` passed.
