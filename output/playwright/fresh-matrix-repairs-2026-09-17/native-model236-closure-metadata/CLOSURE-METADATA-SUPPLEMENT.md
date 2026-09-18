# UAT236 post-closure metadata supplement

This supplement preserves the timing boundary of the earlier strict retention audit. It does not modify the accepted fresh-profile packet or either original audit.

## Pre-closure record

The retained audit at `output/playwright/fresh-matrix-repairs-2026-09-17/native-model236-fresh-retention-review/audit.json` is byte-stable (`f682f7759872dfe81c554f325b27a8c7461cc06835f5aa3f9da4b7eeb9a0d98c`). At capture time it reported **11 passing checks** and **43 matching provenance inputs** for the bounded UAT236 AC3 packet.

The original task content is recoverable from `git show 592521e9fa`. Its recorded 3,955-byte SHA-256 is `f7890b7d82e8beb7c2ee40181972f0fa23690b2ceec77b0fa24a6f4601fc7460`, matching the original input ledger exactly.

## Post-closure recheck

The current task file is intentionally mutable because its Backlog status was closed. It is now 4,989 bytes with SHA-256 `41dbdbc6d2df5663ed35f8e03851941654a641b51e177db1a9f8588d4ce895a2`.

`closure-metadata-audit.py` recalculated every ledger entry without retaining private contents or private paths. All **42 immutable inputs** still match their recorded bytes and hashes. The one mismatch is precisely the public Backlog task above. The zero-context Git diff is constrained to that task and has the closure update shape: status/checklist transitions, a closing evidence note, and final-summary metadata. No retained packet payload, acceptance evidence input, product file, or test file changed in this check.

A current rerun of the original strict audit correctly reports a gap for the changed mutable task input. That is expected after closure and does not revise the pre-closure 11-clear record.

Machine-readable aggregate facts are in `closure-metadata.json`; the verifier has SHA-256 `2bab0c3e0020a7c7ecc7d9828e549d7494e4762cb58ca63e39fecdcc4a635362`.
