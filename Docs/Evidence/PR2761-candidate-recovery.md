# PR #2761 candidate recovery evidence

**Tracking:** TASK-13013.3; runtime fixes through `50dcf5453b1ba10b0eabb505ff32b831f47f8185`.

The repaired API image starts and passes a same-image backup/restore smoke in a disposable single-user SQLite/Redis profile. This is a bounded candidate check. The published 0.1.38 baseline fails startup with `ModuleNotFoundError: mcp_unified`, so a working published-baseline upgrade/rollback has **not** been proved.

## Artifact and environment

- Tested local image: `sha256:e01578be89000870b86c09164ad724160654f410a755a44fb84500857977b630`; amd64 manifest `sha256:0b957248ad6e78d7d86d1090b1687dd6989c5a190bb9292061991b04a04ac58a`.
- Build input was a clean archive of `0cec0bb409ddbd2de0089e1909b4b6b718823de3` plus the exact Dockerfile fix committed in `50dcf5453b`. Application runtime source equality was checked. Embedded documentation/tests/metadata predate that fix commit; this local diagnostic image is not the final published artifact.
- The original candidate image built successfully but failed importing `tldw_profile_core` at startup. The repaired Dockerfile includes its already-declared local package source and checks imports of both local Python packages during image construction. Actual rebuilt import checks and subsequent application startup succeeded.
- One app worker, 2 CPU / 4 GiB limit; Redis 0.5 CPU / 256 MiB; amd64 emulation on an arm64 Docker Desktop host. Unique `pr2761-sqlite-y4oqta` resources, an internal network, no published ports, no production mounts, generated test credentials, offline provider settings. These are test constraints, not an operating envelope.
- [Exact dependency inventory](PR2761-candidate-dependencies.txt): 286 entries, SHA-256 `703c3cbe5b42a7e022af812f3121595033fcf6a304d92171f08c25def4f39baa`. The unmodified dependency ranges resolve floating versions; this does not close TASK-13013.7.

## Rehearsal and checks

1. Confirmed `/internal/ready` returns 200. Through authenticated APIs, read the synthetic account and created one note, one conversation and a 34-byte note attachment.
2. Stopped app writers, saved Redis RDB and stopped Redis. Archived the complete app data volume and separate runtime config. Verified archive safety, checksums, Redis RDB validity, and `PRAGMA integrity_check` on all **12 SQLite databases**.
3. Created a second fresh volume set. Restored data/config with original application ownership. Loaded the RDB with AOF disabled, verified its synthetic marker, enabled AOF and waited for rewrite, then restarted normal AOF operation.
4. First app restart exposed a temporary-controller omission: fresh volume roots were still root-owned, although file ownership was restored. Corrected the two owned mount roots to uid/gid 10001 and fixed the controller. An early readiness request then arrived during startup; after explicit startup completion the endpoint returned 200. Neither adjustment altered application data or skipped an acceptance assertion.
5. Verified unchanged account identity, note content, conversation title and exact attachment bytes; attachment SHA-256 `5fc7520f7125621db3a9da5127c795e3107b39915fb3e4d2a381668618d8c1c2`. Runtime config bytes were identical and Redis retained its marker after AOF restart.
6. Stopped writers again and checked **all 12 restored SQLite databases**, each reporting `ok`. Removed only rehearsal containers and its internal network. Own test volumes, images and private backups remain for inspection; no production or unrelated fixture resources were removed.

The [machine-readable report](PR2761-candidate-recovery.json) preserves image identities, backup/config digests, database paths/schema observations, acceptance results and limits. Credentials and database archives are not checked into the repository.

Temporary scripts and raw artifacts are under `/tmp/pr2761-sqlite-rehearsal-Y4OqTa` (private directory). `control.py`, `recovery.py` and `api_probe.py` contain the executed isolated commands. Their final syntax/security checks passed; subprocess exceptions are scoped to the fixed Docker executable and owned test resources.

## Still open

A verified working deployed rollback target, cross-version seeded migration/restore, PostgreSQL production-reference deployment, multi-user/multi-worker behavior, provider operations, capacity/soak thresholds, encrypted personal-context data and complete lifecycle/erasure certification remain separate gates. This result must not be used to mark those complete.
