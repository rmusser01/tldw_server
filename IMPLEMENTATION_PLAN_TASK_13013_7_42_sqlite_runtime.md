# TASK-13013.7.42 — fixed SQLite runtime

The administrator bundle restore can preserve externally supplied SQLite pages that later reach FTS queries. Upgrade the OS library to official Debian 3.53.4-2, which includes the upstream fixes, without changing the restore API or treating scanner rows as exploit reproductions. Keep Trixie and the Python locks. Fetch the amd64 and arm64 packages with Docker's mandatory SHA256 verification; install only the matching package through a read-only builder mount. Do not add an unstable APT source or waive affected SQLite versions.

Qualification exposed a scanner mapping limitation: Trivy still reports the official fixed package as affected under Trixie's suite status. After verifying the canonical image payloads, add six exact, expiring fixed-package dispositions for the two High advisory IDs across three amd64 images. Preserve all 336 earlier records and keep old versions and unrelated identities blocked. Primary Debian and SQLite records identify the package as fixed; this is not a reachability exception.

## Stage 1: Qualify package and baseline
**Goal**: Bind vendor provenance and demonstrate the old runtime fails the fixed-version check.
**Success Criteria**: Both downloads match Debian hashes; dependencies fit Trixie; benign compatibility tests run.
**Tests**: Version, JSON, WAL, backup/restore and FTS smoke checks; package metadata and shared-library binding.
**Status**: Complete — both packages match Debian SHA256 values; four small Python/architecture images built and passed 16 checks; the old runtime failed only the version gate.

## Stage 2: Update canonical Docker recipes
**Goal**: Install the exact fixed library in all three backend runtimes.
**Success Criteria**: Mandatory checksums; temporary package mounts; unchanged locked dependencies and preserved existing policy records.
**Tests**: Build the actual recipes and run the same compatibility checks plus the existing guarded-import probe.
**Status**: In Progress — all three recipes updated; 261 focused checks, Ruff, Bandit and independent source review passed. Canonical builds follow the implementation commit.

## Stage 3: Verify and retain evidence
**Goal**: Record exact candidates, scan outcomes and review.
**Success Criteria**: Fixed installed package and correct Python binding; no silent suppression; tests and Bandit reviewed; local commit.
**Tests**: Focused pinning/policy tests, offline scans, evidence hash checks and independent review.
**Status**: Not Started
