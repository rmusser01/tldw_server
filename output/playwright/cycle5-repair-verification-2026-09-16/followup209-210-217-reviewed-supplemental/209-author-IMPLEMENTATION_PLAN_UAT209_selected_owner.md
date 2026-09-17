# UAT209 / TASK13260.147 implementation stages

## Stage 1: Establish selected-owner and compatibility boundaries
**Goal:** Real two-owner persistence and cold HTTP causal controls.
**Success criteria:** Privileged PostgreSQL failures, SQLite and existing restricted-RLS controls, native receipt attribution.
**Tests:** Retained initial six reads, mutation/link/organization/Sync/graph controls; nonmutating cold HTTP baseline replay.
**Status:** Complete.

## Stage 2: Apply explicit selected-owner scope
**Goal:** Approved three-file owner predicates with parent guards in existing transactions, unchanged generic defaults and SQLite device IDs.
**Success criteria:** Foreign rows remain absent/unchanged, owned operations and legacy SQLite labels work, caller transactions remain caller owned.
**Tests:** Current permanent suite plus actual cold dependency and stale-owner mutation controls.
**Status:** Complete. Frozen source and all202 focused controls pass. UAT210 mapping and UAT217 literal SQL correction remain separately attributed.

## Stage 3: Verify, freeze and independently review
**Goal:** Stable source/test snapshots, official PostgreSQL zero-skip receipt, adjacent Sync/Notes/lifetime tests and static baseline comparison.
**Success criteria:** Focused and required adjacent checks pass, new findings resolved/associated, exact owner-only diff excludes210/217.
**Tests:** Final permanent suite, Notes graph/folder/store/restore and171/181 lifetimes, Notes organization Sync API, Ruff/Bandit.
**Status:** Complete. Author verification/freeze and independent review complete:202 passed,0 skipped, stable six-path hashes. Parent owns native acceptance, task updates and integration. Reviewer report: `.tmp/uat209-210-217-independent-20260917/REVIEW209-210-217.md`.
