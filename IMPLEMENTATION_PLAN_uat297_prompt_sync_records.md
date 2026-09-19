# UAT297 — Durable Prompt Studio sync records

Tasks: TASK13260.234 and TASK13260.241. Repair the existing PostgreSQL insert without changing optional-sync policy or generic SQL translation.

## Stage 1: causal persistence regression
**Goal**: Demonstrate missing durable sync records on actual PostgreSQL alongside SQLite parity.
**Success Criteria**: Existing public project/prompt mutations expose absent events, with the production content schema and official PostgreSQL fixture.
**Tests**: Project create/update, prompt create/version, persisted payload/identity and subsequent writes.
**Status**: Complete

## Stage 2: minimal repair and review (reopened for UAT304)
**Goal**: Give the sync insert its correct explicit return column, change_id, and the tenant ownership expected by existing PostgreSQL sync RLS while retaining Prompt audit client identity.
**Success Criteria**: Causal and adjacent tests pass on both engines; no new static/security findings; independent review clear.
**Tests**: Focused DB and existing Prompt Studio tests, including non-bypass role with distinct tenant/audit IDs and foreign-read denial; Ruff/Bandit.
**Status**: Complete

## Stage 3: targeted native acceptance
**Goal**: Repeat native Prompt save/update on committed source and real PostgreSQL.
**Success Criteria**: Visible save/reload agrees with durable sync records; no missing-ID errors; evidence/tracker updated.
**Tests**: Native Prompt create/edit/reload, authenticated or read-only DB audit, frozen source parity, owned-process teardown.
**Status**: In Progress

Causal run:2 PostgreSQL failures (missing events),2 SQLite passes. Repair:59 focused tests pass,2 existing SQLite shared-write concurrency skips,0 PostgreSQL skips. Ruff clear; Bandit23 unchanged baseline module findings,0 new/errors. Independent review reports no actionable findings. Durability covers successful mutations; mutation and optional sync logging remain separate transactions.

First native acceptance on d859cd8d4b FAILED: two sync RLS errors under actual restricted role; new UAT304/TASK13260.241 tracks the previously masked tenant/audit-client mismatch. No source edits in running archive; all24,928 entries match. Preserve this failed run and correct regression coverage before the next committed targeted acceptance.

UAT304 causal restricted-role regression reproduces absent owner events1fail/4pass before using tenant_user_id for shared PostgreSQL sync ownership.61 focused tests now pass,2 pre-existing SQLite concurrency skips,0 PG skips. Two actual non-bypass tenants retain audit client web on project/prompt rows, read durable own events, and cannot read foreign events reciprocally. Ruff clean; Bandit23 unchanged baseline module findings/0new/errors; independent review no actionable findings. Original failed native d859cd8d4b run retained and stopped, holder exit0; next targeted native acceptance uses a new committed archive. User approved checkpoint PR immediately after this in-flight repair, before any unrelated fixes/full UAT. Fresh origin/dev remains3cff7962721a60b768464221c1f7fe2a8b25e4d5.
