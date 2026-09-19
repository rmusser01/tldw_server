# UAT297 — Durable Prompt Studio sync records

Task: TASK13260.234. Repair the existing PostgreSQL insert without changing optional-sync policy or generic SQL translation.

## Stage 1: causal persistence regression
**Goal**: Demonstrate missing durable sync records on actual PostgreSQL alongside SQLite parity.
**Success Criteria**: Existing public project/prompt mutations expose absent events, with the production content schema and official PostgreSQL fixture.
**Tests**: Project create/update, prompt create/version, persisted payload/identity and subsequent writes.
**Status**: Complete

## Stage 2: minimal repair and review
**Goal**: Give the sync insert its correct explicit return column, change_id.
**Success Criteria**: Causal and adjacent tests pass on both engines; no new static/security findings; independent review clear.
**Tests**: Focused DB and existing Prompt Studio tests; Ruff/Bandit.
**Status**: Complete

## Stage 3: targeted native acceptance
**Goal**: Repeat native Prompt save/update on committed source and real PostgreSQL.
**Success Criteria**: Visible save/reload agrees with durable sync records; no missing-ID errors; evidence/tracker updated.
**Tests**: Native Prompt create/edit/reload, authenticated or read-only DB audit, frozen source parity, owned-process teardown.
**Status**: In Progress

Causal run:2 PostgreSQL failures (missing events),2 SQLite passes. Repair:59 focused tests pass,2 existing SQLite shared-write concurrency skips,0 PostgreSQL skips. Ruff clear; Bandit23 unchanged baseline module findings,0 new/errors. Independent review reports no actionable findings. Durability covers successful mutations; mutation and optional sync logging remain separate transactions.
