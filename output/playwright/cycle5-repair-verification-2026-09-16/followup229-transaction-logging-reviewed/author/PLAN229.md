# TASK-13260.170 / UAT229 plan

## Stage 1: Prove failure
**Goal**: Reproduce logging masking the original transaction failure.
**Success Criteria**: Real writes distinguish plain and brace-bearing exceptions; injected commit/rollback errors reach all affected log branches.
**Tests**: Original private 1FAIL/1PASS; permanent final6FAIL/9controls.
**Status**: Complete

## Stage 2: Minimal repair
**Goal**: Parameterize four Loguru calls.
**Success Criteria**: No transaction decisions or exception identity/cause changes; exact AST scope.
**Tests**: Focused15PASS; scoped AST, Ruff/Bandit and compile.
**Status**: Complete

## Stage 3: Verify and review
**Goal**: Confirm adjacent transaction behavior and freeze review packet.
**Success Criteria**: Required-PG/SQLite controls pass and reviewer examines exact bytes.
**Tests**: Combined47PASS/0skip; independent review pending.
**Status**: In Progress
