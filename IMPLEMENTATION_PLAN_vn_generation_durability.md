## Stage 1: Recipe snapshot storage
**Goal**: Persist versioned per-variant recipes atomically with new batches.
**Success Criteria**: A batch and all recipes commit together before Jobs enqueue; old batches remain readable.
**Tests**: Repository migration, atomic insert, and legacy batch tests.
**Status**: Complete

## Stage 2: Snapshot worker execution
**Goal**: Generate from immutable recipes for V1 batches.
**Success Criteria**: Source edits after submission cannot change generated requests or item labels; a missing recipe fails closed.
**Tests**: Worker mutation, replay, ownership, and missing-recipe tests.
**Status**: Complete

## Stage 3: Durable variant replay
**Goal**: Recover after duplicate delivery or a worker restart.
**Success Criteria**: A variant publishes and counts once; interrupted attempts can be reconciled.
**Tests**: Duplicate delivery, crash boundary, lease expiry, and counter tests.
**Status**: Not Started

## Stage 4: Browser reload recovery
**Goal**: Rediscover an ambiguous or active batch after reload.
**Success Criteria**: Reload neither loses progress nor starts another batch accidentally.
**Tests**: Frontend component and browser reload scenarios.
**Status**: Not Started

## Stage 5: Verify and deliver
**Goal**: Finish scoped quality gates and review.
**Success Criteria**: Focused tests, lint, Bandit, docs, Backlog records, and a reviewable PR against dev.
**Tests**: Backend and frontend targeted suites plus browser smoke.
**Status**: Not Started
