# Preserve Media evidence identity (UAT408)

Task13260.278.3.2. Native diagnostic8 on SQLite and PostgreSQL receives supported cited answers but the stored citation lacks source_id. Media-level retrieval creates Document.id from the canonical media row but omits media_id/source_type from metadata. The stream deliberately forwards only explicit public provenance; the UI correctly refuses to guess a source ID from a result/chunk ID.

## Stage 1: Causal evidence and regressions
**Goal**: Trace retrieval through stream projection and saved source navigation.
**Success Criteria**: Actual SQLite/official PostgreSQL retrieval reproduces missing canonical identity, including chunk versus media identity.
**Tests**: Existing restricted database fixture and actual stream context projector; positive owned source and foreign-scope controls.
**Status**: Complete

## Stage 2: Minimal provenance repair
**Goal**: Populate public canonical provenance at the retrieval owner.
**Success Criteria**: Source ID is the owning Media row, independent of chunk identity or optional descriptive metadata; private metadata remains excluded.
**Tests**: Red-to-green causal checks, affected retrieval/stream regressions, Ruff/Bandit on touched Python.
**Status**: Complete

## Stage 3: Attributable native follow-up
**Goal**: Confirm citations, source preview/navigation and saved reload on a newly built application.
**Success Criteria**: Exact native SQLite/PostgreSQL citation identity and navigation pass without relaxing the strict oracle.
**Tests**: Committed source and production artifact hashes, bounded UAT390 journey with separate catalog/wire provider values; preserve failed diagnostic8 evidence.
**Status**: In Progress

## Related causal finding UAT409 (TASK13260.278.3.3)
The first repair passes all eight citation assertions, but the final foreign-owner control fails for both SQLite chunk variants (six cases pass, two fail). Raw chunk FTS does not apply SQLite visibility predicates; restricted PostgreSQL RLS blocks the same read. Extract the existing Media search predicate in the database layer and reuse it before chunk ranking/limits. Verify personal, team, organization, empty-principal and administrator behavior through actual retrieval on both engines. Keep the original failing run.

Verification:8 original identity failures; first repair6pass/2foreign-scope failures; expanded visibility baseline62pass/6fail; final68pass on actual SQLite/required restricted PostgreSQL. Adjacent retrieval/stream/evidence122pass and existing Media search23pass. All213 checks have zero failures/skips. Ruff and production-scope Bandit pass with zero findings. Existing dependency and old pytest temporary-cleanup warnings remain recorded. Root diff review verifies bound SQL parameters, filtering before limits, matching shared visibility behavior and explicit source identity overriding descriptive metadata; independent review capacity was exhausted earlier. Native408 acceptance remains open.
