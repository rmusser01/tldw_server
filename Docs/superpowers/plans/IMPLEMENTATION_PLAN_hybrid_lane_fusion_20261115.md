# Hybrid lane fusion repair — TASK-15505.7

Approved scope: preserve request weighting and prevent bounded hybrid fusion from silently becoming vector-only when the two indexes use different chunk identities. ADR-053 rank fusion remains governing; no new architecture ownership or scoring calibration.

## Stage 1: Reproduce integration losses
**Goal:** Freeze tests for disjoint chunk identities, endpoint alpha propagation and pure lanes.
**Success Criteria:** Tests fail on ignored request alpha and starvation while retaining scoped retrieval calls.
**Tests:** Focused hybrid retrieval regression with independently keyed lexical/vector results and arbitrary source text.
**Status:** Complete

## Stage 2: Repair the existing path
**Goal:** Carry alpha through RetrievalConfig and reserve the top eligible result of each nonzero lane when top_k permits.
**Success Criteria:** Weighted RRF order remains deterministic; alpha 0/1 stays pure; no limit or authority change; caches distinguish effective alpha.
**Tests:** Focused regressions plus existing retrieval and public pipeline suites.
**Status:** Complete

## Stage 3: Verify and replay
**Goal:** Independent review and one frozen retrieval-only replay by the parent.
**Success Criteria:** Scoped tests, lint and Bandit pass; held-out misses remain reported; MiniLM index configuration is measured separately without changing model defaults.
**Tests:** Current real Chroma/public endpoint provenance suite and retained 39-case replay.
**Status:** In Progress

Verification: 8 failing behavioral regressions before the repair; 172 connected retrieval/pipeline/real-Chroma/context tests pass after. Scoped Bandit has zero findings. Independent review and one frozen retrieval-only replay remain with the parent.
