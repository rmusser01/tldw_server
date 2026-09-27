# TASK13260.217 / UAT276 — catalogue/detail overlap

## Problem and scope

On immutableac713a4 original PGmulti at CSS1200×953, catalogue x80..460.797 is narrower than its rendered table; its More actions button atx565.953 overlaps the detail beginningx476.797. A normal pointer click is intercepted by detail content. Screenshot and geometry in native-entry274275-review/10–12 are causal browser evidence. Keyboard activation works; that workaround does not resolve the layout.

Diagnose the existing Manager split-layout/list boundary and choose the smallest conventional table overflow/responsive correction. Keep selection, row controls, accessible names and detail behavior. Do not redesign the screen, hide essential actions, force clicks, change transport/backend, or revise UAT criteria.

## Ownership and verification

An entry/frontend implementer owns only the affected WorldBook layout/list file(s) and meaningful focused regression coverage. Root owns Backlog/tracker/plan/Git, immutable source copies and original runtime. Native reviewer exclusively owns original PGmulti until explicit handback. Implementer must not touch that browser/profile/runtime, model9099 or current evidence. An isolated layout harness may use a separate owned port/profile; no provider calls.

Retain before-fix real-browser failure/geometry and any test harness limits. Verify actual layout and pointer hit-target behavior at affected desktop and adjacent responsive sizes; avoid tests that merely assert the chosen CSS string. Run existing applicable selection/edit controls and scoped lint; record Bandit's TypeScript limitation truthfully. Freeze source/hashes/report for independent review. Original PostgreSQL pointer/reload acceptance on a committed immutable upgrade is required for closure.
