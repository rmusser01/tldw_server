# UAT181 HTTP ownership preparation

## Stage1 — Actual causal boundary
Goal: real dependency-wired HTTP→retained PostgreSQL connection→replacement constructor.
Status: Complete; two-case causal RED and full23-case RED retained.

## Stage2 — Reviewable ownership design
Goal: minimal HTTP/maintenance scope, caller contracts and non-HTTP audit.
Status: Design ready for parent review. No production prototype written. New external-binding/detached-child/main-registration controls are explicitly pending the selected API.

## Stage3 — Production implementation and GREEN
Goal: selected owned-lifetime implementation and unchanged causal controls plus API-specific additions.
Status: Not started; parent has held production edits pending review.
