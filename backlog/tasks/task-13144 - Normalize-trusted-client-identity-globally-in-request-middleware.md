---
id: TASK-13144
title: Normalize trusted client identity globally in request middleware
status: To Do
created_date: 2026-08-30 02:45
dependencies:
- TASK-13013.5
labels:
- security
- proxy
- middleware
- architecture
- future
priority: medium
references:
- TASK-13013.5
documentation:
- Docs/superpowers/specs/2026-08-29-task-13013-5-trusted-proxy-login-lockout-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace per-consumer trusted-proxy resolution with one global HTTP and WebSocket middleware contract that rewrites the application-visible client identity while preserving the immutable physical peer separately. Audit every request.client consumer before migration so authorization, audit, setup, WebSocket, and Resource Governor behavior cannot change silently. This is future architecture work and is not part of the current core-release epic.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 HTTP and WebSocket application code receives one canonical trusted client identity while the physical peer remains separately available for trust decisions and audit evidence.
- [ ] #2 Every existing request.client consumer is inventoried and either migrated or explicitly exempted with compatibility tests.
- [ ] #3 AuthNZ and Resource Governor duplicate resolver wrappers are removed only after direct, trusted-proxy, multi-hop, spoofed, malformed, IPv4, IPv6, setup, audit, and WebSocket regression coverage passes.
- [ ] #4 Rollout includes observability, staged enablement, and an operator-safe rollback path without weakening forwarding-header trust boundaries.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests and verification recorded
- [ ] #3 Operator and architecture documentation updated
- [ ] #4 Bandit run for touched code with no new medium/high findings
- [ ] #5 Final summary and rollout/rollback evidence added
<!-- DOD:END -->
