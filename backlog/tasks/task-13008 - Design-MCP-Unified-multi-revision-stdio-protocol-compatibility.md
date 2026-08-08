---
id: TASK-13008
title: Design MCP Unified multi-revision stdio protocol compatibility
status: In Progress
assignee: []
created_date: '2026-08-08 21:17'
updated_date: '2026-08-08 21:32'
labels:
  - mcp-unified
  - protocol
  - stdio
  - design
dependencies: []
references:
  - 'https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning'
  - >-
    https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio
documentation:
  - >-
    Docs/superpowers/specs/2026-08-08-mcp-unified-multi-revision-stdio-protocol-design.md
  - backlog/decisions/001-mcp-unified-multi-revision-stdio-protocol.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the reusable MCP Unified stdio protocol and runtime contract required for downstream consumers such as tldw_chatbook. The design must support the current stateless MCP revision plus the approved legacy compatibility chain without changing existing tldw_server HTTP/WebSocket behavior or beginning implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A design specification defines protocol profiles for 2026-07-28, 2025-11-25, 2025-06-18, 2025-03-26, and 2024-11-05.
- [ ] #2 An ADR records the stdio-only modern compliance boundary, additive public runtime API, legacy compatibility policy, and rejected alternatives.
- [ ] #3 The design defines stateless modern requests, legacy initialization, 2025-03-26-only batching, cancellation, limits, JSON Schema validation, privacy-safe caching, and safe error/logging behavior.
- [ ] #4 The design defines deterministic verification, package release gates, consumer-contract testing, and rollback without modifying implementation code.
- [ ] #5 The design is self-reviewed and linked to the downstream tldw_chatbook migration boundary.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect current standalone gateway, package/release contracts, prior MCP designs, and normative protocol sources.
2. Write the linked specification and ADR with version, security, API, verification, and release decisions.
3. Self-review for protocol compliance, privacy, compatibility, scope creep, and documentation integrity.
4. Validate links and whitespace, commit the design-only change set, and hold implementation planning for user review.

ADR required: yes
ADR path: backlog/decisions/001-mcp-unified-multi-revision-stdio-protocol.md
Reason: This changes a public runtime/dependency boundary, protocol service contract, security behavior, and release compatibility policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Drafted the multi-revision stdio protocol specification and ADR after inspecting the standalone gateway, prior MCP designs, package metadata, release workflow, and dated official MCP sources. The design records the five profile contracts, additive public API, schema/cancellation/limit/privacy boundaries, installed-artifact verification, release ordering, and the downstream Chatbook exposure contract. Self-review corrected the public cancellation-token API, injectable stdio streams, modern-only unsupported-version advertisement, legacy result-field projection, and downstream links. Verification so far: local documentation targets exist, marker scan is clean, and whitespace validation passes. Bandit is not applicable because this change touches documentation/governance only. Status remains In Progress pending user review of the committed design; no implementation plan or code changes have begun.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
