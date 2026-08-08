---
id: TASK-13008
title: Design MCP Unified multi-revision stdio protocol compatibility
status: In Progress
assignee: []
created_date: '2026-08-08 21:17'
updated_date: '2026-08-08 22:04'
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
  - Docs/ADR/032-mcp-unified-multi-revision-stdio-protocol.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the reusable MCP Unified stdio protocol and runtime contract required for downstream consumers such as tldw_chatbook. The design must support the current stateless MCP revision plus the approved legacy compatibility chain without changing existing tldw_server HTTP/WebSocket behavior or beginning implementation.
<!-- SECTION:DESCRIPTION:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A design specification defines explicit profiles for 2026-07-28 plus 2025-11-25 plus 2025-06-18 plus 2025-03-26 plus 2024-11-05
- [ ] #2 Canonical ADR-032 records the stdio-only modern boundary and strict additive API while preserving GatewayStdioServer one-message compatibility and the application-owned data boundary
- [ ] #3 Discovery and unsupported-version responses advertise all five supported revisions while client guidance permits same-process retry only for a mutually supported modern profile
- [ ] #4 The public contract defines optional resource-template extension detection plus exact cancellation and immutable limits and safe application errors and binary stdio ownership and exit semantics
- [ ] #5 The design defines stateless modern requests plus legacy initialization plus 2025-03-26-only batching plus typed IDs plus resultType validation plus MCP-shaped JSON Schema validation plus privacy-safe cache and error behavior
- [ ] #6 Verification covers both stdio surfaces plus all protocol profiles plus no server-generated requests or input_required plus installed-artifact consumer compatibility and release rollback
- [ ] #7 The design is self-reviewed and linked to the downstream tldw_chatbook migration boundary without implementation code changes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect current standalone gateway and package/release contracts plus prior MCP designs plus normative protocol sources.
2. Write the linked specification and canonical ADR-032 with version plus security plus API plus verification plus release decisions while preserving the superseded noncanonical record.
3. Self-review for protocol compliance plus public API compatibility plus privacy plus scope and cross-repository contract integrity.
4. Validate links and whitespace then commit the design-only correction and hold implementation planning for user review.

ADR required: yes
ADR path: Docs/ADR/032-mcp-unified-multi-revision-stdio-protocol.md
Reason: This changes a public runtime and dependency boundary plus protocol service contract plus security behavior plus release compatibility policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Drafted and then review-corrected the multi-revision stdio protocol specification. Canonical ADR-032 supersedes the mistakenly placed backlog decision without rewriting its accepted rationale. The corrected design advertises all five revisions while filtering same-process retries to modern profiles; preserves GatewayStdioServer semantics through a new strict server class; makes resource-template support an optional Protocol; and specifies cancellation tokens plus immutable limits plus safe application errors plus binary stream ownership plus typed IDs plus resultType rules plus MCP-shaped JSON Schema validation. Fresh documentation verification passed: git diff --check returned clean; touched paths are limited to Docs and backlog; local spec/task/ADR targets exist; Backlog parses TASK-13008; required contract markers are present; contradictory superseded phrases are absent; and touched files have no trailing whitespace. Bandit and code tests are not applicable because no implementation code is touched. Status remains In Progress pending user review; implementation planning and code changes have not begun.
<!-- SECTION:NOTES:END -->
