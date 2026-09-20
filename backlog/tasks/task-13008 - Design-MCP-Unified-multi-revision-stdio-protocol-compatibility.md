---
id: TASK-13008
title: Design MCP Unified multi-revision stdio protocol compatibility
status: Done
assignee: []
created_date: '2026-08-08 21:17'
updated_date: '2026-08-08 22:44'
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
  - Docs/ADR/033-mcp-unified-stdio-contract-hardening.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the reusable MCP Unified stdio protocol and runtime contract required for downstream consumers such as tldw_chatbook. The design must support the current stateless MCP revision plus the approved legacy compatibility chain without changing existing tldw_server HTTP/WebSocket behavior or beginning implementation.
<!-- SECTION:DESCRIPTION:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A design specification defines explicit profiles for 2026-07-28 plus 2025-11-25 plus 2025-06-18 plus 2025-03-26 plus 2024-11-05
- [x] #2 Canonical ADR-033 supersedes ADR-032 and records the hardened stdio-only modern boundary and strict additive API while preserving GatewayStdioServer one-message compatibility and the application-owned data boundary
- [x] #3 Discovery and unsupported-version responses advertise all five supported revisions while client guidance permits same-process retry only for a mutually supported modern profile
- [x] #4 The public contract defines a narrow core runtime plus optional resource-template extension detection plus arbitrary current JSON tool results plus exact typed request identity plus cancellation plus immutable JSON/result/catalog/page/batch limits plus safe application errors plus portable binary stdio ownership and exit semantics
- [x] #5 The design defines stateless modern requests plus legacy initialization plus 2025-03-26-only batching plus resultType validation plus current arbitrary-root outputSchema behavior plus legacy text-only projection plus direct process-isolated time-bounded and fully reaped jsonschema validation plus stable fingerprinted pagination plus privacy-safe cache and typed error metadata
- [x] #6 Verification covers both stdio surfaces plus all protocol profiles plus no server-generated requests or input_required plus installed-artifact consumer compatibility and release rollback
- [x] #7 The design is self-reviewed and linked to the downstream tldw_chatbook migration boundary without implementation code changes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect current standalone gateway and package/release contracts plus prior MCP designs plus normative protocol sources.
2. Write the linked specification and canonical ADR-032, then supersede it with ADR-033 after review hardening, while preserving both immutable historical records.
3. Self-review for protocol compliance plus public API compatibility plus privacy plus scope and cross-repository contract integrity.
4. Validate links and whitespace then commit the design-only correction and hold implementation planning for user review.

ADR required: yes
ADR path: Docs/ADR/033-mcp-unified-stdio-contract-hardening.md
Reason: This changes a public runtime and dependency boundary plus protocol service contract plus security behavior plus release compatibility policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Drafted and iteratively review-corrected the multi-revision stdio protocol specification. Canonical ADR-033 supersedes immutable ADR-032, which itself supersedes the mistakenly placed backlog decision, without rewriting accepted history. The corrected design advertises all five revisions while filtering same-process retries to modern profiles; preserves GatewayStdioServer and the legacy GatewayRuntime while adding a narrow strict core runtime; makes resource-template support optional; supports arbitrary-root current output schemas and JSON values with legacy text-only projection; and specifies type-exact request IDs, cancellation, immutable JSON/result/catalog/page/batch limits, fingerprinted cursors, stable tool-error metadata, portable binary stdio, and a direct process-isolated time-bounded jsonschema dependency. Final documentation verification passed: Backlog parses TASK-13008 and all criteria; local spec/task/ADR/index links exist; required contract markers and supersession metadata are present; invalid/stale contract phrases are absent; touched files have no trailing whitespace; and `git diff --check` is clean. Bandit and code tests are not applicable because no implementation code is touched. The design task is complete; implementation planning and code remain deliberately deferred to the next task.
<!-- SECTION:NOTES:END -->
