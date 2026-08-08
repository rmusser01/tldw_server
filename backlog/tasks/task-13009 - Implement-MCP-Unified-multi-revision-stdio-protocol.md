---
id: TASK-13009
title: Implement MCP Unified multi-revision stdio protocol
status: In Progress
assignee: []
created_date: '2026-08-08 23:07'
updated_date: '2026-08-08 23:15'
labels:
  - mcp-unified
  - protocol
  - stdio
  - security
dependencies:
  - TASK-13008
references:
  - >-
    https://github.com/modelcontextprotocol/modelcontextprotocol/tree/5f5440bb26a62e2cf3440b92da5a667efa03b267/schema
documentation:
  - >-
    Docs/superpowers/specs/2026-08-08-mcp-unified-multi-revision-stdio-protocol-design.md
  - >-
    Docs/superpowers/plans/2026-08-08-mcp-unified-multi-revision-stdio-implementation.md
  - Docs/ADR/033-mcp-unified-stdio-contract-hardening.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement and release the reusable MCP Unified stdio protocol runtime defined by the approved multi-revision design so downstream local applications can consume one bounded interoperable gateway package.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All five approved MCP revision profiles interoperate through the public stdio runtime
- [ ] #2 Existing GatewayStdioServer and HTTP or WebSocket behavior remain compatible
- [ ] #3 Modern stateless lifecycle and each legacy initialization lifecycle enforce revision-specific batching
- [ ] #4 Public runtime contracts preserve type-exact request IDs and bounded cancellation concurrency rate JSON result catalog batch and shutdown behavior
- [ ] #5 Tools resources resource templates and prompts support empty catalogs deterministic pagination and stable errors
- [ ] #6 JSON Schema validation uses the declared direct dependency and disposable fully reaped bounded worker processes
- [ ] #7 Current arbitrary JSON structured results and legacy text-only projection preserve protocol-correct cache and error metadata
- [ ] #8 Native and fallback binary stdio paths enforce bounded payload-safe logging and deterministic child cleanup
- [ ] #9 Wheel and sdist installs support Python 3.10 through 3.13 and pass a synthetic downstream consumer contract
- [ ] #10 Focused unit integration security packaging documentation and release verification evidence is recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add public revision profiles limits cancellation errors and strict runtime contracts through TDD.
2. Add pinned schemas bounded validation result projection and authenticated pagination through TDD.
3. Add revision-aware lifecycle dispatch batching rate limits and cancellation through TDD.
4. Add portable binary stdio framing and shutdown while preserving compatibility surfaces.
5. Complete artifact consumer docs security RC publish and post-release verification gates.

ADR required: yes
ADR path: Docs/ADR/033-mcp-unified-stdio-contract-hardening.md
Reason: This implementation directly realizes the accepted public protocol dependency transport and security contract; no new ADR is needed.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
