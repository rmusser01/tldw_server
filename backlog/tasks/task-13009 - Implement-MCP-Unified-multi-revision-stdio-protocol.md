---
id: TASK-13009
title: Implement MCP Unified multi-revision stdio protocol
status: In Progress
assignee: []
created_date: '2026-08-08 23:07'
updated_date: '2026-08-09 00:27'
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
- [x] #1 All five approved MCP revision profiles interoperate through the public stdio runtime
- [x] #2 Existing GatewayStdioServer and HTTP or WebSocket behavior remain compatible
- [x] #3 Modern stateless lifecycle and each legacy initialization lifecycle enforce revision-specific batching
- [x] #4 Public runtime contracts preserve type-exact request IDs and bounded cancellation concurrency rate JSON result catalog batch and shutdown behavior
- [x] #5 Tools resources resource templates and prompts support empty catalogs deterministic pagination and stable errors
- [x] #6 JSON Schema validation uses the declared direct dependency and disposable fully reaped bounded worker processes
- [x] #7 Current arbitrary JSON structured results and legacy text-only projection preserve protocol-correct cache and error metadata
- [x] #8 Native and fallback binary stdio paths enforce bounded payload-safe logging and deterministic child cleanup
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
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

Prepared the reviewed `mcp-unified` 0.2.0 release candidate without publishing it. The public package now documents and tests the five pinned MCP revisions, their lifecycle and batch matrix, strict stdio versus compatibility transports, bounded gateway defaults, cancellation and shutdown escalation, cache and safe-error behavior, and the application-owned local-data/privacy boundary. Root and packaged README/USER_GUIDE resources are byte-identical, package status is `public-alpha`, publishing status is `published`, and the documentation states that 0.2.0 remains a release candidate until the protected publish succeeds.

The artifact gate builds wheel and sdist independently, installs each into a separate clean uncached virtual environment, proves `mcp_unified` imports from that environment's `site-packages`, and runs the five protocol suites plus the synthetic downstream consumer outside the checkout package path. It verifies the direct `jsonschema>=4.23,<5` dependency in both metadata formats and the pinned normative schema commit and five SHA-256 values. The RC helper records bounded, path-sanitized evidence and never performs the publish.

The protected portable-stdio workflow is now capable of proving the artifact consumer contract across exactly five bounded jobs: Ubuntu on Python 3.10, 3.11, 3.12, and 3.13, plus Windows on Python 3.11. Every job retains license-first admission and runs the installed contracts, connection, stdio, and independent wheel/sdist downstream consumer tests. Acceptance criterion 9 remains unchecked until those protected jobs actually pass.

Local release evidence includes 364 passing focused protocol/artifact tests; successful clean wheel and sdist installed-suite runs of 361 tests apiece; 3 passing installed-artifact consumer cases; an authoritative 48/48 Python 3.11.13 RC-helper `all` run and successful `make mcp-unified-publish-dry-run`; Ruff checks for the changed Python files; Python 3.10 and 3.11 compile checks; and a Bandit scan of the entire gateway (19,240 lines, zero findings). The final wheel and sdist SHA-256 values are `eed06e065a5a418be76dba359e290d796ea2c0828549c88a42fe045181ddfaf2` and `fd9419ecfb497ed720995fb8772928fa5c8840bc280fdd960d10485e79f0c893`. The five accepted package-boundary baseline failures, seven independently reproducible broader baseline failures, order/global-state contamination, and the expected external-federation skip are recorded in the Task 5 implementer report. Existing gateway-wide Ruff (5 findings) and mypy (149 errors in 17 files) baselines were not changed or concealed.

ADR-033 remains the governing decision. GPL package metadata was preserved, publish protection still requires the protected environment, manual `MCP_UNIFIED_PUBLISH`, duplicate-version guard, and `MCP_UNIFIED_ALLOW_PUBLISH=1`, and no live upload was attempted. The task intentionally remains **In Progress**: the five-job Python 3.10-3.13/Windows artifact matrix must run in protected CI, followed by merge, protected publish, and fresh PyPI install/metadata verification before acceptance criteria 9-10, DoD 1, and task completion can be checked.
