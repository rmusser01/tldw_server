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

The artifact gate builds wheel and sdist independently, installs each into a separate clean uncached virtual environment, proves `mcp_unified` imports from that environment's `site-packages`, and runs the five protocol suites plus the synthetic downstream consumer outside the checkout package path. It verifies the direct `jsonschema>=4.23,<5` dependency in both metadata formats and the pinned normative schema commit and five SHA-256 values. Fixture staging is fail-closed: only the manifest, NOTICE, and five exact regular non-symlink schemas may be copied, and traversal, absolute paths, duplicates, missing or extra members, and pin/hash mismatches are rejected. The RC helper recursively sanitizes the complete final evidence payload, including nested commands and URL userinfo, while preserving public schema URLs and safe relative paths; it never performs the publish.

The protected portable-stdio workflow is now capable of proving the artifact consumer contract across exactly five bounded jobs: Ubuntu on Python 3.10, 3.11, 3.12, and 3.13, plus Windows on Python 3.11. Every job retains license-first admission and runs the installed contracts, connection, stdio, and independent wheel/sdist downstream consumer tests. Its pull-request paths and `.github/license-first-paths.json` entry are exact ordered peers covering all five installed suites, the artifact consumer and side-effect-free build utility, the normative fixture tree, package boundary/status tests, RC/publish helpers, and both release workflows. Acceptance criterion 9 remains unchecked until those protected jobs actually pass.

Fix-round release evidence includes 388 passing focused protocol/artifact tests; successful clean wheel and sdist installed-suite runs of 362 tests apiece; 26 passing artifact-consumer/security cases with independently strict wheel and sdist stdout/stderr checks; a 24/24 focused confidentiality, fixture-confinement, routing, noisy-output, and utility-side-effect set; and a 4/4 sparse dev artifact gate. The corrected authoritative Python 3.11.13 RC-helper `all` run passed 48/48 top-level gates, and `make mcp-unified-publish-dry-run` succeeded without upload. The RC-verified wheel and sdist SHA-256 values are `9743109056e8b16bf6de080d6b1d57d76b92b951df61b5c9c74a958a258f4e52` and `a37813d35780ebb3e822de4a9cfcdf22bbbf17b8996cfaa76bda9df4661bb4d9`. Ruff checks, Python 3.10/3.11 compile checks, and docs parity/contracts passed. The authoritative command `python -m bandit -r apps/mcp-unified/src/mcp_unified/gateway Helper_Scripts/mcp_unified_rc.py -f json -o <ignored-json>` scanned exactly the gateway plus RC helper: 20,990 lines, zero findings, zero errors, and two skipped tests. The new utility is mypy-clean while three pre-existing RC-helper mypy findings remain documented. The authoritative 3,694-test package/docs partition was not rerun because the review fix changes only release tooling/tests/routing; its five accepted package-boundary failures, seven independently reproducible broader baseline failures, order/global-state contamination, and expected external-federation skip remain recorded in the Task 5 implementer report.

ADR-033 remains the governing decision. GPL package metadata was preserved, publish protection still requires the protected environment, manual `MCP_UNIFIED_PUBLISH`, duplicate-version guard, and `MCP_UNIFIED_ALLOW_PUBLISH=1`, and no live upload was attempted. The release-routing manifest and side-effect-free test utility were added to Task 5 scope during review hardening; no protocol production code changed. The task intentionally remains **In Progress**: the five-job Python 3.10-3.13/Windows artifact matrix must run in protected CI, followed by merge, protected publish, and fresh PyPI install/metadata verification before acceptance criteria 9-10, DoD 1, and task completion can be checked.
