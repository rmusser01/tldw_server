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

The protected portable-stdio workflow is now capable of proving the artifact consumer contract across exactly five bounded jobs: Ubuntu on Python 3.10, 3.11, 3.12, and 3.13, plus Windows on Python 3.11. Every job retains license-first admission and invokes the package-only `portable-gate`, which builds both artifacts and runs all five protocol suites plus the independent wheel/sdist downstream consumer from an explicit host-package-independent mirrored tree. Its pull-request paths and `.github/license-first-paths.json` entry are exact ordered peers covering all five installed suites, the artifact consumer and side-effect-free build utility, the normative fixture tree, package boundary/status tests, RC/publish helpers, the official SDK smoke, and both release workflows. Acceptance criterion 9 remains unchecked until those protected jobs actually pass.

Final whole-branch review added the mandatory official Tier 1 interoperability gate without changing runtime dependencies. Each clean wheel and sdist environment installs `mcp==2.0.0`, pinned to the official Python SDK `v2.0.0` release/tag commit `6f69a37`, proves both SDK and gateway imports come from that environment's `site-packages`, and exercises automatic strict stdio negotiation at `2026-07-28`, tool discovery, and one tool call. The official conformance server harness is URL-oriented, so it remains inapplicable to this stdio-only strict surface and no full-transport or modern-HTTP conformance is claimed. Local Python 3.11 artifact-gate evidence records 16/16 top-level checks, 370/370 installed protocol tests for each artifact, and separate marker-only SDK successes for wheel and sdist; protected execution remains outstanding.

The same review corrected two bounded strict-protocol defects through focused TDD. Output limiting now retains the original request ID whenever any semantically acceptable bounded correlated response fits, before considering null-ID replacements; huge IDs and the exact 79/78-byte fixed-error boundary remain fail-closed. Legacy arbitrary-root text fallback for `2025-06-18`, `2025-03-26`, and `2024-11-05` now validates a supported schema-declared dialect, including Draft 2020-12, while current-revision behavior is preserved and published legacy object-root descriptors remain constrained to the negotiated dialect. Compatibility stdio, HTTP, and WebSocket surfaces were not changed.

Latest whole-branch-review evidence includes 398/398 focused protocol/artifact tests, 32/32 RC-helper tests, 13/13 documentation contracts, 4/4 sparse artifact checks, and the unchanged package-boundary partition of 351 passing plus the same five accepted legacy failures. Ruff and Python 3.10/3.11 compile checks pass; route/workflow parity is 21 ordered unique paths across exactly five portable jobs. The expanded Bandit command covering the entire gateway, RC helper, and official SDK smoke scanned 21,241 lines with zero findings, zero errors, and two skipped tests. The SDK smoke is mypy-clean; the focused production/helper mypy run reports only 19 pre-existing findings outside the new lines. The latest rebuilt wheel and sdist pass `twine check`, preserve direct `jsonschema<5,>=4.23` metadata, and hash to `18c8998bc912157ada038aacc60784546007ccaf1e57918ebbb3b260e6973905` and `b5ab92319359707c25bdcd75b666d5680a5106e6afd41051eec560d97e42ba6a`. Publish dry-run passed and did not upload.

A final fresh PyPI query and final full RC could not start because the approval reviewer reported an environment usage-limit exhaustion and explicitly prohibited an indirect workaround. The last authoritative PyPI query remains `2026-08-09T08:09:30Z`, releases exactly `['0.1.1']`; the last network-backed local artifact gate remains 16/16 with 370/370 installed tests and separate official SDK successes for both artifacts. Protected CI is therefore the next authoritative network execution and the external acceptance criteria remain open.

Fix-round release evidence includes 388 passing focused protocol/artifact tests; successful clean wheel and sdist installed-suite runs of 362 tests apiece; 26 passing artifact-consumer/security cases with independently strict wheel and sdist stdout/stderr checks; a 24/24 focused confidentiality, fixture-confinement, routing, noisy-output, and utility-side-effect set; and a 4/4 sparse dev artifact gate. The corrected authoritative Python 3.11.13 RC-helper `all` run passed 48/48 top-level gates, and `make mcp-unified-publish-dry-run` succeeded without upload. The RC-verified wheel and sdist SHA-256 values are `9743109056e8b16bf6de080d6b1d57d76b92b951df61b5c9c74a958a258f4e52` and `a37813d35780ebb3e822de4a9cfcdf22bbbf17b8996cfaa76bda9df4661bb4d9`. Ruff checks, Python 3.10/3.11 compile checks, and docs parity/contracts passed. The authoritative command `python -m bandit -r apps/mcp-unified/src/mcp_unified/gateway Helper_Scripts/mcp_unified_rc.py -f json -o <ignored-json>` scanned exactly the gateway plus RC helper: 20,990 lines, zero findings, zero errors, and two skipped tests. The new utility is mypy-clean while three pre-existing RC-helper mypy findings remain documented. The authoritative 3,694-test package/docs partition was not rerun because the review fix changes only release tooling/tests/routing; its five accepted package-boundary failures, seven independently reproducible broader baseline failures, order/global-state contamination, and expected external-federation skip remain recorded in the Task 5 implementer report.

ADR-033 remains the governing decision. GPL package metadata was preserved, publish protection still requires the protected environment, manual `MCP_UNIFIED_PUBLISH`, duplicate-version guard, and `MCP_UNIFIED_ALLOW_PUBLISH=1`, and no live upload was attempted. The release-routing manifest, side-effect-free test utility, official SDK smoke, mirrored consumer isolation, and the two reviewed strict-protocol corrections were added to Task 5 scope during review hardening. The task intentionally remains **In Progress**: the five-job Python 3.10-3.13/Windows artifact matrix and official SDK scenarios must run in protected CI, followed by merge, protected publish, and fresh PyPI install/metadata verification before acceptance criteria 9-10, DoD 1, and task completion can be checked.

The protected `0.2.0` publish subsequently completed and fresh macOS/Linux
installation plus the official SDK smoke passed, but the required Windows
portable job exposed a nested-process defect before the downstream migration
was allowed to begin. The Windows named-pipe failure, source-only POSIX test,
sdist dependency metadata, artifact-consumer path matching, and SDK launch
diagnostics were corrected incrementally for `0.2.1`. The decisive evidence
was that the SDK call failed at the exact schema-worker timeout even after
raising the bound, preloading validator dependencies, and moving the target to
a lightweight top-level module: nested `multiprocessing` reconstruction was
the incompatible boundary, not schema evaluation. Native Windows validation
therefore uses the same disposable fail-closed worker through a fixed-argument
Python subprocess with an exact-length owner-only temporary payload and
bounded verdict; POSIX and explicit process-test seams retain
`multiprocessing spawn`. Timeout, cancellation, active shutdown, reaping,
permit release, payload deletion, malformed verdict, and dialect behavior
remain bounded. The task remains **In Progress** until the corrected protected
matrix, merge, `0.2.1` publish, and fresh PyPI artifact verification pass.

Corrective local verification is green before protected execution: all five
source protocol suites pass 381/381; the RC helper passes 34/34; the package
boundary retains exactly its accepted 45 passing plus five unchanged legacy
failures. The clean `portable-gate` passes 18/18 required checks with wheel and
sdist each running 381 installed protocol tests, separate official SDK stdio
smokes, and the 29-case installed-artifact consumer. The wheel SHA-256 is
`696f4dd43dfbc86f8fa6f0a20acf357669bbd5ce3528c687c22198802c3b77cc`;
the sdist SHA-256 is
`80e4adc729470a5c9370aee0bc33a7747601dfd7718b09618c9deb0b1978f033`.
Ruff, formatting, expanded Bandit, Python 3.10/3.11 compilation, README and
USER_GUIDE package-copy parity, and diff checks pass. The public docs disclose
that native Windows briefly uses an owner-only temporary validation payload so
applications with stricter no-temporary-storage requirements can make an
informed privacy decision.
