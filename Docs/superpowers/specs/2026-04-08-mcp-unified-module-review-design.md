# MCP Unified Module Review Design

Date: 2026-04-08
Topic: MCP Unified module full file-by-file review with findings and fixes
Status: Approved design

## Objective

Review the MCP Unified subsystem in `tldw_server` for concrete bugs,
correctness issues, security weaknesses, operational hazards, test gaps, and
maintainability problems using a full file-by-file audit.

The review should produce a code-review-style findings report with concrete fix
guidance for each issue. The emphasis is on actionable, evidence-backed
problems rather than broad redesign advice or generic style commentary.

## Scope

This review is centered on:

- `tldw_Server_API/app/core/MCP_unified`
- `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py`
- MCP Unified tests used to confirm intended behavior or expose coverage gaps

This includes:

- transport and request lifecycle behavior across HTTP and WebSocket entry
  points
- protocol parsing, request validation, response shaping, batching, and
  idempotency behavior
- auth, RBAC, rate limiting, request guards, persona and scope handling
- module framework behavior including registry, base interfaces, and module
  implementations
- external server integration including manager, transports, config schema, and
  runtime credential handling
- governance packs, monitoring, config defaults, and operational safety
- tests and documentation only when they materially define or claim current MCP
  Unified behavior

This review excludes:

- the broader MCP Hub management surface under `/api/v1/mcp/hub`
- unrelated platform endpoints that only mention MCP incidentally
- broad product or UX critique
- speculative architecture rewrites that are not justified by concrete defects
- remediation work during the review unless explicitly requested later
- generated artifacts such as `__pycache__` bytecode files

## Approaches Considered

### Chosen: Full file-by-file audit

Read every source file in the scoped MCP Unified tree, then synthesize findings
across the subsystem with concrete fix guidance.

Why this is preferred:

- it matches the user’s explicit request for a full file-by-file review
- it reduces the chance that low-visibility helpers or safety checks are missed
- it supports subsystem-wide findings without depending on sampling assumptions
- it produces a stronger basis for calling out test gaps and cross-file risks

### Alternative: Entry-point-first audit

Start from endpoint, server, protocol, and config files, then trace only into
high-risk dependencies and hotspot modules.

Trade-offs:

- faster
- better early signal for integration problems
- weaker assurance that every module file actually received direct inspection

### Alternative: Test-led audit

Use the current MCP tests as the map of intended behavior, then inspect the
code paths behind important or weakly covered behaviors.

Trade-offs:

- efficient for regressions and contract drift
- good at exposing missing tests
- risks normalizing existing blind spots in the test suite

## Chosen Method

Use the full file-by-file audit with five ordered passes:

1. `Inventory and boundary map`
   Enumerate the scoped files, identify trust boundaries, major runtime seams,
   and hotspot areas likely to carry cross-file risk.
2. `Direct file inspection`
   Read every scoped source file, recording concrete concerns and cross-file
   dependencies as they appear.
3. `Behavior cross-check`
   Use MCP-focused tests and targeted docs to confirm intended behavior,
   distinguish confirmed defects from probable risks, and identify missing
   coverage.
4. `Finding synthesis`
   Consolidate duplicate symptoms into single findings when they reflect one
   underlying problem spanning multiple files.
5. `Fix guidance`
   Attach concrete remediation guidance and likely test additions to each
   finding without turning the review into a speculative redesign roadmap.

## Review Slices

The file-by-file pass should still be organized so the findings remain
traceable. The review will use these slices:

- transport and request lifecycle:
  `mcp_unified_endpoint.py`, `server.py`, `protocol.py`,
  `security/request_guards.py`, `security/ip_filter.py`
- auth and policy:
  `auth/*`, `persona_scope.py`, and any scope-enforcement logic on the MCP
  request path
- module framework:
  `modules/base.py`, `modules/registry.py`, `modules/disk_space.py`, and all
  files under `modules/implementations/`
- external integration:
  `external_servers/*`, `command_runtime/*`, and related runtime configuration
  or adapter boundaries
- governance and operations:
  `governance_packs/*`, `monitoring/*`, `config.py`, `README.md`,
  `docker/Dockerfile`, and package boundary files where they shape runtime
  expectations

Every tracked, non-generated implementation or runtime artifact in
`tldw_Server_API/app/core/MCP_unified` must be inspected, even if a given file
ends up contributing no reportable issue. Tests remain supporting evidence
rather than primary source files for the file-by-file audit.

## Coverage Ledger

To keep the file-by-file promise auditable, the review should maintain a
working coverage ledger during execution.

The ledger should list:

- every inspected implementation or runtime file in scope
- the review slice that owned it
- whether it produced a finding, a probable risk, or no reportable issue
- any tests or docs consulted for that file when relevant

The final user-facing report may stay compact, but the review process should
not rely on memory or an informal checklist.

## Evidence Model

The review will rely on:

- direct source inspection of every scoped MCP Unified source file
- direct inspection of the main MCP endpoint surface in
  `mcp_unified_endpoint.py`
- targeted test inspection across:
  - `tldw_Server_API/app/core/MCP_unified/tests`
  - `tldw_Server_API/tests/MCP_unified` when those tests clarify intended
    runtime behavior for the scoped subsystem
- targeted documentation inspection where docs claim current MCP Unified
  behavior and materially affect review conclusions
- recent git history only when a suspicious area appears churn-heavy or
  regression-prone

The review is source-first, not test-first. Tests and docs are supporting
evidence, not substitutes for direct code inspection.

Targeted runtime verification may be used selectively when static inspection
alone leaves an important claim unresolved, especially for concurrency,
transport divergence, or safety-guard behavior. It should stay narrow and
evidence-driven rather than turning the audit into broad execution work.

## Findings Model

The review output should be organized as a code review with findings first,
ordered by severity.

Each finding should include:

- severity
- confidence
- affected files with line references
- the concrete bug, risk, or design weakness
- why it matters in practice
- the specific fix that is recommended
- the tests that should be added, tightened, or updated

When one issue spans multiple files, it should be reported once as a single
finding instead of repeated per file.

If a file or slice does not contribute any meaningful issue, it should not be
given filler commentary. The review should stay high-signal.

Observations should be labeled as one of:

- `Confirmed finding`: supported directly by source, tests, or tightly bounded
  verification
- `Probable risk`: a likely issue where the impact or trigger cannot be fully
  proven from available local evidence
- `Improvement`: a concrete change that is not strong evidence of a current
  defect but would reduce future risk or maintenance friction

## Review Focus Areas

The audit should bias toward:

- authentication and authorization boundary mistakes
- unsafe config defaults or confusing configuration precedence
- request validation gaps and malformed input handling
- state, concurrency, cache, or idempotency errors
- transport inconsistencies between HTTP and WebSocket paths
- credential leakage or redaction failures
- fragile fallbacks and fail-open behavior
- module registry and tool execution trust boundaries
- external process or transport safety
- operational hazards in metrics, health, or breaker behavior
- maintainability risks in large or multi-responsibility files when they create
  plausible defect pressure

## Success Criteria

Success for this review means:

- every source file under `tldw_Server_API/app/core/MCP_unified` is inspected
- `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py` is included
- findings are actionable enough to become implementation tasks without
  rediscovering the issue
- fix guidance is concrete and paired with likely test actions
- obvious gaps in tests, validation, auth boundaries, state handling, or config
  safety are called out explicitly

## Execution Boundaries

- The review remains non-invasive and should not silently turn into
  remediation work.
- Improvements should be proposed only when they address a concrete defect,
  risk, or repeated maintenance hazard.
- Runtime safety should not be claimed unless justified by source and available
  tests.
- Residual-risk notes are allowed at the end for complex areas that appear
  acceptable by inspection but remain hard to fully validate statically.

## Final Deliverable

The canonical output will be one findings report in code-review style with:

- findings first, ordered by severity
- concrete remediation guidance for each finding
- explicit test-gap notes tied to those findings
- a compact appendix listing the reviewed slices and files so the review is
  visibly file-by-file rather than sampled
- a note on any selective runtime verification used to confirm or narrow a
  claim

The final response should prioritize bugs, risks, and weak spots over summary.
