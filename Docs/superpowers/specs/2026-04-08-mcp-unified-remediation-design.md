# MCP Unified Remediation Design

Date: 2026-04-08

## Goal

Remediate the confirmed MCP Unified defects from the file-by-file review in one branch, using phased delivery so each class of fixes can be implemented and verified independently.

## Scope

In scope:
- Confirmed MCP Unified product defects in security/scope enforcement, data integrity, runtime hardening, compatibility, and MCP-specific ops packaging
- Supporting test coverage needed to prove each fix
- MCP test-harness cleanup needed to make the MCP suite reliably reflect product behavior

Out of scope:
- Broad architectural rewrites outside the reviewed MCP surfaces
- Unreviewed modules outside the MCP Unified remediation set
- New features unrelated to the identified defects
- Environment limitations that cannot be solved in-repo, such as sandbox policy preventing local socket bind in this session

## Constraints

- Work in one remediation branch
- Execute fixes phase by phase, not as one large patchset
- Use TDD for behavior changes
- Keep product fixes separate from harness-only fixes until the final stabilization phase
- Verify every phase before moving to the next one
- Preserve existing repo patterns unless a reviewed defect requires a boundary cleanup

## Why This Shape

The identified issues do not form one single subsystem problem. They cluster into four largely independent remediation groups:

1. Security and scope boundaries
2. Data integrity and user-visible behavior correctness
3. Runtime hardening and adapter/framework compatibility
4. Ops packaging and test-harness stabilization

Fixing these in order reduces risk:
- Security issues get closed first
- Data corruption and broken behavior get fixed before lower-priority cleanup
- Runtime hardening and compatibility can then build on stable behavior
- Harness cleanup last avoids confusing test-only artifacts with product regressions during product remediation

## Selected Approach

Use one branch with four sequential remediation phases. Each phase will:
- add or tighten failing tests first
- implement the minimum safe fix set for that slice
- run targeted verification for the touched scope
- leave the branch in a green state for that slice before advancing

## Phase Design

### Phase 1: Security and Scope Boundaries

Fix authority and isolation defects where callers can cross scope or workspace boundaries.

Primary targets:
- `modules/implementations/filesystem_module.py`
- `modules/implementations/governance_module.py`
- `modules/implementations/flashcards_module.py`
- `modules/implementations/characters_module.py`
- `modules/implementations/prompts_module.py`
- `modules/implementations/notes_module.py`
- `modules/implementations/mcp_discovery_module.py`
- `modules/implementations/sandbox_module.py`

Required outcomes:
- filesystem workspace resolution fails closed without authenticated owner binding
- governance scope derives from verified context by default and rejects conflicting caller scope unless explicitly authorized
- flashcards respect workspace isolation for list/read/write/export flows
- direct character/prompt/note MCP tools enforce the same persona scope guarantees expected from the knowledge layer

Verification:
- targeted unit/integration tests for each affected tool surface
- negative tests proving out-of-scope access is denied

### Phase 2: Integrity and Behavior Correctness

Fix broken behavior that can fail normal workflows, silently under-return data, or persist invalid state.

Primary targets:
- `modules/implementations/media_module.py`
- `modules/implementations/slides_module.py`
- `modules/implementations/quizzes_module.py`
- `modules/implementations/chats_module.py`
- `modules/implementations/knowledge_module.py`
- `modules/implementations/kanban_module.py`

Required outcomes:
- queue-enabled media ingestion degrades safely instead of hard-failing
- slides reorder validates exact permutations before mutation
- slides patch/update paths keep normalized stored state coherent
- quiz generation cannot claim success with zero valid questions persisted
- quiz updates preserve create-time invariants
- chat title search uses the correct client scoping semantics
- knowledge defaults align with published source coverage

Verification:
- targeted regression tests per defect
- existing MCP module tests for these surfaces stay green

### Phase 3: Runtime Hardening and Compatibility

Fix defects in protocol/server error handling, registry behavior under mutation, and adapter compatibility.

Primary targets:
- `protocol.py`
- `server.py`
- `modules/registry.py`
- `modules/base.py`
- `modules/implementations/external_federation_module.py`
- `external_servers/manager.py`
- `external_servers/transports/base.py`

Required outcomes:
- unexpected handler/auth exceptions are sanitized before logging, telemetry, and user-facing error paths
- module registry health checks and dynamic tool catalogs are mutation-safe
- dynamic external tool refresh can invalidate/rebuild catalog state safely
- external adapter runtime-auth support does not break compatible in-repo or extension adapters

Verification:
- isolated failing tests from the review (`test_log_redaction.py`, `test_external_server_manager.py`) become green
- registry/dynamic-catalog race coverage is added

### Phase 4: Ops and Harness Stabilization

Fix MCP-specific packaging defects and clean shared-state test pollution so the full MCP suite becomes trustworthy.

Primary targets:
- `docker/Dockerfile`
- websocket tests reusing shared app/router state
- Prometheus/shared-singleton tests
- env-mutating config tests

Required outcomes:
- Dockerfile points at real repo paths and a runnable application target
- websocket tests no longer re-register duplicate MCP routes on the shared app
- config and Prometheus tests isolate process-global state correctly
- full MCP suite failures correspond to real product issues, not shared test contamination

Verification:
- full `tldw_Server_API/app/core/MCP_unified/tests` suite rerun
- any remaining failures must be either newly introduced regressions or environment-limited cases explicitly documented

## Testing Strategy

Each phase must include:
- new failing tests for the reviewed defect set in that phase
- targeted pytest commands for touched files/modules
- regression coverage for negative authorization or invalid-state scenarios where applicable

Program-level verification at the end:
- rerun the full MCP Unified suite
- run Bandit on the touched MCP scope
- summarize any environment-limited tests separately from code defects

## File Boundary Guidance

- Prefer fixing the defect where the guarantee belongs instead of layering extra filtering elsewhere
- Use verified request context or verified claims as the source of authority; do not widen authorization from mutable metadata when avoidable
- Keep module-local validation close to the module tool surface unless the correct boundary is the shared protocol or registry layer
- Do not broaden scope into unrelated refactors unless needed to make a reviewed defect fix safe and durable

## Success Criteria

This remediation is complete when:
- all confirmed product defects from the review are addressed
- the major probable risks selected for remediation in the reviewed files are closed where they block safe operation
- targeted phase-level tests pass
- the isolated redaction and external-manager regression tests pass
- the full MCP Unified suite is substantially cleaner, with remaining failures limited to explicitly documented environment constraints if any remain
- the Dockerfile no longer references missing entrypoints or invalid app targets

## Risks

- Some reviewed issues share state through global singletons; harness cleanup may reveal additional hidden coupling while phases are underway
- External federation integration tests that require local socket bind may remain environment-limited in this session even after code is corrected
- Some scope fixes may require tightening expectations in existing tests that were implicitly relying on the current weaker behavior

## Implementation Handoff

Next step: write a phased implementation plan that decomposes these four remediation phases into concrete TDD tasks with exact files, verification commands, and checkpoints.
