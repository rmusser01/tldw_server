---
id: TASK-12031
title: Wire host-level MCP answer generation controls for RPG rules lookup
status: To Do
created_date: 2026-06-26 03:33
dependencies:
- TASK-12030
labels:
- rpg
- mcp
- security
- backend
priority: high
references:
- TASK-12030
documentation:
- Docs/superpowers/plans/2026-06-25-rpg-rules-pack-attachment-retrieval-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Connect the MCP host/protocol layer to the RPG rules lookup answer-mode governance marker so `rpg.rules.lookup` can generate grounded answers over MCP only after the same token-scope, budget, rate-limit, and chat-completions controls used by REST have been enforced. TASK-12030 deliberately leaves MCP answer mode fail-closed without this trusted marker.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 MCP request preparation or execution records a trusted `mcp_rpg_answer_generation_controls` marker only after chat-completions permission, token-scope, budget, and rate-limit checks succeed.
- [ ] #2 `rpg.rules.lookup` with `mode=answer` succeeds over the mounted MCP path when the caller is authorized and controls pass, without callers being able to spoof the marker from tool arguments.
- [ ] #3 Unauthorized or over-budget MCP answer-mode requests fail before lookup and before any LLM call.
- [ ] #4 Tests cover success, missing chat permission, missing/failed generation controls, and marker-spoofing attempts.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Focused MCP protocol/runtime tests pass.
- [ ] #2 RPG MCP module tests pass.
- [ ] #3 Security/Bandit check on touched MCP/RPG scope is recorded.
- [ ] #4 RPG README or MCP docs mention the host-level control path.
<!-- DOD:END -->
