---
id: TASK-13396
title: RG ingress safety net (spec 1 of 2)
status: To Do
assignee: []
created_date: '2026-09-30 01:39'
updated_date: '2026-09-30 03:37'
labels:
  - resource-governance
  - backend
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implements Docs/Design/2026-09-29-rg-ingress-safety-net-design.md in three PRs (relief, coverage, switch and docs). Plan: Docs/superpowers/plans/2026-09-29-rg-ingress-safety-net.md. TASK-13395 closes with PR B.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR A merged: safety-net defaults and permanent-429 fixes in both backends
- [ ] #2 PR B merged: resolver, route index, principal identity, audits, route-map lints, WebUI replay; TASK-13395 closed
- [ ] #3 PR C merged: single RG switch, config hygiene, ADR-056, docs
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PR A (relief) implemented on fix/rg-safety-net-relief:
- shared policy_eval (unknown policy → default → built-in default; requests inheritance; scope_pairs always keeps the caller's own bucket; token clamp);
- memory and Redis backends never deny forever, resize buckets on reload, and memory evicts idle buckets;
- the Redis scope rule is applied at every site, which also fixes a lease leak on mis-scoped entities;
- safety-net YAML: no server-wide buckets except the three email-sending auth policies, generous per-user limits, a default policy;
- policy-reference consistency CI test, and a startup log of undefined route-map targets;
- MCP categories fall back to mcp.default;
- auth endpoints are charged once (guard keyed on an IP-scoped reservation under the same policy).
Each task was reviewed (spec + quality). Broad suites (Resource_Governance, AuthNZ_Unit, Embeddings, lint): 2207 passed; 4 failures are xdist ordering interactions that pass in isolation.
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
