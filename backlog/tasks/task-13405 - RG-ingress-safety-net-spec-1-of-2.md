---
id: TASK-13405
title: RG ingress safety net (spec 1 of 2)
status: To Do
assignee: []
created_date: '2026-09-30 01:39'
updated_date: '2026-09-30 22:35'
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

PR #3066 Qodo wave (2026-09-30). Fixed 9 of 11 findings; declined 2 with a posted rationale. Changes:
- auth skip needs a policy match, an IP entity charged at ingress, and an IP auth entity;
- policy-store failures log at ERROR from both governors;
- Redis token windows quantized to about 1000 members (G = max(1, per_min // 1000));
- eviction rotates in place (also fixes a cursor that skipped keys).
Also fixed a pre-existing bypass: ingress used the client X-Request-ID as the governor op_id, and the memory governor replays the cached decision for a repeated op_id without charging, so a fixed header was never rate limited on any route. The op_id is now always server-generated. Commits 67b07ae36a..d2ecf10cca.

Renumbered from TASK-13396 on 2026-09-30. dev also gained TASK-13396 (Buddy canonical workspace URL, via #3056) before #3066 brought this one in. Content is unchanged; commit messages up to #3066 still say TASK-13396. PR A (#3066) merged 2026-09-30 22:32Z. The ADR is ADR-057, not ADR-056 as AC #3 says (PR #3041 claims 056).
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
