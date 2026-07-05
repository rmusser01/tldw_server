---
id: TASK-12886
title: Harden audit Integrations outbound HTTP policy
status: Done
created_date: 2026-07-04 01:32
labels:
- audit-remediation
- security
- integrations
- egress
priority: medium
references:
- AUDIT-2026-06-27-INTEGRATIONS-001
- AUDIT-2026-06-27-INTEGRATIONS-002
- https://github.com/rmusser01/tldw_server/pull/2604
documentation:
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/integrations-providers.md
- Docs/superpowers/reviews/2026-06-27-repo-audit/remediation-backlog-draft.md
modified_files:
- Docs/superpowers/plans/2026-07-04-audit-integrations-egress-remediation-plan.md
- tldw_Server_API/app/core/Workflows/adapters/research/search.py
- tldw_Server_API/app/core/Workflows/adapters/research/bibliography.py
- tldw_Server_API/app/core/LLM_Calls/tokenizer_resolver.py
- tldw_Server_API/tests/Workflows/adapters/test_research_adapters.py
- tldw_Server_API/tests/Writing/test_tokenizer_resolver_unit.py
updated_date: 2026-07-04 19:08
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remediate comprehensive audit findings AUDIT-2026-06-27-INTEGRATIONS-001 and AUDIT-2026-06-27-INTEGRATIONS-002. Route workflow research adapter outbound calls and tokenizer resolver provider/token-count requests through the central outbound HTTP policy instead of raw clients, while preserving explicit local-provider behavior where supported.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workflow research adapter direct pdf_url downloads go through central egress/proxy/trust_env controls and deny private/loopback targets by policy.
- [x] #2 Workflow research adapter first-party HTTP calls use central HTTP helpers or client factories rather than raw httpx clients.
- [x] #3 Tokenizer resolver _http_post uses the central sync HTTP helper/client path instead of requests.post directly.
- [x] #4 Regression tests cover private/loopback denial, direct pdf_url handling, central client/proxy defaults, and tokenizer base URL handling.
- [x] #5 Focused pytest, diff check, and Bandit verification are recorded before finalizing.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing regression tests for arxiv_download direct pdf_url egress denial and tokenizer _http_post central egress/proxy behavior.
2. Replace workflow research raw httpx call sites with central async HTTP helpers or client factories, keeping existing response behavior.
3. Replace tokenizer_resolver._http_post requests.post usage with the central sync fetch path.
4. Run focused pytest, diff checks, and Bandit on touched production files; record results in the task final summary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented via isolated worktree codex/audit-integrations-egress-2026-07-04. Verification run: focused pytest for research adapter and tokenizer resolver tests passed (116 passed); git diff --check passed; Bandit on touched production files exited 0 and wrote /tmp/bandit_integrations_egress.json. Also confirmed targeted raw httpx.AsyncClient and requests.post patterns were removed from touched paths.
Latest-dev validation refresh (2026-07-04): original branch was stale at merge-base 800a81b5d7f8c941969c915274eb9f878bc2c207, then rebased cleanly onto origin/dev fd5c152b065c408e4e8ee5f08da41589f21cb7f5. Post-rebase merge-base matched origin/dev. Addressed PR review feedback by letting `_managed_afetch` accept an optional pre-existing async client and restoring PubMed search/summary to share one `create_async_client(timeout=30)` context while still calling central `afetch`. Added `test_pubmed_search_adapter_reuses_central_http_client`, which verifies one central client context and two fetch calls using the same client. Passed targeted PubMed reuse test (1 passed). Passed `.venv/bin/python -m pytest tldw_Server_API/tests/Workflows/adapters/test_research_adapters.py tldw_Server_API/tests/Writing/test_tokenizer_resolver_unit.py -q` with 117 passed. Passed Bandit over touched production files with 0 findings in `/tmp/bandit_integrations_egress_latest_dev.json`. Passed `git diff --check`. Raw-client scan over touched production files for `httpx.AsyncClient`, `httpx.Client`, `requests.`, `urllib.request`, and `urlopen(` returned no matches.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Routed workflow research adapter direct HTTP calls for direct arXiv PDF download, PubMed, Semantic Scholar, Google Patents, and DOI resolution through the central HTTP helper path with managed clients that inherit trust_env=False and central egress/proxy enforcement. Replaced tokenizer resolver raw requests.post usage with the central sync fetch helper. Added regression coverage for direct pdf_url central-helper use, private URL denial through central policy, tokenizer central fetch use, tokenizer private URL denial, and PubMed shared-client reuse. Latest validation was refreshed after rebasing onto origin/dev fd5c152b065c408e4e8ee5f08da41589f21cb7f5: 117 focused tests passed, Bandit over touched production files reported 0 findings, `git diff --check` passed, and touched production files no longer contain the scanned raw-client patterns.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
