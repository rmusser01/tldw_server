---
id: TASK-2292
title: Add WebSearch MCP tool over tldw research providers
status: To Do
labels:
- mcp
- web
- research
- search
- tools
- agentic-execution
references:
- https://code.claude.com/docs/en/tools-reference
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement a WebSearch-style MCP tool backed by tldw_server research/websearch providers. Support query refinement within bounded calls, allowed_domains or blocked_domains but not both, result title/url/snippet metadata, no page fetching in search responses, follow-up WebFetch workflow, profile/network grants, provider configuration, rate limits, hooks, and tool-use evaluation metrics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
VERIFICATION (2026-10-14): web_search_module.py + web_research_module.py + web_tool_base.py exist in MCP_unified implementations (38 modules total); web trio tests green. Empty-shell task, no ACs; implementation predates this record. Recommend owning workstream close.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
