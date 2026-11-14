---
id: TASK-2291
title: Add WebFetch MCP tool with domain policy controls
status: To Do
labels:
- mcp
- web
- research
- tools
- agentic-execution
references:
- https://code.claude.com/docs/en/tools-reference
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement a WebFetch-style MCP tool for fetching and extracting URL content for agentic workflows. Cover domain allow/deny/ask rules, robots/outbound safety alignment with existing research/web scraping policy, Markdown-preferred extraction, cache/redirect behavior, size/time limits, prompt-free and prompt-assisted summarization modes, hooks, and metadata-only telemetry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
VERIFICATION (2026-10-14, credit-batch2 recon, worktree from dev 6cbf1e9434): implementation EXISTS and is healthy - modules/implementations/web_fetch_module.py (561 lines) + web_cache + web_rate_limit, registered in server.py behind MCP_ENABLE_WEB_FETCH_MODULE gate; tests 54/54 passed across the web trio. Task had no ACs defined (empty shell). Recommend the owning workstream close or define ACs; do not rebuild.
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
