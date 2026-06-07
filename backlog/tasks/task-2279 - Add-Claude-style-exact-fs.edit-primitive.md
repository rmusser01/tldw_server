---
id: TASK-2279
title: Add Claude-style exact fs.edit primitive
status: To Do
labels:
- mcp
- filesystem
- tools
- agentic-execution
references:
- https://code.claude.com/docs/en/tools-reference#edit-tool-behavior
- Docs/superpowers/specs/2026-06-07-mcp-fs-patch-write-safe-edit-tools-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement a Claude-style fs.edit primitive for exact string replacement as a complement to fs.patch. The tool should enforce read-before-edit via read receipts or hashes, require exact old_string matching, reject fuzzy/regex behavior, require uniqueness unless replace_all is explicit, follow action-aware path grants, and integrate with tool-use redaction and hooks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
