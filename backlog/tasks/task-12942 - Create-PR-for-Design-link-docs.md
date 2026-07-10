---
id: TASK-12942
title: Create PR for Design link docs
status: In Progress
labels:
- docs
references:
- https://github.com/rmusser01/tldw_server/pull/2694
modified_files:
- Docs/Design/Coding_Page.md
- Docs/Design/ETL_Pipeline.md
- Docs/Design/Prompts.md
- Docs/Design/Researcher.md
- Docs/Design/VLMs.md
- Docs/Design/RAG_Links.md
- Docs/Design/ACP_Links.md
- Docs/Design/Agent_Links.md
- Docs/Design/ETL_Pipeline_Links.md
- Docs/Design/Embeddings_Links.md
- Docs/Design/Eval_Links.md
- Docs/Design/Persona_Links.md
- Docs/Design/Research_Links.md
- Docs/Design/SKILLS_md-Links.md
- Docs/Design/Security_Links.md
- Docs/Design/UX_Links.md
- Docs/Design/Writing_Links.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare and maintain PR #2694 against dev containing the Docs/Design link additions and review cleanup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2694 against latest origin/dev and addressed review feedback: renamed the unsafe RAG links filename, fixed malformed research and UX URLs, removed placeholder bullets, removed the Security typo, pruned review-flagged duplicate links, and kept the Design docs as internal Docs/Design notes rather than adding them to the published MkDocs pipeline. Verification: git diff --cached --check passed before commit; targeted bad-pattern scan found no remaining review-flagged malformed URLs/placeholders; pre-commit end-of-file-fixer passed for touched docs/task files after fixing RAG_Links.md. Bandit skipped because this is docs-only plus Backlog metadata.
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
