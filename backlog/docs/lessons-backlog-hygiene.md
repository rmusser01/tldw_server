# Backlog editing evidence

**Incident (PR #3056, 2026-09-29):** The installed JavaScript `backlog` editor
wrapped canonical implementation notes in `NOTES` and duplicated final-summary
markers. The repository Python parser lost the canonical notes and treated
marker text as summary content. Qodo detected both defects after publication.

Use the repository Python CLI with full `TASK-` IDs for later edits. Verify
canonical section parsing and append/summary mutation on disposable copies;
successful CLI output alone does not prove the Markdown remains editable.
