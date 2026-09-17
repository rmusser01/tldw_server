# TASK13260.130 / UAT192 tag result columns

## Stage 1: Reproduce actual backend behavior
Goal: Preserve tag-link, JSON mirror, version and caller rollback contracts.
Success Criteria: Real official PostgreSQL failure with passing SQLite controls.
Tests: Twenty real database cases, including POST create with link assertions, PATCH existing links and PUT tags.
Status: Complete — corrected causal RED8 PostgreSQL failures/12 passing controls/0skip (22.75s). Initial harness used a nonexistent list method; retained separately and not counted as causal evidence.

## Stage 2: Minimal repair
Goal: Read selected columns by name for both supported row types.
Success Criteria: Exactly two production expressions change; SQL and transaction logic unchanged.
Tests: New twenty-case suite plus existing PostgreSQL FTS/tag tests. Update one synthetic tuple fixture to match actual named PostgreSQL rows, retaining its SQL assertions.
Status: In Progress

## Stage 3: Verify and review
Goal: Independently review behavior and scoped security; retain evidence.
Success Criteria: Required PostgreSQL zero skips, SQLite controls, no new lint/security findings.
Tests: Independent targeted rerun and native tag acceptance when the reviewed backend is restarted.
Status: Not Started
