# Preserve stale filing records implementation plan

> **For agentic workers:** Execute inline using the existing backlog-py editor and requesting-code-review; no implementation delegation is needed.

**Goal:** Land the useful records from PRs #2783 and #2810 under fresh IDs before closing those filing PRs, and record verified task closeouts for the same authorized cleanup batch (#3061, #3207, #3176).

**Architecture:** Documentation-only preservation. TASK-13868 and TASK-13869 retain the two media requests with qualified historical evidence; TASK-13870 retains the Actor Pack design request and all nine criteria. Unrelated colliding task records remain unchanged.

**Tech stack:** Backlog.md, backlog-py, GitHub CLI.

**Spec:** User-authorized cleanup of #2783 and #2810; original records pinned at 3f64149e9974c23facc37c0b15d9eef91d3f297d and 724db16258c74226ba38d8a8c5a55b43488bcef1.

**Tracking:** TASK-13867.

**ADR check:** ADR required: no. ADR-002 and ADR-059 govern task tracking. This preservation accepts no product or workflow architecture decision. Future media compatibility and Actor Pack design decisions remain explicitly open in their tasks.

## Global constraints

- Create and edit task records only through backlog-py; use fresh IDs checked against dev and all open PRs.
- Preserve original identity, dates, requirements, references and evidence provenance; qualify outdated statements.
- Leave all three preserved feature/design requests To Do. Record Done for the other three implementation tasks only after their actual verified normal merges. This PR implements none of the preserved requests.
- Preserve unrelated colliding records byte-for-byte.
- Require independent requesting-code-review and normal exact-head/current-base CI before merging.
- Verify replacement records on dev before closing either source PR; retain source branches.

## Stage 1: Preserve the records

**Goal:** Create canonical TASK-13868 through TASK-13870 with source attribution.
**Success criteria:** Actor Pack criteria match the original nine verbatim; media requirements retain the original intent and address the five actionable filing comments.
**Tests:** Compare source and replacement criteria, confirm provenance and To Do statuses; check source plumbing on current dev.
**Status:** Complete.

- [x] Recreate records using backlog-py after searching current backlog and allocating above TASK-13866.
- [x] Qualify model-based routing, get_add_media_form and job plumbing, unknown-field behavior and version-skew direction.
- [x] Retain old live-instance reports as historical evidence without claiming fresh API acceptance.

## Stage 2: Qualify and review

**Goal:** Produce a reviewable documentation-only PR.
**Success criteria:** Canonical task formatting, preservation assertions, applicable hooks and independent review pass.
**Tests:** backlog-py normalize --check on four new records; deterministic source/replacement comparison; pre-commit on changed files; git diff --check.
**Status:** In Progress.

- [x] Run the checks and record their actual results in TASK-13867.
- [x] Dispatch a read-only reviewer with only requirements and exact base/head.
- [x] Initial preservation-only review found no actionable findings.
- [ ] Review the combined final documentation scope, including verified implementation-task closeouts, on its exact current-base head.
- [ ] Satisfy the human Change summary gate for this concrete batch.

Bandit and runtime tests are inapplicable: no Python or runtime files change. Cross-repo reference contents and a live media instance are not revalidated; this limitation is explicit in the preserved tasks.

## Stage 3: Land and retire filing PRs

**Goal:** Preserve the records durably before closing #2783 and #2810.
**Success criteria:** All seven required statuses pass on current-base head, a normal merge lands, dev contains the replacement records, and both old PRs close with replacement links.
**Tests:** Read required statuses and merge/base identity; verify replacement blobs on freshly fetched dev; verify source PR states.
**Status:** In Progress.

- [x] Collect the task-only closeouts for #3061, #3207 and #3176 after verifying their normal merges, retaining prior phase evidence.
- [ ] Publish the replacement PR and satisfy normal required gates.
- [ ] Merge and verify all three records on dev.
- [ ] Close the two source PRs only after that verification, retaining branches and linking replacements.

Repository PR tracking may record a locally complete documentation change while merge/retirement remains pending; report the hosted merge outcome separately and do not claim the future feature/design tasks are complete.
