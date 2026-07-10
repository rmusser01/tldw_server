# Chatbooks Backup And Import UAT UX Design

**Date:** 2026-07-09
**Backlog:** TASK-12095
**Surface:** WebUI Chatbooks, Settings Chatbooks, browser-extension Chatbooks, OpenWebUI import paths
**Status:** Approved design

## Goal

Determine whether users can perform backups and imports from the WebUI and
browser extension, and whether the process is possible, straightforward, and
easy.

The review must produce a senior UX/HCI assessment, not only a technical
capability check. It should identify the smallest remediation set that would
make backup and import trustworthy for self-hosted users, researchers, and
operators who depend on local data control.

## Method

Use a mixed UAT and expert-review method grounded in Nielsen Norman Group
practice:

- Cognitive walkthrough for the primary backup, restore, and OpenWebUI migration
  tasks.
- Heuristic evaluation against NN/g's 10 usability heuristics.
- Severity rating using frequency, impact, and persistence of each problem.
- Task analysis focused on information scent, system status, error prevention,
  recognition over recall, recovery, and match to user mental models.

Findings will be scored with this scale:

- `0`: no issue
- `1`: cosmetic or low-friction issue
- `2`: minor usability problem that slows or confuses some users
- `3`: major usability problem that blocks or misleads many users
- `4`: critical issue causing failure, data-loss risk, or severe trust loss

## Scope

P0 flows:

- WebUI Chatbooks backup-all flow.
- WebUI Chatbooks selective export flow.
- WebUI Chatbooks archive import and restore flow.
- Browser-extension Chatbooks export/import parity where exposed.
- Settings Chatbooks export/import shortcuts.
- User guide and API documentation alignment with actual UI and backend
  behavior.

P1 flows:

- OpenWebUI JSON preview and import.
- OpenWebUI SQLite database preview and import.
- OpenWebUI attachment hydration after import.
- Hydration permissions, server-local paths, allowed roots, and post-import
  recovery.

Out of scope for this review:

- Redesigning the entire Chatbooks product model.
- Replacing the existing Jobs infrastructure.
- Reworking unrelated import/export systems.
- Implementing fixes before the review report and remediation spec are accepted.

## Acceptance Questions

For each task, answer:

- Can the target user find where to start without reading implementation docs?
- Can they predict what data will be included before running the action?
- Can they distinguish backup, export, import, restore, migration, and hydration?
- Can they see progress and final status?
- Can they download or locate the resulting backup without ambiguity?
- Can they prevent destructive or misleading imports?
- Can they recover from unsupported media, conflicts, failures, and permissions
  problems?
- Does the UI behavior match the docs and API contract?

The final verdict will use:

- **Possible:** the flow can be completed by a determined user.
- **Straightforward:** the intended path is discoverable, linear, and mostly
  self-explanatory.
- **Easy:** the flow has low cognitive load, clear defaults, good feedback, and
  recovery for common mistakes.

## Review Artifacts

Produce a review report under `Docs/Reviews/` with:

- Environment and evidence.
- Task walkthroughs.
- A concise verdict for WebUI, extension, Settings, and OpenWebUI migration.
- Severity-ranked findings.
- NN/g heuristic mapping for each major finding.
- Minimal remediation recommendations.
- Test coverage gaps and suggested UAT automation.

Produce a remediation design/spec if findings show the flow is not
straightforward or easy. That spec should prefer the existing Chatbooks page,
jobs, preview, and extension route over new architecture unless the review
proves those surfaces cannot support the needed experience.

## Preliminary Risks To Verify

Initial code and documentation inspection suggests these risks need direct
verification:

- "Backup everything" may be described in docs but not represented as a working
  first-class UI path.
- Settings import may default to options that the backend rejects.
- Settings shortcuts may be much narrower than the main Chatbooks page.
- OpenWebUI hydration may require users to remember or manually discover
  imported conversation ids after import.
- Browser-extension coverage may prove export works but leave import and restore
  acceptance untested.
- Existing E2E tests may validate page mechanics without validating user
  acceptance criteria.

These are hypotheses until the review report cites evidence.

## Execution Plan

1. Re-read the WebUI, Settings, extension, backend endpoint, service, docs, and
   existing test paths that define Chatbooks backup/import behavior.
2. Build a UAT matrix for P0 and P1 flows.
3. Run code/docs inspection first and record evidence.
4. Run targeted live browser UAT only for risky flows where static inspection is
   insufficient.
5. Score findings by severity and map them to NN/g heuristics.
6. Write the review report and minimal remediation spec.
7. Update `TASK-12095` with touched files, verification, skips, and final
   summary.

## Quality Bar

- Findings must cite concrete UI, docs, code, or live-test evidence.
- Recommendations must name the smallest practical fix that would improve user
  acceptance.
- Do not recommend broad rewrites when copy, defaults, routing, or job-result
  affordances solve the problem.
- Do not treat "technically possible through the API" as acceptable UX if the
  WebUI or extension flow hides required state or recovery.
- Accessibility basics are non-negotiable: keyboard reachability, visible
  status, clear labels, and non-color-only state must be considered where live
  UAT is run.

## References

- NN/g: 10 Usability Heuristics for User Interface Design
  https://www.nngroup.com/articles/ten-usability-heuristics/
- NN/g: How to Conduct a Heuristic Evaluation
  https://www.nngroup.com/articles/how-to-conduct-a-heuristic-evaluation/
- NN/g: How to Rate the Severity of Usability Problems
  https://www.nngroup.com/articles/how-to-rate-the-severity-of-usability-problems/
- NN/g: Cognitive Walkthroughs
  https://www.nngroup.com/articles/cognitive-walkthroughs/
