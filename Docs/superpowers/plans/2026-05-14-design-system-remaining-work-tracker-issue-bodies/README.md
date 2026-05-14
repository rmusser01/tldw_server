# Design-System Remaining Work Tracker Issue Bodies

These are local draft GitHub issue bodies generated from:

- `Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md`
- `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-implementation-plan.md`
- `apps/packages/ui/scripts/design-system-product-state-baseline.json`

Human review and approval are required before creating or updating public GitHub issues, labels, PRs, or other public GitHub state from these drafts.

## Baseline Snapshot

Snapshot date: 2026-05-14
Baseline source: `apps/packages/ui/scripts/design-system-product-state-baseline.json`

- Total allowed legacy exceptions: 500
- `antd-product-state-import`: 481
- `canonical-state-label`: 19

This fresh baseline matches the spec snapshot, so no count drift was observed while generating these drafts.

## Source-of-Truth Model

GitHub owns mutable tracker state: current counts, issue status, and latest PR links.

Backlog.md owns execution notes, verification evidence, and PR-specific before/after records.

The live verifier remains the ground truth when tracker records disagree.

## Creation Order

1. Get human review and approval for these local drafts.
2. Search for existing design-system or product-state tracker issues to avoid duplicates.
3. Check for required labels and create missing labels only if approved.
4. Create the epic from `github-epic.md`.
5. Create the Backlog parent.
6. Create migration and governance issues.
7. Create Backlog child tasks.
8. Update all GitHub issue bodies with Backlog links.
9. Update the epic dashboard with issue links and Backlog links.
