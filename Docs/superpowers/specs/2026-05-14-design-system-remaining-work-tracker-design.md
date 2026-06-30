# Design-System Remaining Work Tracker Design

Date: 2026-05-14

## Context

The tldw WebUI and browser extension design-system foundation is in place:

- Contract: `Docs/Design/tldw_web_design_system_contract.md`
- Inventory: `Docs/Design/tldw_web_design_system_inventory.md`
- Shared UI source of truth: `apps/packages/ui/src`
- Product-state guard: `apps/packages/ui/scripts/verify-design-system-product-state.mjs`
- Current baseline: `apps/packages/ui/scripts/design-system-product-state-baseline.json`

The initial proof surface and the existing `TASK-45` migration slices are done,
but the full migration is not done. On `origin/dev` after PR #1637, the product-
state baseline still contains 500 allowed legacy exceptions:

| Rule | Count |
| --- | ---: |
| `antd-product-state-import` | 481 |
| `canonical-state-label` | 19 |

There is no current open GitHub issue whose title or body exactly tracks
`design-system` or `product-state` remaining work. The tracker should become
the durable public roadmap and the local execution ledger for the remaining
program.

## Goals

- Create one public GitHub epic for the full remaining design-system program.
- Mirror that epic with one Backlog.md parent task.
- Track migration work through product-area sub-issues, not individual baseline
  line entries.
- Track governance hardening work beside migration work.
- Keep every product-area closure measurable: the area closes only when its
  current product-state baseline exceptions reach zero.
- Preserve small reviewable implementation slices by letting each product-area
  issue accumulate multiple PR links.

In this design, "sub-issue" means a regular GitHub issue linked from the epic
checklist and cross-linked back to the epic. If GitHub native sub-issues are
available in the repository, they may be used as an enhancement, but the tracker
must not depend on that feature.

## Non-Goals

- Do not create one GitHub issue per baseline entry.
- Do not replace the product-state guard with manual tracking.
- Do not declare the design-system program complete when only the current
  planned slice queue is done.
- Do not force a repo-wide AntD replacement. AntD remains acceptable for
  mechanics such as tables, forms, modals, tooltips, and inputs.

## Source of Truth Model

Use both GitHub and Backlog.md, with different responsibilities.

### GitHub

GitHub is the public roadmap and cross-PR dashboard. It is the canonical home
for mutable tracker state: current counts, open/closed status, and latest PR
links.

- One epic issue owns program status.
- Product-area sub-issues own migration outcomes.
- Governance sub-issues own guard, CI, docs, and component ownership hardening.
- Each issue body contains the current baseline count and links to Backlog tasks
  and PRs.

### Backlog.md

Backlog.md is the local execution ledger. It records implementation notes and
PR-specific before/after evidence, but it should not be treated as the
canonical source for the latest current count when it disagrees with GitHub.

- One parent task mirrors the GitHub epic.
- One child task mirrors each GitHub sub-issue.
- Implementation PRs may create narrower Backlog tasks under the relevant child
  when the area is too large for one PR.
- Backlog notes record exact verification commands, known skips, PR review
  fixes, and per-PR before/after count evidence.

The live verifier is the ground truth. If records conflict, resolve them in
this order: verifier output, GitHub issue, then Backlog task. The two systems
intentionally duplicate only stable tracker metadata: title, scope, issue link,
and Backlog link. Mutable count/status/latest-PR state is updated in GitHub
first, then cited in Backlog notes as execution evidence.

## GitHub Epic

Title:

```text
Epic: Complete tldw WebUI and extension design-system migration
```

Labels:

- `WebUI`
- `enhancement`
- `design-system` if the label exists or can be created

Body sections:

```md
## Purpose

Track the remaining tldw WebUI and browser-extension design-system migration and
governance work after the v1 proof surface.

## Baseline Snapshot

Snapshot date: 2026-05-14
Baseline source: apps/packages/ui/scripts/design-system-product-state-baseline.json

- Total allowed legacy exceptions: 500
- antd-product-state-import: 481
- canonical-state-label: 19

Counts are refreshed after each merged migration PR.

## Migration Burn-Down

| Area | Issue | Initial | Current | Latest PR | Status |
| --- | --- | ---: | ---: | --- | --- |

## Governance and System Hardening

| Track | Issue | Status | Latest PR |
| --- | --- | --- | --- |

## Operating Rules

- Do not add new baseline exceptions unless a sub-issue explicitly accepts them
  as temporary migration debt.
- Product-area sub-issues close only when their current product-state baseline
  count is zero.
- Each migration PR updates the relevant sub-issue with before and after counts.
- Keep PRs small enough for focused review.

## References

- Docs/Design/tldw_web_design_system_contract.md
- Docs/Design/tldw_web_design_system_inventory.md
- apps/packages/ui/scripts/verify-design-system-product-state.mjs
```

## Product-Area Sub-Issues

Product-area sub-issues are ordered by the approved hybrid priority: honor the
contract migration phases, but prioritize denser baseline buckets inside each
phase.

Initial counts are grouped from the current baseline by the ordered path
ownership map below. They are point-in-time counts and should be refreshed when
issues are created.

| Area | Initial Count | Rule Split | Primary Paths |
| --- | ---: | --- | --- |
| Chat and Playground | 2 | `canonical-state-label`: 2 | `Option/Playground`, `sidepanel-chat` |
| Ingestion, Library, and media | 39 | `antd-product-state-import`: 39 | `AudiobookStudio`, `ChunkingPlayground`, `Sources`, `DataTables`, `QuickIngest` |
| Jobs, Scheduler, and Watchlists | 52 | `antd-product-state-import`: 52 | `Watchlists`, `Common/Workflow`, `AgentTasks` |
| MCP and ACP | 45 | `antd-product-state-import`: 43, `canonical-state-label`: 2 | `MCPHub`, `WorkspacePlayground`, `ACPPlayground` |
| Evaluations | 40 | `antd-product-state-import`: 40 | `Option/Evaluations` |
| Settings and account/security | 77 | `antd-product-state-import`: 71, `canonical-state-label`: 6 | `Settings`, `Integrations`, `Setup`, `TTS` |
| Admin and health expansion | 47 | `antd-product-state-import`: 45, `canonical-state-label`: 2 | `Option/Admin` |
| Prompt and Prompt Studio | 38 | `antd-product-state-import`: 38 | `Prompt`, `PromptStudio` |
| Flashcards, Quiz, and study flows | 48 | `antd-product-state-import`: 48 | `Quiz`, `Flashcards`, `StudySuggestions` |
| Document and Workspace surfaces | 13 | `antd-product-state-import`: 13 | `DocumentWorkspace` |
| Character, Persona, and presentation surfaces | 22 | `antd-product-state-import`: 16, `canonical-state-label`: 6 | `Characters`, `PersonaGarden`, `PresentationStudio` |
| Writing and Review surfaces | 22 | `antd-product-state-import`: 21, `canonical-state-label`: 1 | `WritingPlayground`, `Review` |
| Other shared surfaces and long-tail triage | 55 | `antd-product-state-import`: 55 | `Chatbooks`, `Collections`, `ChatWorkflows`, `Speech`, `ScheduledTasks` |

### Ordered Path Ownership Map

Apply these path rules from top to bottom when assigning a baseline finding to a
sub-issue. The first matching row owns the finding. If no row matches, the
finding belongs to `Other shared surfaces and long-tail triage` until the epic
explicitly reassigns it.

| Area | Owned path patterns |
| --- | --- |
| Chat and Playground | `src/components/Option/Playground`, `src/components/Common/Playground`, `src/components/Sidepanel/Chat`, `src/routes/sidepanel-chat.tsx` |
| Ingestion, Library, and media | `src/components/Option/Ingestion`, `src/components/Option/Library`, `src/components/Option/Media`, `src/components/Option/Sources`, `src/components/Option/DataTables`, `src/components/Option/AudiobookStudio`, `src/components/Option/ChunkingPlayground`, `src/components/Common/QuickIngest`, `src/components/Timeline` |
| Jobs, Scheduler, and Watchlists | `src/components/Option/Watchlists`, `src/components/Option/AgentTasks`, `src/components/Common/Workflow` |
| MCP and ACP | `src/components/Option/MCPHub`, `src/components/Option/ACPPlayground`, `src/components/Option/WorkspacePlayground` |
| Evaluations | `src/components/Option/Evaluations` |
| Settings and account/security | `src/components/Option/Settings`, `src/components/Option/Setup`, `src/components/Option/Integrations`, `src/components/Option/TTS` |
| Admin and health expansion | `src/components/Option/Admin` |
| Prompt and Prompt Studio | `src/components/Option/Prompt`, `src/components/Option/PromptStudio` |
| Flashcards, Quiz, and study flows | `src/components/Flashcards`, `src/components/Quiz`, `src/components/StudySuggestions` |
| Document and Workspace surfaces | `src/components/DocumentWorkspace`, `src/components/Option/Workspace` |
| Character, Persona, and presentation surfaces | `src/components/Option/Characters`, `src/components/Option/PresentationStudio`, `src/components/PersonaGarden` |
| Writing and Review surfaces | `src/components/Option/WritingPlayground`, `src/components/Review` |
| Other shared surfaces and long-tail triage | `src/components/Option/Chatbooks`, `src/components/Option/Collections`, `src/components/Option/ChatWorkflows`, `src/components/Option/Speech`, `src/components/Option/ScheduledTasks`, `src/components/Common/Settings`, `src/components/Common/StorageQuotaBanner.tsx`, `src/components/Option/AgentRegistry`, `src/components/Option/Dictionaries`, `src/components/Option/STT`, `src/components/WorkflowEditor`, `src/components/Common/LocaleJsonDiagnostics.tsx`, `src/components/Common/PromptInsertModal.tsx`, `src/components/Option/Items`, `src/components/Option/KanbanPlayground`, `src/components/Option/Models`, `src/components/Option/SharedWithMe`, and any unmatched future path |

The path map is part of the tracker contract. If a future migration creates a
better product-area boundary, update this map in the epic and Backlog parent
before moving counts between sub-issues.

### Product-Area Issue Template

```md
## Scope

Owned paths and product surfaces.

## Current Baseline Debt

Baseline source:
Snapshot date:

- Total:
- antd-product-state-import:
- canonical-state-label:

## Done Criteria

- This area has zero current product-state baseline exceptions.
- Focused tests cover migrated behavior.
- `bun run verify:design-system-state` passes from `apps/packages/ui`.
- `git diff --check` passes.
- Touched-file TypeScript filtering reports no diagnostics, or unrelated
  baseline diagnostics are documented.
- Bandit is run for Python touches or explicitly skipped for UI-only work.

## Tracking

- Parent epic:
- Backlog task:
- PRs:

## Notes

- Keep AntD where it is only mechanics.
- Migrate product state language to shared primitives or the state registry.
- Split implementation into reviewable PRs when the area is too broad.
```

## Governance Sub-Issues

Create governance sub-issues alongside migration sub-issues. These close only
when they deliver an actual guard, documented policy, CI path, or component
ownership decision.

Recommended governance tracks:

| Track | Done Criteria |
| --- | --- |
| Product-state baseline reporting and stale-entry cleanup | Verifier report gives useful grouped totals and stale entries are removed in migration PRs. |
| CI gate tightening path | Documented path from report mode to stricter gates without blocking unrelated work. |
| Token, color, radius, and layout guard design | Spec defines which non-product-state drift to guard next and which mechanics are allowed. |
| Component ownership plan | Button, PageShell, FeatureEmptyState, EmptyState, Badge, Alert, and WebUI-local duplicates have explicit owners and migration rules. |
| Shared component documentation and examples | Canonical primitives have usage examples for product-state, recovery, loading, empty, and status patterns. |
| Browser/WebUI/extension visual QA checklist | A repeatable visual QA checklist exists for shared surfaces across WebUI and extension contexts. |

### Governance Issue Template

```md
## Purpose

What governance risk this track reduces.

## Scope

Included guard, doc, CI, or ownership decision.

## Non-Goals

What this track explicitly does not migrate.

## Done Criteria

- Durable artifact exists and is linked from the epic.
- Verification or review path is documented.
- Follow-up migration tasks know how to use the artifact.

## Tracking

- Parent epic:
- Backlog task:
- PRs:
```

## Maintenance Workflow

### Initial Creation

1. Fetch current `origin/dev`.
2. Regenerate the baseline summary from
   `apps/packages/ui/scripts/design-system-product-state-baseline.json` on
   `origin/dev`.
3. Search GitHub for existing design-system or product-state tracker issues to
   avoid duplicates.
4. Generate issue bodies locally as a reviewable draft artifact.
5. Ask for human approval before creating or updating public GitHub issues.
6. Create the GitHub epic.
7. Create the Backlog parent task.
8. Create product-area GitHub sub-issues and Backlog child tasks.
9. Create governance GitHub sub-issues and Backlog child tasks.
10. Add all links back to the epic and Backlog parent.
11. Record the current baseline snapshot in the epic.

The issue body draft should be committed or attached to the implementation plan
before public issue creation. It is acceptable to skip committing the draft only
if the user explicitly asks for immediate GitHub issue creation after reviewing
the generated bodies in the conversation.

### Per Migration PR

1. Pick one product-area sub-issue.
2. Create a narrow Backlog implementation task if the area issue is too broad.
3. Migrate a small set of product-state findings to shared primitives or
   registry labels.
4. Remove the migrated baseline entries.
5. Run focused tests, `bun run verify:design-system-state`, `git diff --check`,
   and touched-file TypeScript filtering.
6. Update the GitHub sub-issue with before and after counts plus the PR link.
7. Add a Backlog note that links the same PR and records the same before/after
   evidence. The GitHub sub-issue remains the canonical current count.
8. Keep the area sub-issue open until its count reaches zero.

### Per Governance PR

1. Name the governance artifact being delivered.
2. Keep the PR scoped to the guard, doc, CI path, or ownership decision.
3. Add examples or tests when the artifact changes behavior.
4. Update the epic and Backlog child with links and verification notes.

## Staleness Rules

- The epic count is a snapshot, not live truth.
- The verifier output is the live truth.
- Refresh counts after each merged migration PR.
- If a baseline entry disappears without an issue update, treat it as a stale
  tracker entry and correct the relevant sub-issue.
- If a new baseline exception is introduced, it must reference the owning
  product-area or governance issue and explain why it is temporary.
- Until the baseline schema grows a dedicated tracker field, put the issue
  reference in the baseline entry `reason` using `Tracker: #NNNN`, and set
  `migrationQueue` to the product-area or governance slug used by the epic.
- A governance issue should later decide whether to add a validated
  `trackerIssue` field to the baseline schema.
- Do not close an area because it has "enough" coverage. Close it only when the
  current baseline count for that area is zero.

## Close and Reopen Rules

- Close a product-area issue only after a fresh verifier run shows zero findings
  for every path owned by that issue.
- If a later verifier run finds product-state debt in a closed area's owned
  paths, reopen the same GitHub issue and Backlog child task.
- If the reintroduced finding is an implementation regression, create a narrow
  implementation Backlog task under the reopened child and link the fixing PR.
- If the reintroduced finding is an intentional temporary exception, keep the
  area issue open and add a baseline reason with `Tracker: #NNNN`.
- If the finding belongs to no current path owner, assign it to long-tail triage
  first, then update the ordered path ownership map if a better owner is clear.

## Long-Tail Split Rules

The long-tail issue is a triage bucket, not a permanent dumping ground.

- If any long-tail path group has five or more findings at tracker creation,
  create a dedicated sub-issue or explicitly record why it remains long-tail.
- If a later verifier refresh pushes a long-tail path group to five or more
  findings, split it before starting implementation on that group.
- If a long-tail path maps clearly to an existing product area, update the
  ordered path ownership map before migrating it.
- Keep the long-tail issue open until every unmatched or intentionally retained
  path group reaches zero.

## Backlog Structure

Backlog parent title:

```text
Track remaining tldw design-system migration and governance
```

Backlog parent status starts as `To Do` until the GitHub epic and child tasks
exist. Each GitHub sub-issue gets a Backlog child task with the same title and
the GitHub URL in `references`.

Child task labels:

- `design-system`
- `webui`
- `extension`
- `product-state` for migration tasks
- `governance` for hardening tasks

Backlog child tasks should include the initial baseline count and GitHub issue
URL. They may record per-PR before/after counts in notes, but the GitHub
sub-issue owns the current count and latest PR fields.

## Risks and Mitigations

- Risk: The tracker becomes a second baseline file.
  - Mitigation: Track product-area counts and links, not per-line findings.
- Risk: Large product-area issues become unreviewable.
  - Mitigation: Keep product-area issues as area parents and use narrow Backlog
    tasks plus PRs for implementation slices.
- Risk: Governance tracks block migration delivery.
  - Mitigation: Governance issues close only durable artifacts, and migration
    issues can continue using the current guard while stricter guards are
    designed.
- Risk: Counts drift after unrelated merges.
  - Mitigation: Refresh counts from the verifier after each merged migration PR.
- Risk: Public GitHub issue creation amplifies mistakes in generated tracker
  bodies.
  - Mitigation: Generate issue bodies locally and require human approval before
    public mutation.
- Risk: The long-tail bucket hides enough work to become a second epic.
  - Mitigation: Split any long-tail path group with five or more findings into a
    dedicated sub-issue or document why it remains grouped.

## Acceptance Criteria

- A GitHub epic and Backlog parent can be created from this design without
  inventing new structure.
- Every product-area issue has measurable done criteria.
- Governance hardening is tracked separately from migration execution.
- The tracker can survive line-number churn in the baseline file.
- The issue hierarchy supports small PRs while preserving a single public
  program view.
