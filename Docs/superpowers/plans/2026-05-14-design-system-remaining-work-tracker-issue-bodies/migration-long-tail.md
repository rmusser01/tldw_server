# Migrate design-system product state: Other shared surfaces and long-tail triage

Draft only: human review and approval are required before creating or updating any public GitHub issue from this body.

## Scope

Owned paths and product surfaces from the ordered path ownership map:

- `src/components/Option/Chatbooks`
- `src/components/Option/Collections`
- `src/components/Option/ChatWorkflows`
- `src/components/Option/Speech`
- `src/components/Option/ScheduledTasks`
- `src/components/Common/Settings`
- `src/components/Common/StorageQuotaBanner.tsx`
- `src/components/Option/AgentRegistry`
- `src/components/Option/Dictionaries`
- `src/components/Option/STT`
- `src/components/WorkflowEditor`
- `src/components/Common/LocaleJsonDiagnostics.tsx`
- `src/components/Common/PromptInsertModal.tsx`
- `src/components/Option/Items`
- `src/components/Option/KanbanPlayground`
- `src/components/Option/Models`
- `src/components/Option/SharedWithMe`
- Any unmatched future path until the epic explicitly reassigns it.

## Current Baseline Debt

Baseline source: `apps/packages/ui/scripts/design-system-product-state-baseline.json`
Snapshot date: 2026-05-14

- Total: 55
- `antd-product-state-import`: 55
- `canonical-state-label`: 0

Top current path groups:

- `src/components/Option/Chatbooks`: 17
- `src/components/Option/Collections`: 7
- `src/components/Option/ChatWorkflows`: 4
- `src/components/Option/Speech`: 4
- `src/components/Option/ScheduledTasks`: 3
- `src/components/Common/Settings`: 2
- `src/components/Common/StorageQuotaBanner.tsx`: 2
- `src/components/Option/AgentRegistry`: 2

## Done Criteria

- This area has zero current product-state baseline exceptions.
- Focused tests cover migrated behavior.
- `bun run verify:design-system-state` passes from `apps/packages/ui`.
- `git diff --check` passes.
- Touched-file TypeScript filtering reports no diagnostics, or unrelated baseline diagnostics are documented.
- Bandit is run for Python touches or explicitly skipped for UI-only work.

## Tracking

- Parent epic: TBD
- Backlog task: TBD
- PRs: TBD

## Notes

- Keep AntD where it is only mechanics.
- Migrate product state language to shared primitives or the state registry.
- Split implementation into reviewable PRs when the area is too broad.
- If any long-tail path group has five or more findings at tracker creation, create a dedicated sub-issue or explicitly record why it remains long-tail.
