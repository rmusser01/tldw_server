# Epic: Complete tldw WebUI and extension design-system migration

## Purpose

Track the remaining tldw WebUI and browser-extension design-system migration and governance work after the v1 proof surface.

## Baseline Snapshot

Snapshot date: 2026-05-14
Baseline source: `apps/packages/ui/scripts/design-system-product-state-baseline.json`

- Total allowed legacy exceptions: 500
- `antd-product-state-import`: 481
- `canonical-state-label`: 19

Counts are refreshed after each merged migration PR. GitHub owns mutable tracker state, Backlog.md owns execution notes and PR evidence, and the live verifier remains the ground truth.

## Migration Burn-Down

| Area | Issue | Backlog | Initial | Current | Rule Split | Latest PR | Status |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| Chat and Playground | #1658 | TASK-45.44.1 | 2 | 2 | `canonical-state-label`: 2 | TBD | Open |
| Ingestion, Library, and media | #1659 | TASK-45.44.2 | 39 | 39 | `antd-product-state-import`: 39 | TBD | Open |
| Jobs, Scheduler, and Watchlists | #1660 | TASK-45.44.3 | 52 | 52 | `antd-product-state-import`: 52 | TBD | Open |
| MCP and ACP | #1661 | TASK-45.44.4 | 45 | 45 | `antd-product-state-import`: 43, `canonical-state-label`: 2 | TBD | Open |
| Evaluations | #1662 | TASK-45.44.5 | 40 | 40 | `antd-product-state-import`: 40 | TBD | Open |
| Settings and account/security | #1663 | TASK-45.44.6 | 77 | 77 | `antd-product-state-import`: 71, `canonical-state-label`: 6 | TBD | Open |
| Admin and health expansion | #1664 | TASK-45.44.7 | 47 | 47 | `antd-product-state-import`: 45, `canonical-state-label`: 2 | TBD | Open |
| Prompt and Prompt Studio | #1665 | TASK-45.44.8 | 38 | 38 | `antd-product-state-import`: 38 | TBD | Open |
| Flashcards, Quiz, and study flows | #1666 | TASK-45.44.9 | 48 | 48 | `antd-product-state-import`: 48 | TBD | Open |
| Document and Workspace surfaces | #1667 | TASK-45.44.10 | 13 | 13 | `antd-product-state-import`: 13 | TBD | Open |
| Character, Persona, and presentation surfaces | #1668 | TASK-45.44.11 | 22 | 22 | `antd-product-state-import`: 16, `canonical-state-label`: 6 | TBD | Open |
| Writing and Review surfaces | #1669 | TASK-45.44.12 | 22 | 22 | `antd-product-state-import`: 21, `canonical-state-label`: 1 | TBD | Open |
| Other shared surfaces and long-tail triage | #1670 | TASK-45.44.13 | 55 | 55 | `antd-product-state-import`: 55 | TBD | Open |

## Governance and System Hardening

| Track | Issue | Backlog | Status | Latest PR |
| --- | --- | --- | --- | --- |
| Harden design-system baseline reporting and stale-entry cleanup | #1671 | TASK-45.44.14 | Open | TBD |
| Define design-system CI gate tightening path | #1672 | TASK-45.44.15 | Open | TBD |
| Design token, color, radius, and layout drift guards | #1673 | TASK-45.44.16 | Open | TBD |
| Define shared design-system component ownership plan | #1674 | TASK-45.44.17 | Open | TBD |
| Add shared design-system component documentation and examples | #1675 | TASK-45.44.18 | Open | TBD |
| Add Browser/WebUI/extension visual QA checklist | #1676 | TASK-45.44.19 | Open | TBD |

## Operating Rules

- Do not add new baseline exceptions unless a sub-issue explicitly accepts them as temporary migration debt.
- Product-area sub-issues close only when their current product-state baseline count is zero.
- Each migration PR updates the relevant sub-issue with before and after counts.
- Keep PRs small enough for focused review.
- Keep AntD allowed for mechanics such as tables, forms, modals, tooltips, and inputs.
- Resolve conflicting tracker records in this order: verifier output, GitHub issue, then Backlog task.

## References

- `Docs/Design/tldw_web_design_system_contract.md`
- `Docs/Design/tldw_web_design_system_inventory.md`
- `apps/packages/ui/scripts/verify-design-system-product-state.mjs`
- `apps/packages/ui/scripts/design-system-product-state-baseline.json`
