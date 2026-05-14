# Epic: Complete tldw WebUI and extension design-system migration

Draft only: human review and approval are required before creating or updating any public GitHub issue from this body.

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
| Chat and Playground | TBD | TBD | 2 | 2 | `canonical-state-label`: 2 | TBD | Draft |
| Ingestion, Library, and media | TBD | TBD | 39 | 39 | `antd-product-state-import`: 39 | TBD | Draft |
| Jobs, Scheduler, and Watchlists | TBD | TBD | 52 | 52 | `antd-product-state-import`: 52 | TBD | Draft |
| MCP and ACP | TBD | TBD | 45 | 45 | `antd-product-state-import`: 43, `canonical-state-label`: 2 | TBD | Draft |
| Evaluations | TBD | TBD | 40 | 40 | `antd-product-state-import`: 40 | TBD | Draft |
| Settings and account/security | TBD | TBD | 77 | 77 | `antd-product-state-import`: 71, `canonical-state-label`: 6 | TBD | Draft |
| Admin and health expansion | TBD | TBD | 47 | 47 | `antd-product-state-import`: 45, `canonical-state-label`: 2 | TBD | Draft |
| Prompt and Prompt Studio | TBD | TBD | 38 | 38 | `antd-product-state-import`: 38 | TBD | Draft |
| Flashcards, Quiz, and study flows | TBD | TBD | 48 | 48 | `antd-product-state-import`: 48 | TBD | Draft |
| Document and Workspace surfaces | TBD | TBD | 13 | 13 | `antd-product-state-import`: 13 | TBD | Draft |
| Character, Persona, and presentation surfaces | TBD | TBD | 22 | 22 | `antd-product-state-import`: 16, `canonical-state-label`: 6 | TBD | Draft |
| Writing and Review surfaces | TBD | TBD | 22 | 22 | `antd-product-state-import`: 21, `canonical-state-label`: 1 | TBD | Draft |
| Other shared surfaces and long-tail triage | TBD | TBD | 55 | 55 | `antd-product-state-import`: 55 | TBD | Draft |

## Governance and System Hardening

| Track | Issue | Backlog | Status | Latest PR |
| --- | --- | --- | --- | --- |
| Harden design-system baseline reporting and stale-entry cleanup | TBD | TBD | Draft | TBD |
| Define design-system CI gate tightening path | TBD | TBD | Draft | TBD |
| Design token, color, radius, and layout drift guards | TBD | TBD | Draft | TBD |
| Define shared design-system component ownership plan | TBD | TBD | Draft | TBD |
| Add shared design-system component documentation and examples | TBD | TBD | Draft | TBD |
| Add Browser/WebUI/extension visual QA checklist | TBD | TBD | Draft | TBD |

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
