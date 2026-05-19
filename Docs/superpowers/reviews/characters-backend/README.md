# Characters Backend Review Workspace

Stage order: `1 -> 2 -> 3 -> 4 -> 5`

Stage reports:
- [2026-04-15 Rebaseline](./2026-04-15-rebaseline.md)
  - Current-tree Wave 2 rebaseline: confirms the mixed-suite ChaChaNotes `503` failures are still live, while some March lifecycle findings are already closed.
- [Stage 1](./2026-03-23-stage1-review-artifacts.md)
  - Review scaffold only: created the workspace, templates, and command inventory.
- [Stage 2](./2026-03-23-stage2-api-crud-versioning.md)
  - Lifecycle integrity review: found the highest-confidence correctness issues in restore, empty updates, and avatar/version-history behavior.
- [Stage 3](./2026-03-23-stage3-import-validation-export.md)
  - Import/export review: found malformed-text fallback surprises, image-normalization splits, and PNG round-trip limit mismatches.
- [Stage 4](./2026-03-23-stage4-exemplars-worldbooks-search.md)
  - Retrieval/world-book review: found hybrid fallback pagination inaccuracies, a response-field mismatch, and deeper-page performance risk.
- [Stage 5](./2026-03-23-stage5-chat-rate-limit-synthesis.md)
  - Chat/rate-limit review and final synthesis: points to the ChaChaNotes dependency layer as the most likely source of mixed-suite `503` failures and ranks the full backend findings set.

Recommended reading order for maintainers:
- Start with [2026-04-15 Rebaseline](./2026-04-15-rebaseline.md) for the current-tree status before trusting any March finding as still live.
- Start with [Stage 5](./2026-03-23-stage5-chat-rate-limit-synthesis.md) for the ranked synthesis and fix-order guidance.
- Then read [Stage 2](./2026-03-23-stage2-api-crud-versioning.md) for the highest-confidence lifecycle correctness bugs.
- Then read [Stage 3](./2026-03-23-stage3-import-validation-export.md) for import/export contract and round-trip issues.
- Then read [Stage 4](./2026-03-23-stage4-exemplars-worldbooks-search.md) for retrieval and world-book details.
- Read [Stage 1](./2026-03-23-stage1-review-artifacts.md) only if you need the scaffold/setup context.

Rules:
- Write findings before remediation ideas.
- Label uncertain items as assumptions instead of overstating them as bugs.
