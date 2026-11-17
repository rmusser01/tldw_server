# Opening a workspace with 13 saved political reports

TASK-15505.22 / UX-PDF-104. The original browser downloaded every draft and every Claims receipt before it showed the first output. Opening the same workspace now fetches a catalog. Each terminal section report has a title, review status, version, recorded model and creation or import time. Opening a report fetches its evidence and reviews. Copy and export load unopened reports before serializing the workspace.

The measurements use the same saved source workspace, a cloned database and Chromium. Cold means first navigation in a new browser context. Warm means an ordinary reload in the same tab. The measured interval ends when the first saved output card becomes visible. The baseline frontend was pinned to `d284d90d34`. Each build showed the same 13 cards in the same order.

| Browser measurement | Before, runs 1 / 2 | After, runs 1 / 2 |
|---|---:|---:|
| Cold open | 31.642 s / 40.824 s | 14.088 s / 4.480 s |
| Warm reload | 26.607 s / 24.014 s | 7.482 s / 19.268 s |
| Artifact response, decoded bytes per open | 90,797,775 | 5,313 |
| Artifact response, encoded bytes per open | 17,378,105 | 888 |

The new browser made zero full-list or detail requests during these opens. Timings include local development server and browser work. They varied between repetitions. The full artifact list remains available to existing clients. Workspaces containing only output types that require eager mapping can still use a full-list read. A previously imported copy of this workspace has a 5,490-byte catalog. Its Job 61 keeps the original logical version 311 and the destination row version 1.

The frontend regressions exercise 13 catalog entries, opening one report, collecting unopened reports, preserving exact text and review data, and refusing partial exports after a failed read, a changed revision or an account switch. A mixed inventory with an old identifier that cannot appear in an API path falls back to the full list. Backend checks cover workspace isolation, full response parity, hidden or unsupported client metadata, display aliases and the distinction between native and imported revisions.

Browser receipts for both pairs are in `/tmp/ux104-canonical-benchmark/{before,after,before-repeat,after-repeat}.json`. The source and destination database comparison is in `/tmp/ux104-readonly-profile.json`. These measurements cover the observed research workspace. The uncoached participant session remains pending.

The opt-in Playwright regression runs cold and warm against a prepared history. The pinned baseline failed because it downloaded the full list. The updated frontend passed both opens without a full-list or detail request. It records timing and bytes for comparisons. See the [browser UAT runner](../Development/Research_Workspace_Final_UAT_Runner.md) for the command and required environment.

The database and API suite passed all 30 focused tests, including real PostgreSQL role isolation and legacy metadata that PostgreSQL cannot project. The client falls back to the full list for unsupported identifiers and older server versions. The 506 connected restore, request-scope, artifact persistence and generation tests passed, as did the full frontend typecheck. A separate regression confirms that unopened reports with unknown usage do not show a zero-cost estimate.
