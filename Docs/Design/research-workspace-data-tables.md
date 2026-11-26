# Research Workspace Data Tables

- ADR required — yes
- ADR path — [ADR-101](../ADR/101-research-workspace-data-tables.md)
- Reason — The public launch and replay contract spans Workspace, Media, and Jobs.

The capability query at `GET /api/v1/workspaces/{workspace_id}/research-data-tables`
checks the exact configured provider and model under the authenticated
account's Workspace and Media permissions. A POST on the same path accepts an
operation identity, the selected versioned associations, and bounded
generation options. Credentials and user-supplied source bodies are absent
from the public contract. When usage quotas are enabled, the route stays
unavailable until the native worker can meter a Workspace generation.

The worker uses the explicit route and checks the native source revisions and
the digest of the bounded text captured at admission before passing evidence
to the model. Completed rows and CSV,
JSON, and XLSX exports use the same native Data Tables endpoints as other
tables. A retry returns the same native table and job. Conflicting operation
reuse and deleted or stale reservations surface a conflict.

## Implementation plan

1. Exercise route readiness, owner and source fences, interrupted and
   concurrent launches, and native completion with real Media and Jobs data.
2. Add the bounded admission endpoint and bind first Media admission to
   native Jobs. Preserve the legacy generation request and worker route.
3. Run the focused backend checks, including both storage backends where
   available, and refresh the generated public API contract.
