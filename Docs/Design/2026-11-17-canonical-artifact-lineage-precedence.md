# Native saved-job version and lineage display

TASK-15505.26 / UX-PDF-107 addresses a source walkthrough where a card loaded at
v285 reconnects to completed Job61 at v308, then receives review edits at v311.
The generated text and all Claims receipts remain exact, but client metadata
retains v285/v284 lineage and later overrides current display fields. This is
NN/g visibility of system status and consistency (H1/H4), agent severity 2.

The bounded fix applies canonical version and lineage precedence in the existing
shared mapper only when the outer server producer origin is
`research_workspace_output_job`. Its completed-result merge must also propagate
fresh root/version/previous IDs instead of retaining the initial card's IDs.
Canonical missing or null links clear stale self-revision hints. Explicit logical
roots and predecessors pointing to a different artifact remain intact: a saved
`Create new version` parent is distinct from an earlier revision of the same row.
Only the exact `<current artifact ID>:v<positive integer>` namespace denotes a
self-revision; a cross-artifact ID such as `<current ID>:variant` is preserved.
Display card IDs,
resource IDs, exact structured data, Claims findings and review overlays retain
their existing contracts. Canonical `workspaceArtifactVersion` remains the save
concurrency authority.

ACP and other generated resources intentionally have logical versions distinct
from workspace persistence revisions. Imported wrappers retain historical
provenance. The fix does not traverse imported producers, rewrite historical
metadata, or collapse those version domains. The existing resource-v3 versus
workspace-v12 regression remains unchanged.

Tests reproduce stale native-job hints at readback, reconnect from v285 to v308,
then review persistence through v311. They check the actual save request version,
current displayed/exported identity, and unchanged content/Claims/reviews. Null
canonical links, native new-version parent links through reload/export, and
imported outer-origin wrappers receive separate coverage.
Connected persistence, import, remapping and generation suites guard neighboring
contracts. No model, provider, backend or frozen evaluation code changes.

ADR required: no new ADR. [ADR-007](../ADR/007-research-workspace-canonical-first-slice-shell.md)
and [ADR-008](../ADR/008-workspace-split-key-persistence-and-indexeddb-offload.md)
govern the existing workspace shell and persistence. This restores an existing
projection's fidelity without changing storage schema or authority boundaries.

Verification: four initial regressions failed before the fix. Independent review
then identified cross-artifact parent loss; save/reload/export and exact namespace
regressions reproduced it before the final adjustment. The final connected suites
pass 244 tests; 26 selected ACP/version tests also pass (69 unrelated tests excluded
by the name filter). Independent final review passes 107 focused tests and reports
no remaining finding. Scoped production/persistence-test TypeScript has zero
diagnostics. The generation integration test retains 13 pre-existing fixture-cast
diagnostics with none introduced; ESLint retains 14 pre-existing warnings with no
new warnings or errors. Bandit was invoked on touched frontend directories; this
TypeScript-only scope contains no applicable Python files. Frozen backend/core
hashes remain unchanged. Parent-owned live browser and pack qualification remain
separate from these implementation checks.
