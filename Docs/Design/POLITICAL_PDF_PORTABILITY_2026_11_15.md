# Complete research packs and editable migration

Task: TASK-15505.9. The user explicitly requested both offline dissemination and editable cross-server migration after the existing workspace ZIP's limits were explained.

## Design and boundaries

Keep the existing workspace snapshot format readable. Add an explicitly named **Export complete research pack** action alongside the lightweight workspace export. The complete ZIP contains the workspace snapshot, an existing Chatbook archive restricted to the workspace's Media and associated saved Notes, readable reports/Notes and original source files. Include an offline index with relative original-file links and physical PDF page references. Do not silently fall back to an incomplete JSON download when complete export fails. Existing lightweight exports remain importable and are clearly described as snapshots requiring their existing server sources.

Chatbooks remains the owner of Media and saved Notes serialization and restore. The complete pack does not implement a second media-ingestion engine. Enumerate associated Notes through the existing workspace-tag search with complete pagination; include the current canonical Note and preserve any unsaved current draft separately. Export must fail explicitly if selected source bytes or requested saved content are missing. Do not broaden to unrelated account data or include provider/session credentials.

Extend the existing Chatbooks import result with optional verified media and Note identity mappings. Media mappings are emitted only after successful restore or an existing same-owner record with matching import provenance; ambiguous aliases remain unresolved. Preserve the source's exact canonical text/page map and restore original files through existing storage abstractions. Page provenance is validated against canonical text before reuse. Restore results are scoped to the authenticated destination, not supplied by the archive.

The import UI captures the current server/account scope, validates the complete pack and imports its embedded Chatbook using explicit content allowlists. It requires mappings for every linked source and saved Note before constructing the imported workspace. Remap typed Media references in source rows, chat citations, section evidence, artifact lineage, review metadata and Note provenance. Preserve exact quoted text, offsets and page labels. Reset old workspace/artifact/job/chat mutation authority; a new workspace reconciles through the existing canonical Workspace API. Old job checkpoints are historical data and cannot cause polling or paid work on the destination. Update restored Notes' workspace association and provenance through the existing Notes APIs. Missing or ambiguous mappings must never fall back to a numerically identical destination source.

Validate archive paths, expected file entries, declared sizes and total expanded size before expansion. Validate the manifest and selected identities before network writes. Keep the original local workspace unchanged on failure. If the server restored some items before an error or cancellation, explain that partial result and retain recoverable receipts; never claim atomic rollback that the existing Chatbooks engine does not provide. Bind every request and final state update to the captured account/server scope.

## Reviewable units

1. Backend: optional Chatbook import mappings and canonical Media page-provenance preservation; sync/async results, schemas and caller tests.
2. Frontend: portable ZIP codec/offline index, scoped export/import orchestration, explicit UI progress/errors and typed reference remapping; keep helpers separate from the large header component.
3. Acceptance: restore onto a separate isolated server containing a different Media record at the original numeric ID; verify original checksum, all associated Notes, reviewed corrections, chapter evidence and actual citation navigation. Test old bundle compatibility, malformed archives, partial restore and account changes.

## ADR check

ADR required: no new ownership or queue rule. This extends existing Chatbooks serialization/results and the Workspace cache/reconciliation contract under ADR-007, with Notes under ADR-065 and Media storage remaining canonical. The optional portable payload is backward compatible; existing snapshots do not gain an unsupported cross-server guarantee. No new archive engine, dependency, database or authority source is introduced. Reassess if implementation requires changing these owners.
