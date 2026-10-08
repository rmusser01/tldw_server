# Knowledge workstream mechanism correction

TASK-13534; requester explicitly authorized fixes and removal of demonstrated duplicate mechanisms on 2026-10-08. Audit baseline: current dev 97ea9cd5fa7e3a61ee4e7c56f3d9643b7311d511. Preserve the primary checkout.

## Problem and evidence

Review the actual first-parent changes of PR3196, PR3205, PR3211 and PR3213. Git-derived full inventories are authoritative; GitHub file lists cap at100. Independent audits cover45 unique backend production paths and all client production paths. Reports will be published with the final corrections; private originals and failed receipts are retained.

Confirmed corrections: Quick Notes advances the note body base after409 without merging content, enabling a second stale overwrite. Quick Notes and Knowledge Export retain an uncertain write only in component memory despite ordinary collapse/Close unmounts; reuse existing Notes durable drafts and exact pending-write identities. Migrated Workspace restore with missing capture checkpoints can discard a public capture pin and bypass Ask/current-head checks; recover through existing owned WebClipper/Media versions, retaining known pins or surfacing unavailable/ambiguous evidence. Do not infer that the latest version is the accepted snapshot.

Public capture forcibly overrides canonical router/preflight choices to HTTPX and refuses curl. Restore existing selection only after shared transports enforce the same public egress, DNS pinning, fresh credential-free sessions, env/proxy isolation, bounded identity-encoded reads and manual redirect checks. Existing HTTPX stream_response is the consolidation candidate for the copied pin/open/restore code; verify equivalence, including injected headers, certificate policy and strict constructor behavior. Existing public preflight needs the existing async byte bound, not another reader. The installed optional curl0.16.3 source proves trust_env=False alone is ineffective; verify actual CurlOpt/session/request behavior. No new transport, dependency declaration or acquisition subsystem.

Remove unused capture_note_with_provenance only after lifecycle tests cover production plan_compound_note/provenance_step/durable batch. Add notes.provenance discovery using its existing strict Pydantic contract. Keep independent persistence, receipt tables, transaction/RLS, lifecycle and migration contracts; they have no interchangeable earlier implementation in the audited scope.

Fresh locked owning tests reproduce four default quota-eviction failures and one native split-storage spy failure. Repair instrumentation against the actual Storage prototype without weakening assertions. CSV owning33 and serial predecessor120 cases pass, leaving historical cause unknown; observed lazy module readiness is an actual test dependency. Exercise the house act+dynamicImportSettled contract under a controlled delayed real module and retain export assertions; cover CSV download as well as JSON. Do not raise timeouts or call the historical cause proved on a passing rerun.

Investigate owned lint/type/format/security findings under canonical tools. Current189-file client lint finds failures; identical-rules historical blobs distinguish prior findings from workstream additions. Fix every scoped error and introduced finding, including findings introduced by these corrections. Classify unchanged wider warnings individually using identical-rule historical evidence; retain existing owners instead of an unrelated mass type refactor. Do not disable checks or erase failed evidence. Whole-file format results and wider warning ownership remain explicit, with actual baseline evidence; no indiscriminate unrelated refactor.

## Preservation requirements

No destructive revert of all four PRs; remove only demonstrated redundant/unsupported code. Preserve successful ingestion outcomes, immutable retry bodies, source selection, original evidence, Notes tombstones/ownership and server confirmation. No force push to shared dev, gate/queue/AGENTS changes, test bypass or global egress relaxation. Actual backend/caller/test evidence governs corrections; unresolved observations remain unresolved.

## ADR assessment

No new durable rule is proposed: restore existing transport selection and apply accepted ownership, bounds and evidence requirements. ADR026/042/066 govern acquisition and snapshots; ADR031/034/065 govern Notes/Sync; ADR059 governs tracking. Historical substantive additions to accepted ADR065 are a verified governance deviation. Record their exact commits in the review/task; do not silently rewrite accepted rationale or infer missing requester approval. Any genuinely changed durable decision requires a new ADR rather than editing accepted rationale.

## Acceptance

All confirmed defects get red/green behavioral regressions through real existing seams and independent review. Publish full reuse inventory and exact changes, updated tracking including13514, preserved failed receipts, limitations, touched-scope Bandit and canonical checks. User review gets a concrete branch/PR; landing follows the actual queue mode, required exact-head statuses, real rebase where applicable and human-owned Change summary.
