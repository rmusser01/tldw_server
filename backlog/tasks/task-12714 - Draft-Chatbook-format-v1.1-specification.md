---
id: TASK-12714
title: Draft Chatbook format v1.1 specification
status: Done
labels:
- docs
- chatbooks
- spec
documentation:
- Docs/Product/Chatbooks_PRD.md
- Docs/Schemas/chatbooks_manifest_v1.json
- tldw_Server_API/app/core/Explainer/chatbook_adapter.py
modified_files:
- Docs/Product/Chatbooks_Format_v1_1_SPEC.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a focused Chatbook v1.1 format specification that incorporates content envelopes, compatibility rules, file inventory, integrity metadata, typed source references, and deterministic reader behavior while preserving v1 compatibility.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Created a focused Chatbook v1.1 format specification at Docs/Product/Chatbooks_Format_v1_1_SPEC.md. The spec uses the existing version field as 1.1.0, adds compatibility/producer/features/file_inventory concepts, defines v1.1 content envelopes under content_items[].metadata.envelope, preserves file_path/checksum as v1 compatibility aliases, standardizes representations/integrity/lossiness/provenance/source_refs, and defines deterministic preview/import behavior. Local review corrected two self-reference hazards: file_inventory excludes manifest.json, and whole-archive checksums must be job/download metadata or an external sidecar rather than a file inside the ZIP. Spec-review subagent was not spawned because the available multi-agent tool requires explicit user authorization for delegated agents.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created Docs/Product/Chatbooks_Format_v1_1_SPEC.md as a focused Chatbook v1.1 format specification. It defines compatibility goals, v1.1 reader behavior, feature registry, file inventory, content item envelopes, representations, structured integrity metadata, lossiness metadata, provenance, typed source references, relationships, redaction profiles, preview reporting, migration stages, validation requirements, and examples for bundled notes, reference-only media, and Explainer sessions. Verification run: rg trailing-whitespace check on the spec and Backlog task returned no matches; ASCII scan returned no matches; targeted rg confirmed the spec forbids a parallel manifest_version in manifest.json, forbids top-level payload_path, excludes manifest.json from file_inventory, and keeps whole-archive checksums outside the ZIP. Bandit and pytest were not run because this was a docs-only change with no Python code changes. Spec-review subagent was skipped because available multi-agent tooling requires explicit user authorization for delegated agents.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
