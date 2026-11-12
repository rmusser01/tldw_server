# B03 saved native lore preview

Task: TASK-13260.278.18.83.45. This document records the explicit-mode design already reviewed and recorded in the B03 plan/task before implementation; this file was added during validation.

## Problem and contract

The existing prompt-preview endpoint prepares legacy complete-v2 context from the live catalog. Native character Chat uses the saved owner-bound behavior snapshot. Their different contracts mean the default preview cannot establish the native turn's frozen lore, especially after source edits. Preserve the legacy default.

Add a strict boolean `saved_native_lore`, default false, to the existing preparation body. Explicit true returns only the saved lore section and the positive warning `Saved native lore only; other prompt sections are not included.` It returns no admission authority. The caller must check that acknowledgement so older servers ignoring the flag cannot appear qualified.

## Existing authority and supported scope

Use the owned locked history snapshot, authoritative global scope check, canonical ready selection, saved behavior projector and fenced selected-content read. Accept only a complete settled single parent-graph path within the requested limit. Reuse native historical role/placeholder rules and transform only the newly appended user text through the saved deterministic dictionary plan. A validated character with no captured worldbook plan has explicit empty lore. Reuse the frozen matcher; never substitute current catalog data.

Refuse unsupported branches, partial history, media/tools, alternate preparation controls and nondeterministic dictionary prediction with explicit409. Existing foreign-owner403 and wrong-scope/not-found404 remain. This mode does not preview other native prompt sections or certify a later changed turn. It is read-only and cannot append messages.

## Scope and verification

Touched endpoint, shared request schema and real-storage native lore tests. Preserve all other preparation callers and default behavior. Test edit/disable/detach against actual native assembly, true absence, dictionary timing, placeholders, unsupported reads, source-fence drift and locked scope. Add API compatibility tests, scoped security validation and independent review. Combined public schema/API types generation remains a later engineering handoff gate; do not hand-select a fingerprint.

ADR assessment: no new durable rule. ADR-049/050 govern saved history ownership/lifecycle; ADR-059 governs task edits. This is an opt-in projection using existing authority, not a replacement history or admission model.
