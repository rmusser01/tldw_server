# ADR-069: Native Monaco BOM source adapter

**Status:** Accepted
**Date:** 2026-10-11
**Decision owner:** TASK-13520 controller, Rulings51–52
**Related task:** TASK-13520
**Related spec:** Chatbook Docs/superpowers/specs/2026-09-27-structured-document-editing-design.md
**Related plan:** Chatbook Docs/superpowers/plans/2026-09-27-structured-document-editing.md
**Backfilled from:** N/A — controller decision recorded before adapter adoption
**Extends:** [ADR-065](065-portable-note-language-and-negotiated-structured-editing.md)

## Decision

The existing native Monaco adapter may support a leading BOM only when its exact reader `getValue(undefined,true)` and ordinary reader describe one of two concrete representations. Identical strings have native prefix offset zero, including a FEFF pasted into a model initially created without a BOM. Exact source equal to FEFF plus ordinary source has native prefix offset one: Monaco owns that prefix outside its editable range. Unexpected reader differences remain fenced.

Capture exact source and this native representation with the immutable format request. Require current exact source, unchanged representation, canonical BOM flag, proposal prefix, full request key, authority, editability and composition guards synchronously at application. For a model-owned BOM replace only the proposal body; for an in-buffer FEFF replace the complete proposal. Canonical source and persistence remain exact and unchanged in ownership.

Map native offsets to exact UTF-16 offsets by the factual prefix, and map exact offsets back with endpoint clamping. Scalar diagnostics first translate into exact UTF-16; native coordinates then apply this adapter. Native onChange returns exact source and canonical cursor. Canonical value feedback uses the actual native projection, preventing insertion of a second BOM into the ordinary buffer. Formatting remains one native undo transaction with selection and viewport preservation; no retained source undo stack or format setValue is introduced.

## Context

The reviewed candidate read exact BOM source but refused initial BOM formatting. Installed Monaco retains an initial BOM outside getFullModelRange/getOffsetAt. A naive full replacement inserts another BOM. Conversely an ordinary leading FEFF paste stays inside the buffer; unconditional stripping would lose user source. Canonical has_bom alone cannot distinguish these native origins. The integrated editor owns its model directly, rather than using the controlled-value wrapper, but its external canonical feedback needs the same native projection discipline.

## Alternatives considered

- Refuse all BOM source: safe retention but unnecessarily excludes representations the installed native model can preserve.
- Strip every leading FEFF or derive offsets from canonical has_bom: drops in-buffer source or duplicates the model-owned prefix.
- Rewrite canonical source, engines, protocol or generic editor representation: exceeds this native adapter boundary and would undermine exact source ownership.

## Qualification and limits

Retain the existing installed native guard as RED, then qualify initial model-owned and pasted in-buffer origins with exact first format/save, second no-op, native typing and A/B/C/D undo/redo, source/currentness/composition/read-only negatives, Unicode/logical-coordinate navigation, selection/viewport, and LF/CRLF controls. Unexpected representations or concrete failing native gates retain an honest specific source fence. Synthetic composition controls do not qualify physical IME. No new user option, dependency, service, storage or normalization rule is adopted.
