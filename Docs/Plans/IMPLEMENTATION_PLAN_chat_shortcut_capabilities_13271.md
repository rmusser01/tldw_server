# Effective shortcut capabilities — TASK-13271

Accepted all-ten remediation and the existing configured shortcut reference govern this continuation. The shared/full shell currently registers handlers and palette actions even when their UI host is hidden. Compact wide Pro forces its docked sidebar visible, defeating its real Collapse/keyboard/palette commands. Reuse current shortcut enable flags, optional palette callbacks, Help props and compact sidebar state. No new persistent setting, registry or dependency.

Full shells use their existing actual rendering predicates for Sidebar and Quick Chat availability. Quick Chat inside a lazy mobile Drawer is unavailable until its host is mounted/open; a hidden shell cannot toggle latent Quick Chat state. Model palette action is supplied only when its existing shell editor can mount. Help receives the same availability flags. Compact uses sidebarOpen as visibility: entering docked wide Pro opens by default, explicit collapse stays closed until a layout change, leaving docked mode closes the former overlay, and Find conversations explicitly opens in every mode. Review actual media hydration and focus edges before implementation.

ADR assessment: no new durable architecture/API/storage rule; ADR049/050 ownership unaffected,059 tracking. Parent TASK13458 remains open.

## Stage 1: Reproduce actual capability mismatches
**Goal**: Rendered full/shared shell and compact route regressions demonstrate invisible-host actions and defeated docked collapse.
**Success Criteria**: Red negative cases preserve positive visible/legacy/mobile behavior and exact draft.
**Tests**: Real keyboard handlers, lazy palette, Help reference and compact layout/mode transitions with declared I/O boundaries.
**Status**: In Progress

## Stage 2: Match handlers and references to the real hosts
**Goal**: One availability predicate per shell governs shortcut registration, palette callbacks and Help; existing compact state honors all sidebar entry points.
**Success Criteria**: Unavailable bindings are absent and inactive, actual available actions work, no latent store toggle or transcript/settings mutation.
**Tests**: Affected shell/Help/keyboard/compact suites, both client compilers, static checks and both independent source reviews.
**Status**: Not Started

## Stage 3: Verify and record
**Goal**: Qualified emitted build plus the same owned headless browser checks the actual header-hidden reference and compact Pro collapse when runnable.
**Success Criteria**: Existing budgets/ownership remain, Backlog AC3 verified, actual source and scope limitations recorded.
**Tests**: Native keyboard/palette/Collapse/Find controls, exact draft/canonical preservation, Backlog canonical/security applicability and incremental commit.
**Status**: Not Started
