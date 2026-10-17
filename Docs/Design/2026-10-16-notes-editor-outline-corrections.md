# Notes editor and outline corrective behavior (TASK13816 / issue3117)

Status: implementation-scoped corrective specification; verification pending.
Scope: accepted coupled NE03 + NE06 repair in the existing 14-file candidate.

## Existing ownership and architecture

Canonical note body remains Markdown and uses the existing dirty/save machinery.
Keep the textarea and uncontrolled contentEditable bridge; preserve rich nodes and
selection during user input. Reuse existing toolbar, disclosure, converter, hook,
shared route wrappers, test fixtures and both client build owners.
Extend the existing converter only; no replacement parser, framework, dependency, store, runner or persistence format.
ADR assessment: no new ADR required for these corrective behaviors.
ADR026/031/034/065/066 remain governing; accepted rationale stays unchanged.

## NE06 formatting and list editing

The normal toolbar offers explicit H1/H2/H3, unordered, numbered and checklist
operations in both offered input modes. Formatting changes complete affected
logical lines or selected rich list items; preserve unselected text, indentation,
inline suffixes, neighboring nodes and meaningful caret/selection position.
Changing marker kind removes the previous marker; numbered nonblank lines receive
sequential ordinals. Blank selected lines stay blank. An empty current line may
receive the existing recognizable selected placeholder.
Toolbar menus preserve editor-owned selection for mouse and keyboard use.

Plain writable collapsed Enter continues unordered/ordered/task items and retains
suffix text; tasks continue unchecked and ordinals increment across digit widths.
Empty items exit the list while retaining indentation. Active wikilink suggestions
retain Enter/Tab priority. Selection, modifiers, composition, readonly and fenced
Markdown pass through without continuation. Fence state respects delimiter kind,
opening length and valid closing length, including supported indented fences.

Rich tasks use native noneditable checkbox controls within editable list items.
Checkbox property changes must survive recording and an unrelated rich body edit.
Rich task Enter splits only the selected item, preserves inline suffix nodes,
creates an unchecked task and exits empty items without deleting neighbors.
Ordinary rich lists retain native continuation where verified.

Markdown/rich conversion preserves ordered starts, explicit ordinal discontinuity,
checked/unchecked tasks, supported nested list structure and task identity comments
through an actual edited roundtrip. Existing URL escaping/sanitization remains.
This repair does not promise general Markdown fidelity or expand unsupported loose
lists, lazy continuations or fenced-rich conversion.

## NE03 outline and local scrolling

Extract all headings before filtering so original offsets and duplicate slug IDs
remain stable. Filter trimmed exact title headings and case-insensitive literal
Table of contents; show the outline only with at least three meaningful headings.
Reuse the house disclosure and one neutral preference. Six headings default open;
seven default closed; remembered preference wins. Expanded outline is bounded and
independently scrollable, with accessible disclosure relationship and count.

Keep existing viewport height-cap arithmetic authoritative for actual editor and
paired Split preview. The combined task panel and Markdown preview have one Split
scroll owner. Header, Save and normal toolbar remain reachable at required widths.
TOC jumps from Preview first mount Split, then focus with preventScroll and reveal
the actual wrapped textarea offset or current rich heading within its own scroller.
Window and editor-body scrolling remain unchanged; edited/duplicate headings still
target the correct current heading. No newline-count approximation for wrapping.

Split scrolling mirrors the clamped source range ratio in either direction,
handles zero/unequal ranges without oscillation and resets across note/view/input
changes. Disclosure, TOC and scrolling do not mark content dirty or trigger saves.

## Preservation and explicit exclusions

Preserve a526 preview/code/Mermaid/source-jump work and b504 protected native print
work plus root's six dirty print paths; apply reviewed hunks against current bytes.
Keep ordered/task/nested/rich roundtrip, suggestion, IME, fence, readonly and title
TOC semantics. Shared CollapsibleSection changes cover existing Sidebar consumers.
The first-save selectedId Split-to-Edit reset is characterization only, expected
BASE PASS; it remains unchanged and proves no historical mobile or print cause.

## Finite evidence and release boundary

First BASE RED installs only tests that load existing exports and fail assertions
on current behavior: edited ordered/nested/task roundtrip and explicit ordinal
preservation. Do not import new helpers or dereference missing task controls first.
Capture failures before implementation; new-helper tests follow source release.
Use existing component/DOM/hook suites for conversion, toolbar, continuation,
outline and scroll arithmetic; preserve existing owning regressions and URL tests.

Boundary coverage within these owners: six versus seven disclosure defaults;
renamed rich heading and duplicate TOC targets after actual input; rich composing,
modified and actual editorDisabled/readOnly authority Enter fallthrough; keyboard menu selection preservation;
selected nested-item changes leave unselected parent/sibling/task content intact.
Run both existing Sidebar smoke owners: expanded-results and stage46 list-error-count.
No synthetic native input, force, retries, deadline increases or weakened assertions.

After complete NE03+NE06 deterministic evidence and normal owning type checks,
freeze production bytes and run both normal client builds. After finite native
evidence, obtain one independent actual-diff and full-evidence review before commit.
Use fresh normal artifacts for native WebUI and extension options flows, including
narrow wrapped TOC targeting, bounded controls, bidirectional scrolling and actual
rich checkbox/Enter/edit/save/reopen semantics. Retain failure traces via V2.
Product changes after native evidence require fresh normal builds; no intermediate
build replay. No syntax/test/build/native success is claimed by this proposal.
