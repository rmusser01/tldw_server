# World Books UX Progressive Disclosure Redesign

**Status:** Implementation complete (2026-04-10)
**Implementation plan:** `Docs/Plans/2026-04-09-world-books-ux-progressive-disclosure-implementation-plan.md`
**Branch:** `feat/world-books-ux-progressive-disclosure`

**Date:** 2026-04-09
**Page:** `/world-books`
**Approach:** Progressive Disclosure Redesign (Approach B)
**NNG Baseline Score:** ~2.85/5

## Problem Statement

The `/world-books` page is functional for power users but overwhelming and opaque for newcomers. A 2,602-line monolith component renders 8+ modals/drawers. The main view shows too much at once: search + 3 filter selects + 6 equally-weighted toolbar buttons + a debug link banner + a dense table with 8 columns and 7 action buttons per row.

### NNG Heuristic Scores (Current)

| # | Heuristic | Score | Key Issues |
|---|-----------|-------|------------|
| 1 | Visibility of system status | 3.5/5 | Good skeletons/banners; lazy attachment column unexplained |
| 2 | Match between system and real world | 2/5 | Jargon-heavy: "scan depth", "token budget", "recursive scanning" |
| 3 | User control and freedom | 4/5 | Strong: undo on delete, discard confirmations, conflict recovery |
| 4 | Consistency and standards | 3/5 | Good adaptive desktop/mobile; icon meanings unclear (Link2 for attach?) |
| 5 | Error prevention | 3.5/5 | Duplicate name validation; no budget-exceeded warning during entry creation |
| 6 | Recognition rather than recall | 2/5 | 6 toolbar buttons with no grouping or visual hierarchy |
| 7 | Flexibility and efficiency of use | 3.5/5 | Bulk actions, keyboard matrix nav, templates; no shortcuts |
| 8 | Aesthetic and minimalist design | 1.5/5 | Biggest problem -- everything visible at once, 8+ overlay surfaces |
| 9 | Help users recover from errors | 3.5/5 | Good error messages, version conflict recovery |
| 10 | Help and documentation | 2/5 | Tooltips on advanced settings only; no onboarding |

## Design Sections

### 1. First-Run Experience & Empty State

**Replace** the current thin empty state ("No world books yet" + one subtitle line + create button) with:

- **3-step visual flow:** Create a world book -> Add entries with keywords -> Attach to a character or chat. Explains the concept before asking users to act.
- **Concrete example:** "Entry with keyword 'magic system' -> user says 'tell me about the magic system' -> AI receives your lore entry as background context."
- **Template quick-starts surfaced directly** in the empty state (Fantasy Setting, Sci-Fi Lore, Product Knowledge Base) -- currently these are hidden inside the create modal.
- **Import as secondary action** for users migrating from SillyTavern/Kobold.
- Once the user has at least one world book, this collapses to a dismissible tip banner, then disappears.

**NNG impact:** #2 (match real world), #10 (help and documentation).

### 2. Toolbar Reorganization & Visual Hierarchy

**Replace** the flat row of 6 equally-weighted buttons with:

- **"New World Book"** as the only primary button -- clear primary action.
- **"Tools" dropdown** grouping all secondary actions:
  - Analysis: Test Matching, Relationship Matrix, Global Statistics
  - I/O: Import JSON, Export All, Export Selected (when selection active)
  - Debug: Chat Injection Panel (moved from always-visible banner)
- **Filters remain visible:** Search + Status dropdown + Attachment dropdown.
- **Responsive:** On mobile, search goes full-width, filters collapse into a "Filters" popover, Tools + New are the only two buttons.

**NNG impact:** #6 (recognition), #8 (minimalist design).

### 3. Table Simplification & Per-Row Actions

**Replace** the 8-column table with 7 action buttons per row with:

- **Remove** static BookOpen icon column (no information).
- **Merge** Name + Description into one column (description as muted second line).
- **Remove** "Attached To" and "Budget" from the table (visible in detail panel).
- **Columns:** Checkbox, Name+Desc, Entries count, Status, Last Modified, Actions.
- **Only 2 visible action buttons per row:** Edit (pen) + overflow menu (three dots).
- **Overflow menu:** Manage Entries, Duplicate, Quick Attach, Export JSON, Statistics, separator, Delete (danger).
- **Row click** opens the detail panel (Section 4) instead of expanding an entry preview.

**NNG impact:** #8 (minimalist design), #6 (recognition).

### 4. Two-Panel Layout (List + Detail)

**Replace** the modal/drawer pattern with a persistent two-panel layout:

- **List panel (left, ~35%):** Simplified table from Section 3.
- **Detail panel (right, ~65%):** Tabbed interface replacing multiple modals:
  - **Entries tab** (default): Full entry manager with add, edit, bulk-add, filter, search.
  - **Attachments tab:** Attached characters with attach/detach controls.
  - **Stats tab:** Budget utilization, entry counts, keyword analysis.
  - **Settings tab:** Name, description, scan depth, token budget, recursive scanning.
- **Summary bar** at top of detail panel: status, entry count, budget, attachments, last modified.
- **No-selection state:** Prompt with lightweight version of the 3-step visual.

**What stays as modals:** Create World Book, Relationship Matrix, Test Matching, Import. These genuinely need isolated contexts.

**Modals eliminated:** Edit modal, Statistics modal, Quick Attach modal, entries Drawer -- all replaced by detail panel tabs.

**NNG impact:** #8 (minimalist design), #1 (visibility), #6 (recognition), #7 (flexibility).

### 5. Human-Readable Labels & Progressive Disclosure for Settings

**Two-tier labeling** with a "Show technical labels" toggle (persisted in localStorage):

| Current (technical) | Friendly default | Technical toggle adds |
|---|---|---|
| Scan Depth | Messages to search | `scan_depth: 1-20` |
| Token Budget | Context size limit | `token_budget: 50-5000 (~4 chars = 1 token)` |
| Recursive Scanning | Chain matching | `recursive_scanning: max depth N` |

- Friendly labels include inline descriptions explaining the *effect*, not the mechanism.
- Budget utilization bar shown inline in settings so users see impact immediately.
- "Advanced Settings" disclosure relabeled to "Matching & Budget" (describes what it controls, removes intimidation).
- Entry matching options: "Regex match" -> "Pattern matching (regex)" with one-line explainer.

**NNG impact:** #2 (match real world), #7 (flexibility).

### 6. Entry Creation & Budget Feedback

- **Persistent budget bar** at the top of the entries tab, always visible.
- **Per-entry token estimate** shown inline (~18 tokens) so users understand relative cost.
- **Live "budget after save" projection** in the add/edit form. As you type, the budget bar shows projected state after saving.
- **Soft warning (not blocker)** when an entry would push usage over budget. Entry can still be saved, but user understands the trade-off.
- **Filter tabs with counts** (All 12 | Enabled 9 | Disabled 3) replace filter preset dropdowns.
- **Inline entry cards** show keywords, content preview, token estimate, matching mode, status, with Edit/Delete per card.
- **"Matching options" collapsed by default** on the add form. Keywords, Content, Priority are the three visible fields for 90% of entries.

**NNG impact:** #1 (visibility), #5 (error prevention), #6 (recognition).

### 7. Mobile & Responsive Behavior

Three breakpoints with distinct layouts:

- **Desktop (lg+):** Two-panel side-by-side as Section 4.
- **Tablet (md):** Stacked layout. Collapsible list at top (accordion), full-width detail panel below.
- **Mobile (sm):** Navigation stack. List view is default. Tapping a world book pushes a full-width detail view with `<- World Books` back button. "New" as FAB in bottom-right.

Toolbar adapts:
- Desktop: Search + filter selects + Tools dropdown + New button.
- Tablet: Search full-width, filters in popover, Tools + New.
- Mobile: Search full-width, icon buttons for filters and Tools, New as FAB.

Matrix modal: Always list view on mobile (existing `useAttachmentListView` threshold).
Tabs: Scroll horizontally on mobile with fade indicator.

**NNG impact:** #3 (user control), #4 (consistency with platform norms).

### 8. Accessibility & Interaction Polish

**Focus management:**
- Clicking a world book moves focus to detail panel heading.
- `Escape` in detail panel returns focus to selected list row.

**Landmark roles:**
- `<nav aria-label="World books list">` for left panel.
- `<main aria-label="World book detail">` for right panel.

**Aria improvements:**
- Action buttons: `aria-label="Edit Fantasy Lore"` (specific, not generic).
- Budget bar: `role="meter" aria-valuenow="285" aria-valuemax="700"`.
- Filter tabs: `role="tablist"` with `aria-selected`.

**Status indicators:** Add icons alongside color tags (circle-check for enabled, circle-pause for disabled) for color-blind users.

**Keyboard shortcuts (opt-in, off by default):**
- `n` -- New world book
- `e` -- Edit selected
- `Enter` -- Open entries for selected
- `Delete` -- Trigger delete with undo timer
- `?` -- Show shortcut cheat sheet

**Reduced motion:** Pulse animations replaced with static highlights when `prefers-reduced-motion` is set.

## Files Affected

### Frontend (primary changes)
- `apps/packages/ui/src/components/Option/WorldBooks/Manager.tsx` -- Decompose 2,602-line monolith into panel components
- `apps/packages/ui/src/components/Option/WorldBooks/WorldBooksWorkspace.tsx` -- Updated layout wrapper
- `apps/packages/ui/src/components/Option/WorldBooks/WorldBookForm.tsx` -- Two-tier labels
- `apps/packages/ui/src/components/Option/WorldBooks/WorldBookEntryManager.tsx` -- Budget feedback, filter tabs
- New: `WorldBookListPanel.tsx` -- Left panel list
- New: `WorldBookDetailPanel.tsx` -- Right panel with tabs
- New: `WorldBookEmptyState.tsx` -- First-run experience
- New: `WorldBookToolbar.tsx` -- Reorganized toolbar
- New: `WorldBookBudgetBar.tsx` -- Reusable budget indicator
- Utility files: Minimal changes (label mappings, constants)

### Backend
- No backend changes required. All changes are frontend-only.

## Preserved Behaviors

These patterns are well-designed and carry forward unchanged:
- 10-second undo timer on delete
- Discard-changes confirmations on dirty forms
- Version conflict detection and recovery (Load latest / Reapply my edits)
- Optimistic locking via version field
- Accessible switch text ("On"/"Off")
- Bulk selection and bulk actions bar
- Template-based creation
- Import format detection (tldw, SillyTavern, Kobold)
