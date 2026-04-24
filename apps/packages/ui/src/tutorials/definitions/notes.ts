/**
 * Notes Tutorial Definitions
 */

import { StickyNote } from "lucide-react"
import type { TutorialDefinition } from "../registry"

const notesBasics: TutorialDefinition = {
  id: "notes-basics",
  routePattern: "/notes",
  labelKey: "tutorials:notes.basics.label",
  labelFallback: "Notes Basics",
  descriptionKey: "tutorials:notes.basics.description",
  descriptionFallback:
    "Organize notes, apply filters, and edit content with split list/editor workflow",
  icon: StickyNote,
  priority: 1,
  steps: [
    {
      target: '[data-testid="notes-list-region"]',
      titleKey: "tutorials:notes.basics.listTitle",
      titleFallback: "Notes List",
      contentKey: "tutorials:notes.basics.listContent",
      contentFallback:
        "Browse and select notes from the sidebar to load them in the editor.",
      placement: "right",
      disableBeacon: true
    },
    {
      target: '[data-testid="notes-mode-active"], [data-testid="notes-mode-trash"]',
      titleKey: "tutorials:notes.basics.modeTitle",
      titleFallback: "Active and Trash Views",
      contentKey: "tutorials:notes.basics.modeContent",
      contentFallback:
        "Switch between active notes and trash to recover or permanently remove entries.",
      placement: "bottom"
    },
    {
      target: '[data-testid="notes-sort-select"], [data-testid="notes-notebook-select"]',
      titleKey: "tutorials:notes.basics.filtersTitle",
      titleFallback: "Sort and Filter",
      contentKey: "tutorials:notes.basics.filtersContent",
      contentFallback:
        "Sort by date/title and use saved filter or tag filters to narrow your working set.",
      placement: "bottom"
    },
    {
      target: '[data-testid="notes-editor-region"]',
      titleKey: "tutorials:notes.basics.editorTitle",
      titleFallback: "Editor Workspace",
      contentKey: "tutorials:notes.basics.editorContent",
      contentFallback:
        "Edit note content, references, and metadata in the main editor panel.",
      placement: "left"
    },
    {
      target: '[data-testid="notes-save-button"]',
      titleKey: "tutorials:notes.basics.saveTitle",
      titleFallback: "Save and Actions",
      contentKey: "tutorials:notes.basics.saveContent",
      contentFallback:
        "Save updates, duplicate notes, export content, and run helper actions from the header.",
      placement: "bottom"
    },
    {
      target: '[data-testid="notes-keywords-editor"]',
      titleKey: "tutorials:notes.basics.tagsTitle",
      titleFallback: "Tags",
      contentKey: "tutorials:notes.basics.tagsContent",
      contentFallback:
        "Add tags to organize your notes. Use the filter in the sidebar to find notes by tag.",
      placement: "bottom"
    },
    {
      target: '[data-testid="notes-section-connections"]',
      titleKey: "tutorials:notes.basics.connectionsTitle",
      titleFallback: "Note Connections",
      contentKey: "tutorials:notes.basics.connectionsContent",
      contentFallback:
        "See how notes relate to each other. Type [[ in a note to create a link, or add manual connections here.",
      placement: "left"
    },
    {
      target: '[data-testid="notes-section-organize"]',
      titleKey: "tutorials:notes.basics.organizeTitle",
      titleFallback: "Collections and Filters",
      contentKey: "tutorials:notes.basics.organizeContent",
      contentFallback:
        "Group notes into collections, or save tag filters for quick access to related notes.",
      placement: "bottom"
    }
  ]
}

export const notesTutorials: TutorialDefinition[] = [notesBasics]
