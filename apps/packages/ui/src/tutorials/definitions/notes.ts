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
  descriptionKey: "tutorials:notes.basics.quickStartDescription",
  descriptionFallback: "Learn to create, edit, and save a note.",
  icon: StickyNote,
  priority: 1,
  requireFirstTargetForPrompt: true,
  steps: [
    {
      target:
        '[data-testid="notes-new-note-button"], [data-testid="notes-editor-empty-create"], [data-testid="notes-mobile-open-list-button"], [data-testid="notes-desktop-sidebar-toggle"][aria-expanded="false"]',
      titleKey: "tutorials:notes.basics.createTitle",
      titleFallback: "Create a note",
      contentKey: "tutorials:notes.basics.quickStartCreate",
      contentFallback:
        "Choose + or Create note. If the notes list is closed, open it first. Add a title and text, then choose Save.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="notes-title-row"]',
      titleKey: "tutorials:notes.basics.quickStartEditTitle",
      titleFallback: "Add a title and text",
      contentKey: "tutorials:notes.basics.quickStartEdit",
      contentFallback:
        "Give your note a title you can recognize later, then add your text in the editor below.",
      placement: "bottom",
      optional: true
    },
    {
      target: '[data-testid="notes-save-button"]',
      titleKey: "tutorials:notes.basics.quickStartSaveTitle",
      titleFallback: "Save your note",
      contentKey: "tutorials:notes.basics.quickStartSave",
      contentFallback: "Choose Save to keep your note. You can also use Ctrl+S or ⌘S.",
      placement: "bottom",
      optional: true
    }
  ]
}

export const notesTutorials: TutorialDefinition[] = [notesBasics]
