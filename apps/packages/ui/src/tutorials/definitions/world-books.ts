/**
 * World Books Tutorial Definitions
 */

import { BookMarked } from "lucide-react"
import type { TutorialDefinition } from "../registry"

const worldBooksBasics: TutorialDefinition = {
  id: "world-books-basics",
  routePattern: "/world-books",
  labelKey: "tutorials:worldBooks.basics.label",
  labelFallback: "World Books Basics",
  descriptionKey: "tutorials:worldBooks.basics.description",
  descriptionFallback:
    "Create world books, filter by state, and manage reusable lore entries",
  icon: BookMarked,
  priority: 1,
  steps: [
    {
      target: '[data-testid="world-books-tutorial-shell"]',
      titleKey: "tutorials:worldBooks.basics.headerTitle",
      titleFallback: "World Books Workspace",
      contentKey: "tutorials:worldBooks.basics.headerContent",
      contentFallback:
        "World Books hold reusable lore and structured context that can be attached to characters and chat sessions.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="world-books-search-input"]',
      titleKey: "tutorials:worldBooks.basics.searchTitle",
      titleFallback: "Search World Books",
      contentKey: "tutorials:worldBooks.basics.searchContent",
      contentFallback:
        "Search by title, metadata, and content to quickly locate the right world book.",
      placement: "bottom"
    },
    {
      target:
        '[data-testid="world-books-enabled-filter"], [data-testid="world-books-attachment-filter"], [data-testid="world-books-filters-toggle"]',
      titleKey: "tutorials:worldBooks.basics.filtersTitle",
      titleFallback: "Status Filters",
      contentKey: "tutorials:worldBooks.basics.filtersContent",
      contentFallback:
        "Filter by status and attachment before editing or exporting. On smaller screens, open Filters to show these controls.",
      placement: "bottom"
    },
    {
      target:
        '[data-testid="world-books-table"], [data-testid="world-books-library"], [data-testid="world-books-empty-state"]',
      titleKey: "tutorials:worldBooks.basics.tableTitle",
      titleFallback: "World Book Library",
      contentKey: "tutorials:worldBooks.basics.tableContent",
      contentFallback:
        "Select a saved world book to manage entries and attachments. If the library is empty, create or import your first world book.",
      placement: "top"
    },
    {
      target:
        '[data-testid="world-books-new-button"], [data-testid="world-books-import-button"]',
      titleKey: "tutorials:worldBooks.basics.actionsTitle",
      titleFallback: "Create and Import",
      contentKey: "tutorials:worldBooks.basics.actionsContent",
      contentFallback:
        "Create a new world book or import an existing JSON bundle to seed your knowledge base.",
      placement: "left"
    }
  ]
}

export const worldBooksTutorials: TutorialDefinition[] = [worldBooksBasics]
