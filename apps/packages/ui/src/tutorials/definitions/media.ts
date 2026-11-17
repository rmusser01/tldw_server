/**
 * Media Tutorial Definitions
 */

import { BookText } from "lucide-react"
import type { TutorialDefinition } from "../registry"

const mediaBasics: TutorialDefinition = {
  id: "media-basics",
  routePattern: "/media",
  labelKey: "tutorials:media.basics.label",
  labelFallback: "Media Basics",
  descriptionKey: "tutorials:media.basics.description",
  descriptionFallback:
    "Open a saved item and inspect its content.",
  icon: BookText,
  priority: 1,
  requireFirstTargetForPrompt: true,
  steps: [
    {
      target: '[data-testid="media-results-list"]',
      titleKey: "tutorials:media.basics.resultsTitle",
      titleFallback: "Results List",
      contentKey: "tutorials:media.basics.resultsContent",
      contentFallback:
        "Select an item to inspect it. If this list is empty, add a file or URL with Add media.",
      placement: "right",
      disableBeacon: true
    },
    {
      optional: true,
      target:
        '[data-testid="content-scroll-container"], [data-testid="content-viewer-empty"]',
      titleKey: "tutorials:media.basics.viewerTitle",
      titleFallback: "Content Viewer",
      contentKey: "tutorials:media.basics.viewerContent",
      contentFallback:
        "Inspect selected content, metadata, and navigation details in the main viewer pane.",
      placement: "left",
      disableBeacon: true
    },
    {
      optional: true,
      target: '[data-testid="media-search-input"]',
      titleKey: "tutorials:media.basics.searchTitle",
      titleFallback: "Search Media",
      contentKey: "tutorials:media.basics.searchContent",
      contentFallback:
        "Start with a query here to find items by title or content.",
      placement: "bottom",
      disableBeacon: true
    }
  ]
}

export const mediaTutorials: TutorialDefinition[] = [mediaBasics]
