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
    "Search, filter, inspect, and manage ingested media in the library",
  icon: BookText,
  priority: 1,
  steps: [
    {
      target: '[data-testid="media-search-input"]',
      titleKey: "tutorials:media.basics.searchTitle",
      titleFallback: "Search Media",
      contentKey: "tutorials:media.basics.searchContent",
      contentFallback:
        "Start with a query here to find items by title or content.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '#media-search-panel',
      titleKey: "tutorials:media.basics.filtersTitle",
      titleFallback: "Filter and Scope",
      contentKey: "tutorials:media.basics.filtersContent",
      contentFallback:
        "Refine by media types, metadata, keywords, and collection filters before running searches.",
      placement: "bottom"
    },
    {
      target: '[data-testid="media-search-submit"]',
      titleKey: "tutorials:media.basics.submitTitle",
      titleFallback: "Run Search",
      contentKey: "tutorials:media.basics.submitContent",
      contentFallback:
        "Submit your query after adjusting filters to refresh the result set.",
      placement: "bottom"
    },
    {
      target: '[data-testid="media-results-list"]',
      titleKey: "tutorials:media.basics.resultsTitle",
      titleFallback: "Results List",
      contentKey: "tutorials:media.basics.resultsContent",
      contentFallback:
        "Review matched items, select one, and navigate pages to inspect more content.",
      placement: "right",
      disableBeacon: true
    },
    {
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
      target: '[data-testid="media-library-tools-toggle"]',
      titleKey: "tutorials:media.basics.toolsTitle",
      titleFallback: "Library Tools",
      contentKey: "tutorials:media.basics.toolsContent",
      contentFallback:
        "Open library tools for ingest job monitoring and storage/usage statistics.",
      placement: "top"
    }
  ]
}

export const mediaTutorials: TutorialDefinition[] = [mediaBasics]
