/**
 * Knowledge Tutorial Definitions
 */

import { BrainCircuit } from "lucide-react"
import type { TutorialDefinition } from "../registry"

const knowledgeBasics: TutorialDefinition = {
  id: "knowledge-basics",
  routePattern: "/knowledge",
  labelKey: "tutorials:knowledge.basics.label",
  labelFallback: "Knowledge Basics",
  descriptionKey: "tutorials:knowledge.basics.description",
  descriptionFallback:
    "Search your indexed sources and review grounded answers with citations",
  icon: BrainCircuit,
  priority: 1,
  sequence: {
    nextTutorialId: "document-workspace-basics",
    nextRoute: "/document-workspace",
    nextLabelKey: "tutorials:gettingStarted.sequence.documentWorkspaceLabel",
    nextLabelFallback: "Continue in Document Workspace"
  },
  steps: [
    {
      target: "#knowledge-search-input",
      titleKey: "tutorials:knowledge.basics.searchTitle",
      titleFallback: "Ask a Question",
      contentKey: "tutorials:knowledge.basics.searchContent",
      contentFallback:
        "Enter a focused question about your docs, notes, or media to start retrieval.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: "#knowledge-source-selector-toggle",
      titleKey: "tutorials:knowledge.basics.sourcesTitle",
      titleFallback: "Choose Sources",
      contentKey: "tutorials:knowledge.basics.sourcesContent",
      contentFallback:
        "Select which source groups to query so answers stay scoped to what you care about.",
      placement: "bottom"
    },
    {
      target: '[data-testid="knowledge-search-shell"]',
      titleKey: "tutorials:knowledge.basics.contextTitle",
      titleFallback: "Search Context Bar",
      contentKey: "tutorials:knowledge.basics.contextContent",
      contentFallback:
        "Adjust retrieval preset and context options before running your search.",
      placement: "bottom"
    },
    {
      target:
        '[data-testid="knowledge-history-desktop-open"], [data-testid="knowledge-history-desktop-collapsed"], [data-testid="knowledge-history-mobile-open"]',
      titleKey: "tutorials:knowledge.basics.historyTitle",
      titleFallback: "History and Recovery",
      contentKey: "tutorials:knowledge.basics.historyContent",
      contentFallback:
        "Reuse prior threads and queries from history so you can continue investigations quickly.",
      placement: "right"
    },
    {
      target: '[data-testid="knowledge-results-shell"]',
      titleKey: "tutorials:knowledge.basics.resultsTitle",
      titleFallback: "Results Workspace",
      contentKey: "tutorials:knowledge.basics.resultsContent",
      contentFallback:
        "Review answer output, cited sources, and evidence once a search completes.",
      placement: "left"
    }
  ]
}

export const knowledgeTutorials: TutorialDefinition[] = [knowledgeBasics]
