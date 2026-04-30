/**
 * Knowledge Tutorial Definitions
 */

import { BrainCircuit } from "lucide-react"
import type { TutorialDefinition } from "../registry"

const gettingStartedKnowledge: TutorialDefinition = {
  id: "getting-started-knowledge",
  routePattern: "/knowledge",
  labelKey: "tutorials:gettingStarted.knowledge.label",
  labelFallback: "Getting Started: Knowledge",
  descriptionKey: "tutorials:gettingStarted.knowledge.description",
  descriptionFallback:
    "Continue the getting started path by searching your first indexed sources",
  icon: BrainCircuit,
  prerequisites: ["getting-started"],
  priority: 0,
  sequence: {
    nextTutorialId: "document-workspace-basics",
    nextRoute: "/document-workspace",
    nextLabelKey: "tutorials:gettingStarted.sequence.documentWorkspaceLabel",
    nextLabelFallback: "Continue in Document Workspace"
  },
  steps: [
    {
      target: '[data-testid="knowledge-page-root"]',
      titleKey: "tutorials:gettingStarted.knowledge.workspaceTitle",
      titleFallback: "Search Your Knowledge Base",
      contentKey: "tutorials:gettingStarted.knowledge.workspaceContent",
      contentFallback:
        "Knowledge is where your ingested documents, notes, and media become searchable evidence for research questions.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: "#knowledge-search-input",
      titleKey: "tutorials:gettingStarted.knowledge.searchTitle",
      titleFallback: "Ask a Focused Question",
      contentKey: "tutorials:gettingStarted.knowledge.searchContent",
      contentFallback:
        "Start with a focused question. The retrieval pipeline will search across the sources you select.",
      placement: "bottom"
    },
    {
      target: "#knowledge-source-selector-toggle",
      titleKey: "tutorials:gettingStarted.knowledge.sourcesTitle",
      titleFallback: "Pick the Right Sources",
      contentKey: "tutorials:gettingStarted.knowledge.sourcesContent",
      contentFallback:
        "Use source groups to keep answers grounded in the documents or media that matter for the current task.",
      placement: "bottom"
    },
    {
      target: '[data-testid="knowledge-results-shell"]',
      titleKey: "tutorials:gettingStarted.knowledge.resultsTitle",
      titleFallback: "Review Answers and Evidence",
      contentKey: "tutorials:gettingStarted.knowledge.resultsContent",
      contentFallback:
        "After a search, use the results workspace to inspect answers, citations, and supporting source snippets.",
      placement: "left"
    }
  ]
}

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

export const knowledgeTutorials: TutorialDefinition[] = [
  gettingStartedKnowledge,
  knowledgeBasics
]
