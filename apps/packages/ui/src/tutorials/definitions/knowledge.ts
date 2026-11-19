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
  requireFirstTargetForPrompt: true,
  steps: [
    {
      target: "#knowledge-search-input",
      titleKey: "tutorials:knowledge.basics.searchTitle",
      titleFallback: "Ask a Question",
      contentKey: "tutorials:knowledge.basics.searchContent",
      contentFallback:
        "Ask a question your saved sources can answer, such as “What decisions were made in the last meeting?”",
      placement: "bottom",
      disableBeacon: true
    },
    {
      optional: true,
      target: "#knowledge-source-selector-toggle",
      titleKey: "tutorials:knowledge.basics.sourcesTitle",
      titleFallback: "Choose Sources",
      contentKey: "tutorials:knowledge.basics.sourcesContent",
      contentFallback:
        "Select which source groups to query so answers stay scoped to what you care about.",
      placement: "bottom"
    },
    {
      optional: true,
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
