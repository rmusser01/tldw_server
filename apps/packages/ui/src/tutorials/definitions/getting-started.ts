/**
 * Getting Started Tutorial
 * Shown on the home page after first onboarding completion.
 * Guides the user through the three main actions: Chat, Knowledge, and Analysis.
 */

import { Rocket } from "lucide-react"
import type { TutorialDefinition } from "../registry"

const gettingStarted: TutorialDefinition = {
  id: "getting-started",
  routePattern: "/",
  labelKey: "tutorials:gettingStarted.label",
  labelFallback: "Getting Started",
  descriptionKey: "tutorials:gettingStarted.description",
  descriptionFallback:
    "Choose where to start with Chat, Knowledge, or Media.",
  icon: Rocket,
  priority: 0,
  sequence: {
    nextTutorialId: "knowledge-basics",
    nextRoute: "/knowledge",
    nextLabelKey: "tutorials:gettingStarted.sequence.knowledgeLabel",
    nextLabelFallback: "Continue in Knowledge"
  },
  requireFirstTargetForPrompt: true,
  steps: [
    {
      disableBeacon: true,
      target: '[data-testid="companion-home-action-chat"]',
      titleKey: "tutorials:gettingStarted.chat.title",
      titleFallback: "Chat with AI",
      contentKey: "tutorials:gettingStarted.chat.content",
      contentFallback:
        "Open Chat and ask a question. If the model list is empty, configure a provider in Settings first.",
      placement: "bottom",
    },
    {
      optional: true,
      target: '[data-testid="companion-home-action-knowledge"]',
      titleKey: "tutorials:gettingStarted.knowledge.title",
      titleFallback: "Knowledge Base",
      contentKey: "tutorials:gettingStarted.knowledge.content",
      contentFallback:
        "Search and explore your ingested content. Upload documents, web pages, or media to build your personal knowledge base.",
      placement: "bottom",
    },
    {
      optional: true,
      target: '[data-testid="companion-home-action-media-multi"]',
      titleKey: "tutorials:gettingStarted.analysis.title",
      titleFallback: "Media Analysis",
      contentKey: "tutorials:gettingStarted.analysis.content",
      contentFallback:
        "Analyze and compare media files side by side. Great for reviewing transcripts, PDFs, and research materials.",
      placement: "bottom",
    }
  ],
}

export const gettingStartedTutorials: TutorialDefinition[] = [gettingStarted]
