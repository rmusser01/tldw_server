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
    "A quick tour of the main features to help you get the most out of tldw",
  icon: Rocket,
  priority: 0,
  sequence: {
    nextTutorialId: "knowledge-basics",
    nextRoute: "/knowledge",
    nextLabelKey: "tutorials:gettingStarted.sequence.knowledgeLabel",
    nextLabelFallback: "Continue in Knowledge"
  },
  steps: [
    {
      target: '[data-testid="companion-home-shell"]',
      titleKey: "tutorials:gettingStarted.welcome.title",
      titleFallback: "Welcome to tldw!",
      contentKey: "tutorials:gettingStarted.welcome.content",
      contentFallback:
        "This is your home dashboard. From here you can access all of tldw's features. Let's take a quick tour.",
      placement: "bottom",
      disableBeacon: true,
    },
    {
      target: '[data-testid="companion-home-action-chat"]',
      titleKey: "tutorials:gettingStarted.chat.title",
      titleFallback: "Chat with AI",
      contentKey: "tutorials:gettingStarted.chat.content",
      contentFallback:
        "Chat with 16+ AI providers using an OpenAI-compatible interface. You'll need to configure at least one LLM provider in Settings first.",
      placement: "bottom",
    },
    {
      target: '[data-testid="companion-home-action-knowledge"]',
      titleKey: "tutorials:gettingStarted.knowledge.title",
      titleFallback: "Knowledge Base",
      contentKey: "tutorials:gettingStarted.knowledge.content",
      contentFallback:
        "Search and explore your ingested content. Upload documents, web pages, or media to build your personal knowledge base.",
      placement: "bottom",
    },
    {
      target: '[data-testid="companion-home-action-media-multi"]',
      titleKey: "tutorials:gettingStarted.analysis.title",
      titleFallback: "Media Analysis",
      contentKey: "tutorials:gettingStarted.analysis.content",
      contentFallback:
        "Analyze and compare media files side by side. Great for reviewing transcripts, PDFs, and research materials.",
      placement: "bottom",
    },
    {
      target: '[data-testid="companion-home-shell"]',
      titleKey: "tutorials:gettingStarted.nextStep.title",
      titleFallback: "Next: Configure a Provider",
      contentKey: "tutorials:gettingStarted.nextStep.content",
      contentFallback:
        "To start using Chat, go to Settings > tldw Server and add an LLM provider API key (e.g., OpenAI, Anthropic). Then come back and open Chat!",
      placement: "bottom",
    },
  ],
}

export const gettingStartedTutorials: TutorialDefinition[] = [gettingStarted]
