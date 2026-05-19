/**
 * Watchlists Tutorial Definitions
 */

import { Rss } from "lucide-react"
import type { TutorialDefinition } from "../registry"

const watchlistsBasics: TutorialDefinition = {
  id: "watchlists-basics",
  routePattern: "/watchlists",
  labelKey: "tutorials:watchlists.basics.label",
  labelFallback: "Watchlists Basics",
  descriptionKey: "tutorials:watchlists.basics.description",
  descriptionFallback:
    "Learn how to add feeds, create monitors, and review collected articles",
  icon: Rss,
  priority: 1,
  steps: [
    {
      target: '[data-testid="watchlists-overview-onboarding-path-beginner"], [data-testid="watchlists-overview-cta-guided-setup"]',
      titleKey: "tutorials:watchlists.basics.setupTitle",
      titleFallback: "Quick Setup",
      contentKey: "tutorials:watchlists.basics.setupContent",
      contentFallback:
        "Start here to add your first feed and create a monitor. The guided setup walks you through it step by step.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="watchlists-overview-cta-add-feed"]',
      titleKey: "tutorials:watchlists.basics.feedsTitle",
      titleFallback: "Add Feeds",
      contentKey: "tutorials:watchlists.basics.feedsContent",
      contentFallback:
        "Feeds are sources like RSS feeds or websites that your monitors check for new content.",
      placement: "bottom"
    },
    {
      target: '[data-testid="watchlists-overview-cta-create-monitor"]',
      titleKey: "tutorials:watchlists.basics.monitorsTitle",
      titleFallback: "Create Monitors",
      contentKey: "tutorials:watchlists.basics.monitorsContent",
      contentFallback:
        "Monitors run on a schedule to fetch and process content from your feeds. Choose a preset schedule like daily or hourly.",
      placement: "bottom"
    },
    {
      target: '[data-testid="watchlists-items-list"], .ant-tabs-tab',
      titleKey: "tutorials:watchlists.basics.articlesTitle",
      titleFallback: "Review Articles",
      contentKey: "tutorials:watchlists.basics.articlesContent",
      contentFallback:
        "Articles are collected content from successful monitor runs. Switch to the Articles tab to browse and read them.",
      placement: "top"
    }
  ]
}

export const watchlistsTutorials: TutorialDefinition[] = [watchlistsBasics]
