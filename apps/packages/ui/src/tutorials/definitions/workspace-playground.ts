/**
 * Research Studio Tutorial Definitions
 */

import { FlaskConical } from "lucide-react"
import type { TutorialDefinition } from "../registry"

const workspacePlaygroundBasics: TutorialDefinition = {
  id: "workspace-playground-basics",
  routePattern: "/research-studio",
  labelKey: "tutorials:workspacePlayground.basics.label",
  labelFallback: "Research Studio Basics",
  descriptionKey: "tutorials:workspacePlayground.basics.description",
  descriptionFallback:
    "Learn the three-pane workspace flow: sources, chat, and studio outputs",
  icon: FlaskConical,
  priority: 1,
  steps: [
    {
      target: '[data-testid="workspace-header"]',
      titleKey: "tutorials:workspacePlayground.basics.headerTitle",
      titleFallback: "Workspace Header",
      contentKey: "tutorials:workspacePlayground.basics.headerContent",
      contentFallback:
        "Use the header to rename, switch, import, and manage workspaces while tracking system status.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '#workspace-sources-panel, [data-testid="workspace-drawer-left"]',
      titleKey: "tutorials:workspacePlayground.basics.sourcesTitle",
      titleFallback: "Sources Pane",
      contentKey: "tutorials:workspacePlayground.basics.sourcesContent",
      contentFallback:
        "Add and select sources here. Your selected source set controls what the chat and studio use.",
      placement: "right"
    },
    {
      target: '#workspace-main-content',
      titleKey: "tutorials:workspacePlayground.basics.chatTitle",
      titleFallback: "Chat Workspace",
      contentKey: "tutorials:workspacePlayground.basics.chatContent",
      contentFallback:
        "Ask questions against your selected sources and review grounded answers before generating outputs.",
      placement: "left"
    },
    {
      target: '#workspace-studio-panel, [data-testid="workspace-drawer-right"]',
      titleKey: "tutorials:workspacePlayground.basics.studioTitle",
      titleFallback: "Studio Outputs",
      contentKey: "tutorials:workspacePlayground.basics.studioContent",
      contentFallback:
        "Turn source context into summaries, reports, quizzes, flashcards, and other artifacts.",
      placement: "left"
    },
    {
      target: '[data-testid="workspace-workspaces-button"]',
      titleKey: "tutorials:workspacePlayground.basics.switcherTitle",
      titleFallback: "Workspace Switcher",
      contentKey: "tutorials:workspacePlayground.basics.switcherContent",
      contentFallback:
        "Open the workspace switcher to jump between saved workspaces and continue different projects.",
      placement: "bottom"
    }
  ]
}

export const workspacePlaygroundTutorials: TutorialDefinition[] = [
  workspacePlaygroundBasics
]
