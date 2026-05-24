/**
 * Research Workspace Tutorial Definitions
 */

import { FlaskConical } from "lucide-react"
import type { TutorialDefinition } from "../registry"

const researchWorkspaceBasics: TutorialDefinition = {
  id: "research-workspace-basics",
  routePattern: "/research-workspace",
  labelKey: "tutorials:researchWorkspace.basics.label",
  labelFallback: "Research Workspace Basics",
  descriptionKey: "tutorials:researchWorkspace.basics.description",
  descriptionFallback:
    "Learn the three-pane workspace flow: sources, chat, and studio outputs",
  icon: FlaskConical,
  priority: 1,
  steps: [
    {
      target: '[data-testid="workspace-header"]',
      titleKey: "tutorials:researchWorkspace.basics.headerTitle",
      titleFallback: "Workspace Header",
      contentKey: "tutorials:researchWorkspace.basics.headerContent",
      contentFallback:
        "Use the header to rename, switch, import, and manage workspaces while tracking system status.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '#workspace-sources-panel, [data-testid="workspace-drawer-left"]',
      titleKey: "tutorials:researchWorkspace.basics.sourcesTitle",
      titleFallback: "Sources Pane",
      contentKey: "tutorials:researchWorkspace.basics.sourcesContent",
      contentFallback:
        "Add and select sources here. Your selected source set controls what the chat and studio use.",
      placement: "right"
    },
    {
      target: '#workspace-main-content',
      titleKey: "tutorials:researchWorkspace.basics.chatTitle",
      titleFallback: "Chat Workspace",
      contentKey: "tutorials:researchWorkspace.basics.chatContent",
      contentFallback:
        "Ask questions against your selected sources and review grounded answers before generating outputs.",
      placement: "left"
    },
    {
      target: '#workspace-studio-panel, [data-testid="workspace-drawer-right"]',
      titleKey: "tutorials:researchWorkspace.basics.studioTitle",
      titleFallback: "Studio Outputs",
      contentKey: "tutorials:researchWorkspace.basics.studioContent",
      contentFallback:
        "Turn source context into summaries, reports, quizzes, flashcards, and other artifacts.",
      placement: "left"
    },
    {
      target: '[data-testid="workspace-workspaces-button"]',
      titleKey: "tutorials:researchWorkspace.basics.switcherTitle",
      titleFallback: "Workspace Switcher",
      contentKey: "tutorials:researchWorkspace.basics.switcherContent",
      contentFallback:
        "Open the workspace switcher to jump between saved workspaces and continue different projects.",
      placement: "bottom"
    }
  ]
}

export const researchWorkspaceTutorials: TutorialDefinition[] = [
  researchWorkspaceBasics
]
