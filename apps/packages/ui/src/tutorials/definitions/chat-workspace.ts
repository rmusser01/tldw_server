import { SquareTerminal } from "lucide-react"
import type { TutorialDefinition } from "../registry"

export const chatWorkspaceTutorials: TutorialDefinition[] = [
  {
    id: "chat-workspace-basics",
    routePattern: "/chat-workspace",
    labelKey: "tutorials:chatWorkspace.basics.label",
    labelFallback: "Chat Workspace Basics",
    descriptionKey: "tutorials:chatWorkspace.basics.description",
    descriptionFallback: "Prepare workspace chat, stage sources, and check send status",
    icon: SquareTerminal,
    priority: 1,
    steps: [
      {
        target: '[data-testid="chat-workspace-console"]',
        titleKey: "tutorials:chatWorkspace.basics.scopeTitle",
        titleFallback: "Workspace and Assistant",
        contentKey: "tutorials:chatWorkspace.basics.scopeContent",
        contentFallback: "Use the active workspace from Research Workspace and choose a model in Chat before returning here. Inspector reports the workspace, model, and persona, including inherited defaults. On narrow screens, switch between Chat, Sources, and Inspector.",
        disableBeacon: true
      },
      {
        target: '[data-testid="chat-workspace-console"]',
        titleKey: "tutorials:chatWorkspace.basics.sourcesTitle",
        titleFallback: "Review Staged Sources",
        contentKey: "tutorials:chatWorkspace.basics.sourcesContent",
        contentFallback: "Browse opens a bounded read-only preview without staging or sending. Stage adds a ready source to the next send; review it in Chat first. Ready media is sent as retrieval context. Insert summary instead adds a text list of source names and clears staging, not the source contents."
      },
      {
        target: '[aria-label="Chat workspace status"]',
        titleKey: "tutorials:chatWorkspace.basics.statusTitle",
        titleFallback: "Send Status and Recovery",
        contentKey: "tutorials:chatWorkspace.basics.statusContent",
        contentFallback: "The status strip and Inspector show readiness, streaming, and errors. Offline or failed sends keep the current draft and staged context for retry while this page stays open. Stop cancels generation. Switching workspaces or leaving this page can discard unsent work.",
        placement: "top"
      }
    ]
  }
]
