/**
 * Document Workspace Tutorial Definitions
 */

import { FileText } from "lucide-react"
import type { TutorialDefinition } from "../registry"

const documentWorkspaceBasics: TutorialDefinition = {
  id: "document-workspace-basics",
  routePattern: "/document-workspace",
  labelKey: "tutorials:documentWorkspace.basics.label",
  labelFallback: "Document Workspace Basics",
  descriptionKey: "tutorials:documentWorkspace.basics.description",
  descriptionFallback:
    "Read, navigate, annotate, and ask questions about documents in one workspace",
  icon: FileText,
  priority: 1,
  steps: [
    {
      target: '[data-testid="document-workspace-root"]',
      titleKey: "tutorials:documentWorkspace.basics.workspaceTitle",
      titleFallback: "Document Workspace",
      contentKey: "tutorials:documentWorkspace.basics.workspaceContent",
      contentFallback:
        "Use this workspace to read source documents while keeping navigation, notes, citations, quizzes, and chat close at hand.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="document-open-picker-button"]',
      titleKey: "tutorials:documentWorkspace.basics.openDocumentTitle",
      titleFallback: "Open or Select a Document",
      contentKey: "tutorials:documentWorkspace.basics.openDocumentContent",
      contentFallback:
        "Use the open document button to pick a saved source or upload a new PDF or EPUB for study.",
      placement: "bottom"
    },
    {
      target: '[data-testid="document-workspace-toggle-left"]',
      titleKey: "tutorials:documentWorkspace.basics.leftPaneTitle",
      titleFallback: "Document Navigation",
      contentKey: "tutorials:documentWorkspace.basics.leftPaneContent",
      contentFallback:
        "Open the left pane for contents, page thumbnails, references, document info, and quick insights.",
      placement: "bottom"
    },
    {
      target: '[data-testid="document-workspace-toggle-right"]',
      titleKey: "tutorials:documentWorkspace.basics.rightPaneTitle",
      titleFallback: "Document Tools",
      contentKey: "tutorials:documentWorkspace.basics.rightPaneContent",
      contentFallback:
        "Open the right pane for document chat, highlights, citations, and generated quizzes.",
      placement: "bottom"
    },
    {
      target: '[data-testid="document-viewer"]',
      titleKey: "tutorials:documentWorkspace.basics.viewerTitle",
      titleFallback: "Read in Context",
      contentKey: "tutorials:documentWorkspace.basics.viewerContent",
      contentFallback:
        "The center viewer keeps your active PDF or EPUB in focus while side panels add research context.",
      placement: "bottom"
    },
    {
      target: '[data-testid="document-navigation"]',
      titleKey: "tutorials:documentWorkspace.basics.navigationTitle",
      titleFallback: "Jump Through Long Documents",
      contentKey: "tutorials:documentWorkspace.basics.navigationContent",
      contentFallback:
        "Use page navigation, search, zoom, and keyboard shortcuts to move through long documents efficiently.",
      placement: "bottom"
    }
  ]
}

export const documentWorkspaceTutorials: TutorialDefinition[] = [
  documentWorkspaceBasics
]
