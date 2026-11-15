import { Library } from "lucide-react"
import type { TutorialDefinition } from "../registry"

export const collectionsTutorials: TutorialDefinition[] = [
  {
    id: "collections-basics",
    routePattern: "/collections",
    labelKey: "tutorials:collections.basics.label",
    labelFallback: "Collections Basics",
    descriptionKey: "tutorials:collections.basics.description",
    descriptionFallback: "Organize saved articles, highlights, and reading digests.",
    icon: Library,
    priority: 1,
    requireFirstTargetForPrompt: true,
    steps: [
      {
        target: '[data-testid="collections-heading"]',
        titleKey: "tutorials:collections.basics.overviewTitle",
        titleFallback: "Your saved reading",
        contentKey: "tutorials:collections.basics.overviewContent",
        contentFallback: "Collections keeps your reading list, highlights, templates, and digest schedules together.",
        placement: "bottom",
        disableBeacon: true
      },
      {
        target: '.collections-tabs [role="tablist"]',
        titleKey: "tutorials:collections.basics.tabsTitle",
        titleFallback: "Choose a tool",
        contentKey: "tutorials:collections.basics.tabsContent",
        contentFallback: "Use Reading List to manage articles, Highlights to revisit passages, or Import/Export to move your reading list.",
        placement: "bottom"
      }
    ]
  }
]
