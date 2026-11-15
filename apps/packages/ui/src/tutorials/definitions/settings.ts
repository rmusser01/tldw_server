import { Settings } from "lucide-react"
import type { TutorialDefinition } from "../registry"

export const settingsTutorials: TutorialDefinition[] = [
  {
    id: "settings-basics",
    routePattern: "/settings",
    labelKey: "tutorials:settings.basics.label",
    labelFallback: "Settings Basics",
    descriptionKey: "tutorials:settings.basics.description",
    descriptionFallback: "Find the settings you need and see what each section controls.",
    icon: Settings,
    priority: 1,
    requireFirstTargetForPrompt: true,
    steps: [
      {
        disableBeacon: true,
        target: '[data-testid="settings-nav-select"], [data-testid="settings-nav-filter"]',
        titleKey: "tutorials:settings.basics.navigationTitle",
        titleFallback: "Find a setting",
        contentKey: "tutorials:settings.basics.navigationContent",
        contentFallback: "Search for a section on desktop, or choose a section from the Settings menu on mobile.",
        placement: "bottom"
      },
      {
        target: '#settings-route-heading',
        titleKey: "tutorials:settings.basics.overviewTitle",
        titleFallback: "Settings by section",
        contentKey: "tutorials:settings.basics.overviewContent",
        contentFallback: "Check the heading after switching sections so you know which settings your changes will affect.",
        placement: "bottom"
      }
    ]
  }
]
