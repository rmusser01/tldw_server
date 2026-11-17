/**
 * Prompts Tutorial Definitions
 */

import { SquarePen } from "lucide-react"
import type { TutorialDefinition } from "../registry"

const promptsBasics: TutorialDefinition = {
  id: "prompts-basics",
  routePattern: "/prompts",
  labelKey: "tutorials:prompts.basics.label",
  labelFallback: "Prompts Basics",
  descriptionKey: "tutorials:prompts.basics.description",
  descriptionFallback:
    "Save instructions you can reuse in Chat.",
  icon: SquarePen,
  priority: 1,
  requireFirstTargetForPrompt: true,
  steps: [
    {
      disableBeacon: true,
      target: '[data-testid="prompts-add"]',
      titleKey: "tutorials:prompts.basics.createTitle",
      titleFallback: "Create Prompt",
      contentKey: "tutorials:prompts.basics.createContent",
      contentFallback:
        "Choose New prompt and save a set of instructions you use often. Insert the saved prompt in Chat when you need it.",
      placement: "bottom"
    },
    {
      optional: true,
      target: '[data-testid="prompts-search"]',
      titleKey: "tutorials:prompts.basics.searchTitle",
      titleFallback: "Search Prompts",
      contentKey: "tutorials:prompts.basics.searchContent",
      contentFallback:
        "Search prompt names, content, and keywords to quickly find the right template.",
      placement: "bottom"
    },
    {
      optional: true,
      target: '[data-testid="prompts-segmented"]',
      titleKey: "tutorials:prompts.basics.segmentedTitle",
      titleFallback: "Prompt Workspaces",
      contentKey: "tutorials:prompts.basics.segmentedContent",
      contentFallback:
        "Switch between Custom, Copilot, Studio, and Trash to manage different prompt sources.",
      placement: "bottom",
      disableBeacon: true
    }
  ]
}

export const promptsTutorials: TutorialDefinition[] = [promptsBasics]
