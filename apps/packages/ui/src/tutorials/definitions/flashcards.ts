/**
 * Flashcards Tutorial Definitions
 */

import { Library } from "lucide-react"
import type { TutorialDefinition } from "../registry"

const flashcardsBasics: TutorialDefinition = {
  id: "flashcards-basics",
  routePattern: "/flashcards",
  labelKey: "tutorials:flashcards.basics.label",
  labelFallback: "Flashcards Basics",
  descriptionKey: "tutorials:flashcards.basics.description",
  descriptionFallback:
    "Use study, manage, and transfer flows to maintain spaced-repetition decks",
  icon: Library,
  priority: 1,
  steps: [
    {
      target: '[data-testid="flashcards-tabs"]',
      titleKey: "tutorials:flashcards.basics.tabsTitle",
      titleFallback: "Study, Manage, Transfer",
      contentKey: "tutorials:flashcards.basics.tabsContent",
      contentFallback:
        "Switch tabs to study cards, manage deck contents, and import / export card data.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="flashcards-review-topbar"], [data-testid="flashcards-review-empty-card"]',
      titleKey: "tutorials:flashcards.basics.reviewTitle",
      titleFallback: "Study Tab",
      contentKey: "tutorials:flashcards.basics.reviewContent",
      contentFallback:
        "Run review sessions here and score recall quality to update spaced-repetition schedules.",
      placement: "bottom"
    },
    {
      target: '[data-testid="flashcards-manage-topbar"], [data-testid="flashcards-manage-search"]',
      titleKey: "tutorials:flashcards.basics.manageTitle",
      titleFallback: "Manage Tab",
      contentKey: "tutorials:flashcards.basics.manageContent",
      contentFallback:
        "Open the Manage tab to search cards, filter by deck/tags, and perform bulk updates.",
      placement: "bottom"
    },
    {
      target: '[data-testid="flashcards-import-format"], [data-testid="flashcards-import-help-accordion"]',
      titleKey: "tutorials:flashcards.basics.transferTitle",
      titleFallback: "Transfer Tab",
      contentKey: "tutorials:flashcards.basics.transferContent",
      contentFallback:
        "Use Transfer for CSV/JSON/APKG import and export workflows.",
      placement: "left"
    },
    {
      target: '[data-testid="flashcards-to-quiz-cta"]',
      titleKey: "tutorials:flashcards.basics.quizTitle",
      titleFallback: "Bridge to Quiz",
      contentKey: "tutorials:flashcards.basics.quizContent",
      contentFallback:
        "Jump into Quiz mode from your selected deck to validate retention with assessments.",
      placement: "left"
    }
  ]
}

export const flashcardsTutorials: TutorialDefinition[] = [flashcardsBasics]
