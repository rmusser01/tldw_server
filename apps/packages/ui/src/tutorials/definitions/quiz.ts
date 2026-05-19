/**
 * Quiz Playground Tutorial Definitions
 */

import { GraduationCap } from "lucide-react"
import type { TutorialDefinition } from "../registry"

const quizBasics: TutorialDefinition = {
  id: "quiz-basics",
  routePattern: "/quiz",
  labelKey: "tutorials:quiz.basics.label",
  labelFallback: "Quiz Basics",
  descriptionKey: "tutorials:quiz.basics.description",
  descriptionFallback:
    "Learn how to generate, take, and review quizzes from your media content.",
  icon: GraduationCap,
  priority: 1,
  steps: [
    {
      target: '[data-testid="quiz-playground-tabs"]',
      titleKey: "tutorials:quiz.basics.tabsTitle",
      titleFallback: "Quiz Playground",
      contentKey: "tutorials:quiz.basics.tabsContent",
      contentFallback:
        "The Quiz Playground has five tabs. Generate quizzes from your media, create them manually, take quizzes, manage your library, and review results.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="quiz-tab-generate"]',
      titleKey: "tutorials:quiz.basics.generateTitle",
      titleFallback: "Generate from Media",
      contentKey: "tutorials:quiz.basics.generateContent",
      contentFallback:
        "Start here. Select a video, article, or document from your media library, and the AI will generate quiz questions for you.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="quiz-tab-create"]',
      titleKey: "tutorials:quiz.basics.createTitle",
      titleFallback: "Create Manually",
      contentKey: "tutorials:quiz.basics.createContent",
      contentFallback:
        "Prefer to write your own? Create quizzes with multiple choice, true/false, fill-in-the-blank, and matching questions.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="quiz-tab-take"]',
      titleKey: "tutorials:quiz.basics.takeTitle",
      titleFallback: "Take a Quiz",
      contentKey: "tutorials:quiz.basics.takeContent",
      contentFallback:
        "Once you have quizzes, come here to take them. Your answers are saved and scored automatically.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="quiz-tab-results"]',
      titleKey: "tutorials:quiz.basics.resultsTitle",
      titleFallback: "Review Results",
      contentKey: "tutorials:quiz.basics.resultsContent",
      contentFallback:
        "See your scores, review incorrect answers, and track your progress over time.",
      placement: "bottom",
      disableBeacon: true
    }
  ]
}

export const quizTutorials: TutorialDefinition[] = [quizBasics]
