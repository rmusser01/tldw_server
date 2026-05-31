import type { TFunction } from "i18next"
import { getDesignSystemState } from "@/design-system"

export type DemoFlashcardDeck = {
  id: string
  name: string
  summary: string
}

export type DemoNotePreview = {
  id: string
  title: string
  preview: string
  updated_at: string
}

export type DemoMediaPreviewStatusKey = "ready" | "processing"

export type DemoMediaPreview = {
  id: string
  title: string
  meta: string
  statusKey: DemoMediaPreviewStatusKey
  statusLabel: string
}

export type DemoQuizQuestion = {
  id: string
  prompt: string
  explanation: string
} & (
  | {
      type: "multiple_choice"
      options: string[]
      correctAnswer: string
    }
  | {
      type: "true_false"
      correctAnswer: "true" | "false"
    }
  | {
      type: "fill_blank"
      placeholder: string
      correctAnswer: string
      acceptedAnswers: string[]
    }
)

export type DemoQuiz = {
  id: string
  title: string
  description: string
  sourceLabel: string
  difficulty: "easy" | "medium" | "hard"
  passingScore: number
  timeLimitMinutes: number
  questions: DemoQuizQuestion[]
}

export const getDemoFlashcardDecks = (t: TFunction): DemoFlashcardDeck[] => [
  {
    id: "demo-deck-1",
    name: t("option:flashcards.demoSample1Title", {
      defaultValue: "Demo deck: Core concepts"
    }),
    summary: t("option:flashcards.demoSample1Summary", {
      defaultValue: "10 cards · Great for testing spacing and ratings."
    })
  },
  {
    id: "demo-deck-2",
    name: t("option:flashcards.demoSample2Title", {
      defaultValue: "Demo deck: Product terms"
    }),
    summary: t("option:flashcards.demoSample2Summary", {
      defaultValue: "8 cards · Names, acronyms, and key definitions."
    })
  },
  {
    id: "demo-deck-3",
    name: t("option:flashcards.demoSample3Title", {
      defaultValue: "Demo deck: Meeting follow-ups"
    }),
    summary: t("option:flashcards.demoSample3Summary", {
      defaultValue: "6 cards · Example action items to review."
    })
  }
]

export const getDemoNotes = (t: TFunction): DemoNotePreview[] => [
  {
    id: "demo-note-1",
    title: t("option:notesEmpty.demoSample1Title", {
      defaultValue: "Demo note: Weekly meeting recap"
    }),
    preview: t("option:notesEmpty.demoSample1Preview", {
      defaultValue:
        "Decisions, blockers, and follow-ups from a recent team sync."
    }),
    updated_at: t("option:notesEmpty.demoSample1Meta", {
      defaultValue: "Today · 9:32 AM"
    })
  },
  {
    id: "demo-note-2",
    title: t("option:notesEmpty.demoSample2Title", {
      defaultValue: "Demo note: Research highlights"
    }),
    preview: t("option:notesEmpty.demoSample2Preview", {
      defaultValue:
        "Key insights pulled from a long article or paper."
    }),
    updated_at: t("option:notesEmpty.demoSample2Meta", {
      defaultValue: "Yesterday · 4:10 PM"
    })
  },
  {
    id: "demo-note-3",
    title: t("option:notesEmpty.demoSample3Title", {
      defaultValue: "Demo note: Call summary"
    }),
    preview: t("option:notesEmpty.demoSample3Preview", {
      defaultValue:
        "Summary of a customer call with next steps and owners."
    }),
    updated_at: t("option:notesEmpty.demoSample3Meta", {
      defaultValue: "This week"
    })
  }
]

export const getDemoMediaItems = (t: TFunction): DemoMediaPreview[] => {
  const readyStatusLabel = getDesignSystemState("ready").label

  return [
    {
      id: "demo-media-1",
      title: t("review:mediaEmpty.demoSample1Title", {
        defaultValue: "Demo media: Team call recording"
      }),
      meta: t("review:mediaEmpty.demoSample1Meta", {
        defaultValue: "Video · 25 min · Keywords: standup, planning"
      }),
      statusKey: "ready",
      statusLabel: readyStatusLabel
    },
    {
      id: "demo-media-2",
      title: t("review:mediaEmpty.demoSample2Title", {
        defaultValue: "Demo media: Product walkthrough"
      }),
      meta: t("review:mediaEmpty.demoSample2Meta", {
        defaultValue: "Screen recording · 12 min · Keywords: onboarding"
      }),
      statusKey: "processing",
      statusLabel: "Processing"
    },
    {
      id: "demo-media-3",
      title: t("review:mediaEmpty.demoSample3Title", {
        defaultValue: "Demo media: Research article PDF"
      }),
      meta: t("review:mediaEmpty.demoSample3Meta", {
        defaultValue: "PDF · 6 pages · Keywords: summarization"
      }),
      statusKey: "ready",
      statusLabel: readyStatusLabel
    }
  ]
}

export const getDemoQuizzes = (t: TFunction): DemoQuiz[] => [
  {
    id: "demo-quiz-1",
    title: t("option:quiz.demoSample1Title", {
      defaultValue: "Demo quiz: Research workflow fundamentals"
    }),
    description: t("option:quiz.demoSample1Summary", {
      defaultValue: "3 questions · Mixed formats · Quick confidence check."
    }),
    sourceLabel: t("option:quiz.demoSample1Source", {
      defaultValue: "Source: Product walkthrough transcript"
    }),
    difficulty: "easy",
    passingScore: 70,
    timeLimitMinutes: 4,
    questions: [
      {
        id: "demo-quiz-1-q1",
        type: "multiple_choice",
        prompt: t("option:quiz.demoSample1Q1Prompt", {
          defaultValue:
            "Which tab is used to automatically build quizzes from existing media?"
        }),
        options: [
          t("option:quiz.demoSample1Q1Option1", { defaultValue: "Generate" }),
          t("option:quiz.demoSample1Q1Option2", { defaultValue: "Manage" }),
          t("option:quiz.demoSample1Q1Option3", { defaultValue: "Results" })
        ],
        correctAnswer: t("option:quiz.demoSample1Q1Option1", {
          defaultValue: "Generate"
        }),
        explanation: t("option:quiz.demoSample1Q1Explanation", {
          defaultValue:
            "Generate is used to create a quiz directly from your connected content."
        })
      },
      {
        id: "demo-quiz-1-q2",
        type: "true_false",
        prompt: t("option:quiz.demoSample1Q2Prompt", {
          defaultValue: "True or false: Quiz progress should autosave while taking."
        }),
        correctAnswer: "true",
        explanation: t("option:quiz.demoSample1Q2Explanation", {
          defaultValue:
            "Autosave prevents accidental data loss when tabs close or connections drop."
        })
      },
      {
        id: "demo-quiz-1-q3",
        type: "fill_blank",
        prompt: t("option:quiz.demoSample1Q3Prompt", {
          defaultValue:
            "Fill in the blank: A score history view helps learners identify score ____ over time."
        }),
        placeholder: t("option:quiz.demoSample1Q3Placeholder", {
          defaultValue: "Enter one word"
        }),
        correctAnswer: t("option:quiz.demoSample1Q3Correct", {
          defaultValue: "trends"
        }),
        acceptedAnswers: [
          t("option:quiz.demoSample1Q3Correct", {
            defaultValue: "trends"
          }),
          t("option:quiz.demoSample1Q3Accepted1", {
            defaultValue: "improvement"
          })
        ],
        explanation: t("option:quiz.demoSample1Q3Explanation", {
          defaultValue:
            "Trend visibility helps learners focus on long-term improvement, not just one score."
        })
      }
    ]
  },
  {
    id: "demo-quiz-2",
    title: t("option:quiz.demoSample2Title", {
      defaultValue: "Demo quiz: Study loop essentials"
    }),
    description: t("option:quiz.demoSample2Summary", {
      defaultValue: "3 questions · Results-to-retake workflow."
    }),
    sourceLabel: t("option:quiz.demoSample2Source", {
      defaultValue: "Source: Learning systems notes"
    }),
    difficulty: "medium",
    passingScore: 75,
    timeLimitMinutes: 5,
    questions: [
      {
        id: "demo-quiz-2-q1",
        type: "multiple_choice",
        prompt: t("option:quiz.demoSample2Q1Prompt", {
          defaultValue:
            "What is the fastest way to retry a low-scoring attempt from analytics?"
        }),
        options: [
          t("option:quiz.demoSample2Q1Option1", {
            defaultValue: "Use Retake from Results"
          }),
          t("option:quiz.demoSample2Q1Option2", {
            defaultValue: "Create a new quiz manually"
          }),
          t("option:quiz.demoSample2Q1Option3", {
            defaultValue: "Export and re-import the quiz"
          })
        ],
        correctAnswer: t("option:quiz.demoSample2Q1Option1", {
          defaultValue: "Use Retake from Results"
        }),
        explanation: t("option:quiz.demoSample2Q1Explanation", {
          defaultValue:
            "A direct retake action shortens the feedback loop and improves flow."
        })
      },
      {
        id: "demo-quiz-2-q2",
        type: "true_false",
        prompt: t("option:quiz.demoSample2Q2Prompt", {
          defaultValue:
            "True or false: Showing only color for right/wrong is enough for accessibility."
        }),
        correctAnswer: "false",
        explanation: t("option:quiz.demoSample2Q2Explanation", {
          defaultValue:
            "Accessible feedback should include text or icons in addition to color."
        })
      },
      {
        id: "demo-quiz-2-q3",
        type: "fill_blank",
        prompt: t("option:quiz.demoSample2Q3Prompt", {
          defaultValue:
            "Fill in the blank: A pre-quiz screen should state time limit, question count, and ____ score."
        }),
        placeholder: t("option:quiz.demoSample2Q3Placeholder", {
          defaultValue: "Enter one word"
        }),
        correctAnswer: t("option:quiz.demoSample2Q3Correct", {
          defaultValue: "passing"
        }),
        acceptedAnswers: [
          t("option:quiz.demoSample2Q3Correct", {
            defaultValue: "passing"
          })
        ],
        explanation: t("option:quiz.demoSample2Q3Explanation", {
          defaultValue:
            "The passing score sets clear expectations before the attempt begins."
        })
      }
    ]
  }
]
