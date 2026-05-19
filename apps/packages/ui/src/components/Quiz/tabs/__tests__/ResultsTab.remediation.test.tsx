import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ResultsTab } from "../ResultsTab"
import {
  useAllAttemptsQuery,
  useAttemptQuery,
  useAttemptRemediationConversionsQuery,
  useConvertAttemptRemediationQuestionsMutation,
  useGenerateRemediationQuizMutation,
  useQuizAttemptQuestionAssistantQuery,
  useQuizAttemptQuestionAssistantRespondMutation,
  useQuizzesQuery
} from "../../hooks"
import {
  useDecksQuery
} from "@/components/Flashcards/hooks/useFlashcardQueries"
import { useStudySuggestions } from "@/components/StudySuggestions/hooks/useStudySuggestions"

const mocks = vi.hoisted(() => ({
  convertRemediationQuestions: vi.fn(),
  navigate: vi.fn(),
  remediationQuiz: vi.fn(),
  assistantRespond: vi.fn()
}))

const interpolate = (template: string, values: Record<string, unknown> | undefined) => {
  return template.replace(/\{\{\s*([^\s}]+)\s*\}\}/g, (_, key: string) => {
    const value = values?.[key]
    return value == null ? "" : String(value)
  })
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
            [key: string]: unknown
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      const defaultValue = defaultValueOrOptions?.defaultValue
      if (typeof defaultValue === "string") {
        return interpolate(defaultValue, defaultValueOrOptions)
      }
      return key
    }
  })
}))

vi.mock("react-router-dom", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router-dom")>()
  return {
    ...actual,
    useNavigate: () => mocks.navigate
  }
})

vi.mock("../../hooks", () => ({
  useAllAttemptsQuery: vi.fn(),
  useQuizzesQuery: vi.fn(),
  useAttemptQuery: vi.fn(),
  useAttemptRemediationConversionsQuery: vi.fn(),
  useConvertAttemptRemediationQuestionsMutation: vi.fn(),
  useGenerateRemediationQuizMutation: vi.fn(),
  useQuizAttemptQuestionAssistantQuery: vi.fn(),
  useQuizAttemptQuestionAssistantRespondMutation: vi.fn()
}))

vi.mock("@/components/Flashcards/hooks/useFlashcardQueries", () => ({
  useDecksQuery: vi.fn()
}))

vi.mock("@/components/StudySuggestions/hooks/useStudySuggestions", () => ({
  useStudySuggestions: vi.fn()
}))

vi.mock("@/hooks/useTTS", () => ({
  useTTS: () => ({
    speak: vi.fn(),
    cancel: vi.fn(),
    isSpeaking: false
  })
}))

vi.mock("@/hooks/useSpeechRecognition", () => ({
  useSpeechRecognition: () => ({
    supported: false,
    isListening: false,
    transcript: "",
    start: vi.fn(),
    stop: vi.fn(),
    resetTranscript: vi.fn()
  })
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

describe("ResultsTab remediation panel", () => {
  const onRetakeQuiz = vi.fn()
  let assistantQueryState: any
  const assistantRefetchMock = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    window.sessionStorage.clear()
    onRetakeQuiz.mockReset()
    assistantQueryState = null
    assistantRefetchMock.mockReset()
    vi.mocked(useStudySuggestions).mockReturnValue({
      status: "none",
      statusQuery: {} as any,
      snapshot: null,
      activeSnapshotId: null,
      isLoading: false,
      isRefreshing: false,
      refresh: vi.fn(),
      performAction: vi.fn()
    } as any)

    mocks.convertRemediationQuestions.mockResolvedValue({
      attempt_id: 101,
      quiz_id: 7,
      target_deck: { id: 3, name: "Renal Recovery Deck" },
      results: [
        {
          question_id: 13,
          status: "created",
          conversion: {
            id: 902,
            attempt_id: 101,
            quiz_id: 7,
            question_id: 13,
            status: "active",
            orphaned: false,
            superseded_count: 0,
            target_deck_id: 3,
            target_deck_name_snapshot: "Renal Recovery Deck",
            flashcard_count: 1,
            flashcard_uuids_json: ["fc-13"],
            source_ref_id: "quiz-attempt:101:question:13",
            created_at: "2026-03-13T09:16:00Z",
            last_modified: "2026-03-13T09:16:00Z",
            client_id: "test",
            version: 1
          },
          flashcard_uuids: ["fc-13"],
          error: null
        }
      ],
      created_flashcard_uuids: ["fc-13"]
    })
    mocks.remediationQuiz.mockResolvedValue({
      quiz: {
        id: 55,
        name: "Quiz: Remediation"
      },
      questions: []
    })
    mocks.assistantRespond.mockResolvedValue({
      assistant_message: {
        id: 400,
        content: "Review the misconception in your own words."
      }
    })

    vi.mocked(useAllAttemptsQuery).mockReturnValue({
      data: {
        items: [
          {
            id: 101,
            quiz_id: 7,
            started_at: "2026-03-13T09:00:00Z",
            completed_at: "2026-03-13T09:12:00Z",
            score: 1,
            total_possible: 3,
            time_spent_seconds: 720,
            answers: []
          }
        ],
        count: 1
      },
      isLoading: false
    } as any)

    vi.mocked(useQuizzesQuery).mockReturnValue({
      data: {
        items: [
          {
            id: 7,
            name: "Renal Physiology",
            total_questions: 3,
            media_id: 42
          }
        ],
        count: 1
      },
      isLoading: false
    } as any)

    vi.mocked(useAttemptQuery).mockReturnValue({
      data: {
        id: 101,
        quiz_id: 7,
        started_at: "2026-03-13T09:00:00Z",
        completed_at: "2026-03-13T09:12:00Z",
        score: 1,
        total_possible: 3,
        time_spent_seconds: 720,
        answers: [
          {
            question_id: 12,
            user_answer: "Bowman capsule",
            is_correct: false,
            correct_answer: "Glomerulus",
            explanation: "Filtration happens at the glomerulus."
          },
          {
            question_id: 13,
            user_answer: "Descending limb",
            is_correct: false,
            correct_answer: "Ascending limb",
            explanation: "The ascending limb reabsorbs sodium."
          },
          {
            question_id: 14,
            user_answer: "ADH",
            is_correct: true,
            correct_answer: "ADH",
            explanation: "Correct."
          }
        ],
        questions: [
          {
            id: 12,
            quiz_id: 7,
            question_type: "multiple_choice",
            question_text: "Which structure filters blood?",
            options: ["Bowman capsule", "Glomerulus", "Loop of Henle"],
            points: 1,
            order_index: 0,
            deleted: false,
            client_id: "test",
            version: 1
          },
          {
            id: 13,
            quiz_id: 7,
            question_type: "multiple_choice",
            question_text: "Which nephron segment reabsorbs sodium without water?",
            options: ["Descending limb", "Ascending limb", "Collecting duct"],
            points: 1,
            order_index: 1,
            deleted: false,
            client_id: "test",
            version: 1
          },
          {
            id: 14,
            quiz_id: 7,
            question_type: "multiple_choice",
            question_text: "Which hormone raises water reabsorption?",
            options: ["ADH", "Insulin", "Thyroxine"],
            points: 1,
            order_index: 2,
            deleted: false,
            client_id: "test",
            version: 1
          }
        ]
      },
      isLoading: false,
      isFetching: false
    } as any)

    vi.mocked(useGenerateRemediationQuizMutation).mockReturnValue({
      mutateAsync: mocks.remediationQuiz,
      isPending: false
    } as any)
    vi.mocked(useAttemptRemediationConversionsQuery).mockReturnValue({
      data: {
        attempt_id: 101,
        items: [
          {
            id: 901,
            attempt_id: 101,
            quiz_id: 7,
            question_id: 12,
            status: "active",
            orphaned: false,
            superseded_count: 1,
            target_deck_id: 3,
            target_deck_name_snapshot: "Renal Recovery Deck",
            flashcard_count: 1,
            flashcard_uuids_json: ["fc-12"],
            source_ref_id: "quiz-attempt:101:question:12",
            created_at: "2026-03-13T09:15:00Z",
            last_modified: "2026-03-13T09:15:00Z",
            client_id: "test",
            version: 1
          }
        ],
        count: 1,
        superseded_count: 1
      },
      isLoading: false
    } as any)
    vi.mocked(useConvertAttemptRemediationQuestionsMutation).mockReturnValue({
      mutateAsync: mocks.convertRemediationQuestions,
      isPending: false
    } as any)

    assistantRefetchMock.mockImplementation(async () => {
      const nextMessageId = 201 + (assistantQueryState.data?.messages?.length ?? 1)
      assistantQueryState = {
        ...assistantQueryState,
        data: {
          ...assistantQueryState.data,
          thread: {
            ...assistantQueryState.data.thread,
            version: 2,
            message_count: 2
          },
          messages: [
            ...(assistantQueryState.data?.messages ?? []),
            {
              id: nextMessageId,
              thread_id: 19,
              role: "assistant",
              action_type: "follow_up",
              input_modality: "text",
              content: "Latest remediation context",
              structured_payload: {},
              context_snapshot: {},
              provider: "openai",
              model: "gpt-5",
              created_at: "2026-03-13T09:14:00Z",
              client_id: "test"
            }
          ]
        }
      }
      return assistantQueryState
    })
    vi.mocked(useQuizAttemptQuestionAssistantQuery).mockImplementation(
      (_attemptId: number | null | undefined, questionId: number | null | undefined) => {
        if (questionId !== 12) {
          return {
            data: null,
            isLoading: false,
            isError: false,
            refetch: assistantRefetchMock
          } as any
        }

        if (!assistantQueryState) {
          assistantQueryState = {
            data: {
              thread: {
                id: 19,
                context_type: "quiz_attempt_question",
                quiz_attempt_id: 101,
                question_id: 12,
                message_count: 1,
                deleted: false,
                client_id: "test",
                version: 1
              },
              messages: [
                {
                  id: 201,
                  thread_id: 19,
                  role: "assistant",
                  action_type: "explain",
                  input_modality: "text",
                  content: "Review the misconception in your own words.",
                  structured_payload: {},
                  context_snapshot: {},
                  provider: "openai",
                  model: "gpt-5",
                  created_at: "2026-03-13T09:13:00Z",
                  client_id: "test"
                }
              ],
              context_snapshot: {},
              available_actions: ["explain", "follow_up", "freeform"]
            },
            isLoading: false,
            isError: false
          }
        }

        return {
          ...assistantQueryState,
          refetch: assistantRefetchMock
        } as any
      }
    )

    vi.mocked(useQuizAttemptQuestionAssistantRespondMutation).mockReturnValue({
      mutateAsync: mocks.assistantRespond,
      isPending: false
    } as any)

    vi.mocked(useDecksQuery).mockReturnValue({
      data: [{
        id: 3,
        name: "Renal Recovery Deck",
        scheduler_settings: {
          new_steps_minutes: [1, 10],
          relearn_steps_minutes: [10],
          graduating_interval_days: 1,
          easy_interval_days: 4,
          easy_bonus: 1.3,
          interval_modifier: 1,
          max_interval_days: 36500,
          leech_threshold: 8,
          enable_fuzz: false
        }
      }],
      isLoading: false
    } as any)
  })

  const openDetails = async () => {
    render(<ResultsTab onRetakeQuiz={onRetakeQuiz} />)
    fireEvent.click(screen.getByRole("button", { name: /view details/i }))
    await waitFor(() => {
      expect(screen.getByText("Attempt Details")).toBeInTheDocument()
    })
  }

  it("explains a missed question and shows question-scoped assistant history", async () => {
    await openDetails()

    fireEvent.click(screen.getByRole("button", { name: /explain mistake for question 12/i }))

    await waitFor(() => {
      expect(mocks.assistantRespond).toHaveBeenCalledWith({
        attemptId: 101,
        questionId: 12,
        request: { action: "explain" }
      })
    })
    expect(screen.getByText("Review the misconception in your own words.")).toBeInTheDocument()
  })

  it("generates remediation quizzes from selected missed questions", async () => {
    await openDetails()

    fireEvent.click(screen.getByLabelText(/select missed question 12/i))
    fireEvent.click(screen.getByLabelText(/select missed question 13/i))
    fireEvent.click(screen.getByRole("button", { name: /create remediation quiz/i }))

    await waitFor(() => {
      expect(mocks.remediationQuiz).toHaveBeenCalledWith({
        attemptId: 101,
        questionIds: [12, 13]
      })
    })
    await waitFor(() => {
      expect(onRetakeQuiz).toHaveBeenCalledWith(
        expect.objectContaining({
          startQuizId: 55,
          highlightQuizId: 55,
          sourceTab: "results",
          attemptId: 101
        })
      )
    })
  })

  it("does not read remediation conversion state from session storage", async () => {
    const getItemSpy = vi.spyOn(window.sessionStorage, "getItem")

    await openDetails()

    expect(getItemSpy).not.toHaveBeenCalledWith("quiz-results-missed-flashcards-v1")
    expect(getItemSpy).not.toHaveBeenCalledWith("quiz-results-flashcard-deck-map-v1")
  })

  it("opens remediation flashcards and preserves study handoff actions", async () => {
    await openDetails()

    expect(screen.getByText("Converted in deck Renal Recovery Deck.")).toBeInTheDocument()
    expect(
      screen.getByText("Superseded remediation history exists for this question.")
    ).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: /create remediation flashcards/i }))

    await waitFor(() => {
      expect(screen.getByText("Destination deck")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /study linked cards/i }))

    expect(mocks.navigate).toHaveBeenCalledWith(
      expect.stringContaining("/flashcards?tab=review&study_source=quiz&quiz_id=7&attempt_id=101")
    )
    expect(mocks.navigate).toHaveBeenCalledWith(expect.stringContaining("deck_id=3"))
  })

  it("creates missed-question flashcards through the quiz remediation conversion endpoint", async () => {
    await openDetails()

    fireEvent.click(screen.getByRole("button", { name: /create remediation flashcards/i }))

    await waitFor(() => {
      expect(screen.getByText("Destination deck")).toBeInTheDocument()
    })
    expect(screen.getByTestId("quiz-remediation-selected-deck-summary")).toHaveTextContent(
      "Scheduler: 1m,10m -> 1d / easy 4d / leech 8 / fuzz off"
    )

    fireEvent.click(screen.getByRole("button", { name: /^create flashcards$/i }))

    await waitFor(() => {
      expect(mocks.convertRemediationQuestions).toHaveBeenCalledWith({
        attemptId: 101,
        request: {
          question_ids: [13],
          target_deck_id: 3,
          replace_active: false
        }
      })
    })
  })

  it("submits scheduler settings when remediation creates a new deck", async () => {
    await openDetails()

    fireEvent.click(screen.getByRole("button", { name: /create remediation flashcards/i }))

    await waitFor(() => {
      expect(screen.getByText("Destination deck")).toBeInTheDocument()
    })

    const dialog = screen
      .getAllByRole("dialog")
      .find((element) => within(element).queryByText("Destination deck"))
    expect(dialog).not.toBeNull()
    fireEvent.mouseDown(within(dialog as HTMLElement).getByRole("combobox"))
    fireEvent.click(await screen.findByText("Create new deck"))
    fireEvent.change(screen.getByTestId("quiz-remediation-new-deck-name"), {
      target: { value: "Missed Questions Deck" }
    })
    fireEvent.mouseDown(screen.getByTestId("deck-study-defaults-review-prompt-side"))
    fireEvent.click(await screen.findByText("Back first"))
    fireEvent.click(screen.getByTestId("deck-scheduler-editor-preset-fast_acquisition"))
    fireEvent.click(screen.getByRole("button", { name: /^create flashcards$/i }))

    await waitFor(() => {
      expect(mocks.convertRemediationQuestions).toHaveBeenCalledWith({
        attemptId: 101,
        request: {
          question_ids: [13],
          create_deck_name: "Missed Questions Deck",
          create_deck_review_prompt_side: "back",
          create_deck_scheduler_type: "sm2_plus",
          create_deck_scheduler_settings: {
            sm2_plus: {
              new_steps_minutes: [1, 5, 15],
              relearn_steps_minutes: [10],
              graduating_interval_days: 1,
              easy_interval_days: 3,
              easy_bonus: 1.15,
              interval_modifier: 0.9,
              max_interval_days: 3650,
              leech_threshold: 10,
              enable_fuzz: false
            },
            fsrs: {
              target_retention: 0.9,
              maximum_interval_days: 36500,
              enable_fuzz: false
            }
          },
          replace_active: false
        }
      })
    })
  }, 12000)

  it("resubmits already-converted questions with replace_active when confirmed", async () => {
    mocks.convertRemediationQuestions
      .mockResolvedValueOnce({
        attempt_id: 101,
        quiz_id: 7,
        target_deck: { id: 3, name: "Renal Recovery Deck" },
        results: [
          {
            question_id: 12,
            status: "already_exists",
            conversion: {
              id: 901,
              attempt_id: 101,
              quiz_id: 7,
              question_id: 12,
              status: "active",
              orphaned: false,
              superseded_count: 0,
              target_deck_id: 3,
              target_deck_name_snapshot: "Renal Recovery Deck",
              flashcard_count: 1,
              flashcard_uuids_json: ["fc-12"],
              source_ref_id: "quiz-attempt:101:question:12",
              created_at: "2026-03-13T09:15:00Z",
              last_modified: "2026-03-13T09:15:00Z",
              client_id: "test",
              version: 1
            },
            flashcard_uuids: ["fc-12"],
            error: null
          }
        ],
        created_flashcard_uuids: []
      })
      .mockResolvedValueOnce({
        attempt_id: 101,
        quiz_id: 7,
        target_deck: { id: 3, name: "Renal Recovery Deck" },
        results: [
          {
            question_id: 12,
            status: "superseded_and_created",
            conversion: {
              id: 903,
              attempt_id: 101,
              quiz_id: 7,
              question_id: 12,
              status: "active",
              orphaned: false,
              superseded_count: 1,
              target_deck_id: 3,
              target_deck_name_snapshot: "Renal Recovery Deck",
              flashcard_count: 1,
              flashcard_uuids_json: ["fc-12-new"],
              source_ref_id: "quiz-attempt:101:question:12",
              created_at: "2026-03-13T09:18:00Z",
              last_modified: "2026-03-13T09:18:00Z",
              client_id: "test",
              version: 1
            },
            flashcard_uuids: ["fc-12-new"],
            error: null
          }
        ],
        created_flashcard_uuids: ["fc-12-new"]
      })

    await openDetails()

    fireEvent.click(screen.getByRole("button", { name: /create remediation flashcards/i }))

    await waitFor(() => {
      expect(screen.getByText("Destination deck")).toBeInTheDocument()
    })

    const modal = screen.getAllByRole("dialog")[1]
    fireEvent.click(within(modal).getByRole("checkbox", { name: /which structure filters blood/i }))
    fireEvent.click(screen.getByRole("button", { name: /^create flashcards$/i }))

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /convert again anyway/i })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /convert again anyway/i }))

    await waitFor(() => {
      expect(mocks.convertRemediationQuestions).toHaveBeenNthCalledWith(2, {
        attemptId: 101,
        request: {
          question_ids: [12],
          target_deck_id: 3,
          replace_active: true
        }
      })
    })
  }, 12000)

  it("reuses the server-created deck when a new-deck conversion needs replace_active retry", async () => {
    mocks.convertRemediationQuestions
      .mockResolvedValueOnce({
        attempt_id: 101,
        quiz_id: 7,
        target_deck: { id: 44, name: "Retry Deck" },
        results: [
          {
            question_id: 12,
            status: "already_exists",
            conversion: {
              id: 901,
              attempt_id: 101,
              quiz_id: 7,
              question_id: 12,
              status: "active",
              orphaned: false,
              superseded_count: 0,
              target_deck_id: 44,
              target_deck_name_snapshot: "Retry Deck",
              flashcard_count: 1,
              flashcard_uuids_json: ["fc-12"],
              source_ref_id: "quiz-attempt:101:question:12",
              created_at: "2026-03-13T09:15:00Z",
              last_modified: "2026-03-13T09:15:00Z",
              client_id: "test",
              version: 1
            },
            flashcard_uuids: ["fc-12"],
            error: null
          }
        ],
        created_flashcard_uuids: []
      })
      .mockResolvedValueOnce({
        attempt_id: 101,
        quiz_id: 7,
        target_deck: { id: 44, name: "Retry Deck" },
        results: [
          {
            question_id: 12,
            status: "superseded_and_created",
            conversion: {
              id: 903,
              attempt_id: 101,
              quiz_id: 7,
              question_id: 12,
              status: "active",
              orphaned: false,
              superseded_count: 1,
              target_deck_id: 44,
              target_deck_name_snapshot: "Retry Deck",
              flashcard_count: 1,
              flashcard_uuids_json: ["fc-12-new"],
              source_ref_id: "quiz-attempt:101:question:12",
              created_at: "2026-03-13T09:18:00Z",
              last_modified: "2026-03-13T09:18:00Z",
              client_id: "test",
              version: 1
            },
            flashcard_uuids: ["fc-12-new"],
            error: null
          }
        ],
        created_flashcard_uuids: ["fc-12-new"]
      })

    await openDetails()

    fireEvent.click(screen.getByRole("button", { name: /create remediation flashcards/i }))

    await waitFor(() => {
      expect(screen.getByText("Destination deck")).toBeInTheDocument()
    })

    const modal = screen.getAllByRole("dialog")[1]
    fireEvent.mouseDown(within(modal).getByRole("combobox"))
    fireEvent.click(await screen.findByText("Create new deck"))
    fireEvent.change(screen.getByTestId("quiz-remediation-new-deck-name"), {
      target: { value: "Retry Deck" }
    })
    fireEvent.click(within(modal).getByRole("checkbox", { name: /which structure filters blood/i }))
    fireEvent.click(screen.getByRole("button", { name: /^create flashcards$/i }))

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /convert again anyway/i })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /convert again anyway/i }))

    await waitFor(() => {
      expect(mocks.convertRemediationQuestions).toHaveBeenNthCalledWith(2, {
        attemptId: 101,
        request: {
          question_ids: [12],
          target_deck_id: 44,
          replace_active: true
        }
      })
    })
  }, 12000)

  it("drops the deck filter when active remediation conversions span multiple decks", async () => {
    vi.mocked(useAttemptRemediationConversionsQuery).mockReturnValue({
      data: {
        attempt_id: 101,
        items: [
          {
            id: 901,
            attempt_id: 101,
            quiz_id: 7,
            question_id: 12,
            status: "active",
            orphaned: false,
            superseded_count: 0,
            target_deck_id: 3,
            target_deck_name_snapshot: "Renal Recovery Deck",
            flashcard_count: 1,
            flashcard_uuids_json: ["fc-12"],
            source_ref_id: "quiz-attempt:101:question:12",
            created_at: "2026-03-13T09:15:00Z",
            last_modified: "2026-03-13T09:15:00Z",
            client_id: "test",
            version: 1
          },
          {
            id: 902,
            attempt_id: 101,
            quiz_id: 7,
            question_id: 13,
            status: "active",
            orphaned: false,
            superseded_count: 0,
            target_deck_id: 4,
            target_deck_name_snapshot: "Renal Backup Deck",
            flashcard_count: 1,
            flashcard_uuids_json: ["fc-13"],
            source_ref_id: "quiz-attempt:101:question:13",
            created_at: "2026-03-13T09:16:00Z",
            last_modified: "2026-03-13T09:16:00Z",
            client_id: "test",
            version: 1
          }
        ],
        count: 2,
        superseded_count: 0
      },
      isLoading: false
    } as any)
    vi.mocked(useDecksQuery).mockReturnValue({
      data: [
        { id: 3, name: "Renal Recovery Deck" },
        { id: 4, name: "Renal Backup Deck" }
      ],
      isLoading: false
    } as any)

    await openDetails()

    fireEvent.click(screen.getByRole("button", { name: /study linked cards/i }))

    expect(mocks.navigate).toHaveBeenCalledWith(
      "/flashcards?tab=review&study_source=quiz&quiz_id=7&attempt_id=101"
    )
  })

  it("recovers remediation assistant conflicts in place", async () => {
    mocks.assistantRespond
      .mockRejectedValueOnce(Object.assign(new Error("Version mismatch"), { response: { status: 409 } }))
      .mockRejectedValueOnce(Object.assign(new Error("Version mismatch"), { response: { status: 409 } }))
      .mockResolvedValueOnce({
        assistant_message: {
          id: 401,
          content: "Retried remediation response"
        }
      })
    await openDetails()

    fireEvent.click(screen.getByRole("button", { name: /explain mistake for question 12/i }))

    await waitFor(() => {
      expect(assistantRefetchMock).toHaveBeenCalled()
      expect(screen.getByText("Latest remediation context")).toBeInTheDocument()
      expect(screen.getByText("Conversation changed elsewhere.")).toBeInTheDocument()
      expect(screen.getByText("Quiz #7, question 12")).toBeInTheDocument()
    })
    fireEvent.click(screen.getByRole("button", { name: /explain mistake for question 12/i }))
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Retry my message" })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "Retry my message" }))

    await waitFor(() => {
      expect(mocks.assistantRespond).toHaveBeenNthCalledWith(3, {
        attemptId: 101,
        questionId: 12,
        request: { action: "explain" }
      })
    })
  }, 20000)
})
