import React from "react"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router-dom"
import { useServerOnline } from "@/hooks/useServerOnline"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { useDemoMode } from "@/context/demo-mode"
import { getDesignSystemState } from "@/design-system"
import { useScrollToServerCard } from "@/hooks/useScrollToServerCard"
import {
  useConnectionActions,
  useConnectionUxState
} from "@/hooks/useConnectionState"
import FeatureEmptyState from "@/components/Common/FeatureEmptyState"
import ConnectionProblemBanner from "@/components/Common/ConnectionProblemBanner"
import { StatusBadge } from "@/components/Common/StatusBadge"
import {
  getDemoQuizzes,
  type DemoQuiz,
  type DemoQuizQuestion
} from "@/utils/demo-content"
import { QuizPlayground } from "./QuizPlayground"
import { useQuizzesQuery } from "./hooks"

type DemoAnswerMap = Record<string, string>
type DemoPreviewMode = "catalog" | "taking" | "results"

const QUIZ_BETA_TOOLTIP_ID = "quiz-beta-tooltip"

const normalizeDemoAnswer = (value: string | undefined): string => value?.trim().toLowerCase() ?? ""

const isDemoQuestionCorrect = (question: DemoQuizQuestion, answer: string | undefined): boolean => {
  if (!answer) return false

  if (question.type === "fill_blank") {
    const normalized = normalizeDemoAnswer(answer)
    return question.acceptedAnswers.some(
      (candidate) => normalizeDemoAnswer(candidate) === normalized
    )
  }

  return normalizeDemoAnswer(answer) === normalizeDemoAnswer(question.correctAnswer)
}

const QuizBetaBadge: React.FC<{ label: string; description: string }> = ({
  label,
  description
}) => {
  const [isOpen, setIsOpen] = React.useState(false)

  React.useEffect(() => {
    if (!isOpen) return

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsOpen(false)
      }
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => {
      window.removeEventListener("keydown", handleKeyDown)
    }
  }, [isOpen])

  return (
    <div className="relative inline-flex items-center">
      <button
        type="button"
        data-testid="quiz-beta-badge"
        className="rounded-full focus:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2 focus-visible:ring-offset-bg"
        aria-label={description}
        aria-expanded={isOpen}
        aria-describedby={isOpen ? QUIZ_BETA_TOOLTIP_ID : undefined}
        onMouseEnter={() => setIsOpen(true)}
        onMouseLeave={() => setIsOpen(false)}
        onFocus={() => setIsOpen(true)}
        onBlur={() => setIsOpen(false)}
        onClick={() => setIsOpen((previous) => !previous)}
      >
        <StatusBadge variant="warning">{label}</StatusBadge>
      </button>
      {isOpen ? (
        <div
          id={QUIZ_BETA_TOOLTIP_ID}
          role="tooltip"
          data-testid="quiz-beta-tooltip"
          className="absolute right-0 top-[calc(100%+8px)] z-10 w-72 rounded-lg border border-border bg-surface p-2 text-left text-xs text-text shadow-card"
        >
          {description}
        </div>
      ) : null}
    </div>
  )
}

const DemoQuizPreview: React.FC<{ quizzes: DemoQuiz[] }> = ({ quizzes }) => {
  const [selectedQuizId, setSelectedQuizId] = React.useState<string | null>(
    quizzes[0]?.id ?? null
  )
  const [mode, setMode] = React.useState<DemoPreviewMode>("catalog")
  const [questionIndex, setQuestionIndex] = React.useState(0)
  const [answers, setAnswers] = React.useState<DemoAnswerMap>({})
  const [scoreSummary, setScoreSummary] = React.useState<{
    correctCount: number
    total: number
    percent: number
    passed: boolean
  } | null>(null)

  const selectedQuiz = React.useMemo(
    () => quizzes.find((quiz) => quiz.id === selectedQuizId) ?? quizzes[0] ?? null,
    [quizzes, selectedQuizId]
  )

  React.useEffect(() => {
    if (!selectedQuiz && quizzes.length > 0) {
      setSelectedQuizId(quizzes[0].id)
    }
  }, [selectedQuiz, quizzes])

  const activeQuestion =
    mode === "taking" && selectedQuiz
      ? selectedQuiz.questions[questionIndex]
      : null

  const handleStart = React.useCallback(() => {
    setMode("taking")
    setQuestionIndex(0)
    setAnswers({})
    setScoreSummary(null)
  }, [])

  const handleSubmit = React.useCallback(() => {
    if (!selectedQuiz) return

    const total = selectedQuiz.questions.length
    const correctCount = selectedQuiz.questions.reduce((count, question) => {
      return count + (isDemoQuestionCorrect(question, answers[question.id]) ? 1 : 0)
    }, 0)
    const percent = total > 0 ? Math.round((correctCount / total) * 100) : 0
    const passed = percent >= selectedQuiz.passingScore

    setScoreSummary({
      correctCount,
      total,
      percent,
      passed
    })
    setMode("results")
  }, [answers, selectedQuiz])

  if (!selectedQuiz) return null

  if (mode === "catalog") {
    return (
      <section
        data-testid="quiz-demo-preview"
        className="rounded-2xl border border-dashed border-border bg-surface p-4"
      >
        <h3 className="text-sm font-semibold text-text">
          Demo quiz preview (local only)
        </h3>
        <p className="mt-1 text-xs text-text-muted">
          Pick a sample quiz and run through the full take-submit-review flow with local data.
        </p>

        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          {quizzes.map((quiz) => {
            const isSelected = quiz.id === selectedQuiz.id
            return (
              <button
                key={quiz.id}
                type="button"
                data-testid={`quiz-demo-card-${quiz.id}`}
                className={`rounded-lg border px-3 py-2 text-left transition ${
                  isSelected
                    ? "border-primary bg-primary/5"
                    : "border-border bg-bg hover:border-primary/50"
                }`}
                onClick={() => setSelectedQuizId(quiz.id)}
              >
                <div className="text-sm font-medium text-text">{quiz.title}</div>
                <div className="mt-1 text-xs text-text-muted">{quiz.description}</div>
                <div className="mt-2 text-[11px] text-text-muted">
                  {quiz.questions.length} questions · {quiz.timeLimitMinutes} min · pass at{" "}
                  {quiz.passingScore}%
                </div>
              </button>
            )
          })}
        </div>

        <div className="mt-3 rounded-lg border border-border/70 bg-bg p-3">
          <div className="text-xs font-semibold uppercase tracking-[0.08em] text-text-muted">
            Selected sample
          </div>
          <div className="mt-1 text-sm font-medium text-text">{selectedQuiz.title}</div>
          <div className="mt-1 text-xs text-text-muted">{selectedQuiz.sourceLabel}</div>
          <div className="mt-2 text-xs text-text-muted">
            Difficulty: {selectedQuiz.difficulty} · Timer: {selectedQuiz.timeLimitMinutes} minutes
          </div>
        </div>

        <div className="mt-3 flex flex-wrap gap-2">
          <button
            type="button"
            data-testid="quiz-demo-start"
            className="rounded-full bg-primary px-4 py-2 text-xs font-semibold uppercase tracking-[0.08em] text-bg transition hover:bg-primary/90"
            onClick={handleStart}
          >
            Start demo quiz
          </button>
          <span className="inline-flex items-center rounded-full border border-border px-3 py-1 text-[11px] text-text-muted">
            Answers stay in this tab only
          </span>
        </div>
      </section>
    )
  }

  if (mode === "taking" && activeQuestion) {
    const isLastQuestion = questionIndex >= selectedQuiz.questions.length - 1
    const currentAnswer = answers[activeQuestion.id] ?? ""

    return (
      <section
        data-testid="quiz-demo-taking"
        className="rounded-2xl border border-border bg-surface p-4"
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-text">
            {selectedQuiz.title}
          </h3>
          <span className="text-xs text-text-muted">
            Question {questionIndex + 1} / {selectedQuiz.questions.length}
          </span>
        </div>

        <div className="mt-3 rounded-lg border border-border/70 bg-bg p-3">
          <p className="text-sm text-text">{activeQuestion.prompt}</p>

          {activeQuestion.type === "multiple_choice" ? (
            <div className="mt-3 space-y-2">
              {activeQuestion.options.map((option) => (
                <label key={option} className="flex items-start gap-2 text-sm text-text">
                  <input
                    type="radio"
                    name={activeQuestion.id}
                    checked={currentAnswer === option}
                    onChange={() =>
                      setAnswers((previous) => ({
                        ...previous,
                        [activeQuestion.id]: option
                      }))
                    }
                  />
                  <span>{option}</span>
                </label>
              ))}
            </div>
          ) : null}

          {activeQuestion.type === "true_false" ? (
            <div className="mt-3 space-y-2">
              {["true", "false"].map((option) => (
                <label key={option} className="flex items-start gap-2 text-sm text-text">
                  <input
                    type="radio"
                    name={activeQuestion.id}
                    checked={currentAnswer === option}
                    onChange={() =>
                      setAnswers((previous) => ({
                        ...previous,
                        [activeQuestion.id]: option
                      }))
                    }
                  />
                  <span>{option === "true" ? "True" : "False"}</span>
                </label>
              ))}
            </div>
          ) : null}

          {activeQuestion.type === "fill_blank" ? (
            <div className="mt-3">
              <label className="sr-only" htmlFor={`demo-${activeQuestion.id}`}>
                Demo answer
              </label>
              <input
                id={`demo-${activeQuestion.id}`}
                className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text"
                type="text"
                value={currentAnswer}
                placeholder={activeQuestion.placeholder}
                onChange={(event) =>
                  setAnswers((previous) => ({
                    ...previous,
                    [activeQuestion.id]: event.target.value
                  }))
                }
              />
            </div>
          ) : null}
        </div>

        <div className="mt-3 flex flex-wrap gap-2">
          {selectedQuiz.questions.map((question, index) => {
            const answered = Boolean(answers[question.id]?.trim())
            return (
              <button
                key={question.id}
                type="button"
                className={`h-8 w-8 rounded-full border text-xs ${
                  index === questionIndex
                    ? "border-primary bg-primary/10 text-primary"
                    : answered
                      ? "border-success bg-success/10 text-success"
                      : "border-border text-text-muted"
                }`}
                onClick={() => setQuestionIndex(index)}
                aria-label={`Go to question ${index + 1}`}
              >
                {index + 1}
              </button>
            )
          })}
        </div>

        <div className="mt-4 flex flex-wrap gap-2">
          <button
            type="button"
            className="rounded-full border border-border px-3 py-2 text-xs font-semibold uppercase tracking-[0.08em] text-text-muted disabled:cursor-not-allowed disabled:opacity-60"
            onClick={() => setQuestionIndex((value) => Math.max(0, value - 1))}
            disabled={questionIndex === 0}
          >
            Previous
          </button>
          {!isLastQuestion ? (
            <button
              type="button"
              className="rounded-full border border-border px-3 py-2 text-xs font-semibold uppercase tracking-[0.08em] text-text"
              onClick={() =>
                setQuestionIndex((value) =>
                  Math.min(selectedQuiz.questions.length - 1, value + 1)
                )
              }
            >
              Next
            </button>
          ) : (
            <button
              type="button"
              data-testid="quiz-demo-submit"
              className="rounded-full bg-primary px-4 py-2 text-xs font-semibold uppercase tracking-[0.08em] text-bg transition hover:bg-primary/90"
              onClick={handleSubmit}
            >
              Submit demo quiz
            </button>
          )}
          <button
            type="button"
            className="rounded-full border border-border px-3 py-2 text-xs font-semibold uppercase tracking-[0.08em] text-text-muted"
            onClick={() => setMode("catalog")}
          >
            Back to samples
          </button>
        </div>
      </section>
    )
  }

  return (
    <section
      data-testid="quiz-demo-results"
      className="rounded-2xl border border-border bg-surface p-4"
    >
      <h3 className="text-sm font-semibold text-text">Demo results</h3>
      <p className="mt-1 text-sm text-text" data-testid="quiz-demo-score">
        Score: {scoreSummary?.correctCount ?? 0} / {scoreSummary?.total ?? 0} (
        {scoreSummary?.percent ?? 0}%)
      </p>
      <p className="mt-1 text-xs text-text-muted">
        {scoreSummary?.passed
          ? `Pass (threshold ${selectedQuiz.passingScore}%)`
          : `Needs review (threshold ${selectedQuiz.passingScore}%)`}
      </p>

      <div className="mt-3 space-y-2">
        {selectedQuiz.questions.map((question, index) => {
          const correct = isDemoQuestionCorrect(question, answers[question.id])
          return (
            <div
              key={question.id}
              className="rounded-md border border-border/70 bg-bg px-3 py-2 text-xs text-text"
            >
              <div className="font-semibold">
                Q{index + 1}: {correct ? "Correct" : "Needs review"}
              </div>
              <div className="mt-1 text-text-muted">{question.explanation}</div>
            </div>
          )
        })}
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        <button
          type="button"
          className="rounded-full bg-primary px-4 py-2 text-xs font-semibold uppercase tracking-[0.08em] text-bg transition hover:bg-primary/90"
          onClick={handleStart}
        >
          Retake demo quiz
        </button>
        <button
          type="button"
          className="rounded-full border border-border px-3 py-2 text-xs font-semibold uppercase tracking-[0.08em] text-text-muted"
          onClick={() => setMode("catalog")}
        >
          Choose another sample
        </button>
      </div>
    </section>
  )
}

const InlineConnectionWarning = ({
  message,
  retryActionLabel,
  onRetry,
  retryDisabled,
  testId
}: {
  message: string
  retryActionLabel?: string
  onRetry?: () => void
  retryDisabled?: boolean
  testId: string
}) => (
  <div
    data-testid={testId}
    className="rounded-2xl border border-warn/40 bg-warn/10 px-4 py-3 text-sm text-text"
  >
    <div className="font-medium">{message}</div>
    {retryActionLabel && onRetry ? (
      <div className="mt-2 flex justify-start text-xs">
        <button
          type="button"
          onClick={onRetry}
          disabled={retryDisabled}
          className="inline-flex items-center gap-1 text-primary hover:text-primary disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:text-primary"
        >
          {retryActionLabel}
        </button>
      </div>
    ) : null}
  </div>
)

/**
 * QuizWorkspace handles connection state, demo mode, and feature availability.
 * When online and feature is available, it renders QuizPlayground.
 */
export const QuizWorkspace: React.FC = () => {
  const { t } = useTranslation(["option", "common", "settings"])
  const navigate = useNavigate()
  const isOnline = useServerOnline()
  const { demoEnabled } = useDemoMode()
  const { uxState, hasCompletedFirstRun } = useConnectionUxState()
  const { capabilities, loading: capsLoading } = useServerCapabilities()
  const scrollToServerCard = useScrollToServerCard("/quiz")
  const { checkOnce } = useConnectionActions()
  const [checkingConnection, setCheckingConnection] = React.useState(false)

  const quizzesUnsupported = !capsLoading && !!capabilities && !capabilities.hasQuizzes
  const demoQuizzes = React.useMemo(() => getDemoQuizzes(t), [t])

  // Fetch quiz count for FTUX decisions (deduped by react-query when QuizPlayground also queries)
  const { data: quizCountData, isLoading: isQuizCountLoading } = useQuizzesQuery(
    { limit: 1, offset: 0 },
    { enabled: isOnline && !quizzesUnsupported }
  )
  const totalQuizzes = quizCountData?.count ?? 0
  const [showDemo, setShowDemo] = React.useState(false)

  // Item 5: Split beta tooltip by context (connected vs demo)
  const betaDescriptionConnected = t("option:quiz.betaDescriptionConnected", {
    defaultValue:
      "Quiz Playground is in beta. Features and scoring may evolve. Your quiz data is saved to your server."
  })
  const betaDescriptionDemo = t("option:quiz.betaDescriptionDemo", {
    defaultValue:
      "Quiz Playground is in beta. Demo answers are local to this session and will not be saved."
  })
  const betaDescription = isOnline ? betaDescriptionConnected : betaDescriptionDemo

  const handleRetryConnection = React.useCallback(() => {
    if (checkingConnection) return
    setCheckingConnection(true)
    Promise.resolve(checkOnce())
      .catch(() => {
        // errors are surfaced via connection UX state
      })
      .finally(() => {
        setCheckingConnection(false)
      })
  }, [checkOnce, checkingConnection])

  const offlineBannerProps = React.useMemo(() => {
    if (uxState === "error_auth" || uxState === "configuring_auth") {
      return {
        badgeLabel: "Credentials required",
        title: t("option:quiz.authTitle", {
          defaultValue: "Add your credentials to use Quiz Playground"
        }),
        description: t("option:quiz.authDescription", {
          defaultValue:
            "Quiz Playground needs valid credentials before it can create or review quizzes against your tldw server."
        }),
        examples: [
          t("option:quiz.authExample1", {
            defaultValue:
              "Use the connection card at the top of this page to repair your server URL or API key."
          })
        ],
        primaryActionLabel: t("option:connectionCard.buttonGoToServerCard", {
          defaultValue: "Go to server card"
        }),
        onPrimaryAction: scrollToServerCard
      }
    }
    if (uxState === "unconfigured" || uxState === "configuring_url") {
      const setupRequiredState = getDesignSystemState("setup_required")

      return {
        badgeLabel: setupRequiredState.label,
        title: t("option:quiz.setupTitle", {
          defaultValue: "Finish setup to use Quiz Playground"
        }),
        description: hasCompletedFirstRun
          ? t("option:quiz.setupDescriptionReturning", {
              defaultValue:
                "Quiz Playground still needs a configured tldw server before it can load your real quizzes."
            })
          : t("option:quiz.setupDescriptionFirstRun", {
              defaultValue:
                "Finish connecting your tldw server before Quiz Playground can load your real quizzes."
            }),
        examples: [
          t("option:quiz.setupExample1", {
            defaultValue:
              "Use the connection card at the top of this page to finish server setup."
          })
        ],
        primaryActionLabel: t("option:connectionCard.buttonGoToServerCard", {
          defaultValue: "Go to server card"
        }),
        onPrimaryAction: scrollToServerCard
      }
    }
    if (uxState === "error_unreachable") {
      return {
        badgeLabel: "Server unreachable",
        title: t("option:quiz.unreachableTitle", {
          defaultValue: "Can't reach your tldw server right now"
        }),
        description: t("option:quiz.unreachableDescription", {
          defaultValue:
            "Quiz Playground depends on a reachable tldw server. Review the server card above, then retry the connection check."
        }),
        examples: [
          t("option:quiz.unreachableExample1", {
            defaultValue:
              "If your server is running, confirm the URL in the connection card still points to the right host."
          })
        ],
        primaryActionLabel: t("option:connectionCard.buttonGoToServerCard", {
          defaultValue: "Go to server card"
        }),
        onPrimaryAction: scrollToServerCard,
        retryActionLabel: t("option:buttonRetry", "Retry connection"),
        onRetry: handleRetryConnection,
        retryDisabled: checkingConnection
      }
    }
    return {
      badgeLabel: "Not connected",
      title: t("option:quiz.emptyConnectTitle", {
        defaultValue: "Connect to use Quiz Playground"
      }),
      description: t("option:quiz.emptyConnectDescription", {
        defaultValue:
          "This view needs a connected server. Use the server connection card above to fix your connection, then return here to create and take quizzes."
      }),
      examples: [
        t("option:quiz.emptyConnectExample1", {
          defaultValue:
            "Use the connection card at the top of this page to add your server URL and API key."
        })
      ],
      primaryActionLabel: t("option:connectionCard.buttonGoToServerCard", {
        defaultValue: "Go to server card"
      }),
      onPrimaryAction: scrollToServerCard,
      retryActionLabel: t("option:buttonRetry", "Retry connection"),
      onRetry: handleRetryConnection,
      retryDisabled: checkingConnection
    }
  }, [
    checkingConnection,
    handleRetryConnection,
    hasCompletedFirstRun,
    scrollToServerCard,
    t,
    uxState
  ])

  const demoConnectionWarning = React.useMemo(() => {
    if (uxState === "error_auth" || uxState === "configuring_auth") {
      return {
        message:
          "Demo stays available, but your Quiz Playground credentials need attention."
      }
    }
    if (uxState === "unconfigured" || uxState === "configuring_url") {
      return {
        message: "Demo stays available while you finish Quiz Playground setup."
      }
    }
    if (uxState === "error_unreachable") {
      return {
        message: "Demo stays available, but your tldw server is unreachable.",
        retryActionLabel: t("option:buttonRetry", "Retry connection"),
        onRetry: handleRetryConnection,
        retryDisabled: checkingConnection
      }
    }
    return null
  }, [checkingConnection, handleRetryConnection, t, uxState])

  // Offline state - show demo or connection banner
  if (!isOnline) {
    return demoEnabled ? (
      <div className="space-y-4">
        <div className="flex justify-end">
          <QuizBetaBadge
            label={t("common:beta", { defaultValue: "Beta" })}
            description={betaDescription}
          />
        </div>
        {demoConnectionWarning ? (
          <InlineConnectionWarning
            testId="quiz-demo-connection-warning"
            message={demoConnectionWarning.message}
            retryActionLabel={demoConnectionWarning.retryActionLabel}
            onRetry={demoConnectionWarning.onRetry}
            retryDisabled={demoConnectionWarning.retryDisabled}
          />
        ) : null}
        <FeatureEmptyState
          title={
            <span className="inline-flex items-center gap-2">
              <StatusBadge variant="demo">Demo</StatusBadge>
              <span>
                {t("option:quiz.demoTitle", {
                  defaultValue: "Explore Quiz Playground in demo mode"
                })}
              </span>
            </span>
          }
          description={t("option:quiz.demoDescription", {
            defaultValue:
              "This demo shows how Quiz Playground can help you create and take quizzes from your content. Connect your own server later to generate quizzes from your media."
          })}
          examples={[
            t("option:quiz.demoExample1", {
              defaultValue:
                "Generate quizzes automatically from videos, articles, or documents."
            }),
            t("option:quiz.demoExample2", {
              defaultValue:
                "Create custom quizzes with multiple choice, true/false, and fill-in-the-blank questions."
            }),
            t("option:quiz.demoExample3", {
              defaultValue:
                "Track your quiz results and review incorrect answers to improve retention."
            })
          ]}
          primaryActionLabel={t("option:connectionCard.buttonGoToServerCard", {
            defaultValue: "Go to server card"
          })}
          onPrimaryAction={scrollToServerCard}
        />
        <DemoQuizPreview quizzes={demoQuizzes} />
      </div>
    ) : (
        <div className="space-y-4">
          <div className="flex justify-end">
            <QuizBetaBadge
              label={t("common:beta", { defaultValue: "Beta" })}
              description={betaDescription}
            />
          </div>
          <ConnectionProblemBanner
            badgeLabel={offlineBannerProps.badgeLabel}
            title={offlineBannerProps.title}
            description={offlineBannerProps.description}
            examples={offlineBannerProps.examples}
            primaryActionLabel={offlineBannerProps.primaryActionLabel}
            onPrimaryAction={offlineBannerProps.onPrimaryAction}
            retryActionLabel={offlineBannerProps.retryActionLabel}
            onRetry={offlineBannerProps.onRetry}
            retryDisabled={offlineBannerProps.retryDisabled}
          />
        </div>
      )
  }

  // Feature not supported on this server
  if (quizzesUnsupported) {
    const specVersion = capabilities?.specVersion ?? "unknown"

    return (
      <div className="space-y-4">
        <div className="flex justify-end">
          <QuizBetaBadge
            label={t("common:beta", { defaultValue: "Beta" })}
            description={betaDescription}
          />
        </div>
        <FeatureEmptyState
          title={
            <span className="inline-flex items-center gap-2">
              <StatusBadge variant="error">Feature unavailable</StatusBadge>
              <span>
                {t("option:quiz.offlineTitle", {
                  defaultValue: "Quiz API not available on this server"
                })}
              </span>
            </span>
          }
          description={t("option:quiz.offlineDescription", {
            defaultValue:
              "This server does not advertise Quiz endpoints in its capability spec (reported: {{specVersion}}). Quizzes require tldw_server v0.1.0+ with /api/v1/quizzes enabled.",
            specVersion
          })}
          examples={[
            t("option:quiz.offlineExample1", {
              defaultValue:
                "Open Health & diagnostics to verify server version and advertised API routes."
            }),
            t("option:quiz.offlineExample2", {
              defaultValue:
                "Update the server, then refresh this page so capabilities are re-detected."
            }),
            t("option:quiz.offlineExample3", {
              defaultValue:
                "If you self-host, confirm /api/v1/quizzes appears in /openapi.json."
            })
          ]}
          primaryActionLabel={t("settings:healthSummary.diagnostics", {
            defaultValue: "Health & diagnostics"
          })}
          onPrimaryAction={() => navigate("/settings/health")}
          secondaryActionLabel={t("option:quiz.offlineSecondaryAction", {
            defaultValue: "Open setup guide"
          })}
          onSecondaryAction={() => navigate("/documentation")}
        />
      </div>
    )
  }

  // Online and feature supported - render main playground
  return (
    <div className="space-y-3">
      <div className="flex justify-end">
        <QuizBetaBadge
          label={t("common:beta", { defaultValue: "Beta" })}
          description={betaDescription}
        />
      </div>
      <QuizPlayground />
      {!isQuizCountLoading && totalQuizzes === 0 && (
        <div className="mt-4 text-center" data-testid="quiz-connected-demo-toggle">
          {showDemo ? (
            <>
              <button
                type="button"
                className="mb-3 text-sm text-primary hover:text-primary/80"
                onClick={() => setShowDemo(false)}
              >
                {t("option:quiz.hideDemoQuiz", { defaultValue: "Hide demo" })}
              </button>
              <DemoQuizPreview quizzes={demoQuizzes} />
            </>
          ) : (
            <button
              type="button"
              className="text-sm text-primary hover:text-primary/80"
              data-testid="quiz-try-demo-button"
              onClick={() => setShowDemo(true)}
            >
              {t("option:quiz.tryDemoQuiz", {
                defaultValue: "Try a demo quiz to see how it works"
              })}
            </button>
          )}
        </div>
      )}
    </div>
  )
}

export default QuizWorkspace
