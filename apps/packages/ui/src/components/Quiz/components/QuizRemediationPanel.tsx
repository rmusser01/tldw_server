import React from "react"
import { Button, Card, Checkbox, Empty, List, Space, Tag, Typography } from "antd"
import { BookOutlined, FileAddOutlined, MessageOutlined } from "@ant-design/icons"
import { useTranslation } from "react-i18next"
import { FlashcardStudyAssistantPanel } from "@/components/Flashcards/components"
import type { StudyAssistantRespondRequest } from "@/services/flashcards"
import {
  useQuizAttemptQuestionAssistantQuery,
  useQuizAttemptQuestionAssistantRespondMutation
} from "../hooks"

const { Text } = Typography

export type MissedQuestionEntry = {
  questionId: number
  questionText: string
  correctAnswerText: string
  userAnswerText: string
  explanation: string | null
  alreadyConverted: boolean
  orphaned: boolean
  convertedDeckName: string | null
  linkedFlashcardCount: number
  hasSupersededHistory: boolean
}

interface QuizRemediationPanelProps {
  attemptId: number
  quizId: number
  missedQuestionEntries: MissedQuestionEntry[]
  selectedMissedQuestions: Record<number, boolean>
  onSelectedMissedQuestionsChange: React.Dispatch<React.SetStateAction<Record<number, boolean>>>
  onCreateRemediationQuiz: (questionIds: number[]) => Promise<void>
  onCreateRemediationFlashcards: () => void
  onStudyLinkedCards: () => void
  remediationQuizPending?: boolean
}

const DEFAULT_AVAILABLE_ACTIONS = ["explain", "follow_up", "fact_check", "freeform"] as const

export const QuizRemediationPanel: React.FC<QuizRemediationPanelProps> = ({
  attemptId,
  quizId,
  missedQuestionEntries,
  selectedMissedQuestions,
  onSelectedMissedQuestionsChange,
  onCreateRemediationQuiz,
  onCreateRemediationFlashcards,
  onStudyLinkedCards,
  remediationQuizPending = false
}) => {
  const { t } = useTranslation(["option", "common"])
  const [activeQuestionId, setActiveQuestionId] = React.useState<number | null>(
    () => missedQuestionEntries[0]?.questionId ?? null
  )
  const [assistantAutoRequest, setAssistantAutoRequest] = React.useState<{
    token: number
    request: StudyAssistantRespondRequest
  } | null>(null)
  const assistantAutoRequestTokenRef = React.useRef(0)

  React.useEffect(() => {
    if (missedQuestionEntries.length === 0) {
      setActiveQuestionId(null)
      return
    }
    const activeStillPresent = activeQuestionId != null
      && missedQuestionEntries.some((entry) => entry.questionId === activeQuestionId)
    if (!activeStillPresent) {
      setActiveQuestionId(missedQuestionEntries[0]?.questionId ?? null)
    }
  }, [activeQuestionId, missedQuestionEntries])

  const selectedQuestionIds = React.useMemo(
    () => missedQuestionEntries
      .filter((entry) => selectedMissedQuestions[entry.questionId])
      .map((entry) => entry.questionId),
    [missedQuestionEntries, selectedMissedQuestions]
  )

  const activeQuestion = React.useMemo(
    () => missedQuestionEntries.find((entry) => entry.questionId === activeQuestionId) ?? null,
    [activeQuestionId, missedQuestionEntries]
  )

  const assistantQuery = useQuizAttemptQuestionAssistantQuery(
    attemptId,
    activeQuestionId,
    { enabled: activeQuestionId != null }
  )
  const assistantRespondMutation = useQuizAttemptQuestionAssistantRespondMutation()

  const handleExplainQuestion = React.useCallback(
    async (questionId: number) => {
      setActiveQuestionId(questionId)
      assistantAutoRequestTokenRef.current += 1
      setAssistantAutoRequest({
        token: assistantAutoRequestTokenRef.current,
        request: { action: "explain" }
      })
    },
    []
  )

  const handleAssistantRespond = React.useCallback(
    async (request: StudyAssistantRespondRequest) => {
      if (activeQuestionId == null) return null
      return await assistantRespondMutation.mutateAsync({
        attemptId,
        questionId: activeQuestionId,
        request
      })
    },
    [activeQuestionId, assistantRespondMutation, attemptId]
  )

  if (missedQuestionEntries.length === 0) {
    return (
      <Card size="small" title={t("option:quiz.remediationTitle", { defaultValue: "Remediation" })}>
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={t("option:quiz.noMissedQuestionsRemediation", {
            defaultValue: "No missed questions to review for this attempt."
          })}
        />
      </Card>
    )
  }

  return (
    <Card
      size="small"
      title={t("option:quiz.remediationTitle", { defaultValue: "Remediation" })}
      extra={(
        <Tag color="gold">
          {t("option:quiz.remediationQuestionCount", {
            defaultValue: "{{count}} missed",
            count: missedQuestionEntries.length
          })}
        </Tag>
      )}
    >
      <Space orientation="vertical" size={16} className="w-full">
        <div className="rounded border border-border bg-surface p-3">
          <Space orientation="vertical" size={12} className="w-full">
            <Checkbox
              checked={
                missedQuestionEntries.length > 0 &&
                selectedQuestionIds.length === missedQuestionEntries.length
              }
              indeterminate={
                selectedQuestionIds.length > 0 &&
                selectedQuestionIds.length < missedQuestionEntries.length
              }
              onChange={(event) => {
                const checked = event.target.checked
                onSelectedMissedQuestionsChange(
                  missedQuestionEntries.reduce<Record<number, boolean>>((acc, entry) => {
                    acc[entry.questionId] = checked
                    return acc
                  }, {})
                )
              }}
            >
              {t("option:quiz.selectAllMissedQuestions", {
                defaultValue: "Select all missed questions"
              })}
            </Checkbox>

            <List
              dataSource={missedQuestionEntries}
              renderItem={(entry) => (
                <List.Item>
                  <div className="w-full space-y-2">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div className="min-w-0 flex-1">
                        <Checkbox
                          aria-label={t("option:quiz.selectMissedQuestionAria", {
                            defaultValue: "Select missed question {{id}}",
                            id: entry.questionId
                          })}
                          checked={Boolean(selectedMissedQuestions[entry.questionId])}
                          onChange={(event) => {
                            const checked = event.target.checked
                            onSelectedMissedQuestionsChange((previous) => ({
                              ...previous,
                              [entry.questionId]: checked
                            }))
                          }}
                        >
                          <span className="font-medium">{entry.questionText}</span>
                        </Checkbox>
                        <div className="pl-6 pt-1 text-xs text-text-muted">
                          {t("option:quiz.correctAnswerLabel", { defaultValue: "Correct answer" })}:{" "}
                          {entry.correctAnswerText}
                        </div>
                        <div className="pl-6 text-xs text-text-subtle">
                          {t("option:quiz.yourAnswer", { defaultValue: "Your answer" })}:{" "}
                          {entry.userAnswerText}
                        </div>
                        {entry.orphaned ? (
                          <div className="pl-6 text-xs text-warning">
                            {t("option:quiz.linkedCardsDeleted", {
                              defaultValue: "Linked cards were deleted. Convert again to recreate them."
                            })}
                          </div>
                        ) : entry.alreadyConverted ? (
                          <div className="pl-6 text-xs text-text-subtle">
                            {entry.convertedDeckName
                              ? t("option:quiz.alreadyConvertedDeckLabel", {
                                  defaultValue: "Converted in deck {{deckName}}.",
                                  deckName: entry.convertedDeckName
                                })
                              : t("option:quiz.alreadyConvertedRemediation", {
                                  defaultValue: "Remediation flashcards already exist for this miss."
                                })}
                          </div>
                        ) : null}
                        {entry.hasSupersededHistory && (
                          <div className="pl-6 text-xs text-text-subtle">
                            {t("option:quiz.supersededRemediationHistory", {
                              defaultValue: "Superseded remediation history exists for this question."
                            })}
                          </div>
                        )}
                      </div>
                      <Button
                        size="small"
                        icon={<MessageOutlined />}
                        aria-label={t("option:quiz.explainMistakeAria", {
                          defaultValue: "Explain mistake for question {{id}}",
                          id: entry.questionId
                        })}
                        onClick={() => {
                          void handleExplainQuestion(entry.questionId)
                        }}
                      >
                        {t("option:quiz.explainMistake", { defaultValue: "Explain mistake" })}
                      </Button>
                    </div>
                  </div>
                </List.Item>
              )}
            />

            <div className="flex flex-wrap gap-2">
              <Button
                type="primary"
                icon={<FileAddOutlined />}
                disabled={selectedQuestionIds.length === 0}
                loading={remediationQuizPending}
                onClick={() => {
                  void onCreateRemediationQuiz(selectedQuestionIds)
                }}
              >
                {t("option:quiz.createRemediationQuiz", { defaultValue: "Create Remediation Quiz" })}
              </Button>
              <Button
                icon={<BookOutlined />}
                disabled={missedQuestionEntries.length === 0}
                onClick={onCreateRemediationFlashcards}
              >
                {t("option:quiz.createRemediationFlashcards", {
                  defaultValue: "Create Remediation Flashcards"
                })}
              </Button>
              <Button onClick={onStudyLinkedCards}>
                {t("option:quiz.studyLinkedCards", { defaultValue: "Study Linked Cards" })}
              </Button>
            </div>
            <Text type="secondary" className="text-xs">
              {t("option:quiz.remediationHint", {
                defaultValue: "Use the assistant to explain a miss, then turn selected misses into a follow-up quiz or flashcard deck."
              })}
            </Text>
          </Space>
        </div>

        {activeQuestion ? (
          <div className="space-y-3">
            <div>
              <Text strong>
                {t("option:quiz.activeRemediationQuestion", {
                  defaultValue: "Active review question"
                })}
              </Text>
              <div className="text-sm text-text-muted">
                {activeQuestion.questionText}
              </div>
              <div className="text-xs text-text-subtle">
                {t("option:quiz.quizQuestionReference", {
                  defaultValue: "Quiz #{{quizId}}, question {{questionId}}",
                  quizId,
                  questionId: activeQuestion.questionId
                })}
              </div>
            </div>
            <FlashcardStudyAssistantPanel
              cardUuid={`quiz-attempt-${attemptId}-question-${activeQuestion.questionId}`}
              threadVersion={assistantQuery.data?.thread.version ?? null}
              messages={assistantQuery.data?.messages ?? []}
              availableActions={assistantQuery.data?.available_actions ?? [...DEFAULT_AVAILABLE_ACTIONS]}
              isLoading={assistantQuery.isLoading}
              isError={assistantQuery.isError}
              queryError={assistantQuery.error}
              isResponding={assistantRespondMutation.isPending}
              onReloadContext={() => assistantQuery.refetch()}
              onRespond={handleAssistantRespond}
              autoSubmitRequest={assistantAutoRequest}
            />
          </div>
        ) : null}
      </Space>
    </Card>
  )
}

export default QuizRemediationPanel
