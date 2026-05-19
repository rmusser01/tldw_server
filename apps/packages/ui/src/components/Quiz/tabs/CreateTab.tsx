import React from "react"
import {
  Alert,
  Button,
  Card,
  Checkbox,
  Form,
  Input,
  InputNumber,
  Modal,
  Progress,
  Select,
  Space,
  Switch,
  Typography,
  message
} from "antd"
import { useTranslation } from "react-i18next"
import {
  PlusOutlined,
  DeleteOutlined,
  SaveOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
  MinusCircleOutlined
} from "@ant-design/icons"
import { useCreateQuizMutation, useCreateQuestionMutation } from "../hooks"
import type { QuestionType, QuestionCreate } from "@/services/quizzes"
import type { TakeTabNavigationIntent } from "../navigation"
import { normalizeMatchingAnswerMap } from "../utils/matchingAnswer"
import { checkStorageBeforeWrite, notifyStorageWrite } from "@/utils/storage-guard"
import { estimateStorageCost } from "@/utils/storage-budget"

const CREATE_ORIENTATION_DISMISSED_KEY = "tldw_quiz_create_orientation_dismissed"

interface CreateTabProps {
  onNavigateToTake: (intent?: TakeTabNavigationIntent) => void
  onDirtyStateChange?: (dirty: boolean) => void
}

interface QuestionFormData {
  key: string
  question_type: QuestionType
  question_text: string
  options: string[]
  correct_answer: number | string | number[] | Record<string, string>
  explanation?: string
  hint?: string
  hint_penalty_points?: number
}

type QuizCreateDraft = {
  name: string
  description: string
  timeLimit: number | null
  passingScore: number | null
  questions: QuestionFormData[]
  updatedAt: number
}

const CREATE_TAB_DRAFT_KEY = "quiz-create-draft-v1"
const MIN_MULTIPLE_CHOICE_OPTIONS = 2
const MAX_MULTIPLE_CHOICE_OPTIONS = 6
const MIN_MATCHING_PAIRS = 2
const MAX_MATCHING_PAIRS = 8
const TOUCH_TARGET_CLASS = "min-h-11 px-4"

type SaveProgressState = {
  phase: "creating_quiz" | "saving_questions"
  current: number
  total: number
}

class QuestionSaveError extends Error {
  questionNumber: number
  totalQuestions: number

  constructor(questionNumber: number, totalQuestions: number) {
    super("Failed to save question during quiz creation")
    this.name = "QuestionSaveError"
    this.questionNumber = questionNumber
    this.totalQuestions = totalQuestions
  }
}

const isFormValidationError = (error: unknown): boolean => {
  if (!error || typeof error !== "object") return false
  const maybeValidationError = error as { errorFields?: unknown }
  return Array.isArray(maybeValidationError.errorFields)
}

const readCreateDraft = (): QuizCreateDraft | null => {
  if (typeof window === "undefined") return null
  try {
    const raw = window.localStorage.getItem(CREATE_TAB_DRAFT_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as Partial<QuizCreateDraft>
    if (!Array.isArray(parsed.questions)) return null
    return {
      name: typeof parsed.name === "string" ? parsed.name : "",
      description: typeof parsed.description === "string" ? parsed.description : "",
      timeLimit: typeof parsed.timeLimit === "number" ? parsed.timeLimit : null,
      passingScore: typeof parsed.passingScore === "number" ? parsed.passingScore : null,
      questions: parsed.questions as QuestionFormData[],
      updatedAt: typeof parsed.updatedAt === "number" ? parsed.updatedAt : Date.now()
    }
  } catch {
    return null
  }
}

const writeCreateDraft = (draft: QuizCreateDraft): boolean | { saved: boolean; recommendation: string | null } => {
  if (typeof window === "undefined") return false
  try {
    const serialized = JSON.stringify(draft)
    const guard = checkStorageBeforeWrite(estimateStorageCost(serialized), CREATE_TAB_DRAFT_KEY)
    // Advisory only — always attempt the write
    window.localStorage.setItem(CREATE_TAB_DRAFT_KEY, serialized)
    notifyStorageWrite()
    if (guard.recommendation) {
      return { saved: true, recommendation: guard.recommendation }
    }
    return true
  } catch {
    return false
  }
}

const clearCreateDraft = (): void => {
  if (typeof window === "undefined") return
  try {
    window.localStorage.removeItem(CREATE_TAB_DRAFT_KEY)
  } catch {
    // ignore storage clear failures
  }
}

export const CreateTab: React.FC<CreateTabProps> = ({ onNavigateToTake, onDirtyStateChange }) => {
  const { t } = useTranslation(["option", "common"])
  const [form] = Form.useForm()
  const [questions, setQuestions] = React.useState<QuestionFormData[]>([])
  const [messageApi, contextHolder] = message.useMessage()
  const [pendingDraft, setPendingDraft] = React.useState<QuizCreateDraft | null>(null)
  const [draftStorageUnavailable, setDraftStorageUnavailable] = React.useState(false)
  const [draftWarningDismissed, setDraftWarningDismissed] = React.useState(false)
  const [orientationDismissed, setOrientationDismissed] = React.useState(() => {
    try {
      return window.localStorage.getItem(CREATE_ORIENTATION_DISMISSED_KEY) === "1"
    } catch {
      return false
    }
  })
  const [previewOpen, setPreviewOpen] = React.useState(false)
  const [saveProgress, setSaveProgress] = React.useState<SaveProgressState | null>(null)
  const hasLoadedDraft = React.useRef(false)
  const lastRecommendationRef = React.useRef<string | null>(null)

  const createQuizMutation = useCreateQuizMutation()
  const createQuestionMutation = useCreateQuestionMutation()

  const nameValue = Form.useWatch("name", form) as string | undefined
  const descriptionValue = Form.useWatch("description", form) as string | undefined
  const timeLimitValue = Form.useWatch("timeLimit", form) as number | undefined
  const passingScoreValue = Form.useWatch("passingScore", form) as number | undefined
  const isSaving = Boolean(saveProgress) || createQuizMutation.isPending || createQuestionMutation.isPending

  const isDirty = React.useMemo(() => {
    const hasQuizDetails = Boolean(
      (nameValue ?? "").trim() ||
      (descriptionValue ?? "").trim() ||
      timeLimitValue != null ||
      passingScoreValue != null
    )
    const hasQuestionData = questions.some((question) => {
      if (question.question_text.trim()) return true
      if ((question.explanation ?? "").trim()) return true
      if ((question.hint ?? "").trim()) return true
      if ((question.hint_penalty_points ?? 0) > 0) return true
      if (question.question_type === "fill_blank") {
        return String(question.correct_answer ?? "").trim().length > 0
      }
      if (question.question_type === "multiple_choice") {
        return question.options.some((option) => option.trim().length > 0)
      }
      if (question.question_type === "multi_select") {
        return question.options.some((option) => option.trim().length > 0)
      }
      if (question.question_type === "matching") {
        const map = normalizeMatchingAnswerMap(question.correct_answer)
        return (
          question.options.some((option) => option.trim().length > 0) ||
          Object.keys(map).length > 0
        )
      }
      return true
    })
    return hasQuizDetails || hasQuestionData || questions.length > 0
  }, [descriptionValue, nameValue, passingScoreValue, questions, timeLimitValue])

  const currentDraft = React.useMemo<QuizCreateDraft>(() => ({
    name: (nameValue ?? "").trim(),
    description: descriptionValue ?? "",
    timeLimit: typeof timeLimitValue === "number" ? timeLimitValue : null,
    passingScore: typeof passingScoreValue === "number" ? passingScoreValue : null,
    questions,
    updatedAt: Date.now()
  }), [descriptionValue, nameValue, passingScoreValue, questions, timeLimitValue])

  React.useEffect(() => {
    if (hasLoadedDraft.current) return
    hasLoadedDraft.current = true
    const existingDraft = readCreateDraft()
    if (!existingDraft) return
    if (existingDraft.name || existingDraft.description || existingDraft.questions.length > 0) {
      setPendingDraft(existingDraft)
    }
  }, [])

  React.useEffect(() => {
    onDirtyStateChange?.(isDirty)
  }, [isDirty, onDirtyStateChange])

  React.useEffect(() => {
    if (pendingDraft) return
    if (!isDirty) {
      clearCreateDraft()
      return
    }

    const timeoutId = window.setTimeout(() => {
      const result = writeCreateDraft(currentDraft)
      if (!result) {
        setDraftStorageUnavailable(true)
        lastRecommendationRef.current = null
      } else {
        setDraftStorageUnavailable(false)
        if (typeof result === "object" && result.recommendation) {
          if (lastRecommendationRef.current !== result.recommendation) {
            lastRecommendationRef.current = result.recommendation
            messageApi.warning(result.recommendation)
          }
        } else {
          lastRecommendationRef.current = null
        }
      }
    }, 300)

    return () => {
      window.clearTimeout(timeoutId)
    }
  }, [currentDraft, isDirty, messageApi, pendingDraft])

  React.useEffect(() => {
    if (!isDirty) return
    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault()
      event.returnValue = ""
    }
    window.addEventListener("beforeunload", handleBeforeUnload)
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload)
    }
  }, [isDirty])

  const addQuestion = () => {
    const newQuestion: QuestionFormData = {
      key: crypto.randomUUID(),
      question_type: "multiple_choice",
      question_text: "",
      options: ["", "", "", ""],
      correct_answer: 0,
      explanation: "",
      hint: "",
      hint_penalty_points: 0
    }
    setQuestions([...questions, newQuestion])
  }

  const removeQuestion = (key: string) => {
    setQuestions(questions.filter((q) => q.key !== key))
  }

  const updateQuestion = (key: string, updates: Partial<QuestionFormData>) => {
    setQuestions(
      questions.map((q) => (q.key === key ? { ...q, ...updates } : q))
    )
  }

  const moveQuestion = (key: string, direction: "up" | "down") => {
    setQuestions((prev) => {
      const index = prev.findIndex((question) => question.key === key)
      if (index < 0) return prev
      if (direction === "up" && index === 0) return prev
      if (direction === "down" && index === prev.length - 1) return prev

      const nextIndex = direction === "up" ? index - 1 : index + 1
      const cloned = [...prev]
      const [item] = cloned.splice(index, 1)
      cloned.splice(nextIndex, 0, item)
      return cloned
    })
  }

  const addMultipleChoiceOption = (key: string) => {
    setQuestions((prev) => prev.map((question) => {
      if (
        question.key !== key ||
        (question.question_type !== "multiple_choice" && question.question_type !== "multi_select")
      ) {
        return question
      }
      if (question.options.length >= MAX_MULTIPLE_CHOICE_OPTIONS) {
        return question
      }
      return {
        ...question,
        options: [...question.options, ""]
      }
    }))
  }

  const removeMultipleChoiceOption = (key: string, optionIndex: number) => {
    setQuestions((prev) => prev.map((question) => {
      if (
        question.key !== key ||
        (question.question_type !== "multiple_choice" && question.question_type !== "multi_select")
      ) {
        return question
      }
      if (question.options.length <= MIN_MULTIPLE_CHOICE_OPTIONS) {
        return question
      }

      const nextOptions = question.options.filter((_, idx) => idx !== optionIndex)
      if (question.question_type === "multi_select") {
        const selected = Array.isArray(question.correct_answer)
          ? question.correct_answer
            .map((entry) => Number(entry))
            .filter((entry) => Number.isFinite(entry))
          : []
        const nextSelected = Array.from(new Set(
          selected
            .filter((entry) => entry !== optionIndex)
            .map((entry) => (entry > optionIndex ? entry - 1 : entry))
            .filter((entry) => entry >= 0 && entry < nextOptions.length)
        )).sort((a, b) => a - b)

        return {
          ...question,
          options: nextOptions,
          correct_answer: nextSelected
        }
      }

      const currentAnswerIndex = Number(question.correct_answer)
      let nextAnswerIndex = Number.isNaN(currentAnswerIndex) ? 0 : currentAnswerIndex
      if (optionIndex < nextAnswerIndex) {
        nextAnswerIndex -= 1
      } else if (optionIndex === nextAnswerIndex) {
        nextAnswerIndex = Math.max(0, nextAnswerIndex - 1)
      }
      if (nextAnswerIndex >= nextOptions.length) {
        nextAnswerIndex = Math.max(0, nextOptions.length - 1)
      }
      return {
        ...question,
        options: nextOptions,
        correct_answer: nextAnswerIndex
      }
    }))
  }

  const addMatchingPair = (key: string) => {
    setQuestions((prev) => prev.map((question) => {
      if (question.key !== key || question.question_type !== "matching") {
        return question
      }
      if (question.options.length >= MAX_MATCHING_PAIRS) {
        return question
      }
      return {
        ...question,
        options: [...question.options, ""]
      }
    }))
  }

  const removeMatchingPair = (key: string, pairIndex: number) => {
    setQuestions((prev) => prev.map((question) => {
      if (question.key !== key || question.question_type !== "matching") {
        return question
      }
      if (question.options.length <= MIN_MATCHING_PAIRS) {
        return question
      }
      const removedLeft = question.options[pairIndex]?.trim()
      const nextOptions = question.options.filter((_, idx) => idx !== pairIndex)
      const nextMap = { ...normalizeMatchingAnswerMap(question.correct_answer) }
      if (removedLeft) {
        delete nextMap[removedLeft]
      }
      return {
        ...question,
        options: nextOptions,
        correct_answer: nextMap
      }
    }))
  }

  const handleSave = async () => {
    try {
      const values = await form.validateFields()

      if (questions.length === 0) {
        messageApi.warning(
          t("option:quiz.addQuestionsFirst", { defaultValue: "Please add at least one question" })
        )
        return
      }

      const normalizedQuestions: QuestionCreate[] = []
      for (let i = 0; i < questions.length; i++) {
        const q = questions[i]
        let questionOptions: string[] | undefined
        let normalizedCorrectAnswer: number | string | number[] | Record<string, string> = q.correct_answer

        if (q.question_type === "multiple_choice" || q.question_type === "multi_select") {
          const trimmedOptions = q.options.map((option) => option.trim())
          const normalizedOptions = trimmedOptions.filter(Boolean)
          if (normalizedOptions.length < MIN_MULTIPLE_CHOICE_OPTIONS) {
            messageApi.warning(
              t("option:quiz.optionsRequired", {
                defaultValue: "Please provide at least two options."
              })
            )
            return
          }
          questionOptions = normalizedOptions
          const indexMap = new Map<number, number>()
          let nextIndex = 0
          trimmedOptions.forEach((option, originalIndex) => {
            if (!option) return
            indexMap.set(originalIndex, nextIndex)
            nextIndex += 1
          })
          if (q.question_type === "multiple_choice") {
            const rawIndex = Number(q.correct_answer)
            const fallbackIndex = 0
            const mapped = indexMap.get(Number.isNaN(rawIndex) ? fallbackIndex : rawIndex)
            const safeIndex = mapped ?? fallbackIndex
            normalizedCorrectAnswer = Math.max(0, Math.min(safeIndex, normalizedOptions.length - 1))
          } else {
            const selectedIndicesRaw = Array.isArray(q.correct_answer)
              ? q.correct_answer
              : []
            const selectedIndices = selectedIndicesRaw
              .map((entry) => Number(entry))
              .filter((entry) => Number.isFinite(entry))
              .map((entry) => indexMap.get(entry))
              .filter((entry): entry is number => typeof entry === "number")
            const uniqueSelected = Array.from(new Set(selectedIndices)).sort((a, b) => a - b)
            if (uniqueSelected.length === 0) {
              messageApi.warning(
                t("option:quiz.multiSelectCorrectRequired", {
                  defaultValue: "Select at least one correct option."
                })
              )
              return
            }
            normalizedCorrectAnswer = uniqueSelected
          }
        } else if (q.question_type === "true_false") {
          normalizedCorrectAnswer =
            String(q.correct_answer || "true").toLowerCase() === "true" ? "true" : "false"
        } else if (q.question_type === "matching") {
          const rawLeftOptions = q.options.map((option) => option.trim())
          const leftOptions = rawLeftOptions.filter(Boolean)
          const answerMap = normalizeMatchingAnswerMap(q.correct_answer)
          const normalizedPairs = leftOptions
            .map((left) => {
              const right = String(answerMap[left] ?? "").trim()
              return { left, right }
            })
            .filter((pair) => pair.left.length > 0 && pair.right.length > 0)

          if (normalizedPairs.length < MIN_MATCHING_PAIRS) {
            messageApi.warning(
              t("option:quiz.matchingPairsRequired", {
                defaultValue: "Provide at least two complete matching pairs."
              })
            )
            return
          }

          questionOptions = normalizedPairs.map((pair) => pair.left)
          normalizedCorrectAnswer = Object.fromEntries(
            normalizedPairs.map((pair) => [pair.left, pair.right])
          )
        } else {
          normalizedCorrectAnswer = String(q.correct_answer || "").trim()
        }

        const hintText = (q.hint ?? "").trim()
        normalizedQuestions.push({
          question_type: q.question_type,
          question_text: q.question_text,
          options: questionOptions,
          correct_answer: normalizedCorrectAnswer,
          explanation: q.explanation || undefined,
          ...(hintText ? { hint: hintText } : {}),
          ...((q.hint_penalty_points ?? 0) > 0 ? { hint_penalty_points: q.hint_penalty_points } : {}),
          order_index: i
        })
      }

      setSaveProgress({
        phase: "creating_quiz",
        current: 0,
        total: normalizedQuestions.length
      })

      const quiz = await createQuizMutation.mutateAsync({
        name: values.name,
        description: values.description || undefined,
        time_limit_seconds: values.timeLimit ? values.timeLimit * 60 : undefined,
        passing_score: values.passingScore || undefined
      })

      for (let i = 0; i < normalizedQuestions.length; i++) {
        setSaveProgress({
          phase: "saving_questions",
          current: i + 1,
          total: normalizedQuestions.length
        })
        try {
          await createQuestionMutation.mutateAsync({
            quizId: quiz.id,
            question: normalizedQuestions[i]
          })
        } catch {
          throw new QuestionSaveError(i + 1, normalizedQuestions.length)
        }
      }

      messageApi.success(
        t("option:quiz.createSuccess", { defaultValue: "Quiz created successfully!" })
      )

      // Reset form
      form.resetFields()
      setQuestions([])
      clearCreateDraft()
      onDirtyStateChange?.(false)
      setPreviewOpen(false)
      onNavigateToTake({
        highlightQuizId: quiz.id,
        sourceTab: "create"
      })
    } catch (error) {
      if (isFormValidationError(error)) {
        return
      }
      if (error instanceof QuestionSaveError) {
        messageApi.error(
          t("option:quiz.createPartialError", {
            defaultValue:
              "Quiz was created, but saving failed at question {{question}} of {{total}}.",
            question: error.questionNumber,
            total: error.totalQuestions
          })
        )
        return
      }
      messageApi.error(
        t("option:quiz.createError", { defaultValue: "Failed to create quiz" })
      )
    } finally {
      setSaveProgress(null)
    }
  }

  const restoreDraft = () => {
    if (!pendingDraft) return
    form.setFieldsValue({
      name: pendingDraft.name,
      description: pendingDraft.description || undefined,
      timeLimit: pendingDraft.timeLimit ?? undefined,
      passingScore: pendingDraft.passingScore ?? undefined
    })
    setQuestions(pendingDraft.questions)
    setPendingDraft(null)
    messageApi.success(
      t("option:quiz.draftRestored", {
        defaultValue: "Recovered your previous draft."
      })
    )
  }

  const discardDraft = () => {
    clearCreateDraft()
    setPendingDraft(null)
  }

  const renderPreviewQuestion = (question: QuestionFormData, index: number) => {
    if (question.question_type === "multiple_choice") {
      return (
        <div className="space-y-1">
          {question.options.filter(Boolean).map((option, optionIndex) => (
            <div key={`${question.key}-preview-option-${optionIndex}`} className="text-sm text-text-muted">
              {String.fromCharCode(65 + optionIndex)}. {option}
            </div>
          ))}
        </div>
      )
    }
    if (question.question_type === "true_false") {
      return (
        <div className="text-sm text-text-muted">
          {t("option:quiz.trueFalse", { defaultValue: "True/False" })}
        </div>
      )
    }
    if (question.question_type === "multi_select") {
      return (
        <div className="text-sm text-text-muted italic">
          {t("option:quiz.multiSelect", { defaultValue: "Multi-Select" })}
        </div>
      )
    }
    if (question.question_type === "matching") {
      return (
        <div className="text-sm text-text-muted italic">
          {t("option:quiz.matching", { defaultValue: "Matching" })}
        </div>
      )
    }
    return (
      <div className="text-sm text-text-muted italic">
        {t("option:quiz.fillBlank", { defaultValue: "Fill in the Blank" })}
      </div>
    )
  }

  const renderQuestionEditor = (question: QuestionFormData, index: number) => {
    return (
      <Card
        key={question.key}
        size="small"
        className="mb-4"
        title={`${t("option:quiz.question", { defaultValue: "Question" })} ${index + 1}`}
        extra={
          <Space>
            <Button
              type="text"
              icon={<ArrowUpOutlined />}
              onClick={() => moveQuestion(question.key, "up")}
              disabled={isSaving || index === 0}
              className="min-h-11 min-w-11"
              aria-label={t("option:quiz.moveQuestionUp", {
                defaultValue: "Move question {{number}} up",
                number: index + 1
              })}
            />
            <Button
              type="text"
              icon={<ArrowDownOutlined />}
              onClick={() => moveQuestion(question.key, "down")}
              disabled={isSaving || index === questions.length - 1}
              className="min-h-11 min-w-11"
              aria-label={t("option:quiz.moveQuestionDown", {
                defaultValue: "Move question {{number}} down",
                number: index + 1
              })}
            />
            <Button
              type="text"
              danger
              icon={<DeleteOutlined />}
              onClick={() => removeQuestion(question.key)}
              disabled={isSaving}
              className="min-h-11 min-w-11"
              aria-label={t("option:quiz.removeQuestion", {
                defaultValue: "Remove question {{number}}",
                number: index + 1
              })}
            />
          </Space>
        }
      >
        <Space orientation="vertical" className="w-full">
          <Select
            value={question.question_type}
            disabled={isSaving}
            onChange={(value) => {
              const updates: Partial<QuestionFormData> = { question_type: value }
              if (value === "multiple_choice") {
                updates.correct_answer = 0
                if (question.options.length < MIN_MULTIPLE_CHOICE_OPTIONS) {
                  const paddedOptions = [...question.options]
                  while (paddedOptions.length < MIN_MULTIPLE_CHOICE_OPTIONS) {
                    paddedOptions.push("")
                  }
                  updates.options = paddedOptions
                }
              } else if (value === "multi_select") {
                updates.correct_answer = []
                if (question.options.length < MIN_MULTIPLE_CHOICE_OPTIONS) {
                  const paddedOptions = [...question.options]
                  while (paddedOptions.length < MIN_MULTIPLE_CHOICE_OPTIONS) {
                    paddedOptions.push("")
                  }
                  updates.options = paddedOptions
                }
              } else if (value === "matching") {
                updates.correct_answer = {}
                const paddedOptions = [...question.options]
                while (paddedOptions.length < MIN_MATCHING_PAIRS) {
                  paddedOptions.push("")
                }
                updates.options = paddedOptions
              } else if (value === "true_false") {
                updates.correct_answer = "true"
              } else {
                updates.correct_answer = ""
              }
              updateQuestion(question.key, updates)
            }}
            options={[
              { label: t("option:quiz.multipleChoice", { defaultValue: "Multiple Choice" }), value: "multiple_choice" },
              { label: t("option:quiz.multiSelect", { defaultValue: "Multi-Select" }), value: "multi_select" },
              { label: t("option:quiz.matching", { defaultValue: "Matching" }), value: "matching" },
              { label: t("option:quiz.trueFalse", { defaultValue: "True/False" }), value: "true_false" },
              { label: t("option:quiz.fillBlank", { defaultValue: "Fill in the Blank" }), value: "fill_blank" }
            ]}
            className="w-full sm:w-48"
          />

          <Input.TextArea
            placeholder={t("option:quiz.questionText", { defaultValue: "Enter your question..." })}
            value={question.question_text}
            onChange={(e) => updateQuestion(question.key, { question_text: e.target.value })}
            rows={2}
            disabled={isSaving}
          />

          {(question.question_type === "multiple_choice" || question.question_type === "multi_select") && (
            <div className="space-y-2">
              <div className="text-sm font-medium">
                {t("option:quiz.options", { defaultValue: "Options" })}
              </div>
              {question.options.map((option, optIndex) => (
                <div
                  key={optIndex}
                  data-testid={`create-option-row-${index}-${optIndex}`}
                  className="flex flex-col gap-2 sm:flex-row sm:items-center"
                >
                  <label className="inline-flex min-h-11 min-w-11 items-center justify-center rounded border border-border p-2">
                    {question.question_type === "multi_select" ? (
                      <Checkbox
                        checked={Array.isArray(question.correct_answer) && question.correct_answer.includes(optIndex)}
                        onChange={(event) => {
                          const existing = Array.isArray(question.correct_answer)
                            ? question.correct_answer
                              .map((entry) => Number(entry))
                              .filter((entry) => Number.isFinite(entry))
                            : []
                          const next = event.target.checked
                            ? Array.from(new Set([...existing, optIndex])).sort((a, b) => a - b)
                            : existing.filter((entry) => entry !== optIndex)
                          updateQuestion(question.key, { correct_answer: next })
                        }}
                        disabled={isSaving}
                        aria-label={t("option:quiz.markOptionCorrect", {
                          defaultValue: "Mark option {{number}} as correct",
                          number: optIndex + 1
                        })}
                      />
                    ) : (
                      <input
                        type="radio"
                        name={`correct-${question.key}`}
                        checked={Number(question.correct_answer) === optIndex}
                        onChange={() => updateQuestion(question.key, { correct_answer: optIndex })}
                        disabled={isSaving}
                        className="h-4 w-4"
                        aria-label={t("option:quiz.markOptionCorrect", {
                          defaultValue: "Mark option {{number}} as correct",
                          number: optIndex + 1
                        })}
                      />
                    )}
                  </label>
                  <Input
                    placeholder={`${t("option:quiz.option", { defaultValue: "Option" })} ${optIndex + 1}`}
                    value={option}
                    onChange={(e) => {
                      const newOptions = [...question.options]
                      newOptions[optIndex] = e.target.value
                      updateQuestion(question.key, { options: newOptions })
                    }}
                    disabled={isSaving}
                    className="w-full sm:flex-1"
                  />
                  <Button
                    type="text"
                    icon={<MinusCircleOutlined />}
                    className="min-h-11 min-w-11 self-start sm:self-auto"
                    aria-label={t("option:quiz.removeOption", {
                      defaultValue: "Remove option {{option}} for question {{question}}",
                      option: optIndex + 1,
                      question: index + 1
                    })}
                    onClick={() => removeMultipleChoiceOption(question.key, optIndex)}
                    disabled={isSaving || question.options.length <= MIN_MULTIPLE_CHOICE_OPTIONS}
                  />
                </div>
              ))}
              <Button
                type="dashed"
                icon={<PlusOutlined />}
                onClick={() => addMultipleChoiceOption(question.key)}
                disabled={isSaving || question.options.length >= MAX_MULTIPLE_CHOICE_OPTIONS}
                className={TOUCH_TARGET_CLASS}
              >
                {t("option:quiz.addOption", { defaultValue: "Add Option" })}
              </Button>
            </div>
          )}

          {question.question_type === "true_false" && (
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
              <span>{t("option:quiz.correctAnswer", { defaultValue: "Correct answer" })}:</span>
              <Switch
                checkedChildren={t("option:quiz.true", { defaultValue: "True" })}
                unCheckedChildren={t("option:quiz.false", { defaultValue: "False" })}
                checked={question.correct_answer === "true"}
                disabled={isSaving}
                className="min-h-11"
                onChange={(checked) =>
                  updateQuestion(question.key, { correct_answer: checked ? "true" : "false" })
                }
              />
            </div>
          )}

          {question.question_type === "fill_blank" && (
            <Space orientation="vertical" className="w-full" size={4}>
              <Input
                placeholder={t("option:quiz.correctAnswerPlaceholder", {
                  defaultValue: "Enter the correct answer..."
                })}
                value={typeof question.correct_answer === "string" ? question.correct_answer : ""}
                onChange={(e) => updateQuestion(question.key, { correct_answer: e.target.value })}
                disabled={isSaving}
              />
              <Typography.Text type="secondary" className="text-xs">
                {t("option:quiz.fillBlankAuthoringHelp", {
                  defaultValue:
                    "Use `answer1 || answer2` for alternates. Prefix with `~` (or `~0.85:`) to allow fuzzy matching."
                })}
              </Typography.Text>
            </Space>
          )}

          {question.question_type === "matching" && (
            <div className="space-y-2">
              <div className="text-sm font-medium">
                {t("option:quiz.matchingPairs", { defaultValue: "Matching Pairs" })}
              </div>
              {question.options.map((leftOption, pairIndex) => {
                const answerMap = normalizeMatchingAnswerMap(question.correct_answer)
                const leftTrimmed = leftOption.trim()
                const rightValue = leftTrimmed ? (answerMap[leftTrimmed] ?? "") : ""
                return (
                  <div
                    key={pairIndex}
                    className="grid grid-cols-1 gap-2 sm:grid-cols-[1fr_1fr_auto] sm:items-center"
                  >
                    <Input
                      placeholder={t("option:quiz.matchingLeftPlaceholder", {
                        defaultValue: "Left item"
                      })}
                      value={leftOption}
                      onChange={(e) => {
                        const nextLeftRaw = e.target.value
                        const previousLeft = question.options[pairIndex] ?? ""
                        const previousTrimmed = previousLeft.trim()
                        const nextTrimmed = nextLeftRaw.trim()
                        const nextOptions = [...question.options]
                        nextOptions[pairIndex] = nextLeftRaw
                        const nextMap = { ...normalizeMatchingAnswerMap(question.correct_answer) }
                        const carryValue = previousTrimmed ? nextMap[previousTrimmed] : undefined
                        if (previousTrimmed) {
                          delete nextMap[previousTrimmed]
                        }
                        if (nextTrimmed && carryValue) {
                          nextMap[nextTrimmed] = carryValue
                        }
                        updateQuestion(question.key, {
                          options: nextOptions,
                          correct_answer: nextMap
                        })
                      }}
                      disabled={isSaving}
                    />
                    <Input
                      placeholder={t("option:quiz.matchingRightPlaceholder", {
                        defaultValue: "Matching item"
                      })}
                      value={rightValue}
                      onChange={(e) => {
                        const nextMap = { ...normalizeMatchingAnswerMap(question.correct_answer) }
                        const mapKey = (question.options[pairIndex] ?? "").trim()
                        if (!mapKey) {
                          updateQuestion(question.key, { correct_answer: nextMap })
                          return
                        }
                        const nextRight = e.target.value.trim()
                        if (nextRight) {
                          nextMap[mapKey] = nextRight
                        } else {
                          delete nextMap[mapKey]
                        }
                        updateQuestion(question.key, { correct_answer: nextMap })
                      }}
                      disabled={isSaving}
                    />
                    <Button
                      type="text"
                      icon={<MinusCircleOutlined />}
                      className="min-h-11 min-w-11 self-start sm:self-auto"
                      aria-label={t("option:quiz.removeMatchingPair", {
                        defaultValue: "Remove matching pair {{pair}} for question {{question}}",
                        pair: pairIndex + 1,
                        question: index + 1
                      })}
                      onClick={() => removeMatchingPair(question.key, pairIndex)}
                      disabled={isSaving || question.options.length <= MIN_MATCHING_PAIRS}
                    />
                  </div>
                )
              })}
              <Button
                type="dashed"
                icon={<PlusOutlined />}
                onClick={() => addMatchingPair(question.key)}
                disabled={isSaving || question.options.length >= MAX_MATCHING_PAIRS}
                className={TOUCH_TARGET_CLASS}
              >
                {t("option:quiz.addMatchingPair", { defaultValue: "Add Pair" })}
              </Button>
            </div>
          )}

          <Input.TextArea
            placeholder={t("option:quiz.explanationPlaceholder", {
              defaultValue: "Explanation (shown after answering)..."
            })}
            value={question.explanation}
            onChange={(e) => updateQuestion(question.key, { explanation: e.target.value })}
            rows={2}
            disabled={isSaving}
          />
          <Typography.Text type="secondary" className="text-xs">
            {t("option:quiz.explanationVisibilityHelp", {
              defaultValue: "Shown to the learner after they submit the quiz."
            })}
          </Typography.Text>
          <Input.TextArea
            placeholder={t("option:quiz.hintPlaceholder", {
              defaultValue: "Optional hint (learner can reveal during quiz)..."
            })}
            value={question.hint ?? ""}
            onChange={(e) => updateQuestion(question.key, { hint: e.target.value })}
            rows={2}
            disabled={isSaving}
          />
          <InputNumber
            min={0}
            className="w-full sm:w-56"
            value={question.hint_penalty_points ?? 0}
            onChange={(value) => updateQuestion(question.key, { hint_penalty_points: Number(value) || 0 })}
            placeholder={t("option:quiz.hintPenaltyPoints", {
              defaultValue: "Hint penalty (points)"
            })}
            disabled={isSaving}
          />
          <Typography.Text type="secondary" className="text-xs">
            {t("option:quiz.hintPenaltyHelp", {
              defaultValue: "Applied only when the learner answers correctly after revealing the hint."
            })}
          </Typography.Text>
        </Space>
      </Card>
    )
  }

  return (
    <div className="max-w-3xl space-y-6">
      {contextHolder}

      {pendingDraft && (
        <Alert
          type="info"
          showIcon
          title={t("option:quiz.draftRecoveryTitle", {
            defaultValue: "Saved draft found"
          })}
          description={t("option:quiz.draftRecoveryDescription", {
            defaultValue: "Restore your previous Create Quiz draft?"
          })}
          action={(
            <Space>
              <Button size="small" onClick={discardDraft}>
                {t("common:discard", { defaultValue: "Discard" })}
              </Button>
              <Button type="primary" size="small" onClick={restoreDraft}>
                {t("common:restore", { defaultValue: "Restore" })}
              </Button>
            </Space>
          )}
        />
      )}

      {draftStorageUnavailable && !draftWarningDismissed && (
        <Alert
          type="warning"
          showIcon
          closable
          onClose={() => setDraftWarningDismissed(true)}
          title={t("option:quiz.draftStorageUnavailable", {
            defaultValue:
              "Draft autosave unavailable — your progress will not be preserved if you leave."
          })}
        />
      )}

      {!orientationDismissed && (
        <Alert
          type="info"
          showIcon
          closable
          data-testid="quiz-create-orientation"
          onClose={() => {
            setOrientationDismissed(true)
            try {
              window.localStorage.setItem(CREATE_ORIENTATION_DISMISSED_KEY, "1")
            } catch {
              // localStorage unavailable — dismiss for this session only
            }
          }}
          message={t("option:quiz.createOrientationTitle", {
            defaultValue: "Create a Quiz"
          })}
          description={t("option:quiz.createOrientationDescription", {
            defaultValue:
              "Name your quiz, add questions (multiple choice, true/false, fill-in-the-blank, matching, multi-select), set a passing score, and save. Tip: start with 5\u201310 questions."
          })}
        />
      )}

      <Card
        title={t("option:quiz.quizDetails", { defaultValue: "Quiz Details" })}
        size="small"
      >
        <Form form={form} layout="vertical">
          <Form.Item
            name="name"
            label={t("option:quiz.quizName", { defaultValue: "Quiz Name" })}
            rules={[{ required: true, message: t("option:quiz.nameRequired", { defaultValue: "Please enter a quiz name" }) }]}
          >
            <Input placeholder={t("option:quiz.namePlaceholder", { defaultValue: "e.g., Biology Chapter 5" })} />
          </Form.Item>

          <Form.Item
            name="description"
            label={t("option:quiz.description", { defaultValue: "Description" })}
          >
            <Input.TextArea
              placeholder={t("option:quiz.descriptionPlaceholder", { defaultValue: "Optional description..." })}
              rows={2}
            />
          </Form.Item>

          <div data-testid="create-details-grid" className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <Form.Item
              name="timeLimit"
              label={t("option:quiz.timeLimit", { defaultValue: "Time Limit (minutes)" })}
            >
              <InputNumber min={1} max={180} className="w-full" placeholder="Optional" />
            </Form.Item>

            <Form.Item
              name="passingScore"
              label={t("option:quiz.passingScore", { defaultValue: "Passing Score (%)" })}
            >
              <InputNumber min={1} max={100} className="w-full" placeholder="Optional" />
            </Form.Item>
          </div>
        </Form>
      </Card>

      <div>
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-medium">
            {t("option:quiz.questionsSection", { defaultValue: "Questions" })} ({questions.length})
          </h3>
          <Button
            type="dashed"
            icon={<PlusOutlined />}
            onClick={addQuestion}
            disabled={isSaving}
            className={TOUCH_TARGET_CLASS}
          >
            {t("option:quiz.addQuestion", { defaultValue: "Add Question" })}
          </Button>
        </div>

        {questions.length === 0 ? (
          <Card className="py-8 text-center">
            <p className="mb-4 text-text-subtle">
              {t("option:quiz.noQuestionsYet", { defaultValue: "No questions added yet" })}
            </p>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={addQuestion}
              disabled={isSaving}
              className={TOUCH_TARGET_CLASS}
            >
              {t("option:quiz.addFirstQuestion", { defaultValue: "Add Your First Question" })}
            </Button>
          </Card>
        ) : (
          questions.map((q, i) => renderQuestionEditor(q, i))
        )}
      </div>

      <Space className="w-full" orientation="vertical">
        {saveProgress && (
          <Card size="small">
            <Space orientation="vertical" className="w-full">
              <Typography.Text strong>
                {saveProgress.phase === "creating_quiz"
                  ? t("option:quiz.savingQuizProgress", {
                    defaultValue: "Creating quiz..."
                  })
                  : t("option:quiz.savingQuestionProgress", {
                    defaultValue: "Saving question {{current}} of {{total}}...",
                    current: saveProgress.current,
                    total: saveProgress.total
                  })}
              </Typography.Text>
              {saveProgress.phase === "saving_questions" && (
                <Progress
                  percent={Math.max(
                    0,
                    Math.min(100, Math.round((saveProgress.current / Math.max(saveProgress.total, 1)) * 100))
                  )}
                  status="active"
                  aria-label={t("option:quiz.savingProgressAria", {
                    defaultValue: "Quiz save progress"
                  })}
                />
              )}
            </Space>
          </Card>
        )}
        <Button
          onClick={() => setPreviewOpen(true)}
          disabled={questions.length === 0 || isSaving}
          className={TOUCH_TARGET_CLASS}
          block
        >
          {t("option:quiz.preview", { defaultValue: "Preview" })}
        </Button>
        <Button
          type="primary"
          icon={<SaveOutlined />}
          size="large"
          onClick={handleSave}
          loading={isSaving}
          disabled={questions.length === 0 || isSaving}
          className={TOUCH_TARGET_CLASS}
          block
        >
          {t("option:quiz.saveQuiz", { defaultValue: "Save Quiz" })}
        </Button>
      </Space>

      <Modal
        title={t("option:quiz.previewTitle", { defaultValue: "Quiz Preview" })}
        open={previewOpen}
        onCancel={() => setPreviewOpen(false)}
        footer={[
          <Button key="close" onClick={() => setPreviewOpen(false)}>
            {t("common:close", { defaultValue: "Close" })}
          </Button>
        ]}
        width={760}
      >
        <div className="space-y-4">
          <div className="space-y-1">
            <Typography.Title level={4} className="!mb-0">
              {(nameValue ?? "").trim() || t("option:quiz.untitledQuiz", { defaultValue: "Untitled Quiz" })}
            </Typography.Title>
            {(descriptionValue ?? "").trim() && (
              <Typography.Paragraph className="text-text-muted mb-0">
                {descriptionValue}
              </Typography.Paragraph>
            )}
          </div>

          {questions.map((question, index) => (
            <Card
              key={`preview-${question.key}`}
              size="small"
              title={`${t("option:quiz.question", { defaultValue: "Question" })} ${index + 1}`}
            >
              <div className="space-y-2">
                <Typography.Text strong>{question.question_text || "—"}</Typography.Text>
                {renderPreviewQuestion(question, index)}
                {(question.explanation ?? "").trim() && (
                  <Typography.Text type="secondary" className="block text-sm">
                    {t("option:quiz.explanation", { defaultValue: "Explanation" })}: {question.explanation}
                  </Typography.Text>
                )}
                {(question.hint ?? "").trim() && (
                  <Typography.Text type="secondary" className="block text-sm">
                    {t("option:quiz.hint", { defaultValue: "Hint" })}: {question.hint}
                  </Typography.Text>
                )}
                {(question.hint_penalty_points ?? 0) > 0 && (
                  <Typography.Text type="secondary" className="block text-xs">
                    {t("option:quiz.hintPenaltyPreview", {
                      defaultValue: "Hint penalty: -{{points}} point(s)",
                      points: question.hint_penalty_points
                    })}
                  </Typography.Text>
                )}
              </div>
            </Card>
          ))}
        </div>
      </Modal>
    </div>
  )
}

export default CreateTab
