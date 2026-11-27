import React from "react"
import {
  Button,
  Card,
  Empty,
  Input,
  InputNumber,
  Select,
  Space,
  Spin,
  Switch,
  Tabs,
  Tag,
  Typography
} from "antd"
import {
  ClipboardList,
  CopyPlus,
  MessageCircleMore,
  PencilLine,
  Play,
  Plus,
  Sparkles,
  XCircle
} from "lucide-react"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router-dom"

import { DismissibleBetaAlert } from "@/components/Common/DismissibleBetaAlert"
import { PageShell } from "@/components/Common/PageShell"
import WorkspaceConnectionGate from "@/components/Common/WorkspaceConnectionGate"
import { Alert } from "@/components/ui/primitives"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import {
  useCancelChatWorkflowRun,
  useChatWorkflowRun,
  useChatWorkflowTemplates,
  useChatWorkflowTranscript,
  useContinueChatWorkflowRun,
  useCreateChatWorkflowTemplate,
  useGenerateChatWorkflowDraft,
  useStartChatWorkflowRun,
  useSubmitChatWorkflowAnswer,
  useSubmitChatWorkflowRound,
  useUpdateChatWorkflowTemplate
} from "@/hooks/useChatWorkflows"
import { CHAT_PATH } from "@/routes/route-paths"
import type {
  ChatWorkflowDialogueConfig,
  ChatWorkflowRun,
  ChatWorkflowTemplate,
  ChatWorkflowTemplateDraft,
  ChatWorkflowTemplateStep,
  ChatWorkflowTranscriptMessage,
  ChatWorkflowTranscriptRole
} from "@/types/chat-workflows"
import { SETTINGS_SERVER_CHAT_ID_PARAM } from "@/utils/settings-return"

const { Text, Title } = Typography
const { TextArea } = Input

type ChatWorkflowTabKey = "library" | "builder" | "generate" | "run"

const createDefaultDialogueConfig = (): ChatWorkflowDialogueConfig => ({
  goal_prompt: "",
  opening_prompt_mode: "base_question",
  opening_prompt_text: "",
  user_role_label: "User",
  debate_llm_config: {
    provider: "openai",
    model: "gpt-4o-mini"
  },
  moderator_llm_config: {
    provider: "openai",
    model: "gpt-4o-mini"
  },
  max_rounds: 4,
  finish_conditions: [],
  context_refs: [],
  debate_instruction_prompt: "",
  moderator_instruction_prompt: ""
})

const createStepId = (): string => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID()
  }
  return `step-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`
}

const createEmptyStep = (index: number): ChatWorkflowTemplateStep => ({
  id: createStepId(),
  step_index: index,
  step_type: "question_step",
  label: `Step ${index + 1}`,
  base_question: "",
  question_mode: "stock",
  phrasing_instructions: "",
  context_refs: []
})

const createEmptyDraft = (): ChatWorkflowTemplateDraft => ({
  title: "",
  description: "",
  version: 1,
  steps: [createEmptyStep(0)]
})

const createSocraticDialogueDraft = (): ChatWorkflowTemplateDraft => ({
  title: "Socratic Dialogue",
  description: "Pressure-test a thesis with a debate LLM and an LLM moderator.",
  version: 1,
  steps: [
    {
      id: createStepId(),
      step_index: 0,
      step_type: "dialogue_round_step",
      label: "Socratic dialogue",
      base_question: "State your current thesis or position.",
      question_mode: "stock",
      context_refs: [],
      dialogue_config: {
        goal_prompt: "Stress-test the user's thesis until the reasoning is clarified.",
        opening_prompt_mode: "base_question",
        opening_prompt_text: "",
        user_role_label: "User",
        debate_llm_config: {
          provider: "openai",
          model: "gpt-4o-mini"
        },
        moderator_llm_config: {
          provider: "openai",
          model: "gpt-4o-mini"
        },
        max_rounds: 4,
        finish_conditions: [
          "The thesis has been adequately challenged or refined."
        ],
        context_refs: [],
        debate_instruction_prompt:
          "Challenge weak assumptions, vague evidence, and unsupported causal claims.",
        moderator_instruction_prompt:
          "Return structured control output only. Decide whether the dialogue should continue or finish."
      }
    }
  ]
})

const cloneTemplateToDraft = (
  template: Pick<ChatWorkflowTemplateDraft, "title" | "description" | "version" | "steps">
): ChatWorkflowTemplateDraft => ({
  title: template.title,
  description: template.description || "",
  version: template.version,
  steps: template.steps.map((step, index) => ({
    ...step,
    id: step.id || createStepId(),
    step_index: index,
    step_type: step.step_type || "question_step",
    label: step.label || `Step ${index + 1}`,
    phrasing_instructions: step.phrasing_instructions || "",
    context_refs: Array.isArray(step.context_refs) ? [...step.context_refs] : [],
    dialogue_config:
      step.step_type === "dialogue_round_step"
        ? {
            ...createDefaultDialogueConfig(),
            ...step.dialogue_config,
            debate_llm_config: {
              ...createDefaultDialogueConfig().debate_llm_config,
              ...step.dialogue_config?.debate_llm_config
            },
            moderator_llm_config: {
              ...createDefaultDialogueConfig().moderator_llm_config,
              ...step.dialogue_config?.moderator_llm_config
            },
            finish_conditions: Array.isArray(step.dialogue_config?.finish_conditions)
              ? [...step.dialogue_config.finish_conditions]
              : [],
            context_refs: Array.isArray(step.dialogue_config?.context_refs)
              ? [...step.dialogue_config.context_refs]
              : []
          }
        : undefined
  }))
})

const normalizeDraftForSubmit = (
  draft: ChatWorkflowTemplateDraft
): ChatWorkflowTemplateDraft => ({
  title: draft.title.trim(),
  description: draft.description?.trim() || undefined,
  version: draft.version || 1,
  steps: draft.steps.map((step, index) => ({
    id: step.id.trim() || `step-${index + 1}`,
    step_index: index,
    step_type: step.step_type || "question_step",
    label: step.label?.trim() || `Step ${index + 1}`,
    base_question: step.base_question.trim(),
    question_mode: step.question_mode,
    phrasing_instructions: step.phrasing_instructions?.trim() || undefined,
    context_refs: Array.isArray(step.context_refs) ? step.context_refs : [],
    dialogue_config:
      step.step_type === "dialogue_round_step" && step.dialogue_config
        ? {
            ...step.dialogue_config,
            goal_prompt: step.dialogue_config.goal_prompt.trim(),
            opening_prompt_text:
              step.dialogue_config.opening_prompt_text?.trim() || undefined,
            user_role_label: step.dialogue_config.user_role_label.trim(),
            finish_conditions: step.dialogue_config.finish_conditions
              .flatMap((entry) => entry.split("\n"))
              .map((entry) => entry.trim())
              .filter(Boolean),
            debate_instruction_prompt:
              step.dialogue_config.debate_instruction_prompt.trim(),
            moderator_instruction_prompt:
              step.dialogue_config.moderator_instruction_prompt.trim(),
            debate_llm_config: {
              ...step.dialogue_config.debate_llm_config,
              provider:
                step.dialogue_config.debate_llm_config.provider?.trim() ||
                undefined,
              model: step.dialogue_config.debate_llm_config.model.trim()
            },
            moderator_llm_config: {
              ...step.dialogue_config.moderator_llm_config,
              provider:
                step.dialogue_config.moderator_llm_config.provider?.trim() ||
                undefined,
              model: step.dialogue_config.moderator_llm_config.model.trim()
            }
          }
        : undefined
  }))
})

const duplicateDraft = (template: ChatWorkflowTemplate): ChatWorkflowTemplateDraft => {
  const cloned = cloneTemplateToDraft(template)
  return {
    ...cloned,
    title: `${template.title} copy`
  }
}

const buildRunHistoryMessages = (
  run: ChatWorkflowRun | undefined,
  transcriptMessages: ChatWorkflowTranscriptMessage[] | undefined
): ChatWorkflowTranscriptMessage[] => {
  if (Array.isArray(transcriptMessages) && transcriptMessages.length > 0) {
    return transcriptMessages
  }
  if (!run) {
    return []
  }

  if (Array.isArray(run.rounds) && run.rounds.length > 0) {
    return run.rounds.flatMap((round) => {
      const messages: ChatWorkflowTranscriptMessage[] = [
        {
          role: "user",
          content: round.user_message,
          step_index: run.current_step_index
        }
      ]
      if (round.debate_llm_message) {
        messages.push({
          role: "debate_llm",
          content: round.debate_llm_message,
          step_index: run.current_step_index
        })
      }
      const moderatorParts = [round.moderator_summary, round.next_user_prompt].filter(
        Boolean
      )
      if (moderatorParts.length > 0) {
        messages.push({
          role: "moderator",
          content: moderatorParts.join("\n\n"),
          step_index: run.current_step_index
        })
      }
      return messages
    })
  }

  return run.answers.flatMap((answer) => [
    {
      role: "assistant",
      content: answer.displayed_question,
      step_index: answer.step_index
    },
    {
      role: "user",
      content: answer.answer_text,
      step_index: answer.step_index
    }
  ])
}

const getHistoryLabel = (message: ChatWorkflowTranscriptMessage): string => {
  const stepSuffix =
    typeof message.step_index === "number" ? ` ${message.step_index + 1}` : ""
  const labels: Record<ChatWorkflowTranscriptRole, string> = {
    assistant: `Question${stepSuffix}`,
    user: `Response${stepSuffix}`,
    debate_llm: `Debate${stepSuffix}`,
    moderator: `Moderator${stepSuffix}`
  }
  return labels[message.role] || `Message${stepSuffix}`
}

export const ChatWorkflowsPage: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  const navigate = useNavigate()
  const notification = useAntdNotification()
  const [activeTab, setActiveTab] = React.useState<ChatWorkflowTabKey>("library")
  const [editingTemplateId, setEditingTemplateId] = React.useState<number | null>(
    null
  )
  const [draft, setDraft] = React.useState<ChatWorkflowTemplateDraft>(() =>
    createEmptyDraft()
  )
  const [generateGoal, setGenerateGoal] = React.useState("")
  const [generateBaseQuestion, setGenerateBaseQuestion] = React.useState("")
  const [desiredStepCount, setDesiredStepCount] = React.useState(4)
  const [activeRunId, setActiveRunId] = React.useState<string | null>(null)
  const [activeRunTemplate, setActiveRunTemplate] =
    React.useState<ChatWorkflowTemplateDraft | null>(null)
  const [runAnswerText, setRunAnswerText] = React.useState("")

  const templatesQuery = useChatWorkflowTemplates()
  const createTemplateMutation = useCreateChatWorkflowTemplate()
  const updateTemplateMutation = useUpdateChatWorkflowTemplate(
    editingTemplateId || 0
  )
  const generateDraftMutation = useGenerateChatWorkflowDraft()
  const startRunMutation = useStartChatWorkflowRun()
  const activeRunQuery = useChatWorkflowRun(activeRunId, {
    enabled: Boolean(activeRunId)
  })
  const activeTranscriptQuery = useChatWorkflowTranscript(activeRunId, {
    enabled: Boolean(activeRunId)
  })
  const submitAnswerMutation = useSubmitChatWorkflowAnswer(activeRunId || "")
  const submitRoundMutation = useSubmitChatWorkflowRound(
    activeRunId || "",
    activeRunQuery.data?.current_round_index || 0
  )
  const cancelRunMutation = useCancelChatWorkflowRun(activeRunId || "")
  const continueRunMutation = useContinueChatWorkflowRun(activeRunId || "")

  const openNewTemplate = React.useCallback(() => {
    setEditingTemplateId(null)
    setDraft(createEmptyDraft())
    setActiveTab("builder")
  }, [])

  const openTemplateForEditing = React.useCallback((template: ChatWorkflowTemplate) => {
    setEditingTemplateId(template.id)
    setDraft(cloneTemplateToDraft(template))
    setActiveTab("builder")
  }, [])

  const openTemplateAsCopy = React.useCallback((template: ChatWorkflowTemplate) => {
    setEditingTemplateId(null)
    setDraft(duplicateDraft(template))
    setActiveTab("builder")
  }, [])

  const openSocraticTemplate = React.useCallback(() => {
    setEditingTemplateId(null)
    setDraft(createSocraticDialogueDraft())
    setActiveTab("builder")
  }, [])

  const updateDraftField = React.useCallback(
    (field: "title" | "description", value: string) => {
      setDraft((current) => ({
        ...current,
        [field]: value
      }))
    },
    []
  )

  const updateStep = React.useCallback(
    (
      stepIndex: number,
      field: keyof ChatWorkflowTemplateStep,
      value: ChatWorkflowTemplateStep[keyof ChatWorkflowTemplateStep]
    ) => {
      setDraft((current) => ({
        ...current,
        steps: current.steps.map((step, index) =>
          index === stepIndex
            ? ({
                ...step,
                [field]: value
              } as ChatWorkflowTemplateStep)
            : step
        )
      }))
    },
    []
  )

  const updateDialogueConfig = React.useCallback(
    (
      stepIndex: number,
      field: keyof ChatWorkflowDialogueConfig,
      value: ChatWorkflowDialogueConfig[keyof ChatWorkflowDialogueConfig]
    ) => {
      setDraft((current) => ({
        ...current,
        steps: current.steps.map((step, index) => {
          if (index !== stepIndex) {
            return step
          }

          const dialogueConfig = {
            ...createDefaultDialogueConfig(),
            ...step.dialogue_config
          }

          return {
            ...step,
            dialogue_config: {
              ...dialogueConfig,
              [field]: value
            }
          }
        })
      }))
    },
    []
  )

  const updateDialogueSelection = React.useCallback(
    (
      stepIndex: number,
      selectionKey: "debate_llm_config" | "moderator_llm_config",
      field: "provider" | "model" | "temperature" | "max_tokens" | "top_p",
      value: string | number | null
    ) => {
      setDraft((current) => ({
        ...current,
        steps: current.steps.map((step, index) => {
          if (index !== stepIndex) {
            return step
          }

          const dialogueConfig = {
            ...createDefaultDialogueConfig(),
            ...step.dialogue_config
          }
          const selection = {
            ...dialogueConfig[selectionKey],
            [field]: value
          }

          return {
            ...step,
            dialogue_config: {
              ...dialogueConfig,
              [selectionKey]: selection
            }
          }
        })
      }))
    },
    []
  )

  const updateStepType = React.useCallback(
    (stepIndex: number, stepType: ChatWorkflowTemplateStep["step_type"]) => {
      setDraft((current) => ({
        ...current,
        steps: current.steps.map((step, index) => {
          if (index !== stepIndex) {
            return step
          }

          if (stepType === "dialogue_round_step") {
            return {
              ...step,
              step_type: "dialogue_round_step",
              question_mode: "stock",
              phrasing_instructions: "",
              dialogue_config: step.dialogue_config || createDefaultDialogueConfig()
            }
          }

          return {
            ...step,
            step_type: "question_step",
            dialogue_config: undefined
          }
        })
      }))
    },
    []
  )

  const addStep = React.useCallback(() => {
    setDraft((current) => ({
      ...current,
      steps: [...current.steps, createEmptyStep(current.steps.length)]
    }))
  }, [])

  const removeStep = React.useCallback((stepIndex: number) => {
    setDraft((current) => {
      const nextSteps = current.steps.filter((_, index) => index !== stepIndex)
      const normalizedSteps =
        nextSteps.length > 0 ? nextSteps : [createEmptyStep(0)]
      return {
        ...current,
        steps: normalizedSteps.map((step, index) => ({
          ...step,
          step_index: index,
          label: step.label || `Step ${index + 1}`
        }))
      }
    })
  }, [])

  const validateDraft = React.useCallback(
    (candidate: ChatWorkflowTemplateDraft): ChatWorkflowTemplateDraft | null => {
      const payload = normalizeDraftForSubmit(candidate)
      if (!payload.title) {
        notification.error({
          message: t("common:error", "Error"),
          description: "Template title is required."
        })
        return null
      }
      if (payload.steps.some((step) => !step.base_question)) {
        notification.error({
          message: t("common:error", "Error"),
          description:
            "Each step needs a question before you can start or save the workflow."
        })
        return null
      }
      const invalidDialogueStep = payload.steps.find((step) => {
        if (step.step_type !== "dialogue_round_step" || !step.dialogue_config) {
          return false
        }
        const debateModel = step.dialogue_config.debate_llm_config.model?.trim()
        const moderatorModel = step.dialogue_config.moderator_llm_config.model?.trim()
        const needsCustomPrompt =
          step.dialogue_config.opening_prompt_mode === "custom_prompt" &&
          !step.dialogue_config.opening_prompt_text
        return (
          !step.dialogue_config.goal_prompt ||
          !step.dialogue_config.user_role_label ||
          !step.dialogue_config.debate_instruction_prompt ||
          !step.dialogue_config.moderator_instruction_prompt ||
          !debateModel ||
          !moderatorModel ||
          needsCustomPrompt
        )
      })
      if (invalidDialogueStep) {
        notification.error({
          message: t("common:error", "Error"),
          description:
            "Dialogue steps need a goal, user role, debate/moderator instructions, and both model selections."
        })
        return null
      }
      return payload
    },
    [notification, t]
  )

  const saveTemplate = React.useCallback(async () => {
    const payload = validateDraft(draft)
    if (!payload) {
      return
    }

    try {
      const saved =
        editingTemplateId != null
          ? await updateTemplateMutation.mutateAsync(payload)
          : await createTemplateMutation.mutateAsync(payload)
      setEditingTemplateId(saved.id)
      setDraft(cloneTemplateToDraft(saved))
      notification.success({
        message:
          editingTemplateId != null
            ? "Template updated"
            : "Template created",
        description:
          editingTemplateId != null
            ? "Your workflow changes are saved."
            : "Your workflow is ready for structured Q&A."
      })
      setActiveTab("library")
    } catch (error) {
      notification.error({
        message: t("common:error", "Error"),
        description:
          error instanceof Error
            ? error.message
            : "Unable to save the workflow template."
      })
    }
  }, [
    createTemplateMutation,
    draft,
    editingTemplateId,
    notification,
    t,
    updateTemplateMutation,
    validateDraft
  ])

  const generateDraft = React.useCallback(async () => {
    try {
      const generated = await generateDraftMutation.mutateAsync({
        goal: generateGoal.trim(),
        base_question: generateBaseQuestion.trim(),
        desired_step_count: desiredStepCount,
        context_refs: []
      })
      setEditingTemplateId(null)
      setDraft(cloneTemplateToDraft(generated.template_draft))
      setActiveTab("builder")
      notification.success({
        message: "Draft generated",
        description: "Review the suggested steps, then save the template."
      })
    } catch (error) {
      notification.error({
        message: t("common:error", "Error"),
        description:
          error instanceof Error
            ? error.message
            : "Unable to generate a workflow draft."
      })
    }
  }, [
    desiredStepCount,
    generateBaseQuestion,
    generateDraftMutation,
    generateGoal,
    notification,
    t
  ])

  const startRunFromTemplate = React.useCallback(
    async (template: ChatWorkflowTemplate) => {
      try {
        const run = await startRunMutation.mutateAsync({
          template_id: template.id,
          selected_context_refs: []
        })
        setActiveRunId(run.run_id)
        setActiveRunTemplate(cloneTemplateToDraft(template))
        setRunAnswerText("")
        setActiveTab("run")
        notification.success({
          message: "Workflow started",
          description: "Answer each prompt in sequence to complete the run."
        })
      } catch (error) {
        notification.error({
          message: t("common:error", "Error"),
          description:
            error instanceof Error
              ? error.message
              : "Unable to start the workflow run."
        })
      }
    },
    [notification, startRunMutation, t]
  )

  const startRunFromDraft = React.useCallback(async () => {
    const payload = validateDraft(draft)
    if (!payload) {
      return
    }

    try {
      const run = await startRunMutation.mutateAsync({
        template_draft: payload,
        selected_context_refs: []
      })
      setActiveRunId(run.run_id)
      setActiveRunTemplate(payload)
      setRunAnswerText("")
      setActiveTab("run")
      notification.success({
        message: "Draft started as a run",
        description: "This execution uses the current draft snapshot."
      })
    } catch (error) {
      notification.error({
        message: t("common:error", "Error"),
        description:
          error instanceof Error
            ? error.message
            : "Unable to start the workflow run."
      })
    }
  }, [draft, notification, startRunMutation, t, validateDraft])

  const activeRun = activeRunQuery.data
  const terminalRunNotice = activeRun?.status === "completed"
    ? {
        variant: "success" as const,
        title: "Workflow complete",
        description: "The guided run is finished. Continue into chat only if you want to leave the structured flow."
      }
    : activeRun?.status === "canceled"
      ? {
          variant: "warning" as const,
          title: "Workflow canceled",
          description: "This run has been stopped. You can return to the library or launch a fresh run."
        }
      : null
  const activeRunPrompt = activeRun?.current_prompt || activeRun?.current_question
  const isDialogueStep = activeRun?.current_step_kind === "dialogue_round_step"
  const runHistoryMessages = buildRunHistoryMessages(
    activeRun,
    activeTranscriptQuery.data?.messages
  )
  const totalRunSteps =
    activeRunTemplate?.steps.length ??
    Math.max(activeRun?.answers.length || 0, activeRun?.current_step_index || 0, 1)
  const completedRunSteps =
    activeRun?.status === "completed"
      ? totalRunSteps
      : activeRun?.answers.length || 0

  const submitCurrentAnswer = React.useCallback(async () => {
    if (!activeRunId || !activeRun) {
      return
    }

    const answer = runAnswerText.trim()
    if (!answer) {
      notification.error({
        message: t("common:error", "Error"),
        description: isDialogueStep
          ? "Write a response before continuing the dialogue."
          : "Write an answer before moving to the next step."
      })
      return
    }

    try {
      const nextRun = isDialogueStep
        ? await submitRoundMutation.mutateAsync({
            user_message: answer
          })
        : await submitAnswerMutation.mutateAsync({
            step_index: activeRun.current_step_index,
            answer_text: answer
          })
      setRunAnswerText("")
      if (nextRun.status === "completed") {
        notification.success({
          message: "Workflow complete",
          description: "You can stop here or continue into normal chat."
        })
      }
    } catch (error) {
      notification.error({
        message: t("common:error", "Error"),
        description:
          error instanceof Error
            ? error.message
            : "Unable to submit the workflow answer."
      })
    }
  }, [
    activeRun,
    activeRunId,
    isDialogueStep,
    notification,
    runAnswerText,
    submitRoundMutation,
    submitAnswerMutation,
    t
  ])

  const cancelActiveRun = React.useCallback(async () => {
    if (!activeRunId) {
      return
    }

    try {
      await cancelRunMutation.mutateAsync()
      notification.success({
        message: "Workflow canceled",
        description: "The run has been stopped and will not ask more questions."
      })
    } catch (error) {
      notification.error({
        message: t("common:error", "Error"),
        description:
          error instanceof Error ? error.message : "Unable to cancel the workflow."
      })
    }
  }, [activeRunId, cancelRunMutation, notification, t])

  const continueToChat = React.useCallback(async () => {
    if (!activeRunId) {
      return
    }

    try {
      const response = await continueRunMutation.mutateAsync()
      const params = new URLSearchParams({
        [SETTINGS_SERVER_CHAT_ID_PARAM]: response.conversation_id
      })
      navigate(`${CHAT_PATH}?${params.toString()}`)
    } catch (error) {
      notification.error({
        message: t("common:error", "Error"),
        description:
          error instanceof Error
            ? error.message
            : "Unable to continue the workflow in normal chat."
      })
    }
  }, [activeRunId, continueRunMutation, navigate, notification, t])

  const templates = templatesQuery.data || []

  return (
    <WorkspaceConnectionGate
      featureName="Chat Workflows"
      setupDescription="Chat Workflows depends on your connected tldw server to create templates, run structured sessions, and hand off completed conversations."
      maxWidthClassName="max-w-6xl"
    >
      <PageShell className="py-6" maxWidthClassName="max-w-6xl">
      <div className="mb-6 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-border bg-surface px-3 py-1 text-xs font-medium uppercase tracking-[0.16em] text-text-muted">
            <ClipboardList className="h-3.5 w-3.5" />
            Structured QA
          </div>
          <Title level={2} className="!mb-1">
            Chat Workflows
          </Title>
          <Text className="text-sm text-text-muted">
            Build guided Q&amp;A flows that ask one focused question at a time, preserve the run state,
            and hand off to free chat only when the structure is complete.
          </Text>
        </div>
        <Space wrap>
          <Button icon={<Plus className="h-4 w-4" />} onClick={openNewTemplate}>
            New Template
          </Button>
          <Button icon={<MessageCircleMore className="h-4 w-4" />} onClick={openSocraticTemplate}>
            Use Socratic Dialogue
          </Button>
          <Button
            type="primary"
            icon={<Sparkles className="h-4 w-4" />}
            onClick={() => setActiveTab("generate")}>
            Open Generator
          </Button>
        </Space>
      </div>

      <DismissibleBetaAlert
        storageKey="beta-dismissed:chat-workflows"
        message="Beta feature"
        description="Template authoring, guided run playback, and free-chat handoff are ready. Context pickers and resumable runs land next."
        className="mb-6"
      />

      <p role="status" aria-label="Workflow run status" aria-atomic="true" className="sr-only">
        {activeTab === "run" && activeRunId && !activeRunQuery.isError && terminalRunNotice
          ? `${terminalRunNotice.title}. ${terminalRunNotice.description}`
          : null}
      </p>

      <Tabs
        activeKey={activeTab}
        destroyOnHidden
        onChange={(key) => setActiveTab(key as ChatWorkflowTabKey)}
        items={[
          {
            key: "library",
            label: "Library",
            children: (
              <div className="space-y-4">
                <Alert
                  variant="info"
                  aria-live="off"
                  title={<span className="text-text">Reusable templates</span>}
                >
                  <span className="text-text">
                    Keep authored workflows here, then launch a guided run whenever you want the assistant to ask one question at a time.
                  </span>
                </Alert>
                {templatesQuery.isLoading ? (
                  <div className="flex min-h-[240px] items-center justify-center">
                    <Spin />
                  </div>
                ) : templates.length === 0 ? (
                  <Empty
                    description="No chat workflow templates yet. Start with a blank builder or generate a draft."
                  />
                ) : (
                  <div className="grid gap-4 lg:grid-cols-2">
                    {templates.map((template) => (
                      <Card
                        key={template.id}
                        className="border border-border shadow-sm"
                        title={
                          <div className="flex items-center justify-between gap-3">
                            <span>{template.title}</span>
                            <Tag color={template.status === "active" ? "green" : "default"}>
                              {template.steps.length} steps
                            </Tag>
                          </div>
                        }>
                        <div className="space-y-4">
                          <Text className="block text-sm text-text-muted">
                            {template.description || "No description yet."}
                          </Text>
                          <div className="space-y-2">
                            {template.steps.slice(0, 3).map((step) => (
                              <div
                                key={step.id}
                                className="rounded-lg border border-dashed border-border bg-surface p-3">
                                <Text strong className="block text-xs uppercase text-text-muted">
                                  {step.label || `Step ${step.step_index + 1}`}
                                </Text>
                                <Text className="text-sm">{step.base_question}</Text>
                              </div>
                            ))}
                          </div>
                          <Space wrap>
                            <Button
                              icon={<PencilLine className="h-4 w-4" />}
                              onClick={() => openTemplateForEditing(template)}>
                              Edit in Builder
                            </Button>
                            <Button
                              icon={<CopyPlus className="h-4 w-4" />}
                              onClick={() => openTemplateAsCopy(template)}>
                              Load as Copy
                            </Button>
                            <Button
                              type="primary"
                              icon={<Play className="h-4 w-4" />}
                              onClick={() => void startRunFromTemplate(template)}
                              loading={startRunMutation.isPending}>
                              Start Run
                            </Button>
                          </Space>
                        </div>
                      </Card>
                    ))}
                  </div>
                )}
              </div>
            )
          },
          {
            key: "builder",
            label: "Builder",
            children: (
              <div className="grid gap-6 xl:grid-cols-[minmax(0,1.6fr)_minmax(300px,0.9fr)]">
                <Card
                  title={
                    editingTemplateId != null
                      ? `Editing template #${editingTemplateId}`
                      : "New template"
                  }
                  extra={
                    <Space>
                      <Button onClick={openNewTemplate}>Reset Draft</Button>
                      <Button
                        icon={<Play className="h-4 w-4" />}
                        onClick={() => void startRunFromDraft()}
                        loading={startRunMutation.isPending}>
                        Start Run
                      </Button>
                      <Button
                        type="primary"
                        onClick={saveTemplate}
                        loading={
                          createTemplateMutation.isPending ||
                          updateTemplateMutation.isPending
                        }>
                        Save Template
                      </Button>
                    </Space>
                  }>
                  <div className="space-y-5">
                    <div className="grid gap-4 md:grid-cols-2">
                      <label className="block">
                        <span className="mb-1 block text-sm font-medium">
                          Template title
                        </span>
                        <Input
                          aria-label="Template title"
                          placeholder="Discovery workflow"
                          value={draft.title}
                          onChange={(event) =>
                            updateDraftField("title", event.target.value)
                          }
                        />
                      </label>
                      <label className="block">
                        <span className="mb-1 block text-sm font-medium">
                          Template description
                        </span>
                        <Input
                          aria-label="Template description"
                          placeholder="Capture goal, constraints, and success criteria."
                          value={draft.description || ""}
                          onChange={(event) =>
                            updateDraftField("description", event.target.value)
                          }
                        />
                      </label>
                    </div>

                    <div className="space-y-4">
                      {draft.steps.map((step, index) => (
                        <Card
                          key={step.id}
                          size="small"
                          className="border border-border bg-surface"
                          title={
                            <div className="flex items-center gap-2">
                        <span>{step.label || `Step ${index + 1}`}</span>
                              <Tag>
                                {step.step_type === "dialogue_round_step"
                                  ? "Dialogue"
                                  : step.question_mode === "stock"
                                    ? "Stock"
                                    : "LLM phrased"}
                              </Tag>
                            </div>
                          }
                          extra={
                            <Button
                              danger
                              size="small"
                              onClick={() => removeStep(index)}>
                              Remove
                            </Button>
                          }>
                          <div className="space-y-4">
                            <label className="block">
                              <span className="mb-1 block text-sm font-medium">
                                Step type
                              </span>
                              <Select
                                aria-label={`Type for step ${index + 1}`}
                                value={step.step_type || "question_step"}
                                options={[
                                  { label: "Question step", value: "question_step" },
                                  {
                                    label: "Dialogue round step",
                                    value: "dialogue_round_step"
                                  }
                                ]}
                                onChange={(value) =>
                                  updateStepType(
                                    index,
                                    value as ChatWorkflowTemplateStep["step_type"]
                                  )
                                }
                              />
                            </label>
                            <label className="block">
                              <span className="mb-1 block text-sm font-medium">
                                Step label
                              </span>
                              <Input
                                aria-label={`Label for step ${index + 1}`}
                                placeholder={`Step ${index + 1}`}
                                value={step.label || ""}
                                onChange={(event) =>
                                  updateStep(index, "label", event.target.value)
                                }
                              />
                            </label>
                            <label className="block">
                              <span className="mb-1 block text-sm font-medium">
                                {step.step_type === "dialogue_round_step"
                                  ? `Opening prompt for step ${index + 1}`
                                  : `Question for step ${index + 1}`}
                              </span>
                              <TextArea
                                aria-label={`Question for step ${index + 1}`}
                                autoSize={{ minRows: 2, maxRows: 5 }}
                                placeholder={
                                  step.step_type === "dialogue_round_step"
                                    ? "State the position the user should defend first."
                                    : "Ask one concrete question at this step."
                                }
                                value={step.base_question}
                                onChange={(event) =>
                                  updateStep(index, "base_question", event.target.value)
                                }
                              />
                            </label>
                            {step.step_type === "dialogue_round_step" &&
                            step.dialogue_config ? (
                              <div className="space-y-4 rounded-2xl border border-dashed border-border px-4 py-4">
                                <div className="grid gap-4 md:grid-cols-2">
                                  <label className="block">
                                    <span className="mb-1 block text-sm font-medium">
                                      Dialogue goal
                                    </span>
                                    <TextArea
                                      aria-label={`Dialogue goal for step ${index + 1}`}
                                      autoSize={{ minRows: 2, maxRows: 4 }}
                                      placeholder="What should this dialogue pressure-test?"
                                      value={step.dialogue_config.goal_prompt}
                                      onChange={(event) =>
                                        updateDialogueConfig(
                                          index,
                                          "goal_prompt",
                                          event.target.value
                                        )
                                      }
                                    />
                                  </label>
                                  <label className="block">
                                    <span className="mb-1 block text-sm font-medium">
                                      User role label
                                    </span>
                                    <Input
                                      aria-label={`User role label for step ${index + 1}`}
                                      placeholder="User"
                                      value={step.dialogue_config.user_role_label}
                                      onChange={(event) =>
                                        updateDialogueConfig(
                                          index,
                                          "user_role_label",
                                          event.target.value
                                        )
                                      }
                                    />
                                  </label>
                                  <label className="block">
                                    <span className="mb-1 block text-sm font-medium">
                                      Opening prompt mode
                                    </span>
                                    <Select
                                      aria-label={`Opening prompt mode for step ${index + 1}`}
                                      value={step.dialogue_config.opening_prompt_mode || "base_question"}
                                      options={[
                                        {
                                          label: "Use base question",
                                          value: "base_question"
                                        },
                                        {
                                          label: "Use custom opening prompt",
                                          value: "custom_prompt"
                                        }
                                      ]}
                                      onChange={(value) =>
                                        updateDialogueConfig(
                                          index,
                                          "opening_prompt_mode",
                                          value as ChatWorkflowDialogueConfig["opening_prompt_mode"]
                                        )
                                      }
                                    />
                                  </label>
                                  <label className="block">
                                    <span className="mb-1 block text-sm font-medium">
                                      Max rounds
                                    </span>
                                    <InputNumber
                                      min={1}
                                      max={12}
                                      value={step.dialogue_config.max_rounds}
                                      onChange={(value) =>
                                        updateDialogueConfig(
                                          index,
                                          "max_rounds",
                                          Number(value) || 1
                                        )
                                      }
                                    />
                                  </label>
                                </div>
                                {step.dialogue_config.opening_prompt_mode === "custom_prompt" ? (
                                  <label className="block">
                                    <span className="mb-1 block text-sm font-medium">
                                      Custom opening prompt
                                    </span>
                                    <TextArea
                                      aria-label={`Custom opening prompt for step ${index + 1}`}
                                      autoSize={{ minRows: 2, maxRows: 4 }}
                                      placeholder="How should the first moderator prompt read?"
                                      value={step.dialogue_config.opening_prompt_text || ""}
                                      onChange={(event) =>
                                        updateDialogueConfig(
                                          index,
                                          "opening_prompt_text",
                                          event.target.value
                                        )
                                      }
                                    />
                                  </label>
                                ) : null}
                                <label className="block">
                                  <span className="mb-1 block text-sm font-medium">
                                    Finish conditions
                                  </span>
                                  <TextArea
                                    aria-label={`Finish conditions for step ${index + 1}`}
                                    autoSize={{ minRows: 2, maxRows: 4 }}
                                    placeholder="One condition per line."
                                    value={step.dialogue_config.finish_conditions.join("\n")}
                                    onChange={(event) =>
                                      updateDialogueConfig(
                                        index,
                                        "finish_conditions",
                                        event.target.value
                                          .split("\n")
                                          .map((entry) => entry.trim())
                                          .filter(Boolean)
                                      )
                                    }
                                  />
                                </label>
                                <div className="grid gap-4 md:grid-cols-2">
                                  <label className="block">
                                    <span className="mb-1 block text-sm font-medium">
                                      Debate instructions
                                    </span>
                                    <TextArea
                                      aria-label={`Debate instructions for step ${index + 1}`}
                                      autoSize={{ minRows: 3, maxRows: 5 }}
                                      placeholder="How should the debate LLM challenge the user?"
                                      value={step.dialogue_config.debate_instruction_prompt}
                                      onChange={(event) =>
                                        updateDialogueConfig(
                                          index,
                                          "debate_instruction_prompt",
                                          event.target.value
                                        )
                                      }
                                    />
                                  </label>
                                  <label className="block">
                                    <span className="mb-1 block text-sm font-medium">
                                      Moderator instructions
                                    </span>
                                    <TextArea
                                      aria-label={`Moderator instructions for step ${index + 1}`}
                                      autoSize={{ minRows: 3, maxRows: 5 }}
                                      placeholder="What should the moderator optimize for?"
                                      value={step.dialogue_config.moderator_instruction_prompt}
                                      onChange={(event) =>
                                        updateDialogueConfig(
                                          index,
                                          "moderator_instruction_prompt",
                                          event.target.value
                                        )
                                      }
                                    />
                                  </label>
                                </div>
                                <div className="grid gap-4 md:grid-cols-2">
                                  <div className="space-y-3 rounded-xl border border-border bg-background px-4 py-3">
                                    <Text strong className="block">
                                      Debate LLM
                                    </Text>
                                    <Input
                                      aria-label={`Debate provider for step ${index + 1}`}
                                      placeholder="Provider"
                                      value={step.dialogue_config.debate_llm_config.provider || ""}
                                      onChange={(event) =>
                                        updateDialogueSelection(
                                          index,
                                          "debate_llm_config",
                                          "provider",
                                          event.target.value
                                        )
                                      }
                                    />
                                    <Input
                                      aria-label={`Debate model for step ${index + 1}`}
                                      placeholder="Model"
                                      value={step.dialogue_config.debate_llm_config.model}
                                      onChange={(event) =>
                                        updateDialogueSelection(
                                          index,
                                          "debate_llm_config",
                                          "model",
                                          event.target.value
                                        )
                                      }
                                    />
                                  </div>
                                  <div className="space-y-3 rounded-xl border border-border bg-background px-4 py-3">
                                    <Text strong className="block">
                                      Moderator LLM
                                    </Text>
                                    <Input
                                      aria-label={`Moderator provider for step ${index + 1}`}
                                      placeholder="Provider"
                                      value={step.dialogue_config.moderator_llm_config.provider || ""}
                                      onChange={(event) =>
                                        updateDialogueSelection(
                                          index,
                                          "moderator_llm_config",
                                          "provider",
                                          event.target.value
                                        )
                                      }
                                    />
                                    <Input
                                      aria-label={`Moderator model for step ${index + 1}`}
                                      placeholder="Model"
                                      value={step.dialogue_config.moderator_llm_config.model}
                                      onChange={(event) =>
                                        updateDialogueSelection(
                                          index,
                                          "moderator_llm_config",
                                          "model",
                                          event.target.value
                                        )
                                      }
                                    />
                                  </div>
                                </div>
                              </div>
                            ) : (
                              <>
                                <div className="flex flex-col gap-2 rounded-lg border border-dashed border-border px-4 py-3 md:flex-row md:items-center md:justify-between">
                                  <div>
                                    <Text strong className="block">
                                      Use LLM phrasing
                                    </Text>
                                    <Text className="text-sm text-text-muted">
                                      Keep the authored intent, but let the server rephrase the displayed question at run time.
                                    </Text>
                                  </div>
                                  <Switch
                                    checked={step.question_mode === "llm_phrased"}
                                    onChange={(checked) =>
                                      updateStep(
                                        index,
                                        "question_mode",
                                        checked ? "llm_phrased" : "stock"
                                      )
                                    }
                                  />
                                </div>
                                {step.question_mode === "llm_phrased" ? (
                                  <label className="block">
                                    <span className="mb-1 block text-sm font-medium">
                                      Phrasing instructions
                                    </span>
                                    <TextArea
                                      aria-label={`Phrasing instructions for step ${index + 1}`}
                                      autoSize={{ minRows: 2, maxRows: 4 }}
                                      placeholder="Example: keep it concise and ask for operational detail."
                                      value={step.phrasing_instructions || ""}
                                      onChange={(event) =>
                                        updateStep(
                                          index,
                                          "phrasing_instructions",
                                          event.target.value
                                        )
                                      }
                                    />
                                  </label>
                                ) : null}
                              </>
                            )}
                          </div>
                        </Card>
                      ))}
                    </div>

                    <Button icon={<Plus className="h-4 w-4" />} onClick={addStep}>
                      Add Step
                    </Button>
                  </div>
                </Card>

                <div className="space-y-4">
                  <Card title="Builder notes">
                    <div className="space-y-3 text-sm text-text-muted">
                      <p>
                        Keep each step narrow. If a question could produce two different kinds of answer, split it into two steps.
                      </p>
                      <p>
                        Starting a run from the builder freezes the current draft, so later edits in the template do not mutate the live session.
                      </p>
                      <p>
                        Dialogue round steps keep the workflow linear while allowing multiple moderated rounds inside a single step.
                      </p>
                    </div>
                  </Card>
                  <Card title="Run behavior">
                    <div className="space-y-2 text-sm text-text-muted">
                      <p>
                        Saved templates can be launched from the library. Unsaved drafts can also run immediately when you want to validate the question flow before saving it.
                      </p>
                    </div>
                  </Card>
                </div>
              </div>
            )
          },
          {
            key: "generate",
            label: "Generate",
            children: (
              <Card title="Draft from a goal">
                <div className="grid gap-4 lg:grid-cols-[minmax(0,1.3fr)_minmax(260px,0.7fr)]">
                  <div className="space-y-4">
                    <label className="block">
                      <span className="mb-1 block text-sm font-medium">
                        Workflow goal
                      </span>
                      <Input
                        aria-label="Workflow goal"
                        placeholder="E.g. turn an intake call into a crisp implementation brief"
                        value={generateGoal}
                        onChange={(event) => setGenerateGoal(event.target.value)}
                      />
                    </label>
                    <label className="block">
                      <span className="mb-1 block text-sm font-medium">
                        Base question override
                      </span>
                      <Input
                        aria-label="Base question override"
                        placeholder="Optional: What must this workflow learn before it can finish?"
                        value={generateBaseQuestion}
                        onChange={(event) => setGenerateBaseQuestion(event.target.value)}
                      />
                    </label>
                    <label className="block">
                      <span className="mb-1 block text-sm font-medium">
                        Desired step count
                      </span>
                      <InputNumber
                        min={1}
                        max={12}
                        value={desiredStepCount}
                        onChange={(value) => setDesiredStepCount(Number(value) || 4)}
                      />
                    </label>
                    <Space>
                      <Button
                        type="primary"
                        icon={<Sparkles className="h-4 w-4" />}
                        onClick={generateDraft}
                        loading={generateDraftMutation.isPending}>
                        Generate Draft
                      </Button>
                    </Space>
                  </div>
                  <div className="rounded-2xl border border-dashed border-border bg-surface p-5">
                    <Text strong className="block">
                      Suggested use
                    </Text>
                    <Text className="mt-2 block text-sm text-text-muted">
                      Start with the outcome you need, not the first question you want to ask. The draft generator is best when it can infer progression from a concrete target.
                    </Text>
                    <Text className="mt-4 block text-sm text-text-muted">
                      Example goals:
                    </Text>
                    <ul className="mt-2 list-disc space-y-2 pl-5 text-sm text-text-muted">
                      <li>Prepare an implementation brief for a new feature.</li>
                      <li>Qualify an inbound lead before handing off to sales.</li>
                      <li>Capture a bug report with enough detail to reproduce it.</li>
                    </ul>
                  </div>
                </div>
              </Card>
            )
          },
          {
            key: "run",
            label: "Run",
            children: !activeRunId ? (
              <Empty description="Start a workflow from the library or builder to open the guided run screen." />
            ) : activeRunQuery.isLoading && !activeRun ? (
              <div className="flex min-h-[280px] items-center justify-center">
                <Spin />
              </div>
            ) : activeRunQuery.isError ? (
              <Alert
                variant="error"
                title={<span className="text-text">Unable to load the active workflow run.</span>}
              >
                <span className="text-text">
                  {activeRunQuery.error?.message || "Try starting the run again."}
                </span>
              </Alert>
            ) : activeRun ? (
              <div className="grid gap-6 xl:grid-cols-[minmax(0,1.35fr)_minmax(320px,0.85fr)]">
                <Card
                  title="Run in progress"
                  extra={
                    <Space wrap>
                      <Tag
                        color={
                          activeRun.status === "completed"
                            ? "green"
                            : activeRun.status === "canceled"
                              ? "default"
                              : "blue"
                        }>
                        {activeRun.status}
                      </Tag>
                      <Button onClick={() => setActiveTab("library")}>Back to Library</Button>
                    </Space>
                  }>
                  <div className="space-y-5">
                    <div className="rounded-2xl border border-border bg-surface p-4">
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <div>
                          <Text strong className="block">
                            {activeRunTemplate?.title || "Current workflow"}
                          </Text>
                          <Text className="text-sm text-text-muted">
                            {completedRunSteps} of {totalRunSteps} steps answered
                          </Text>
                        </div>
                        <Space size={8} wrap>
                          <Tag>
                            {activeRun.status === "active"
                              ? `Step ${Math.min(activeRun.current_step_index + 1, totalRunSteps)} of ${totalRunSteps}`
                              : `${completedRunSteps} / ${totalRunSteps}`}
                          </Tag>
                          {isDialogueStep ? (
                            <Tag color="gold">
                              Round {(activeRun.current_round_index || 0) + 1}
                            </Tag>
                          ) : null}
                        </Space>
                      </div>
                    </div>

                    {terminalRunNotice && (
                      <div aria-hidden="true">
                        <Alert
                          variant={terminalRunNotice.variant}
                          aria-live="off"
                          title={<span className="text-text">{terminalRunNotice.title}</span>}
                        >
                          <span className="text-text">{terminalRunNotice.description}</span>
                        </Alert>
                      </div>
                    )}

                    <Card size="small" title="Run history" className="border border-border bg-surface">
                      {runHistoryMessages.length === 0 ? (
                        <Empty
                          image={Empty.PRESENTED_IMAGE_SIMPLE}
                          description="Answers will appear here as the workflow progresses."
                        />
                      ) : (
                        <div className="space-y-3">
                          {runHistoryMessages.map((message, index) => (
                            <div
                              key={`${message.role}-${message.step_index ?? "free"}-${index}`}
                              className="rounded-xl border border-border px-4 py-3">
                              <Text strong className="block text-xs uppercase text-text-muted">
                                {getHistoryLabel(message)}
                              </Text>
                              <Text className="block whitespace-pre-wrap text-sm">
                                {message.content}
                              </Text>
                            </div>
                          ))}
                        </div>
                      )}
                    </Card>

                    {activeRun.status === "active" && activeRunPrompt ? (
                      <Card
                        size="small"
                        title={isDialogueStep ? "Current round" : "Current question"}
                        className="border border-border">
                        <div className="space-y-4">
                          <Text className="block whitespace-pre-wrap text-base">
                            {activeRunPrompt}
                          </Text>
                          <label className="block">
                            <span className="mb-1 block text-sm font-medium">
                              {isDialogueStep
                                ? "Response for current round"
                                : "Answer for current step"}
                            </span>
                            <TextArea
                              aria-label={
                                isDialogueStep
                                  ? "Response for current round"
                                  : "Answer for current step"
                              }
                              autoSize={{ minRows: 4, maxRows: 8 }}
                              placeholder={
                                isDialogueStep
                                  ? "Write the next response you want entered into the dialogue."
                                  : "Write the answer you want saved for this step."
                              }
                              value={runAnswerText}
                              onChange={(event) => setRunAnswerText(event.target.value)}
                            />
                          </label>
                          <Space wrap>
                            <Button
                              type="primary"
                              icon={<Play className="h-4 w-4" />}
                              onClick={() => void submitCurrentAnswer()}
                              loading={
                                isDialogueStep
                                  ? submitRoundMutation.isPending
                                  : submitAnswerMutation.isPending
                              }>
                              {isDialogueStep ? "Submit Response" : "Submit Answer"}
                            </Button>
                            <Button
                              danger
                              icon={<XCircle className="h-4 w-4" />}
                              onClick={() => void cancelActiveRun()}
                              loading={cancelRunMutation.isPending}>
                              Cancel Run
                            </Button>
                          </Space>
                        </div>
                      </Card>
                    ) : null}
                  </div>
                </Card>

                <div className="space-y-4">
                  <Card title="Run controls">
                    <div className="space-y-3 text-sm text-text-muted">
                      <p>
                        Workflow runs are immutable snapshots. Editing the template later does not change this execution.
                      </p>
                      {isDialogueStep ? (
                        <p>
                          Dialogue rounds stay inside the current workflow step until the moderator decides to finish or the round cap is reached.
                        </p>
                      ) : null}
                      <p>
                        The free-chat handoff is explicit and only appears after the workflow reaches a completed state.
                      </p>
                    </div>
                    {activeRun.status === "completed" ? (
                      <div className="mt-4">
                        <Button
                          type="primary"
                          icon={<MessageCircleMore className="h-4 w-4" />}
                          onClick={() => void continueToChat()}
                          loading={continueRunMutation.isPending}>
                          Continue to Chat
                        </Button>
                      </div>
                    ) : null}
                  </Card>
                  <Card title="Context">
                    <div className="space-y-2 text-sm text-text-muted">
                      <p>
                        Attached context selection stays explicit in v1. This screen currently uses the authored template snapshot and prior answers only.
                      </p>
                    </div>
                  </Card>
                </div>
              </div>
            ) : (
              <Empty description="Start a workflow to load the run screen." />
            )
          }
        ]}
      />
      </PageShell>
    </WorkspaceConnectionGate>
  )
}

export default ChatWorkflowsPage
