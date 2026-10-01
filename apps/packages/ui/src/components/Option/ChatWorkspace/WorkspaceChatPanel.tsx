import React from "react"
import { X } from "lucide-react"

import { PlaygroundMessage, type MessageRecoveryAction } from "@/components/Common/Playground/Message"
import { HistorySelectionReview } from "@/components/Common/Playground/HistorySelectionReview"
import { useWorkspaceChatCheckpoint } from "@/hooks/chat/useWorkspaceChatCheckpoint"
import type { HistoryTurnRecovery } from "@/db/dexie/types"
import {
  type ChatSubmitResult,
  isChatSubmitSuccess,
  normalizeChatSubmitResult
} from "@/hooks/chat/chat-action-utils"
import { cancelChatMacroRun } from "@/services/chat-macros"
import { fetchChatModels } from "@/services/tldw-server"
import { resolveServicePromptScope } from "@/services/service-prompts"
import type { Message } from "@/store/option"
import { useStoreChatModelSettings, type ChatModelSettings } from "@/store/model"
import { useMessageOption } from "@/hooks/useMessageOption"
import { useSetting } from "@/hooks/useSetting"
import { useDarkModeStore } from "@/hooks/useDarkmode"
import {
  CUSTOM_THEMES_SETTING,
  THEME_PRESET_SETTING
} from "@/services/settings/ui-settings"
import { getDefaultTheme, getThemeById } from "@/themes/presets"
import { meetsTextContrast } from "@/themes/contrast"
import type { AssistantSelection } from "@/types/assistant-selection"
import type { ChatScope } from "@/types/chat-scope"
import type {
  EffectiveWorkspaceAssistantDefault,
  WorkspacePersonaMemoryMode
} from "@/types/workspace"
import { sanitizeServerErrorMessage } from "@/utils/server-error-message"
import { isGreetingMessageType } from "@/utils/character-greetings"

import { ContextStagingCard } from "./ContextStagingCard"
import { MacroRunDetailDrawer } from "./MacroRunDetailDrawer"
import {
  MacroStatusCard,
  isChatMacroStatusComplete,
  type ChatMacroStatusMetadata
} from "./MacroStatusCard"
import {
  formatStagedSourceInsertText,
  getReadyStagedMediaIds
} from "./staging"
import type {
  ChatWorkspaceAssistantSource,
  ChatWorkspaceRuntimeState,
  StagedWorkspaceSource
} from "./types"
import { normalizeWorkspaceId } from "./workspaceIdentity"

export type WorkspaceChatPanelProps = {
  workspaceId?: string | null
  workspaceReady?: boolean
  workspaceName?: string | null
  stagedSources: StagedWorkspaceSource[]
  onClearStagedSources: () => void
  onRemoveStagedSource?: (sourceId: string) => void
  backendAvailable: boolean
  effectiveAssistantDefault?: EffectiveWorkspaceAssistantDefault | null
  onRuntimeStateChange?: (state: ChatWorkspaceRuntimeState) => void
}

const noop = () => undefined

type WorkspaceSubmit = Parameters<ReturnType<typeof useMessageOption>["onSubmit"]>[0]
type RetryModel = Awaited<ReturnType<typeof fetchChatModels>>[number]
const getRetryModelKey = (model: RetryModel) =>
  JSON.stringify([model.provider ?? "", model.model])
type FailedWorkspaceTurn = {
  request: WorkspaceSubmit & {
    requestOverrides: NonNullable<WorkspaceSubmit["requestOverrides"]> & {
      currentChatModelSettings: ChatModelSettings
    }
  }
  memory: ReturnType<typeof useMessageOption>["history"]
  priorMessages: Message[]
  workspaceId: string | null
  conversationId: string | null
  assistantKey: string
  userMessageId?: string
  assistantMessageId?: string
}

type WorkspacePanelMessage = Message &
  Partial<{
    content: string
    text: string
  }>

const getMessageText = (message: WorkspacePanelMessage): string => {
  if (typeof message?.message === "string") return message.message
  if (typeof message?.content === "string") return message.content
  if (typeof message?.text === "string") return message.text
  return ""
}

const getMessageRole = (message: WorkspacePanelMessage): "user" | "assistant" | "system" => {
  if (message?.role === "system") return "system"
  if (message?.role === "user" || message?.isBot === false) return "user"
  return "assistant"
}

const getWorkspaceAttemptMessages = (attempt: FailedWorkspaceTurn, messages: Message[]) => {
  let greetingCount = 0
  while (
    isGreetingMessageType(messages[greetingCount]?.messageType) &&
    !attempt.priorMessages.some((message) => message.id === messages[greetingCount]?.id)
  ) {
    greetingCount += 1
  }
  if (!attempt.priorMessages.every((message, index) => message.id === messages[greetingCount + index]?.id)) {
    return null
  }
  const userIndex = greetingCount + attempt.priorMessages.length
  const remaining = messages.slice(userIndex)
  const [user, assistant] = remaining
  if (
    remaining.length > 2 ||
    (user && (!user.id || getMessageRole(user) !== "user" || getMessageText(user) !== attempt.request.message)) ||
    (assistant && (!assistant.id || getMessageRole(assistant) !== "assistant" ||
      (assistant.parentMessageId != null && assistant.parentMessageId !== user?.id)))
  ) {
    return null
  }
  return { user, assistant, priorMessages: messages.slice(0, userIndex) }
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const optionalString = (value: unknown): string | null =>
  typeof value === "string" && value.trim() ? value : null

const optionalNumber = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null

const getChatMacroMetadata = (
  metadataExtra: unknown
): ChatMacroStatusMetadata | null => {
  if (!isRecord(metadataExtra) || !isRecord(metadataExtra.chat_macro)) return null
  const raw = metadataExtra.chat_macro
  const runId = optionalString(raw.run_id)
  const status = optionalString(raw.status)
  if (!runId || !status) return null

  return {
    run_id: runId,
    name: optionalString(raw.name),
    command: optionalString(raw.command),
    status,
    detail_url: optionalString(raw.detail_url),
    output_profile: optionalString(raw.output_profile),
    branch_count: optionalNumber(raw.branch_count)
  }
}

type InheritedWorkspacePersonaAssistant = AssistantSelection & { kind: "persona" }

const normalizePersonaMemoryMode = (
  value: WorkspacePersonaMemoryMode | null | undefined
): WorkspacePersonaMemoryMode => {
  return value === "read_write" ? "read_write" : "read_only"
}

const buildInheritedWorkspaceAssistant = (
  effectiveAssistantDefault: EffectiveWorkspaceAssistantDefault | null | undefined,
  workspaceReady: boolean
): InheritedWorkspacePersonaAssistant | null => {
  if (!workspaceReady || effectiveAssistantDefault?.status !== "available") {
    return null
  }
  if (
    effectiveAssistantDefault.assistantKind !== "persona" ||
    !effectiveAssistantDefault.assistantId
  ) {
    return null
  }

  const personaMemoryMode = normalizePersonaMemoryMode(
    effectiveAssistantDefault.personaMemoryMode
  )

  return {
    kind: "persona",
    id: effectiveAssistantDefault.assistantId,
    name: effectiveAssistantDefault.label ?? "Workspace Persona",
    metadata: {
      selectionMode: "tracked",
      source: "workspace",
      personaMemoryMode
    }
  }
}

export const WorkspaceChatPanel = ({
  workspaceId,
  workspaceReady: workspaceHydrated = true,
  workspaceName,
  stagedSources,
  onClearStagedSources,
  onRemoveStagedSource,
  backendAvailable,
  effectiveAssistantDefault,
  onRuntimeStateChange
}: WorkspaceChatPanelProps) => {
  const composerRef = React.useRef<HTMLTextAreaElement>(null)
  const retryModelRef = React.useRef<HTMLSelectElement>(null)
  const retryModelId = React.useId()
  const requestInFlight = React.useRef(false)
  const activeTurnRef = React.useRef<FailedWorkspaceTurn | null>(null)
  const [draft, setDraftState] = React.useState("")
  const pendingReprepareUUID = React.useRef<string | null>(null)
  const [sendError, setSendError] = React.useState<string | null>(null)
  const [failedTurn, setFailedTurn] = React.useState<FailedWorkspaceTurn | null>(null)
  const [settledAttempt, setSettledAttempt] = React.useState<{
    attempt: FailedWorkspaceTurn
    result: ChatSubmitResult
    boundTurn?: FailedWorkspaceTurn
    previousTurn?: FailedWorkspaceTurn
  } | null>(null)
  const [preparationPending, setPreparationPending] = React.useState(false)
  const [recoveryPending, setRecoveryPending] = React.useState(false)
  const [modelPickerOpen, setModelPickerOpen] = React.useState(false)
  const [retryModel, setRetryModel] = React.useState("")
  const [retryModels, setRetryModels] = React.useState<Awaited<ReturnType<typeof fetchChatModels>>>([])
  const currentChatModelSettings = useStoreChatModelSettings()
  const [themeId] = useSetting(THEME_PRESET_SETTING)
  const [customThemes] = useSetting(CUSTOM_THEMES_SETTING)
  const mode = useDarkModeStore((state) => state.mode)
  const palette = (getThemeById(themeId, customThemes) ?? getDefaultTheme()).palette[
    mode === "dark" ? "dark" : "light"
  ]
  const primaryForeground = meetsTextContrast("255 255 255", palette.primaryStrong)
    ? "#ffffff"
    : "#000000"
  const [modelsLoading, setModelsLoading] = React.useState(false)
  const [modelsError, setModelsError] = React.useState<string | null>(null)
  const [detailRunId, setDetailRunId] = React.useState<string | null>(null)
  const [macroStatusOverrides, setMacroStatusOverrides] = React.useState<
    Record<string, string>
  >({})
  const normalizedWorkspaceId = React.useMemo(
    () => normalizeWorkspaceId(workspaceId),
    [workspaceId]
  )
  const workspaceReady = workspaceHydrated && normalizedWorkspaceId !== null
  const chatBackendAvailable = backendAvailable && workspaceReady

  const scope = React.useMemo<ChatScope>(
    () =>
      normalizedWorkspaceId
        ? { type: "workspace", workspaceId: normalizedWorkspaceId }
        : { type: "global" },
    [normalizedWorkspaceId]
  )
  const inheritedAssistant = React.useMemo(
    () => buildInheritedWorkspaceAssistant(effectiveAssistantDefault, workspaceReady),
    [effectiveAssistantDefault, workspaceReady]
  )
  const inheritedPersonaMemoryMode = inheritedAssistant
    ? normalizePersonaMemoryMode(
        effectiveAssistantDefault?.personaMemoryMode ?? null
      )
    : null
  const messageOptionArgs = React.useMemo(
    () =>
      inheritedAssistant
        ? {
            hydrateServerChat: workspaceReady,
            scope,
            inheritedAssistant,
            inheritedPersonaMemoryMode
          }
        : { scope, hydrateServerChat: workspaceReady },
    [inheritedAssistant, inheritedPersonaMemoryMode, scope, workspaceReady]
  )

  const chat = useMessageOption(messageOptionArgs)
  const {
    messages,
    history,
    setMessages,
    onSubmit,
    streaming,
    isLoading,
    isProcessing,
    stopStreamingRequest,
    selectedModel,
    temporaryChat,
    selectedAssistant,
    selectedAssistantSource,
    serverChatAssistantKind,
    serverChatAssistantId,
    serverChatId,
    serverChatLoadState,
    serverChatLoadError
  } = chat
  const checkpoint = useWorkspaceChatCheckpoint({
    workspaceId: normalizedWorkspaceId, workspaceReady, draft, setDraft: setDraftState, chat
  })
  const { setDraft } = checkpoint
  const recoveryReference = checkpoint.controller?.getReference()
  React.useEffect(() => {
    pendingReprepareUUID.current = null
  }, [normalizedWorkspaceId, checkpoint.referenceId, recoveryReference?.owner_key, recoveryReference?.conversation_id])
  const reprepareRecovery = React.useCallback((turn: HistoryTurnRecovery) => {
    const current = checkpoint.controller?.getCurrent()
    const reference = checkpoint.controller?.getReference()
    if (turn.persistence !== "server" || !turn.logical_user_message_id ||
      !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(turn.logical_user_message_id) ||
      current?.status !== "ready" || current.owner?.kind !== "native" || !current.owner.validate_lease() ||
      reference?.owner_key !== turn.owner_key || reference.conversation_id !== turn.conversation_id ||
      turn.conversation_id !== serverChatId || temporaryChat) return
    pendingReprepareUUID.current = turn.logical_user_message_id
    setDraft(turn.input_text)
    composerRef.current?.focus()
  }, [checkpoint.controller, serverChatId, setDraft, temporaryChat])
  const historyLoading = serverChatLoadState === "loading" || checkpoint.restoring
  const historyLoadError =
    serverChatLoadError ||
    (serverChatLoadState === "failed" ? "Chat history unavailable" : null)
  const sending = isLoading || isProcessing || preparationPending || recoveryPending
  const isSending = streaming || sending
  const historyReady = !historyLoading && !historyLoadError

  const selectedMatchesWorkspaceAssistant =
    selectedAssistantSource === "workspace" &&
    selectedAssistant?.kind === "persona" &&
    inheritedAssistant?.kind === "persona" &&
    selectedAssistant.id === inheritedAssistant.id
  const restoredWorkspaceAssistant =
    !selectedAssistant &&
    inheritedAssistant?.kind === "persona" &&
    serverChatAssistantKind === "persona" &&
    serverChatAssistantId === inheritedAssistant.id
  const usingWorkspaceAssistant =
    Boolean(inheritedAssistant) &&
    (selectedMatchesWorkspaceAssistant ||
      restoredWorkspaceAssistant ||
      (selectedAssistantSource === "workspace" &&
        !selectedAssistant &&
        messages.length === 0))
  const assistantSource: ChatWorkspaceAssistantSource = usingWorkspaceAssistant
    ? "workspace"
    : selectedAssistant || serverChatAssistantKind || serverChatAssistantId
      ? "explicit"
      : effectiveAssistantDefault?.status === "unavailable"
        ? "unavailable"
        : "none"
  const runtimeSelectedPersonaLabel =
    assistantSource === "workspace"
      ? inheritedAssistant?.name ?? selectedAssistant?.name ?? null
      : selectedAssistant?.name ??
        (assistantSource === "explicit" && serverChatAssistantKind === "persona"
          ? "Persona"
          : null)
  const assistantKey = JSON.stringify([
    selectedAssistant?.kind ?? serverChatAssistantKind ?? inheritedAssistant?.kind ?? null,
    selectedAssistant?.id ?? serverChatAssistantId ?? inheritedAssistant?.id ?? null
  ])

  React.useEffect(() => {
    if (!checkpoint.active) setDraft("")
    setSendError(null)
    setMacroStatusOverrides({})
    setFailedTurn(null)
    setSettledAttempt(null)
    setModelPickerOpen(false)
    activeTurnRef.current = null
    requestInFlight.current = false
    setPreparationPending(false)
    setRecoveryPending(false)
  }, [normalizedWorkspaceId, checkpoint.referenceId, checkpoint.active, setDraft])

  // A skipped result can precede the transition publishing final interruption metadata.
  React.useEffect(() => {
    if (!settledAttempt) return
    const { attempt, result, boundTurn, previousTurn } = settledAttempt
    if (activeTurnRef.current !== attempt) {
      setSettledAttempt(null)
      return
    }
    const target = boundTurn ?? attempt
    const turn = getWorkspaceAttemptMessages(target, messages)
    const unreplacedTurn = previousTurn && turn &&
      previousTurn.assistantMessageId === turn.assistant?.id ? previousTurn : undefined
    const intentionalSkip = result.status === "skipped" && [
      "Request cancelled", "Request scope changed", "Character chat turn aborted"
    ].includes(result.reason)
    if (
      (intentionalSkip && !unreplacedTurn) || !turn || target.workspaceId !== normalizedWorkspaceId || target.assistantKey !== assistantKey ||
      (target.conversationId != null && target.conversationId !== serverChatId) ||
      (target.conversationId == null && serverChatId != null && (!turn.user || boundTurn)) ||
      (boundTurn && (boundTurn.userMessageId !== turn.user?.id || boundTurn.assistantMessageId !== turn.assistant?.id))
    ) {
      setSettledAttempt(null)
      setFailedTurn(null)
      return
    }
    const capturedTurn = unreplacedTurn ?? boundTurn ?? {
      ...target,
      conversationId: serverChatId,
      priorMessages: turn.priorMessages,
      userMessageId: turn.user?.id,
      assistantMessageId: turn.assistant?.id
    }
    if (!unreplacedTurn && result.status === "skipped" && turn.assistant?.generationInfo?.streamTransportInterrupted !== true) {
      if (!boundTurn) setSettledAttempt({ attempt, result, boundTurn: capturedTurn })
      return
    }
    setSettledAttempt(null)
    setSendError(intentionalSkip ? null : result.status === "failed" ? result.errorMessage || "Send failed" : result.status === "skipped" ? result.reason : null)
    setFailedTurn(capturedTurn)
  }, [assistantKey, messages, normalizedWorkspaceId, serverChatId, settledAttempt])

  React.useEffect(() => {
    if (!modelPickerOpen) return
    let cancelled = false
    setModelsLoading(true)
    setModelsError(null)
    setRetryModels([])
    void fetchChatModels({ returnEmpty: true }).then((models) => {
      if (!cancelled) setRetryModels(models)
    }).catch((error) => {
      if (!cancelled) setModelsError(sanitizeServerErrorMessage(error, "Unable to load models"))
    }).finally(() => {
      if (!cancelled) setModelsLoading(false)
    })
    return () => { cancelled = true }
  }, [modelPickerOpen])

  React.useEffect(() => {
    if (modelPickerOpen && !modelsLoading) retryModelRef.current?.focus()
  }, [modelPickerOpen, modelsLoading])

  React.useEffect(() => {
    onRuntimeStateChange?.({
      backendAvailable: chatBackendAvailable,
      streaming,
      sending,
      historyLoading,
      historyLoadError,
      sendError,
      selectedModelLabel: selectedModel || "No model selected",
      hasModelSelected: Boolean(selectedModel),
      selectedPersonaLabel: runtimeSelectedPersonaLabel,
      assistantSource,
      workspaceAssistantDegradedReason:
        assistantSource === "unavailable"
          ? effectiveAssistantDefault?.degradedReason ?? null
          : null
    })
  }, [
    assistantSource,
    chatBackendAvailable,
    effectiveAssistantDefault?.degradedReason,
    onRuntimeStateChange,
    runtimeSelectedPersonaLabel,
    selectedModel,
    sendError,
    sending,
    historyLoading,
    historyLoadError,
    streaming
  ])

  const hasStagedContext = stagedSources.length > 0
  const readyMediaIds = React.useMemo(
    () => getReadyStagedMediaIds(stagedSources),
    [stagedSources]
  )
  const hasReadyMedia = readyMediaIds.length > 0
  const hasUncarriedStagedContext = stagedSources.some(
    (source) =>
      source.availability !== "ready" ||
      typeof source.mediaId !== "number" ||
      !Number.isInteger(source.mediaId) ||
      source.mediaId <= 0
  )
  const trimmedDraft = draft.trim()
  const sendDisabled =
    !chatBackendAvailable ||
    !historyReady ||
    isSending ||
    (!trimmedDraft && !hasStagedContext)
  const conversationInstanceId = normalizedWorkspaceId ?? "workspace-chat"

  const insertStagedSummary = React.useCallback(() => {
    const insertText = formatStagedSourceInsertText(stagedSources)
    if (!insertText) return

    pendingReprepareUUID.current = null
    setDraft((current) => {
      if (!current) return insertText
      const separator = current.endsWith("\n") ? "" : "\n\n"
      return `${current}${separator}${insertText}`
    })
    onClearStagedSources()
    composerRef.current?.focus()
  }, [onClearStagedSources, stagedSources, setDraft])

  const submitMessage = React.useCallback(async () => {
    if (sendDisabled) return
    if (!chatBackendAvailable) return

    setSendError(null)
    const fallbackContext = formatStagedSourceInsertText(stagedSources).trim()
    const sendMessage =
      trimmedDraft && hasStagedContext && hasUncarriedStagedContext && fallbackContext
        ? `${trimmedDraft}\n\n${fallbackContext}`
        : trimmedDraft || fallbackContext

    if (requestInFlight.current) return
    const reference = checkpoint.controller?.getReference()
    if (pendingReprepareUUID.current && !checkpoint.controller?.recoveries.some(({ turn }) =>
      turn.persistence === "server" && turn.logical_user_message_id === pendingReprepareUUID.current &&
      turn.owner_key === reference?.owner_key && turn.conversation_id === serverChatId &&
      turn.input_text === sendMessage)) pendingReprepareUUID.current = null
    requestInFlight.current = true
    let attempt: FailedWorkspaceTurn | undefined
    try {
      attempt = {
        workspaceId: normalizedWorkspaceId,
        conversationId: serverChatId,
        assistantKey,
        priorMessages: [...messages],
        memory: [...history],
        request: {
          message: sendMessage,
          image: "",
          requestOverrides: {
            selectedModel,
            ...(!temporaryChat ? { tldwTurn: { user_message_id: pendingReprepareUUID.current ?? crypto.randomUUID() } } : {}),
            currentChatModelSettings: { ...currentChatModelSettings },
            ragMediaIds: readyMediaIds,
            fileRetrievalEnabled: hasReadyMedia,
            chatMode: hasReadyMedia ? "rag" : "normal",
            ...(usingWorkspaceAssistant && inheritedAssistant
              ? {
                  assistant_kind: "persona" as const,
                  assistant_id: inheritedAssistant.id,
                  persona_memory_mode: inheritedPersonaMemoryMode ?? "read_only"
                }
              : {})
          }
        }
      }
      activeTurnRef.current = attempt
      setPreparationPending(true)
      setSettledAttempt(null)
      setFailedTurn(null)
      setModelPickerOpen(false)

      const capturedScope = await resolveServicePromptScope()
      if (activeTurnRef.current !== attempt) return
      attempt.request.requestOverrides.requestScope = Object.freeze({
        config: capturedScope.config,
        userId: capturedScope.userId
      })
      const result = normalizeChatSubmitResult(
        (await onSubmit(attempt.request)) as ChatSubmitResult | undefined
      )
      if (activeTurnRef.current !== attempt) return

      if (isChatSubmitSuccess(result)) {
        pendingReprepareUUID.current = null
        setDraft("")
        onClearStagedSources()
        return
      }

      setSettledAttempt({ attempt, result })
    } catch (error) {
      if (!attempt) {
        setSendError("Unable to create a secure turn ID. Use HTTPS or localhost and retry with browser cryptography enabled.")
        return
      }
      if (activeTurnRef.current !== attempt) return
      if (!attempt.request.requestOverrides.requestScope) {
        setSendError(sanitizeServerErrorMessage(error, "Unable to resolve the server and account. Reconnect and retry."))
        return
      }
      setSettledAttempt({ attempt, result: { status: "failed", errorMessage: "Send failed" } })
    } finally {
      if (!attempt || activeTurnRef.current === attempt) requestInFlight.current = false
      if (activeTurnRef.current === attempt) setPreparationPending(false)
    }
  }, [
    assistantKey,
    chatBackendAvailable,
    checkpoint.controller,
    currentChatModelSettings,
    hasReadyMedia,
    hasStagedContext,
    hasUncarriedStagedContext,
    inheritedAssistant,
    inheritedPersonaMemoryMode,
    history,
    messages,
    normalizedWorkspaceId,
    onClearStagedSources,
    onSubmit,
    readyMediaIds,
    sendDisabled,
    selectedModel,
    serverChatId,
    setDraft,
    stagedSources,
    trimmedDraft,
    temporaryChat,
    usingWorkspaceAssistant
  ])

  const currentFailedTurn = failedTurn ? getWorkspaceAttemptMessages(failedTurn, messages) : null
  const failedUser = currentFailedTurn?.user
  const failedAssistant = currentFailedTurn?.assistant
  const ownsFailedTurn = Boolean(
    failedTurn && currentFailedTurn &&
    failedTurn.workspaceId === normalizedWorkspaceId &&
    failedTurn.conversationId === serverChatId &&
    failedTurn.assistantKey === assistantKey &&
    failedTurn.userMessageId === failedUser?.id &&
    failedTurn.assistantMessageId === failedAssistant?.id
  )
  const recoveryDisabled = !ownsFailedTurn || !chatBackendAvailable || !historyReady || isSending

  const retryFailedTurn = async (model?: RetryModel) => {
    if (!failedTurn || recoveryDisabled || requestInFlight.current) return
    requestInFlight.current = true
    setRecoveryPending(true)
    setSendError(null)
    setModelPickerOpen(false)
    const attempt = {
      ...failedTurn,
      request: {
        ...failedTurn.request,
        requestOverrides: {
          ...failedTurn.request.requestOverrides,
          ...(model ? {
            selectedModel: model.model,
            currentChatModelSettings: {
              ...failedTurn.request.requestOverrides.currentChatModelSettings,
              apiProvider: model.provider
            }
          } : {})
        }
      }
    }
    activeTurnRef.current = attempt
    setSettledAttempt(null)
    // Keep the existing user turn, replacing only its failed assistant response.
    const retryMessages = failedAssistant ? messages.slice(0, -1) : [...messages]
    setMessages(retryMessages)
    let result: ChatSubmitResult
    try {
      result = normalizeChatSubmitResult(await onSubmit({
        ...attempt.request,
        isRegenerate: Boolean(failedUser),
        messages: retryMessages,
        memory: attempt.memory,
        regenerateFromMessage: failedAssistant,
        serverChatIdOverride: attempt.conversationId
      }))
    } catch {
      result = { status: "failed", errorMessage: "Send failed" }
    } finally {
      if (activeTurnRef.current === attempt) {
        requestInFlight.current = false
        setRecoveryPending(false)
        composerRef.current?.focus()
      }
    }
    if (activeTurnRef.current !== attempt) return
    if (isChatSubmitSuccess(result)) setFailedTurn(null)
    else {
      setMessages((current) => failedAssistant && current.length === retryMessages.length &&
        current.every((message, index) => message.id === retryMessages[index].id)
        ? [...current, failedAssistant] : current)
      setSettledAttempt({ attempt, result, previousTurn: failedTurn })
    }
  }

  const recoveryActions: MessageRecoveryAction[] = [
    {
      id: "retry", label: "Retry same model", disabled: recoveryDisabled,
      onClick: () => { void retryFailedTurn() }
    },
    {
      id: "switch", label: "Switch model", disabled: recoveryDisabled,
      onClick: () => {
        setRetryModel(JSON.stringify([
          failedTurn?.request.requestOverrides.currentChatModelSettings.apiProvider ?? "",
          failedTurn?.request.requestOverrides?.selectedModel ?? ""
        ]))
        setModelPickerOpen(true)
      }
    },
    {
      id: "fallback", label: "Try provider fallback", disabled: true,
      disabledReason: "Provider fallback is unavailable in Chat Workspace. Select another model instead.",
      onClick: noop
    }
  ]

  const handleSubmit = React.useCallback(
    (event: React.FormEvent<HTMLFormElement>) => {
      event.preventDefault()
      void submitMessage()
    },
    [submitMessage]
  )

  const handleComposerKeyDown = React.useCallback(
    (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault()
        void submitMessage()
      }
    },
    [submitMessage]
  )

  const handleCancelMacroRun = React.useCallback(async (runId: string) => {
    setSendError(null)
    try {
      const response = await cancelChatMacroRun(runId)
      if (!response.ok || !response.data) {
        setSendError(
          sanitizeServerErrorMessage(
            response.error,
            `Unable to cancel macro run (${response.status})`
          )
        )
        return
      }
      setMacroStatusOverrides((current) => ({
        ...current,
        [runId]: response.data.status
      }))
    } catch (error) {
      setSendError(sanitizeServerErrorMessage(error, "Unable to cancel macro run"))
    }
  }, [])

  const handleOpenMacroRunDetail = React.useCallback((runId: string) => {
    setDetailRunId(runId)
  }, [])

  return (
    <section
      aria-label="Chat workspace panel"
      className="flex h-full min-h-0 flex-col bg-bg text-text"
    >
      <div className="shrink-0 border-b border-border px-4 py-3">
        <h2 className="text-sm font-semibold">
          {workspaceName ? `${workspaceName} chat` : "Workspace chat"}
        </h2>
      </div>

      <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-4 py-3">
        {checkpoint.controller ? (
          <HistorySelectionReview selection={checkpoint.controller} onReprepareRecovery={reprepareRecovery} />
        ) : null}
        {Array.isArray(messages) && messages.length > 0 ? (
          messages.map((message: WorkspacePanelMessage, index: number) => {
            const rawMacroMetadata = getChatMacroMetadata(message?.metadataExtra)
            const macroMetadata = rawMacroMetadata
              ? {
                  ...rawMacroMetadata,
                  status:
                    macroStatusOverrides[rawMacroMetadata.run_id] ||
                    rawMacroMetadata.status
                }
              : null
            if (macroMetadata && !isChatMacroStatusComplete(macroMetadata.status)) {
              return (
                <MacroStatusCard
                  key={macroMetadata.run_id}
                  metadata={macroMetadata}
                  onCancel={handleCancelMacroRun}
                  onOpenDetail={handleOpenMacroRunDetail}
                />
              )
            }

            const role = getMessageRole(message)
            const messageText = getMessageText(message)
            const messageId =
              message?.id != null ? String(message.id) : `workspace-message-${index}`

            return (
              <PlaygroundMessage
                key={messageId}
                conversationInstanceId={conversationInstanceId}
                messageId={messageId}
                message={messageText}
                sources={message?.sources}
                images={message?.images}
                documents={message?.documents}
                toolCalls={message?.toolCalls}
                toolResults={message?.toolResults}
                currentMessageIndex={index}
                totalMessages={messages.length}
                isBot={role !== "user"}
                name={message?.name ?? (role === "user" ? "You" : "Assistant")}
                role={role}
                isProcessing={false}
                isStreaming={Boolean(streaming && index === messages.length - 1)}
                createdAt={message?.createdAt}
                metadataExtra={message?.metadataExtra}
                generationInfo={message?.generationInfo}
                scope={scope}
                recoveryActions={message.id === failedTurn?.assistantMessageId ? recoveryActions : []}
                dynamicUISurface="workspace"
                headingOffset={1}
                hideEditAndRegenerate
                hideContinue
                hideSourceActions
                onRegenerate={noop}
                onEditFormSubmit={noop}
                onContinue={noop}
              />
            )
          })
        ) : (
          <p className="rounded-md border border-dashed border-border bg-surface px-3 py-2 text-sm text-text-muted">
            Start a workspace chat from the composer below.
          </p>
        )}
      </div>

      <div className="flex max-h-[75%] min-h-0 flex-col gap-3 overflow-y-auto border-t border-border bg-surface2/40 px-4 py-3">
        {modelPickerOpen && ownsFailedTurn ? (
          <div
            role="group"
            aria-label="Failed request model"
            className="flex flex-wrap items-end gap-2"
            onKeyDown={(event) => {
              if (event.key === "Escape") {
                setModelPickerOpen(false)
                composerRef.current?.focus()
              }
            }}
          >
            <label htmlFor={retryModelId} className="flex w-full min-w-0 flex-none flex-col gap-1 text-sm sm:w-auto sm:flex-1">
              Retry model
              <select
                id={retryModelId}
                ref={retryModelRef}
                value={retryModel}
                onChange={(event) => setRetryModel(event.target.value)}
                disabled={modelsLoading || recoveryDisabled}
                className="min-h-11 w-full min-w-0 rounded-md border border-border bg-bg px-2 text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              >
                <option value="">Select a model</option>
                {retryModels.map((model) => (
                  <option key={getRetryModelKey(model)} value={getRetryModelKey(model)}>
                    {model.nickname || model.model}{model.provider ? ` (${model.provider})` : ""}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="button"
              disabled={recoveryDisabled || modelsLoading || !retryModels.some((model) => getRetryModelKey(model) === retryModel)}
              onClick={() => {
                const model = retryModels.find((entry) => getRetryModelKey(entry) === retryModel)
                if (model) void retryFailedTurn(model)
              }}
              className="min-h-11 rounded-md border border-border px-3 text-sm disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
            >
              Retry with selected model
            </button>
            <button
              type="button"
              aria-label="Close model selector"
              title="Close model selector"
              className="flex size-11 items-center justify-center rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              onClick={() => {
                setModelPickerOpen(false)
                composerRef.current?.focus()
              }}
            ><X size={18} aria-hidden="true" /></button>
            {modelsLoading ? <p role="status" className="w-full text-sm">Loading models</p> : null}
            {modelsError || (!modelsLoading && retryModels.length === 0) ? (
              <p role="status" className="w-full text-sm">{modelsError || "No models available"}</p>
            ) : null}
          </div>
        ) : null}
        {ownsFailedTurn && !failedAssistant ? (
          <div className="flex flex-wrap gap-2">
            {recoveryActions.map((action) => (
              <span key={action.id} className="flex max-w-full flex-col gap-1">
                <button
                  type="button"
                  onClick={action.onClick}
                  disabled={action.disabled}
                  aria-description={action.disabledReason}
                  className="min-h-11 rounded-md border border-border px-3 text-sm disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                >{action.label}</button>
                {action.disabledReason ? <span className="text-xs text-text-muted">{action.disabledReason}</span> : null}
              </span>
            ))}
          </div>
        ) : null}
        {hasStagedContext || sendError || !workspaceReady ? (
          <div className="min-h-20 overflow-y-auto p-1">
            {hasStagedContext ? (
              <ContextStagingCard
                sources={stagedSources}
                isSending={isSending}
                canSend={chatBackendAvailable && historyReady}
                onClear={() => {
                  onClearStagedSources()
                  composerRef.current?.focus()
                }}
                onInsert={insertStagedSummary}
                onSend={submitMessage}
                onRemoveSource={
                  onRemoveStagedSource
                    ? (sourceId) => {
                        onRemoveStagedSource(sourceId)
                        composerRef.current?.focus()
                      }
                    : undefined
                }
              />
            ) : null}

            {sendError ? (
              <p className="text-sm font-medium text-text" role="alert">
                {sendError}
              </p>
            ) : null}

            {!workspaceReady ? (
              <p
                className="text-sm text-text-muted"
                role="status"
                aria-live="polite"
              >
                Loading workspace context
              </p>
            ) : null}
          </div>
        ) : null}

        {isSending ? (
          <div className="flex shrink-0 items-center justify-between gap-3 text-sm text-text-muted">
            <span>{streaming ? "Streaming" : "Sending"}</span>
            {streaming ? (
              <button
                type="button"
                className="inline-flex min-h-11 items-center rounded-md border border-border px-3 py-1.5 text-sm font-medium text-text transition-colors hover:bg-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus xl:min-h-8"
                onClick={() => {
                  stopStreamingRequest()
                  composerRef.current?.focus()
                }}
                aria-label="Stop generating"
              >
                Stop generating
              </button>
            ) : null}
          </div>
        ) : null}

        <form className="flex shrink-0 flex-col gap-2" onSubmit={handleSubmit}>
          <textarea
            ref={composerRef}
            aria-label="Chat workspace message"
            className="h-[88px] min-h-[64px] max-h-[25dvh] resize-y rounded-md border border-border bg-bg px-3 py-2 text-sm text-text outline-none transition-colors placeholder:text-text-muted focus:border-primary focus:ring-2 focus:ring-focus"
            value={draft}
            onChange={(event) => {
              pendingReprepareUUID.current = null
              setDraft(event.target.value)
            }}
            onKeyDown={handleComposerKeyDown}
            placeholder="Ask about this workspace"
          />
          <div className="flex justify-end">
            <button
              type="submit"
              className="inline-flex min-h-11 items-center rounded-md bg-primaryStrong px-4 py-2 text-sm font-medium transition-colors hover:bg-primaryStrong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50 xl:min-h-9"
              style={{ color: primaryForeground }}
              disabled={sendDisabled}
              aria-label="Send message"
            >
              Send message
            </button>
          </div>
        </form>
      </div>

      <MacroRunDetailDrawer
        runId={detailRunId}
        open={Boolean(detailRunId)}
        onClose={() => setDetailRunId(null)}
      />
    </section>
  )
}
