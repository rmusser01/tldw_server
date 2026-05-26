import React from "react"
import { Button, Card, Empty, Input, Space, Tag, Typography } from "antd"
import { Lightbulb, MessageSquareText, Sparkles, Volume2 } from "lucide-react"
import { useTranslation } from "react-i18next"
import { Link as RouterLink, useInRouterContext } from "react-router-dom"
import { Alert } from "@/components/ui/primitives"
import { useSpeechRecognition } from "@/hooks/useSpeechRecognition"
import { useTTS } from "@/hooks/useTTS"
import type {
  StudyAssistantAction,
  StudyAssistantContextResponse,
  StudyAssistantMessage,
  StudyAssistantRespondRequest
} from "@/services/flashcards"
import { MarkdownWithBoundary } from "./MarkdownWithBoundary"
import { VoiceTranscriptComposer } from "./VoiceTranscriptComposer"

type AssistantErrorKind = "network" | "server" | "no_llm"

function extractHttpStatus(err: unknown): number | undefined {
  if (!err || typeof err !== "object") return undefined
  const e = err as Record<string, unknown>
  // Top-level: err.status, err.statusCode
  if (typeof e.status === "number") return e.status
  if (typeof e.statusCode === "number") return e.statusCode
  // Nested: err.response.status, err.response.statusCode
  const resp = e.response
  if (resp && typeof resp === "object") {
    const r = resp as Record<string, unknown>
    if (typeof r.status === "number") return r.status
    if (typeof r.statusCode === "number") return r.statusCode
  }
  // Data envelope: err.data.status, err.data.statusCode
  const data = e.data
  if (data && typeof data === "object") {
    const d = data as Record<string, unknown>
    if (typeof d.status === "number") return d.status
    if (typeof d.statusCode === "number") return d.statusCode
  }
  // Deep nested: err.response.data.status, err.response.data.statusCode
  if (resp && typeof resp === "object") {
    const rData = (resp as Record<string, unknown>).data
    if (rData && typeof rData === "object") {
      const rd = rData as Record<string, unknown>
      if (typeof rd.status === "number") return rd.status
      if (typeof rd.statusCode === "number") return rd.statusCode
    }
  }
  return undefined
}

function classifyAssistantError(err: unknown): AssistantErrorKind {
  const isNetwork =
    err instanceof TypeError ||
    (err instanceof Error && /network|fetch|timeout/i.test(err.message))
  if (isNetwork) return "network"
  const httpStatus = extractHttpStatus(err)
  if (typeof httpStatus === "number" && httpStatus >= 400) return "server"
  return "no_llm"
}

const { Text, Title } = Typography

const DEFAULT_ACTIONS: StudyAssistantAction[] = [
  "explain",
  "mnemonic",
  "follow_up",
  "fact_check",
  "freeform"
]

interface FlashcardStudyAssistantPanelProps {
  cardUuid: string
  threadVersion?: number | null
  messages: StudyAssistantMessage[]
  availableActions?: StudyAssistantAction[] | null
  assistantContext?: unknown
  isLoading?: boolean
  isError?: boolean
  queryError?: unknown
  isResponding?: boolean
  onReloadContext?: () => Promise<unknown>
  onRespond: (request: StudyAssistantRespondRequest) => Promise<unknown>
  autoSubmitRequest?: {
    token: number
    request: StudyAssistantRespondRequest
  } | null
  defaultExpanded?: boolean
}

type SubmitOutcome = "success" | "conflict" | "error"

const hasStudyAssistantContextShape = (value: unknown): value is StudyAssistantContextResponse =>
  typeof value === "object" &&
  value !== null &&
  "thread" in value &&
  "messages" in value &&
  "available_actions" in value

const extractStudyAssistantContext = (value: unknown): StudyAssistantContextResponse | null => {
  if (hasStudyAssistantContextShape(value)) {
    return value
  }
  if (
    typeof value === "object" &&
    value !== null &&
    "data" in value &&
    hasStudyAssistantContextShape((value as { data?: unknown }).data)
  ) {
    return (value as { data: StudyAssistantContextResponse }).data
  }
  return null
}

const isConflictError = (error: unknown): boolean =>
  typeof error === "object" &&
  error !== null &&
  "response" in error &&
  typeof (error as { response?: { status?: unknown } }).response?.status === "number" &&
  (error as { response?: { status?: number } }).response?.status === 409

const ACTION_LABELS: Record<StudyAssistantAction, string> = {
  explain: "Explain",
  mnemonic: "Mnemonic",
  follow_up: "Follow-up",
  fact_check: "Fact-check me",
  freeform: "Ask assistant"
}

type RemediationCitationLike = {
  citation_text?: string | null
  quote?: string | null
  label?: string | null
  source_type?: string
  source_id?: string
  locator?: string | null
  ordinal?: unknown
}

type RemediationDeepDiveTargetLike = {
  source_type?: string
  source_id?: string
  citation_ordinal?: unknown
  route_kind?: string | null
  route?: string | null
  available?: boolean
  fallback_reason?: string | null
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const toCleanString = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

const toSafeHref = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const href = value.trim()
  if (!href) return null
  if (href.startsWith("/")) return href

  try {
    const parsed = new URL(href)
    return parsed.protocol === "http:" || parsed.protocol === "https:" ? parsed.toString() : null
  } catch {
    return null
  }
}

const normalizeCitation = (value: unknown): RemediationCitationLike | null => {
  if (!isRecord(value)) return null
  const sourceType = toCleanString(value.source_type)
  const sourceId = toCleanString(value.source_id)
  if (!sourceType || !sourceId) return null
  return {
    citation_text: toCleanString(value.citation_text),
    quote: toCleanString(value.quote),
    label: toCleanString(value.label),
    source_type: sourceType,
    source_id: sourceId,
    locator: toCleanString(value.locator),
    ordinal: value.ordinal
  }
}

const normalizeCitationList = (value: unknown): RemediationCitationLike[] => {
  if (!Array.isArray(value)) return []
  return value.map(normalizeCitation).filter((entry): entry is RemediationCitationLike => entry != null)
}

const normalizeDeepDiveTarget = (value: unknown): RemediationDeepDiveTargetLike | null => {
  if (!isRecord(value)) return null
  const sourceType = toCleanString(value.source_type)
  const sourceId = toCleanString(value.source_id)
  if (!sourceType || !sourceId) return null
  return {
    source_type: sourceType,
    source_id: sourceId,
    citation_ordinal: value.citation_ordinal,
    route_kind: toCleanString(value.route_kind),
    route: toSafeHref(value.route),
    available: typeof value.available === "boolean" ? value.available : true,
    fallback_reason: toCleanString(value.fallback_reason)
  }
}

export const FlashcardStudyAssistantPanel: React.FC<FlashcardStudyAssistantPanelProps> = ({
  cardUuid,
  threadVersion = null,
  messages,
  availableActions,
  assistantContext = null,
  isLoading = false,
  isError = false,
  queryError,
  isResponding = false,
  onReloadContext,
  onRespond,
  autoSubmitRequest = null,
  defaultExpanded = false
}) => {
  const { t } = useTranslation(["option", "common"])
  const inRouterContext = useInRouterContext()

  const classifiedQueryError = React.useMemo(() => {
    if (!isError || !queryError) return null
    const kind = classifyAssistantError(queryError)
    if (kind === "network") {
      return t("option:flashcards.studyAssistantNetworkError", {
        defaultValue: "Could not reach the server. Check your connection and try again."
      })
    }
    if (kind === "server") {
      return t("option:flashcards.studyAssistantServerError", {
        defaultValue: "The server returned an error. Please try again or check server logs."
      })
    }
    return t("option:flashcards.studyAssistantNoLlm", {
      defaultValue: "Study assistant requires an LLM provider. Configure one in Settings \u2192 LLM Providers."
    })
  }, [isError, queryError, t])
  const assistantUnavailableTitle = t("option:flashcards.studyAssistantUnavailable", {
    defaultValue: "Study assistant unavailable"
  })

  const { speak, isSpeaking } = useTTS()
  const {
    supported,
    isListening,
    transcript,
    start,
    stop,
    resetTranscript
  } = useSpeechRecognition()
  const [assistantError, setAssistantError] = React.useState<string | null>(null)
  const [conflictRequest, setConflictRequest] = React.useState<StudyAssistantRespondRequest | null>(null)
  const [isConflictRecovering, setIsConflictRecovering] = React.useState(false)
  const [reloadedContext, setReloadedContext] = React.useState<StudyAssistantContextResponse | null>(null)
  const [followUpText, setFollowUpText] = React.useState("")
  const [factCheckTranscript, setFactCheckTranscript] = React.useState("")
  const [factCheckOpen, setFactCheckOpen] = React.useState(false)
  const [isExpanded, setIsExpanded] = React.useState(defaultExpanded)
  const lastAutoSubmitTokenRef = React.useRef<number | null>(null)
  const resetTranscriptRef = React.useRef(resetTranscript)
  const assistantWarningMessage = assistantError ?? classifiedQueryError

  React.useEffect(() => {
    resetTranscriptRef.current = resetTranscript
  }, [resetTranscript])

  const activeContextOverride = React.useMemo(() => {
    if (!reloadedContext) return null
    if (threadVersion == null) return reloadedContext
    return reloadedContext.thread.version > threadVersion ? reloadedContext : null
  }, [reloadedContext, threadVersion])

  const displayedMessages = activeContextOverride?.messages ?? messages
  const activeAssistantContext = React.useMemo(
    () => activeContextOverride ?? extractStudyAssistantContext(assistantContext),
    [activeContextOverride, assistantContext]
  )
  const resolvedActions = React.useMemo(() => {
    const sourceActions = activeContextOverride?.available_actions ?? availableActions
    return Array.from(new Set((sourceActions && sourceActions.length ? sourceActions : DEFAULT_ACTIONS)))
  }, [activeContextOverride?.available_actions, availableActions])
  const latestAssistantMessage = React.useMemo(
    () => [...displayedMessages].reverse().find((message) => message.role === "assistant") ?? null,
    [displayedMessages]
  )
  const remediationContext = React.useMemo(() => {
    const assistantRoot = isRecord(activeAssistantContext) ? activeAssistantContext : null
    const assistantPayload = isRecord(latestAssistantMessage?.structured_payload)
      ? latestAssistantMessage.structured_payload
      : null
    const rootCitations = normalizeCitationList(assistantRoot?.citations)
    const payloadCitations = normalizeCitationList(assistantPayload?.citations)
    const citations = rootCitations.length > 0 ? rootCitations : payloadCitations
    const primaryCitation =
      normalizeCitation(assistantRoot?.primary_citation) ??
      normalizeCitation(assistantPayload?.primary_citation) ??
      citations[0] ??
      null
    const deepDiveTarget =
      normalizeDeepDiveTarget(assistantRoot?.deep_dive_target) ??
      normalizeDeepDiveTarget(assistantPayload?.deep_dive_target)
    const supportingQuote =
      toCleanString(primaryCitation?.citation_text) ??
      toCleanString(primaryCitation?.quote) ??
      toCleanString(assistantPayload?.supporting_quote) ??
      toCleanString(assistantPayload?.supporting_quote_text) ??
      toCleanString(citations[0]?.citation_text) ??
      toCleanString(citations[0]?.quote) ??
      null
    return {
      citations,
      primaryCitation,
      deepDiveTarget,
      supportingQuote
    }
  }, [activeAssistantContext, latestAssistantMessage?.structured_payload])

  React.useEffect(() => {
    setAssistantError(null)
    setConflictRequest(null)
    setIsConflictRecovering(false)
    setReloadedContext(null)
    setFollowUpText("")
    setFactCheckTranscript("")
    setFactCheckOpen(false)
    resetTranscriptRef.current()
  }, [cardUuid])

  React.useEffect(() => {
    if (!reloadedContext || threadVersion == null) return
    if (threadVersion >= reloadedContext.thread.version) {
      setReloadedContext(null)
    }
  }, [reloadedContext, threadVersion])

  React.useEffect(() => {
    if (!factCheckOpen || !isListening) return
    setFactCheckTranscript(transcript)
  }, [factCheckOpen, isListening, transcript])

  const reloadLatestContext = React.useCallback(async () => {
    if (!onReloadContext) return true
    setIsConflictRecovering(true)
    try {
      const refreshed = await onReloadContext()
      const latestContext = extractStudyAssistantContext(refreshed)
      if (latestContext) {
        setReloadedContext(latestContext)
      }
      return true
    } catch (err) {
      const kind = classifyAssistantError(err)
      setAssistantError(
        kind === "network"
          ? t("option:flashcards.studyAssistantNetworkError", {
              defaultValue:
                "Could not reach the server. Check your connection and try again."
            })
          : kind === "server"
            ? t("option:flashcards.studyAssistantServerError", {
                defaultValue:
                  "The server returned an error. Please try again or check server logs."
              })
            : t("option:flashcards.studyAssistantNoLlm", {
                defaultValue:
                  "Study assistant requires an LLM provider. Configure one in Settings \u2192 LLM Providers."
              })
      )
      return false
    } finally {
      setIsConflictRecovering(false)
    }
  }, [onReloadContext, t])

  const submitRequest = React.useCallback(
    async (request: StudyAssistantRespondRequest): Promise<SubmitOutcome> => {
      try {
        setAssistantError(null)
        setConflictRequest(null)
        await onRespond(request)
        return "success"
      } catch (error) {
        if (isConflictError(error)) {
          setConflictRequest(request)
          const reloaded = await reloadLatestContext()
          if (!reloaded) {
            return "error"
          }
          return "conflict"
        }
        const kind = classifyAssistantError(error)
        setAssistantError(
          kind === "network"
            ? t("option:flashcards.studyAssistantNetworkError", {
                defaultValue:
                  "Could not reach the server. Check your connection and try again."
              })
            : kind === "server"
              ? t("option:flashcards.studyAssistantServerError", {
                  defaultValue:
                    "The server returned an error. Please try again or check server logs."
                })
              : t("option:flashcards.studyAssistantNoLlm", {
                  defaultValue:
                    "Study assistant requires an LLM provider. Configure one in Settings \u2192 LLM Providers."
                })
        )
        return "error"
      }
    },
    [onRespond, reloadLatestContext, t]
  )

  React.useEffect(() => {
    if (!autoSubmitRequest) return
    if (lastAutoSubmitTokenRef.current === autoSubmitRequest.token) return
    lastAutoSubmitTokenRef.current = autoSubmitRequest.token
    setIsExpanded(true)
    void submitRequest(autoSubmitRequest.request)
  }, [autoSubmitRequest, submitRequest])

  React.useEffect(() => {
    setIsExpanded(defaultExpanded)
  }, [cardUuid, defaultExpanded])

  const handleQuickAction = React.useCallback(
    async (action: StudyAssistantAction) => {
      if (action === "fact_check") {
        setFactCheckOpen(true)
        setFactCheckTranscript(transcript)
        if (supported && !isListening) {
          resetTranscript()
          start()
        }
        return
      }
      await submitRequest({ action })
    },
    [isListening, resetTranscript, start, submitRequest, supported, transcript]
  )

  const handleSubmitFollowUp = React.useCallback(async () => {
    const trimmed = followUpText.trim()
    if (!trimmed) return
    const action: StudyAssistantAction = resolvedActions.includes("follow_up")
      ? "follow_up"
      : "freeform"
    const outcome = await submitRequest({
      action,
      message: trimmed,
      input_modality: "text"
    })
    if (outcome === "success") {
      setFollowUpText("")
    }
  }, [followUpText, resolvedActions, submitRequest])

  const handleSubmitFactCheck = React.useCallback(async () => {
    const trimmed = factCheckTranscript.trim()
    if (!trimmed) return
    const outcome = await submitRequest({
      action: "fact_check",
      message: trimmed,
      input_modality: "voice_transcript"
    })
    if (outcome !== "success") return
    if (isListening) {
      stop()
    }
    setFactCheckOpen(false)
    setFactCheckTranscript("")
    resetTranscript()
  }, [factCheckTranscript, isListening, resetTranscript, stop, submitRequest])

  const handleCancelFactCheck = React.useCallback(() => {
    if (isListening) {
      stop()
    }
    setFactCheckOpen(false)
    setFactCheckTranscript("")
    resetTranscript()
  }, [isListening, resetTranscript, stop])

  const handlePlayReply = React.useCallback(
    (content: string) => {
      if (!content.trim()) return
      void speak({ utterance: content })
    },
    [speak]
  )

  const handleReloadLatest = React.useCallback(async () => {
    const reloaded = await reloadLatestContext()
    if (!reloaded) return
    setConflictRequest(null)
  }, [reloadLatestContext])

  const handleRetryConflict = React.useCallback(async () => {
    if (!conflictRequest) return
    await submitRequest(conflictRequest)
  }, [conflictRequest, submitRequest])

  const retryLabel = React.useMemo(() => {
    if (!conflictRequest) return null
    if (conflictRequest.input_modality === "voice_transcript") {
      return t("option:flashcards.studyAssistantRetryTranscript", {
        defaultValue: "Retry transcript review"
      })
    }
    return t("option:flashcards.studyAssistantRetryMessage", {
      defaultValue: "Retry my message"
    })
  }, [conflictRequest, t])
  const remediationLabel = t("option:flashcards.studyAssistantRemediation", {
    defaultValue: "Remediation"
  })
  const deepDiveLabel = t("option:flashcards.studyAssistantDeepDive", {
    defaultValue: "Deep dive to source"
  })

  return (
    <Card
      size="small"
      data-testid="flashcards-review-study-assistant"
      extra={
        <Button
          size="small"
          type={isExpanded ? "default" : "primary"}
          onClick={() => setIsExpanded((current) => !current)}
          aria-expanded={isExpanded}
          data-testid="flashcards-study-assistant-toggle"
        >
          {isExpanded
            ? t("option:flashcards.studyAssistantCollapse", {
                defaultValue: "Hide assistant"
              })
            : t("option:flashcards.studyAssistantExpand", {
                defaultValue: "Open assistant"
              })}
        </Button>
      }
      title={
        <div className="flex items-center gap-2">
          <Sparkles className="size-4 text-primary" aria-hidden="true" />
          <span>
            {t("option:flashcards.studyAssistantTitle", {
              defaultValue: "Study assistant"
            })}
          </span>
        </div>
      }
    >
      {!isExpanded ? (
        <Space orientation="vertical" size={8} className="w-full">
          <Text type="secondary">
            {t("option:flashcards.studyAssistantCollapsedPrompt", {
              defaultValue: "Need help understanding this card?"
            })}
          </Text>
          <Text type="secondary" className="text-xs">
            {t("option:flashcards.studyAssistantCollapsedDescription", {
              defaultValue:
                "Open the assistant when you want an explanation, mnemonic, or fact-check without leaving review."
            })}
          </Text>
        </Space>
      ) : (
        <Space orientation="vertical" size={12} className="w-full">
          {(assistantWarningMessage || isError) && (
            <Alert
              variant="warning"
              title={assistantUnavailableTitle}
            >
              {assistantWarningMessage}
            </Alert>
          )}
          {conflictRequest && (
            <div className="rounded border border-amber-300 bg-amber-50 p-3">
              <div className="font-medium">
                {t("option:flashcards.studyAssistantConflictTitle", {
                  defaultValue: "Conversation changed elsewhere."
                })}
              </div>
              <div className="text-sm text-text-muted">
                {t("option:flashcards.studyAssistantConflictDescription", {
                  defaultValue: "Reload the latest thread or retry your request against the updated conversation."
                })}
              </div>
              <Space className="mt-3">
                <Button
                  size="small"
                  onClick={() => void handleReloadLatest()}
                  loading={isConflictRecovering}
                >
                  {t("option:flashcards.studyAssistantReloadLatest", {
                    defaultValue: "Reload latest"
                  })}
                </Button>
                <Button
                  size="small"
                  type="primary"
                  onClick={() => void handleRetryConflict()}
                  loading={isResponding}
                >
                  {retryLabel}
                </Button>
              </Space>
            </div>
          )}
          <div className="flex flex-wrap gap-2">
            {resolvedActions.includes("explain") && (
              <Button
                onClick={() => void handleQuickAction("explain")}
                loading={isResponding || isConflictRecovering}
                icon={<Lightbulb className="size-4" />}
              >
                {t("option:flashcards.studyAssistantExplain", {
                  defaultValue: ACTION_LABELS.explain
                })}
              </Button>
            )}
            {resolvedActions.includes("mnemonic") && (
              <Button
                onClick={() => void handleQuickAction("mnemonic")}
                loading={isResponding || isConflictRecovering}
              >
                {t("option:flashcards.studyAssistantMnemonic", {
                  defaultValue: ACTION_LABELS.mnemonic
                })}
              </Button>
            )}
            {supported && resolvedActions.includes("fact_check") && (
              <Button
                onClick={() => void handleQuickAction("fact_check")}
                loading={isResponding || isConflictRecovering}
              >
                {t("option:flashcards.studyAssistantFactCheck", {
                  defaultValue: ACTION_LABELS.fact_check
                })}
              </Button>
            )}
          </div>
          {factCheckOpen && (
            <VoiceTranscriptComposer
              transcript={factCheckTranscript}
              isListening={isListening}
              supported={supported}
              isSubmitting={isResponding || isConflictRecovering}
              onTranscriptChange={setFactCheckTranscript}
              onStartListening={() => start()}
              onStopListening={stop}
              onCancel={handleCancelFactCheck}
              onSubmit={() => void handleSubmitFactCheck()}
            />
          )}
          {resolvedActions.includes("follow_up") || resolvedActions.includes("freeform") ? (
            <div className="rounded border border-border bg-surface p-3">
              <Space orientation="vertical" size={8} className="w-full">
                <Text strong>
                  {t("option:flashcards.studyAssistantChatPrompt", {
                    defaultValue: "Ask a follow-up"
                  })}
                </Text>
                <Input.TextArea
                  aria-label={t("option:flashcards.studyAssistantChatInput", {
                    defaultValue: "Ask the study assistant"
                  })}
                  value={followUpText}
                  onChange={(event) => setFollowUpText(event.target.value)}
                  rows={3}
                  placeholder={t("option:flashcards.studyAssistantChatPlaceholder", {
                    defaultValue: "Ask for clarification, examples, or a simpler explanation."
                  })}
                />
                <div className="flex justify-end">
                  <Button
                    type="primary"
                    icon={<MessageSquareText className="size-4" />}
                    loading={isResponding || isConflictRecovering}
                    disabled={!followUpText.trim()}
                    onClick={() => void handleSubmitFollowUp()}
                  >
                    {t("option:flashcards.studyAssistantChatSend", {
                      defaultValue: ACTION_LABELS.freeform
                    })}
                  </Button>
                </div>
              </Space>
            </div>
          ) : null}
          {(remediationContext.supportingQuote || remediationContext.deepDiveTarget) && (
            <div
              className="rounded border border-border bg-surface p-3"
              data-testid="flashcards-study-assistant-remediation"
            >
              <Space orientation="vertical" size={8} className="w-full">
                <Text strong>{remediationLabel}</Text>
                {remediationContext.supportingQuote && (
                  <div className="rounded border border-border/70 bg-background p-3">
                    <Text type="secondary" className="mb-1 block text-xs uppercase tracking-wide">
                      {t("option:flashcards.studyAssistantSupportingQuote", {
                        defaultValue: "Supporting quote"
                      })}
                    </Text>
                    <Text className="block text-sm text-text">
                      “{remediationContext.supportingQuote}”
                    </Text>
                  </div>
                )}
                {remediationContext.deepDiveTarget?.route ? (
                  remediationContext.deepDiveTarget.route.startsWith("/") && inRouterContext ? (
                    <RouterLink
                      to={remediationContext.deepDiveTarget.route}
                      className="text-sm text-primary hover:text-primary/80"
                    >
                      {deepDiveLabel}
                    </RouterLink>
                  ) : (
                    <Typography.Link
                      href={remediationContext.deepDiveTarget.route}
                      className="text-sm"
                    >
                      {deepDiveLabel}
                    </Typography.Link>
                  )
                ) : null}
              </Space>
            </div>
          )}
          <div>
            <Title level={5} className="!mb-2">
              {t("option:flashcards.studyAssistantHistory", {
                defaultValue: "Session"
              })}
            </Title>
            {isLoading ? (
              <Text type="secondary">
                {t("option:flashcards.studyAssistantLoading", {
                  defaultValue: "Loading assistant history..."
                })}
              </Text>
            ) : displayedMessages.length ? (
              <div className="flex flex-col gap-3">
                {displayedMessages.map((message) => (
                  <div
                    key={message.id}
                    className="rounded border border-border bg-surface p-3"
                  >
                    <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                      <Space size={8}>
                        <Tag color={message.role === "assistant" ? "blue" : "default"}>
                          {message.role === "assistant"
                            ? t("option:flashcards.studyAssistantRoleAssistant", {
                                defaultValue: "Assistant"
                              })
                            : t("option:flashcards.studyAssistantRoleUser", {
                                defaultValue: "You"
                              })}
                        </Tag>
                        <Text type="secondary" className="text-xs">
                          {t(`option:flashcards.studyAssistantAction.${message.action_type}`, {
                            defaultValue: ACTION_LABELS[message.action_type]
                          })}
                        </Text>
                      </Space>
                      {message.role === "assistant" && (
                        <Button
                          size="small"
                          icon={<Volume2 className="size-4" />}
                          loading={isSpeaking}
                          onClick={() => handlePlayReply(message.content)}
                        >
                          {t("option:flashcards.studyAssistantPlayReply", {
                            defaultValue: "Play reply"
                          })}
                        </Button>
                      )}
                    </div>
                    <MarkdownWithBoundary
                      content={message.content}
                      size="sm"
                      className="prose-headings:!text-text prose-p:!text-text prose-li:!text-text prose-strong:!text-text"
                    />
                  </div>
                ))}
              </div>
            ) : (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description={t("option:flashcards.studyAssistantEmpty", {
                  defaultValue: "Use the assistant to explain or check this card without leaving review mode."
                })}
              />
            )}
          </div>
        </Space>
      )}
    </Card>
  )
}

export default FlashcardStudyAssistantPanel
