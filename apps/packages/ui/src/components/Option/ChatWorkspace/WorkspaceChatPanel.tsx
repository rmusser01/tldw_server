import React from "react"

import { PlaygroundMessage } from "@/components/Common/Playground/Message"
import {
  type ChatSubmitResult,
  isChatSubmitSuccess,
  normalizeChatSubmitResult
} from "@/hooks/chat/chat-action-utils"
import type { Message } from "@/store/option"
import { useMessageOption } from "@/hooks/useMessageOption"
import type { ChatScope } from "@/types/chat-scope"

import { ContextStagingCard } from "./ContextStagingCard"
import {
  formatStagedSourceInsertText,
  getReadyStagedMediaIds
} from "./staging"
import type {
  ChatWorkspaceRuntimeState,
  StagedWorkspaceSource
} from "./types"
import { normalizeWorkspaceId } from "./workspaceIdentity"

export type WorkspaceChatPanelProps = {
  workspaceId?: string | null
  workspaceName?: string | null
  stagedSources: StagedWorkspaceSource[]
  onClearStagedSources: () => void
  backendAvailable: boolean
  onRuntimeStateChange?: (state: ChatWorkspaceRuntimeState) => void
}

const noop = () => undefined

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
  if (message?.role === "user" || message?.isBot === false) return "user"
  if (message?.role === "system") return "system"
  return "assistant"
}

export const WorkspaceChatPanel = ({
  workspaceId,
  workspaceName,
  stagedSources,
  onClearStagedSources,
  backendAvailable,
  onRuntimeStateChange
}: WorkspaceChatPanelProps) => {
  const [draft, setDraft] = React.useState("")
  const [sendError, setSendError] = React.useState<string | null>(null)
  const normalizedWorkspaceId = React.useMemo(
    () => normalizeWorkspaceId(workspaceId),
    [workspaceId]
  )
  const workspaceReady = normalizedWorkspaceId !== null
  const chatBackendAvailable = backendAvailable && workspaceReady

  const scope = React.useMemo<ChatScope>(
    () =>
      normalizedWorkspaceId
        ? { type: "workspace", workspaceId: normalizedWorkspaceId }
        : { type: "global" },
    [normalizedWorkspaceId]
  )

  const {
    messages,
    onSubmit,
    streaming,
    isLoading,
    isProcessing,
    stopStreamingRequest,
    selectedModel,
    selectedAssistant
  } = useMessageOption({ scope })

  React.useEffect(() => {
    setDraft("")
    setSendError(null)
  }, [normalizedWorkspaceId])

  React.useEffect(() => {
    onRuntimeStateChange?.({
      backendAvailable: chatBackendAvailable,
      streaming,
      selectedModelLabel: selectedModel || "No model selected",
      selectedPersonaLabel: selectedAssistant?.name ?? null
    })
  }, [
    chatBackendAvailable,
    onRuntimeStateChange,
    selectedAssistant?.name,
    selectedModel,
    streaming
  ])

  const isSending = streaming || isLoading || isProcessing
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
    isSending ||
    (!trimmedDraft && !hasStagedContext)
  const conversationInstanceId = normalizedWorkspaceId ?? "workspace-chat"

  const insertStagedSummary = React.useCallback(() => {
    const insertText = formatStagedSourceInsertText(stagedSources)
    if (!insertText) return

    setDraft((current) => {
      if (!current) return insertText
      const separator = current.endsWith("\n") ? "" : "\n\n"
      return `${current}${separator}${insertText}`
    })
    onClearStagedSources()
  }, [onClearStagedSources, stagedSources])

  const submitMessage = React.useCallback(async () => {
    if (sendDisabled) return
    if (!chatBackendAvailable) return

    setSendError(null)
    const fallbackContext = formatStagedSourceInsertText(stagedSources).trim()
    const sendMessage =
      trimmedDraft && hasStagedContext && hasUncarriedStagedContext && fallbackContext
        ? `${trimmedDraft}\n\n${fallbackContext}`
        : trimmedDraft || fallbackContext

    try {
      const result = normalizeChatSubmitResult(
        (await onSubmit({
          message: sendMessage,
          image: "",
          requestOverrides: {
            ragMediaIds: readyMediaIds,
            fileRetrievalEnabled: hasReadyMedia,
            chatMode: hasReadyMedia ? "rag" : "normal"
          }
        })) as ChatSubmitResult | undefined
      )

      if (isChatSubmitSuccess(result)) {
        setDraft("")
        onClearStagedSources()
        return
      }

      if (result.status === "skipped") {
        return
      }

      setSendError(result.errorMessage || "Send failed")
    } catch {
      setSendError("Send failed")
    }
  }, [
    chatBackendAvailable,
    hasReadyMedia,
    hasUncarriedStagedContext,
    onClearStagedSources,
    onSubmit,
    readyMediaIds,
    sendDisabled,
    stagedSources,
    trimmedDraft
  ])

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

  return (
    <section
      aria-label="Chat workspace panel"
      className="flex h-full min-h-0 flex-col bg-background text-text"
    >
      <div className="border-b border-border px-4 py-3">
        <h2 className="text-sm font-semibold">
          {workspaceName ? `${workspaceName} chat` : "Workspace chat"}
        </h2>
      </div>

      <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-4 py-3">
        {Array.isArray(messages) && messages.length > 0 ? (
          messages.map((message: WorkspacePanelMessage, index: number) => {
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
                hideEditAndRegenerate
                hideContinue
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

      <div className="flex flex-col gap-3 border-t border-border bg-surface2/40 px-4 py-3">
        {hasStagedContext ? (
          <ContextStagingCard
            sources={stagedSources}
            isSending={isSending}
            canSend={chatBackendAvailable}
            onClear={onClearStagedSources}
            onInsert={insertStagedSummary}
            onSend={submitMessage}
          />
        ) : null}

        {(streaming || isProcessing || isLoading) ? (
          <div className="flex items-center justify-between gap-3 text-sm text-text-muted">
            <span>{streaming ? "Streaming" : "Sending"}</span>
            {streaming ? (
              <button
                type="button"
                className="inline-flex min-h-[32px] items-center rounded-md border border-border px-3 py-1.5 text-sm font-medium text-text transition-colors hover:bg-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                onClick={() => stopStreamingRequest()}
                aria-label="Stop generating"
              >
                Stop generating
              </button>
            ) : null}
          </div>
        ) : null}

        {sendError ? (
          <p className="text-sm font-medium text-danger" role="alert">
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

        <form className="flex flex-col gap-2" onSubmit={handleSubmit}>
          <textarea
            aria-label="Chat workspace message"
            className="min-h-[88px] resize-y rounded-md border border-border bg-background px-3 py-2 text-sm text-text outline-none transition-colors placeholder:text-text-muted focus:border-primary focus:ring-2 focus:ring-focus"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={handleComposerKeyDown}
            placeholder="Ask about this workspace"
          />
          <div className="flex justify-end">
            <button
              type="submit"
              className="inline-flex min-h-[36px] items-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-primaryStrong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50"
              disabled={sendDisabled}
              aria-label="Send message"
            >
              Send message
            </button>
          </div>
        </form>
      </div>
    </section>
  )
}
