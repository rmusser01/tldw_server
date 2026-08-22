import React from "react"
import { Link, useNavigate } from "react-router-dom"
import { Loader2, Send } from "lucide-react"
import { useTranslation } from "react-i18next"
import { Markdown } from "@/components/Common/Markdown"
import { ChatModelSelectorDropdown } from "@/components/Option/Playground/ChatModelSelectorDropdown"
import { useModelSelector } from "@/hooks/playground"
import {
  getCanonicalModelKey,
  getModelId,
  getModelProvider
} from "@/hooks/playground/modelSelectorUtils"
import type { fetchChatModels } from "@/services/tldw-server"
import type { SharedCitation } from "@/types/shared-workspace"
import { resolveStartupSelectedModel } from "@/utils/model-startup-selection"
import type { SharedResearchWorkspaceController } from "./useSharedResearchWorkspace"

type ChatModel = Awaited<ReturnType<typeof fetchChatModels>>[number]

type SharedWorkspaceChatPaneProps = {
  controller: SharedResearchWorkspaceController
  onPreviewCitation: (
    sourceId: string,
    chunkIndex: number | undefined,
    trigger: HTMLElement
  ) => void
}

const CitationButton: React.FC<{
  citation: SharedCitation
  index: number
  onPreviewCitation: SharedWorkspaceChatPaneProps["onPreviewCitation"]
}> = ({ citation, index, onPreviewCitation }) => {
  const { t } = useTranslation("playground")
  const activate = (target: HTMLButtonElement) =>
    onPreviewCitation(
      citation.source_id,
      citation.locator.chunk ?? undefined,
      target
    )

  return (
    <button
      type="button"
      aria-label={t(
        "sharedWorkspace.openCitation",
        "Open citation {{index}} from {{title}}",
        { index: index + 1, title: citation.source_title }
      )}
      onClick={(event) => activate(event.currentTarget)}
      onKeyDown={(event) => {
        if (event.key !== "Enter" && event.key !== " ") return
        event.preventDefault()
        activate(event.currentTarget)
      }}
      className="min-w-0 rounded-md border border-border bg-surface2 px-2.5 py-2 text-left outline-none transition-colors hover:border-primary/50 hover:bg-surface focus-visible:ring-2 focus-visible:ring-focus"
    >
      <span className="block truncate text-xs font-semibold text-primary">
        [{index + 1}] {citation.source_title}
      </span>
      <Markdown
        message={citation.quote}
        allowExternalImages={false}
        enableMermaidDiagrams={false}
        className="prose mt-0.5 max-w-none break-words text-xs text-text-muted dark:prose-invert"
      />
    </button>
  )
}

export const SharedWorkspaceChatPane: React.FC<
  SharedWorkspaceChatPaneProps
> = ({ controller, onPreviewCitation }) => {
  const { t } = useTranslation("playground")
  const navigate = useNavigate()
  const { setModel, setProvider, state } = controller
  const [catalogModels, setCatalogModels] = React.useState<ChatModel[]>([])
  const [modelsLoading, setModelsLoading] = React.useState(true)
  const [selectedModel, setSelectedModel] = React.useState<string | null>(null)
  const [announcement, setAnnouncement] = React.useState("")
  const seededDefaultRef = React.useRef<string | null>(null)
  const awaitingAnswerRef = React.useRef(false)
  const messagesEndRef = React.useRef<HTMLDivElement>(null)
  const generationDefault = state.bootstrap?.generation_default

  React.useEffect(() => {
    let active = true
    setModelsLoading(true)
    void import("@/services/tldw-server")
      .then(({ fetchChatModels: loadChatModels }) =>
        loadChatModels({ returnEmpty: true })
      )
      .then((models) => {
        if (active) setCatalogModels(Array.isArray(models) ? models : [])
      })
      .catch(() => {
        if (active) setCatalogModels([])
      })
      .finally(() => {
        if (active) setModelsLoading(false)
      })
    return () => {
      active = false
    }
  }, [])

  const composerModels = React.useMemo<ChatModel[]>(() => {
    if (!generationDefault?.ready) return catalogModels
    const hasDefault = catalogModels.some(
      (model) =>
        getModelProvider(model) === generationDefault.provider.toLowerCase() &&
        getModelId(model) === generationDefault.model
    )
    if (hasDefault) return catalogModels
    return [
      {
        model: generationDefault.model,
        provider: generationDefault.provider,
        nickname: generationDefault.model,
        configured: true,
        is_configured: true
      },
      ...catalogModels
    ]
  }, [catalogModels, generationDefault])

  React.useEffect(() => {
    if (!generationDefault?.ready) return
    const key = getCanonicalModelKey(
      generationDefault.provider,
      generationDefault.model
    )
    if (seededDefaultRef.current === key) return
    seededDefaultRef.current = key
    setSelectedModel(key)
    setProvider(generationDefault.provider)
    setModel(generationDefault.model)
  }, [
    generationDefault?.model,
    generationDefault?.provider,
    generationDefault?.ready,
    setModel,
    setProvider
  ])

  React.useEffect(() => {
    if (generationDefault?.ready || selectedModel) return
    const startup = resolveStartupSelectedModel({
      currentModel: selectedModel,
      models: composerModels,
      preferredModelIds: [],
      isCurrentModelHydrating: false,
      arePreferencesHydrating: false
    })
    if (startup) setSelectedModel(startup)
  }, [composerModels, generationDefault?.ready, selectedModel])

  React.useEffect(() => {
    if (!selectedModel) return
    const selected = composerModels.find(
      (model) =>
        getCanonicalModelKey(model).toLowerCase() === selectedModel.toLowerCase()
    )
    if (!selected) return
    const provider = getModelProvider(selected)
    const model = getModelId(selected)
    if (provider && provider !== state.provider) setProvider(provider)
    if (model && model !== state.model) setModel(model)
  }, [
    composerModels,
    selectedModel,
    setModel,
    setProvider,
    state.model,
    state.provider
  ])

  const modelSelector = useModelSelector({
    composerModels,
    selectedModel,
    setSelectedModel,
    navigate,
    modelsLoading
  })

  const sourceCount =
    state.sourceScopeMode === "all"
      ? state.sourceSummary?.queryable ?? 0
      : state.selectedSourceIds.length
  const invalidScope =
    sourceCount === 0 ||
    sourceCount > 500 ||
    (state.sourceScopeMode === "all" &&
      (state.sourceSummary?.queryable ?? 0) > 500)
  const submitting = state.pendingSubmission?.status === "submitting"
  const rateLimited = state.rateLimitRemainingMs > 0
  const canSubmit =
    Boolean(state.draft.trim()) &&
    state.allowedActions.ask_grounded_questions.allowed &&
    generationDefault?.ready === true &&
    Boolean(state.provider && state.model && selectedModel) &&
    !invalidScope &&
    !submitting &&
    !rateLimited
  const submissionCode = state.errors.submission?.code
  const directError =
    submissionCode === "shared_source_changed"
      ? t(
          "sharedWorkspace.sourceConflict",
          "The shared source set changed. Refresh sources before trying again."
        )
      : submissionCode === "shared_generation_unavailable"
        ? t(
            "sharedWorkspace.noProvider",
            "Choose a configured model before asking a question."
          )
        : submissionCode === "shared_context_budget_exceeded"
          ? t(
              "sharedWorkspace.contextBudget",
              "The selected sources exceed this model's context budget. Choose fewer sources and try again."
            )
          : submissionCode === "shared_retrieval_unavailable"
            ? t(
                "sharedWorkspace.retrievalUnavailable",
                "Shared source retrieval is temporarily unavailable. Try again."
              )
            : state.errors.submission?.message || null

  React.useEffect(() => {
    if (!state.errors.submission) return
    awaitingAnswerRef.current = false
    if (state.rateLimitRemainingMs > 0) {
      setAnnouncement(
        t("sharedWorkspace.rateLimited", "Try again in {{count}} second", {
          count: Math.max(1, Math.ceil(state.rateLimitRemainingMs / 1000))
        })
      )
      return
    }
    setAnnouncement(t("sharedWorkspace.questionNotSent", "Question not sent"))
  }, [state.errors.submission, state.rateLimitRemainingMs, t])

  React.useEffect(() => {
    if (
      !awaitingAnswerRef.current ||
      !state.messages.some((message) => message.role === "assistant")
    ) {
      return
    }
    awaitingAnswerRef.current = false
    setAnnouncement(t("sharedWorkspace.answerAdded", "Answer added"))
  }, [state.messages, t])

  React.useEffect(() => {
    const messagesEnd = messagesEndRef.current
    if (typeof messagesEnd?.scrollIntoView === "function") {
      messagesEnd.scrollIntoView({ block: "nearest" })
    }
  }, [state.messages.length])

  const submit = async () => {
    if (!canSubmit) return
    awaitingAnswerRef.current = true
    setAnnouncement(
      t("sharedWorkspace.askingStatus", "Asking shared workspace")
    )
    await controller.submitDraft()
  }

  return (
    <section
      data-testid="shared-workspace-chat-pane"
      aria-labelledby="shared-workspace-chat-heading"
      className="flex min-h-0 min-w-0 flex-col overflow-hidden bg-bg"
    >
      <div className="flex h-12 shrink-0 items-center justify-between gap-2 border-b border-border bg-surface px-3">
        <div className="min-w-0">
          <h2
            id="shared-workspace-chat-heading"
            className="text-sm font-semibold"
          >
            {t("sharedWorkspace.chat", "Chat")}
          </h2>
          <p className="truncate text-xs text-text-muted">
            {t("sharedWorkspace.scopeCount", "{{count}} sources in scope", {
              count: sourceCount
            })}
          </p>
        </div>
        <ChatModelSelectorDropdown
          apiModelLabel={modelSelector.apiModelLabel}
          connectionStatusLabel={t(
            "sharedWorkspace.modelConfigured",
            "Configured"
          )}
          modelDropdownMenuItems={modelSelector.modelDropdownMenuItems}
          modelDropdownOpen={modelSelector.modelDropdownOpen}
          modelSearchQuery={modelSelector.modelSearchQuery}
          modelSelectorWarning={modelSelector.modelSelectorWarning}
          modelSortMode={modelSelector.modelSortMode}
          placement="bottomLeft"
          resolvedProviderKey={modelSelector.resolvedProviderKey}
          selectedModel={selectedModel}
          setModelDropdownOpen={modelSelector.setModelDropdownOpen}
          setModelSearchQuery={modelSelector.setModelSearchQuery}
          setModelSortMode={modelSelector.setModelSortMode}
        />
      </div>

      <div
        role="log"
        aria-label={t(
          "sharedWorkspace.messagesLabel",
          "Shared workspace messages"
        )}
        className="min-h-0 min-w-0 flex-1 overflow-y-auto px-3 py-4 sm:px-5"
      >
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-3">
          {state.nextBefore ? (
            <button
              type="button"
              onClick={() => void controller.loadOlderHistory()}
              className="mx-auto h-9 rounded-md px-3 text-xs font-medium text-primary outline-none hover:bg-surface2 focus-visible:ring-2 focus-visible:ring-focus"
              aria-label={t(
                "sharedWorkspace.loadOlder",
                "Load older messages"
              )}
            >
              {t("sharedWorkspace.loadOlder", "Load older messages")}
            </button>
          ) : null}
          {state.messages.map((message) => (
            <article
              key={message.message_id}
              data-message-id={message.message_id}
              className={
                message.role === "user"
                  ? "ml-auto max-w-[85%] rounded-md bg-primary/10 px-3 py-2"
                  : "max-w-[72ch] border-b border-border px-1 py-3"
              }
            >
              <span className="sr-only">
                {message.role === "user"
                  ? t("sharedWorkspace.you", "You")
                  : t("sharedWorkspace.assistant", "Assistant")}
              </span>
              <Markdown
                message={message.content}
                allowExternalImages={false}
                enableMermaidDiagrams={false}
                className="prose max-w-none break-words text-sm dark:prose-invert"
              />
              {message.citations.length ? (
                <div className="mt-3 grid min-w-0 gap-2">
                  {message.citations.map((citation, index) => (
                    <CitationButton
                      key={citation.citation_id}
                      citation={citation}
                      index={index}
                      onPreviewCitation={onPreviewCitation}
                    />
                  ))}
                </div>
              ) : null}
            </article>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="shrink-0 border-t border-border bg-surface px-3 py-3 sm:px-5">
        <div className="mx-auto w-full max-w-3xl space-y-2">
          {!generationDefault?.ready ? (
            <div className="flex flex-wrap items-center justify-between gap-2 text-sm text-warn">
              <span>
                {t(
                  "sharedWorkspace.noProvider",
                  "Choose a configured model before asking a question."
                )}
              </span>
              <Link
                to="/settings/tldw"
                className="rounded-md px-2 py-1 font-medium text-primary outline-none focus-visible:ring-2 focus-visible:ring-focus"
              >
                {t(
                  "sharedWorkspace.openModelSettings",
                  "Open model settings"
                )}
              </Link>
            </div>
          ) : null}
          {directError ? (
            <div className="flex flex-wrap items-center justify-between gap-2 text-sm text-danger">
              <span>{directError}</span>
              {state.errors.submission?.code === "shared_source_changed" ? (
                <button
                  type="button"
                  onClick={() => void controller.refreshSources()}
                  className="h-9 rounded-md px-2 font-medium text-primary outline-none focus-visible:ring-2 focus-visible:ring-focus"
                >
                  {t("sharedWorkspace.refreshSources", "Refresh sources")}
                </button>
              ) : null}
              {state.pendingSubmission?.status === "retryable" ? (
                <button
                  type="button"
                  onClick={() => void controller.retryPending()}
                  disabled={rateLimited}
                  className="h-9 rounded-md px-2 font-medium text-primary outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:opacity-40"
                >
                  {t("sharedWorkspace.retry", "Retry")}
                </button>
              ) : null}
            </div>
          ) : null}
          <div className="grid min-w-0 grid-cols-[minmax(0,1fr)_2.75rem] items-end gap-2">
            <label className="min-w-0">
              <span className="sr-only">
                {t(
                  "sharedWorkspace.composerLabel",
                  "Ask about shared sources"
                )}
              </span>
              <textarea
                aria-label={t(
                  "sharedWorkspace.composerLabel",
                  "Ask about shared sources"
                )}
                value={state.draft}
                onChange={(event) => controller.setDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (
                    event.key === "Enter" &&
                    !event.shiftKey &&
                    !event.nativeEvent.isComposing
                  ) {
                    event.preventDefault()
                    void submit()
                  }
                }}
                rows={2}
                className="max-h-32 min-h-[2.75rem] w-full resize-none rounded-md border border-border bg-surface2 px-3 py-2 text-sm outline-none focus-visible:border-primary focus-visible:ring-2 focus-visible:ring-focus"
              />
            </label>
            <button
              type="button"
              aria-label={t(
                "sharedWorkspace.askButton",
                "Ask shared workspace"
              )}
              disabled={!canSubmit}
              onClick={() => void submit()}
              className="inline-flex h-11 w-11 items-center justify-center rounded-md bg-primary text-white outline-none transition-colors hover:bg-primary/90 focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {submitting ? (
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
              ) : (
                <Send className="h-4 w-4" aria-hidden="true" />
              )}
            </button>
          </div>
          <div role="status" aria-live="polite" className="min-h-4 text-xs text-text-muted">
            {rateLimited
              ? t(
                  "sharedWorkspace.rateLimited",
                  "Try again in {{count}} second",
                  {
                    count: Math.max(
                      1,
                      Math.ceil(state.rateLimitRemainingMs / 1000)
                    )
                  }
                )
              : announcement}
          </div>
        </div>
      </div>
    </section>
  )
}
